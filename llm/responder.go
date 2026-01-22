package llm

import (
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"unicode"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/shared"

	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

// citationMarkerRegex matches OpenAI-style citation markers like 【1】 or 【1†L1-L15】
var citationMarkerRegex = regexp.MustCompile(`【\d+[^】]*】`)

// StreamingCitationStripper handles citation markers that may span multiple streaming chunks
type StreamingCitationStripper struct {
	buffer   strings.Builder
	inMarker bool
}

// NewStreamingCitationStripper creates a new streaming citation stripper
func NewStreamingCitationStripper() *StreamingCitationStripper {
	return &StreamingCitationStripper{}
}

// Process takes incoming content and returns cleaned content to emit.
func (s *StreamingCitationStripper) Process(content string) string {
	var result strings.Builder
	for _, r := range content {
		if s.inMarker {
			s.buffer.WriteRune(r)
			if r == '】' {
				marker := s.buffer.String()
				if !isValidCitationMarker(marker) {
					result.WriteString(marker)
				}
				s.buffer.Reset()
				s.inMarker = false
			}
		} else if r == '【' {
			s.inMarker = true
			s.buffer.WriteRune(r)
		} else {
			result.WriteRune(r)
		}
	}
	return result.String()
}

// Flush returns any remaining buffered content (call at end of stream)
func (s *StreamingCitationStripper) Flush() string {
	content := s.buffer.String()
	s.buffer.Reset()
	s.inMarker = false
	return content
}

// isValidCitationMarker checks if the string matches the citation pattern
func isValidCitationMarker(str string) bool {
	if !strings.HasPrefix(str, "【") || !strings.HasSuffix(str, "】") {
		return false
	}
	inner := strings.TrimPrefix(str, "【")
	inner = strings.TrimSuffix(inner, "】")
	if len(inner) == 0 {
		return false
	}
	return unicode.IsDigit(rune(inner[0]))
}

// ChatCompletionStream is the stream type returned by NewStreaming
type ChatCompletionStream = ssestream.Stream[openai.ChatCompletionChunk]

// ChatClient defines the interface for chat completion operations
type ChatClient interface {
	New(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error)
	NewStreaming(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) *ChatCompletionStream
}

// TinfoilResponder implements Responder using a Tinfoil client
type TinfoilResponder struct {
	client ChatClient
}

// NewTinfoilResponder creates a new TinfoilResponder
func NewTinfoilResponder(client ChatClient) *TinfoilResponder {
	return &TinfoilResponder{client: client}
}

// Complete makes a non-streaming completion call
func (r *TinfoilResponder) Complete(ctx context.Context, params pipeline.ResponderParams, opts ...option.RequestOption) (*pipeline.ResponderResultData, error) {
	chatParams := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(params.Model),
		Messages: params.Messages,
	}
	if params.Temperature != nil {
		chatParams.Temperature = openai.Float(*params.Temperature)
	}
	if params.MaxTokens != nil {
		chatParams.MaxTokens = openai.Int(*params.MaxTokens)
	}

	resp, err := r.client.New(ctx, chatParams, opts...)
	if err != nil {
		return nil, err
	}

	var content string
	if len(resp.Choices) > 0 {
		// Strip citation markers from content
		content = StripCitationMarkers(resp.Choices[0].Message.Content)
	}

	return &pipeline.ResponderResultData{
		ID:      resp.ID,
		Model:   resp.Model,
		Object:  string(resp.Object),
		Created: resp.Created,
		Content: content,
		Usage:   resp.Usage,
	}, nil
}

// Stream makes a streaming completion call
func (r *TinfoilResponder) Stream(ctx context.Context, params pipeline.ResponderParams, annotations []pipeline.Annotation, reasoning string, reasoningItems []pipeline.ReasoningItem, emitter pipeline.EventEmitter, opts ...option.RequestOption) error {
	chatParams := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(params.Model),
		Messages: params.Messages,
	}
	if params.Temperature != nil {
		chatParams.Temperature = openai.Float(*params.Temperature)
	}
	if params.MaxTokens != nil {
		chatParams.MaxTokens = openai.Int(*params.MaxTokens)
	}

	stream := r.client.NewStreaming(ctx, chatParams, opts...)
	metadataSent := false
	stripper := NewStreamingCitationStripper()

	for stream.Next() {
		chunk := stream.Current()

		// Send metadata (annotations + reasoning + reasoning items) on first chunk with choices
		if !metadataSent && len(chunk.Choices) > 0 && (len(annotations) > 0 || reasoning != "" || len(reasoningItems) > 0) {
			if err := emitter.EmitMetadata(annotations, reasoning, reasoningItems); err != nil {
				return err
			}
			metadataSent = true
		}

		// Strip citation markers from streaming content (handles markers spanning chunks)
		chunkData := chunk.RawJSON()
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			cleaned := stripper.Process(chunk.Choices[0].Delta.Content)
			chunkData = updateChunkContent(chunkData, cleaned)
		}

		if err := emitter.EmitChunk([]byte(chunkData)); err != nil {
			return err
		}
	}

	// Flush any remaining buffered content
	if remaining := stripper.Flush(); remaining != "" {
		if err := emitter.EmitChunk([]byte(`{"choices":[{"delta":{"content":"` + remaining + `"}}]}`)); err != nil {
			return err
		}
	}

	if err := stream.Err(); err != nil {
		return emitter.EmitError(err)
	}

	return emitter.EmitDone()
}

// StripCitationMarkers removes OpenAI-style citation markers from content
func StripCitationMarkers(content string) string {
	return citationMarkerRegex.ReplaceAllString(content, "")
}

// updateChunkContent replaces the content in a streaming chunk JSON
func updateChunkContent(chunkData string, newContent string) string {
	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(chunkData), &parsed); err != nil {
		return chunkData
	}

	choices, ok := parsed["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return chunkData
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return chunkData
	}

	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		return chunkData
	}

	delta["content"] = newContent
	marshaled, err := json.Marshal(parsed)
	if err != nil {
		return chunkData
	}

	return string(marshaled)
}
