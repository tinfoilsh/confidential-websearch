package llm

import (
	"context"
	"encoding/json"
	"regexp"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/packages/ssestream"
	"github.com/openai/openai-go/v2/shared"

	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

// citationMarkerRegex matches OpenAI-style citation markers like 【1】 or 【1†L1-L15】
var citationMarkerRegex = regexp.MustCompile(`【\d+[^】]*】`)

// ResponderParams contains parameters for a responder LLM call
type ResponderParams struct {
	Model       string
	Messages    []openai.ChatCompletionMessageParamUnion
	Temperature *float64
	MaxTokens   *int64
}

// ResponderResult contains the result of a non-streaming responder call
type ResponderResult struct {
	ID      string
	Model   string
	Object  string
	Created int64
	Content string
	Usage   interface{}
}

// ChatCompletionStream is the stream type returned by NewStreaming
type ChatCompletionStream = ssestream.Stream[openai.ChatCompletionChunk]

// ChatClient defines the interface for chat completion operations
type ChatClient interface {
	New(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error)
	NewStreaming(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) *ChatCompletionStream
}

// Responder handles the final LLM call to generate responses
type Responder interface {
	// Complete makes a non-streaming completion call
	Complete(ctx context.Context, params ResponderParams, opts ...option.RequestOption) (*ResponderResult, error)

	// Stream makes a streaming completion call, emitting chunks via the emitter
	Stream(ctx context.Context, params ResponderParams, annotations []pipeline.Annotation, reasoning string, emitter pipeline.EventEmitter, opts ...option.RequestOption) error
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
func (r *TinfoilResponder) Complete(ctx context.Context, params ResponderParams, opts ...option.RequestOption) (*ResponderResult, error) {
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

	return &ResponderResult{
		ID:      resp.ID,
		Model:   resp.Model,
		Object:  string(resp.Object),
		Created: resp.Created,
		Content: content,
		Usage:   resp.Usage,
	}, nil
}

// Stream makes a streaming completion call
func (r *TinfoilResponder) Stream(ctx context.Context, params ResponderParams, annotations []pipeline.Annotation, reasoning string, emitter pipeline.EventEmitter, opts ...option.RequestOption) error {
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

	for stream.Next() {
		chunk := stream.Current()

		// Send metadata (annotations + reasoning) on first chunk with choices
		if !metadataSent && len(chunk.Choices) > 0 && (len(annotations) > 0 || reasoning != "") {
			if err := emitter.EmitMetadata(annotations, reasoning); err != nil {
				return err
			}
			metadataSent = true
		}

		// Strip citation markers from streaming content
		chunkData := chunk.RawJSON()
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			chunkData = stripCitationMarkersFromChunk(chunkData, chunk.Choices[0].Delta.Content)
		}

		if err := emitter.EmitChunk([]byte(chunkData)); err != nil {
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

// stripCitationMarkersFromChunk removes citation markers from a streaming chunk
func stripCitationMarkersFromChunk(chunkData string, originalContent string) string {
	cleaned := StripCitationMarkers(originalContent)
	if cleaned == originalContent {
		return chunkData
	}

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

	delta["content"] = cleaned
	marshaled, err := json.Marshal(parsed)
	if err != nil {
		return chunkData
	}

	return string(marshaled)
}
