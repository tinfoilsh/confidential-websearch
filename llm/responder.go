package llm

import (
	"context"
	"encoding/json"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/shared"
	log "github.com/sirupsen/logrus"

	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

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
		content = resp.Choices[0].Message.Content
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

	// Debug: Log message count and approximate size
	totalMsgSize := 0
	for _, msg := range params.Messages {
		msgBytes, _ := json.Marshal(msg)
		totalMsgSize += len(msgBytes)
	}
	log.Debugf("[Responder.Stream] Starting stream: model=%s, messages=%d, totalMsgSize=%d bytes, annotations=%d",
		params.Model, len(params.Messages), totalMsgSize, len(annotations))

	log.Debug("[Responder.Stream] Calling client.NewStreaming()...")
	stream := r.client.NewStreaming(ctx, chatParams, opts...)
	log.Debug("[Responder.Stream] NewStreaming() returned, entering stream loop...")

	metadataSent := false
	chunkCount := 0

	log.Debug("[Responder.Stream] Calling stream.Next() for first chunk...")
	for stream.Next() {
		chunkCount++
		if chunkCount == 1 {
			log.Debug("[Responder.Stream] Received first chunk from upstream!")
		}
		if chunkCount <= 3 || chunkCount%100 == 0 {
			log.Debugf("[Responder.Stream] Processing chunk #%d", chunkCount)
		}

		chunk := stream.Current()

		// Send metadata (annotations + reasoning + reasoning items) on first chunk with choices
		if !metadataSent && len(chunk.Choices) > 0 && (len(annotations) > 0 || reasoning != "" || len(reasoningItems) > 0) {
			log.Debugf("[Responder.Stream] Emitting metadata: annotations=%d, reasoning=%d chars, reasoningItems=%d",
				len(annotations), len(reasoning), len(reasoningItems))
			if err := emitter.EmitMetadata(annotations, reasoning, reasoningItems); err != nil {
				log.Errorf("[Responder.Stream] EmitMetadata failed: %v", err)
				return err
			}
			metadataSent = true
		}

		if err := emitter.EmitChunk([]byte(chunk.RawJSON())); err != nil {
			log.Errorf("[Responder.Stream] EmitChunk failed: %v", err)
			return err
		}
	}

	log.Debugf("[Responder.Stream] Stream loop ended after %d chunks", chunkCount)

	if err := stream.Err(); err != nil {
		log.Errorf("[Responder.Stream] Stream error: %v", err)
		return emitter.EmitError(err)
	}

	log.Debug("[Responder.Stream] Stream completed successfully, emitting done")
	return emitter.EmitDone()
}
