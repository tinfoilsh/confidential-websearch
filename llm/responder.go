package llm

import (
	"context"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/shared"

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
func (r *TinfoilResponder) Stream(ctx context.Context, params pipeline.ResponderParams, annotations []pipeline.Annotation, reasoning string, emitter pipeline.EventEmitter, opts ...option.RequestOption) error {
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
			if err := emitter.EmitMetadata(chunk.ID, chunk.Created, chunk.Model, annotations, reasoning); err != nil {
				return err
			}
			metadataSent = true
		}

		if err := emitter.EmitChunk([]byte(chunk.RawJSON())); err != nil {
			return err
		}
	}

	if err := stream.Err(); err != nil {
		return emitter.EmitError(err)
	}

	return emitter.EmitDone()
}
