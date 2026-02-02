package llm

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

// MockChatClient implements ChatClient for testing
type MockChatClient struct {
	NewFunc          func(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error)
	NewStreamingFunc func(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) *ChatCompletionStream
}

func (m *MockChatClient) New(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
	if m.NewFunc != nil {
		return m.NewFunc(ctx, params, opts...)
	}
	return nil, errors.New("not implemented")
}

func (m *MockChatClient) NewStreaming(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) *ChatCompletionStream {
	if m.NewStreamingFunc != nil {
		return m.NewStreamingFunc(ctx, params, opts...)
	}
	return nil
}

// MockEventEmitter implements pipeline.EventEmitter for testing
type MockEventEmitter struct {
	mu              sync.Mutex
	SearchCalls     []mockSearchCall
	MetadataCalls   []mockMetadataCall
	Chunks          [][]byte
	Errors          []error
	DoneCalled      bool
	EmitChunkErr    error
	EmitMetadataErr error
	EmitDoneErr     error
}

type mockSearchCall struct {
	ID     string
	Status string
	Query  string
	Reason string
}

type mockMetadataCall struct {
	Annotations     []pipeline.Annotation
	SearchReasoning string
}

func (m *MockEventEmitter) EmitSearchCall(id, status, query, reason string, created int64, model string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.SearchCalls = append(m.SearchCalls, mockSearchCall{ID: id, Status: status, Query: query, Reason: reason})
	return nil
}

func (m *MockEventEmitter) EmitMetadata(id string, created int64, model string, annotations []pipeline.Annotation, reasoning string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.EmitMetadataErr != nil {
		return m.EmitMetadataErr
	}
	m.MetadataCalls = append(m.MetadataCalls, mockMetadataCall{Annotations: annotations, SearchReasoning: reasoning})
	return nil
}

func (m *MockEventEmitter) EmitChunk(data []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.EmitChunkErr != nil {
		return m.EmitChunkErr
	}
	m.Chunks = append(m.Chunks, data)
	return nil
}

func (m *MockEventEmitter) EmitError(err error) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Errors = append(m.Errors, err)
	return nil
}

func (m *MockEventEmitter) EmitDone() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.EmitDoneErr != nil {
		return m.EmitDoneErr
	}
	m.DoneCalled = true
	return nil
}

func (m *MockEventEmitter) EmitResponseStart() error {
	return nil
}

func (m *MockEventEmitter) EmitMessageStart(itemID string) error {
	return nil
}

func (m *MockEventEmitter) EmitMessageEnd(text string, annotations []pipeline.Annotation) error {
	return nil
}

func TestTinfoilResponderComplete(t *testing.T) {
	mockClient := &MockChatClient{
		NewFunc: func(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
			return &openai.ChatCompletion{
				ID:      "chatcmpl-123",
				Model:   "gpt-4",
				Object:  "chat.completion",
				Created: 1234567890,
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Content: "Hello【1】 world",
						},
					},
				},
			}, nil
		},
	}

	responder := NewTinfoilResponder(mockClient)

	params := pipeline.ResponderParams{
		Model:    "gpt-4",
		Messages: []openai.ChatCompletionMessageParamUnion{},
	}

	result, err := responder.Complete(context.Background(), params)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.ID != "chatcmpl-123" {
		t.Errorf("expected ID chatcmpl-123, got %s", result.ID)
	}

	if result.Model != "gpt-4" {
		t.Errorf("expected model gpt-4, got %s", result.Model)
	}

	if result.Content != "Hello【1】 world" {
		t.Errorf("expected content 'Hello【1】 world', got %q", result.Content)
	}
}

func TestTinfoilResponderCompleteError(t *testing.T) {
	expectedErr := errors.New("API error")
	mockClient := &MockChatClient{
		NewFunc: func(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
			return nil, expectedErr
		},
	}

	responder := NewTinfoilResponder(mockClient)

	params := pipeline.ResponderParams{
		Model:    "gpt-4",
		Messages: []openai.ChatCompletionMessageParamUnion{},
	}

	_, err := responder.Complete(context.Background(), params)
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	if !errors.Is(err, expectedErr) {
		t.Errorf("expected error %v, got %v", expectedErr, err)
	}
}

func TestTinfoilResponderCompleteWithParams(t *testing.T) {
	var capturedParams openai.ChatCompletionNewParams
	mockClient := &MockChatClient{
		NewFunc: func(ctx context.Context, params openai.ChatCompletionNewParams, opts ...option.RequestOption) (*openai.ChatCompletion, error) {
			capturedParams = params
			return &openai.ChatCompletion{
				Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Content: "test"}}},
			}, nil
		},
	}

	responder := NewTinfoilResponder(mockClient)

	temp := 0.7
	maxTokens := int64(100)
	params := pipeline.ResponderParams{
		Model:       "gpt-4",
		Messages:    []openai.ChatCompletionMessageParamUnion{},
		Temperature: &temp,
		MaxTokens:   &maxTokens,
	}

	_, err := responder.Complete(context.Background(), params)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify that temperature and max_tokens were set
	if !capturedParams.Temperature.Valid() {
		t.Error("expected temperature to be set")
	}

	if !capturedParams.MaxTokens.Valid() {
		t.Error("expected max_tokens to be set")
	}
}

func TestNewTinfoilResponder(t *testing.T) {
	mockClient := &MockChatClient{}
	responder := NewTinfoilResponder(mockClient)

	if responder == nil {
		t.Fatal("expected non-nil responder")
	}

	if responder.client != mockClient {
		t.Error("expected client to be set")
	}
}
