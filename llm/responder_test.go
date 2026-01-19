package llm

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"

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
}

type mockMetadataCall struct {
	Annotations []pipeline.Annotation
	Reasoning   string
}

func (m *MockEventEmitter) EmitSearchCall(id, status, query string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.SearchCalls = append(m.SearchCalls, mockSearchCall{ID: id, Status: status, Query: query})
	return nil
}

func (m *MockEventEmitter) EmitMetadata(annotations []pipeline.Annotation, reasoning string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.EmitMetadataErr != nil {
		return m.EmitMetadataErr
	}
	m.MetadataCalls = append(m.MetadataCalls, mockMetadataCall{Annotations: annotations, Reasoning: reasoning})
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

func TestStripCitationMarkers(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "simple citation",
			input:    "Hello【1】world",
			expected: "Helloworld",
		},
		{
			name:     "citation with line reference",
			input:    "Hello【1†L1-L15】world",
			expected: "Helloworld",
		},
		{
			name:     "multiple citations",
			input:    "First【1】second【2】third【3】",
			expected: "Firstsecondthird",
		},
		{
			name:     "no citations",
			input:    "Hello world",
			expected: "Hello world",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "citation at start",
			input:    "【1】Hello",
			expected: "Hello",
		},
		{
			name:     "citation at end",
			input:    "Hello【1】",
			expected: "Hello",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := StripCitationMarkers(tt.input)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestStripCitationMarkersFromChunk(t *testing.T) {
	tests := []struct {
		name            string
		chunkData       string
		originalContent string
		expectChanged   bool
	}{
		{
			name:            "content with citation",
			chunkData:       `{"choices":[{"delta":{"content":"Hello【1】"}}]}`,
			originalContent: "Hello【1】",
			expectChanged:   true,
		},
		{
			name:            "content without citation",
			chunkData:       `{"choices":[{"delta":{"content":"Hello"}}]}`,
			originalContent: "Hello",
			expectChanged:   false,
		},
		{
			name:            "invalid json",
			chunkData:       `not json`,
			originalContent: "Hello【1】",
			expectChanged:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := stripCitationMarkersFromChunk(tt.chunkData, tt.originalContent)
			changed := result != tt.chunkData
			if changed != tt.expectChanged {
				t.Errorf("expected changed=%v, got changed=%v", tt.expectChanged, changed)
			}
		})
	}
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

	params := ResponderParams{
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

	// Content should have citation markers stripped
	if result.Content != "Hello world" {
		t.Errorf("expected content 'Hello world', got %q", result.Content)
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

	params := ResponderParams{
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
	params := ResponderParams{
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
