package llm

import (
	"context"
	"errors"
	"strings"
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
	Reason string
}

type mockMetadataCall struct {
	Annotations    []pipeline.Annotation
	Reasoning      string
	ReasoningItems []pipeline.ReasoningItem
}

func (m *MockEventEmitter) EmitSearchCall(id, status, query, reason string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.SearchCalls = append(m.SearchCalls, mockSearchCall{ID: id, Status: status, Query: query, Reason: reason})
	return nil
}

func (m *MockEventEmitter) EmitMetadata(annotations []pipeline.Annotation, reasoning string, reasoningItems []pipeline.ReasoningItem) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.EmitMetadataErr != nil {
		return m.EmitMetadataErr
	}
	m.MetadataCalls = append(m.MetadataCalls, mockMetadataCall{Annotations: annotations, Reasoning: reasoning, ReasoningItems: reasoningItems})
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

func TestUpdateChunkContent(t *testing.T) {
	tests := []struct {
		name       string
		chunkData  string
		newContent string
		wantJSON   bool
	}{
		{
			name:       "update content",
			chunkData:  `{"choices":[{"delta":{"content":"Hello【1】"}}]}`,
			newContent: "Hello",
			wantJSON:   true,
		},
		{
			name:       "invalid json returns original",
			chunkData:  `not json`,
			newContent: "Hello",
			wantJSON:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := updateChunkContent(tt.chunkData, tt.newContent)
			if tt.wantJSON {
				if !strings.Contains(result, tt.newContent) {
					t.Errorf("expected result to contain %q, got %q", tt.newContent, result)
				}
			} else {
				if result != tt.chunkData {
					t.Errorf("expected unchanged chunkData, got %q", result)
				}
			}
		})
	}
}

func TestStreamingCitationStripper(t *testing.T) {
	tests := []struct {
		name     string
		chunks   []string
		expected string
	}{
		{
			name:     "marker in single chunk",
			chunks:   []string{"Hello【1】world"},
			expected: "Helloworld",
		},
		{
			name:     "marker spans two chunks",
			chunks:   []string{"Hello【1", "†L1-L9】world"},
			expected: "Helloworld",
		},
		{
			name:     "marker spans three chunks",
			chunks:   []string{"Hello【", "4†L1", "-L9】world"},
			expected: "Helloworld",
		},
		{
			name:     "multiple markers",
			chunks:   []string{"A【1】B【2】C"},
			expected: "ABC",
		},
		{
			name:     "no markers",
			chunks:   []string{"Hello ", "world"},
			expected: "Hello world",
		},
		{
			name:     "incomplete marker at end",
			chunks:   []string{"Hello【1"},
			expected: "Hello【1",
		},
		{
			name:     "invalid marker (no digit)",
			chunks:   []string{"Hello【abc】world"},
			expected: "Hello【abc】world",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stripper := NewStreamingCitationStripper()
			var result strings.Builder
			for _, chunk := range tt.chunks {
				result.WriteString(stripper.Process(chunk))
			}
			result.WriteString(stripper.Flush())
			if result.String() != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result.String())
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
