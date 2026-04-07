package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tinfoilsh/confidential-websearch/agent"
	"github.com/tinfoilsh/confidential-websearch/engine"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
	"github.com/tinfoilsh/confidential-websearch/search"
)

// toJSON marshals a string to json.RawMessage for tests
func toJSON(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

type MockRunner struct {
	RunFunc    func(ctx context.Context, req *pipeline.Request) (*engine.Result, error)
	StreamFunc func(ctx context.Context, req *pipeline.Request, emitter pipeline.EventEmitter) (*engine.Result, error)
}

func (m *MockRunner) Run(ctx context.Context, req *pipeline.Request) (*engine.Result, error) {
	if m.RunFunc != nil {
		return m.RunFunc(ctx, req)
	}
	return nil, nil
}

func (m *MockRunner) Stream(ctx context.Context, req *pipeline.Request, emitter pipeline.EventEmitter) (*engine.Result, error) {
	if m.StreamFunc != nil {
		return m.StreamFunc(ctx, req, emitter)
	}
	return nil, nil
}

func TestHandleHealth(t *testing.T) {
	srv := &Server{}
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	srv.HandleHealth(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp map[string]string
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["status"] != "ok" {
		t.Errorf("expected status ok, got %s", resp["status"])
	}
}

func TestHandleRoot(t *testing.T) {
	srv := &Server{}
	req := httptest.NewRequest("GET", "/", nil)
	w := httptest.NewRecorder()

	srv.HandleRoot(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp map[string]string
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["service"] != "confidential-websearch" {
		t.Errorf("expected service 'confidential-websearch', got %s", resp["service"])
	}
	if resp["status"] != "ok" {
		t.Errorf("expected status ok, got %s", resp["status"])
	}
}

func TestHandleChatCompletions_MethodNotAllowed(t *testing.T) {
	srv := &Server{}
	methods := []string{"GET", "PUT", "DELETE", "PATCH"}

	for _, method := range methods {
		req := httptest.NewRequest(method, "/v1/chat/completions", nil)
		w := httptest.NewRecorder()

		srv.HandleChatCompletions(w, req)

		if w.Code != http.StatusMethodNotAllowed {
			t.Errorf("%s: expected 405, got %d", method, w.Code)
		}
	}
}

func TestHandleChatCompletions_InvalidJSON(t *testing.T) {
	srv := &Server{}
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("not valid json"))
	w := httptest.NewRecorder()

	srv.HandleChatCompletions(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestHandleChatCompletions_EmptyBody(t *testing.T) {
	srv := &Server{}
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(""))
	w := httptest.NewRecorder()

	srv.HandleChatCompletions(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestHandleChatCompletions_InvalidSearchContextSize(t *testing.T) {
	srv := &Server{Runner: engine.NewService(nil, nil, nil, nil)}
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{
		"model":"gpt-4",
		"messages":[{"role":"user","content":"Hello"}],
		"web_search_options":{"search_context_size":"maximum"}
	}`))
	w := httptest.NewRecorder()

	srv.HandleChatCompletions(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

func TestHandleChatCompletions_NonStreaming_Success(t *testing.T) {
	mockRunner := &MockRunner{
		RunFunc: func(ctx context.Context, req *pipeline.Request) (*engine.Result, error) {
			return &engine.Result{
				ID:              "resp_123",
				Model:           "gpt-4",
				Object:          "chat.completion",
				Created:         1234567890,
				Content:         "Here is the response",
				SearchReasoning: "Searched for test query",
				SearchResults: []agent.ToolCall{
					{
						ID:    "call_123",
						Query: "test query",
						Results: []search.Result{
							{Title: "Result 1", URL: "https://example.com"},
						},
					},
				},
			}, nil
		},
	}

	srv := &Server{Runner: mockRunner}
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}],"stream":false}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	srv.HandleChatCompletions(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp ChatCompletionResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if resp.ID != "resp_123" {
		t.Errorf("expected ID 'resp_123', got '%s'", resp.ID)
	}
	if resp.Model != "gpt-4" {
		t.Errorf("expected model 'gpt-4', got '%s'", resp.Model)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}
	if resp.Choices[0].Message.Content != "Here is the response" {
		t.Errorf("expected content 'Here is the response', got '%s'", resp.Choices[0].Message.Content)
	}
	if resp.Choices[0].Message.SearchReasoning != "Searched for test query" {
		t.Errorf("expected search reasoning, got '%s'", resp.Choices[0].Message.SearchReasoning)
	}
}

func TestHandleChatCompletions_FallsBackToLegacyAnnotations(t *testing.T) {
	mockRunner := &MockRunner{
		RunFunc: func(ctx context.Context, req *pipeline.Request) (*engine.Result, error) {
			return &engine.Result{
				ID:      "resp_123",
				Model:   "gpt-4",
				Object:  "chat.completion",
				Created: 1234567890,
				Content: "Here is the response",
				SearchResults: []agent.ToolCall{
					{
						ID:    "call_123",
						Query: "test query",
						Results: []search.Result{
							{Title: "Result 1", URL: "https://example.com"},
						},
					},
				},
			}, nil
		},
	}

	srv := &Server{Runner: mockRunner}
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}],"stream":false}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	srv.HandleChatCompletions(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp ChatCompletionResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}
	if len(resp.Choices[0].Message.Annotations) != 1 {
		t.Fatalf("expected 1 fallback annotation, got %d", len(resp.Choices[0].Message.Annotations))
	}
	if resp.Choices[0].Message.Annotations[0].URLCitation.URL != "https://example.com" {
		t.Fatalf("expected fallback annotation URL https://example.com, got %q", resp.Choices[0].Message.Annotations[0].URLCitation.URL)
	}
}

func TestHandleChatCompletions_NonStreaming_PreservesFetchStatuses(t *testing.T) {
	mockRunner := &MockRunner{
		RunFunc: func(ctx context.Context, req *pipeline.Request) (*engine.Result, error) {
			return &engine.Result{
				ID:      "resp_123",
				Model:   "gpt-4",
				Object:  "chat.completion",
				Created: 1234567890,
				Content: "Here is the response",
				FetchCalls: []engine.FetchCall{
					{ID: "fetch_ok", Status: pipeline.EmitStatusCompleted, URL: "https://example.com"},
					{ID: "fetch_failed", Status: pipeline.EmitStatusFailed, URL: "https://blocked.example"},
				},
			}, nil
		},
	}

	srv := &Server{Runner: mockRunner}
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}],"stream":false}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	srv.HandleChatCompletions(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp ChatCompletionResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}
	if len(resp.Choices[0].Message.FetchCalls) != 2 {
		t.Fatalf("expected 2 fetch calls, got %d", len(resp.Choices[0].Message.FetchCalls))
	}
	if resp.Choices[0].Message.FetchCalls[1].Status != StatusFailed {
		t.Fatalf("expected failed fetch status, got %q", resp.Choices[0].Message.FetchCalls[1].Status)
	}
}

func TestHandleResponses_MethodNotAllowed(t *testing.T) {
	srv := &Server{}
	methods := []string{"GET", "PUT", "DELETE", "PATCH"}

	for _, method := range methods {
		req := httptest.NewRequest(method, "/v1/responses", nil)
		w := httptest.NewRecorder()

		srv.HandleResponses(w, req)

		if w.Code != http.StatusMethodNotAllowed {
			t.Errorf("%s: expected 405, got %d", method, w.Code)
		}
	}
}

func TestHandleResponses_InvalidJSON(t *testing.T) {
	srv := &Server{}
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader("not valid json"))
	w := httptest.NewRecorder()

	srv.HandleResponses(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestHandleResponses_InvalidSearchContextSize(t *testing.T) {
	srv := &Server{Runner: engine.NewService(nil, nil, nil, nil)}
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{
		"model":"gpt-4",
		"input":"Hello",
		"tools":[{"type":"web_search","search_context_size":"maximum"}]
	}`))
	w := httptest.NewRecorder()

	srv.HandleResponses(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

func TestHandleResponses_PreservesFetchStatuses(t *testing.T) {
	mockRunner := &MockRunner{
		RunFunc: func(ctx context.Context, req *pipeline.Request) (*engine.Result, error) {
			return &engine.Result{
				ID:      "resp_123",
				Model:   "gpt-4",
				Object:  "response",
				Created: 1234567890,
				Content: "Here is the response",
				FetchCalls: []engine.FetchCall{
					{ID: "fetch_ok", Status: pipeline.EmitStatusCompleted, URL: "https://example.com"},
					{ID: "fetch_failed", Status: pipeline.EmitStatusFailed, URL: "https://blocked.example"},
				},
			}, nil
		},
	}

	srv := &Server{Runner: mockRunner}
	body := `{"model":"gpt-4","input":"Hello","tools":[{"type":"web_search"}]}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	w := httptest.NewRecorder()

	srv.HandleResponses(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp struct {
		Output []ResponsesOutput `json:"output"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if len(resp.Output) < 2 {
		t.Fatalf("expected fetch calls in output, got %+v", resp.Output)
	}
	if resp.Output[0].Status != StatusCompleted {
		t.Fatalf("expected first fetch status completed, got %q", resp.Output[0].Status)
	}
	if resp.Output[1].Status != StatusFailed {
		t.Fatalf("expected second fetch status failed, got %q", resp.Output[1].Status)
	}
}

func TestRecoveryMiddleware_NoPanic(t *testing.T) {
	handler := RecoveryMiddleware(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("success"))
	})

	req := httptest.NewRequest("GET", "/", nil)
	w := httptest.NewRecorder()

	handler(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if w.Body.String() != "success" {
		t.Errorf("expected 'success', got '%s'", w.Body.String())
	}
}

func TestRecoveryMiddleware_Panic(t *testing.T) {
	handler := RecoveryMiddleware(func(w http.ResponseWriter, r *http.Request) {
		panic("something went wrong")
	})

	req := httptest.NewRequest("GET", "/", nil)
	w := httptest.NewRecorder()

	handler(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("expected 500, got %d", w.Code)
	}

	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	errObj, ok := resp["error"].(map[string]any)
	if !ok {
		t.Fatal("expected error object in response")
	}
	if errObj["message"] != "internal server error" {
		t.Errorf("expected 'internal server error', got '%s'", errObj["message"])
	}
}

func TestJsonError(t *testing.T) {
	w := httptest.NewRecorder()
	jsonError(w, "test error message", http.StatusBadRequest)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
	if w.Header().Get("Content-Type") != "application/json" {
		t.Errorf("expected Content-Type application/json")
	}

	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	errObj, ok := resp["error"].(map[string]any)
	if !ok {
		t.Fatal("expected error object in response")
	}
	if errObj["message"] != "test error message" {
		t.Errorf("expected 'test error message', got '%s'", errObj["message"])
	}
}

func TestJsonErrorResponse(t *testing.T) {
	w := httptest.NewRecorder()
	body := map[string]any{
		"error": map[string]string{
			"message": "detailed error",
			"type":    pipeline.ErrTypeInvalidRequest,
		},
	}
	jsonErrorResponse(w, http.StatusUnprocessableEntity, body)

	if w.Code != http.StatusUnprocessableEntity {
		t.Errorf("expected 422, got %d", w.Code)
	}
	if w.Header().Get("Content-Type") != "application/json" {
		t.Errorf("expected Content-Type application/json")
	}
}

func TestConvertMessages(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: toJSON("Hello")},
		{Role: "assistant", Content: toJSON("Hi there")},
		{Role: "user", Content: toJSON("Search for something")},
	}

	result := convertMessages(msgs)

	if len(result) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(result))
	}
	if result[0].Role != "user" || result[0].GetTextContent() != "Hello" {
		t.Error("first message mismatch")
	}
	if result[1].Role != "assistant" || result[1].GetTextContent() != "Hi there" {
		t.Error("second message mismatch")
	}
}

func TestBuildFlatAnnotations(t *testing.T) {
	annotationsInput := []pipeline.Annotation{
		{
			Type: pipeline.AnnotationTypeURLCitation,
			URLCitation: pipeline.URLCitation{
				Title:      "Title 1",
				URL:        "https://example.com/1",
				StartIndex: 10,
				EndIndex:   14,
			},
		},
		{
			Type: pipeline.AnnotationTypeURLCitation,
			URLCitation: pipeline.URLCitation{
				Title:      "Title 2",
				URL:        "https://example.com/2",
				StartIndex: 20,
				EndIndex:   24,
			},
		},
		{
			Type: pipeline.AnnotationTypeURLCitation,
			URLCitation: pipeline.URLCitation{
				Title:      "Title 3",
				URL:        "https://example.com/3",
				StartIndex: 30,
				EndIndex:   34,
			},
		},
	}

	annotations := buildFlatAnnotations(annotationsInput)

	if len(annotations) != 3 {
		t.Fatalf("expected 3 annotations, got %d", len(annotations))
	}

	if annotations[0].Type != "url_citation" {
		t.Errorf("expected type 'url_citation', got '%s'", annotations[0].Type)
	}
	if annotations[0].Title != "Title 1" {
		t.Errorf("expected title 'Title 1', got '%s'", annotations[0].Title)
	}
	if annotations[0].URL != "https://example.com/1" {
		t.Errorf("expected URL 'https://example.com/1', got '%s'", annotations[0].URL)
	}
	if annotations[0].StartIndex != 10 || annotations[0].EndIndex != 14 {
		t.Errorf("expected start/end indexes 10/14, got %d/%d", annotations[0].StartIndex, annotations[0].EndIndex)
	}
}

func TestBuildFlatAnnotations_Empty(t *testing.T) {
	annotations := buildFlatAnnotations(nil)
	if annotations != nil {
		t.Errorf("expected nil for empty tool calls, got %v", annotations)
	}

	annotations = buildFlatAnnotations([]pipeline.Annotation{})
	if annotations != nil {
		t.Errorf("expected nil for empty annotations slice, got %v", annotations)
	}
}

func TestParseRequestBody_Success(t *testing.T) {
	body := bytes.NewReader([]byte(`{"model":"gpt-4","messages":[]}`))
	req := httptest.NewRequest("POST", "/", body)

	var result IncomingRequest
	err := parseRequestBody(req, &result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Model != "gpt-4" {
		t.Errorf("expected model 'gpt-4', got '%s'", result.Model)
	}
}

func TestParseRequestBody_InvalidJSON(t *testing.T) {
	body := bytes.NewReader([]byte(`not valid json`))
	req := httptest.NewRequest("POST", "/", body)

	var result IncomingRequest
	err := parseRequestBody(req, &result)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}

	if !strings.Contains(err.Error(), "failed to parse request") {
		t.Errorf("expected parse error, got: %v", err)
	}
}

func TestParseRequestBody_EmptyBody(t *testing.T) {
	body := bytes.NewReader([]byte(``))
	req := httptest.NewRequest("POST", "/", body)

	var result IncomingRequest
	err := parseRequestBody(req, &result)
	if err == nil {
		t.Fatal("expected error for empty body")
	}
}

func TestChatCompletions_PipelineError(t *testing.T) {
	mockRunner := &MockRunner{
		RunFunc: func(ctx context.Context, req *pipeline.Request) (*engine.Result, error) {
			return nil, &pipeline.PipelineError{
				Stage: "validate",
				Err:   errors.New("validation failed"),
			}
		},
	}

	srv := &Server{Runner: mockRunner}
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	srv.HandleChatCompletions(w, req)

	if w.Code == http.StatusOK {
		t.Error("expected error status code")
	}
}

func TestMaxRequestBodySize_Constant(t *testing.T) {
	expected := int64(200 << 20) // 200 MB
	if MaxRequestBodySize != expected {
		t.Errorf("expected MaxRequestBodySize to be %d, got %d", expected, MaxRequestBodySize)
	}
}
