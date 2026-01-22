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
	"time"

	"github.com/openai/openai-go/v2/option"

	"github.com/tinfoilsh/confidential-websearch/agent"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
	"github.com/tinfoilsh/confidential-websearch/search"
)

// MockPipeline implements a test double for Pipeline
type MockPipeline struct {
	ExecuteFunc func(ctx context.Context, req *pipeline.Request, emitter pipeline.EventEmitter, reqOpts ...option.RequestOption) (*pipeline.Context, error)
}

func (m *MockPipeline) Execute(ctx context.Context, req *pipeline.Request, emitter pipeline.EventEmitter, reqOpts ...option.RequestOption) (*pipeline.Context, error) {
	if m.ExecuteFunc != nil {
		return m.ExecuteFunc(ctx, req, emitter, reqOpts...)
	}
	return nil, nil
}

// TestableServer wraps Server with a mockable pipeline
type TestableServer struct {
	Server
	mockPipeline *MockPipeline
}

func (ts *TestableServer) HandleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		jsonError(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, MaxRequestBodySize)

	var req IncomingRequest
	if err := parseRequestBody(r, &req); err != nil {
		jsonError(w, err.Error(), http.StatusBadRequest)
		return
	}

	pipelineReq := &pipeline.Request{
		Model:       req.Model,
		Messages:    convertMessages(req.Messages),
		Stream:      req.Stream,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Format:      pipeline.FormatChatCompletion,
	}

	pctx, err := ts.mockPipeline.Execute(r.Context(), pipelineReq, nil)
	if pctx != nil && pctx.Cancel != nil {
		defer pctx.Cancel()
	}

	if err != nil {
		status, body := pipeline.ErrorResponse(err)
		jsonErrorResponse(w, status, body)
		return
	}

	result := pctx.ResponderResult.(*pipeline.ResponderResultData)
	annotations := pipeline.BuildAnnotations(pctx.SearchResults)

	response := ChatCompletionResponse{
		ID:      result.ID,
		Object:  result.Object,
		Created: result.Created,
		Model:   result.Model,
		Usage:   result.Usage,
		Choices: []ChatCompletionChoiceOutput{
			{
				Index:        0,
				FinishReason: "stop",
				Message: ChatCompletionMessageOutput{
					Role:            "assistant",
					Content:         result.Content,
					Annotations:     annotations,
					SearchReasoning: pctx.AgentResult.AgentReasoning,
				},
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
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

func TestHandleChatCompletions_NonStreaming_Success(t *testing.T) {
	mockPipeline := &MockPipeline{
		ExecuteFunc: func(ctx context.Context, req *pipeline.Request, emitter pipeline.EventEmitter, reqOpts ...option.RequestOption) (*pipeline.Context, error) {
			cancelCtx, cancel := context.WithCancel(ctx)
			return &pipeline.Context{
				Context: cancelCtx,
				Cancel:  cancel,
				AgentResult: &agent.Result{
					AgentReasoning: "Searched for test query",
				},
				SearchResults: []agent.ToolCall{
					{
						ID:    "call_123",
						Query: "test query",
						Results: []search.Result{
							{Title: "Result 1", URL: "https://example.com"},
						},
					},
				},
				ResponderResult: &pipeline.ResponderResultData{
					ID:      "resp_123",
					Model:   "gpt-4",
					Object:  "chat.completion",
					Created: 1234567890,
					Content: "Here is the response",
				},
			}, nil
		},
	}

	ts := &TestableServer{mockPipeline: mockPipeline}
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}],"stream":false}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	ts.HandleChatCompletions(w, req)

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

	var resp map[string]string
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["error"] != "internal server error" {
		t.Errorf("expected 'internal server error', got '%s'", resp["error"])
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

	var resp map[string]string
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["error"] != "test error message" {
		t.Errorf("expected 'test error message', got '%s'", resp["error"])
	}
}

func TestJsonErrorResponse(t *testing.T) {
	w := httptest.NewRecorder()
	body := map[string]interface{}{
		"error": map[string]string{
			"message": "detailed error",
			"type":    "invalid_request_error",
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
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there", SearchReasoning: "No search needed"},
		{Role: "user", Content: "Search for something"},
	}

	result := convertMessages(msgs)

	if len(result) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(result))
	}
	if result[0].Role != "user" || result[0].Content != "Hello" {
		t.Error("first message mismatch")
	}
	if result[1].Role != "assistant" || result[1].Content != "Hi there" {
		t.Error("second message mismatch")
	}
	if result[1].SearchReasoning != "No search needed" {
		t.Error("search reasoning not preserved")
	}
}

func TestBuildFlatAnnotations(t *testing.T) {
	toolCalls := []agent.ToolCall{
		{
			ID:    "call_1",
			Query: "query 1",
			Results: []search.Result{
				{Title: "Title 1", URL: "https://example.com/1", Content: "Content 1", PublishedDate: "2024-01-01"},
				{Title: "Title 2", URL: "https://example.com/2", Content: "Content 2"},
			},
		},
		{
			ID:    "call_2",
			Query: "query 2",
			Results: []search.Result{
				{Title: "Title 3", URL: "https://example.com/3", Content: "Content 3"},
			},
		},
	}

	annotations := buildFlatAnnotations(toolCalls)

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
	if annotations[0].PublishedDate != "2024-01-01" {
		t.Errorf("expected published date '2024-01-01', got '%s'", annotations[0].PublishedDate)
	}
}

func TestBuildFlatAnnotations_Empty(t *testing.T) {
	annotations := buildFlatAnnotations(nil)
	if annotations != nil {
		t.Errorf("expected nil for empty tool calls, got %v", annotations)
	}

	annotations = buildFlatAnnotations([]agent.ToolCall{})
	if annotations != nil {
		t.Errorf("expected nil for empty tool calls slice, got %v", annotations)
	}
}

func TestExtractRequestOptions(t *testing.T) {
	req := httptest.NewRequest("POST", "/", nil)
	req.Header.Set("Authorization", "Bearer test-token")

	opts := extractRequestOptions(req)

	if len(opts) != 1 {
		t.Errorf("expected 1 option, got %d", len(opts))
	}
}

func TestExtractRequestOptions_NoAuth(t *testing.T) {
	req := httptest.NewRequest("POST", "/", nil)

	opts := extractRequestOptions(req)

	if len(opts) != 0 {
		t.Errorf("expected 0 options, got %d", len(opts))
	}
}

func TestExtractUserQuery(t *testing.T) {
	messages := []pipeline.Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "First question"},
		{Role: "assistant", Content: "First answer"},
		{Role: "user", Content: "Second question"},
	}

	query := extractUserQuery(messages)
	if query != "Second question" {
		t.Errorf("expected 'Second question', got '%s'", query)
	}
}

func TestExtractUserQuery_NoUserMessage(t *testing.T) {
	messages := []pipeline.Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "assistant", Content: "Hello"},
	}

	query := extractUserQuery(messages)
	if query != "" {
		t.Errorf("expected empty string, got '%s'", query)
	}
}

func TestExtractUserQuery_EmptyMessages(t *testing.T) {
	query := extractUserQuery(nil)
	if query != "" {
		t.Errorf("expected empty string, got '%s'", query)
	}

	query = extractUserQuery([]pipeline.Message{})
	if query != "" {
		t.Errorf("expected empty string, got '%s'", query)
	}
}

func TestExtractUserQuery_EmptyContent(t *testing.T) {
	messages := []pipeline.Message{
		{Role: "user", Content: ""},
		{Role: "user", Content: "Valid content"},
		{Role: "user", Content: ""},
	}

	query := extractUserQuery(messages)
	if query != "Valid content" {
		t.Errorf("expected 'Valid content', got '%s'", query)
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
	mockPipeline := &MockPipeline{
		ExecuteFunc: func(ctx context.Context, req *pipeline.Request, emitter pipeline.EventEmitter, reqOpts ...option.RequestOption) (*pipeline.Context, error) {
			cancelCtx, cancel := context.WithCancel(ctx)
			return &pipeline.Context{
				Context: cancelCtx,
				Cancel:  cancel,
			}, &pipeline.PipelineError{
				Stage: "validate",
				Err:   errors.New("validation failed"),
			}
		},
	}

	ts := &TestableServer{mockPipeline: mockPipeline}
	body := `{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	ts.HandleChatCompletions(w, req)

	if w.Code == http.StatusOK {
		t.Error("expected error status code")
	}
}

func TestRequestTimeout_Constant(t *testing.T) {
	if RequestTimeout != 2*time.Minute {
		t.Errorf("expected RequestTimeout to be 2 minutes, got %v", RequestTimeout)
	}
}

func TestMaxRequestBodySize_Constant(t *testing.T) {
	expected := int64(200 << 20) // 200 MB
	if MaxRequestBodySize != expected {
		t.Errorf("expected MaxRequestBodySize to be %d, got %d", expected, MaxRequestBodySize)
	}
}
