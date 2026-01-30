package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

func TestNewSSEEmitter_Success(t *testing.T) {
	w := httptest.NewRecorder()

	emitter, err := NewSSEEmitter(w)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if emitter == nil {
		t.Fatal("expected non-nil emitter")
	}

	if w.Header().Get("Content-Type") != "text/event-stream" {
		t.Errorf("expected Content-Type text/event-stream, got %s", w.Header().Get("Content-Type"))
	}

	if w.Header().Get("Cache-Control") != "no-cache" {
		t.Errorf("expected Cache-Control no-cache, got %s", w.Header().Get("Cache-Control"))
	}

	if w.Header().Get("Connection") != "keep-alive" {
		t.Errorf("expected Connection keep-alive, got %s", w.Header().Get("Connection"))
	}
}

// nonFlushingWriter is a ResponseWriter that doesn't implement Flusher
type nonFlushingWriter struct {
	header http.Header
}

func (w *nonFlushingWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}

func (w *nonFlushingWriter) Write(data []byte) (int, error) {
	return len(data), nil
}

func (w *nonFlushingWriter) WriteHeader(statusCode int) {}

func TestNewSSEEmitter_NoFlusher(t *testing.T) {
	w := &nonFlushingWriter{}

	_, err := NewSSEEmitter(w)
	if err == nil {
		t.Fatal("expected error for non-flushing writer")
	}

	if !strings.Contains(err.Error(), "streaming not supported") {
		t.Errorf("expected 'streaming not supported' error, got: %v", err)
	}
}

func TestSSEEmitter_EmitSearchCall(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, _ := NewSSEEmitter(w)

	err := emitter.EmitSearchCall("call_123", "in_progress", "test query", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	body := w.Body.String()
	if !strings.Contains(body, "data:") {
		t.Error("expected SSE data prefix")
	}
	if !strings.Contains(body, "web_search_call") {
		t.Error("expected web_search_call type")
	}
	if !strings.Contains(body, "call_123") {
		t.Error("expected call ID")
	}
	if !strings.Contains(body, "in_progress") {
		t.Error("expected status")
	}
	if !strings.Contains(body, "test query") {
		t.Error("expected query")
	}
}

func TestSSEEmitter_EmitSearchCall_NoQuery(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, _ := NewSSEEmitter(w)

	err := emitter.EmitSearchCall("call_123", "completed", "", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	body := w.Body.String()
	if strings.Contains(body, `"action"`) {
		t.Error("expected no action when query is empty")
	}
}

func TestSSEEmitter_EmitSearchCall_Blocked(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, _ := NewSSEEmitter(w)

	err := emitter.EmitSearchCall("call_123", "blocked", "john.smith@gmail.com", "email identifies individual")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	body := w.Body.String()
	if !strings.Contains(body, "blocked") {
		t.Error("expected blocked status")
	}
	if !strings.Contains(body, "john.smith@gmail.com") {
		t.Error("expected query in action")
	}
	if !strings.Contains(body, "email identifies individual") {
		t.Error("expected reason")
	}
}

func TestSSEEmitter_EmitMetadata(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, _ := NewSSEEmitter(w)

	annotations := []pipeline.Annotation{
		{
			Type: "url_citation",
			URLCitation: pipeline.URLCitation{
				URL:   "https://example.com",
				Title: "Test Title",
			},
		},
	}

	err := emitter.EmitMetadata(annotations, "Search reasoning")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	body := w.Body.String()
	if !strings.Contains(body, "data:") {
		t.Error("expected SSE data prefix")
	}
	if !strings.Contains(body, "chat.completion.chunk") {
		t.Error("expected chat.completion.chunk object")
	}
	if !strings.Contains(body, "Test Title") {
		t.Error("expected annotation title")
	}
	if !strings.Contains(body, "https://example.com") {
		t.Error("expected annotation URL")
	}
	if !strings.Contains(body, "Search reasoning") {
		t.Error("expected search reasoning")
	}
}

func TestSSEEmitter_EmitMetadata_Empty(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, _ := NewSSEEmitter(w)

	err := emitter.EmitMetadata(nil, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	body := w.Body.String()
	if body != "" {
		t.Errorf("expected empty body for empty metadata, got: %s", body)
	}
}

func TestSSEEmitter_EmitChunk(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, _ := NewSSEEmitter(w)

	chunk := []byte(`{"id":"chunk_1","choices":[{"delta":{"content":"Hello"}}]}`)
	err := emitter.EmitChunk(chunk)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	body := w.Body.String()
	if !strings.Contains(body, "data: {") {
		t.Error("expected SSE data prefix with JSON")
	}
	if !strings.Contains(body, "chunk_1") {
		t.Error("expected chunk ID in output")
	}
	if !strings.Contains(body, "Hello") {
		t.Error("expected content in output")
	}
}

func TestSSEEmitter_EmitError(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, _ := NewSSEEmitter(w)

	testErr := &testError{msg: "something went wrong"}
	err := emitter.EmitError(testErr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	body := w.Body.String()
	if !strings.Contains(body, "data:") {
		t.Error("expected SSE data prefix")
	}
	if !strings.Contains(body, "error") {
		t.Error("expected error key")
	}
	if !strings.Contains(body, "something went wrong") {
		t.Error("expected error message")
	}
	if !strings.Contains(body, "api_error") {
		t.Error("expected api_error type")
	}
}

type testError struct {
	msg string
}

func (e *testError) Error() string {
	return e.msg
}

func TestSSEEmitter_EmitDone(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, _ := NewSSEEmitter(w)

	err := emitter.EmitDone()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	body := w.Body.String()
	if !strings.Contains(body, "data: [DONE]") {
		t.Errorf("expected 'data: [DONE]', got: %s", body)
	}
}

func TestSSEEmitter_ImplementsEventEmitter(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, _ := NewSSEEmitter(w)

	// Verify the interface is implemented by using it as EventEmitter
	var _ pipeline.EventEmitter = emitter
}

func TestSSEEmitter_MultipleEvents(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, _ := NewSSEEmitter(w)

	// Emit multiple events in sequence
	emitter.EmitSearchCall("call_1", "in_progress", "query 1", "")
	emitter.EmitSearchCall("call_1", "completed", "", "")
	emitter.EmitChunk([]byte(`{"content":"hello"}`))
	emitter.EmitDone()

	body := w.Body.String()

	// Each event should be separated by double newlines
	events := strings.Split(body, "\n\n")
	// Last element is empty due to trailing \n\n
	if len(events) < 4 {
		t.Errorf("expected at least 4 events, got %d", len(events))
	}

	if !strings.Contains(body, "in_progress") {
		t.Error("expected in_progress status")
	}
	if !strings.Contains(body, "completed") {
		t.Error("expected completed status")
	}
	if !strings.Contains(body, "[DONE]") {
		t.Error("expected [DONE]")
	}
}
