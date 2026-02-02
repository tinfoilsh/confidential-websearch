package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

// SSEEmitter implements pipeline.EventEmitter for Server-Sent Events
type SSEEmitter struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

// NewSSEEmitter creates a new SSE emitter from a response writer
func NewSSEEmitter(w http.ResponseWriter) (*SSEEmitter, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("streaming not supported")
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	return &SSEEmitter{w: w, flusher: flusher}, nil
}

// emit writes SSE data and flushes
func (e *SSEEmitter) emit(data []byte) error {
	if _, err := fmt.Fprintf(e.w, "data: %s\n\n", data); err != nil {
		return err
	}
	e.flusher.Flush()
	return nil
}

// EmitSearchCall emits a custom web search status event.
// Wrapped in chat.completion.chunk envelope so SDKs don't fail parsing.
func (e *SSEEmitter) EmitSearchCall(id, status, query, reason string, created int64, model string) error {
	event := WebSearchCall{
		Type:    "web_search_call",
		ID:      id,
		Status:  status,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []any{},
	}

	if created == 0 {
		event.Created = time.Now().Unix()
	}

	if query != "" {
		event.Action = &WebSearchAction{
			Type:  "search",
			Query: query,
		}
	}

	if reason != "" {
		event.Reason = reason
	}

	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	return e.emit(data)
}

// EmitMetadata emits annotations and reasoning as a custom chunk.
// This is an extension - OpenAI doesn't stream annotations in Chat Completions.
func (e *SSEEmitter) EmitMetadata(id string, created int64, model string, annotations []pipeline.Annotation, reasoning string) error {
	if len(annotations) == 0 && reasoning == "" {
		return nil
	}

	chunk := StreamingChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []StreamingChoice{
			{
				Index: 0,
				Delta: StreamingDelta{
					Annotations:     annotations,
					SearchReasoning: reasoning,
				},
			},
		},
	}

	data, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	return e.emit(data)
}

// EmitChunk emits a raw data chunk
func (e *SSEEmitter) EmitChunk(data []byte) error {
	return e.emit(data)
}

// EmitError emits an error event
func (e *SSEEmitter) EmitError(err error) error {
	errData, marshalErr := json.Marshal(map[string]any{
		"error": map[string]string{
			"message": err.Error(),
			"type":    "api_error",
		},
	})
	if marshalErr != nil {
		return marshalErr
	}
	return e.emit(errData)
}

// EmitDone emits the final done signal
func (e *SSEEmitter) EmitDone() error {
	return e.emit([]byte("[DONE]"))
}

// Verify SSEEmitter implements EventEmitter
var _ pipeline.EventEmitter = (*SSEEmitter)(nil)
