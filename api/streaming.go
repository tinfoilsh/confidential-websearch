package api

import (
	"encoding/json"
	"fmt"
	"net/http"

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

// EmitSearchCall emits a web search call event
func (e *SSEEmitter) EmitSearchCall(id, status, query, reason string) error {
	event := WebSearchCall{
		Type:   "web_search_call",
		ID:     id,
		Status: status,
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

	if _, err := fmt.Fprintf(e.w, "data: %s\n\n", data); err != nil {
		return err
	}
	e.flusher.Flush()
	return nil
}

// EmitMetadata emits annotations and reasoning as a custom chunk
func (e *SSEEmitter) EmitMetadata(annotations []pipeline.Annotation, reasoning string) error {
	if len(annotations) == 0 && reasoning == "" {
		return nil
	}

	chunk := StreamingChunk{
		Object: "chat.completion.chunk",
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

	if _, err := fmt.Fprintf(e.w, "data: %s\n\n", data); err != nil {
		return err
	}
	e.flusher.Flush()
	return nil
}

// EmitChunk emits a raw data chunk
func (e *SSEEmitter) EmitChunk(data []byte) error {
	if _, err := fmt.Fprintf(e.w, "data: %s\n\n", data); err != nil {
		return err
	}
	e.flusher.Flush()
	return nil
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

	if _, err := fmt.Fprintf(e.w, "data: %s\n\n", errData); err != nil {
		return err
	}
	e.flusher.Flush()
	return nil
}

// EmitDone emits the final done signal
func (e *SSEEmitter) EmitDone() error {
	if _, err := fmt.Fprintf(e.w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	e.flusher.Flush()
	return nil
}

// Verify SSEEmitter implements EventEmitter
var _ pipeline.EventEmitter = (*SSEEmitter)(nil)
