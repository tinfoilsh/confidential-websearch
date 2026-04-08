package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	openai "github.com/openai/openai-go/v3"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

// ResponsesEmitter implements pipeline.EventEmitter for Responses API streaming.
// Emits OpenAI-conformant response.* events.
type ResponsesEmitter struct {
	w             http.ResponseWriter
	flusher       http.Flusher
	responseID    string
	model         string
	createdAt     int64
	seqNum        atomic.Int64
	mu            sync.Mutex
	nextOutputIdx int
	itemIndexes   map[string]int
	messageItemID string
	messageIdx    int
	reasoning     string
}

// NewResponsesEmitter creates a new Responses API SSE emitter
func NewResponsesEmitter(w http.ResponseWriter, responseID, model string) (*ResponsesEmitter, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("streaming not supported")
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	return &ResponsesEmitter{
		w:           w,
		flusher:     flusher,
		responseID:  responseID,
		model:       model,
		createdAt:   time.Now().Unix(),
		itemIndexes: make(map[string]int),
	}, nil
}

// emit writes an SSE event with type and data
func (e *ResponsesEmitter) emit(eventType string, data any) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(e.w, "event: %s\ndata: %s\n\n", eventType, jsonData); err != nil {
		return err
	}
	e.flusher.Flush()
	return nil
}

func (e *ResponsesEmitter) nextSeq() int64 {
	return e.seqNum.Add(1)
}

func (e *ResponsesEmitter) reserveOutputIndex(itemID string) int {
	e.mu.Lock()
	defer e.mu.Unlock()

	if index, ok := e.itemIndexes[itemID]; ok {
		return index
	}

	index := e.nextOutputIdx
	e.itemIndexes[itemID] = index
	e.nextOutputIdx++
	return index
}

func (e *ResponsesEmitter) outputIndexFor(itemID string) int {
	e.mu.Lock()
	defer e.mu.Unlock()

	if index, ok := e.itemIndexes[itemID]; ok {
		return index
	}

	index := e.nextOutputIdx
	e.itemIndexes[itemID] = index
	e.nextOutputIdx++
	return index
}

func (e *ResponsesEmitter) currentMessageIndex() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.messageIdx
}

// EmitResponseStart emits response.created and response.in_progress events
func (e *ResponsesEmitter) EmitResponseStart() error {
	// Emit response.created
	if err := e.emit("response.created", map[string]any{
		"type":            "response.created",
		"sequence_number": e.nextSeq(),
		"response": map[string]any{
			"id":         e.responseID,
			"object":     ObjectResponse,
			"created_at": e.createdAt,
			"status":     StatusInProgress,
			"model":      e.model,
			"output":     []any{},
		},
	}); err != nil {
		return err
	}

	// Emit response.in_progress
	return e.emit("response.in_progress", map[string]any{
		"type":            "response.in_progress",
		"sequence_number": e.nextSeq(),
		"response": map[string]any{
			"id":         e.responseID,
			"object":     ObjectResponse,
			"created_at": e.createdAt,
			"status":     StatusInProgress,
			"model":      e.model,
		},
	})
}

// emitWebSearchCallEvents emits the standard event sequence for a web_search_call item
func (e *ResponsesEmitter) emitWebSearchCallEvents(itemID, status string, action map[string]any, reason string) error {
	switch status {
	case StatusInProgress:
		outputIdx := e.reserveOutputIndex(itemID)
		item := map[string]any{
			"type":   ItemTypeWebSearchCall,
			"id":     itemID,
			"status": StatusInProgress,
			"action": action,
		}
		if err := e.emit("response.output_item.added", map[string]any{
			"type":            "response.output_item.added",
			"sequence_number": e.nextSeq(),
			"output_index":    outputIdx,
			"item":            item,
		}); err != nil {
			return err
		}
		return e.emit("response.web_search_call.in_progress", map[string]any{
			"type":            "response.web_search_call.in_progress",
			"sequence_number": e.nextSeq(),
			"output_index":    outputIdx,
			"item_id":         itemID,
		})

	case StatusCompleted:
		outputIdx := e.outputIndexFor(itemID)
		if err := e.emit("response.web_search_call.completed", map[string]any{
			"type":            "response.web_search_call.completed",
			"sequence_number": e.nextSeq(),
			"output_index":    outputIdx,
			"item_id":         itemID,
		}); err != nil {
			return err
		}
		if err := e.emit("response.output_item.done", map[string]any{
			"type":            "response.output_item.done",
			"sequence_number": e.nextSeq(),
			"output_index":    outputIdx,
			"item": map[string]any{
				"type":   ItemTypeWebSearchCall,
				"id":     itemID,
				"status": StatusCompleted,
				"action": action,
			},
		}); err != nil {
			return err
		}
		return nil

	case StatusBlocked, StatusFailed:
		outputIdx := e.reserveOutputIndex(itemID)
		item := map[string]any{
			"type":   ItemTypeWebSearchCall,
			"id":     itemID,
			"status": status,
		}
		if reason != "" {
			item["reason"] = reason
		}
		if action != nil {
			item["action"] = action
		}
		if err := e.emit("response.output_item.added", map[string]any{
			"type":            "response.output_item.added",
			"sequence_number": e.nextSeq(),
			"output_index":    outputIdx,
			"item":            item,
		}); err != nil {
			return err
		}
		// Done item omits action
		doneItem := map[string]any{
			"type":   ItemTypeWebSearchCall,
			"id":     itemID,
			"status": status,
		}
		if reason != "" {
			doneItem["reason"] = reason
		}
		if err := e.emit("response.output_item.done", map[string]any{
			"type":            "response.output_item.done",
			"sequence_number": e.nextSeq(),
			"output_index":    outputIdx,
			"item":            doneItem,
		}); err != nil {
			return err
		}
		return nil
	}

	return nil
}

// EmitSearchCall emits web search status events (response.web_search_call.*)
func (e *ResponsesEmitter) EmitSearchCall(id, status, query, reason string, created int64, model string) error {
	itemID := IDPrefixWebSearch + id

	if status == "searching" {
		outputIdx := e.outputIndexFor(itemID)
		return e.emit("response.web_search_call.searching", map[string]any{
			"type":            "response.web_search_call.searching",
			"sequence_number": e.nextSeq(),
			"output_index":    outputIdx,
			"item_id":         itemID,
		})
	}

	action := map[string]any{
		"type":  ActionTypeSearch,
		"query": query,
	}
	return e.emitWebSearchCallEvents(itemID, status, action, reason)
}

// EmitFetchCall emits URL fetch status events (action.type "open_page")
func (e *ResponsesEmitter) EmitFetchCall(id, status, url string, created int64, model string) error {
	itemID := IDPrefixWebSearch + id

	action := map[string]any{
		"type": ActionTypeOpenPage,
		"url":  url,
	}
	return e.emitWebSearchCallEvents(itemID, status, action, "")
}

// EmitMetadata stores reasoning for later emission in EmitMessageEnd.
// Annotations are emitted via EmitMessageEnd, not here.
func (e *ResponsesEmitter) EmitMetadata(id string, created int64, model string, annotations []pipeline.Annotation, reasoning string) error {
	e.reasoning = reasoning
	return nil
}

// EmitMessageStart emits the message output item start
func (e *ResponsesEmitter) EmitMessageStart(itemID string) error {
	e.messageItemID = itemID
	outputIdx := e.reserveOutputIndex(itemID)
	e.mu.Lock()
	e.messageIdx = outputIdx
	e.mu.Unlock()

	// Emit output_item.added for message
	if err := e.emit("response.output_item.added", map[string]any{
		"type":            "response.output_item.added",
		"sequence_number": e.nextSeq(),
		"output_index":    outputIdx,
		"item": map[string]any{
			"type":   ItemTypeMessage,
			"id":     itemID,
			"role":   RoleAssistant,
			"status": StatusInProgress,
		},
	}); err != nil {
		return err
	}

	// Emit content_part.added
	return e.emit("response.content_part.added", map[string]any{
		"type":            "response.content_part.added",
		"sequence_number": e.nextSeq(),
		"output_index":    outputIdx,
		"content_index":   0,
		"part": map[string]any{
			"type": ContentTypeOutputText,
			"text": "",
		},
	})
}

// EmitChunk emits a content delta (response.output_text.delta)
func (e *ResponsesEmitter) EmitChunk(data []byte) error {
	// Parse the incoming chunk to extract the delta content
	var chunk struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(data, &chunk); err != nil {
		// If we can't parse, skip
		return nil
	}

	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta.Content == "" {
		return nil
	}

	outputIdx := e.currentMessageIndex()
	return e.emit("response.output_text.delta", map[string]any{
		"type":            "response.output_text.delta",
		"sequence_number": e.nextSeq(),
		"output_index":    outputIdx,
		"content_index":   0,
		"delta":           chunk.Choices[0].Delta.Content,
	})
}

// EmitMessageEnd emits the message output item completion
func (e *ResponsesEmitter) EmitMessageEnd(text string, annotations []pipeline.Annotation) error {
	outputIdx := e.currentMessageIndex()

	// Emit annotations
	for i, ann := range annotations {
		if err := e.emit("response.output_text.annotation.added", map[string]any{
			"type":             "response.output_text.annotation.added",
			"sequence_number":  e.nextSeq(),
			"output_index":     outputIdx,
			"content_index":    0,
			"annotation_index": i,
			"annotation": map[string]any{
				"type":        ContentTypeURLCitation,
				"url":         ann.URLCitation.URL,
				"title":       ann.URLCitation.Title,
				"start_index": ann.URLCitation.StartIndex,
				"end_index":   ann.URLCitation.EndIndex,
			},
		}); err != nil {
			return err
		}
	}

	// Emit output_text.done
	if err := e.emit("response.output_text.done", map[string]any{
		"type":            "response.output_text.done",
		"sequence_number": e.nextSeq(),
		"output_index":    outputIdx,
		"content_index":   0,
		"text":            text,
	}); err != nil {
		return err
	}

	// Emit content_part.done
	part := map[string]any{
		"type":        ContentTypeOutputText,
		"text":        text,
		"annotations": buildFlatAnnotations(annotations),
	}
	if e.reasoning != "" {
		part["search_reasoning"] = e.reasoning
	}
	if err := e.emit("response.content_part.done", map[string]any{
		"type":            "response.content_part.done",
		"sequence_number": e.nextSeq(),
		"output_index":    outputIdx,
		"content_index":   0,
		"part":            part,
	}); err != nil {
		return err
	}

	// Emit output_item.done
	return e.emit("response.output_item.done", map[string]any{
		"type":            "response.output_item.done",
		"sequence_number": e.nextSeq(),
		"output_index":    outputIdx,
		"item": map[string]any{
			"type":   ItemTypeMessage,
			"id":     e.messageItemID,
			"role":   RoleAssistant,
			"status": StatusCompleted,
		},
	})
}

// EmitError emits an error event
func (e *ResponsesEmitter) EmitError(err error) error {
	// Pass through upstream API errors with their original structured fields
	var apiErr *openai.Error
	if errors.As(err, &apiErr) && apiErr.RawJSON() != "" {
		var rawError map[string]any
		if json.Unmarshal([]byte(apiErr.RawJSON()), &rawError) == nil {
			return e.emit("error", map[string]any{
				"type":            "error",
				"sequence_number": e.nextSeq(),
				"error":           rawError,
			})
		}
	}

	return e.emit("error", map[string]any{
		"type":            "error",
		"sequence_number": e.nextSeq(),
		"error": map[string]any{
			"type":    pipeline.ErrTypeServer,
			"code":    pipeline.ErrTypeServer,
			"message": sanitizeErrorMessage(err),
			"param":   nil,
		},
	})
}

// EmitDone emits the response.completed event
func (e *ResponsesEmitter) EmitDone() error {
	return e.emit("response.completed", map[string]any{
		"type":            "response.completed",
		"sequence_number": e.nextSeq(),
		"response": map[string]any{
			"id":         e.responseID,
			"object":     ObjectResponse,
			"created_at": e.createdAt,
			"status":     StatusCompleted,
			"model":      e.model,
		},
	})
}

// Verify ResponsesEmitter implements EventEmitter
var _ pipeline.EventEmitter = (*ResponsesEmitter)(nil)
