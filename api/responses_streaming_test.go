package api

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	openai "github.com/openai/openai-go/v3"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

func TestResponsesEmitter_ReusesOutputIndexPerToolItem(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, err := NewResponsesEmitter(w, "resp_test", "gpt-oss-120b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := emitter.EmitSearchCall("a", StatusInProgress, "query a", "", 0, "gpt-oss-120b"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitSearchCall("b", StatusInProgress, "query b", "", 0, "gpt-oss-120b"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitSearchCall("a", StatusCompleted, "query a", "", 0, "gpt-oss-120b"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitSearchCall("b", StatusCompleted, "query b", "", 0, "gpt-oss-120b"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	events := decodeSSEDataLines(t, w.Body.String())
	aIndexes := collectOutputIndexes(events, IDPrefixWebSearch+"a")
	bIndexes := collectOutputIndexes(events, IDPrefixWebSearch+"b")

	if len(aIndexes) != 1 {
		t.Fatalf("expected one output index for item a, got %v", aIndexes)
	}
	if len(bIndexes) != 1 {
		t.Fatalf("expected one output index for item b, got %v", bIndexes)
	}
	if aIndexes[0] == bIndexes[0] {
		t.Fatalf("expected distinct output indexes, got %d for both items", aIndexes[0])
	}
}

func TestResponsesEmitter_UsesFlatAnnotationsInContentPartDone(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, err := NewResponsesEmitter(w, "resp_test", "gpt-oss-120b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := emitter.EmitMessageStart("msg_test"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitMessageEnd("answer【1】", []pipeline.Annotation{
		{
			Type: pipeline.AnnotationTypeURLCitation,
			URLCitation: pipeline.URLCitation{
				URL:        "https://example.com",
				Title:      "Example",
				StartIndex: 6,
				EndIndex:   10,
			},
		},
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	events := decodeSSEDataLines(t, w.Body.String())
	for _, event := range events {
		if event["type"] != "response.content_part.done" {
			continue
		}
		part, ok := event["part"].(map[string]any)
		if !ok {
			t.Fatalf("expected part payload, got %+v", event["part"])
		}
		annotations, ok := part["annotations"].([]any)
		if !ok || len(annotations) != 1 {
			t.Fatalf("expected one flat annotation, got %+v", part["annotations"])
		}
		annotation, ok := annotations[0].(map[string]any)
		if !ok {
			t.Fatalf("expected annotation object, got %+v", annotations[0])
		}
		if _, nested := annotation["url_citation"]; nested {
			t.Fatalf("expected flat annotation shape, got %+v", annotation)
		}
		if annotation["url"] != "https://example.com" {
			t.Fatalf("expected annotation url, got %+v", annotation)
		}
		return
	}

	t.Fatal("expected response.content_part.done event")
}

func TestResponsesEmitter_DoesNotReAddBlockedToolItem(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, err := NewResponsesEmitter(w, "resp_test", "gpt-oss-120b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := emitter.EmitSearchCall("a", StatusInProgress, "query a", "", 0, "gpt-oss-120b"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitSearchCall("a", StatusBlocked, "query a", "blocked", 0, "gpt-oss-120b"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	events := decodeSSEDataLines(t, w.Body.String())
	addedCount := 0
	for _, event := range events {
		if event["type"] != "response.output_item.added" {
			continue
		}
		item, ok := event["item"].(map[string]any)
		if !ok || item["id"] != IDPrefixWebSearch+"a" {
			continue
		}
		addedCount++
	}
	if addedCount != 1 {
		t.Fatalf("expected one output_item.added for blocked item, got %d", addedCount)
	}
}

func TestResponsesEmitter_EmitDoneIncludesUsage(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, err := NewResponsesEmitter(w, "resp_test", "gpt-oss-120b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := emitter.EmitDone("resp_test", 123, "gpt-oss-120b", openai.CompletionUsage{
		PromptTokens:     7,
		CompletionTokens: 4,
		TotalTokens:      11,
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	events := decodeSSEDataLines(t, w.Body.String())
	for _, event := range events {
		if event["type"] != "response.completed" {
			continue
		}
		response, ok := event["response"].(map[string]any)
		if !ok {
			t.Fatalf("expected response payload, got %+v", event["response"])
		}
		usage, ok := response["usage"].(map[string]any)
		if !ok {
			t.Fatalf("expected usage payload, got %+v", response["usage"])
		}
		if usage["input_tokens"] != float64(7) || usage["output_tokens"] != float64(4) || usage["total_tokens"] != float64(11) {
			t.Fatalf("unexpected usage payload: %+v", usage)
		}
		return
	}

	t.Fatal("expected response.completed event")
}

func TestResponsesEmitter_MessageEventsMatchSDKShape(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, err := NewResponsesEmitter(w, "resp_test", "gpt-oss-120b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := emitter.EmitMessageStart("msg_test"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitChunk([]byte(`{"choices":[{"delta":{"content":"Hello"}}]}`)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitMessageEnd("Hello", []pipeline.Annotation{
		{
			Type: pipeline.AnnotationTypeURLCitation,
			URLCitation: pipeline.URLCitation{
				URL:        "https://example.com",
				Title:      "Example",
				StartIndex: 0,
				EndIndex:   5,
			},
		},
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitDone("resp_test", 123, "gpt-oss-120b", openai.CompletionUsage{
		PromptTokens:     2,
		CompletionTokens: 1,
		TotalTokens:      3,
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	events := decodeSSEDataLines(t, w.Body.String())
	for _, event := range events {
		switch event["type"] {
		case "response.output_item.added":
			item, ok := event["item"].(map[string]any)
			if !ok || item["id"] != "msg_test" {
				continue
			}
			content, ok := item["content"].([]any)
			if !ok || len(content) != 0 {
				t.Fatalf("expected empty content array on message add, got %+v", item["content"])
			}
		case "response.content_part.added", "response.content_part.done", "response.output_text.delta", "response.output_text.done", "response.output_text.annotation.added":
			if event["item_id"] != "msg_test" {
				t.Fatalf("expected item_id on %s, got %+v", event["type"], event)
			}
			if event["type"] == "response.output_text.delta" || event["type"] == "response.output_text.done" {
				logprobs, ok := event["logprobs"].([]any)
				if !ok || len(logprobs) != 0 {
					t.Fatalf("expected empty logprobs on %s, got %+v", event["type"], event["logprobs"])
				}
			}
		case "response.completed":
			response, ok := event["response"].(map[string]any)
			if !ok {
				t.Fatalf("expected response payload, got %+v", event["response"])
			}
			output, ok := response["output"].([]any)
			if !ok || len(output) != 1 {
				t.Fatalf("expected one output item in completed response, got %+v", response["output"])
			}
			message, ok := output[0].(map[string]any)
			if !ok {
				t.Fatalf("expected message output item, got %+v", output[0])
			}
			content, ok := message["content"].([]any)
			if !ok || len(content) != 1 {
				t.Fatalf("expected one content part in completed response, got %+v", message["content"])
			}
		}
	}
}

func TestResponsesEmitter_FlushesPartialMessageBeforeToolCalls(t *testing.T) {
	w := httptest.NewRecorder()
	emitter, err := NewResponsesEmitter(w, "resp_test", "gpt-oss-120b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := emitter.EmitMessageStart("msg_pre"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitChunk([]byte(`{"choices":[{"delta":{"content":"Let me check that."}}]}`)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitSearchCall("search_1", StatusInProgress, "golang", "", 0, "gpt-oss-120b"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitSearchCall("search_1", StatusCompleted, "golang", "", 0, "gpt-oss-120b"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitMessageStart("msg_final"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitChunk([]byte(`{"choices":[{"delta":{"content":"Here is the answer."}}]}`)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitMessageEnd("Here is the answer.", nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := emitter.EmitDone("resp_test", 123, "gpt-oss-120b", openai.CompletionUsage{
		PromptTokens:     2,
		CompletionTokens: 2,
		TotalTokens:      4,
	}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	events := decodeSSEDataLines(t, w.Body.String())
	for _, event := range events {
		if event["type"] != "response.completed" {
			continue
		}
		response, ok := event["response"].(map[string]any)
		if !ok {
			t.Fatalf("expected response payload, got %+v", event["response"])
		}
		output, ok := response["output"].([]any)
		if !ok || len(output) != 3 {
			t.Fatalf("expected three output items in completed response, got %+v", response["output"])
		}

		firstMessage := output[0].(map[string]any)
		firstContent := firstMessage["content"].([]any)
		firstPart := firstContent[0].(map[string]any)
		if firstPart["text"] != "Let me check that." {
			t.Fatalf("expected flushed partial text, got %+v", firstPart["text"])
		}

		toolItem := output[1].(map[string]any)
		if toolItem["type"] != ItemTypeWebSearchCall || toolItem["status"] != StatusCompleted {
			t.Fatalf("expected completed tool item, got %+v", toolItem)
		}

		finalMessage := output[2].(map[string]any)
		finalContent := finalMessage["content"].([]any)
		finalPart := finalContent[0].(map[string]any)
		if finalPart["text"] != "Here is the answer." {
			t.Fatalf("expected final message text, got %+v", finalPart["text"])
		}
		return
	}

	t.Fatal("expected response.completed event")
}

func decodeSSEDataLines(t *testing.T, body string) []map[string]any {
	t.Helper()

	var events []map[string]any
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
			t.Fatalf("failed to decode event %q: %v", line, err)
		}
		events = append(events, event)
	}
	return events
}

func collectOutputIndexes(events []map[string]any, itemID string) []int {
	seen := make(map[int]struct{})
	var indexes []int

	for _, event := range events {
		currentID := ""
		if rawID, ok := event["item_id"].(string); ok {
			currentID = rawID
		}
		if item, ok := event["item"].(map[string]any); ok && currentID == "" {
			if rawID, ok := item["id"].(string); ok {
				currentID = rawID
			}
		}
		if currentID != itemID {
			continue
		}

		value, ok := event["output_index"].(float64)
		if !ok {
			continue
		}
		index := int(value)
		if _, ok := seen[index]; ok {
			continue
		}
		seen[index] = struct{}{}
		indexes = append(indexes, index)
	}

	return indexes
}
