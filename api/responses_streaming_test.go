package api

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

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
