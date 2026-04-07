package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tinfoilsh/confidential-websearch/engine"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

type captureRunner struct {
	t     *testing.T
	check func(*pipeline.Request)
}

func (r *captureRunner) Run(ctx context.Context, req *pipeline.Request) (*engine.Result, error) {
	r.t.Helper()
	if r.check != nil {
		r.check(req)
	}

	return &engine.Result{
		ID:      "resp_123",
		Model:   req.Model,
		Object:  "chat.completion",
		Created: 123,
		Content: "ok",
	}, nil
}

func (r *captureRunner) Stream(ctx context.Context, req *pipeline.Request, emitter pipeline.EventEmitter) (*engine.Result, error) {
	return r.Run(ctx, req)
}

func TestHandleChatCompletions_MapsFeatureFlagsToPipelineRequest(t *testing.T) {
	srv := &Server{
		Runner: &captureRunner{
			t: t,
			check: func(req *pipeline.Request) {
				if !req.WebSearchEnabled {
					t.Fatal("expected web search to be enabled")
				}
				if !req.PIICheckEnabled {
					t.Fatal("expected pii check to be enabled")
				}
				if !req.InjectionCheckEnabled {
					t.Fatal("expected injection check to be enabled")
				}
				if req.Format != pipeline.FormatChatCompletion {
					t.Fatalf("expected chat completion format, got %v", req.Format)
				}
			},
		},
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
		"model":"gpt-oss-120b",
		"messages":[{"role":"user","content":"hello"}],
		"web_search_options":{},
		"pii_check_options":{},
		"injection_check_options":{}
	}`))
	rec := httptest.NewRecorder()

	srv.HandleChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
}

func TestHandleResponses_MapsFeatureFlagsToPipelineRequest(t *testing.T) {
	srv := &Server{
		Runner: &captureRunner{
			t: t,
			check: func(req *pipeline.Request) {
				if !req.WebSearchEnabled {
					t.Fatal("expected web search to be enabled")
				}
				if !req.PIICheckEnabled {
					t.Fatal("expected pii check to be enabled")
				}
				if !req.InjectionCheckEnabled {
					t.Fatal("expected injection check to be enabled")
				}
				if req.Format != pipeline.FormatResponses {
					t.Fatalf("expected responses format, got %v", req.Format)
				}
				if req.Input != "hello" {
					t.Fatalf("expected input to be preserved, got %q", req.Input)
				}
			},
		},
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model":"gpt-oss-120b",
		"input":"hello",
		"tools":[{"type":"web_search"}],
		"pii_check_options":{},
		"injection_check_options":{}
	}`))
	rec := httptest.NewRecorder()

	srv.HandleResponses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var body struct {
		Output []ResponsesOutput `json:"output"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}
	if len(body.Output) != 1 || body.Output[0].Type != ItemTypeMessage {
		t.Fatalf("expected final message output, got %+v", body.Output)
	}
}
