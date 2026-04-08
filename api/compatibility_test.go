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
				if req.SearchContextSize != pipeline.SearchContextSizeLow {
					t.Fatalf("expected search context size low, got %q", req.SearchContextSize)
				}
				if req.UserLocation == nil || req.UserLocation.Country != "US" || req.UserLocation.City != "San Francisco" || req.UserLocation.Region != "CA" {
					t.Fatalf("expected user location to be mapped, got %+v", req.UserLocation)
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
		"web_search_options":{
			"search_context_size":"low",
			"user_location":{
				"type":"approximate",
				"approximate":{
					"country":"US",
					"city":"San Francisco",
					"region":"CA"
				}
			}
		},
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
				if req.SearchContextSize != pipeline.SearchContextSizeHigh {
					t.Fatalf("expected search context size high, got %q", req.SearchContextSize)
				}
				if req.UserLocation == nil || req.UserLocation.Country != "FR" {
					t.Fatalf("expected user location country FR, got %+v", req.UserLocation)
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
				if string(req.Input) != `"hello"` {
					t.Fatalf("expected input to be preserved, got %s", string(req.Input))
				}
			},
		},
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model":"gpt-oss-120b",
		"input":"hello",
		"tools":[{
			"type":"web_search",
			"search_context_size":"high",
			"user_location":{
				"type":"approximate",
				"approximate":{
					"country":"FR"
				}
			}
		}],
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
