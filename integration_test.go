//go:build integration

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/api"
	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/engine"
	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
)

// responsesClientWrapper wraps the responses service to implement engine.ResponsesClient
type responsesClientWrapper struct {
	inner *responses.ResponseService
}

func (c responsesClientWrapper) New(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) (*responses.Response, error) {
	return c.inner.New(ctx, body, opts...)
}

func (c responsesClientWrapper) NewStreaming(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) engine.ResponseStream {
	return c.inner.NewStreaming(ctx, body, opts...)
}

func setupIntegrationServer(t *testing.T) *api.Server {
	t.Helper()

	exaKey := os.Getenv("EXA_API_KEY")
	if exaKey == "" {
		t.Skip("EXA_API_KEY not set, skipping integration test")
	}

	tinfoilKey := os.Getenv("TINFOIL_API_KEY")
	if tinfoilKey == "" {
		t.Skip("TINFOIL_API_KEY not set, skipping integration test")
	}

	client, err := tinfoil.NewClient(option.WithAPIKey(tinfoilKey))
	if err != nil {
		t.Fatalf("Failed to create Tinfoil client: %v", err)
	}

	cfg := &config.Config{
		// AgentModel removed in refactor:           "gpt-oss-120b",
		ExaAPIKey:            exaKey,
		SafeguardModel:       "gpt-oss-safeguard-120b",
		EnableInjectionCheck: true,
	}

	searcher, err := search.NewProvider(search.Config{ExaAPIKey: cfg.ExaAPIKey})
	if err != nil {
		t.Fatalf("Failed to create search provider: %v", err)
	}

	var fetcher engine.URLFetcher
	cloudflareAccountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	cloudflareAPIToken := os.Getenv("CLOUDFLARE_API_TOKEN")
	if cloudflareAccountID != "" && cloudflareAPIToken != "" {
		fetcher = fetch.NewFetcher(cloudflareAccountID, cloudflareAPIToken)
	}

	safeguardClient := safeguard.NewClient(client, cfg.SafeguardModel)

	service := engine.NewService(
		responsesClient{inner: &client.Responses},
		searcher,
		fetcher,
		safeguardClient,
		engine.WithChatCompletionsClient(chatCompletionsClient{inner: &client.Chat.Completions}),
		engine.WithModelCatalog(engine.NewHTTPModelCatalog(nil)),
		engine.WithToolSummaryModel(cfg.ToolSummaryModel),
	)

	return &api.Server{
		Runner:                       service,
		DefaultPIICheckEnabled:       cfg.EnablePIICheck,
		DefaultInjectionCheckEnabled: cfg.EnableInjectionCheck,
	}
}

func TestIntegration_HealthEndpoint(t *testing.T) {
	srv := &api.Server{}
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()

	srv.HandleHealth(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	body, _ := io.ReadAll(w.Body)
	var resp map[string]string
	json.Unmarshal(body, &resp)

	if resp["status"] != "ok" {
		t.Errorf("expected status ok, got %s", resp["status"])
	}
}

func TestIntegration_RootEndpoint(t *testing.T) {
	srv := &api.Server{}
	req := httptest.NewRequest("GET", "/", nil)
	w := httptest.NewRecorder()

	srv.HandleRoot(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	body, _ := io.ReadAll(w.Body)
	var resp map[string]string
	json.Unmarshal(body, &resp)

	if resp["service"] != "confidential-websearch" {
		t.Errorf("expected service 'confidential-websearch', got %s", resp["service"])
	}
}

func TestIntegration_ChatCompletions_SimpleQuery(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	srv := setupIntegrationServer(t)

	requestBody := map[string]any{
		"model": "gpt-oss-120b",
		"messages": []map[string]string{
			{"role": "user", "content": "What is 2+2?"},
		},
		"stream": false,
	}
	bodyBytes, _ := json.Marshal(requestBody)

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		srv.HandleChatCompletions(w, req)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Minute):
		t.Fatal("request timed out")
	}

	if w.Code != http.StatusOK {
		body, _ := io.ReadAll(w.Body)
		t.Fatalf("expected 200, got %d: %s", w.Code, string(body))
	}

	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)

	if _, ok := resp["choices"]; !ok {
		t.Error("response should have choices")
	}
}

func TestIntegration_ChatCompletions_SearchQuery(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	srv := setupIntegrationServer(t)

	requestBody := map[string]any{
		"model": "gpt-oss-120b",
		"messages": []map[string]string{
			{"role": "user", "content": "What is the current weather in San Francisco? Search the web for the latest information."},
		},
		"stream":             false,
		"web_search_options": map[string]any{},
	}
	bodyBytes, _ := json.Marshal(requestBody)

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		srv.HandleChatCompletions(w, req)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Minute):
		t.Fatal("request timed out")
	}

	if w.Code != http.StatusOK {
		body, _ := io.ReadAll(w.Body)
		t.Fatalf("expected 200, got %d: %s", w.Code, string(body))
	}

	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)

	choices, ok := resp["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatal("response should have choices")
	}

	choice := choices[0].(map[string]any)
	message := choice["message"].(map[string]any)

	if message["content"] == "" {
		t.Error("response content should not be empty")
	}
}

func TestIntegration_Responses_SimpleQuery(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	srv := setupIntegrationServer(t)

	requestBody := map[string]any{
		"model": "gpt-oss-120b",
		"input": "Tell me a joke",
	}
	bodyBytes, _ := json.Marshal(requestBody)

	req := httptest.NewRequest("POST", "/v1/responses", bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		srv.HandleResponses(w, req)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Minute):
		t.Fatal("request timed out")
	}

	if w.Code != http.StatusOK {
		body, _ := io.ReadAll(w.Body)
		t.Fatalf("expected 200, got %d: %s", w.Code, string(body))
	}

	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)

	if resp["object"] != "response" {
		t.Errorf("expected object 'response', got '%v'", resp["object"])
	}

	output, ok := resp["output"].([]any)
	if !ok || len(output) == 0 {
		t.Fatal("response should have output")
	}
}

func TestIntegration_ChatCompletions_InvalidRequest(t *testing.T) {
	srv := setupIntegrationServer(t)

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader([]byte("invalid json")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	srv.HandleChatCompletions(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestIntegration_Responses_InvalidRequest(t *testing.T) {
	srv := setupIntegrationServer(t)

	req := httptest.NewRequest("POST", "/v1/responses", bytes.NewReader([]byte("invalid json")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	srv.HandleResponses(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}
