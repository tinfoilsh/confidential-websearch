//go:build integration

package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/openai/openai-go/v3/option"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/agent"
	"github.com/tinfoilsh/confidential-websearch/api"
	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/llm"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
)

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
		AgentModel:           "gpt-oss-120b-free",
		ExaAPIKey:            exaKey,
		SafeguardModel:       "gpt-oss-safeguard-120b",
		EnableInjectionCheck: true,
	}

	searcher, err := search.NewProvider(search.Config{ExaAPIKey: cfg.ExaAPIKey})
	if err != nil {
		t.Fatalf("Failed to create search provider: %v", err)
	}

	baseAgent := agent.New(client, cfg.AgentModel)
	responder := llm.NewTinfoilResponder(&client.Chat.Completions)
	messageBuilder := llm.NewMessageBuilder()

	safeguardClient := safeguard.NewClient(client, cfg.SafeguardModel)

	agentRunner := agent.NewSafeAgent(baseAgent, safeguardClient)

	p := pipeline.NewPipeline([]pipeline.Stage{
		&pipeline.ValidateStage{},
		&pipeline.AgentStage{Agent: agentRunner},
		&pipeline.SearchStage{Searcher: searcher},
		&pipeline.FilterResultsStage{
			Checker: safeguardClient,
			Policy:  safeguard.PromptInjectionPolicy,
			Enabled: cfg.EnableInjectionCheck,
		},
		&pipeline.BuildMessagesStage{Builder: messageBuilder},
		&pipeline.ResponderStage{Responder: responder},
	}, api.RequestTimeout)

	return &api.Server{
		Cfg:      cfg,
		Client:   client,
		Pipeline: p,
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

	requestBody := map[string]interface{}{
		"model": "gpt-oss-120b-free",
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

	var resp map[string]interface{}
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

	requestBody := map[string]interface{}{
		"model": "gpt-oss-120b-free",
		"messages": []map[string]string{
			{"role": "user", "content": "What is the current weather in San Francisco? Search the web for the latest information."},
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

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)

	choices, ok := resp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Fatal("response should have choices")
	}

	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})

	if message["content"] == "" {
		t.Error("response content should not be empty")
	}
}

func TestIntegration_Responses_SimpleQuery(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	srv := setupIntegrationServer(t)

	requestBody := map[string]interface{}{
		"model": "gpt-oss-120b-free",
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

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)

	if resp["object"] != "response" {
		t.Errorf("expected object 'response', got '%v'", resp["object"])
	}

	output, ok := resp["output"].([]interface{})
	if !ok || len(output) == 0 {
		t.Fatal("response should have output")
	}
}

func TestIntegration_ChatCompletions_InvalidRequest(t *testing.T) {
	srv := &api.Server{}

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader([]byte("invalid json")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	srv.HandleChatCompletions(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestIntegration_Responses_InvalidRequest(t *testing.T) {
	srv := &api.Server{}

	req := httptest.NewRequest("POST", "/v1/responses", bytes.NewReader([]byte("invalid json")))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	srv.HandleResponses(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}
