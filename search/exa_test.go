package search

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func newTestProvider(serverURL string) *ExaProvider {
	return &ExaProvider{
		apiKey:     "test-api-key",
		httpClient: &http.Client{Timeout: 5 * time.Second},
		baseURL:    serverURL,
	}
}

func TestExaProvider_Search_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/search" {
			t.Errorf("expected /search, got %s", r.URL.Path)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected Content-Type application/json")
		}
		if r.Header.Get("x-api-key") != "test-api-key" {
			t.Errorf("expected x-api-key header")
		}

		var reqBody exaRequest
		json.NewDecoder(r.Body).Decode(&reqBody)
		if reqBody.Query != "test query" {
			t.Errorf("expected query 'test query', got '%s'", reqBody.Query)
		}
		if reqBody.NumResults != 3 {
			t.Errorf("expected numResults 3, got %d", reqBody.NumResults)
		}

		resp := exaResponse{
			Results: []struct {
				Title         string `json:"title"`
				URL           string `json:"url"`
				Text          string `json:"text"`
				Favicon       string `json:"favicon"`
				PublishedDate string `json:"publishedDate"`
			}{
				{
					Title:         "Result 1",
					URL:           "https://example.com/1",
					Text:          "Content 1",
					Favicon:       "https://example.com/favicon.ico",
					PublishedDate: "2024-01-01",
				},
				{
					Title:         "Result 2",
					URL:           "https://example.com/2",
					Text:          "Content 2",
					Favicon:       "",
					PublishedDate: "",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := newTestProvider(server.URL)
	results, err := provider.Search(context.Background(), "test query", 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}

	if results[0].Title != "Result 1" {
		t.Errorf("expected title 'Result 1', got '%s'", results[0].Title)
	}
	if results[0].URL != "https://example.com/1" {
		t.Errorf("expected URL 'https://example.com/1', got '%s'", results[0].URL)
	}
	if results[0].Content != "Content 1" {
		t.Errorf("expected content 'Content 1', got '%s'", results[0].Content)
	}
	if results[0].Favicon != "https://example.com/favicon.ico" {
		t.Errorf("expected favicon, got '%s'", results[0].Favicon)
	}
	if results[0].PublishedDate != "2024-01-01" {
		t.Errorf("expected date '2024-01-01', got '%s'", results[0].PublishedDate)
	}
}

func TestExaProvider_Search_EmptyResults(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := exaResponse{
			Results: []struct {
				Title         string `json:"title"`
				URL           string `json:"url"`
				Text          string `json:"text"`
				Favicon       string `json:"favicon"`
				PublishedDate string `json:"publishedDate"`
			}{},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := newTestProvider(server.URL)
	results, err := provider.Search(context.Background(), "no results query", 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}

func TestExaProvider_Search_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("internal server error"))
	}))
	defer server.Close()

	provider := newTestProvider(server.URL)
	_, err := provider.Search(context.Background(), "test query", 3)
	if err == nil {
		t.Fatal("expected error for 500 response")
	}

	if !strings.Contains(err.Error(), "500") {
		t.Errorf("expected error to contain status code, got: %v", err)
	}
}

func TestExaProvider_Search_BadRequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error": "invalid request"}`))
	}))
	defer server.Close()

	provider := newTestProvider(server.URL)
	_, err := provider.Search(context.Background(), "test query", 3)
	if err == nil {
		t.Fatal("expected error for 400 response")
	}

	if !strings.Contains(err.Error(), "400") {
		t.Errorf("expected error to contain status code, got: %v", err)
	}
}

func TestExaProvider_Search_MalformedJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte("not valid json"))
	}))
	defer server.Close()

	provider := newTestProvider(server.URL)
	_, err := provider.Search(context.Background(), "test query", 3)
	if err == nil {
		t.Fatal("expected error for malformed JSON")
	}

	if !strings.Contains(err.Error(), "failed to parse response") {
		t.Errorf("expected parse error, got: %v", err)
	}
}

func TestExaProvider_Search_ExaErrorField(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := exaResponse{
			Error: "rate limit exceeded",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := newTestProvider(server.URL)
	_, err := provider.Search(context.Background(), "test query", 3)
	if err == nil {
		t.Fatal("expected error when API returns error field")
	}

	if !strings.Contains(err.Error(), "rate limit exceeded") {
		t.Errorf("expected error message to contain 'rate limit exceeded', got: %v", err)
	}
}

func TestExaProvider_Search_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(exaResponse{})
	}))
	defer server.Close()

	provider := newTestProvider(server.URL)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := provider.Search(ctx, "test query", 3)
	if err == nil {
		t.Fatal("expected error when context is cancelled")
	}
}

func TestExaProvider_Search_Timeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(exaResponse{})
	}))
	defer server.Close()

	provider := &ExaProvider{
		apiKey:     "test-api-key",
		httpClient: &http.Client{Timeout: 50 * time.Millisecond},
		baseURL:    server.URL,
	}

	_, err := provider.Search(context.Background(), "test query", 3)
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestExaProvider_Search_RequestBody(t *testing.T) {
	var receivedReq exaRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedReq)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(exaResponse{})
	}))
	defer server.Close()

	provider := newTestProvider(server.URL)
	provider.Search(context.Background(), "my search query", 10)

	if receivedReq.Query != "my search query" {
		t.Errorf("expected query 'my search query', got '%s'", receivedReq.Query)
	}
	if receivedReq.Type != "fast" {
		t.Errorf("expected type 'fast', got '%s'", receivedReq.Type)
	}
	if receivedReq.NumResults != 10 {
		t.Errorf("expected numResults 10, got %d", receivedReq.NumResults)
	}
	if receivedReq.Contents == nil || receivedReq.Contents.Text == nil {
		t.Error("expected contents.text to be set")
	}
}
