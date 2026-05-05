package fetch

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func newTestFetcher(apiURL string) *Fetcher {
	return &Fetcher{
		client: http.DefaultClient,
		apiURL: apiURL,
		apiKey: "test-key",
	}
}

func writeExaResponse(w http.ResponseWriter, results []exaContentsResult) {
	resp := exaContentsResponse{Results: results}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func TestFetchURLs_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("unexpected api-key header: %s", r.Header.Get("x-api-key"))
		}

		var req exaContentsRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "decode error", http.StatusBadRequest)
			return
		}
		if !req.Text {
			t.Error("expected text=true in request")
		}
		if req.MaxAgeHours != exaMaxAgeHours {
			t.Errorf("expected maxAgeHours=%d, got %d", exaMaxAgeHours, req.MaxAgeHours)
		}

		results := make([]exaContentsResult, len(req.URLs))
		for i, u := range req.URLs {
			results[i] = exaContentsResult{URL: u, Text: "text from " + u}
		}
		writeExaResponse(w, results)
	}))
	defer server.Close()

	fetcher := newTestFetcher(server.URL)
	pages := fetcher.FetchURLs(context.Background(), []string{"https://example.com"})

	if len(pages) != 1 {
		t.Fatalf("expected 1 page, got %d", len(pages))
	}
	if pages[0].URL != "https://example.com" {
		t.Errorf("expected URL %q, got %q", "https://example.com", pages[0].URL)
	}
	if pages[0].Content != "text from https://example.com" {
		t.Errorf("unexpected content: %q", pages[0].Content)
	}
}

func TestFetchURLResults_PreservesInputOrder(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req exaContentsRequest
		json.NewDecoder(r.Body).Decode(&req)
		writeExaResponse(w, []exaContentsResult{
			{URL: req.URLs[1], Text: "second"},
			{URL: req.URLs[0], Text: "first"},
		})
	}))
	defer server.Close()

	fetcher := newTestFetcher(server.URL)
	urls := []string{"https://example.com/a", "https://example.com/b"}
	results := fetcher.FetchURLResults(context.Background(), urls)

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].URL != urls[0] || results[1].URL != urls[1] {
		t.Fatalf("expected input order preserved, got %+v", results)
	}
	if results[0].Content != "first" || results[1].Content != "second" {
		t.Fatalf("expected matched content, got %+v", results)
	}
}

func TestFetchURLResults_EmptyTextMarksFailed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req exaContentsRequest
		json.NewDecoder(r.Body).Decode(&req)
		writeExaResponse(w, []exaContentsResult{
			{URL: req.URLs[0], Text: ""},
		})
	}))
	defer server.Close()

	fetcher := newTestFetcher(server.URL)
	results := fetcher.FetchURLResults(context.Background(), []string{"https://example.com"})

	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Status != FetchStatusFailed {
		t.Fatalf("expected failed status, got %q", results[0].Status)
	}
	if results[0].Error == "" {
		t.Fatal("expected error message for empty content")
	}
}

func TestFetchURLs_NonOKStatusMarksAllFailed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("internal error"))
	}))
	defer server.Close()

	fetcher := newTestFetcher(server.URL)
	results := fetcher.FetchURLResults(context.Background(), []string{
		"https://example.com/a",
		"https://example.com/b",
	})

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	for i, r := range results {
		if r.Status != FetchStatusFailed {
			t.Fatalf("result %d: expected failed status, got %q", i, r.Status)
		}
		if r.Error == "" {
			t.Fatalf("result %d: expected error", i)
		}
	}
}

func TestFetchURLs_MalformedJSONMarksAllFailed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("not json"))
	}))
	defer server.Close()

	fetcher := newTestFetcher(server.URL)
	pages := fetcher.FetchURLs(context.Background(), []string{"https://example.com"})

	if len(pages) != 0 {
		t.Fatalf("expected 0 pages on malformed response, got %d", len(pages))
	}
}

func TestFetchURLs_ContextCanceled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeExaResponse(w, []exaContentsResult{{URL: "https://example.com", Text: "ok"}})
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	fetcher := newTestFetcher(server.URL)
	pages := fetcher.FetchURLs(ctx, []string{"https://example.com"})

	if len(pages) != 0 {
		t.Fatalf("expected 0 pages for canceled context, got %d", len(pages))
	}
}

func TestFetchURLs_EmptyInput(t *testing.T) {
	fetcher := newTestFetcher("http://unreachable.invalid")
	results := fetcher.FetchURLResults(context.Background(), nil)
	if len(results) != 0 {
		t.Fatalf("expected 0 results for empty input, got %d", len(results))
	}
}

func TestTruncate(t *testing.T) {
	short := "hello"
	if truncate(short) != short {
		t.Error("short string should not be truncated")
	}

	long := make([]byte, maxContentLength+100)
	for i := range long {
		long[i] = 'a'
	}
	result := truncate(string(long))
	if len(result) <= maxContentLength {
		t.Error("truncated result should include truncation notice")
	}
	if result[maxContentLength:] != "\n\n[content truncated]" {
		t.Error("expected truncation notice")
	}
}
