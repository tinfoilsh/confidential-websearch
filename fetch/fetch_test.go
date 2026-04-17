package fetch

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
)

// newTestFetcher creates a Fetcher that points at a test server instead of Cloudflare
func newTestFetcher(apiURL string) *Fetcher {
	return &Fetcher{
		client:   http.DefaultClient,
		apiURL:   apiURL,
		apiToken: "test-token",
	}
}

// writeCFResponse writes a Cloudflare-style JSON response
func writeCFResponse(w http.ResponseWriter, success bool, result string, errors []cfError) {
	resp := cloudflareResponse{
		Success: success,
		Result:  result,
		Errors:  errors,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func TestFetchURLs_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.Header.Get("Authorization") != "Bearer test-token" {
			t.Errorf("unexpected auth header: %s", r.Header.Get("Authorization"))
		}

		var req cloudflareRequest
		json.NewDecoder(r.Body).Decode(&req)

		writeCFResponse(w, true, "# Example\n\nThis is markdown from "+req.URL, nil)
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
	if pages[0].Content != "# Example\n\nThis is markdown from https://example.com" {
		t.Errorf("unexpected content: %q", pages[0].Content)
	}
}

func TestFetchURLs_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeCFResponse(w, false, "", []cfError{{Code: 1000, Message: "invalid URL"}})
	}))
	defer server.Close()

	fetcher := newTestFetcher(server.URL)
	pages := fetcher.FetchURLs(context.Background(), []string{"https://bad.example"})

	if len(pages) != 0 {
		t.Fatalf("expected 0 pages for API error, got %d", len(pages))
	}
}

func TestFetchURLs_NonOKStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("internal error"))
	}))
	defer server.Close()

	fetcher := newTestFetcher(server.URL)
	pages := fetcher.FetchURLs(context.Background(), []string{"https://example.com"})

	if len(pages) != 0 {
		t.Fatalf("expected 0 pages for 500, got %d", len(pages))
	}
}

func TestFetchURLs_ParallelMultipleURLs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req cloudflareRequest
		json.NewDecoder(r.Body).Decode(&req)
		writeCFResponse(w, true, "page for "+req.URL, nil)
	}))
	defer server.Close()

	fetcher := newTestFetcher(server.URL)
	urls := []string{
		"https://example.com/a",
		"https://example.com/b",
		"https://example.com/c",
	}
	pages := fetcher.FetchURLs(context.Background(), urls)

	if len(pages) != 3 {
		t.Fatalf("expected 3 pages, got %d", len(pages))
	}
}

func TestFetchURLs_ContextCanceled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeCFResponse(w, true, "content", nil)
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

func TestFetchURLResults_ContextCanceledPreservesInputOrderMetadata(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeCFResponse(w, true, "content", nil)
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	fetcher := newTestFetcher(server.URL)
	urls := []string{"https://example.com/a", "https://example.com/b"}
	results := fetcher.FetchURLResults(ctx, urls)

	if len(results) != len(urls) {
		t.Fatalf("expected %d results, got %d", len(urls), len(results))
	}
	for i, result := range results {
		if result.URL != urls[i] {
			t.Fatalf("expected result %d url %q, got %q", i, urls[i], result.URL)
		}
		if result.Status != FetchStatusFailed {
			t.Fatalf("expected result %d status %q, got %q", i, FetchStatusFailed, result.Status)
		}
	}
}

func TestFetchURLs_EmptyResult(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeCFResponse(w, true, "   ", nil)
	}))
	defer server.Close()

	fetcher := newTestFetcher(server.URL)
	pages := fetcher.FetchURLs(context.Background(), []string{"https://example.com"})

	if len(pages) != 0 {
		t.Fatalf("expected 0 pages for empty content, got %d", len(pages))
	}
}

func TestValidateTargetURL_RejectsUnsafeTargets(t *testing.T) {
	credentialURL := (&url.URL{
		Scheme: "https",
		Host:   "example.com",
		User:   url.UserPassword("username", "placeholder"),
	}).String()

	tests := []string{
		"http://127.0.0.1",
		"http://[::1]",
		"http://localhost",
		"http://service.internal",
		"file:///etc/passwd",
		credentialURL,
		"https://192.168.1.10",
	}

	for _, rawURL := range tests {
		t.Run(rawURL, func(t *testing.T) {
			if err := validateTargetURL(context.Background(), rawURL); err == nil {
				t.Fatalf("expected %q to be rejected", rawURL)
			}
		})
	}
}

func TestValidateTargetURL_AllowsPublicHTTPSTarget(t *testing.T) {
	if err := validateTargetURL(context.Background(), "https://93.184.216.34"); err != nil {
		t.Fatalf("expected public IP target to be allowed, got %v", err)
	}
}

func TestFetchURLs_SkipsUnsafeTargetsBeforeCallingAPI(t *testing.T) {
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		writeCFResponse(w, true, "content", nil)
	}))
	defer server.Close()

	fetcher := newTestFetcher(server.URL)
	pages := fetcher.FetchURLs(context.Background(), []string{
		"http://127.0.0.1",
		"http://localhost",
		"https://192.168.1.10",
	})

	if len(pages) != 0 {
		t.Fatalf("expected 0 pages for unsafe targets, got %d", len(pages))
	}
	if calls.Load() != 0 {
		t.Fatalf("expected Cloudflare API not to be called, got %d calls", calls.Load())
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
