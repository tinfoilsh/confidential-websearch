package fetch

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestExtractURLs(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "no URLs",
			input:    "hello world",
			expected: nil,
		},
		{
			name:     "single URL",
			input:    "check out https://example.com/page",
			expected: []string{"https://example.com/page"},
		},
		{
			name:     "multiple URLs",
			input:    "see https://a.com and https://b.com",
			expected: []string{"https://a.com", "https://b.com"},
		},
		{
			name:     "URL with trailing punctuation",
			input:    "visit https://example.com/page.",
			expected: []string{"https://example.com/page"},
		},
		{
			name:     "URL with trailing comma",
			input:    "see https://a.com, https://b.com, and more",
			expected: []string{"https://a.com", "https://b.com"},
		},
		{
			name:     "duplicate URLs",
			input:    "https://example.com and https://example.com again",
			expected: []string{"https://example.com"},
		},
		{
			name:     "http URL",
			input:    "old site http://example.com/page",
			expected: []string{"http://example.com/page"},
		},
		{
			name:     "URL with query params",
			input:    "see https://example.com/search?q=test&page=1",
			expected: []string{"https://example.com/search?q=test&page=1"},
		},
		{
			name:     "URL with fragment",
			input:    "see https://example.com/page#section",
			expected: []string{"https://example.com/page#section"},
		},
		{
			name:     "bare domain",
			input:    "what does example.com say about this?",
			expected: []string{"https://example.com"},
		},
		{
			name:     "bare domain with path",
			input:    "check out example.com/about for more info",
			expected: []string{"https://example.com/about"},
		},
		{
			name:     "bare subdomain",
			input:    "see docs.example.com for details",
			expected: []string{"https://docs.example.com"},
		},
		{
			name:     "bare domain at start of text",
			input:    "example.com has great content",
			expected: []string{"https://example.com"},
		},
		{
			name:     "bare domain not duplicated with full URL",
			input:    "visit https://example.com or example.com",
			expected: []string{"https://example.com"},
		},
		{
			name:     "not a domain - single word",
			input:    "hello world",
			expected: nil,
		},
		{
			name:     "not a domain - no TLD",
			input:    "check localhost for details",
			expected: nil,
		},
		{
			name:     "not a URL - file extension txt",
			input:    "open myfile.txt please",
			expected: nil,
		},
		{
			name:     "not a URL - file extension json",
			input:    "edit config.json now",
			expected: nil,
		},
		{
			name:     "not a URL - file extension yaml",
			input:    "check values.yaml",
			expected: nil,
		},
		{
			name:     "valid bare domain with real TLD",
			input:    "go to google.com",
			expected: []string{"https://google.com"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractURLs(tt.input)
			if len(result) != len(tt.expected) {
				t.Fatalf("expected %d URLs, got %d: %v", len(tt.expected), len(result), result)
			}
			for i, u := range result {
				if u != tt.expected[i] {
					t.Errorf("URL[%d]: expected %q, got %q", i, tt.expected[i], u)
				}
			}
		})
	}
}

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
	urls := []string{"https://a.com", "https://b.com", "https://c.com"}
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
