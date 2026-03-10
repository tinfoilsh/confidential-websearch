package fetch

import (
	"context"
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

func TestFetchURLs_PlainText(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte("This is plain text content"))
	}))
	defer server.Close()

	fetcher := NewFetcher()
	pages := fetcher.FetchURLs(context.Background(), []string{server.URL})

	if len(pages) != 1 {
		t.Fatalf("expected 1 page, got %d", len(pages))
	}
	if pages[0].Content != "This is plain text content" {
		t.Errorf("unexpected content: %q", pages[0].Content)
	}
	if pages[0].URL != server.URL {
		t.Errorf("expected URL %q, got %q", server.URL, pages[0].URL)
	}
}

func TestFetchURLs_HTML(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(`<html><body><h1>Title</h1><p>Hello world</p><script>var x = 1;</script></body></html>`))
	}))
	defer server.Close()

	fetcher := NewFetcher()
	pages := fetcher.FetchURLs(context.Background(), []string{server.URL})

	if len(pages) != 1 {
		t.Fatalf("expected 1 page, got %d", len(pages))
	}
	// Should contain the text content, converted to markdown
	if pages[0].Content == "" {
		t.Error("expected non-empty content")
	}
}

func TestFetchURLs_NonOKStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	fetcher := NewFetcher()
	pages := fetcher.FetchURLs(context.Background(), []string{server.URL})

	if len(pages) != 0 {
		t.Fatalf("expected 0 pages for 404, got %d", len(pages))
	}
}

func TestFetchURLs_UnsupportedContentType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/pdf")
		w.Write([]byte("binary data"))
	}))
	defer server.Close()

	fetcher := NewFetcher()
	pages := fetcher.FetchURLs(context.Background(), []string{server.URL})

	if len(pages) != 0 {
		t.Fatalf("expected 0 pages for unsupported content type, got %d", len(pages))
	}
}

func TestFetchURLs_Markdown(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/markdown")
		w.Write([]byte("# Hello\n\nThis is markdown"))
	}))
	defer server.Close()

	fetcher := NewFetcher()
	pages := fetcher.FetchURLs(context.Background(), []string{server.URL})

	if len(pages) != 1 {
		t.Fatalf("expected 1 page, got %d", len(pages))
	}
	if pages[0].Content != "# Hello\n\nThis is markdown" {
		t.Errorf("unexpected content: %q", pages[0].Content)
	}
}

func TestFetchURLs_ParallelMultipleURLs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte("page " + r.URL.Path))
	}))
	defer server.Close()

	fetcher := NewFetcher()
	urls := []string{server.URL + "/a", server.URL + "/b", server.URL + "/c"}
	pages := fetcher.FetchURLs(context.Background(), urls)

	if len(pages) != 3 {
		t.Fatalf("expected 3 pages, got %d", len(pages))
	}
}

func TestFetchURLs_ContextCanceled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte("content"))
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	fetcher := NewFetcher()
	pages := fetcher.FetchURLs(ctx, []string{server.URL})

	if len(pages) != 0 {
		t.Fatalf("expected 0 pages for canceled context, got %d", len(pages))
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
