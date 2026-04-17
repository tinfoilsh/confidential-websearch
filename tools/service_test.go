package tools

import (
	"context"
	"errors"
	"testing"

	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
)

type stubSearcher struct {
	results []search.Result
	err     error
	opts    search.Options
}

func (s *stubSearcher) Name() string { return "stub" }

func (s *stubSearcher) Search(_ context.Context, _ string, opts search.Options) ([]search.Result, error) {
	s.opts = opts
	return s.results, s.err
}

type stubFetcher struct {
	results []fetch.URLResult
}

func (f *stubFetcher) FetchURLs(_ context.Context, urls []string) []fetch.FetchedPage {
	pages := make([]fetch.FetchedPage, 0, len(urls))
	for _, u := range urls {
		for _, r := range f.results {
			if r.URL == u && r.Status == fetch.FetchStatusCompleted {
				pages = append(pages, fetch.FetchedPage{URL: r.URL, Content: r.Content})
			}
		}
	}
	return pages
}

func (f *stubFetcher) FetchURLResults(_ context.Context, _ []string) []fetch.URLResult {
	return f.results
}

type stubSafeguard struct {
	blocked map[string]string
}

func (s *stubSafeguard) Check(_ context.Context, _ string, content string) (*safeguard.CheckResult, error) {
	if reason, ok := s.blocked[content]; ok {
		return &safeguard.CheckResult{Violation: true, Rationale: reason}, nil
	}
	return &safeguard.CheckResult{}, nil
}

func TestSearch_RequiresQuery(t *testing.T) {
	service := NewService(&stubSearcher{}, nil, nil)
	_, err := service.Search(context.Background(), "", Options{})
	if err == nil {
		t.Fatal("expected error for empty query")
	}
}

func TestSearch_ReturnsResults(t *testing.T) {
	searcher := &stubSearcher{
		results: []search.Result{{Title: "One", URL: "https://example.com/1"}},
	}
	service := NewService(searcher, nil, nil)

	outcome, err := service.Search(context.Background(), "golang", Options{MaxResults: 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(outcome.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(outcome.Results))
	}
	if searcher.opts.MaxResults != 3 {
		t.Fatalf("expected MaxResults=3 forwarded, got %d", searcher.opts.MaxResults)
	}
}

func TestSearch_DefaultMaxResults(t *testing.T) {
	searcher := &stubSearcher{}
	service := NewService(searcher, nil, nil)

	if _, err := service.Search(context.Background(), "golang", Options{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if searcher.opts.MaxResults != defaultMaxResults {
		t.Fatalf("expected default MaxResults=%d, got %d", defaultMaxResults, searcher.opts.MaxResults)
	}
}

func TestSearch_PIIBlocksQuery(t *testing.T) {
	searcher := &stubSearcher{results: []search.Result{{Title: "hit"}}}
	sg := &stubSafeguard{blocked: map[string]string{"john@example.com": "email detected"}}
	service := NewService(searcher, nil, sg)

	outcome, err := service.Search(context.Background(), "john@example.com", Options{PIICheckEnabled: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if outcome.BlockedReason == "" {
		t.Fatal("expected query to be blocked")
	}
}

func TestSearch_InjectionCheckFiltersResults(t *testing.T) {
	searcher := &stubSearcher{
		results: []search.Result{
			{Title: "Safe", URL: "https://example.com/safe", Content: "safe"},
			{Title: "Unsafe", URL: "https://example.com/unsafe", Content: "bad instructions"},
		},
	}
	sg := &stubSafeguard{blocked: map[string]string{"bad instructions": "injection"}}
	service := NewService(searcher, nil, sg)

	outcome, err := service.Search(context.Background(), "topic", Options{InjectionCheckEnabled: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(outcome.Results) != 1 {
		t.Fatalf("expected injection-free result only, got %d", len(outcome.Results))
	}
	if outcome.Results[0].Title != "Safe" {
		t.Fatalf("expected Safe result to remain, got %q", outcome.Results[0].Title)
	}
}

func TestSearch_UpstreamErrorPropagates(t *testing.T) {
	service := NewService(&stubSearcher{err: errors.New("boom")}, nil, nil)
	if _, err := service.Search(context.Background(), "topic", Options{}); err == nil {
		t.Fatal("expected upstream error to propagate")
	}
}

func TestFetchDetailed_PreservesURLOrder(t *testing.T) {
	fetcher := &stubFetcher{
		results: []fetch.URLResult{
			{URL: "https://example.com/a", Status: fetch.FetchStatusCompleted, Content: "A"},
			{URL: "https://example.com/b", Status: fetch.FetchStatusFailed, Error: "blocked"},
		},
	}
	service := NewService(nil, fetcher, nil)

	results := service.FetchDetailed(context.Background(), []string{
		"https://example.com/a",
		"https://example.com/b",
	}, Options{})

	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Status != fetch.FetchStatusCompleted || results[1].Status != fetch.FetchStatusFailed {
		t.Fatalf("unexpected statuses: %+v", results)
	}
}

func TestFetchDetailed_InjectionCheckMarksFailure(t *testing.T) {
	fetcher := &stubFetcher{
		results: []fetch.URLResult{
			{URL: "https://example.com/a", Status: fetch.FetchStatusCompleted, Content: "safe"},
			{URL: "https://example.com/b", Status: fetch.FetchStatusCompleted, Content: "bad instructions"},
		},
	}
	sg := &stubSafeguard{blocked: map[string]string{"bad instructions": "injection"}}
	service := NewService(nil, fetcher, sg)

	results := service.FetchDetailed(context.Background(), []string{
		"https://example.com/a",
		"https://example.com/b",
	}, Options{InjectionCheckEnabled: true})

	if results[0].Status != fetch.FetchStatusCompleted {
		t.Fatalf("expected safe page to remain completed, got %+v", results[0])
	}
	if results[1].Status != fetch.FetchStatusFailed {
		t.Fatalf("expected unsafe page to be marked failed, got %+v", results[1])
	}
	if results[1].Content != "" {
		t.Fatalf("expected unsafe content to be cleared, got %q", results[1].Content)
	}
}

func TestFetch_EmptyURLsReturnsNil(t *testing.T) {
	service := NewService(nil, &stubFetcher{}, nil)
	if pages := service.Fetch(context.Background(), nil, Options{}); pages != nil {
		t.Fatalf("expected nil pages, got %+v", pages)
	}
}

func TestFetchDetailed_CapsURLs(t *testing.T) {
	urls := make([]string, maxFetchURLs+5)
	results := make([]fetch.URLResult, 0, maxFetchURLs)
	for i := range urls {
		urls[i] = "https://example.com/" + string(rune('a'+i%26))
		if i < maxFetchURLs {
			results = append(results, fetch.URLResult{URL: urls[i], Status: fetch.FetchStatusCompleted, Content: "ok"})
		}
	}
	fetcher := &stubFetcher{results: results}
	service := NewService(nil, fetcher, nil)

	got := service.FetchDetailed(context.Background(), urls, Options{})
	if len(got) != maxFetchURLs {
		t.Fatalf("expected cap at %d, got %d", maxFetchURLs, len(got))
	}
}
