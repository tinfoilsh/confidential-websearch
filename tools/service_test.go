package tools

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	log "github.com/sirupsen/logrus"

	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
)

type stubSearcher struct {
	results []search.Result
	err     error
	called  bool
	query   string
	opts    search.Options
}

func (s *stubSearcher) Name() string { return "stub" }

func (s *stubSearcher) Search(_ context.Context, query string, opts search.Options) ([]search.Result, error) {
	s.called = true
	s.query = query
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
	err     error
}

func (s *stubSafeguard) Check(_ context.Context, content string) (*safeguard.CheckResult, error) {
	if s.err != nil {
		return nil, s.err
	}
	if reason, ok := s.blocked[content]; ok {
		return &safeguard.CheckResult{Violation: true, Rationale: reason}, nil
	}
	return &safeguard.CheckResult{}, nil
}

type stubPIIRedactor struct {
	redacted string
	err      error
}

func (s *stubPIIRedactor) Redact(_ context.Context, _ string) (string, error) {
	return s.redacted, s.err
}

func ptrBool(v bool) *bool { return &v }

func TestSearch_RequiresQuery(t *testing.T) {
	service := NewService(&stubSearcher{}, nil, nil, nil, nil)
	_, err := service.Search(context.Background(), "", Options{})
	if err == nil {
		t.Fatal("expected error for empty query")
	}
}

func TestSearch_ReturnsResults(t *testing.T) {
	searcher := &stubSearcher{
		results: []search.Result{{Title: "One", URL: "https://example.com/1"}},
	}
	service := NewService(searcher, nil, nil, nil, nil)

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
	service := NewService(searcher, nil, nil, nil, nil)

	if _, err := service.Search(context.Background(), "golang", Options{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if searcher.opts.MaxResults != defaultMaxResults {
		t.Fatalf("expected default MaxResults=%d, got %d", defaultMaxResults, searcher.opts.MaxResults)
	}
}

func TestSearch_CapsMaxResults(t *testing.T) {
	results := make([]search.Result, MaxSearchResults+5)
	searcher := &stubSearcher{results: results}
	service := NewService(searcher, nil, nil, nil, nil)

	outcome, err := service.Search(context.Background(), "golang", Options{MaxResults: MaxSearchResults + 5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if searcher.opts.MaxResults != MaxSearchResults {
		t.Fatalf("expected provider limit %d, got %d", MaxSearchResults, searcher.opts.MaxResults)
	}
	if len(outcome.Results) != MaxSearchResults {
		t.Fatalf("expected at most %d results, got %d", MaxSearchResults, len(outcome.Results))
	}
}

func TestSearch_PIIRedactsQuery(t *testing.T) {
	searcher := &stubSearcher{results: []search.Result{{Title: "hit"}}}
	redactor := &stubPIIRedactor{redacted: "hiking trails"}
	service := NewService(searcher, nil, nil, redactor, nil)

	outcome, err := service.Search(context.Background(), "john@example.com hiking trails", Options{PIICheckEnabled: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(outcome.Results) != 1 {
		t.Fatalf("expected search to proceed, got %d results", len(outcome.Results))
	}
	if searcher.query != "hiking trails" {
		t.Fatalf("expected redacted query, got %q", searcher.query)
	}
}

func TestSearch_PIIRedactionFailureStopsSearch(t *testing.T) {
	searcher := &stubSearcher{}
	redactor := &stubPIIRedactor{err: errors.New("redaction failed")}
	service := NewService(searcher, nil, nil, redactor, nil)

	_, err := service.Search(context.Background(), "query", Options{PIICheckEnabled: true})
	if err == nil {
		t.Fatal("expected PII redaction failure")
	}
	if searcher.query != "" {
		t.Fatal("expected search not to run")
	}
}

func TestSearch_PIIOnlyQuerySkipsProvider(t *testing.T) {
	searcher := &stubSearcher{}
	redactor := &stubPIIRedactor{redacted: " "}
	service := NewService(searcher, nil, nil, redactor, nil)

	outcome, err := service.Search(context.Background(), "john@example.com", Options{PIICheckEnabled: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if searcher.called {
		t.Fatal("expected empty redacted query not to reach search provider")
	}
	if len(outcome.Results) != 0 {
		t.Fatalf("expected no results, got %d", len(outcome.Results))
	}
}

func TestInjectionSafeguardErrorsDoNotReachLogs(t *testing.T) {
	const providerDetail = "provider-secret-sentinel"
	var logs bytes.Buffer
	previousOutput := log.StandardLogger().Out
	log.SetOutput(&logs)
	defer log.SetOutput(previousOutput)

	searcher := &stubSearcher{
		results: []search.Result{{
			URL:     "https://example.com",
			Content: "content",
		}},
	}
	fetcher := &stubFetcher{
		results: []fetch.URLResult{{
			URL:     "https://example.com",
			Status:  fetch.FetchStatusCompleted,
			Content: "content",
		}},
	}
	service := NewService(
		searcher,
		fetcher,
		&stubSafeguard{err: errors.New(providerDetail)},
		nil,
		nil,
	)
	options := Options{InjectionCheckEnabled: ptrBool(true)}

	_, err := service.Search(context.Background(), "query", options)
	if err != nil {
		t.Fatalf("unexpected search error: %v", err)
	}
	_ = service.Fetch(
		context.Background(),
		[]string{"https://example.com"},
		options,
	)
	_ = service.FetchDetailed(
		context.Background(),
		[]string{"https://example.com"},
		options,
	)

	if strings.Contains(logs.String(), providerDetail) {
		t.Fatalf("log exposed safeguard response: %s", logs.String())
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
	service := NewService(searcher, nil, sg, nil, nil)

	outcome, err := service.Search(context.Background(), "topic", Options{InjectionCheckEnabled: ptrBool(true)})
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
	service := NewService(&stubSearcher{err: errors.New("boom")}, nil, nil, nil, nil)
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
	service := NewService(nil, fetcher, nil, nil, nil)

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
	service := NewService(nil, fetcher, sg, nil, nil)

	results := service.FetchDetailed(context.Background(), []string{
		"https://example.com/a",
		"https://example.com/b",
	}, Options{InjectionCheckEnabled: ptrBool(true)})

	if results[0].Status != fetch.FetchStatusCompleted {
		t.Fatalf("expected safe page to remain completed, got %+v", results[0])
	}
	if results[1].Status != fetch.FetchStatusFailed {
		t.Fatalf("expected unsafe page to be marked failed, got %+v", results[1])
	}
	if results[1].Content != "" {
		t.Fatalf("expected unsafe content to be cleared, got %q", results[1].Content)
	}
	if results[1].Error != blockedContentError {
		t.Fatalf("expected sanitized block error, got %q", results[1].Error)
	}
}

func TestFetch_EmptyURLsReturnsNil(t *testing.T) {
	service := NewService(nil, &stubFetcher{}, nil, nil, nil)
	if pages := service.Fetch(context.Background(), nil, Options{}); pages != nil {
		t.Fatalf("expected nil pages, got %+v", pages)
	}
}

func TestFetchDetailed_CapsURLs(t *testing.T) {
	urls := make([]string, MaxFetchURLs+5)
	results := make([]fetch.URLResult, 0, MaxFetchURLs)
	for i := range urls {
		urls[i] = "https://example.com/" + string(rune('a'+i%26))
		if i < MaxFetchURLs {
			results = append(results, fetch.URLResult{URL: urls[i], Status: fetch.FetchStatusCompleted, Content: "ok"})
		}
	}
	fetcher := &stubFetcher{results: results}
	service := NewService(nil, fetcher, nil, nil, nil)

	got := service.FetchDetailed(context.Background(), urls, Options{})
	if len(got) != MaxFetchURLs {
		t.Fatalf("expected cap at %d, got %d", MaxFetchURLs, len(got))
	}
}

type stubRanker struct {
	inBucket map[string]bool
}

func (r stubRanker) InTopBucket(host string) bool { return r.inBucket[host] }

type recordingSafeguard struct {
	blocked map[string]string
	mu      sync.Mutex
	checked []string
}

func (r *recordingSafeguard) Check(_ context.Context, content string) (*safeguard.CheckResult, error) {
	r.mu.Lock()
	r.checked = append(r.checked, content)
	r.mu.Unlock()
	if reason, ok := r.blocked[content]; ok {
		return &safeguard.CheckResult{Violation: true, Rationale: reason}, nil
	}
	return &safeguard.CheckResult{}, nil
}

func (r *recordingSafeguard) snapshot() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, len(r.checked))
	copy(out, r.checked)
	return out
}

func TestSearch_DefaultSkipsTopBucketHosts(t *testing.T) {
	searcher := &stubSearcher{
		results: []search.Result{
			{Title: "Top", URL: "https://example.com/safe", Content: "top-content"},
			{Title: "Tail", URL: "https://obscure.test/safe", Content: "tail-content"},
		},
	}
	sg := &recordingSafeguard{}
	ranker := stubRanker{inBucket: map[string]bool{"example.com": true}}
	service := NewService(searcher, nil, sg, nil, ranker)

	if _, err := service.Search(context.Background(), "topic", Options{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	checked := sg.snapshot()
	if len(checked) != 1 || checked[0] != "tail-content" {
		t.Fatalf("expected only tail-content to be checked, got %v", checked)
	}
}

func TestSearch_ExplicitOptInChecksAll(t *testing.T) {
	searcher := &stubSearcher{
		results: []search.Result{
			{Title: "Top", URL: "https://example.com/safe", Content: "top-content"},
			{Title: "Tail", URL: "https://obscure.test/safe", Content: "tail-content"},
		},
	}
	sg := &recordingSafeguard{}
	ranker := stubRanker{inBucket: map[string]bool{"example.com": true}}
	service := NewService(searcher, nil, sg, nil, ranker)

	if _, err := service.Search(context.Background(), "topic", Options{InjectionCheckEnabled: ptrBool(true)}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	checked := sg.snapshot()
	if len(checked) != 2 {
		t.Fatalf("expected both items to be checked, got %v", checked)
	}
}

func TestSearch_ExplicitOptOutChecksNothing(t *testing.T) {
	searcher := &stubSearcher{
		results: []search.Result{
			{Title: "Tail", URL: "https://obscure.test/safe", Content: "tail-content"},
		},
	}
	sg := &recordingSafeguard{}
	service := NewService(searcher, nil, sg, nil, nil)

	if _, err := service.Search(context.Background(), "topic", Options{InjectionCheckEnabled: ptrBool(false)}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	checked := sg.snapshot()
	if len(checked) != 0 {
		t.Fatalf("expected no safeguard calls, got %v", checked)
	}
}

func TestFetchDetailed_DefaultSkipsTopBucketHosts(t *testing.T) {
	fetcher := &stubFetcher{
		results: []fetch.URLResult{
			{URL: "https://example.com/a", Status: fetch.FetchStatusCompleted, Content: "top-content"},
			{URL: "https://obscure.test/b", Status: fetch.FetchStatusCompleted, Content: "tail-content"},
		},
	}
	sg := &recordingSafeguard{}
	ranker := stubRanker{inBucket: map[string]bool{"example.com": true}}
	service := NewService(nil, fetcher, sg, nil, ranker)

	results := service.FetchDetailed(context.Background(), []string{
		"https://example.com/a",
		"https://obscure.test/b",
	}, Options{})

	checked := sg.snapshot()
	if len(checked) != 1 || checked[0] != "tail-content" {
		t.Fatalf("expected only tail-content to be checked, got %v", checked)
	}
	for _, r := range results {
		if r.Status != fetch.FetchStatusCompleted {
			t.Fatalf("expected both pages to remain completed, got %+v", results)
		}
	}
}

func TestFetchDetailed_ExplicitOptInChecksAll(t *testing.T) {
	fetcher := &stubFetcher{
		results: []fetch.URLResult{
			{URL: "https://example.com/a", Status: fetch.FetchStatusCompleted, Content: "top-content"},
			{URL: "https://obscure.test/b", Status: fetch.FetchStatusCompleted, Content: "tail-content"},
		},
	}
	sg := &recordingSafeguard{}
	ranker := stubRanker{inBucket: map[string]bool{"example.com": true}}
	service := NewService(nil, fetcher, sg, nil, ranker)

	service.FetchDetailed(context.Background(), []string{
		"https://example.com/a",
		"https://obscure.test/b",
	}, Options{InjectionCheckEnabled: ptrBool(true)})

	checked := sg.snapshot()
	if len(checked) != 2 {
		t.Fatalf("expected both items to be checked, got %v", checked)
	}
}

func TestFetchDetailed_ExplicitOptOutChecksNothing(t *testing.T) {
	fetcher := &stubFetcher{
		results: []fetch.URLResult{
			{URL: "https://example.com/a", Status: fetch.FetchStatusCompleted, Content: "bad instructions"},
		},
	}
	sg := &recordingSafeguard{blocked: map[string]string{"bad instructions": "injection"}}
	service := NewService(nil, fetcher, sg, nil, nil)

	results := service.FetchDetailed(context.Background(), []string{"https://example.com/a"}, Options{InjectionCheckEnabled: ptrBool(false)})
	checked := sg.snapshot()
	if len(checked) != 0 {
		t.Fatalf("expected no safeguard calls, got %v", checked)
	}
	if results[0].Status != fetch.FetchStatusCompleted {
		t.Fatalf("expected page to remain completed, got %+v", results[0])
	}
}
