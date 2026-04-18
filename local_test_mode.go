package main

import (
	"context"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/search"
	"github.com/tinfoilsh/confidential-websearch/tools"
)

func isLocalTestMode() bool {
	return os.Getenv("LOCAL_TEST_MODE") == "1"
}

// LocalCallRecorder captures the most recent search and fetch calls seen by
// the fixture-mode MCP service. The eval harness reads this record through
// the /debug/last-call endpoint to assert that request-shaping options
// (user_location, allowed_domains, excluded_domains, category, etc.) are
// plumbed end-to-end from the caller through the router into the MCP tools.
//
// The recorder is only wired up when LOCAL_TEST_MODE=1 so it cannot leak
// request metadata from a real deployment.
type LocalCallRecorder struct {
	mu sync.Mutex

	lastSearchAt    time.Time
	lastSearchQuery string
	lastSearchOpts  search.Options

	lastFetchAt   time.Time
	lastFetchURLs []string
}

// NewLocalCallRecorder returns an empty recorder ready to be handed to the
// fixture searcher/fetcher.
func NewLocalCallRecorder() *LocalCallRecorder {
	return &LocalCallRecorder{}
}

func (r *LocalCallRecorder) recordSearch(query string, opts search.Options) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lastSearchAt = time.Now().UTC()
	r.lastSearchQuery = query
	r.lastSearchOpts = opts
}

func (r *LocalCallRecorder) recordFetch(urls []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lastFetchAt = time.Now().UTC()
	copied := make([]string, len(urls))
	copy(copied, urls)
	r.lastFetchURLs = copied
}

// LastSearch returns a snapshot of the most recent search call.
func (r *LocalCallRecorder) LastSearch() (time.Time, string, search.Options) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.lastSearchAt, r.lastSearchQuery, r.lastSearchOpts
}

// LastFetch returns a snapshot of the most recent fetch call.
func (r *LocalCallRecorder) LastFetch() (time.Time, []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	copied := make([]string, len(r.lastFetchURLs))
	copy(copied, r.lastFetchURLs)
	return r.lastFetchAt, copied
}

func newLocalTestService() (*tools.Service, *LocalCallRecorder) {
	recorder := NewLocalCallRecorder()
	searcher := localTestSearcher{recorder: recorder}
	fetcher := localTestFetcher{recorder: recorder}
	return tools.NewService(searcher, fetcher, nil), recorder
}

type localTestSearcher struct {
	recorder *LocalCallRecorder
}

func (localTestSearcher) Name() string {
	return "local-test"
}

func (s localTestSearcher) Search(_ context.Context, query string, opts search.Options) ([]search.Result, error) {
	if s.recorder != nil {
		s.recorder.recordSearch(query, opts)
	}
	results := []search.Result{
		{
			Title:   "Local Cat Almanac 2026",
			URL:     "https://local.test/cats/almanac",
			Content: "A short almanac entry about Nimbus the cat and the morning routine.",
		},
		{
			Title:   "Neighborhood Cat Gazette",
			URL:     "https://local.test/cats/gazette",
			Content: "A local column about cats in the sunroom and the cushions they prefer.",
		},
	}

	if strings.Contains(strings.ToLower(query), "gazette") {
		results[0], results[1] = results[1], results[0]
	}

	if opts.MaxResults > 0 && len(results) > opts.MaxResults {
		results = results[:opts.MaxResults]
	}

	return results, nil
}

type localTestFetcher struct {
	recorder *LocalCallRecorder
}

func (f localTestFetcher) FetchURLs(ctx context.Context, urls []string) []fetch.FetchedPage {
	results := f.FetchURLResults(ctx, urls)
	pages := make([]fetch.FetchedPage, 0, len(results))
	for _, result := range results {
		if result.Status != fetch.FetchStatusCompleted {
			continue
		}
		pages = append(pages, fetch.FetchedPage{
			URL:     result.URL,
			Content: result.Content,
		})
	}
	return pages
}

func (f localTestFetcher) FetchURLResults(_ context.Context, urls []string) []fetch.URLResult {
	if f.recorder != nil {
		f.recorder.recordFetch(urls)
	}
	results := make([]fetch.URLResult, 0, len(urls))
	for _, rawURL := range urls {
		switch rawURL {
		case "https://local.test/cats/almanac":
			results = append(results, fetch.URLResult{
				URL:     rawURL,
				Status:  fetch.FetchStatusCompleted,
				Content: "Local Cat Almanac 2026 says Nimbus naps for exactly 17 minutes after breakfast before inspecting the window.",
			})
		case "https://local.test/cats/gazette":
			results = append(results, fetch.URLResult{
				URL:     rawURL,
				Status:  fetch.FetchStatusCompleted,
				Content: "Neighborhood Cat Gazette reports that the cats in the sunroom prefer saffron cushions because they stay warm in the afternoon light.",
			})
		default:
			results = append(results, fetch.URLResult{
				URL:    rawURL,
				Status: fetch.FetchStatusFailed,
				Error:  "local test fixture not found",
			})
		}
	}
	return results
}
