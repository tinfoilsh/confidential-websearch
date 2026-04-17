package main

import (
	"context"
	"os"
	"strings"

	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/search"
	"github.com/tinfoilsh/confidential-websearch/tools"
)

func isLocalTestMode() bool {
	return os.Getenv("LOCAL_TEST_MODE") == "1"
}

func newLocalTestService() *tools.Service {
	return tools.NewService(localTestSearcher{}, localTestFetcher{}, nil)
}

type localTestSearcher struct{}

func (localTestSearcher) Name() string {
	return "local-test"
}

func (localTestSearcher) Search(_ context.Context, query string, opts search.Options) ([]search.Result, error) {
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

type localTestFetcher struct{}

func (localTestFetcher) FetchURLs(ctx context.Context, urls []string) []fetch.FetchedPage {
	results := localTestFetcher{}.FetchURLResults(ctx, urls)
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

func (localTestFetcher) FetchURLResults(_ context.Context, urls []string) []fetch.URLResult {
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
