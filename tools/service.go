package tools

import (
	"context"
	"fmt"

	log "github.com/sirupsen/logrus"

	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
)

const (
	// maxFetchURLs caps how many URLs a single fetch tool call may process.
	maxFetchURLs = 20

	// defaultMaxResults is the search result count applied when the caller
	// does not specify max_results.
	defaultMaxResults = 8

	// defaultMaxContentChars is the per-result snippet/text budget applied
	// when the caller does not specify max_content_chars.
	defaultMaxContentChars = 700
)

type URLFetcher interface {
	FetchURLs(ctx context.Context, urls []string) []fetch.FetchedPage
}

type DetailedURLFetcher interface {
	FetchURLResults(ctx context.Context, urls []string) []fetch.URLResult
}

type SafeguardChecker interface {
	Check(ctx context.Context, policy, content string) (*safeguard.CheckResult, error)
}

type Options struct {
	MaxResults            int
	MaxContentCharacters  int
	ContentMode           search.ContentMode
	PIICheckEnabled       bool
	InjectionCheckEnabled bool
	UserLocationCountry   string
	AllowedDomains        []string
	ExcludedDomains       []string
	Category              search.Category
	StartPublishedDate    string
	EndPublishedDate      string
	MaxAgeHours           *int
}

type SearchOutcome struct {
	Results       []search.Result
	BlockedReason string
}

type Service struct {
	searcher  search.Provider
	fetcher   URLFetcher
	safeguard SafeguardChecker
}

func NewService(searcher search.Provider, fetcher URLFetcher, safeguardChecker SafeguardChecker) *Service {
	return &Service{
		searcher:  searcher,
		fetcher:   fetcher,
		safeguard: safeguardChecker,
	}
}

func (s *Service) Search(ctx context.Context, query string, opts Options) (SearchOutcome, error) {
	if query == "" {
		return SearchOutcome{}, fmt.Errorf("query is required")
	}

	if opts.PIICheckEnabled && s.safeguard != nil {
		check, err := s.safeguard.Check(ctx, safeguard.PIILeakagePolicy, query)
		if err != nil {
			log.WithError(err).Warn("PII safeguard unavailable; allowing search to continue")
		} else if check.Violation {
			return SearchOutcome{BlockedReason: check.Rationale}, nil
		}
	}

	maxResults := opts.MaxResults
	if maxResults <= 0 {
		maxResults = defaultMaxResults
	}

	maxContentChars := opts.MaxContentCharacters
	if maxContentChars <= 0 {
		maxContentChars = defaultMaxContentChars
	}

	contentMode := opts.ContentMode
	if contentMode == "" {
		contentMode = search.ContentModeHighlights
	}

	results, err := s.searcher.Search(ctx, query, search.Options{
		MaxResults:           maxResults,
		MaxContentCharacters: maxContentChars,
		ContentMode:          contentMode,
		UserLocationCountry:  opts.UserLocationCountry,
		AllowedDomains:       opts.AllowedDomains,
		ExcludedDomains:      opts.ExcludedDomains,
		Category:             opts.Category,
		StartPublishedDate:   opts.StartPublishedDate,
		EndPublishedDate:     opts.EndPublishedDate,
		MaxAgeHours:          opts.MaxAgeHours,
	})
	if err != nil {
		return SearchOutcome{}, err
	}

	if opts.InjectionCheckEnabled && len(results) > 0 && s.safeguard != nil {
		results = filterSearchResults(ctx, s.safeguard, results)
	}

	return SearchOutcome{Results: results}, nil
}

func (s *Service) Fetch(ctx context.Context, urls []string, opts Options) []fetch.FetchedPage {
	if len(urls) == 0 || s.fetcher == nil {
		return nil
	}
	if len(urls) > maxFetchURLs {
		urls = urls[:maxFetchURLs]
	}

	pages := s.fetcher.FetchURLs(ctx, urls)
	if opts.InjectionCheckEnabled && len(pages) > 0 && s.safeguard != nil {
		pages = filterFetchedPages(ctx, s.safeguard, pages)
	}

	return pages
}

func (s *Service) FetchDetailed(ctx context.Context, urls []string, opts Options) []fetch.URLResult {
	if len(urls) == 0 || s.fetcher == nil {
		return nil
	}
	if len(urls) > maxFetchURLs {
		urls = urls[:maxFetchURLs]
	}

	detailedFetcher, ok := s.fetcher.(DetailedURLFetcher)
	if !ok {
		pages := s.Fetch(ctx, urls, opts)
		pagesByURL := make(map[string][]fetch.FetchedPage, len(pages))
		for _, page := range pages {
			pagesByURL[page.URL] = append(pagesByURL[page.URL], page)
		}

		results := make([]fetch.URLResult, 0, len(urls))
		for _, rawURL := range urls {
			queue := pagesByURL[rawURL]
			if len(queue) == 0 {
				results = append(results, fetch.URLResult{
					URL:    rawURL,
					Status: fetch.FetchStatusFailed,
					Error:  "fetch failed",
				})
				continue
			}

			page := queue[0]
			pagesByURL[rawURL] = queue[1:]
			results = append(results, fetch.URLResult{
				URL:     rawURL,
				Status:  fetch.FetchStatusCompleted,
				Content: page.Content,
			})
		}
		return results
	}

	results := detailedFetcher.FetchURLResults(ctx, urls)
	if opts.InjectionCheckEnabled && len(results) > 0 && s.safeguard != nil {
		results = filterFetchResults(ctx, s.safeguard, results)
	}

	return results
}

func filterSearchResults(ctx context.Context, checker SafeguardChecker, results []search.Result) []search.Result {
	contents := make([]string, len(results))
	for i, result := range results {
		contents[i] = result.Content
	}

	checks := safeguard.CheckItems(ctx, checker, safeguard.PromptInjectionPolicy, contents)
	filtered := make([]search.Result, 0, len(results))
	for i, check := range checks {
		if check.Err != nil {
			log.WithError(check.Err).Warn("prompt injection safeguard unavailable; keeping search result")
			filtered = append(filtered, results[i])
			continue
		}
		if !check.Violation {
			filtered = append(filtered, results[i])
		}
	}
	return filtered
}

func filterFetchedPages(ctx context.Context, checker SafeguardChecker, pages []fetch.FetchedPage) []fetch.FetchedPage {
	contents := make([]string, len(pages))
	for i, page := range pages {
		contents[i] = page.Content
	}

	checks := safeguard.CheckItems(ctx, checker, safeguard.PromptInjectionPolicy, contents)
	filtered := make([]fetch.FetchedPage, 0, len(pages))
	for i, check := range checks {
		if check.Err != nil {
			log.WithError(check.Err).Warn("prompt injection safeguard unavailable; keeping fetched page")
			filtered = append(filtered, pages[i])
			continue
		}
		if !check.Violation {
			filtered = append(filtered, pages[i])
		}
	}
	return filtered
}

func filterFetchResults(ctx context.Context, checker SafeguardChecker, results []fetch.URLResult) []fetch.URLResult {
	indexes := make([]int, 0, len(results))
	contents := make([]string, 0, len(results))
	filtered := make([]fetch.URLResult, len(results))
	copy(filtered, results)

	for i, result := range results {
		if result.Status != fetch.FetchStatusCompleted || result.Content == "" {
			continue
		}
		indexes = append(indexes, i)
		contents = append(contents, result.Content)
	}
	if len(contents) == 0 {
		return filtered
	}

	checks := safeguard.CheckItems(ctx, checker, safeguard.PromptInjectionPolicy, contents)
	for i, check := range checks {
		if check.Err != nil {
			log.WithError(check.Err).Warn("prompt injection safeguard unavailable; keeping fetched result")
			continue
		}
		if !check.Violation {
			continue
		}

		resultIndex := indexes[i]
		filtered[resultIndex].Status = fetch.FetchStatusFailed
		filtered[resultIndex].Content = ""
		filtered[resultIndex].Error = check.Rationale
	}

	return filtered
}
