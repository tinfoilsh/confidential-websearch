package main

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/search"
	"github.com/tinfoilsh/confidential-websearch/tools"
)

// Per-request safety override headers set by the model-router when the caller
// sends `pii_check_options` or `prompt_injection_check_options` on the
// originating OpenAI request. When the header is absent the server falls back
// to its env-driven defaults, so self-hosted deployments keep working without
// the router in front of them.
const (
	headerPIICheck       = "X-Tinfoil-Tool-PII-Check"
	headerInjectionCheck = "X-Tinfoil-Tool-Injection-Check"
)

// resolveSafetyFlag picks between a per-request header override and the
// server-level default. An empty header means the caller (router) said nothing
// and the env-backed config value wins; any other value is parsed as a bool
// with the same semantics as `strconv.ParseBool`. Unparseable header values
// fall back to the default so a malformed header can never silently weaken
// filtering compared to what the operator configured.
func resolveSafetyFlag(req *http.Request, header string, fallback bool) bool {
	if req == nil {
		return fallback
	}
	value := strings.TrimSpace(req.Header.Get(header))
	if value == "" {
		return fallback
	}
	switch strings.ToLower(value) {
	case "true", "1":
		return true
	case "false", "0":
		return false
	default:
		return fallback
	}
}

type SearchArgs struct {
	Query               string   `json:"query" jsonschema:"Natural language search query. Be specific and descriptive for better results. Max ~400 characters."`
	MaxResults          int      `json:"max_results,omitempty" jsonschema:"Number of results to return (1-30). Defaults to 10 if omitted. Use fewer for focused queries and more for broad research."`
	UserLocationCountry string   `json:"user_location_country,omitempty" jsonschema:"ISO 3166-1 alpha-2 country code to bias results toward (e.g. 'US', 'GB', 'DE'). Maps to OpenAI web_search_options.user_location.approximate.country."`
	AllowedDomains      []string `json:"allowed_domains,omitempty" jsonschema:"If set, only return results from these domains. Maps to OpenAI filters.allowed_domains."`
	ExcludedDomains     []string `json:"excluded_domains,omitempty" jsonschema:"If set, drop results from these domains. Useful for filtering out aggregators, SEO farms, or known-low-quality sources."`
	ContentMode         string   `json:"content_mode,omitempty" jsonschema:"Per-result content granularity: 'highlights' (default) returns key excerpts relevant to the query; 'text' returns the full page text as markdown."`
	MaxContentChars     int      `json:"max_content_chars,omitempty" jsonschema:"Per-result character budget for the snippet/text returned in each hit. Defaults to 700. Higher values return more context at the cost of more tokens."`
	Category            string   `json:"category,omitempty" jsonschema:"Restrict results to one of: 'company', 'people', 'research paper', 'news', 'personal site', 'financial report'. Note: 'company' and 'people' disable date filters and excluded_domains."`
	StartPublishedDate  string   `json:"start_published_date,omitempty" jsonschema:"Only include results published at or after this ISO-8601 date (e.g. '2024-01-01' or '2024-01-01T00:00:00Z')."`
	EndPublishedDate    string   `json:"end_published_date,omitempty" jsonschema:"Only include results published at or before this ISO-8601 date."`
	MaxAgeHours         *int     `json:"max_age_hours,omitempty" jsonschema:"Cache freshness control. 0 forces livecrawl on every result (freshest, slowest). -1 disables livecrawl (cache-only, fastest). Omit for Exa's default (livecrawl only when uncached)."`
}

type SearchResult struct {
	Results []search.Result `json:"results"`
}

type FetchArgs struct {
	URLs           []string `json:"urls" jsonschema:"One or more HTTP/HTTPS URLs to fetch. Each page is rendered in a headless browser and converted to clean markdown. Maximum 20 URLs per request."`
	AllowedDomains []string `json:"allowed_domains,omitempty" jsonschema:"If set, reject any URL whose host is not in the list. Maps to OpenAI filters.allowed_domains."`
}

type FetchResult struct {
	Pages   []fetch.FetchedPage `json:"pages,omitempty"`
	Results []fetch.URLResult   `json:"results"`
}

func newSearchHandler(svc *tools.Service, cfg *config.Config, httpReq *http.Request) mcp.ToolHandlerFor[SearchArgs, SearchResult] {
	return func(ctx context.Context, req *mcp.CallToolRequest, args SearchArgs) (*mcp.CallToolResult, SearchResult, error) {
		if args.Query == "" {
			return nil, SearchResult{}, fmt.Errorf("the 'query' parameter is required: provide a non-empty search query string describing what you want to find")
		}

		contentMode, err := parseContentMode(args.ContentMode)
		if err != nil {
			return nil, SearchResult{}, err
		}

		category, err := parseCategory(args.Category)
		if err != nil {
			return nil, SearchResult{}, err
		}

		startPublishedDate, err := normalizeDate("start_published_date", args.StartPublishedDate)
		if err != nil {
			return nil, SearchResult{}, err
		}
		endPublishedDate, err := normalizeDate("end_published_date", args.EndPublishedDate)
		if err != nil {
			return nil, SearchResult{}, err
		}

		excludedDomains := normalizeDomains(args.ExcludedDomains)

		if err := validateCategoryCompatibility(category, startPublishedDate, endPublishedDate, excludedDomains); err != nil {
			return nil, SearchResult{}, err
		}

		outcome, err := svc.Search(ctx, args.Query, tools.Options{
			MaxResults:            args.MaxResults,
			MaxContentCharacters:  args.MaxContentChars,
			ContentMode:           contentMode,
			PIICheckEnabled:       resolveSafetyFlag(httpReq, headerPIICheck, cfg.EnablePIICheck),
			InjectionCheckEnabled: resolveSafetyFlag(httpReq, headerInjectionCheck, cfg.EnableInjectionCheck),
			UserLocationCountry:   strings.ToUpper(strings.TrimSpace(args.UserLocationCountry)),
			AllowedDomains:        normalizeDomains(args.AllowedDomains),
			ExcludedDomains:       excludedDomains,
			Category:              category,
			StartPublishedDate:    startPublishedDate,
			EndPublishedDate:      endPublishedDate,
			MaxAgeHours:           args.MaxAgeHours,
		})
		if err != nil {
			return nil, SearchResult{}, fmt.Errorf("search request failed: %w — try rephrasing the query to be simpler or more specific, or retry after a short delay", err)
		}
		if outcome.BlockedReason != "" {
			return nil, SearchResult{}, fmt.Errorf("query was blocked by safety filters: %s — rephrase the query to remove personal information or sensitive content, then retry", outcome.BlockedReason)
		}

		return nil, SearchResult{Results: outcome.Results}, nil
	}
}

func newFetchHandler(svc *tools.Service, cfg *config.Config, httpReq *http.Request) mcp.ToolHandlerFor[FetchArgs, FetchResult] {
	return func(ctx context.Context, req *mcp.CallToolRequest, args FetchArgs) (*mcp.CallToolResult, FetchResult, error) {
		if len(args.URLs) == 0 {
			return nil, FetchResult{}, fmt.Errorf("the 'urls' parameter is required: provide at least one valid HTTP or HTTPS URL to fetch")
		}

		urls, rejected := splitAllowedURLs(args.URLs, normalizeDomains(args.AllowedDomains))
		if len(urls) == 0 {
			return nil, FetchResult{Results: rejected}, nil
		}

		results := svc.FetchDetailed(ctx, urls, tools.Options{
			PIICheckEnabled:       false,
			InjectionCheckEnabled: resolveSafetyFlag(httpReq, headerInjectionCheck, cfg.EnableInjectionCheck),
		})
		results = append(results, rejected...)

		pages := make([]fetch.FetchedPage, 0, len(results))
		for _, result := range results {
			if result.Status != fetch.FetchStatusCompleted || result.Content == "" {
				continue
			}
			pages = append(pages, fetch.FetchedPage{
				URL:     result.URL,
				Content: result.Content,
			})
		}

		return nil, FetchResult{Pages: pages, Results: results}, nil
	}
}

// parseContentMode normalizes the optional content_mode argument into the
// search package's enum. An empty string means "use the service default"
// (highlights) so existing callers keep their current behavior.
func parseContentMode(raw string) (search.ContentMode, error) {
	trimmed := strings.ToLower(strings.TrimSpace(raw))
	switch trimmed {
	case "":
		return "", nil
	case string(search.ContentModeText):
		return search.ContentModeText, nil
	case string(search.ContentModeHighlights):
		return search.ContentModeHighlights, nil
	default:
		return "", fmt.Errorf("invalid content_mode %q: expected 'text' or 'highlights'", raw)
	}
}

// parseCategory validates the category argument against Exa's enum. Empty
// means no category filter.
func parseCategory(raw string) (search.Category, error) {
	trimmed := strings.ToLower(strings.TrimSpace(raw))
	if trimmed == "" {
		return "", nil
	}
	switch search.Category(trimmed) {
	case search.CategoryCompany,
		search.CategoryPeople,
		search.CategoryResearchPaper,
		search.CategoryNews,
		search.CategoryPersonalSite,
		search.CategoryFinancialReport:
		return search.Category(trimmed), nil
	default:
		return "", fmt.Errorf("invalid category %q: expected one of 'company', 'people', 'research paper', 'news', 'personal site', 'financial report'", raw)
	}
}

// normalizeDate validates an ISO-8601 date/datetime. Accepts both the date-only
// (YYYY-MM-DD) and full RFC 3339 shapes Exa documents. Returns the canonical
// RFC 3339 form with a UTC midnight when only a date was supplied.
func normalizeDate(field, raw string) (string, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return "", nil
	}
	if t, err := time.Parse(time.RFC3339, trimmed); err == nil {
		return t.UTC().Format(time.RFC3339), nil
	}
	if t, err := time.Parse("2006-01-02", trimmed); err == nil {
		return t.UTC().Format(time.RFC3339), nil
	}
	return "", fmt.Errorf("invalid %s %q: expected ISO-8601 date (YYYY-MM-DD) or RFC 3339 datetime", field, raw)
}

// validateCategoryCompatibility rejects filter combinations that Exa will 400
// on, so callers get a clear error rather than an opaque provider failure.
func validateCategoryCompatibility(category search.Category, startPublished, endPublished string, excludedDomains []string) error {
	if category != search.CategoryCompany && category != search.CategoryPeople {
		return nil
	}
	if startPublished != "" || endPublished != "" {
		return fmt.Errorf("category %q does not support date filters: remove start_published_date / end_published_date", category)
	}
	if len(excludedDomains) > 0 {
		return fmt.Errorf("category %q does not support excluded_domains: remove excluded_domains", category)
	}
	return nil
}

// normalizeDomains trims whitespace and lowercases each entry, dropping empties
// and duplicates so downstream matching is case-insensitive and stable.
func normalizeDomains(domains []string) []string {
	if len(domains) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(domains))
	out := make([]string, 0, len(domains))
	for _, domain := range domains {
		trimmed := strings.ToLower(strings.TrimSpace(domain))
		if trimmed == "" {
			continue
		}
		if _, dup := seen[trimmed]; dup {
			continue
		}
		seen[trimmed] = struct{}{}
		out = append(out, trimmed)
	}
	return out
}

// splitAllowedURLs partitions the requested URLs into ones whose host is
// allowed by the domain filter and pre-built failure records for the rest.
// When the filter is empty, every URL is considered allowed.
func splitAllowedURLs(urls, allowedDomains []string) ([]string, []fetch.URLResult) {
	if len(allowedDomains) == 0 {
		return urls, nil
	}
	allowed := make([]string, 0, len(urls))
	rejected := make([]fetch.URLResult, 0)
	for _, raw := range urls {
		if hostAllowed(raw, allowedDomains) {
			allowed = append(allowed, raw)
			continue
		}
		rejected = append(rejected, fetch.URLResult{
			URL:    raw,
			Status: fetch.FetchStatusFailed,
			Error:  fmt.Sprintf("host is not in allowed_domains filter (%s)", strings.Join(allowedDomains, ", ")),
		})
	}
	return allowed, rejected
}

// hostAllowed reports whether rawURL's host is in the allowed-domain list.
// A domain entry matches either the exact host or any subdomain of it, mirroring
// the behavior documented by OpenAI for filters.allowed_domains.
func hostAllowed(rawURL string, allowedDomains []string) bool {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil {
		return false
	}
	host := strings.ToLower(parsed.Hostname())
	if host == "" {
		return false
	}
	for _, domain := range allowedDomains {
		if host == domain || strings.HasSuffix(host, "."+domain) {
			return true
		}
	}
	return false
}
