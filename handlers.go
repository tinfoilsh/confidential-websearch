package main

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"

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

		outcome, err := svc.Search(ctx, args.Query, tools.Options{
			MaxResults:            args.MaxResults,
			PIICheckEnabled:       resolveSafetyFlag(httpReq, headerPIICheck, cfg.EnablePIICheck),
			InjectionCheckEnabled: resolveSafetyFlag(httpReq, headerInjectionCheck, cfg.EnableInjectionCheck),
			UserLocationCountry:   strings.ToUpper(strings.TrimSpace(args.UserLocationCountry)),
			AllowedDomains:        normalizeDomains(args.AllowedDomains),
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
