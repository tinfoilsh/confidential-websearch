package main

import (
	"context"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
)

type SearchArgs struct {
	Query      string `json:"query" jsonschema:"the search query to execute"`
	MaxResults int    `json:"max_results,omitempty" jsonschema:"maximum number of results to return,default 10"`
}

type SearchResult struct {
	Results []search.Result `json:"results"`
}

type FetchArgs struct {
	URLs []string `json:"urls" jsonschema:"URLs to fetch as markdown,max 5"`
}

type FetchResult struct {
	Pages []fetch.FetchedPage `json:"pages"`
}

func newSearchHandler(searcher search.Provider, sg *safeguard.Client, cfg *config.Config) mcp.ToolHandlerFor[SearchArgs, SearchResult] {
	return func(ctx context.Context, req *mcp.CallToolRequest, args SearchArgs) (*mcp.CallToolResult, SearchResult, error) {
		if args.Query == "" {
			return nil, SearchResult{}, fmt.Errorf("query is required")
		}

		if cfg.EnablePIICheck {
			check, err := sg.Check(ctx, safeguard.PIILeakagePolicy, args.Query)
			if err == nil && check.Violation {
				return nil, SearchResult{}, fmt.Errorf("query blocked: %s", check.Rationale)
			}
		}

		maxResults := args.MaxResults
		if maxResults <= 0 {
			maxResults = config.DefaultMaxSearchResults
		}
		if maxResults > 20 {
			maxResults = 20
		}

		results, err := searcher.Search(ctx, args.Query, maxResults)
		if err != nil {
			return nil, SearchResult{}, fmt.Errorf("search failed: %w", err)
		}

		if cfg.EnableInjectionCheck && len(results) > 0 {
			contents := make([]string, len(results))
			for i, r := range results {
				contents[i] = r.Content
			}
			checks := safeguard.CheckItems(ctx, sg, safeguard.PromptInjectionPolicy, contents)
			var filtered []search.Result
			for _, c := range checks {
				if c.Err != nil || !c.Violation {
					filtered = append(filtered, results[c.Index])
				}
			}
			results = filtered
		}

		return nil, SearchResult{Results: results}, nil
	}
}

func newFetchHandler(fetcher *fetch.Fetcher, sg *safeguard.Client, cfg *config.Config) mcp.ToolHandlerFor[FetchArgs, FetchResult] {
	return func(ctx context.Context, req *mcp.CallToolRequest, args FetchArgs) (*mcp.CallToolResult, FetchResult, error) {
		if len(args.URLs) == 0 {
			return nil, FetchResult{}, fmt.Errorf("at least one URL is required")
		}
		if len(args.URLs) > 5 {
			args.URLs = args.URLs[:5]
		}

		pages := fetcher.FetchURLs(ctx, args.URLs)

		if cfg.EnableInjectionCheck && len(pages) > 0 {
			contents := make([]string, len(pages))
			for i, p := range pages {
				contents[i] = p.Content
			}
			checks := safeguard.CheckItems(ctx, sg, safeguard.PromptInjectionPolicy, contents)
			var filtered []fetch.FetchedPage
			for _, c := range checks {
				if c.Err != nil || !c.Violation {
					filtered = append(filtered, pages[c.Index])
				}
			}
			pages = filtered
		}

		return nil, FetchResult{Pages: pages}, nil
	}
}
