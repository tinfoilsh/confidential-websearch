package main

import (
	"context"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/engine"
	"github.com/tinfoilsh/confidential-websearch/fetch"
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
	Pages   []fetch.FetchedPage `json:"pages,omitempty"`
	Results []fetch.URLResult   `json:"results"`
}

func newSearchHandler(service *engine.Service, cfg *config.Config) mcp.ToolHandlerFor[SearchArgs, SearchResult] {
	return func(ctx context.Context, req *mcp.CallToolRequest, args SearchArgs) (*mcp.CallToolResult, SearchResult, error) {
		if args.Query == "" {
			return nil, SearchResult{}, fmt.Errorf("query is required")
		}

		outcome, err := service.Search(ctx, args.Query, engine.ToolOptions{
			MaxResults:            args.MaxResults,
			PIICheckEnabled:       cfg.EnablePIICheck,
			InjectionCheckEnabled: cfg.EnableInjectionCheck,
		})
		if err != nil {
			return nil, SearchResult{}, fmt.Errorf("search failed: %w", err)
		}
		if outcome.BlockedReason != "" {
			return nil, SearchResult{}, fmt.Errorf("query blocked: %s", outcome.BlockedReason)
		}

		return nil, SearchResult{Results: outcome.Results}, nil
	}
}

func newFetchHandler(service *engine.Service, cfg *config.Config) mcp.ToolHandlerFor[FetchArgs, FetchResult] {
	return func(ctx context.Context, req *mcp.CallToolRequest, args FetchArgs) (*mcp.CallToolResult, FetchResult, error) {
		if len(args.URLs) == 0 {
			return nil, FetchResult{}, fmt.Errorf("at least one URL is required")
		}

		results := service.FetchDetailed(ctx, args.URLs, engine.ToolOptions{
			PIICheckEnabled:       false,
			InjectionCheckEnabled: cfg.EnableInjectionCheck,
		})

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
