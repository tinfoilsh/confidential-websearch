package main

import (
	"context"
	"net/http"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/tools"
	"github.com/tinfoilsh/confidential-websearch/usage"
)

func newMCPServer(svc *tools.Service, cfg *config.Config, reporter *usage.Reporter, request *http.Request) *mcp.Server {
	server := mcp.NewServer(&mcp.Implementation{
		Name:    "confidential-websearch",
		Version: version,
	}, nil)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "search",
		Description: "Search the web for current information. Returns ranked results with titles, URLs, snippets, and publication dates.",
	}, newSearchHandlerWithUsage(svc, cfg, reporter, request))

	mcp.AddTool(server, &mcp.Tool{
		Name:        "fetch",
		Description: "Fetch a web page and return the rendered markdown content for deeper reading after search.",
	}, newFetchHandlerWithUsage(svc, cfg, reporter, request))

	return server
}

func newSearchHandlerWithUsage(svc *tools.Service, cfg *config.Config, reporter *usage.Reporter, httpReq *http.Request) mcp.ToolHandlerFor[SearchArgs, SearchResult] {
	inner := newSearchHandler(svc, cfg)
	return func(ctx context.Context, req *mcp.CallToolRequest, args SearchArgs) (*mcp.CallToolResult, SearchResult, error) {
		reporter.ReportSession(ctx, httpReq)
		return inner(ctx, req, args)
	}
}

func newFetchHandlerWithUsage(svc *tools.Service, cfg *config.Config, reporter *usage.Reporter, httpReq *http.Request) mcp.ToolHandlerFor[FetchArgs, FetchResult] {
	inner := newFetchHandler(svc, cfg)
	return func(ctx context.Context, req *mcp.CallToolRequest, args FetchArgs) (*mcp.CallToolResult, FetchResult, error) {
		reporter.ReportSession(ctx, httpReq)
		return inner(ctx, req, args)
	}
}
