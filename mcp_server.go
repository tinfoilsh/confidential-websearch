package main

import (
	"context"
	"fmt"
	"net/http"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/tools"
	"github.com/tinfoilsh/confidential-websearch/usage"
)

// searchToolDescription enumerates the accepted arguments so models that
// were trained on other search-tool schemas (e.g. Tavily's recency_days,
// Brave's topn) get a strong hint about what this tool actually takes
// before they start hallucinating alternatives. The enumeration is the
// source of truth paired with permissiveSchema's additionalProperties:true
// safety net.
const searchToolDescription = "Search the web for current information. Returns ranked results with titles, URLs, snippets, and publication dates. Accepted arguments: query (required, string); max_results (int, 1-30); user_location_country (ISO country code); allowed_domains ([]string); excluded_domains ([]string); content_mode (\"highlights\"|\"text\"); max_content_chars (int); category (string); start_published_date (ISO-8601); end_published_date (ISO-8601); max_age_hours (int, -1 disables livecrawl, 0 forces it). Unknown arguments are silently ignored."

const fetchToolDescription = "Fetch one or more web pages and return the rendered markdown content for deeper reading after search. Accepted arguments: urls (required, []string, max 20); allowed_domains ([]string). Unknown arguments are silently ignored."

func newMCPServer(svc *tools.Service, cfg *config.Config, reporter *usage.Reporter, request *http.Request) *mcp.Server {
	server := mcp.NewServer(&mcp.Implementation{
		Name:    "confidential-websearch",
		Version: version,
	}, nil)

	searchSchema, err := permissiveSchema[SearchArgs]()
	if err != nil {
		panic(fmt.Sprintf("building search input schema: %v", err))
	}
	fetchSchema, err := permissiveSchema[FetchArgs]()
	if err != nil {
		panic(fmt.Sprintf("building fetch input schema: %v", err))
	}

	mcp.AddTool(server, &mcp.Tool{
		Name:        "search",
		Description: searchToolDescription,
		InputSchema: searchSchema,
	}, newSearchHandlerWithUsage(svc, cfg, reporter, request))

	mcp.AddTool(server, &mcp.Tool{
		Name:        "fetch",
		Description: fetchToolDescription,
		InputSchema: fetchSchema,
	}, newFetchHandlerWithUsage(svc, cfg, reporter, request))

	return server
}

// permissiveSchema returns the schema inferred from T with
// additionalProperties:false replaced by an open policy. Chat and
// responses model outputs frequently include argument keys that belong
// to other search-tool training distributions (Tavily's recency_days,
// Brave's topn, etc). Dropping the strict additionalProperties check
// lets the handler ignore those keys silently instead of failing the
// tool call with a schema validation error that the model then
// interprets as an instruction to retry. The per-field descriptions
// and required flags from the struct tags are preserved.
func permissiveSchema[T any]() (*jsonschema.Schema, error) {
	schema, err := jsonschema.For[T](nil)
	if err != nil {
		return nil, err
	}
	schema.AdditionalProperties = nil
	return schema, nil
}

func newSearchHandlerWithUsage(svc *tools.Service, cfg *config.Config, reporter *usage.Reporter, httpReq *http.Request) mcp.ToolHandlerFor[SearchArgs, SearchResult] {
	inner := newSearchHandler(svc, cfg, httpReq)
	return func(ctx context.Context, req *mcp.CallToolRequest, args SearchArgs) (*mcp.CallToolResult, SearchResult, error) {
		reporter.ReportSession(ctx, httpReq)
		return inner(ctx, req, args)
	}
}

func newFetchHandlerWithUsage(svc *tools.Service, cfg *config.Config, reporter *usage.Reporter, httpReq *http.Request) mcp.ToolHandlerFor[FetchArgs, FetchResult] {
	inner := newFetchHandler(svc, cfg, httpReq)
	return func(ctx context.Context, req *mcp.CallToolRequest, args FetchArgs) (*mcp.CallToolResult, FetchResult, error) {
		reporter.ReportSession(ctx, httpReq)
		return inner(ctx, req, args)
	}
}
