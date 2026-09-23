package server

import (
	"context"
	"fmt"
	"net/http"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	log "github.com/sirupsen/logrus"

	"github.com/tinfoilsh/confidential-websearch/internal/config"
	"github.com/tinfoilsh/confidential-websearch/internal/tools"
	"github.com/tinfoilsh/confidential-websearch/internal/usage"
)

// NewMCPServer builds an MCP server exposing the search and fetch tools for a
// single HTTP request. The request is captured so per-request safety headers
// and usage context can be read by the tool handlers.
func NewMCPServer(svc *tools.Service, cfg *config.Config, descriptions config.ToolDescriptions, reporter *usage.Reporter, version string, request *http.Request) *mcp.Server {
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
		Description: descriptions.Search,
		InputSchema: searchSchema,
	}, instrumentTool("search", newSearchHandlerWithUsage(svc, cfg, reporter, request)))

	mcp.AddTool(server, &mcp.Tool{
		Name:        "fetch",
		Description: descriptions.Fetch,
		InputSchema: fetchSchema,
	}, instrumentTool("fetch", newFetchHandlerWithUsage(svc, cfg, reporter, request)))

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
		if err := reporter.ReportSession(ctx, httpReq); err != nil {
			return nil, SearchResult{}, err
		}
		callResult, result, err := inner(ctx, req, args)
		if err == nil && result.PIIChecked {
			if reportErr := reporter.ReportPIICheck(ctx, httpReq); reportErr != nil {
				log.WithError(reportErr).Error("failed to report privacy filter usage")
			}
		}
		return callResult, result, err
	}
}

func newFetchHandlerWithUsage(svc *tools.Service, cfg *config.Config, reporter *usage.Reporter, httpReq *http.Request) mcp.ToolHandlerFor[FetchArgs, FetchResult] {
	inner := newFetchHandler(svc, cfg, httpReq)
	return func(ctx context.Context, req *mcp.CallToolRequest, args FetchArgs) (*mcp.CallToolResult, FetchResult, error) {
		if err := reporter.ReportSession(ctx, httpReq); err != nil {
			return nil, FetchResult{}, err
		}
		return inner(ctx, req, args)
	}
}
