package agent

import (
	"context"

	"github.com/openai/openai-go/v3/responses"

	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/search"
)

// URLFetcher fetches URL contents
type URLFetcher interface {
	FetchURLs(ctx context.Context, urls []string) []fetch.FetchedPage
}

// SearchProvider executes web searches
type SearchProvider interface {
	Search(ctx context.Context, query string, opts search.Options) ([]search.Result, error)
}

// SearchArgs represents the arguments for a web search
type SearchArgs struct {
	Query string `json:"query"`
}

// ToolCall represents a completed search tool call with its results
type ToolCall struct {
	ID      string          `json:"id"`
	Query   string          `json:"query"`
	Results []search.Result `json:"results"`
}

// BlockedQuery represents a search query that was blocked by a filter
type BlockedQuery struct {
	ID     string `json:"id"`
	Query  string `json:"query"`
	Reason string `json:"reason"`
}

// FetchArgs represents the arguments for a URL fetch
type FetchArgs struct {
	URL string `json:"url"`
}

// ContextMessage represents a message in the conversation context for the agent
type ContextMessage struct {
	Role    string // "user", "assistant", or "system"
	Content string // The message content
}

// Result contains executed tool results from the agent loop
type Result struct {
	SearchResults   []ToolCall          // Executed search results
	FetchedPages    []fetch.FetchedPage // Pages fetched during the agent loop
	BlockedQueries  []BlockedQuery      // Queries blocked by PII filter
	SearchReasoning string              // Agent's reasoning about decisions
}

// ChunkCallback is called for each streaming event from the agent LLM
type ChunkCallback func(event responses.ResponseStreamEventUnion)

// ToolEventCallback is called when a tool execution starts or completes.
// toolType is "search" or "fetch", id is the tool call ID, status is
// "in_progress"/"completed"/"failed"/"blocked", detail is the query or URL.
type ToolEventCallback func(toolType, id, status, detail string)

// SearchToolParams is the JSON schema for the search tool
var SearchToolParams = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"query": map[string]any{
			"type":        "string",
			"description": "The search query",
		},
	},
	"required": []string{"query"},
}

// FetchToolParams is the JSON schema for the fetch tool
var FetchToolParams = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"url": map[string]any{
			"type":        "string",
			"description": "The URL to fetch",
		},
	},
	"required": []string{"url"},
}
