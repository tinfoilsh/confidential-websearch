package agent

import (
	"github.com/openai/openai-go/v3/responses"

	"github.com/tinfoilsh/confidential-websearch/search"
)

// SearchArgs represents the arguments for a web search
type SearchArgs struct {
	Query string `json:"query"`
}

// ToolCall represents a completed tool call with its results
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

// PendingSearch represents a search query to be executed by the pipeline
type PendingSearch struct {
	ID    string `json:"id"`
	Query string `json:"query"`
}

// ContextMessage represents a message in the conversation context for the agent
type ContextMessage struct {
	Role    string // "user", "assistant", or "system"
	Content string // The message content
}

// Result contains the agent's decision about what to search
type Result struct {
	PendingSearches []PendingSearch // Searches to execute (after PII filtering)
	BlockedQueries  []BlockedQuery  // Queries blocked by PII filter
	SearchReasoning string          // Agent's reasoning about search decisions
}

// ChunkCallback is called for each streaming event from the agent LLM
type ChunkCallback func(event responses.ResponseStreamEventUnion)

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
