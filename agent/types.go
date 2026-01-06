package agent

import (
	"github.com/openai/openai-go/v2/responses"

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

// Result contains the search results gathered by the agent
type Result struct {
	ToolCalls      []ToolCall
	AgentReasoning string
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
