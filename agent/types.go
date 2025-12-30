package agent

import (
	"strings"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/shared"

	"github.com/tinfoilsh/confidential-websearch/search"
)

// SearchArgs represents the arguments for a web search
type SearchArgs struct {
	Query string `json:"query"`
}

// toolCallBuilder accumulates streaming tool call deltas
type toolCallBuilder struct {
	id        string
	name      string
	arguments strings.Builder
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

// ChunkCallback is called for each streaming chunk from the agent LLM
type ChunkCallback func(chunk openai.ChatCompletionChunk)

// SearchToolParams is the JSON schema for the search tool
var SearchToolParams = shared.FunctionParameters{
	"type": "object",
	"properties": map[string]interface{}{
		"query": map[string]interface{}{
			"type":        "string",
			"description": "The search query",
		},
	},
	"required": []string{"query"},
}
