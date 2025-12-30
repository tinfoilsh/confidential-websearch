package agent

import (
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/shared"
)

// SearchArgs represents the arguments for a web search
type SearchArgs struct {
	Query string `json:"query"`
}

// ChunkCallback is called for each streaming chunk from the agent LLM
type ChunkCallback func(chunk openai.ChatCompletionChunk)

// StatusCallback is called for status updates (search progress, etc.)
type StatusCallback func(status string)

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
