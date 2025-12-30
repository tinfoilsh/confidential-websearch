package agent

import "github.com/openai/openai-go/v2/shared"

// SearchArgs represents the arguments for a web search
type SearchArgs struct {
	Query string `json:"query"`
}

// WebSearchToolParams is the JSON schema for the web_search tool
var WebSearchToolParams = shared.FunctionParameters{
	"type": "object",
	"properties": map[string]interface{}{
		"query": map[string]interface{}{
			"type":        "string",
			"description": "The search query",
		},
	},
	"required": []string{"query"},
}
