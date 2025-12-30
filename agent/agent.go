package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/shared"
	log "github.com/sirupsen/logrus"

	"github.com/tinfoilsh/confidential-websearch/search"
)

// Agent handles the tool-calling loop with the small model
type Agent struct {
	client   *openai.Client
	model    string
	searcher search.Provider
}

// ToolCall represents a tool call made by the agent
type ToolCall struct {
	ID        string `json:"id"`
	Query     string `json:"query"`
	ResultIdx int    `json:"result_idx"` // Index into SearchResults for this call's results
}

// Result contains the search results gathered by the agent
type Result struct {
	SearchResults []search.Result
	ToolCalls     []ToolCall // Track individual tool calls for proper message construction
}

// New creates a new agent
func New(client *openai.Client, model string, searcher search.Provider) *Agent {
	return &Agent{
		client:   client,
		model:    model,
		searcher: searcher,
	}
}

// Run executes the agent loop to gather search results
// reqOpts are forwarded to the LLM calls (e.g., for API key forwarding)
func (a *Agent) Run(ctx context.Context, userQuery string, reqOpts ...option.RequestOption) (*Result, error) {
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(`You are a research assistant. Your job is to decide if you need to search the web to answer the user's question.

If the question requires current information, recent events, or facts you're uncertain about, use the web_search tool.
If the question is about general knowledge, logic, or creative tasks, you can answer directly without searching.

Be concise in your search queries. You can make multiple searches if needed.`),
		openai.UserMessage(userQuery),
	}

	result := &Result{}

	// Define the web_search tool using the helper function
	webSearchTool := openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
		Name:        "web_search",
		Description: openai.String("Search the web for current information. Use this when you need up-to-date information or facts you're uncertain about."),
		Parameters:  WebSearchToolParams,
	})

	// Max 5 iterations to prevent infinite loops
	for i := 0; i < 5; i++ {
		log.Debugf("Agent iteration %d", i+1)

		resp, err := a.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:       shared.ChatModel(a.model),
			Messages:    messages,
			Tools:       []openai.ChatCompletionToolUnionParam{webSearchTool},
			Temperature: openai.Float(0.3),
			MaxTokens:   openai.Int(1024),
		}, reqOpts...)
		if err != nil {
			return nil, fmt.Errorf("agent LLM call failed: %w", err)
		}

		if len(resp.Choices) == 0 {
			return nil, fmt.Errorf("agent LLM returned no choices")
		}

		choice := resp.Choices[0]

		// No tool calls - agent is done deciding
		if choice.FinishReason != "tool_calls" || len(choice.Message.ToolCalls) == 0 {
			log.Debugf("Agent finished without tool calls (finish_reason: %s)", choice.FinishReason)
			return result, nil
		}

		// Add assistant message with tool calls (convert response to param)
		messages = append(messages, choice.Message.ToParam())

		// Execute searches in parallel
		var wg sync.WaitGroup
		var mu sync.Mutex
		toolResults := make([]openai.ChatCompletionMessageParamUnion, 0)

		for _, tc := range choice.Message.ToolCalls {
			tcUnion := tc.AsAny()
			if tcUnion == nil {
				continue
			}

			funcCall, ok := tcUnion.(openai.ChatCompletionMessageFunctionToolCall)
			if !ok || funcCall.Function.Name != "web_search" {
				log.Warnf("Unknown or non-function tool call")
				continue
			}

			wg.Add(1)
			go func(toolCallID string, arguments string) {
				defer wg.Done()

				var args SearchArgs
				if err := json.Unmarshal([]byte(arguments), &args); err != nil {
					log.Errorf("Failed to parse tool arguments: %v", err)
					return
				}

				log.Infof("Searching: %s", args.Query)
				searchResults, err := a.searcher.Search(ctx, args.Query, 5)

				mu.Lock()
				defer mu.Unlock()

				var content string
				if err != nil {
					log.Errorf("Search failed: %v", err)
					content = fmt.Sprintf(`{"error": "%s"}`, err.Error())
				} else {
					// Track this tool call
					result.ToolCalls = append(result.ToolCalls, ToolCall{
						ID:        toolCallID,
						Query:     args.Query,
						ResultIdx: len(result.SearchResults),
					})
					result.SearchResults = append(result.SearchResults, searchResults...)
					b, _ := json.Marshal(searchResults)
					content = string(b)
				}

				toolResults = append(toolResults, openai.ToolMessage(content, toolCallID))
			}(funcCall.ID, funcCall.Function.Arguments)
		}
		wg.Wait()

		// Add tool results to messages
		messages = append(messages, toolResults...)
	}

	log.Warn("Agent reached max iterations")
	return result, nil
}
