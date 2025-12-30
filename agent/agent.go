package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
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
	SearchResults  []search.Result
	ToolCalls      []ToolCall // Track individual tool calls for proper message construction
	AgentReasoning string     // Reasoning content from the agent (e.g., reasoning_content field)
}

// New creates a new agent
func New(client *openai.Client, model string, searcher search.Provider) *Agent {
	return &Agent{
		client:   client,
		model:    model,
		searcher: searcher,
	}
}

// Run executes a single-shot tool call to gather search results (non-streaming)
func (a *Agent) Run(ctx context.Context, userQuery string, reqOpts ...option.RequestOption) (*Result, error) {
	return a.RunStreaming(ctx, userQuery, nil, nil, reqOpts...)
}

// RunStreaming executes the agent with streaming support
// - onChunk: called for each streaming chunk from the agent LLM (can be nil)
// - onStatus: called for status updates like search progress (can be nil)
func (a *Agent) RunStreaming(ctx context.Context, userQuery string, onChunk ChunkCallback, onStatus StatusCallback, reqOpts ...option.RequestOption) (*Result, error) {
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(`You are a research assistant. Decide if you need to search the web to answer the user's question.

If the question requires current information, recent events, or facts you're uncertain about, use the web_search tool.
If the question is about general knowledge, logic, or creative tasks, you can answer directly without searching.

Be concise in your search queries. You can call web_search multiple times in parallel if needed.`),
		openai.UserMessage(userQuery),
	}

	result := &Result{}

	// Define the web_search tool
	webSearchTool := openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
		Name:        "web_search",
		Description: openai.String("Search the web for current information. Use this when you need up-to-date information or facts you're uncertain about."),
		Parameters:  WebSearchToolParams,
	})

	params := openai.ChatCompletionNewParams{
		Model:       shared.ChatModel(a.model),
		Messages:    messages,
		Tools:       []openai.ChatCompletionToolUnionParam{webSearchTool},
		Temperature: openai.Float(0.3),
		MaxTokens:   openai.Int(1024),
	}

	// Use streaming if callback provided, otherwise use regular call
	var toolCalls []openai.ChatCompletionMessageToolCallUnion
	var reasoning string

	if onChunk != nil {
		// Streaming mode - forward chunks and accumulate tool calls
		stream := a.client.Chat.Completions.NewStreaming(ctx, params, reqOpts...)

		// Track tool call deltas by index
		toolCallBuilders := make(map[int]*struct {
			id        string
			name      string
			arguments strings.Builder
		})

		var reasoningBuilder strings.Builder

		for stream.Next() {
			chunk := stream.Current()

			// Forward the chunk
			onChunk(chunk)

			// Accumulate from the chunk
			if len(chunk.Choices) > 0 {
				delta := chunk.Choices[0].Delta

				// Accumulate content (standard reasoning)
				if delta.Content != "" {
					reasoningBuilder.WriteString(delta.Content)
				}

				// Also check for reasoning_content in raw JSON (gpt-oss style)
				if rawJSON := delta.RawJSON(); rawJSON != "" {
					var deltaData map[string]interface{}
					if err := json.Unmarshal([]byte(rawJSON), &deltaData); err == nil {
						if rc, ok := deltaData["reasoning_content"].(string); ok && rc != "" {
							reasoningBuilder.WriteString(rc)
						}
					}
				}

				// Accumulate tool calls
				for _, tcDelta := range delta.ToolCalls {
					idx := int(tcDelta.Index)
					if _, exists := toolCallBuilders[idx]; !exists {
						toolCallBuilders[idx] = &struct {
							id        string
							name      string
							arguments strings.Builder
						}{}
					}
					builder := toolCallBuilders[idx]

					if tcDelta.ID != "" {
						builder.id = tcDelta.ID
					}
					if tcDelta.Function.Name != "" {
						builder.name = tcDelta.Function.Name
					}
					if tcDelta.Function.Arguments != "" {
						builder.arguments.WriteString(tcDelta.Function.Arguments)
					}
				}
			}
		}
		if err := stream.Err(); err != nil {
			return nil, fmt.Errorf("agent streaming failed: %w", err)
		}

		reasoning = reasoningBuilder.String()

		// Convert builders to tool calls
		for _, builder := range toolCallBuilders {
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnion{
				ID: builder.id,
				Function: openai.ChatCompletionMessageFunctionToolCallFunction{
					Name:      builder.name,
					Arguments: builder.arguments.String(),
				},
			})
		}
	} else {
		// Non-streaming mode
		resp, err := a.client.Chat.Completions.New(ctx, params, reqOpts...)
		if err != nil {
			return nil, fmt.Errorf("agent LLM call failed: %w", err)
		}

		if len(resp.Choices) == 0 {
			return nil, fmt.Errorf("agent LLM returned no choices")
		}

		choice := resp.Choices[0]
		toolCalls = choice.Message.ToolCalls

		// Extract reasoning_content from raw JSON (used by some models like gpt-oss)
		if rawJSON := choice.Message.RawJSON(); rawJSON != "" {
			var msgData map[string]interface{}
			if err := json.Unmarshal([]byte(rawJSON), &msgData); err == nil {
				if rc, ok := msgData["reasoning_content"].(string); ok && rc != "" {
					reasoning = rc
				} else if content, ok := msgData["content"].(string); ok && content != "" {
					reasoning = content
				}
			}
		}
	}

	result.AgentReasoning = reasoning
	if reasoning != "" {
		log.Debugf("Agent reasoning: %s", reasoning)
	}

	// No tool calls - model decided no search is needed
	if len(toolCalls) == 0 {
		log.Debug("Agent decided no search needed")
		return result, nil
	}

	log.Debugf("Agent requested %d search(es)", len(toolCalls))

	// Helper to emit status
	emitStatus := func(status string) {
		if onStatus != nil {
			onStatus(status)
		}
	}

	// Execute all searches in parallel
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, tc := range toolCalls {
		toolCallID := tc.ID
		functionName := tc.Function.Name
		functionArgs := tc.Function.Arguments

		// Handle variations like "web_search" or "web_search.exec"
		if !strings.HasPrefix(functionName, "web_search") {
			log.Warnf("Unknown tool: %s", functionName)
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

			query := strings.TrimSpace(args.Query)
			if query == "" {
				log.Warn("Skipping empty search query")
				return
			}

			emitStatus(fmt.Sprintf("🔍 Searching: %s", query))
			log.Infof("Searching: %s", query)

			searchResults, err := a.searcher.Search(ctx, query, 5)
			if err != nil {
				log.Errorf("Search failed: %v", err)
				emitStatus(fmt.Sprintf("❌ Search failed: %s", query))
				return
			}

			emitStatus(fmt.Sprintf("✓ Found %d results for: %s", len(searchResults), query))

			mu.Lock()
			defer mu.Unlock()

			result.ToolCalls = append(result.ToolCalls, ToolCall{
				ID:        toolCallID,
				Query:     query,
				ResultIdx: len(result.SearchResults),
			})
			result.SearchResults = append(result.SearchResults, searchResults...)
		}(toolCallID, functionArgs)
	}
	wg.Wait()

	return result, nil
}
