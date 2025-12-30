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
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/search"
)

// extractReasoningContent extracts reasoning_content or content from raw JSON response.
func extractReasoningContent(rawJSON string) string {
	if rawJSON == "" {
		return ""
	}
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(rawJSON), &data); err != nil {
		return ""
	}
	if rc, ok := data["reasoning_content"].(string); ok && rc != "" {
		return rc
	}
	if content, ok := data["content"].(string); ok {
		return content
	}
	return ""
}

// Agent handles the tool-calling loop with the small model
type Agent struct {
	client   *tinfoil.Client
	model    string
	searcher search.Provider
}

// New creates a new agent
func New(client *tinfoil.Client, model string, searcher search.Provider) *Agent {
	return &Agent{client: client, model: model, searcher: searcher}
}

// Run executes a single-shot tool call to gather search results (non-streaming)
func (a *Agent) Run(ctx context.Context, userQuery string, reqOpts ...option.RequestOption) (*Result, error) {
	return a.RunStreaming(ctx, userQuery, nil, reqOpts...)
}

// RunStreaming executes the agent with optional streaming support
func (a *Agent) RunStreaming(ctx context.Context, userQuery string, onChunk ChunkCallback, reqOpts ...option.RequestOption) (*Result, error) {
	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(`You are a research assistant. Use the search tool for current information or facts you're uncertain about. You can call search multiple times in parallel for complex queries.`),
		openai.UserMessage(userQuery),
	}

	searchTool := openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
		Name:        "search",
		Description: openai.String("Search the web for current information."),
		Parameters:  SearchToolParams,
	})

	params := openai.ChatCompletionNewParams{
		Model:       shared.ChatModel(a.model),
		Messages:    messages,
		Tools:       []openai.ChatCompletionToolUnionParam{searchTool},
		Temperature: openai.Float(0.3),
		MaxTokens:   openai.Int(1024),
	}

	var toolCalls []openai.ChatCompletionMessageToolCallUnion
	var reasoning string

	if onChunk != nil {
		stream := a.client.Chat.Completions.NewStreaming(ctx, params, reqOpts...)
		toolCallBuilders := make(map[int]*toolCallBuilder)
		var reasoningBuilder strings.Builder

		for stream.Next() {
			chunk := stream.Current()
			onChunk(chunk)

			if len(chunk.Choices) > 0 {
				delta := chunk.Choices[0].Delta
				if delta.Content != "" {
					reasoningBuilder.WriteString(delta.Content)
				}
				if rc := extractReasoningContent(delta.RawJSON()); rc != "" {
					reasoningBuilder.WriteString(rc)
				}
				for _, tcDelta := range delta.ToolCalls {
					idx := int(tcDelta.Index)
					if toolCallBuilders[idx] == nil {
						toolCallBuilders[idx] = &toolCallBuilder{}
					}
					b := toolCallBuilders[idx]
					if tcDelta.ID != "" {
						b.id = tcDelta.ID
					}
					if tcDelta.Function.Name != "" {
						b.name = tcDelta.Function.Name
					}
					if tcDelta.Function.Arguments != "" {
						b.arguments.WriteString(tcDelta.Function.Arguments)
					}
				}
			}
		}
		if err := stream.Err(); err != nil {
			return nil, fmt.Errorf("agent streaming failed: %w", err)
		}

		reasoning = reasoningBuilder.String()
		for _, b := range toolCallBuilders {
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnion{
				ID:       b.id,
				Function: openai.ChatCompletionMessageFunctionToolCallFunction{Name: b.name, Arguments: b.arguments.String()},
			})
		}
	} else {
		resp, err := a.client.Chat.Completions.New(ctx, params, reqOpts...)
		if err != nil {
			return nil, fmt.Errorf("agent LLM call failed: %w", err)
		}
		if len(resp.Choices) == 0 {
			return nil, fmt.Errorf("agent LLM returned no choices")
		}
		toolCalls = resp.Choices[0].Message.ToolCalls
		reasoning = extractReasoningContent(resp.Choices[0].Message.RawJSON())
	}

	result := &Result{AgentReasoning: reasoning}

	if len(toolCalls) == 0 {
		log.Debug("Agent decided no search needed")
		return result, nil
	}

	log.Debugf("Agent requested %d search(es)", len(toolCalls))

	// Execute searches in parallel
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, tc := range toolCalls {
		if tc.Function.Name != "search" {
			continue
		}

		var args SearchArgs
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			log.Errorf("Failed to parse search arguments: %v", err)
			continue
		}

		query := strings.TrimSpace(args.Query)
		if query == "" {
			continue
		}

		wg.Add(1)
		go func(id, q string) {
			defer wg.Done()
			log.Infof("Searching: %s", q)

			searchResults, err := a.searcher.Search(ctx, q, 5)
			if err != nil {
				log.Errorf("Search failed: %v", err)
				return
			}

			mu.Lock()
			result.ToolCalls = append(result.ToolCalls, ToolCall{ID: id, Query: q, Results: searchResults})
			mu.Unlock()
		}(tc.ID, query)
	}
	wg.Wait()

	return result, nil
}
