package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/responses"
	"github.com/openai/openai-go/v2/shared"
	log "github.com/sirupsen/logrus"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/search"
)

const (
	agentTemperature  = 0.3
	agentMaxTokens    = 1024
	defaultMaxResults = 5
)

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
	searchTool := responses.ToolParamOfFunction(
		"search",
		SearchToolParams,
		false,
	)
	searchTool.OfFunction.Description = openai.String("Search the web for current information.")

	params := responses.ResponseNewParams{
		Model:           shared.ResponsesModel(a.model),
		Instructions:    openai.String(`You are a research assistant. Use the search tool for current information or facts you're uncertain about. You can call search multiple times in parallel for complex queries.`),
		Input:           responses.ResponseNewParamsInputUnion{OfString: openai.String(userQuery)},
		Tools:           []responses.ToolUnionParam{searchTool},
		Temperature:     openai.Float(agentTemperature),
		MaxOutputTokens: openai.Int(agentMaxTokens),
	}

	// Track function calls by output index
	type functionCall struct {
		id        string
		name      string
		arguments strings.Builder
	}
	functionCalls := make(map[int]*functionCall)
	var reasoningBuilder strings.Builder

	if onChunk != nil {
		stream := a.client.Responses.NewStreaming(ctx, params, reqOpts...)

		for stream.Next() {
			event := stream.Current()
			onChunk(event)

			switch event.Type {
			case "response.reasoning_text.delta":
				reasoningBuilder.WriteString(event.Delta)

			case "response.output_item.added":
				if event.Item.Type == "function_call" {
					fc := event.Item.AsFunctionCall()
					functionCalls[int(event.OutputIndex)] = &functionCall{
						id:   fc.CallID,
						name: fc.Name,
					}
				}

			case "response.function_call_arguments.delta":
				idx := int(event.OutputIndex)
				if functionCalls[idx] != nil {
					functionCalls[idx].arguments.WriteString(event.Arguments)
				}
			}
		}
		if err := stream.Err(); err != nil {
			return nil, fmt.Errorf("agent streaming failed: %w", err)
		}
	} else {
		resp, err := a.client.Responses.New(ctx, params, reqOpts...)
		if err != nil {
			return nil, fmt.Errorf("agent LLM call failed: %w", err)
		}

		for i, item := range resp.Output {
			switch item.Type {
			case "reasoning":
				for _, part := range item.AsReasoning().Summary {
					if part.Type == "summary_text" {
						reasoningBuilder.WriteString(part.Text)
					}
				}
			case "function_call":
				fc := item.AsFunctionCall()
				functionCalls[i] = &functionCall{
					id:   fc.CallID,
					name: fc.Name,
				}
				functionCalls[i].arguments.WriteString(fc.Arguments)
			}
		}
	}

	result := &Result{AgentReasoning: reasoningBuilder.String()}

	if len(functionCalls) == 0 {
		log.Debug("Agent decided no search needed")
		return result, nil
	}

	log.Debugf("Agent requested %d search(es)", len(functionCalls))

	// Execute searches in parallel
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, fc := range functionCalls {
		if fc.name != "search" {
			continue
		}

		var args SearchArgs
		if err := json.Unmarshal([]byte(fc.arguments.String()), &args); err != nil {
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

			searchResults, err := a.searcher.Search(ctx, q, defaultMaxResults)
			if err != nil {
				log.Errorf("Search failed: %v", err)
				return
			}

			mu.Lock()
			result.ToolCalls = append(result.ToolCalls, ToolCall{ID: id, Query: q, Results: searchResults})
			mu.Unlock()
		}(fc.id, query)
	}
	wg.Wait()

	return result, nil
}
