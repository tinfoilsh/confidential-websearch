package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/responses"
	"github.com/openai/openai-go/v2/shared"
	log "github.com/sirupsen/logrus"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/search"
)

// SearchFilter is called with queries before search execution.
// Returns the filtered list of queries that should proceed.
type SearchFilter func(ctx context.Context, queries []string) []string

// Agent handles the tool-calling loop with the small model
type Agent struct {
	client       *tinfoil.Client
	model        string
	searcher     search.Provider
	filterMu     sync.RWMutex
	searchFilter SearchFilter
}

// New creates a new agent
func New(client *tinfoil.Client, model string, searcher search.Provider) *Agent {
	return &Agent{client: client, model: model, searcher: searcher}
}

// SetSearchFilter sets a filter to be applied before search execution.
// The filter receives all queries and returns only those that should proceed.
// This method is safe for concurrent use.
func (a *Agent) SetSearchFilter(filter SearchFilter) {
	a.filterMu.Lock()
	a.searchFilter = filter
	a.filterMu.Unlock()
}

// getSearchFilter returns the current search filter (thread-safe).
func (a *Agent) getSearchFilter() SearchFilter {
	a.filterMu.RLock()
	filter := a.searchFilter
	a.filterMu.RUnlock()
	return filter
}

// Run executes a single-shot tool call to gather search results (non-streaming)
func (a *Agent) Run(ctx context.Context, userQuery string) (*Result, error) {
	return a.runWithFilter(ctx, userQuery, nil, a.getSearchFilter())
}

// RunStreaming executes the agent with optional streaming support
func (a *Agent) RunStreaming(ctx context.Context, userQuery string, onChunk ChunkCallback) (*Result, error) {
	return a.runWithFilter(ctx, userQuery, onChunk, a.getSearchFilter())
}

// RunWithFilter executes the agent with a specific search filter (thread-safe for concurrent use)
func (a *Agent) RunWithFilter(ctx context.Context, userQuery string, filter SearchFilter) (*Result, error) {
	return a.runWithFilter(ctx, userQuery, nil, filter)
}

// runWithFilter is the internal implementation that accepts a filter parameter
func (a *Agent) runWithFilter(ctx context.Context, userQuery string, onChunk ChunkCallback, filter SearchFilter) (*Result, error) {
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
		Temperature:     openai.Float(config.AgentTemperature),
		MaxOutputTokens: openai.Int(config.AgentMaxTokens),
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
		stream := a.client.Responses.NewStreaming(ctx, params)

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
				// Arguments come in the Delta field for streaming events
				args := event.Delta
				if args == "" {
					args = event.Arguments
				}
				if functionCalls[idx] != nil {
					functionCalls[idx].arguments.WriteString(args)
				}
			}
		}
		if err := stream.Err(); err != nil {
			return nil, fmt.Errorf("agent streaming failed: %w", err)
		}
	} else {
		resp, err := a.client.Responses.New(ctx, params)
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

	// Collect queries from function calls
	type queryItem struct {
		id    string
		query string
	}
	var queries []queryItem

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

		queries = append(queries, queryItem{id: fc.id, query: query})
	}

	if len(queries) == 0 {
		log.Debug("No valid search queries")
		return result, nil
	}

	// Apply filter if set
	if filter != nil {
		queryStrings := make([]string, len(queries))
		for i, q := range queries {
			queryStrings[i] = q.query
		}

		allowed := filter(ctx, queryStrings)
		allowedSet := make(map[string]bool, len(allowed))
		for _, q := range allowed {
			allowedSet[q] = true
		}

		var filtered []queryItem
		for _, q := range queries {
			if allowedSet[q.query] {
				filtered = append(filtered, q)
			}
		}
		queries = filtered

		if len(queries) == 0 {
			log.Debug("All queries filtered out")
			return result, nil
		}
	}

	// Execute searches in parallel
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, q := range queries {
		wg.Add(1)
		go func(id, query string) {
			defer wg.Done()
			log.Infof("Searching: %s", query)

			searchResults, err := a.searcher.Search(ctx, query, config.DefaultMaxSearchResults)
			if err != nil {
				log.Errorf("Search failed: %v", err)
				return
			}

			mu.Lock()
			result.ToolCalls = append(result.ToolCalls, ToolCall{ID: id, Query: query, Results: searchResults})
			mu.Unlock()
		}(q.id, q.query)
	}
	wg.Wait()

	return result, nil
}
