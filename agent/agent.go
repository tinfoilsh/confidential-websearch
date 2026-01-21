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

// FilterResult contains the outcome of filtering search queries
type FilterResult struct {
	Allowed []string
	Blocked []BlockedQuery
}

// SearchFilter is called with queries before search execution.
// Returns the filter result containing allowed and blocked queries.
type SearchFilter func(ctx context.Context, queries []string) FilterResult

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

// RunStreamingWithFilter executes the agent with streaming and a custom filter
func (a *Agent) RunStreamingWithFilter(ctx context.Context, userQuery string, onChunk ChunkCallback, filter SearchFilter) (*Result, error) {
	return a.runWithFilter(ctx, userQuery, onChunk, filter)
}

// RunWithContext executes the agent with full conversation context, reasoning items, and system prompt.
// This method builds proper Responses API input from the conversation history.
func (a *Agent) RunWithContext(ctx context.Context, messages []ContextMessage, systemPrompt string, onChunk ChunkCallback) (*Result, error) {
	return a.runWithContext(ctx, messages, systemPrompt, onChunk, a.getSearchFilter())
}

// runWithContext is the internal implementation for context-aware agent execution
func (a *Agent) runWithContext(ctx context.Context, messages []ContextMessage, systemPrompt string, onChunk ChunkCallback, filter SearchFilter) (*Result, error) {
	searchTool := responses.ToolParamOfFunction(
		"search",
		SearchToolParams,
		false,
	)
	searchTool.OfFunction.Description = openai.String("Search the web for current information.")

	// Build Responses API input from conversation context
	var input []responses.ResponseInputItemUnionParam

	// Add conversation history
	for _, msg := range messages {
		switch msg.Role {
		case "user":
			input = append(input, responses.ResponseInputItemParamOfMessage(msg.Content, responses.EasyInputMessageRoleUser))
		case "assistant":
			input = append(input, responses.ResponseInputItemParamOfMessage(msg.Content, responses.EasyInputMessageRoleAssistant))
			// Add reasoning items from previous turn as item references
			for _, ri := range msg.ReasoningItems {
				input = append(input, responses.ResponseInputItemParamOfItemReference(ri.ID))
			}
		}
	}

	params := responses.ResponseNewParams{
		Model:           shared.ResponsesModel(a.model),
		Input:           responses.ResponseNewParamsInputUnion{OfInputItemList: input},
		Tools:           []responses.ToolUnionParam{searchTool},
		Temperature:     openai.Float(config.AgentTemperature),
		MaxOutputTokens: openai.Int(config.AgentMaxTokens),
	}

	// Use system prompt as instructions (client's system prompt)
	if systemPrompt != "" {
		params.Instructions = openai.String(systemPrompt)
	}

	// Track function calls by output index
	type functionCall struct {
		id        string
		name      string
		arguments strings.Builder
	}
	functionCalls := make(map[int]*functionCall)
	var reasoningBuilder strings.Builder
	var reasoningItems []ReasoningItem

	if onChunk != nil {
		// Create cancellable context for early abort
		streamCtx, cancelStream := context.WithCancel(ctx)
		defer cancelStream()

		stream := a.client.Responses.NewStreaming(streamCtx, params)

		var currentReasoningItem *ReasoningItem
		aborted := false

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
				} else if event.Item.Type == "reasoning" {
					ri := event.Item.AsReasoning()
					currentReasoningItem = &ReasoningItem{
						ID:   ri.ID,
						Type: "reasoning",
					}
				} else if event.Item.Type == "message" {
					// Agent is generating content instead of tool calls - abort early
					log.Debug("Agent starting content generation, aborting (no search needed)")
					cancelStream()
					aborted = true
				}

			case "response.output_item.done":
				if event.Item.Type == "reasoning" && currentReasoningItem != nil {
					ri := event.Item.AsReasoning()
					for _, s := range ri.Summary {
						if s.Type == "summary_text" {
							currentReasoningItem.Summary = append(currentReasoningItem.Summary, ReasoningSummaryPart{
								Type: "summary_text",
								Text: s.Text,
							})
						}
					}
					reasoningItems = append(reasoningItems, *currentReasoningItem)
					currentReasoningItem = nil
				}

			case "response.function_call_arguments.delta":
				idx := int(event.OutputIndex)
				args := event.Delta
				if args == "" {
					args = event.Arguments
				}
				if functionCalls[idx] != nil {
					functionCalls[idx].arguments.WriteString(args)
				}
			}

			if aborted {
				break
			}
		}
		// Only check for errors if we didn't abort intentionally
		if !aborted {
			if err := stream.Err(); err != nil {
				return nil, fmt.Errorf("agent streaming failed: %w", err)
			}
		}
	} else {
		resp, err := a.client.Responses.New(ctx, params)
		if err != nil {
			return nil, fmt.Errorf("agent LLM call failed: %w", err)
		}

		for i, item := range resp.Output {
			switch item.Type {
			case "reasoning":
				ri := item.AsReasoning()
				reasoningItem := ReasoningItem{
					ID:   ri.ID,
					Type: "reasoning",
				}
				for _, part := range ri.Summary {
					if part.Type == "summary_text" {
						reasoningBuilder.WriteString(part.Text)
						reasoningItem.Summary = append(reasoningItem.Summary, ReasoningSummaryPart{
							Type: "summary_text",
							Text: part.Text,
						})
					}
				}
				reasoningItems = append(reasoningItems, reasoningItem)
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

	result := &Result{
		AgentReasoning: reasoningBuilder.String(),
		ReasoningItems: reasoningItems,
	}

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

		filterResult := filter(ctx, queryStrings)

		allowedSet := make(map[string]bool, len(filterResult.Allowed))
		for _, q := range filterResult.Allowed {
			allowedSet[q] = true
		}

		blockedReasons := make(map[string]string)
		for _, b := range filterResult.Blocked {
			blockedReasons[b.Query] = b.Reason
		}

		var filtered []queryItem
		for _, q := range queries {
			if allowedSet[q.query] {
				filtered = append(filtered, q)
			} else if reason, blocked := blockedReasons[q.query]; blocked {
				result.BlockedQueries = append(result.BlockedQueries, BlockedQuery{
					ID:     q.id,
					Query:  q.query,
					Reason: reason,
				})
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

		filterResult := filter(ctx, queryStrings)

		// Build allowed set for O(1) lookup
		allowedSet := make(map[string]bool, len(filterResult.Allowed))
		for _, q := range filterResult.Allowed {
			allowedSet[q] = true
		}

		// Map blocked reasons by query string
		blockedReasons := make(map[string]string)
		for _, b := range filterResult.Blocked {
			blockedReasons[b.Query] = b.Reason
		}

		// Separate allowed and blocked queries
		var filtered []queryItem
		for _, q := range queries {
			if allowedSet[q.query] {
				filtered = append(filtered, q)
			} else if reason, blocked := blockedReasons[q.query]; blocked {
				result.BlockedQueries = append(result.BlockedQueries, BlockedQuery{
					ID:     q.id,
					Query:  q.query,
					Reason: reason,
				})
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
