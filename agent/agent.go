package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	log "github.com/sirupsen/logrus"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/config"
)

const agentInstructions = `You are a search routing agent. Your task is to decide if a web search would help answer the user's request.

Focus primarily on the most recent user message when deciding what to search for. Earlier messages provide context, but the latest message should drive your search decisions.

If you're unsure whether a search would help, lean towards searching - it's better to have information available than to miss something useful.

If a search IS needed: Call the search tool with an appropriate query.
If NO search is needed: Do not call any tools and do not output any text.`

// FilterResult contains the outcome of filtering search queries
type FilterResult struct {
	Allowed []string
	Blocked []BlockedQuery
}

// SearchFilter is called with queries before search execution.
// Returns the filter result containing allowed and blocked queries.
type SearchFilter func(ctx context.Context, queries []string) FilterResult

// Agent handles the tool-calling loop with the LLM
type Agent struct {
	client *tinfoil.Client
	model  string
}

// New creates a new agent
func New(client *tinfoil.Client, model string) *Agent {
	return &Agent{client: client, model: model}
}

// functionCall tracks a single function call being built from stream events
type functionCall struct {
	id        string
	name      string
	arguments strings.Builder
}

// streamParser handles parsing of streaming response events
type streamParser struct {
	functionCalls map[int]*functionCall
	reasoning     strings.Builder
	aborted       bool
	cancel        context.CancelFunc
}

func newStreamParser(cancel context.CancelFunc) *streamParser {
	return &streamParser{
		functionCalls: make(map[int]*functionCall),
		cancel:        cancel,
	}
}

func (p *streamParser) handleEvent(event responses.ResponseStreamEventUnion) {
	switch event.Type {
	case "response.reasoning_text.delta":
		p.reasoning.WriteString(event.Delta)

	case "response.output_item.added":
		if event.Item.Type == "function_call" {
			fc := event.Item.AsFunctionCall()
			p.functionCalls[int(event.OutputIndex)] = &functionCall{
				id:   fc.CallID,
				name: fc.Name,
			}
		} else if event.Item.Type == "message" {
			log.Debug("Agent starting content generation, aborting (no search needed)")
			p.cancel()
			p.aborted = true
		}

	case "response.function_call_arguments.delta":
		idx := int(event.OutputIndex)
		args := event.Delta
		if args == "" {
			args = event.Arguments
		}
		if p.functionCalls[idx] != nil {
			p.functionCalls[idx].arguments.WriteString(args)
		}
	}
}

func (p *streamParser) parseResponse(resp *responses.Response) {
	for i, item := range resp.Output {
		switch item.Type {
		case "reasoning":
			ri := item.AsReasoning()
			for _, part := range ri.Summary {
				if part.Type == "summary_text" {
					p.reasoning.WriteString(part.Text)
				}
			}
		case "function_call":
			fc := item.AsFunctionCall()
			p.functionCalls[i] = &functionCall{
				id:   fc.CallID,
				name: fc.Name,
			}
			p.functionCalls[i].arguments.WriteString(fc.Arguments)
		}
	}
}

// RunWithContext executes the agent with conversation context and optional streaming.
// This is the main entry point for agent execution and implements AgentRunner interface.
func (a *Agent) RunWithContext(ctx context.Context, messages []ContextMessage, systemPrompt string, onChunk ChunkCallback) (*Result, error) {
	return a.run(ctx, messages, systemPrompt, onChunk, nil)
}

// RunWithFilter executes the agent with a custom search filter.
// Used by SafeAgent to inject PII filtering.
func (a *Agent) RunWithFilter(ctx context.Context, messages []ContextMessage, systemPrompt string, onChunk ChunkCallback, filter SearchFilter) (*Result, error) {
	return a.run(ctx, messages, systemPrompt, onChunk, filter)
}

// run is the internal implementation
func (a *Agent) run(ctx context.Context, messages []ContextMessage, systemPrompt string, onChunk ChunkCallback, filter SearchFilter) (*Result, error) {
	searchTool := responses.ToolParamOfFunction(
		"search",
		SearchToolParams,
		false,
	)
	searchTool.OfFunction.Description = openai.String("Search the web for current information.")

	// Build Responses API input from conversation context
	var input []responses.ResponseInputItemUnionParam

	// Limit to last N turns to avoid draining context in long conversations
	maxMessages := config.AgentMaxTurns * 2
	startIdx := 0
	if len(messages) > maxMessages {
		startIdx = len(messages) - maxMessages
	}
	recentMessages := messages[startIdx:]

	// Add conversation history
	for _, msg := range recentMessages {
		switch msg.Role {
		case "user":
			input = append(input, responses.ResponseInputItemParamOfMessage(msg.Content, responses.EasyInputMessageRoleUser))
		case "assistant":
			input = append(input, responses.ResponseInputItemParamOfMessage(msg.Content, responses.EasyInputMessageRoleAssistant))
		}
	}

	params := responses.ResponseNewParams{
		Model:           shared.ResponsesModel(a.model),
		Input:           responses.ResponseNewParamsInputUnion{OfInputItemList: input},
		Tools:           []responses.ToolUnionParam{searchTool},
		Temperature:     openai.Float(config.AgentTemperature),
		MaxOutputTokens: openai.Int(config.AgentMaxTokens),
	}

	// Build complete instructions with context
	var fullInstructions strings.Builder
	fullInstructions.WriteString(agentInstructions)
	fullInstructions.WriteString(fmt.Sprintf("\n\nCurrent date and time: %s", time.Now().Format("Monday, January 2, 2006 at 3:04 PM MST")))
	if systemPrompt != "" {
		fullInstructions.WriteString(fmt.Sprintf("\n\nThe user specified the following system prompt for the conversation, which you can use to draw context from in your decision:\n\"%s\"", systemPrompt))
	}

	params.Instructions = openai.String(fullInstructions.String())

	var parser *streamParser
	if onChunk != nil {
		streamCtx, cancelStream := context.WithCancel(ctx)
		defer cancelStream()

		parser = newStreamParser(cancelStream)
		stream := a.client.Responses.NewStreaming(streamCtx, params)

		for stream.Next() {
			event := stream.Current()
			onChunk(event)
			parser.handleEvent(event)
			if parser.aborted {
				break
			}
		}

		if !parser.aborted {
			if err := stream.Err(); err != nil {
				return nil, fmt.Errorf("agent streaming failed: %w", err)
			}
		}
	} else {
		resp, err := a.client.Responses.New(ctx, params)
		if err != nil {
			return nil, fmt.Errorf("agent LLM call failed: %w", err)
		}

		parser = newStreamParser(nil)
		parser.parseResponse(resp)
	}

	result := &Result{
		SearchReasoning: parser.reasoning.String(),
	}

	if len(parser.functionCalls) == 0 {
		log.Debug("Agent decided no search needed")
		return result, nil
	}

	log.Debugf("Agent requested %d search(es)", len(parser.functionCalls))

	// Collect queries from function calls
	type queryItem struct {
		id    string
		query string
	}
	var queries []queryItem

	for _, fc := range parser.functionCalls {
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

	// Return pending searches for the pipeline to execute
	for _, q := range queries {
		result.PendingSearches = append(result.PendingSearches, PendingSearch{
			ID:    q.id,
			Query: q.query,
		})
	}

	log.Debugf("Agent returning %d pending search(es)", len(result.PendingSearches))
	return result, nil
}
