package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	log "github.com/sirupsen/logrus"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/config"
)

const agentInstructions = `You are a search routing agent. Your task is to decide if a web search or URL fetch would help answer the user's request.

You have two tools:
- search: Search the web for information. Use this when the user asks a question that requires up-to-date or broad information.
- fetch: Fetch the contents of a specific URL. Use this when the user shares or references a URL and you need its contents.

You can call multiple tools in a single response. For example, you can fetch a URL and search for related information at the same time.

Focus primarily on the most recent user message. Earlier messages provide context, but the latest message should drive your decisions.

If you're unsure whether a search would help, lean towards searching - it's better to have information available than to miss something useful.

If no tools are needed: Do not call any tools and do not output any text.`

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
	client   *tinfoil.Client
	model    string
	fetcher  URLFetcher
	searcher SearchProvider
}

// New creates a new agent
func New(client *tinfoil.Client, model string, fetcher URLFetcher, searcher SearchProvider) *Agent {
	return &Agent{client: client, model: model, fetcher: fetcher, searcher: searcher}
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
func (a *Agent) RunWithContext(ctx context.Context, messages []ContextMessage, systemPrompt string, onChunk ChunkCallback, onToolEvent ToolEventCallback) (*Result, error) {
	return a.run(ctx, messages, systemPrompt, onChunk, nil, onToolEvent)
}

// RunWithFilter executes the agent with a custom search filter.
// Used by SafeAgent to inject PII filtering.
func (a *Agent) RunWithFilter(ctx context.Context, messages []ContextMessage, systemPrompt string, onChunk ChunkCallback, filter SearchFilter, onToolEvent ToolEventCallback) (*Result, error) {
	return a.run(ctx, messages, systemPrompt, onChunk, filter, onToolEvent)
}

// run is the internal implementation
func (a *Agent) run(ctx context.Context, messages []ContextMessage, systemPrompt string, onChunk ChunkCallback, filter SearchFilter, onToolEvent ToolEventCallback) (*Result, error) {
	searchTool := responses.ToolParamOfFunction(
		"search",
		SearchToolParams,
		false,
	)
	searchTool.OfFunction.Description = openai.String("Search the web for current information.")

	fetchTool := responses.ToolParamOfFunction(
		"fetch",
		FetchToolParams,
		false,
	)
	fetchTool.OfFunction.Description = openai.String("Fetch the contents of a specific URL.")

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

	// Build complete instructions with context
	var fullInstructions strings.Builder
	fullInstructions.WriteString(agentInstructions)
	fullInstructions.WriteString(fmt.Sprintf("\n\nCurrent date and time: %s", time.Now().Format("Monday, January 2, 2006 at 3:04 PM MST")))
	if systemPrompt != "" {
		fullInstructions.WriteString(fmt.Sprintf("\n\nThe user specified the following system prompt for the conversation, which you can use to draw context from in your decision:\n\"%s\"", systemPrompt))
	}

	emitToolEvent := func(toolType, id, status, detail string) {
		if onToolEvent != nil {
			onToolEvent(toolType, id, status, detail)
		}
	}

	result := &Result{}
	var allReasoning strings.Builder

	for iteration := range config.AgentMaxIterations {
		if ctx.Err() != nil {
			break
		}

		params := responses.ResponseNewParams{
			Model:           shared.ResponsesModel(a.model),
			Input:           responses.ResponseNewParamsInputUnion{OfInputItemList: input},
			Tools:           []responses.ToolUnionParam{searchTool, fetchTool},
			Temperature:     openai.Float(config.AgentTemperature),
			MaxOutputTokens: openai.Int(config.AgentMaxTokens),
			Instructions:    openai.String(fullInstructions.String()),
		}

		var parser *streamParser
		if onChunk != nil {
			streamCtx, cancelStream := context.WithCancel(ctx)

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
					cancelStream()
					return nil, fmt.Errorf("agent streaming failed: %w", err)
				}
			}
			cancelStream()
		} else {
			resp, err := a.client.Responses.New(ctx, params)
			if err != nil {
				return nil, fmt.Errorf("agent LLM call failed: %w", err)
			}

			parser = newStreamParser(nil)
			parser.parseResponse(resp)
		}

		// Accumulate reasoning across iterations
		if parser.reasoning.Len() > 0 {
			allReasoning.WriteString(parser.reasoning.String())
		}

		if parser.aborted || len(parser.functionCalls) == 0 {
			log.Debugf("Agent decided no more tools needed (iteration %d)", iteration+1)
			break
		}

		log.Debugf("Agent requested %d tool call(s) (iteration %d)", len(parser.functionCalls), iteration+1)

		// Feed back all output items in order per OpenAI docs:
		// reasoning → function_call → function_call_output
		// This preserves the model's chain-of-thought across iterations.
		if parser.reasoning.Len() > 0 {
			input = append(input, responses.ResponseInputItemParamOfReasoning("", []responses.ResponseReasoningItemSummaryParam{
				{Text: parser.reasoning.String()},
			}))
		}

		// Sort by output index to preserve the model's intended tool-call order
		sortedIndices := make([]int, 0, len(parser.functionCalls))
		for idx := range parser.functionCalls {
			sortedIndices = append(sortedIndices, idx)
		}
		slices.Sort(sortedIndices)

		// Phase 1: Parse tool calls and append function_call items to input
		type parsedCall struct {
			fc       *functionCall
			toolType string // "search" or "fetch"
			detail   string // query or URL
		}
		var parsed []parsedCall
		processedCalls := make(map[string]bool)

		for _, idx := range sortedIndices {
			fc := parser.functionCalls[idx]
			switch fc.name {
			case "search":
				var args SearchArgs
				if err := json.Unmarshal([]byte(fc.arguments.String()), &args); err != nil {
					log.Errorf("Failed to parse search arguments: %v", err)
					continue
				}
				query := strings.TrimSpace(args.Query)
				if query == "" {
					continue
				}
				input = append(input, responses.ResponseInputItemParamOfFunctionCall(fc.arguments.String(), fc.id, "search"))
				processedCalls[fc.id] = true
				parsed = append(parsed, parsedCall{fc: fc, toolType: "search", detail: query})

			case "fetch":
				var args FetchArgs
				if err := json.Unmarshal([]byte(fc.arguments.String()), &args); err != nil {
					log.Errorf("Failed to parse fetch arguments: %v", err)
					continue
				}
				fetchURL := strings.TrimSpace(args.URL)
				if fetchURL == "" {
					continue
				}
				input = append(input, responses.ResponseInputItemParamOfFunctionCall(fc.arguments.String(), fc.id, "fetch"))
				processedCalls[fc.id] = true
				parsed = append(parsed, parsedCall{fc: fc, toolType: "fetch", detail: fetchURL})
			}
		}

		// Phase 2: Execute all tool calls concurrently
		var wg sync.WaitGroup
		var mu sync.Mutex
		toolOutputs := make(map[string]string)

		for _, pc := range parsed {
			wg.Add(1)
			go func(pc parsedCall) {
				defer wg.Done()

				switch pc.toolType {
				case "search":
					// Apply PII filter before executing search
					if filter != nil {
						filterResult := filter(ctx, []string{pc.detail})
						if len(filterResult.Blocked) > 0 {
							mu.Lock()
							result.BlockedQueries = append(result.BlockedQueries, BlockedQuery{
								ID:     pc.fc.id,
								Query:  pc.detail,
								Reason: filterResult.Blocked[0].Reason,
							})
							toolOutputs[pc.fc.id] = "Search blocked: " + filterResult.Blocked[0].Reason
							mu.Unlock()
							emitToolEvent("search", pc.fc.id, "blocked", pc.detail)
							log.Debugf("Search blocked by PII filter: %s", pc.detail)
							return
						}
					}

					if a.searcher != nil {
						emitToolEvent("search", pc.fc.id, "in_progress", pc.detail)
						searchResults, err := a.searcher.Search(ctx, pc.detail, config.DefaultMaxSearchResults)
						if err != nil {
							mu.Lock()
							toolOutputs[pc.fc.id] = "Search failed: " + err.Error()
							mu.Unlock()
							emitToolEvent("search", pc.fc.id, "failed", pc.detail)
							log.Errorf("Search failed for %q: %v", pc.detail, err)
						} else {
							var sb strings.Builder
							for i, r := range searchResults {
								fmt.Fprintf(&sb, "[%d] %s\n%s\n%s\n\n", i+1, r.Title, r.URL, r.Content)
							}
							mu.Lock()
							result.SearchResults = append(result.SearchResults, ToolCall{
								ID:      pc.fc.id,
								Query:   pc.detail,
								Results: searchResults,
							})
							toolOutputs[pc.fc.id] = sb.String()
							mu.Unlock()
							emitToolEvent("search", pc.fc.id, "completed", pc.detail)
							log.Debugf("Search %q returned %d results", pc.detail, len(searchResults))
						}
					}

				case "fetch":
					if a.fetcher != nil {
						emitToolEvent("fetch", pc.fc.id, "in_progress", pc.detail)
						pages := a.fetcher.FetchURLs(ctx, []string{pc.detail})
						if len(pages) > 0 {
							mu.Lock()
							result.FetchedPages = append(result.FetchedPages, pages[0])
							toolOutputs[pc.fc.id] = pages[0].Content
							mu.Unlock()
							emitToolEvent("fetch", pc.fc.id, "completed", pc.detail)
							log.Debugf("Fetched %s (%d chars)", pc.detail, len(pages[0].Content))
						} else {
							mu.Lock()
							toolOutputs[pc.fc.id] = "Failed to fetch URL."
							mu.Unlock()
							emitToolEvent("fetch", pc.fc.id, "failed", pc.detail)
							log.Debugf("Failed to fetch %s", pc.detail)
						}
					}
				}
			}(pc)
		}
		wg.Wait()

		// Phase 3: Feed back function_call_output items in sorted order
		for _, idx := range sortedIndices {
			fc := parser.functionCalls[idx]
			if !processedCalls[fc.id] {
				continue
			}
			output := toolOutputs[fc.id]
			if output == "" {
				output = "ok"
			}
			input = append(input, responses.ResponseInputItemParamOfFunctionCallOutput(fc.id, output))
		}
	}

	result.SearchReasoning = allReasoning.String()

	log.Debugf("Agent returning %d search result(s), %d fetched page(s)", len(result.SearchResults), len(result.FetchedPages))
	return result, nil
}
