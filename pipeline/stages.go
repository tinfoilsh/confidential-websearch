package pipeline

import (
	"context"
	"encoding/json"
	"sync"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	log "github.com/sirupsen/logrus"

	"github.com/tinfoilsh/confidential-websearch/agent"
	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
)

// Stage represents a single processing step in the pipeline
type Stage interface {
	Name() string
	Execute(ctx *Context) error
}

// AgentRunner defines the interface for running the agent with full context
type AgentRunner interface {
	RunWithContext(ctx context.Context, messages []agent.ContextMessage, systemPrompt string, onChunk agent.ChunkCallback) (*agent.Result, error)
}

// SafeguardChecker defines the interface for safety checks
type SafeguardChecker interface {
	Check(ctx context.Context, policy, content string) (*safeguard.CheckResult, error)
}

// MessageBuilder defines the interface for building responder messages
type MessageBuilder interface {
	Build(inputMessages []Message, searchResults []agent.ToolCall) []openai.ChatCompletionMessageParamUnion
}

// ResponderParams contains parameters for a responder LLM call
type ResponderParams struct {
	Model       string
	Messages    []openai.ChatCompletionMessageParamUnion
	Temperature *float64
	MaxTokens   *int64
}

// ResponderResultData contains the result of a non-streaming responder call
type ResponderResultData struct {
	ID      string
	Model   string
	Object  string
	Created int64
	Content string
	Usage   openai.CompletionUsage
}

// Responder defines the interface for making responder LLM calls
type Responder interface {
	Complete(ctx context.Context, params ResponderParams, opts ...option.RequestOption) (*ResponderResultData, error)
	Stream(ctx context.Context, params ResponderParams, annotations []Annotation, reasoning string, emitter EventEmitter, opts ...option.RequestOption) error
}

// ValidateStage validates the incoming request
type ValidateStage struct{}

func (s *ValidateStage) Name() string { return "validate" }

func (s *ValidateStage) Execute(ctx *Context) error {
	if ctx.Request == nil {
		return &ValidationError{Message: "request is nil"}
	}

	if ctx.Request.Model == "" {
		return &ValidationError{Field: "model", Message: "model parameter is required"}
	}

	// Extract user query from messages (last user message)
	if ctx.Request.Format == FormatChatCompletion {
		for i := len(ctx.Request.Messages) - 1; i >= 0; i-- {
			if ctx.Request.Messages[i].Role == "user" {
				if text := ctx.Request.Messages[i].GetTextContent(); text != "" {
					ctx.UserQuery = text
					break
				}
			}
		}
		if ctx.UserQuery == "" {
			return &ValidationError{Message: "no user message found"}
		}
	} else {
		// Responses API format
		if ctx.Request.Input == "" {
			return &ValidationError{Field: "input", Message: "input parameter is required"}
		}
		ctx.UserQuery = ctx.Request.Input
	}

	return nil
}

// AgentStage runs the agent to determine if search is needed
type AgentStage struct {
	Agent AgentRunner
}

func (s *AgentStage) Name() string { return "agent" }

func (s *AgentStage) Execute(ctx *Context) error {
	// Skip agent if web search is not enabled
	if !ctx.Request.WebSearchEnabled {
		return nil
	}

	ctx.State.Transition(StateProcessing, map[string]any{"query": ctx.UserQuery})

	systemPrompt, messages := extractAgentContext(ctx.Request.Messages)
	// Fallback for Responses API: if messages is empty, use ctx.UserQuery
	if len(messages) == 0 && ctx.UserQuery != "" {
		messages = []agent.ContextMessage{{Role: "user", Content: ctx.UserQuery}}
	}

	// Pass PII check setting to agent via context
	agentCtx := agent.WithPIICheckEnabled(ctx.Context, ctx.Request.PIICheckEnabled)

	result, err := s.Agent.RunWithContext(agentCtx, messages, systemPrompt, nil)
	if err != nil {
		return &AgentError{Err: err}
	}

	ctx.AgentResult = result
	return nil
}

// SearchStage executes pending searches from the agent
type SearchStage struct {
	Searcher search.Provider
}

func (s *SearchStage) Name() string { return "search" }

func (s *SearchStage) Execute(ctx *Context) error {
	// Emit blocked query events first (before early return check)
	if ctx.AgentResult != nil && ctx.Emitter != nil {
		for _, bq := range ctx.AgentResult.BlockedQueries {
			ctx.Emitter.EmitSearchCall(bq.ID, "blocked", bq.Query, bq.Reason, 0, ctx.Request.Model)
		}
	}

	if ctx.AgentResult == nil || len(ctx.AgentResult.PendingSearches) == 0 {
		return nil
	}

	// Emit in_progress events for all pending searches
	if ctx.Emitter != nil {
		for _, ps := range ctx.AgentResult.PendingSearches {
			ctx.Emitter.EmitSearchCall(ps.ID, "in_progress", ps.Query, "", 0, ctx.Request.Model)
		}
	}

	// Execute searches in parallel
	var wg sync.WaitGroup
	var mu sync.Mutex
	var results []agent.ToolCall

	for _, ps := range ctx.AgentResult.PendingSearches {
		wg.Add(1)
		go func(pending agent.PendingSearch) {
			defer wg.Done()

			log.Debug("Executing search")
			searchResults, err := s.Searcher.Search(ctx.Context, pending.Query, config.DefaultMaxSearchResults)
			if err != nil {
				log.Errorf("Search failed: %v", err)
				if ctx.Emitter != nil {
					ctx.Emitter.EmitSearchCall(pending.ID, "failed", pending.Query, err.Error(), 0, ctx.Request.Model)
				}
				return
			}

			mu.Lock()
			results = append(results, agent.ToolCall{
				ID:      pending.ID,
				Query:   pending.Query,
				Results: searchResults,
			})
			mu.Unlock()

			// Emit completed event
			if ctx.Emitter != nil {
				ctx.Emitter.EmitSearchCall(pending.ID, "completed", pending.Query, "", 0, ctx.Request.Model)
			}
		}(ps)
	}

	wg.Wait()
	ctx.SearchResults = results
	return nil
}

// FilterResultsStage checks search results for prompt injection
type FilterResultsStage struct {
	Checker SafeguardChecker
	Policy  string
	Enabled bool
}

func (s *FilterResultsStage) Name() string { return "filter_results" }

func (s *FilterResultsStage) Execute(ctx *Context) error {
	// Check if injection check is enabled in request, fall back to stage default
	enabled := s.Enabled
	if ctx.Request != nil {
		enabled = ctx.Request.InjectionCheckEnabled
	}
	if !enabled || s.Checker == nil || len(ctx.SearchResults) == 0 {
		return nil
	}

	// Flatten all search result contents for parallel checking
	type resultRef struct {
		tcIdx int
		rIdx  int
	}
	var refs []resultRef
	var contents []string
	for tcIdx, tc := range ctx.SearchResults {
		for rIdx, r := range tc.Results {
			refs = append(refs, resultRef{tcIdx, rIdx})
			contents = append(contents, r.Content)
		}
	}

	checkResults := safeguard.CheckItems(ctx.Context, s.Checker, s.Policy, contents)

	// Build flagged map from results
	flagged := make(map[int]map[int]bool)
	for i, r := range checkResults {
		ref := refs[i]
		if r.Err != nil {
			log.Errorf("Injection check failed: %v", r.Err)
			// On error, allow the result to proceed
			continue
		}
		if r.Violation {
			log.Warnf("Prompt injection detected: %s", r.Rationale)
			if flagged[ref.tcIdx] == nil {
				flagged[ref.tcIdx] = make(map[int]bool)
			}
			flagged[ref.tcIdx][ref.rIdx] = true
		}
	}

	// Filter out flagged results
	var filtered []agent.ToolCall
	totalFiltered := 0
	for tcIdx, tc := range ctx.SearchResults {
		var cleanResults []search.Result
		for rIdx, r := range tc.Results {
			if flagged[tcIdx] != nil && flagged[tcIdx][rIdx] {
				totalFiltered++
				continue
			}
			cleanResults = append(cleanResults, r)
		}
		if len(cleanResults) > 0 {
			filtered = append(filtered, agent.ToolCall{
				ID:      tc.ID,
				Query:   tc.Query,
				Results: cleanResults,
			})
		}
	}

	if totalFiltered > 0 {
		log.Infof("Injection filter: removed %d results", totalFiltered)
	}

	ctx.SearchResults = filtered
	return nil
}

// BuildMessagesStage constructs the messages for the responder LLM
type BuildMessagesStage struct {
	Builder MessageBuilder
}

func (s *BuildMessagesStage) Name() string { return "build_messages" }

func (s *BuildMessagesStage) Execute(ctx *Context) error {
	messages := ctx.Request.Messages

	// For Responses API, construct messages from Input since Messages is empty
	if ctx.Request.Format == FormatResponses && len(messages) == 0 {
		contentJSON, _ := json.Marshal(ctx.UserQuery)
		messages = []Message{{Role: "user", Content: contentJSON}}
	}

	ctx.ResponderMessages = s.Builder.Build(messages, ctx.SearchResults)
	return nil
}

// ResponderStage calls the responder LLM to generate the final response
type ResponderStage struct {
	Responder Responder
}

func (s *ResponderStage) Name() string { return "responder" }

func (s *ResponderStage) Execute(ctx *Context) error {
	ctx.State.Transition(StateResponding, nil)

	params := ResponderParams{
		Model:       ctx.Request.Model,
		Messages:    ctx.ResponderMessages,
		Temperature: ctx.Request.Temperature,
		MaxTokens:   ctx.Request.MaxTokens,
	}

	if ctx.IsStreaming() {
		annotations := BuildAnnotations(ctx.SearchResults)
		reasoning := ""
		if ctx.AgentResult != nil {
			reasoning = ctx.AgentResult.SearchReasoning
		}

		err := s.Responder.Stream(ctx.Context, params, annotations, reasoning, ctx.Emitter, ctx.ReqOpts...)
		if err != nil {
			return &ResponderError{Err: err}
		}
	} else {
		result, err := s.Responder.Complete(ctx.Context, params, ctx.ReqOpts...)
		if err != nil {
			return &ResponderError{Err: err}
		}

		// Store result for handler to use
		ctx.ResponderResult = result
	}

	ctx.State.Transition(StateCompleted, nil)
	return nil
}

// BuildAnnotations creates URL citations from search results
func BuildAnnotations(searchResults []agent.ToolCall) []Annotation {
	if len(searchResults) == 0 {
		return nil
	}

	var annotations []Annotation
	for _, tc := range searchResults {
		for _, r := range tc.Results {
			annotations = append(annotations, Annotation{
				Type: "url_citation",
				URLCitation: URLCitation{
					URL:   r.URL,
					Title: r.Title,
				},
			})
		}
	}
	return annotations
}

// extractAgentContext extracts the system prompt and converts pipeline messages
// to agent.ContextMessage format for the new RunWithContext method.
// Only text content is passed to the agent (images are handled by the responder).
func extractAgentContext(messages []Message) (systemPrompt string, agentMessages []agent.ContextMessage) {
	for _, msg := range messages {
		text := msg.GetTextContent()
		if msg.Role == "system" {
			systemPrompt = text
			continue
		}

		agentMessages = append(agentMessages, agent.ContextMessage{
			Role:    msg.Role,
			Content: text,
		})
	}
	return
}
