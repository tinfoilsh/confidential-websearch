package agent

import (
	"context"
	"fmt"
	"sync"

	log "github.com/sirupsen/logrus"

	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
)

// SafeAgent wraps the base Agent with safety checks for PII and prompt injection
type SafeAgent struct {
	agent                *Agent
	safeguardClient      *safeguard.Client
	enablePIICheck       bool
	enableInjectionCheck bool
}

// NewSafeAgent creates a new SafeAgent wrapper
func NewSafeAgent(agent *Agent, safeguardClient *safeguard.Client) *SafeAgent {
	return &SafeAgent{
		agent:                agent,
		safeguardClient:      safeguardClient,
		enablePIICheck:       true,
		enableInjectionCheck: true,
	}
}

// SetPIICheckEnabled enables or disables PII checking
func (s *SafeAgent) SetPIICheckEnabled(enabled bool) {
	s.enablePIICheck = enabled
}

// SetInjectionCheckEnabled enables or disables prompt injection checking
func (s *SafeAgent) SetInjectionCheckEnabled(enabled bool) {
	s.enableInjectionCheck = enabled
}

// Run executes the agent with safety checks (non-streaming)
func (s *SafeAgent) Run(ctx context.Context, userQuery string) (*Result, error) {
	return s.RunWithContext(ctx, userQuery, "")
}

// RunWithContext executes the agent with safety checks and conversation context for PII detection
func (s *SafeAgent) RunWithContext(ctx context.Context, userQuery string, conversationContext string) (*Result, error) {
	// Create PII filter if enabled (passed as parameter for thread-safety)
	var filter SearchFilter
	if s.enablePIICheck && s.safeguardClient != nil {
		filter = s.createPIIFilter(ctx, conversationContext)
	}

	// Run the base agent with the filter (thread-safe)
	result, err := s.agent.RunWithFilter(ctx, userQuery, filter)
	if err != nil {
		return nil, err
	}

	// Apply injection filtering if enabled
	if s.enableInjectionCheck && s.safeguardClient != nil && len(result.ToolCalls) > 0 {
		result.ToolCalls = s.filterInjectedResults(ctx, result.ToolCalls)
	}

	return result, nil
}

// createPIIFilter returns a SearchFilter that checks queries for PII
func (s *SafeAgent) createPIIFilter(ctx context.Context, conversationContext string) SearchFilter {
	return func(filterCtx context.Context, queries []string) []string {
		if len(queries) == 0 {
			return queries
		}

		// Check all queries in parallel
		type checkResult struct {
			index   int
			allowed bool
		}

		results := make(chan checkResult, len(queries))
		var wg sync.WaitGroup

		for i, query := range queries {
			wg.Add(1)
			go func(idx int, q string) {
				defer wg.Done()

				// Build content with conversation context if available
				content := q
				if conversationContext != "" {
					content = fmt.Sprintf("Conversation context:\n%s\n\nSearch query to evaluate:\n%s", conversationContext, q)
				}

				checkResult, err := s.safeguardClient.Check(filterCtx, safeguard.PIILeakagePolicy, content)
				if err != nil {
					log.Errorf("PII check failed: %v", err)
					// On error, allow the query to proceed
					results <- struct {
						index   int
						allowed bool
					}{idx, true}
					return
				}

				if checkResult.Violation {
					log.Warnf("PII detected in query: %s - %s", checkResult.Category, checkResult.Rationale)
				}

				results <- struct {
					index   int
					allowed bool
				}{idx, !checkResult.Violation}
			}(i, query)
		}

		// Close results channel when all checks complete
		go func() {
			wg.Wait()
			close(results)
		}()

		// Collect results preserving order
		allowed := make([]bool, len(queries))
		for r := range results {
			allowed[r.index] = r.allowed
		}

		// Build filtered list
		var filtered []string
		for i, query := range queries {
			if allowed[i] {
				filtered = append(filtered, query)
			}
		}

		if len(filtered) < len(queries) {
			log.Infof("PII filter: %d/%d queries allowed", len(filtered), len(queries))
		}

		return filtered
	}
}

// filterInjectedResults checks search results for prompt injection and removes flagged ones
func (s *SafeAgent) filterInjectedResults(ctx context.Context, toolCalls []ToolCall) []ToolCall {
	// Flatten all results for parallel checking
	type resultRef struct {
		toolCallIdx int
		resultIdx   int
		content     string
	}

	var allResults []resultRef
	for ti, tc := range toolCalls {
		for ri, r := range tc.Results {
			allResults = append(allResults, resultRef{
				toolCallIdx: ti,
				resultIdx:   ri,
				content:     fmt.Sprintf("Title: %s\nURL: %s\nContent: %s", r.Title, r.URL, r.Content),
			})
		}
	}

	if len(allResults) == 0 {
		return toolCalls
	}

	// Check all results in parallel
	type checkResult struct {
		ref     resultRef
		flagged bool
	}

	results := make(chan checkResult, len(allResults))
	var wg sync.WaitGroup

	for _, ref := range allResults {
		wg.Add(1)
		go func(r resultRef) {
			defer wg.Done()

			check, err := s.safeguardClient.Check(ctx, safeguard.PromptInjectionPolicy, r.content)
			if err != nil {
				log.Errorf("Injection check failed: %v", err)
				// On error, allow the result to proceed
				results <- checkResult{ref: r, flagged: false}
				return
			}

			if check.Violation {
				log.Warnf("Prompt injection detected in result from %s: %s - %s",
					toolCalls[r.toolCallIdx].Results[r.resultIdx].URL,
					check.Category, check.Rationale)
			}

			results <- checkResult{ref: r, flagged: check.Violation}
		}(ref)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	// Build set of flagged results
	flagged := make(map[string]bool)
	for r := range results {
		if r.flagged {
			key := fmt.Sprintf("%d-%d", r.ref.toolCallIdx, r.ref.resultIdx)
			flagged[key] = true
		}
	}

	if len(flagged) == 0 {
		return toolCalls
	}

	// Filter out flagged results
	var filtered []ToolCall
	totalFiltered := 0
	for ti, tc := range toolCalls {
		var cleanResults []search.Result
		for ri, r := range tc.Results {
			key := fmt.Sprintf("%d-%d", ti, ri)
			if !flagged[key] {
				cleanResults = append(cleanResults, r)
			} else {
				totalFiltered++
			}
		}

		// Only include tool call if it has remaining results
		if len(cleanResults) > 0 {
			filtered = append(filtered, ToolCall{
				ID:      tc.ID,
				Query:   tc.Query,
				Results: cleanResults,
			})
		}
	}

	log.Infof("Injection filter: removed %d flagged results", totalFiltered)

	return filtered
}
