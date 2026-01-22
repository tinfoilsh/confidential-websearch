package agent

import (
	"context"
	"sync"

	log "github.com/sirupsen/logrus"

	"github.com/tinfoilsh/confidential-websearch/safeguard"
)

// SafeguardChecker is an interface for safety checking (allows mocking in tests)
type SafeguardChecker interface {
	Check(ctx context.Context, policy, content string) (*safeguard.CheckResult, error)
}

// SafeAgent wraps the base Agent with PII filtering on search queries
type SafeAgent struct {
	agent           *Agent
	safeguardClient *safeguard.Client
	enablePIICheck  bool
}

// NewSafeAgent creates a new SafeAgent wrapper
func NewSafeAgent(agent *Agent, safeguardClient *safeguard.Client) *SafeAgent {
	return &SafeAgent{
		agent:           agent,
		safeguardClient: safeguardClient,
		enablePIICheck:  true,
	}
}

// SetPIICheckEnabled enables or disables PII checking
func (s *SafeAgent) SetPIICheckEnabled(enabled bool) {
	s.enablePIICheck = enabled
}

// RunWithContext implements AgentRunner interface.
// It forwards full conversation context to the base agent while applying PII filtering.
func (s *SafeAgent) RunWithContext(ctx context.Context, messages []ContextMessage, systemPrompt string, onChunk ChunkCallback) (*Result, error) {
	var filter SearchFilter
	if s.enablePIICheck && s.safeguardClient != nil {
		filter = s.createPIIFilter(ctx)
	}

	return s.agent.RunWithFilter(ctx, messages, systemPrompt, onChunk, filter)
}

// createPIIFilter creates a PII filter that checks query content
func (s *SafeAgent) createPIIFilter(ctx context.Context) SearchFilter {
	return s.createPIIFilterWithClient(ctx, s.safeguardClient)
}

// createPIIFilterWithClient returns a SearchFilter using the provided checker (for testing)
func (s *SafeAgent) createPIIFilterWithClient(ctx context.Context, client SafeguardChecker) SearchFilter {
	return func(filterCtx context.Context, queries []string) FilterResult {
		if len(queries) == 0 {
			return FilterResult{Allowed: queries}
		}

		if client == nil {
			return FilterResult{Allowed: queries}
		}

		// Check all queries in parallel
		type checkResult struct {
			index     int
			allowed   bool
			rationale string
		}

		results := make(chan checkResult, len(queries))
		var wg sync.WaitGroup

		for i, query := range queries {
			wg.Add(1)
			go func(idx int, q string) {
				defer wg.Done()

				check, err := client.Check(filterCtx, safeguard.PIILeakagePolicy, q)
				if err != nil {
					log.Errorf("PII check failed: %v", err)
					// On error, allow the query to proceed
					results <- checkResult{idx, true, ""}
					return
				}

				if check.Violation {
					log.Warnf("PII detected in query: %s", check.Rationale)
				}

				results <- checkResult{idx, !check.Violation, check.Rationale}
			}(i, query)
		}

		// Close results channel when all checks complete
		go func() {
			wg.Wait()
			close(results)
		}()

		// Collect results preserving order
		type queryResult struct {
			allowed   bool
			rationale string
		}
		queryResults := make([]queryResult, len(queries))
		for r := range results {
			queryResults[r.index] = queryResult{r.allowed, r.rationale}
		}

		// Build filter result
		var filterResult FilterResult
		for i, query := range queries {
			if queryResults[i].allowed {
				filterResult.Allowed = append(filterResult.Allowed, query)
			} else {
				filterResult.Blocked = append(filterResult.Blocked, BlockedQuery{
					Query:  query,
					Reason: queryResults[i].rationale,
				})
			}
		}

		if len(filterResult.Blocked) > 0 {
			log.Infof("PII filter: %d/%d queries allowed", len(filterResult.Allowed), len(queries))
		}

		return filterResult
	}
}
