package safeguard

import (
	"context"
	"sync"
)

// Checker defines the interface for safety checks
type Checker interface {
	Check(ctx context.Context, policy, content string) (*CheckResult, error)
}

// ItemResult contains the check result for a single item
type ItemResult struct {
	Index     int
	Violation bool
	Rationale string
	Err       error
}

// CheckItems runs parallel safety checks on multiple items.
// Returns results in index order. On error, Err is set and Violation defaults to false.
func CheckItems(ctx context.Context, checker Checker, policy string, items []string) []ItemResult {
	if len(items) == 0 {
		return nil
	}

	results := make(chan ItemResult, len(items))
	var wg sync.WaitGroup

	for i, item := range items {
		wg.Add(1)
		go func(idx int, content string) {
			defer wg.Done()

			check, err := checker.Check(ctx, policy, content)
			if err != nil {
				results <- ItemResult{Index: idx, Err: err}
				return
			}

			results <- ItemResult{
				Index:     idx,
				Violation: check.Violation,
				Rationale: check.Rationale,
			}
		}(i, item)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results in index order
	ordered := make([]ItemResult, len(items))
	for r := range results {
		ordered[r.Index] = r
	}

	return ordered
}
