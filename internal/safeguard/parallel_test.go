package safeguard

import (
	"context"
	"testing"
	"time"
)

const deadlineTolerance = time.Second

type deadlineChecker struct {
	hasDeadline bool
	remaining   time.Duration
}

func (c *deadlineChecker) Check(ctx context.Context, _ string) (*CheckResult, error) {
	deadline, ok := ctx.Deadline()
	c.hasDeadline = ok
	if ok {
		c.remaining = time.Until(deadline)
	}
	return &CheckResult{}, nil
}

func TestCheckItemsAppliesRequestTimeout(t *testing.T) {
	checker := &deadlineChecker{}

	results := CheckItems(context.Background(), checker, []string{"content"})

	if len(results) != 1 || results[0].Err != nil {
		t.Fatalf("unexpected results: %+v", results)
	}
	if !checker.hasDeadline {
		t.Fatal("expected checker context to have a deadline")
	}
	assertRequestTimeout(t, checker.remaining)
}

func assertRequestTimeout(t *testing.T, remaining time.Duration) {
	t.Helper()
	if remaining < safeguardRequestTimeout-deadlineTolerance || remaining > safeguardRequestTimeout {
		t.Fatalf("expected timeout near %v, got %v", safeguardRequestTimeout, remaining)
	}
}
