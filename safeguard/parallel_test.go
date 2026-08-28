package safeguard

import (
	"context"
	"testing"
	"time"
)

type deadlineChecker struct {
	deadline time.Time
}

func (c *deadlineChecker) Check(ctx context.Context, _ string) (*CheckResult, error) {
	deadline, ok := ctx.Deadline()
	if ok {
		c.deadline = deadline
	}
	return &CheckResult{}, nil
}

func TestCheckItemsAppliesRequestTimeout(t *testing.T) {
	checker := &deadlineChecker{}
	startedAt := time.Now()

	results := CheckItems(context.Background(), checker, []string{"content"})

	if len(results) != 1 || results[0].Err != nil {
		t.Fatalf("unexpected results: %+v", results)
	}
	if checker.deadline.IsZero() {
		t.Fatal("expected checker context to have a deadline")
	}
	remaining := checker.deadline.Sub(startedAt)
	if remaining < safeguardRequestTimeout-time.Second || remaining > safeguardRequestTimeout+time.Second {
		t.Fatalf("expected deadline near %v, got %v", safeguardRequestTimeout, remaining)
	}
}
