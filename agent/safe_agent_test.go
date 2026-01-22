package agent

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/tinfoilsh/confidential-websearch/safeguard"
)

// MockSafeguardClient implements a mock for safeguard.Client
type MockSafeguardClient struct {
	CheckFunc func(ctx context.Context, policy, content string) (*safeguard.CheckResult, error)
}

func (m *MockSafeguardClient) Check(ctx context.Context, policy, content string) (*safeguard.CheckResult, error) {
	if m.CheckFunc != nil {
		return m.CheckFunc(ctx, policy, content)
	}
	return &safeguard.CheckResult{Violation: false, Rationale: "mock: no violation"}, nil
}

func TestNewSafeAgent(t *testing.T) {
	sa := NewSafeAgent(nil, nil)

	if sa == nil {
		t.Fatal("expected non-nil SafeAgent")
	}

	if !sa.enablePIICheck {
		t.Error("expected PII check to be enabled by default")
	}
}

func TestSetPIICheckEnabled(t *testing.T) {
	sa := NewSafeAgent(nil, nil)

	sa.SetPIICheckEnabled(false)
	if sa.enablePIICheck {
		t.Error("expected PII check to be disabled")
	}

	sa.SetPIICheckEnabled(true)
	if !sa.enablePIICheck {
		t.Error("expected PII check to be enabled")
	}
}

func TestCreatePIIFilter_EmptyQueries(t *testing.T) {
	mockClient := &MockSafeguardClient{}
	sa := &SafeAgent{
		safeguardClient: nil,
		enablePIICheck:  true,
	}

	filter := sa.createPIIFilterWithClient(context.Background(), mockClient)

	result := filter(context.Background(), []string{})
	if len(result.Allowed) != 0 {
		t.Errorf("expected empty result for empty input, got %d items", len(result.Allowed))
	}
	if len(result.Blocked) != 0 {
		t.Errorf("expected no blocked queries for empty input, got %d items", len(result.Blocked))
	}
}

func TestCreatePIIFilter_AllowsCleanQueries(t *testing.T) {
	var checkCount atomic.Int32
	mockClient := &MockSafeguardClient{
		CheckFunc: func(ctx context.Context, policy, content string) (*safeguard.CheckResult, error) {
			checkCount.Add(1)
			return &safeguard.CheckResult{Violation: false, Rationale: "no PII detected"}, nil
		},
	}

	sa := &SafeAgent{
		enablePIICheck: true,
	}

	filter := sa.createPIIFilterWithClient(context.Background(), mockClient)

	queries := []string{"weather in paris", "stock price of AAPL"}
	result := filter(context.Background(), queries)

	if len(result.Allowed) != 2 {
		t.Errorf("expected 2 queries allowed, got %d", len(result.Allowed))
	}

	if len(result.Blocked) != 0 {
		t.Errorf("expected no blocked queries, got %d", len(result.Blocked))
	}

	if checkCount.Load() != 2 {
		t.Errorf("expected 2 checks, got %d", checkCount.Load())
	}
}

func TestCreatePIIFilter_BlocksPIIQueries(t *testing.T) {
	mockClient := &MockSafeguardClient{
		CheckFunc: func(ctx context.Context, policy, content string) (*safeguard.CheckResult, error) {
			// Block queries containing "SSN"
			if strings.Contains(content, "SSN") {
				return &safeguard.CheckResult{Violation: true, Rationale: "SSN detected"}, nil
			}
			return &safeguard.CheckResult{Violation: false, Rationale: "clean"}, nil
		},
	}

	sa := &SafeAgent{
		enablePIICheck: true,
	}

	filter := sa.createPIIFilterWithClient(context.Background(), mockClient)

	queries := []string{"weather in paris", "my SSN is 123-45-6789", "stock price"}
	result := filter(context.Background(), queries)

	if len(result.Allowed) != 2 {
		t.Errorf("expected 2 queries allowed (1 blocked), got %d", len(result.Allowed))
	}

	// Verify the blocked query is not in allowed results
	for _, q := range result.Allowed {
		if strings.Contains(q, "SSN") {
			t.Error("SSN query should have been blocked")
		}
	}

	// Verify blocked query info
	if len(result.Blocked) != 1 {
		t.Fatalf("expected 1 blocked query, got %d", len(result.Blocked))
	}
	if !strings.Contains(result.Blocked[0].Query, "SSN") {
		t.Error("blocked query should contain SSN")
	}
	if result.Blocked[0].Reason != "SSN detected" {
		t.Errorf("expected reason 'SSN detected', got '%s'", result.Blocked[0].Reason)
	}
}

func TestCreatePIIFilter_FailOpenOnError(t *testing.T) {
	mockClient := &MockSafeguardClient{
		CheckFunc: func(ctx context.Context, policy, content string) (*safeguard.CheckResult, error) {
			return nil, errors.New("safeguard service unavailable")
		},
	}

	sa := &SafeAgent{
		enablePIICheck: true,
	}

	filter := sa.createPIIFilterWithClient(context.Background(), mockClient)

	queries := []string{"query1", "query2"}
	result := filter(context.Background(), queries)

	// Fail-open: all queries should be allowed when service errors
	if len(result.Allowed) != 2 {
		t.Errorf("expected all queries allowed on error (fail-open), got %d", len(result.Allowed))
	}
	if len(result.Blocked) != 0 {
		t.Errorf("expected no blocked queries on error (fail-open), got %d", len(result.Blocked))
	}
}

func TestCreatePIIFilter_QueryOnly(t *testing.T) {
	var receivedContent string
	mockClient := &MockSafeguardClient{
		CheckFunc: func(ctx context.Context, policy, content string) (*safeguard.CheckResult, error) {
			receivedContent = content
			return &safeguard.CheckResult{Violation: false, Rationale: "clean"}, nil
		},
	}

	sa := &SafeAgent{
		enablePIICheck: true,
	}

	filter := sa.createPIIFilterWithClient(context.Background(), mockClient)

	filter(context.Background(), []string{"test query"})

	// PII filter should only receive the query, not any conversation context
	if receivedContent != "test query" {
		t.Errorf("expected only query content, got '%s'", receivedContent)
	}
}
