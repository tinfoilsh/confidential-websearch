package main

import (
	"context"
	"errors"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/prometheus/client_golang/prometheus/testutil"
)

type noArgs struct{}
type noOut struct{}

func toolCalls(t *testing.T, tool, outcome string) float64 {
	t.Helper()
	return testutil.ToFloat64(metricToolCalls.WithLabelValues(tool, outcome))
}

func TestInstrumentToolCountsOutcomes(t *testing.T) {
	const tool = "test_outcomes"
	okBefore := toolCalls(t, tool, outcomeOK)
	errBefore := toolCalls(t, tool, outcomeError)

	succeed := instrumentTool(tool, func(context.Context, *mcp.CallToolRequest, noArgs) (*mcp.CallToolResult, noOut, error) {
		return &mcp.CallToolResult{}, noOut{}, nil
	})
	failErr := instrumentTool(tool, func(context.Context, *mcp.CallToolRequest, noArgs) (*mcp.CallToolResult, noOut, error) {
		return nil, noOut{}, errors.New("boom")
	})
	failResult := instrumentTool(tool, func(context.Context, *mcp.CallToolRequest, noArgs) (*mcp.CallToolResult, noOut, error) {
		return &mcp.CallToolResult{IsError: true}, noOut{}, nil
	})

	ctx := context.Background()
	if _, _, err := succeed(ctx, nil, noArgs{}); err != nil {
		t.Fatalf("succeed returned error: %v", err)
	}
	if _, _, err := failErr(ctx, nil, noArgs{}); err == nil {
		t.Fatal("failErr should propagate the handler error")
	}
	if res, _, err := failResult(ctx, nil, noArgs{}); err != nil || res == nil || !res.IsError {
		t.Fatalf("failResult should pass through the IsError result, got res=%v err=%v", res, err)
	}

	if got := toolCalls(t, tool, outcomeOK) - okBefore; got != 1 {
		t.Fatalf("ok outcomes = %v, want 1", got)
	}
	if got := toolCalls(t, tool, outcomeError) - errBefore; got != 2 {
		t.Fatalf("error outcomes = %v, want 2 (one Go error, one IsError result)", got)
	}
	if got := testutil.ToFloat64(metricToolInflight.WithLabelValues(tool)); got != 0 {
		t.Fatalf("inflight after completion = %v, want 0", got)
	}
}

func TestInstrumentToolTracksInflight(t *testing.T) {
	const tool = "test_inflight"
	entered := make(chan struct{})
	release := make(chan struct{})

	blocking := instrumentTool(tool, func(context.Context, *mcp.CallToolRequest, noArgs) (*mcp.CallToolResult, noOut, error) {
		close(entered)
		<-release
		return &mcp.CallToolResult{}, noOut{}, nil
	})

	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _, _ = blocking(context.Background(), nil, noArgs{})
	}()

	<-entered
	if got := testutil.ToFloat64(metricToolInflight.WithLabelValues(tool)); got != 1 {
		t.Fatalf("inflight during call = %v, want 1", got)
	}
	close(release)
	<-done
	if got := testutil.ToFloat64(metricToolInflight.WithLabelValues(tool)); got != 0 {
		t.Fatalf("inflight after call = %v, want 0", got)
	}
}
