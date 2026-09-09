package main

import (
	"context"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Tool-level metrics. Go runtime and process collectors are registered by
// the default registry, so /metrics also carries memory, GC, and fd stats.
var (
	metricToolCalls = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "websearch_tool_calls_total",
		Help: "MCP tool invocations by tool name and outcome.",
	}, []string{"tool", "outcome"})

	metricToolDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "websearch_tool_duration_seconds",
		Help:    "Wall-clock duration of MCP tool invocations.",
		Buckets: prometheus.ExponentialBuckets(0.05, 2, 12),
	}, []string{"tool"})

	metricToolInflight = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "websearch_tool_inflight",
		Help: "MCP tool invocations currently executing.",
	}, []string{"tool"})
)

const (
	outcomeOK    = "ok"
	outcomeError = "error"
)

// instrumentTool wraps an MCP tool handler with call, duration, and
// in-flight metrics. A non-nil error from the handler is counted as an
// error outcome; MCP-level tool errors returned inside the result (IsError)
// are also counted so provider failures surfaced to the model are visible.
func instrumentTool[In, Out any](tool string, next mcp.ToolHandlerFor[In, Out]) mcp.ToolHandlerFor[In, Out] {
	return func(ctx context.Context, req *mcp.CallToolRequest, args In) (*mcp.CallToolResult, Out, error) {
		metricToolInflight.WithLabelValues(tool).Inc()
		start := time.Now()

		result, out, err := next(ctx, req, args)

		metricToolInflight.WithLabelValues(tool).Dec()
		metricToolDuration.WithLabelValues(tool).Observe(time.Since(start).Seconds())

		outcome := outcomeOK
		if err != nil || (result != nil && result.IsError) {
			outcome = outcomeError
		}
		metricToolCalls.WithLabelValues(tool, outcome).Inc()

		return result, out, err
	}
}
