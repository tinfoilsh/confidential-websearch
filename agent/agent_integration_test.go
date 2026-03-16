//go:build integration

package agent

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/openai/openai-go/v3/option"
	"github.com/tinfoilsh/tinfoil-go"
)

func setupAgent(t *testing.T) *Agent {
	t.Helper()

	tinfoilKey := os.Getenv("TINFOIL_API_KEY")
	if tinfoilKey == "" {
		t.Skip("TINFOIL_API_KEY not set, skipping integration test")
	}

	client, err := tinfoil.NewClient(option.WithAPIKey(tinfoilKey))
	if err != nil {
		t.Fatalf("Failed to create Tinfoil client: %v", err)
	}

	return New(client, "gpt-oss-120b")
}

func runAgent(t *testing.T, agent *Agent, query string) *Result {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	messages := []ContextMessage{{Role: "user", Content: query}}
	result, err := agent.RunWithContext(ctx, messages, "", nil)
	if err != nil {
		t.Fatalf("Agent failed: %v", err)
	}
	return result
}

// TestAgent_SearchQuery verifies a factual question triggers at least one search.
func TestAgent_SearchQuery(t *testing.T) {
	agent := setupAgent(t)
	result := runAgent(t, agent, "What is the latest news about SpaceX?")

	if len(result.PendingSearches) == 0 {
		t.Error("expected at least one search, got none")
	}
}

// TestAgent_FetchQuery verifies a URL in the query triggers at least one fetch.
func TestAgent_FetchQuery(t *testing.T) {
	agent := setupAgent(t)
	result := runAgent(t, agent, "What is on https://example.com?")

	if len(result.PendingFetches) == 0 {
		t.Error("expected at least one fetch, got none")
	}
}

// TestAgent_MultipleTools verifies the agent uses tools when a request needs both fetch and search.
func TestAgent_MultipleTools(t *testing.T) {
	agent := setupAgent(t)
	result := runAgent(t, agent, "Summarize https://example.com and search for similar websites")

	totalTools := len(result.PendingFetches) + len(result.PendingSearches)
	if totalTools == 0 {
		t.Error("expected at least one tool call, got none")
	}
	if totalTools < 2 {
		t.Logf("warning: expected 2 tool calls (fetch + search), got %d (fetches=%d, searches=%d)",
			totalTools, len(result.PendingFetches), len(result.PendingSearches))
	}
}

// TestAgent_NoToolsNeeded verifies simple queries don't trigger tools.
// This is inherently non-deterministic so we log rather than fail on unexpected tool calls.
func TestAgent_NoToolsNeeded(t *testing.T) {
	agent := setupAgent(t)
	result := runAgent(t, agent, "What is 2+2?")

	if len(result.PendingSearches) > 0 {
		t.Logf("warning: agent searched for simple math (got %d searches) — non-deterministic, not failing", len(result.PendingSearches))
	}
	if len(result.PendingFetches) > 0 {
		t.Errorf("expected no fetches for simple math, got %d", len(result.PendingFetches))
	}
}

// TestAgent_MultipleFetches verifies multiple URLs trigger multiple fetches.
func TestAgent_MultipleFetches(t *testing.T) {
	agent := setupAgent(t)
	result := runAgent(t, agent, "Compare the contents of https://example.com and https://example.org")

	if len(result.PendingFetches) < 2 {
		t.Errorf("expected at least 2 fetches for 2 URLs, got %d", len(result.PendingFetches))
	}
}
