//go:build integration

package agent

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/openai/openai-go/v3/option"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/search"
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

	var fetcher URLFetcher
	cfAccountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	cfAPIToken := os.Getenv("CLOUDFLARE_API_TOKEN")
	if cfAccountID != "" && cfAPIToken != "" {
		fetcher = fetch.NewFetcher(cfAccountID, cfAPIToken)
	}

	var searcher SearchProvider
	exaKey := os.Getenv("EXA_API_KEY")
	if exaKey != "" {
		s, err := search.NewProvider(search.Config{ExaAPIKey: exaKey})
		if err != nil {
			t.Fatalf("Failed to create search provider: %v", err)
		}
		searcher = s
	}

	return New(client, "gpt-oss-120b", fetcher, searcher)
}

func runAgent(t *testing.T, agent *Agent, query string) *Result {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	messages := []ContextMessage{{Role: "user", Content: query}}
	result, err := agent.RunWithContext(ctx, messages, "", nil, nil)
	if err != nil {
		t.Fatalf("Agent failed: %v", err)
	}
	return result
}

// TestAgent_SearchQuery verifies a factual question triggers at least one search.
func TestAgent_SearchQuery(t *testing.T) {
	agent := setupAgent(t)
	if agent.searcher == nil {
		t.Skip("EXA_API_KEY not set, skipping search test")
	}
	result := runAgent(t, agent, "What is the latest news about SpaceX?")

	if len(result.SearchResults) == 0 {
		t.Error("expected at least one search, got none")
	}
}

// TestAgent_FetchQuery verifies a URL in the query triggers at least one fetch.
func TestAgent_FetchQuery(t *testing.T) {
	agent := setupAgent(t)
	if agent.fetcher == nil {
		t.Skip("CLOUDFLARE credentials not set, skipping fetch test")
	}
	result := runAgent(t, agent, "What is on https://example.com?")

	if len(result.FetchedPages) == 0 {
		t.Error("expected at least one fetched page, got none")
	}
}

// TestAgent_MultipleTools verifies the agent uses tools when a request needs both fetch and search.
func TestAgent_MultipleTools(t *testing.T) {
	agent := setupAgent(t)
	if agent.fetcher == nil || agent.searcher == nil {
		t.Skip("CLOUDFLARE or EXA credentials not set, skipping multi-tool test")
	}
	result := runAgent(t, agent, "Summarize https://example.com and search for similar websites")

	totalTools := len(result.FetchedPages) + len(result.SearchResults)
	if totalTools == 0 {
		t.Error("expected at least one tool call, got none")
	}
	if totalTools < 2 {
		t.Logf("warning: expected 2 tool calls (fetch + search), got %d (fetches=%d, searches=%d)",
			totalTools, len(result.FetchedPages), len(result.SearchResults))
	}
}

// TestAgent_NoToolsNeeded verifies simple queries don't trigger tools.
func TestAgent_NoToolsNeeded(t *testing.T) {
	agent := setupAgent(t)
	result := runAgent(t, agent, "What is 2+2?")

	if len(result.SearchResults) > 0 {
		t.Logf("warning: agent searched for simple math (got %d searches) — non-deterministic, not failing", len(result.SearchResults))
	}
	if len(result.FetchedPages) > 0 {
		t.Errorf("expected no fetches for simple math, got %d", len(result.FetchedPages))
	}
}

// TestAgent_MultipleFetches verifies multiple URLs trigger multiple fetches.
func TestAgent_MultipleFetches(t *testing.T) {
	agent := setupAgent(t)
	if agent.fetcher == nil {
		t.Skip("CLOUDFLARE credentials not set, skipping fetch test")
	}
	result := runAgent(t, agent, "Compare the contents of https://example.com and https://example.org")

	if len(result.FetchedPages) < 2 {
		t.Errorf("expected at least 2 fetched pages for 2 URLs, got %d", len(result.FetchedPages))
	}
}
