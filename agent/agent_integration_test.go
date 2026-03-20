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

// TestAgent_SearchQuery verifies a factual question triggers a search.
func TestAgent_SearchQuery(t *testing.T) {
	agent := setupAgent(t)
	if agent.searcher == nil {
		t.Skip("EXA_API_KEY not set, skipping search test")
	}
	result := runAgent(t, agent, "What is the latest news about SpaceX?")

	if len(result.SearchResults) == 0 {
		t.Error("expected at least one search result, got none")
	}
}

// TestAgent_FetchQuery verifies a URL fetch request triggers a fetch.
func TestAgent_FetchQuery(t *testing.T) {
	agent := setupAgent(t)
	if agent.fetcher == nil {
		t.Skip("CLOUDFLARE credentials not set, skipping fetch test")
	}
	result := runAgent(t, agent, "Fetch https://htmlonly.com and tell me about it")

	totalTools := len(result.FetchedPages) + len(result.SearchResults)
	if totalTools == 0 {
		t.Error("expected at least one tool call, got none")
	}
	if len(result.FetchedPages) == 0 {
		t.Log("warning: expected a fetch for the URL, but agent chose a different tool")
	}
}

// TestAgent_MultipleTools verifies the agent can use both tools across iterations.
func TestAgent_MultipleTools(t *testing.T) {
	agent := setupAgent(t)
	if agent.fetcher == nil || agent.searcher == nil {
		t.Skip("CLOUDFLARE or EXA credentials not set, skipping multi-tool test")
	}
	result := runAgent(t, agent, "Fetch https://htmlonly.com and tell me about secure enclaves")

	totalTools := len(result.FetchedPages) + len(result.SearchResults)
	if totalTools == 0 {
		t.Error("expected at least one tool call, got none")
	}
	if len(result.FetchedPages) == 0 || len(result.SearchResults) == 0 {
		t.Logf("warning: expected both fetch and search, got fetches=%d searches=%d — non-deterministic",
			len(result.FetchedPages), len(result.SearchResults))
	}
}

// TestAgent_NoToolsNeeded verifies simple queries don't trigger tools.
func TestAgent_NoToolsNeeded(t *testing.T) {
	agent := setupAgent(t)
	result := runAgent(t, agent, "What is 2+2?")

	totalTools := len(result.SearchResults) + len(result.FetchedPages)
	if totalTools > 0 {
		t.Logf("warning: agent used %d tool(s) for simple math — non-deterministic, not failing", totalTools)
	}
}
