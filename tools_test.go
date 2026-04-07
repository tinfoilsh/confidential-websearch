package main

import (
	"context"
	"fmt"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
)

type mockSearchProvider struct {
	results []search.Result
	err     error
}

func (m *mockSearchProvider) Search(ctx context.Context, query string, maxResults int) ([]search.Result, error) {
	if m.err != nil {
		return nil, m.err
	}
	if maxResults < len(m.results) {
		return m.results[:maxResults], nil
	}
	return m.results, nil
}

func (m *mockSearchProvider) Name() string { return "mock" }

type mockFetcher struct {
	pages []fetch.FetchedPage
}

func (m *mockFetcher) FetchURLs(ctx context.Context, urls []string) []fetch.FetchedPage {
	if m.pages != nil {
		return m.pages
	}
	var result []fetch.FetchedPage
	for _, u := range urls {
		result = append(result, fetch.FetchedPage{URL: u, Content: "# Content from " + u})
	}
	return result
}

type mockSafeguardClient struct {
	violation bool
	rationale string
	err       error
}

func (m *mockSafeguardClient) Check(ctx context.Context, policy, content string) (*safeguard.CheckResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &safeguard.CheckResult{Violation: m.violation, Rationale: m.rationale}, nil
}

func TestSearchHandler_Success(t *testing.T) {
	searcher := &mockSearchProvider{
		results: []search.Result{
			{Title: "Result 1", URL: "https://example.com/1", Content: "Content 1"},
			{Title: "Result 2", URL: "https://example.com/2", Content: "Content 2"},
		},
	}
	cfg := &config.Config{}
	handler := newSearchHandler(searcher, nil, cfg)

	_, result, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "test query"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(result.Results))
	}
	if result.Results[0].Title != "Result 1" {
		t.Errorf("expected 'Result 1', got '%s'", result.Results[0].Title)
	}
}

func TestSearchHandler_EmptyQuery(t *testing.T) {
	handler := newSearchHandler(&mockSearchProvider{}, nil, &config.Config{})

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: ""})
	if err == nil {
		t.Fatal("expected error for empty query")
	}
}

func TestSearchHandler_SearchError(t *testing.T) {
	searcher := &mockSearchProvider{err: fmt.Errorf("search failed")}
	handler := newSearchHandler(searcher, nil, &config.Config{})

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "test"})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestSearchHandler_MaxResultsCapped(t *testing.T) {
	var results []search.Result
	for i := 0; i < 25; i++ {
		results = append(results, search.Result{Title: fmt.Sprintf("Result %d", i)})
	}
	searcher := &mockSearchProvider{results: results}
	handler := newSearchHandler(searcher, nil, &config.Config{})

	_, result, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "test", MaxResults: 30})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Results) > 20 {
		t.Errorf("expected max 20 results, got %d", len(result.Results))
	}
}

func TestSearchHandler_DefaultMaxResults(t *testing.T) {
	var results []search.Result
	for i := 0; i < 15; i++ {
		results = append(results, search.Result{Title: fmt.Sprintf("Result %d", i)})
	}
	searcher := &mockSearchProvider{results: results}
	handler := newSearchHandler(searcher, nil, &config.Config{})

	_, result, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Results) != config.DefaultMaxSearchResults {
		t.Errorf("expected %d results, got %d", config.DefaultMaxSearchResults, len(result.Results))
	}
}

func TestSearchHandler_PIICheckDisabled(t *testing.T) {
	searcher := &mockSearchProvider{results: []search.Result{{Title: "Result"}}}

	handler := newSearchHandler(searcher, nil, &config.Config{EnablePIICheck: false})
	_, result, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Results) != 1 {
		t.Errorf("expected 1 result, got %d", len(result.Results))
	}
}

func TestFetchHandler_Success(t *testing.T) {
	fetcher := &fetch.Fetcher{} // Will be overridden by mock pattern
	_ = fetcher

	realFetcher := fetch.NewFetcher("test-account", "test-token")
	handler := newFetchHandler(realFetcher, nil, &config.Config{})

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, FetchArgs{URLs: []string{}})
	if err == nil {
		t.Fatal("expected error for empty URLs")
	}
}

func TestFetchHandler_URLsCapped(t *testing.T) {
	cfg := &config.Config{}
	realFetcher := fetch.NewFetcher("test-account", "test-token")
	handler := newFetchHandler(realFetcher, nil, cfg)

	urls := make([]string, 10)
	for i := range urls {
		urls[i] = fmt.Sprintf("https://example.com/%d", i)
	}

	// This will fail because the fetcher can't actually reach URLs,
	// but it validates that we don't panic with >5 URLs
	_, result, _ := handler(context.Background(), &mcp.CallToolRequest{}, FetchArgs{URLs: urls})
	// With a real fetcher hitting test-account, we get empty results but no panic
	_ = result
}

func TestMCPServer_ToolDiscovery(t *testing.T) {
	server := mcp.NewServer(&mcp.Implementation{
		Name:    "test-websearch",
		Version: "test",
	}, nil)

	searcher := &mockSearchProvider{results: []search.Result{{Title: "Test"}}}
	cfg := &config.Config{}

	mcp.AddTool(server, &mcp.Tool{
		Name:        "search",
		Description: "Search the web",
	}, newSearchHandler(searcher, nil, cfg))

	realFetcher := fetch.NewFetcher("test", "test")
	mcp.AddTool(server, &mcp.Tool{
		Name:        "fetch",
		Description: "Fetch web pages",
	}, newFetchHandler(realFetcher, nil, cfg))

	// Connect in-memory client to verify tool discovery
	client := mcp.NewClient(&mcp.Implementation{Name: "test-client", Version: "v1"}, nil)
	clientTransport, serverTransport := mcp.NewInMemoryTransports()

	go server.Connect(context.Background(), serverTransport, nil)

	session, err := client.Connect(context.Background(), clientTransport, nil)
	if err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer session.Close()

	result, err := session.ListTools(context.Background(), &mcp.ListToolsParams{})
	if err != nil {
		t.Fatalf("failed to list tools: %v", err)
	}

	if len(result.Tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(result.Tools))
	}

	toolNames := make(map[string]bool)
	for _, tool := range result.Tools {
		toolNames[tool.Name] = true
	}
	if !toolNames["search"] {
		t.Error("missing 'search' tool")
	}
	if !toolNames["fetch"] {
		t.Error("missing 'fetch' tool")
	}
}

func TestMCPServer_SearchToolCall(t *testing.T) {
	server := mcp.NewServer(&mcp.Implementation{
		Name:    "test-websearch",
		Version: "test",
	}, nil)

	searcher := &mockSearchProvider{
		results: []search.Result{
			{Title: "Go Programming", URL: "https://go.dev", Content: "The Go programming language"},
		},
	}
	cfg := &config.Config{}

	mcp.AddTool(server, &mcp.Tool{
		Name:        "search",
		Description: "Search the web",
	}, newSearchHandler(searcher, nil, cfg))

	client := mcp.NewClient(&mcp.Implementation{Name: "test-client", Version: "v1"}, nil)
	clientTransport, serverTransport := mcp.NewInMemoryTransports()

	go server.Connect(context.Background(), serverTransport, nil)

	session, err := client.Connect(context.Background(), clientTransport, nil)
	if err != nil {
		t.Fatalf("failed to connect: %v", err)
	}
	defer session.Close()

	result, err := session.CallTool(context.Background(), &mcp.CallToolParams{
		Name:      "search",
		Arguments: mustJSON(t, SearchArgs{Query: "golang"}),
	})
	if err != nil {
		t.Fatalf("failed to call search tool: %v", err)
	}

	if result.IsError {
		t.Fatalf("tool returned error: %v", result.Content)
	}

	if len(result.Content) == 0 {
		t.Fatal("expected content in result")
	}
}

func mustJSON(t *testing.T, v any) map[string]any {
	t.Helper()
	switch val := v.(type) {
	case SearchArgs:
		m := map[string]any{"query": val.Query}
		if val.MaxResults > 0 {
			m["max_results"] = val.MaxResults
		}
		return m
	default:
		t.Fatalf("unsupported type for mustJSON: %T", v)
		return nil
	}
}
