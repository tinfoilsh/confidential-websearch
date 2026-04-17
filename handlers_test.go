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
	"github.com/tinfoilsh/confidential-websearch/tools"
)

type mockSearchProvider struct {
	results []search.Result
	err     error
}

func (m *mockSearchProvider) Search(ctx context.Context, query string, opts search.Options) ([]search.Result, error) {
	if m.err != nil {
		return nil, m.err
	}
	if opts.MaxResults < len(m.results) {
		return m.results[:opts.MaxResults], nil
	}
	return m.results, nil
}

func (m *mockSearchProvider) Name() string { return "mock" }

type mockFetcher struct {
	pages   []fetch.FetchedPage
	results []fetch.URLResult
}

type mockSafeguard struct {
	blocked map[string]string
}

func (m *mockSafeguard) Check(ctx context.Context, policy, content string) (*safeguard.CheckResult, error) {
	if reason, ok := m.blocked[content]; ok {
		return &safeguard.CheckResult{Violation: true, Rationale: reason}, nil
	}
	return &safeguard.CheckResult{}, nil
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

func (m *mockFetcher) FetchURLResults(ctx context.Context, urls []string) []fetch.URLResult {
	if m.results != nil {
		return m.results
	}

	results := make([]fetch.URLResult, 0, len(urls))
	for _, u := range urls {
		results = append(results, fetch.URLResult{
			URL:     u,
			Status:  fetch.FetchStatusCompleted,
			Content: "# Content from " + u,
		})
	}
	return results
}

func TestSearchHandler_Success(t *testing.T) {
	searcher := &mockSearchProvider{
		results: []search.Result{
			{Title: "Result 1", URL: "https://example.com/1", Content: "Content 1"},
			{Title: "Result 2", URL: "https://example.com/2", Content: "Content 2"},
		},
	}
	cfg := &config.Config{}
	svc := tools.NewService(searcher, nil, nil)
	handler := newSearchHandler(svc, cfg)

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
	svc := tools.NewService(&mockSearchProvider{}, nil, nil)
	handler := newSearchHandler(svc, &config.Config{})

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: ""})
	if err == nil {
		t.Fatal("expected error for empty query")
	}
}

func TestSearchHandler_SearchError(t *testing.T) {
	searcher := &mockSearchProvider{err: fmt.Errorf("search failed")}
	svc := tools.NewService(searcher, nil, nil)
	handler := newSearchHandler(svc, &config.Config{})

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "test"})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestSearchHandler_ForwardsMaxResults(t *testing.T) {
	var results []search.Result
	for i := 0; i < 25; i++ {
		results = append(results, search.Result{Title: fmt.Sprintf("Result %d", i)})
	}
	searcher := &mockSearchProvider{results: results}
	svc := tools.NewService(searcher, nil, nil)
	handler := newSearchHandler(svc, &config.Config{})

	_, result, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "test", MaxResults: 30})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Results) != 25 {
		t.Errorf("expected 25 results (all available), got %d", len(result.Results))
	}
}

func TestSearchHandler_PIICheckDisabled(t *testing.T) {
	searcher := &mockSearchProvider{results: []search.Result{{Title: "Result"}}}

	svc := tools.NewService(searcher, nil, nil)
	handler := newSearchHandler(svc, &config.Config{EnablePIICheck: false})
	_, result, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Results) != 1 {
		t.Errorf("expected 1 result, got %d", len(result.Results))
	}
}

func TestSearchHandler_InjectionCheckEnabled(t *testing.T) {
	searcher := &mockSearchProvider{
		results: []search.Result{
			{Title: "Safe", URL: "https://example.com/safe", Content: "safe"},
			{Title: "Unsafe", URL: "https://example.com/unsafe", Content: "unsafe"},
		},
	}
	svc := tools.NewService(searcher, nil, &mockSafeguard{
		blocked: map[string]string{"unsafe": "prompt injection detected"},
	})
	handler := newSearchHandler(svc, &config.Config{EnableInjectionCheck: true})

	_, result, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Results) != 1 {
		t.Fatalf("expected one filtered result, got %d", len(result.Results))
	}
	if result.Results[0].Title != "Safe" {
		t.Fatalf("expected safe result to remain, got %q", result.Results[0].Title)
	}
}

func TestFetchHandler_EmptyURLs(t *testing.T) {
	realFetcher := fetch.NewFetcher("test-account", "test-token")
	svc := tools.NewService(nil, realFetcher, nil)
	handler := newFetchHandler(svc, &config.Config{})

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, FetchArgs{URLs: []string{}})
	if err == nil {
		t.Fatal("expected error for empty URLs")
	}
}

func TestFetchHandler_PreservesPerURLResultsInOrder(t *testing.T) {
	svc := tools.NewService(nil, &mockFetcher{
		results: []fetch.URLResult{
			{URL: "https://example.com/a", Status: fetch.FetchStatusCompleted, Content: "# A"},
			{URL: "https://example.com/b", Status: fetch.FetchStatusFailed, Error: "blocked"},
			{URL: "https://example.com/c", Status: fetch.FetchStatusCompleted, Content: "# C"},
		},
	}, nil)
	handler := newFetchHandler(svc, &config.Config{})

	_, result, err := handler(context.Background(), &mcp.CallToolRequest{}, FetchArgs{
		URLs: []string{"https://example.com/a", "https://example.com/b", "https://example.com/c"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Results) != 3 {
		t.Fatalf("expected 3 per-url results, got %d", len(result.Results))
	}
	if result.Results[1].URL != "https://example.com/b" || result.Results[1].Status != fetch.FetchStatusFailed {
		t.Fatalf("expected second result to preserve failed URL ordering, got %+v", result.Results[1])
	}
	if len(result.Pages) != 2 {
		t.Fatalf("expected successful pages to remain available, got %d", len(result.Pages))
	}
	if result.Pages[0].URL != "https://example.com/a" || result.Pages[1].URL != "https://example.com/c" {
		t.Fatalf("expected successful pages to preserve order, got %+v", result.Pages)
	}
}

func TestMCPServer_ToolDiscovery(t *testing.T) {
	server := mcp.NewServer(&mcp.Implementation{
		Name:    "test-websearch",
		Version: "test",
	}, nil)

	searcher := &mockSearchProvider{results: []search.Result{{Title: "Test"}}}
	cfg := &config.Config{}
	svc := tools.NewService(searcher, fetch.NewFetcher("test", "test"), nil)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "search",
		Description: "Search the web",
	}, newSearchHandler(svc, cfg))

	mcp.AddTool(server, &mcp.Tool{
		Name:        "fetch",
		Description: "Fetch web pages",
	}, newFetchHandler(svc, cfg))

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
	svc := tools.NewService(searcher, nil, nil)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "search",
		Description: "Search the web",
	}, newSearchHandler(svc, cfg))

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
