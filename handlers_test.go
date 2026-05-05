package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
	"github.com/tinfoilsh/confidential-websearch/tools"
)

type mockSearchProvider struct {
	results  []search.Result
	err      error
	lastOpts search.Options
}

func (m *mockSearchProvider) Search(ctx context.Context, query string, opts search.Options) ([]search.Result, error) {
	m.lastOpts = opts
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
	handler := newSearchHandler(svc, cfg, nil)

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
	handler := newSearchHandler(svc, &config.Config{}, nil)

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: ""})
	if err == nil {
		t.Fatal("expected error for empty query")
	}
}

func TestSearchHandler_SearchError(t *testing.T) {
	searcher := &mockSearchProvider{err: fmt.Errorf("search failed")}
	svc := tools.NewService(searcher, nil, nil)
	handler := newSearchHandler(svc, &config.Config{}, nil)

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
	handler := newSearchHandler(svc, &config.Config{}, nil)

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
	handler := newSearchHandler(svc, &config.Config{EnablePIICheck: false}, nil)
	_, result, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Results) != 1 {
		t.Errorf("expected 1 result, got %d", len(result.Results))
	}
}

func TestSearchHandler_InjectionCheckSkippedForSearchResults(t *testing.T) {
	searcher := &mockSearchProvider{
		results: []search.Result{
			{Title: "Safe", URL: "https://example.com/safe", Content: "safe"},
			{Title: "Unsafe", URL: "https://example.com/unsafe", Content: "unsafe"},
		},
	}
	svc := tools.NewService(searcher, nil, &mockSafeguard{
		blocked: map[string]string{"unsafe": "prompt injection detected"},
	})
	handler := newSearchHandler(svc, &config.Config{EnableInjectionCheck: true}, nil)

	_, result, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Results) != 2 {
		t.Fatalf("expected search results to pass through unfiltered, got %d", len(result.Results))
	}
}

func TestSearchHandler_ForwardsUserLocationAndAllowedDomains(t *testing.T) {
	searcher := &mockSearchProvider{results: []search.Result{{Title: "r"}}}
	svc := tools.NewService(searcher, nil, nil)
	handler := newSearchHandler(svc, &config.Config{}, nil)

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{
		Query:               "news",
		UserLocationCountry: " gb ",
		AllowedDomains:      []string{"BBC.co.uk", "bbc.co.uk", " theguardian.com "},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if searcher.lastOpts.UserLocationCountry != "GB" {
		t.Fatalf("expected user_location_country GB, got %q", searcher.lastOpts.UserLocationCountry)
	}
	expected := []string{"bbc.co.uk", "theguardian.com"}
	if !stringSlicesEqual(searcher.lastOpts.AllowedDomains, expected) {
		t.Fatalf("expected allowed_domains %v, got %v", expected, searcher.lastOpts.AllowedDomains)
	}
}

func TestSearchHandler_ForwardsContentModeAndMaxContentChars(t *testing.T) {
	searcher := &mockSearchProvider{results: []search.Result{{Title: "r"}}}
	svc := tools.NewService(searcher, nil, nil)
	handler := newSearchHandler(svc, &config.Config{}, nil)

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{
		Query:           "test",
		ContentMode:     "text",
		MaxContentChars: 2500,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if searcher.lastOpts.ContentMode != search.ContentModeText {
		t.Fatalf("expected content_mode=text forwarded, got %q", searcher.lastOpts.ContentMode)
	}
	if searcher.lastOpts.MaxContentCharacters != 2500 {
		t.Fatalf("expected max_content_chars=2500 forwarded, got %d", searcher.lastOpts.MaxContentCharacters)
	}
}

func TestSearchHandler_ContentModeDefault(t *testing.T) {
	searcher := &mockSearchProvider{results: []search.Result{{Title: "r"}}}
	svc := tools.NewService(searcher, nil, nil)
	handler := newSearchHandler(svc, &config.Config{}, nil)

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if searcher.lastOpts.ContentMode != search.ContentModeHighlights {
		t.Fatalf("expected default content_mode=highlights, got %q", searcher.lastOpts.ContentMode)
	}
	if searcher.lastOpts.MaxContentCharacters != 700 {
		t.Fatalf("expected default max_content_chars=700, got %d", searcher.lastOpts.MaxContentCharacters)
	}
}

func TestSearchHandler_ForwardsNewExaKnobs(t *testing.T) {
	searcher := &mockSearchProvider{results: []search.Result{{Title: "r"}}}
	svc := tools.NewService(searcher, nil, nil)
	handler := newSearchHandler(svc, &config.Config{}, nil)

	zero := 0
	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{
		Query:              "test",
		ExcludedDomains:    []string{"Aggregator.Example", " spam.example "},
		Category:           "news",
		StartPublishedDate: "2024-01-01",
		EndPublishedDate:   "2024-12-31T23:59:59Z",
		MaxAgeHours:        &zero,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []string{"aggregator.example", "spam.example"}
	if !stringSlicesEqual(searcher.lastOpts.ExcludedDomains, expected) {
		t.Errorf("expected excluded_domains %v, got %v", expected, searcher.lastOpts.ExcludedDomains)
	}
	if searcher.lastOpts.Category != search.CategoryNews {
		t.Errorf("expected category 'news', got %q", searcher.lastOpts.Category)
	}
	if searcher.lastOpts.StartPublishedDate != "2024-01-01T00:00:00Z" {
		t.Errorf("expected normalized start date, got %q", searcher.lastOpts.StartPublishedDate)
	}
	if searcher.lastOpts.EndPublishedDate != "2024-12-31T23:59:59Z" {
		t.Errorf("expected end date preserved, got %q", searcher.lastOpts.EndPublishedDate)
	}
	if searcher.lastOpts.MaxAgeHours == nil || *searcher.lastOpts.MaxAgeHours != 0 {
		t.Errorf("expected max_age_hours=0 forwarded, got %v", searcher.lastOpts.MaxAgeHours)
	}
}

func TestSearchHandler_InvalidCategory(t *testing.T) {
	svc := tools.NewService(&mockSearchProvider{}, nil, nil)
	handler := newSearchHandler(svc, &config.Config{}, nil)

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{
		Query:    "test",
		Category: "sports",
	})
	if err == nil {
		t.Fatal("expected error for invalid category")
	}
}

func TestSearchHandler_InvalidDate(t *testing.T) {
	svc := tools.NewService(&mockSearchProvider{}, nil, nil)
	handler := newSearchHandler(svc, &config.Config{}, nil)

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{
		Query:              "test",
		StartPublishedDate: "not-a-date",
	})
	if err == nil {
		t.Fatal("expected error for invalid start_published_date")
	}
}

func TestSearchHandler_CategoryIncompatibleFilters(t *testing.T) {
	svc := tools.NewService(&mockSearchProvider{}, nil, nil)
	handler := newSearchHandler(svc, &config.Config{}, nil)

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{
		Query:              "test",
		Category:           "company",
		StartPublishedDate: "2024-01-01",
	})
	if err == nil {
		t.Fatal("expected error: company category does not support date filters")
	}

	_, _, err = handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{
		Query:           "test",
		Category:        "people",
		ExcludedDomains: []string{"foo.example"},
	})
	if err == nil {
		t.Fatal("expected error: people category does not support excluded_domains")
	}
}

func TestSearchHandler_InvalidContentMode(t *testing.T) {
	svc := tools.NewService(&mockSearchProvider{}, nil, nil)
	handler := newSearchHandler(svc, &config.Config{}, nil)

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{
		Query:       "test",
		ContentMode: "summary",
	})
	if err == nil {
		t.Fatal("expected error for invalid content_mode")
	}
}

func TestFetchHandler_EmptyURLs(t *testing.T) {
	realFetcher := fetch.NewFetcher("test-account", "test-token")
	svc := tools.NewService(nil, realFetcher, nil)
	handler := newFetchHandler(svc, &config.Config{}, nil)

	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, FetchArgs{URLs: []string{}})
	if err == nil {
		t.Fatal("expected error for empty URLs")
	}
}

func TestFetchHandler_AllowedDomainsRejectsOutsideHosts(t *testing.T) {
	svc := tools.NewService(nil, &mockFetcher{
		results: []fetch.URLResult{
			{URL: "https://docs.python.org/3/tutorial", Status: fetch.FetchStatusCompleted, Content: "# Tutorial"},
		},
	}, nil)
	handler := newFetchHandler(svc, &config.Config{}, nil)

	_, result, err := handler(context.Background(), &mcp.CallToolRequest{}, FetchArgs{
		URLs: []string{
			"https://docs.python.org/3/tutorial",
			"https://evil.example.com/takeover",
		},
		AllowedDomains: []string{"python.org"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Results) != 2 {
		t.Fatalf("expected 2 per-url results, got %d", len(result.Results))
	}
	statusByURL := map[string]string{}
	for _, r := range result.Results {
		statusByURL[r.URL] = r.Status
	}
	if statusByURL["https://docs.python.org/3/tutorial"] != fetch.FetchStatusCompleted {
		t.Fatalf("expected allowed URL to be fetched, got %+v", statusByURL)
	}
	if statusByURL["https://evil.example.com/takeover"] != fetch.FetchStatusFailed {
		t.Fatalf("expected out-of-domain URL to be rejected, got %+v", statusByURL)
	}
}

func TestFetchHandler_AllowedDomainsAllRejected(t *testing.T) {
	svc := tools.NewService(nil, &mockFetcher{}, nil)
	handler := newFetchHandler(svc, &config.Config{}, nil)

	_, result, err := handler(context.Background(), &mcp.CallToolRequest{}, FetchArgs{
		URLs:           []string{"https://evil.example.com/takeover"},
		AllowedDomains: []string{"python.org"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Results) != 1 || result.Results[0].Status != fetch.FetchStatusFailed {
		t.Fatalf("expected single rejected result, got %+v", result.Results)
	}
	if len(result.Pages) != 0 {
		t.Fatalf("expected no pages when all URLs rejected, got %d", len(result.Pages))
	}
}

func TestHostAllowed(t *testing.T) {
	cases := []struct {
		url      string
		allowed  []string
		expected bool
	}{
		{"https://python.org/about", []string{"python.org"}, true},
		{"https://docs.python.org/tutorial", []string{"python.org"}, true},
		{"https://python.org.attacker.com", []string{"python.org"}, false},
		{"https://evil.example.com", []string{"python.org"}, false},
		{"https://api.bbc.co.uk/news", []string{"bbc.co.uk"}, true},
		{"not a url", []string{"python.org"}, false},
	}
	for _, tc := range cases {
		got := hostAllowed(tc.url, tc.allowed)
		if got != tc.expected {
			t.Errorf("hostAllowed(%q, %v) = %v, want %v", tc.url, tc.allowed, got, tc.expected)
		}
	}
}

func TestResolveSafetyFlag(t *testing.T) {
	cases := []struct {
		name     string
		header   string
		fallback bool
		want     bool
	}{
		{"absent header uses fallback true", "", true, true},
		{"absent header uses fallback false", "", false, false},
		{"header true overrides fallback false", "true", false, true},
		{"header false overrides fallback true", "false", true, false},
		{"header 1 overrides fallback false", "1", false, true},
		{"header 0 overrides fallback true", "0", true, false},
		{"malformed header falls back", "maybe", true, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/mcp", nil)
			if tc.header != "" {
				req.Header.Set(headerPIICheck, tc.header)
			}
			got := resolveSafetyFlag(req, headerPIICheck, tc.fallback)
			if got != tc.want {
				t.Errorf("resolveSafetyFlag header=%q fallback=%v = %v, want %v", tc.header, tc.fallback, got, tc.want)
			}
		})
	}
}

func TestSearchHandler_HeaderOverridesEnvDefaults(t *testing.T) {
	searcher := &mockSearchProvider{results: []search.Result{{Title: "r", URL: "https://example.com/r", Content: "ok"}}}
	sg := &mockSafeguard{blocked: map[string]string{"john@example.com": "email detected"}}
	svc := tools.NewService(searcher, nil, sg)

	req := httptest.NewRequest(http.MethodPost, "/mcp", nil)
	req.Header.Set(headerPIICheck, "true")
	req.Header.Set(headerInjectionCheck, "false")

	handler := newSearchHandler(svc, &config.Config{EnablePIICheck: false, EnableInjectionCheck: true}, req)
	_, _, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "john@example.com"})
	if err == nil {
		t.Fatalf("expected PII block when header opts PII check on; got nil error")
	}
}

func TestSearchHandler_AbsentHeadersFallBackToEnv(t *testing.T) {
	searcher := &mockSearchProvider{results: []search.Result{{Title: "r", URL: "https://example.com/r", Content: "ok"}}}
	svc := tools.NewService(searcher, nil, &mockSafeguard{
		blocked: map[string]string{"john@example.com": "email detected"},
	})

	req := httptest.NewRequest(http.MethodPost, "/mcp", nil)

	handler := newSearchHandler(svc, &config.Config{EnablePIICheck: false, EnableInjectionCheck: false}, req)
	_, result, err := handler(context.Background(), &mcp.CallToolRequest{}, SearchArgs{Query: "john@example.com"})
	if err != nil {
		t.Fatalf("unexpected error when both env defaults off and no header present: %v", err)
	}
	if len(result.Results) != 1 {
		t.Fatalf("expected 1 result when PII check disabled, got %d", len(result.Results))
	}
}

func stringSlicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestFetchHandler_PreservesPerURLResultsInOrder(t *testing.T) {
	svc := tools.NewService(nil, &mockFetcher{
		results: []fetch.URLResult{
			{URL: "https://example.com/a", Status: fetch.FetchStatusCompleted, Content: "# A"},
			{URL: "https://example.com/b", Status: fetch.FetchStatusFailed, Error: "blocked"},
			{URL: "https://example.com/c", Status: fetch.FetchStatusCompleted, Content: "# C"},
		},
	}, nil)
	handler := newFetchHandler(svc, &config.Config{}, nil)

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
	}, newSearchHandler(svc, cfg, nil))

	mcp.AddTool(server, &mcp.Tool{
		Name:        "fetch",
		Description: "Fetch web pages",
	}, newFetchHandler(svc, cfg, nil))

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
	}, newSearchHandler(svc, cfg, nil))

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
