package llm

import (
	"strings"
	"testing"

	"github.com/tinfoilsh/confidential-websearch/agent"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
	"github.com/tinfoilsh/confidential-websearch/search"
)

func TestFormatSearchResult(t *testing.T) {
	result := FormatSearchResult(1, "Test Title", "https://example.com", "Test content here")

	expected := "[1] Test Title\nURL: https://example.com\nTest content here\n\n"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestBuildSimpleMessages(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there!"},
		{Role: "user", Content: "What is the weather?"},
	}

	result := builder.Build(messages, nil)

	if len(result) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(result))
	}

	// Check system message
	if result[0].OfSystem == nil {
		t.Error("expected system message at index 0")
	}

	// Check user messages
	if result[1].OfUser == nil {
		t.Error("expected user message at index 1")
	}

	// Check assistant message
	if result[2].OfAssistant == nil {
		t.Error("expected assistant message at index 2")
	}

	// Check last user message
	if result[3].OfUser == nil {
		t.Error("expected user message at index 3")
	}
}

func TestBuildWithSearchResults(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "user", Content: "What is the latest news?"},
	}

	searchResults := []agent.ToolCall{
		{
			ID:    "call_123",
			Query: "latest news today",
			Results: []search.Result{
				{
					Title:   "News Article 1",
					URL:     "https://news.example.com/1",
					Content: "Breaking news content",
				},
				{
					Title:   "News Article 2",
					URL:     "https://news.example.com/2",
					Content: "More news content",
				},
			},
		},
	}

	result := builder.Build(messages, searchResults)

	// Should have: user message + user message with search results
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}

	// Check user message
	if result[0].OfUser == nil {
		t.Error("expected user message at index 0")
	}

	// Check search results message (as user message with plain text)
	if result[1].OfUser == nil {
		t.Error("expected user message with search results at index 1")
	}
	content := result[1].OfUser.Content.OfString.Value
	if !strings.Contains(content, "Search results:") {
		t.Error("search results message should contain 'Search results:'")
	}
	if !strings.Contains(content, "News Article 1") {
		t.Error("search results message should contain first result")
	}
}

func TestBuildWithMultipleToolCalls(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "user", Content: "Compare weather in NYC and LA"},
	}

	searchResults := []agent.ToolCall{
		{
			ID:    "call_1",
			Query: "weather NYC",
			Results: []search.Result{
				{Title: "NYC Weather", URL: "https://weather.com/nyc", Content: "Cold"},
			},
		},
		{
			ID:    "call_2",
			Query: "weather LA",
			Results: []search.Result{
				{Title: "LA Weather", URL: "https://weather.com/la", Content: "Warm"},
			},
		},
	}

	result := builder.Build(messages, searchResults)

	// Should have: user message + user message with all search results combined
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}

	// Check search results contain both results
	content := result[1].OfUser.Content.OfString.Value
	if !strings.Contains(content, "NYC Weather") {
		t.Error("search results should contain NYC weather")
	}
	if !strings.Contains(content, "LA Weather") {
		t.Error("search results should contain LA weather")
	}
}

func TestBuildWithHistoricalAnnotations(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "user", Content: "What is the weather?"},
		{
			Role:            "assistant",
			Content:         "The weather is sunny.",
			SearchReasoning: "I searched for weather",
			Annotations: []pipeline.Annotation{
				{
					Type: "url_citation",
					URLCitation: pipeline.URLCitation{
						Title:   "Weather Report",
						URL:     "https://weather.com",
						Content: "Sunny skies expected",
					},
				},
			},
		},
		{Role: "user", Content: "What about tomorrow?"},
	}

	result := builder.Build(messages, nil)

	// Should have: user + assistant (with sources as text) + user
	if len(result) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(result))
	}

	// First message should be user
	if result[0].OfUser == nil {
		t.Error("expected user message at index 0")
	}

	// Second should be assistant with sources embedded in content
	if result[1].OfAssistant == nil {
		t.Error("expected assistant message at index 1")
	}
	content := result[1].OfAssistant.Content.OfString.Value
	if !strings.Contains(content, "The weather is sunny.") {
		t.Error("assistant content should contain original response")
	}
	if !strings.Contains(content, "Sources used:") {
		t.Error("assistant content should contain sources section")
	}
	if !strings.Contains(content, "Weather Report") {
		t.Error("assistant content should contain source title")
	}

	// Third should be user
	if result[2].OfUser == nil {
		t.Error("expected user message at index 2")
	}
}

func TestBuildWithEmptySearchResults(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "user", Content: "Hello"},
	}

	// Empty search results
	searchResults := []agent.ToolCall{}

	result := builder.Build(messages, searchResults)

	// Should only have the user message (no search results injection)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
}

func TestBuildWithNilSearchResults(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "user", Content: "Hello"},
	}

	result := builder.Build(messages, nil)

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
}

func TestSearchResultsContainFormattedResults(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "user", Content: "Search test"},
	}

	searchResults := []agent.ToolCall{
		{
			ID:    "call_1",
			Query: "test query",
			Results: []search.Result{
				{Title: "Result 1", URL: "https://example.com/1", Content: "Content 1"},
				{Title: "Result 2", URL: "https://example.com/2", Content: "Content 2"},
			},
		},
	}

	result := builder.Build(messages, searchResults)

	// Get the search results message (now a user message with plain text)
	if result[1].OfUser == nil {
		t.Fatal("expected user message with search results")
	}

	content := result[1].OfUser.Content.OfString.Value
	if !strings.Contains(content, "[1] Result 1") {
		t.Error("search results should contain first result")
	}
	if !strings.Contains(content, "[2] Result 2") {
		t.Error("search results should contain second result")
	}
	if !strings.Contains(content, "https://example.com/1") {
		t.Error("search results should contain first URL")
	}
}

func TestSearchResultsMessageContainsPrompt(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "user", Content: "Search for something"},
	}

	searchResults := []agent.ToolCall{
		{ID: "call_1", Query: "something", Results: []search.Result{{Title: "T", URL: "U", Content: "C"}}},
	}

	result := builder.Build(messages, searchResults)

	// Last message should contain both search results and the prompt
	lastMsg := result[len(result)-1]
	if lastMsg.OfUser == nil {
		t.Fatal("expected user message as last message")
	}

	content := lastMsg.OfUser.Content.OfString.Value
	if !strings.Contains(content, "Search results:") {
		t.Error("message should contain search results header")
	}
	if !strings.Contains(content, "Answer using these search results.") {
		t.Errorf("message should contain answer prompt, got %q", content)
	}
}
