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

func TestBuildWithAgentResults(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "user", Content: "What is the latest news?"},
	}

	agentResult := &agent.Result{
		AgentReasoning: "I should search for news",
		ToolCalls: []agent.ToolCall{
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
		},
	}

	result := builder.Build(messages, agentResult)

	// Should have: user message + assistant (tool calls) + tool result + user prompt
	if len(result) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(result))
	}

	// Check user message
	if result[0].OfUser == nil {
		t.Error("expected user message at index 0")
	}

	// Check assistant message with tool calls
	if result[1].OfAssistant == nil {
		t.Error("expected assistant message with tool calls at index 1")
	}
	if len(result[1].OfAssistant.ToolCalls) != 1 {
		t.Error("expected 1 tool call")
	}

	// Check tool result message
	if result[2].OfTool == nil {
		t.Error("expected tool message at index 2")
	}

	// Check final user prompt
	if result[3].OfUser == nil {
		t.Error("expected user message at index 3")
	}
}

func TestBuildWithMultipleToolCalls(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "user", Content: "Compare weather in NYC and LA"},
	}

	agentResult := &agent.Result{
		ToolCalls: []agent.ToolCall{
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
		},
	}

	result := builder.Build(messages, agentResult)

	// Should have: user message + assistant (2 tool calls) + tool result 1 + tool result 2 + user prompt
	if len(result) != 5 {
		t.Fatalf("expected 5 messages, got %d", len(result))
	}

	// Check assistant has 2 tool calls
	if result[1].OfAssistant == nil || len(result[1].OfAssistant.ToolCalls) != 2 {
		t.Error("expected 2 tool calls in assistant message")
	}

	// Check both tool results
	if result[2].OfTool == nil {
		t.Error("expected first tool message at index 2")
	}
	if result[3].OfTool == nil {
		t.Error("expected second tool message at index 3")
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

	// Should have: user + assistant(tool call) + tool result + assistant response + user
	if len(result) != 5 {
		t.Fatalf("expected 5 messages, got %d", len(result))
	}

	// First message should be user
	if result[0].OfUser == nil {
		t.Error("expected user message at index 0")
	}

	// Second should be assistant with tool calls (historical)
	if result[1].OfAssistant == nil {
		t.Error("expected assistant message at index 1")
	}
	if len(result[1].OfAssistant.ToolCalls) != 1 {
		t.Error("expected 1 tool call for historical context")
	}

	// Third should be tool result
	if result[2].OfTool == nil {
		t.Error("expected tool message at index 2")
	}

	// Fourth should be assistant response
	if result[3].OfAssistant == nil {
		t.Error("expected assistant message at index 3")
	}

	// Fifth should be user
	if result[4].OfUser == nil {
		t.Error("expected user message at index 4")
	}
}

func TestBuildWithEmptyAgentResult(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "user", Content: "Hello"},
	}

	// Empty agent result (no tool calls)
	agentResult := &agent.Result{
		AgentReasoning: "No search needed",
		ToolCalls:      []agent.ToolCall{},
	}

	result := builder.Build(messages, agentResult)

	// Should only have the user message (no tool call injection)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
}

func TestBuildWithNilAgentResult(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "user", Content: "Hello"},
	}

	result := builder.Build(messages, nil)

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
}

func TestToolResultContainsSearchResults(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "user", Content: "Search test"},
	}

	agentResult := &agent.Result{
		ToolCalls: []agent.ToolCall{
			{
				ID:    "call_1",
				Query: "test query",
				Results: []search.Result{
					{Title: "Result 1", URL: "https://example.com/1", Content: "Content 1"},
					{Title: "Result 2", URL: "https://example.com/2", Content: "Content 2"},
				},
			},
		},
	}

	result := builder.Build(messages, agentResult)

	// Get the tool result message
	toolMsg := result[2].OfTool
	if toolMsg == nil {
		t.Fatal("expected tool message")
	}

	content := toolMsg.Content.OfString.Value
	if !strings.Contains(content, "[1] Result 1") {
		t.Error("tool content should contain first result")
	}
	if !strings.Contains(content, "[2] Result 2") {
		t.Error("tool content should contain second result")
	}
	if !strings.Contains(content, "https://example.com/1") {
		t.Error("tool content should contain first URL")
	}
}

func TestFinalUserPromptAddedWithSearchResults(t *testing.T) {
	builder := NewMessageBuilder()

	messages := []pipeline.Message{
		{Role: "user", Content: "Search for something"},
	}

	agentResult := &agent.Result{
		ToolCalls: []agent.ToolCall{
			{ID: "call_1", Query: "something", Results: []search.Result{{Title: "T", URL: "U", Content: "C"}}},
		},
	}

	result := builder.Build(messages, agentResult)

	// Last message should be the prompt to use search results
	lastMsg := result[len(result)-1]
	if lastMsg.OfUser == nil {
		t.Fatal("expected user message as last message")
	}

	content := lastMsg.OfUser.Content.OfString.Value
	if content != "Answer using the search results above." {
		t.Errorf("expected final prompt, got %q", content)
	}
}
