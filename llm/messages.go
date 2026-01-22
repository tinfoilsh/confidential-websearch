package llm

import (
	"fmt"

	"github.com/openai/openai-go/v3"

	"github.com/tinfoilsh/confidential-websearch/agent"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

// MessageBuilder constructs message arrays for the responder LLM
type MessageBuilder struct{}

// NewMessageBuilder creates a new MessageBuilder
func NewMessageBuilder() *MessageBuilder {
	return &MessageBuilder{}
}

// Build creates the message array for the responder LLM call
func (b *MessageBuilder) Build(inputMessages []pipeline.Message, searchResults []agent.ToolCall) []openai.ChatCompletionMessageParamUnion {
	var messages []openai.ChatCompletionMessageParamUnion

	for _, msg := range inputMessages {
		switch msg.Role {
		case "system":
			messages = append(messages, openai.SystemMessage(msg.Content))
		case "user":
			messages = append(messages, openai.UserMessage(msg.Content))
		case "assistant":
			if len(msg.Annotations) > 0 {
				messages = b.injectHistoricalSearchContext(messages, msg)
			} else {
				messages = append(messages, openai.AssistantMessage(msg.Content))
			}
		}
	}

	// If we have search results, include them as plain text context.
	// We avoid tool call format because the responder LLM doesn't have tools configured
	// and may try to echo/reproduce tool call syntax, breaking the response.
	if len(searchResults) > 0 {
		var resultsText string
		for _, tc := range searchResults {
			for i, sr := range tc.Results {
				resultsText += FormatSearchResult(i+1, sr.Title, sr.URL, sr.Content)
			}
		}
		messages = append(messages, openai.UserMessage("Search results:\n\n"+resultsText+"\nAnswer using these search results."))
	}

	return messages
}

// injectHistoricalSearchContext adds historical search context as plain text.
// We avoid using tool call format here because the responder LLM (which doesn't
// have tools configured) may try to echo/reproduce tool call syntax, breaking the response.
func (b *MessageBuilder) injectHistoricalSearchContext(messages []openai.ChatCompletionMessageParamUnion, msg pipeline.Message) []openai.ChatCompletionMessageParamUnion {
	var sourcesSummary string
	for i, ann := range msg.Annotations {
		if ann.Type == "url_citation" {
			sourcesSummary += fmt.Sprintf("[%d] %s (%s)\n", i+1, ann.URLCitation.Title, ann.URLCitation.URL)
		}
	}

	content := msg.Content
	if sourcesSummary != "" {
		content = fmt.Sprintf("%s\n\nSources used:\n%s", msg.Content, sourcesSummary)
	}

	messages = append(messages, openai.AssistantMessage(content))
	return messages
}

// FormatSearchResult formats a search result for inclusion in tool message content
func FormatSearchResult(index int, title, url, content string) string {
	return fmt.Sprintf("[%d] %s\nURL: %s\n%s\n\n", index, title, url, content)
}
