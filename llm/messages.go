package llm

import (
	"encoding/json"
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
			messages = append(messages, openai.SystemMessage(contentToString(msg.Content)))
		case "user":
			messages = append(messages, buildUserMessage(msg.Content))
		case "assistant":
			if len(msg.Annotations) > 0 {
				messages = b.injectHistoricalSearchContext(messages, msg)
			} else {
				messages = append(messages, openai.AssistantMessage(contentToString(msg.Content)))
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
		messages = append(messages, openai.UserMessage("Search results:\n\n"+resultsText+"\nUse these search results to answer. When you use information from a source and you think it's important to cite the provided source, cite it using lenticular brackets like 【1】, 【2】, etc. Do not overuse citations."))
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

	content := contentToString(msg.Content)
	if sourcesSummary != "" {
		content = fmt.Sprintf("%s\n\nSources used:\n%s", content, sourcesSummary)
	}

	messages = append(messages, openai.AssistantMessage(content))
	return messages
}

// FormatSearchResult formats a search result for inclusion in tool message content
func FormatSearchResult(index int, title, url, content string) string {
	return fmt.Sprintf("【%d】%s\nURL: %s\n%s\n\n", index, title, url, content)
}

// contentToString extracts text from Content for system/assistant messages.
func contentToString(content json.RawMessage) string {
	return pipeline.ExtractTextContent(content)
}

// buildUserMessage creates a user message, passing content through verbatim via SDK unmarshalling.
func buildUserMessage(content json.RawMessage) openai.ChatCompletionMessageParamUnion {
	var contentUnion openai.ChatCompletionUserMessageParamContentUnion
	if err := json.Unmarshal(content, &contentUnion); err != nil {
		return openai.UserMessage("")
	}
	return openai.ChatCompletionMessageParamUnion{
		OfUser: &openai.ChatCompletionUserMessageParam{
			Content: contentUnion,
		},
	}
}
