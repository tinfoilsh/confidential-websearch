package api

import (
	"context"
	"time"

	"github.com/openai/openai-go/v3/option"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

const (
	MaxRequestBodySize = 200 << 20 // 200 MB
	RequestTimeout     = 2 * time.Minute
)

// Server holds all dependencies for the HTTP handlers
type Server struct {
	Cfg      *config.Config
	Client   *tinfoil.Client
	Pipeline *pipeline.Pipeline
}

// IncomingRequest represents the incoming chat request
type IncomingRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream"`
	Temperature *float64  `json:"temperature,omitempty"`
	MaxTokens   *int64    `json:"max_tokens,omitempty"`
}

// ReasoningSummaryPart represents a part of the reasoning summary
type ReasoningSummaryPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ReasoningItem represents a reasoning item for multi-turn context
type ReasoningItem struct {
	ID      string                 `json:"id"`
	Type    string                 `json:"type"`
	Summary []ReasoningSummaryPart `json:"summary"`
}

// Message represents a chat message in the incoming request
type Message struct {
	Role        string                `json:"role"`
	Content     string                `json:"content"`
	Annotations []pipeline.Annotation `json:"annotations,omitempty"`
}

// WebSearchCall represents a search operation in streaming output
type WebSearchCall struct {
	Type   string           `json:"type"`
	ID     string           `json:"id"`
	Status string           `json:"status"`
	Action *WebSearchAction `json:"action,omitempty"`
	Reason string           `json:"reason,omitempty"`
}

// StreamingDelta represents a delta in a streaming chunk
type StreamingDelta struct {
	Annotations     []pipeline.Annotation `json:"annotations,omitempty"`
	SearchReasoning string                `json:"search_reasoning,omitempty"`
	ReasoningItems  []ReasoningItem       `json:"reasoning_items,omitempty"`
}

// StreamingChoice represents a choice in a streaming chunk
type StreamingChoice struct {
	Index int64          `json:"index"`
	Delta StreamingDelta `json:"delta"`
}

// StreamingChunk represents a custom streaming chunk for annotations
type StreamingChunk struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []StreamingChoice `json:"choices"`
}

// FlatAnnotation represents a url_citation annotation (Responses API format)
type FlatAnnotation struct {
	Type          string `json:"type"`
	Title         string `json:"title"`
	URL           string `json:"url"`
	Content       string `json:"content,omitempty"`
	PublishedDate string `json:"published_date,omitempty"`
}

// WebSearchAction contains the search query details
type WebSearchAction struct {
	Type  string `json:"type"`
	Query string `json:"query"`
}

// ResponsesOutput represents the output array in Responses API
type ResponsesOutput struct {
	Type    string             `json:"type"`
	ID      string             `json:"id"`
	Status  string             `json:"status"`
	Role    string             `json:"role,omitempty"`
	Content []ResponsesContent `json:"content,omitempty"`
	Action  *WebSearchAction   `json:"action,omitempty"`
}

// ResponsesContent represents content in Responses API message output
type ResponsesContent struct {
	Type            string           `json:"type"`
	Text            string           `json:"text"`
	Annotations     []FlatAnnotation `json:"annotations,omitempty"`
	SearchReasoning string           `json:"search_reasoning,omitempty"`
	ReasoningItems  []ReasoningItem  `json:"reasoning_items,omitempty"`
}

// ResponsesRequest represents the incoming request for the Responses API
type ResponsesRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

// RequestContext holds common request processing state
type RequestContext struct {
	Ctx     context.Context
	Cancel  context.CancelFunc
	ReqOpts []option.RequestOption
}

// ChatCompletionResponse represents the non-streaming chat completion response
type ChatCompletionResponse struct {
	ID      string                       `json:"id"`
	Object  string                       `json:"object"`
	Created int64                        `json:"created"`
	Model   string                       `json:"model"`
	Choices []ChatCompletionChoiceOutput `json:"choices"`
	Usage   interface{}                  `json:"usage,omitempty"`
}

// ChatCompletionChoiceOutput represents a choice in the response
type ChatCompletionChoiceOutput struct {
	Index        int64                        `json:"index"`
	FinishReason string                       `json:"finish_reason"`
	Message      ChatCompletionMessageOutput  `json:"message"`
}

// ChatCompletionMessageOutput represents the assistant message with annotations
type ChatCompletionMessageOutput struct {
	Role            string                `json:"role"`
	Content         string                `json:"content"`
	Annotations     []pipeline.Annotation `json:"annotations,omitempty"`
	SearchReasoning string                `json:"search_reasoning,omitempty"`
	ReasoningItems  []ReasoningItem       `json:"reasoning_items,omitempty"`
}
