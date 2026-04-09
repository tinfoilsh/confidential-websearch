package api

import (
	"encoding/json"
	"sync"
	"time"

	"github.com/openai/openai-go/v3"

	"github.com/tinfoilsh/confidential-websearch/engine"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

const (
	MaxRequestBodySize = 20 << 20 // 20 MB

	// Object types
	ObjectChatCompletionChunk = "chat.completion.chunk"
	ObjectResponse            = "response"

	// Item types
	ItemTypeWebSearchCall = "web_search_call"
	ItemTypeMessage       = "message"

	// Content types
	ContentTypeOutputText  = "output_text"
	ContentTypeURLCitation = "url_citation"

	// Action types
	ActionTypeSearch   = "search"
	ActionTypeOpenPage = "open_page"

	// Status values
	StatusInProgress = "in_progress"
	StatusCompleted  = "completed"
	StatusFailed     = "failed"
	StatusBlocked    = "blocked"

	// Roles
	RoleAssistant = "assistant"

	// Finish reasons
	FinishReasonStop = "stop"

	// ID prefixes
	IDPrefixWebSearch = "ws_"
	IDPrefixResponse  = "resp_"
	IDPrefixMessage   = "msg_"

	responseContinuationTTL      = 24 * time.Hour
	responseContinuationMaxSize  = 100_000
	responseContinuationPruneInt = 10 * time.Minute
)

// Server holds all dependencies for the HTTP handlers
type Server struct {
	Runner                       engine.Runner
	DefaultPIICheckEnabled       bool
	DefaultInjectionCheckEnabled bool

	responseStoreOnce sync.Once
	responseStore     *responseContinuationStore
}

type responseContinuationStore struct {
	mu        sync.RWMutex
	entries   map[string]responseContinuationEntry
	stopPrune chan struct{}
}

type responseContinuationEntry struct {
	upstreamID string
	expiresAt  time.Time
}

// WebSearchOptions enables the custom web search tool
type WebSearchOptions struct {
	SearchContextSize string            `json:"search_context_size,omitempty"` // "low", "medium", "high"
	UserLocation      *UserLocation     `json:"user_location,omitempty"`
	Filters           *WebSearchFilters `json:"filters,omitempty"`
}

// UserLocation provides location context for web search
type UserLocation struct {
	Type        string               `json:"type,omitempty"` // "approximate"
	Approximate *ApproximateLocation `json:"approximate,omitempty"`
}

// ApproximateLocation contains approximate location details
type ApproximateLocation struct {
	Country string `json:"country,omitempty"`
	City    string `json:"city,omitempty"`
	Region  string `json:"region,omitempty"`
}

// PIICheckOptions enables PII checking when present
type PIICheckOptions struct{}

// InjectionCheckOptions enables injection checking when present
type InjectionCheckOptions struct{}

// IncomingRequest represents the incoming chat request
type IncomingRequest struct {
	Model                 string                 `json:"model"`
	Messages              []Message              `json:"messages"`
	Stream                bool                   `json:"stream"`
	Temperature           *float64               `json:"temperature,omitempty"`
	MaxTokens             *int64                 `json:"max_tokens,omitempty"`
	WebSearchOptions      *WebSearchOptions      `json:"web_search_options,omitempty"`
	PIICheckOptions       *PIICheckOptions       `json:"pii_check_options,omitempty"`
	InjectionCheckOptions *InjectionCheckOptions `json:"injection_check_options,omitempty"`
}

// Message represents a chat message in the incoming request
type Message struct {
	Role        string                `json:"role"`
	Content     json.RawMessage       `json:"content"` // Preserved verbatim for responder
	Annotations []pipeline.Annotation `json:"annotations,omitempty"`
}

// WebSearchCall represents a custom streaming event for search status.
// Uses chat.completion.chunk envelope so SDKs don't fail parsing, but the
// content (type, status, action) is a custom extension not in OpenAI's spec.
type WebSearchCall struct {
	Type    string           `json:"type"`
	ID      string           `json:"id"`
	Status  string           `json:"status"`
	Action  *WebSearchAction `json:"action,omitempty"`
	Reason  string           `json:"reason,omitempty"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []any            `json:"choices"`
}

// StreamingDelta represents a delta in a streaming chunk
type StreamingDelta struct {
	Annotations     []pipeline.Annotation `json:"annotations,omitempty"`
	SearchReasoning string                `json:"search_reasoning,omitempty"`
}

// StreamingChoice represents a choice in a streaming chunk
type StreamingChoice struct {
	Index int64          `json:"index"`
	Delta StreamingDelta `json:"delta"`
}

// StreamingChunk represents a custom streaming chunk for annotations.
// This is an extension - OpenAI doesn't stream annotations in Chat Completions.
type StreamingChunk struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []StreamingChoice `json:"choices"`
}

// FlatAnnotation represents a url_citation annotation (Responses API format)
type FlatAnnotation struct {
	Type       string `json:"type"`
	URL        string `json:"url"`
	Title      string `json:"title"`
	StartIndex int    `json:"start_index"`
	EndIndex   int    `json:"end_index"`
}

// WebSearchAction contains the search query details
type WebSearchAction struct {
	Type  string `json:"type"`
	Query string `json:"query,omitempty"`
	URL   string `json:"url,omitempty"`
}

// ResponsesOutput represents the output array in Responses API
type ResponsesOutput struct {
	Type    string             `json:"type"`
	ID      string             `json:"id"`
	Status  string             `json:"status"`
	Reason  string             `json:"reason,omitempty"`
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
}

// WebSearchFilters restricts search results to specific domains (OpenAI-compatible)
type WebSearchFilters struct {
	AllowedDomains []string `json:"allowed_domains,omitempty"`
}

// ResponsesTool represents a tool in the Responses API request
type ResponsesTool struct {
	Type              string            `json:"type"` // "web_search"
	SearchContextSize string            `json:"search_context_size,omitempty"`
	UserLocation      *UserLocation     `json:"user_location,omitempty"`
	Filters           *WebSearchFilters `json:"filters,omitempty"`
}

// ResponsesUsage represents token usage in the Responses API
type ResponsesUsage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
	TotalTokens  int64 `json:"total_tokens"`
}

// ResponsesRequest represents the incoming request for the Responses API
type ResponsesRequest struct {
	Model                 string                 `json:"model"`
	Input                 json.RawMessage        `json:"input"`
	PreviousResponseID    string                 `json:"previous_response_id,omitempty"`
	Stream                bool                   `json:"stream,omitempty"`
	Tools                 []ResponsesTool        `json:"tools,omitempty"`
	PIICheckOptions       *PIICheckOptions       `json:"pii_check_options,omitempty"`
	InjectionCheckOptions *InjectionCheckOptions `json:"injection_check_options,omitempty"`
}

// ChatCompletionResponse represents the non-streaming chat completion response
type ChatCompletionResponse struct {
	ID      string                       `json:"id"`
	Object  string                       `json:"object"`
	Created int64                        `json:"created"`
	Model   string                       `json:"model"`
	Choices []ChatCompletionChoiceOutput `json:"choices"`
	Usage   openai.CompletionUsage       `json:"usage,omitempty"`
}

// ChatCompletionChoiceOutput represents a choice in the response
type ChatCompletionChoiceOutput struct {
	Index        int64                       `json:"index"`
	FinishReason string                      `json:"finish_reason"`
	Message      ChatCompletionMessageOutput `json:"message"`
}

// BlockedSearch represents a search that was blocked by a filter
type BlockedSearch struct {
	ID     string `json:"id"`
	Query  string `json:"query"`
	Reason string `json:"reason,omitempty"`
}

// ChatCompletionMessageOutput represents the assistant message.
// Includes standard OpenAI fields plus custom extensions:
//   - annotations: URL citations from web search (standard format)
//   - search_reasoning: Agent's reasoning about search decisions (extension)
//   - blocked_searches: Queries blocked by safety filters (extension)
//
// FetchCall represents a URL fetch that was performed
type FetchCall struct {
	ID     string           `json:"id"`
	Status string           `json:"status"`
	Action *WebSearchAction `json:"action"`
}

type ChatCompletionMessageOutput struct {
	Role            string                `json:"role"`
	Content         string                `json:"content"`
	Annotations     []pipeline.Annotation `json:"annotations,omitempty"`
	SearchReasoning string                `json:"search_reasoning,omitempty"`
	BlockedSearches []BlockedSearch       `json:"blocked_searches,omitempty"`
	FetchCalls      []FetchCall           `json:"fetch_calls,omitempty"`
}
