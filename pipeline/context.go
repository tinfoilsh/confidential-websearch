package pipeline

import (
	"context"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/tinfoilsh/confidential-websearch/agent"
)

// APIFormat indicates which API format the request uses
type APIFormat int

const (
	FormatChatCompletion APIFormat = iota
	FormatResponses
)

// ReasoningSummaryPart represents a part of the reasoning summary (mirrors agent.ReasoningSummaryPart)
type ReasoningSummaryPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ReasoningItem represents a reasoning item from the Responses API (mirrors agent.ReasoningItem)
type ReasoningItem struct {
	ID      string                 `json:"id"`
	Type    string                 `json:"type"`
	Summary []ReasoningSummaryPart `json:"summary"`
}

// Message represents a chat message
type Message struct {
	Role        string       `json:"role"`
	Content     string       `json:"content"`
	Annotations []Annotation `json:"annotations,omitempty"`
}

// Annotation represents a url_citation annotation
type Annotation struct {
	Type        string      `json:"type"`
	URLCitation URLCitation `json:"url_citation"`
}

// URLCitation contains the citation details
type URLCitation struct {
	Title         string `json:"title"`
	URL           string `json:"url"`
	Content       string `json:"content,omitempty"`
	PublishedDate string `json:"published_date,omitempty"`
}

// Tool type constants
const (
	ToolTypeWebSearch      = "web_search"
	ToolTypePIICheck       = "pii_check"
	ToolTypeInjectionCheck = "injection_check"
)

// Request is the unified internal request representation
type Request struct {
	Model       string
	Messages    []Message
	Input       string    // For Responses API
	Stream      bool
	Temperature *float64
	MaxTokens   *int64
	Format      APIFormat
	AuthHeader  string
	Tools       []string  // Enabled tools: web_search, pii_check, injection_check
}

// HasTool returns true if the specified tool is enabled in the request
func (r *Request) HasTool(toolType string) bool {
	if r == nil {
		return false
	}
	for _, t := range r.Tools {
		if t == toolType {
			return true
		}
	}
	return false
}

// Context carries all request data through the pipeline
type Context struct {
	context.Context

	// Identification
	RequestID string

	// Input
	Request   *Request
	UserQuery string

	// Request options for LLM calls
	ReqOpts []option.RequestOption

	// Intermediate results
	AgentResult       *agent.Result
	SearchResults     []agent.ToolCall // Executed search results
	ResponderMessages []openai.ChatCompletionMessageParamUnion
	ResponderResult   interface{} // Holds *llm.ResponderResult for non-streaming

	// State tracking
	State StateTracker

	// Streaming support (nil for non-streaming)
	Emitter EventEmitter

	// Cancel function for timeout
	Cancel context.CancelFunc
}

// NewContext creates a new pipeline context
func NewContext(ctx context.Context, requestID string, req *Request) *Context {
	return &Context{
		Context:   ctx,
		RequestID: requestID,
		Request:   req,
		State:     NewStateTracker(),
	}
}

// IsStreaming returns true if this is a streaming request
func (c *Context) IsStreaming() bool {
	return c.Emitter != nil
}

// EventEmitter handles streaming output events
type EventEmitter interface {
	// EmitSearchCall emits a web search call event (reason is optional, used for blocked status)
	EmitSearchCall(id, status, query, reason string) error

	// EmitMetadata emits annotations, reasoning, and reasoning items before content
	EmitMetadata(annotations []Annotation, reasoning string, reasoningItems []ReasoningItem) error

	// EmitChunk emits a raw chunk of data
	EmitChunk(data []byte) error

	// EmitError emits an error event
	EmitError(err error) error

	// EmitDone emits the final done signal
	EmitDone() error
}
