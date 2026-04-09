package pipeline

import (
	"encoding/json"

	openai "github.com/openai/openai-go/v3"
	"github.com/tinfoilsh/confidential-websearch/agent"
)

// APIFormat indicates which API format the request uses
type APIFormat int

const (
	FormatChatCompletion APIFormat = iota
	FormatResponses
)

type SearchContextSize string

const (
	SearchContextSizeLow    SearchContextSize = "low"
	SearchContextSizeMedium SearchContextSize = "medium"
	SearchContextSizeHigh   SearchContextSize = "high"
)

// Emitter status values
const (
	EmitStatusInProgress = "in_progress"
	EmitStatusCompleted  = "completed"
	EmitStatusFailed     = "failed"
	EmitStatusBlocked    = "blocked"
)

// Annotation type
const (
	AnnotationTypeURLCitation = "url_citation"
)

// Message represents a chat message
type Message struct {
	Role        string          `json:"role"`
	Content     json.RawMessage `json:"content"` // Preserved verbatim for responder; text extracted for agent
	Annotations []Annotation    `json:"annotations,omitempty"`
}

// GetTextContent extracts text from Content (handles both string and multimodal array).
// Only used for agent operations (search, PII, prompt injection) - not for responder.
func (m *Message) GetTextContent() string {
	return ExtractTextContent(m.Content)
}

// ExtractTextContent extracts text from a json.RawMessage that may be a string or multimodal array.
func ExtractTextContent(content json.RawMessage) string {
	var s string
	if json.Unmarshal(content, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(content, &parts) == nil {
		for _, p := range parts {
			if p.Type == "text" {
				return p.Text
			}
		}
	}
	return ""
}

// Annotation represents a url_citation annotation
type Annotation struct {
	Type        string      `json:"type"`
	URLCitation URLCitation `json:"url_citation"`
}

// URLCitation contains the citation details (OpenAI-compatible)
type URLCitation struct {
	StartIndex int    `json:"start_index"`
	EndIndex   int    `json:"end_index"`
	URL        string `json:"url"`
	Title      string `json:"title"`
}

type UserLocation struct {
	Country string
	City    string
	Region  string
}

// Request is the unified internal request representation
type Request struct {
	Model              string
	Messages           []Message
	Input              json.RawMessage // For Responses API
	PreviousResponseID string
	Stream             bool
	Temperature        *float64
	MaxTokens          *int64
	Format             APIFormat
	AuthHeader         string

	// Feature flags
	WebSearchEnabled      bool
	PIICheckEnabled       bool
	InjectionCheckEnabled bool
	SearchContextSize     SearchContextSize
	UserLocation          *UserLocation
	AllowedDomains        []string
	StreamIncludeUsage    bool
}

// EventEmitter handles streaming output events
type EventEmitter interface {
	// EmitSearchCall emits a web search call event with OpenAI-compatible fields
	// (reason is optional, used for blocked status; created/model for SDK compatibility)
	EmitSearchCall(id, status, query, reason string, created int64, model string) error

	// EmitFetchCall emits a URL fetch event (action.type "open_page")
	EmitFetchCall(id, status, url string, created int64, model string) error

	// EmitMetadata emits annotations and reasoning before content
	EmitMetadata(id string, created int64, model string, annotations []Annotation, reasoning string) error

	// EmitChunk emits a raw chunk of data
	EmitChunk(data []byte) error

	// EmitError emits an error event
	EmitError(err error) error

	// EmitDone emits the final done signal
	EmitDone(id string, created int64, model string, usage openai.CompletionUsage) error

	// Responses API lifecycle methods (no-op for Chat Completions)
	EmitResponseStart() error
	EmitMessageStart(itemID string) error
	EmitMessageEnd(text string, annotations []Annotation) error
}

// BuildAnnotations creates URL citations from search results.
func BuildAnnotations(searchResults []agent.ToolCall) []Annotation {
	if len(searchResults) == 0 {
		return nil
	}

	var annotations []Annotation
	for _, toolCall := range searchResults {
		for _, result := range toolCall.Results {
			annotations = append(annotations, Annotation{
				Type: AnnotationTypeURLCitation,
				URLCitation: URLCitation{
					URL:   result.URL,
					Title: result.Title,
				},
			})
		}
	}

	return annotations
}
