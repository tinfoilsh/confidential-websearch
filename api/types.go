package api

import (
	"context"
	"time"

	"github.com/openai/openai-go/v2/option"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/agent"
	"github.com/tinfoilsh/confidential-websearch/config"
)

const (
	MaxRequestBodySize = 200 << 20 // 200 MB
	RequestTimeout     = 2 * time.Minute
)

// Server holds all dependencies for the HTTP handlers
type Server struct {
	Cfg    *config.Config
	Client *tinfoil.Client
	Agent  *agent.Agent
}

// IncomingRequest represents the incoming chat request
type IncomingRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream"`
	Temperature *float64  `json:"temperature,omitempty"`
	MaxTokens   *int64    `json:"max_tokens,omitempty"`
}

// Message represents a chat message in the incoming request
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// URLCitation contains the citation details
type URLCitation struct {
	Title         string `json:"title"`
	URL           string `json:"url"`
	Content       string `json:"content,omitempty"`
	PublishedDate string `json:"published_date,omitempty"`
}

// Annotation represents a url_citation annotation
type Annotation struct {
	Type        string      `json:"type"`
	URLCitation URLCitation `json:"url_citation"`
}

// WebSearchCall represents a search operation in streaming output
type WebSearchCall struct {
	Type   string           `json:"type"`
	ID     string           `json:"id"`
	Status string           `json:"status"`
	Action *WebSearchAction `json:"action,omitempty"`
}

// StreamingDelta represents a delta in a streaming chunk
type StreamingDelta struct {
	Annotations []Annotation `json:"annotations,omitempty"`
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
	Type        string           `json:"type"`
	Text        string           `json:"text"`
	Annotations []FlatAnnotation `json:"annotations,omitempty"`
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
