package pipeline

import (
	"context"
	"strings"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"

	"github.com/tinfoilsh/confidential-websearch/agent"
)

// Stage represents a single processing step in the pipeline
type Stage interface {
	Name() string
	Execute(ctx *Context) error
}

// AgentRunner defines the interface for running the agent
type AgentRunner interface {
	Run(ctx context.Context, userQuery string) (*agent.Result, error)
}

// AgentWithContext can receive conversation context per-request (thread-safe)
type AgentWithContext interface {
	AgentRunner
	RunWithContext(ctx context.Context, userQuery string, conversationContext string) (*agent.Result, error)
}

// MessageBuilder defines the interface for building responder messages
type MessageBuilder interface {
	Build(inputMessages []Message, agentResult *agent.Result) []openai.ChatCompletionMessageParamUnion
}

// ResponderParams contains parameters for a responder LLM call
type ResponderParams struct {
	Model       string
	Messages    []openai.ChatCompletionMessageParamUnion
	Temperature *float64
	MaxTokens   *int64
}

// ResponderResultData contains the result of a non-streaming responder call
type ResponderResultData struct {
	ID      string
	Model   string
	Object  string
	Created int64
	Content string
	Usage   interface{}
}

// Responder defines the interface for making responder LLM calls
type Responder interface {
	Complete(ctx context.Context, params ResponderParams, opts ...option.RequestOption) (*ResponderResultData, error)
	Stream(ctx context.Context, params ResponderParams, annotations []Annotation, reasoning string, emitter EventEmitter, opts ...option.RequestOption) error
}

// ValidateStage validates the incoming request
type ValidateStage struct{}

func (s *ValidateStage) Name() string { return "validate" }

func (s *ValidateStage) Execute(ctx *Context) error {
	if ctx.Request == nil {
		return &ValidationError{Message: "request is nil"}
	}

	if ctx.Request.Model == "" {
		return &ValidationError{Field: "model", Message: "model parameter is required"}
	}

	// Extract user query from messages (last user message)
	if ctx.Request.Format == FormatChatCompletion {
		for i := len(ctx.Request.Messages) - 1; i >= 0; i-- {
			if ctx.Request.Messages[i].Role == "user" && ctx.Request.Messages[i].Content != "" {
				ctx.UserQuery = ctx.Request.Messages[i].Content
				break
			}
		}
		if ctx.UserQuery == "" {
			return &ValidationError{Message: "no user message found"}
		}
	} else {
		// Responses API format
		if ctx.Request.Input == "" {
			return &ValidationError{Field: "input", Message: "input parameter is required"}
		}
		ctx.UserQuery = ctx.Request.Input
	}

	return nil
}

// AgentStage runs the agent to determine if search is needed
type AgentStage struct {
	Agent AgentRunner
}

func (s *AgentStage) Name() string { return "agent" }

func (s *AgentStage) Execute(ctx *Context) error {
	ctx.State.Transition(StateAgentStarted, map[string]interface{}{"query": ctx.UserQuery})

	var result *agent.Result
	var err error

	// Use RunWithContext if agent supports it (thread-safe conversation context for PII detection)
	if agentWithCtx, ok := s.Agent.(AgentWithContext); ok {
		conversationCtx := buildConversationContext(ctx.Request.Messages)
		result, err = agentWithCtx.RunWithContext(ctx.Context, ctx.UserQuery, conversationCtx)
	} else {
		result, err = s.Agent.Run(ctx.Context, ctx.UserQuery)
	}

	if err != nil {
		return &AgentError{Err: err}
	}

	ctx.AgentResult = result

	// Track search state transitions
	if len(result.ToolCalls) > 0 {
		ctx.State.Transition(StateSearchStarted, map[string]interface{}{"count": len(result.ToolCalls)})
		ctx.State.Transition(StateSearchCompleted, map[string]interface{}{"results": len(result.ToolCalls)})
	}

	ctx.State.Transition(StateAgentCompleted, nil)
	return nil
}

// BuildMessagesStage constructs the messages for the responder LLM
type BuildMessagesStage struct {
	Builder MessageBuilder
}

func (s *BuildMessagesStage) Name() string { return "build_messages" }

func (s *BuildMessagesStage) Execute(ctx *Context) error {
	messages := ctx.Request.Messages

	// For Responses API, construct messages from Input since Messages is empty
	if ctx.Request.Format == FormatResponses && len(messages) == 0 {
		messages = []Message{{Role: "user", Content: ctx.UserQuery}}
	}

	ctx.ResponderMessages = s.Builder.Build(messages, ctx.AgentResult)
	return nil
}

// ResponderStage calls the responder LLM to generate the final response
type ResponderStage struct {
	Responder Responder
}

func (s *ResponderStage) Name() string { return "responder" }

func (s *ResponderStage) Execute(ctx *Context) error {
	ctx.State.Transition(StateResponderStarted, nil)

	params := ResponderParams{
		Model:       ctx.Request.Model,
		Messages:    ctx.ResponderMessages,
		Temperature: ctx.Request.Temperature,
		MaxTokens:   ctx.Request.MaxTokens,
	}

	if ctx.IsStreaming() {
		ctx.State.Transition(StateResponderStreaming, nil)

		// Emit search completion events before streaming response content
		if ctx.AgentResult != nil && ctx.Emitter != nil {
			for _, tc := range ctx.AgentResult.ToolCalls {
				ctx.Emitter.EmitSearchCall(tc.ID, "completed", tc.Query, "")
			}
		}

		annotations := BuildAnnotations(ctx.AgentResult)
		reasoning := ""
		if ctx.AgentResult != nil {
			reasoning = ctx.AgentResult.AgentReasoning
		}

		err := s.Responder.Stream(ctx.Context, params, annotations, reasoning, ctx.Emitter, ctx.ReqOpts...)
		if err != nil {
			return &ResponderError{Err: err}
		}
	} else {
		result, err := s.Responder.Complete(ctx.Context, params, ctx.ReqOpts...)
		if err != nil {
			return &ResponderError{Err: err}
		}

		// Store result for handler to use
		ctx.ResponderResult = result
	}

	ctx.State.Transition(StateCompleted, nil)
	return nil
}

// BuildAnnotations creates URL citations from agent tool calls
func BuildAnnotations(agentResult *agent.Result) []Annotation {
	if agentResult == nil {
		return nil
	}

	var annotations []Annotation
	for _, tc := range agentResult.ToolCalls {
		for _, r := range tc.Results {
			// Use Summary for display if available, otherwise fall back to Content
			displayContent := r.Summary
			if displayContent == "" {
				displayContent = r.Content
			}
			annotations = append(annotations, Annotation{
				Type: "url_citation",
				URLCitation: URLCitation{
					URL:           r.URL,
					Title:         r.Title,
					Content:       displayContent,
					PublishedDate: r.PublishedDate,
				},
			})
		}
	}
	return annotations
}

// buildConversationContext formats conversation messages for PII detection context
func buildConversationContext(messages []Message) string {
	if len(messages) == 0 {
		return ""
	}

	var sb strings.Builder
	for _, msg := range messages {
		sb.WriteString(msg.Role)
		sb.WriteString(": ")
		sb.WriteString(msg.Content)
		sb.WriteString("\n")
	}
	return sb.String()
}
