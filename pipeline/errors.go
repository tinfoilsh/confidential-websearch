package pipeline

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	openai "github.com/openai/openai-go/v3"
)

// OpenAI API error type constants
const (
	ErrTypeInvalidRequest = "invalid_request_error"
	ErrTypeServer         = "server_error"
)

// PipelineError wraps errors that occur during pipeline execution
type PipelineError struct {
	Stage string
	Err   error
}

func (e *PipelineError) Error() string {
	return fmt.Sprintf("pipeline failed at stage %q: %v", e.Stage, e.Err)
}

func (e *PipelineError) Unwrap() error {
	return e.Err
}

// ValidationError indicates invalid request parameters
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	if e.Field != "" {
		return fmt.Sprintf("validation error on field %q: %s", e.Field, e.Message)
	}
	return fmt.Sprintf("validation error: %s", e.Message)
}

// AgentError indicates the agent LLM call failed
type AgentError struct {
	Err error
}

func (e *AgentError) Error() string {
	return fmt.Sprintf("agent error: %v", e.Err)
}

func (e *AgentError) Unwrap() error {
	return e.Err
}

// ResponderError indicates the responder LLM call failed
type ResponderError struct {
	Err error
}

func (e *ResponderError) Error() string {
	return fmt.Sprintf("responder error: %v", e.Err)
}

func (e *ResponderError) Unwrap() error {
	return e.Err
}

// ErrorResponse maps an error to an HTTP status code and response body
func ErrorResponse(err error) (int, map[string]any) {
	var validationErr *ValidationError
	var agentErr *AgentError
	var responderErr *ResponderError
	var pipelineErr *PipelineError

	// Check for pipeline error first and unwrap
	if errors.As(err, &pipelineErr) {
		err = pipelineErr.Err
	}

	// Pass through upstream API errors with their original status code and structured fields
	var apiErr *openai.Error
	if errors.As(err, &apiErr) && apiErr.RawJSON() != "" {
		var rawError map[string]any
		if json.Unmarshal([]byte(apiErr.RawJSON()), &rawError) == nil {
			return apiErr.StatusCode, map[string]any{"error": rawError}
		}
	}

	switch {
	case errors.As(err, &validationErr):
		return http.StatusBadRequest, map[string]any{
			"error": map[string]any{
				"message": validationErr.Message,
				"type":    ErrTypeInvalidRequest,
				"param":   validationErr.Field,
				"code":    nil,
			},
		}

	case errors.As(err, &agentErr):
		return http.StatusInternalServerError, map[string]any{
			"error": map[string]any{
				"message": "agent processing failed",
				"type":    ErrTypeServer,
				"param":   nil,
				"code":    nil,
			},
		}

	case errors.As(err, &responderErr):
		return http.StatusInternalServerError, map[string]any{
			"error": map[string]any{
				"message": "response generation failed",
				"type":    ErrTypeServer,
				"param":   nil,
				"code":    nil,
			},
		}

	default:
		return http.StatusInternalServerError, map[string]any{
			"error": map[string]any{
				"message": "internal server error",
				"type":    ErrTypeServer,
				"param":   nil,
				"code":    nil,
			},
		}
	}
}
