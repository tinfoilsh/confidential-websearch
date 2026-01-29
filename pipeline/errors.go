package pipeline

import (
	"errors"
	"fmt"
	"net/http"
)

// PipelineError wraps errors that occur during pipeline execution
type PipelineError struct {
	Stage     string
	Err       error
	Retryable bool
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

	switch {
	case errors.As(err, &validationErr):
		return http.StatusBadRequest, map[string]any{
			"error": map[string]string{
				"message": validationErr.Message,
				"type":    "validation_error",
				"field":   validationErr.Field,
			},
		}

	case errors.As(err, &agentErr):
		return http.StatusInternalServerError, map[string]any{
			"error": map[string]string{
				"message": "agent processing failed",
				"type":    "agent_error",
			},
		}

	case errors.As(err, &responderErr):
		return http.StatusInternalServerError, map[string]any{
			"error": map[string]string{
				"message": "response generation failed",
				"type":    "responder_error",
			},
		}

	default:
		return http.StatusInternalServerError, map[string]any{
			"error": map[string]string{
				"message": "internal server error",
				"type":    "api_error",
			},
		}
	}
}
