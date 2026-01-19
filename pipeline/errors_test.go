package pipeline

import (
	"errors"
	"net/http"
	"testing"
)

func TestPipelineError(t *testing.T) {
	inner := errors.New("inner error")
	err := &PipelineError{
		Stage:     "validate",
		Err:       inner,
		Retryable: false,
	}

	expected := `pipeline failed at stage "validate": inner error`
	if err.Error() != expected {
		t.Errorf("expected %q, got %q", expected, err.Error())
	}

	if !errors.Is(err, inner) {
		t.Error("Unwrap should return inner error")
	}
}

func TestValidationError(t *testing.T) {
	tests := []struct {
		name     string
		field    string
		message  string
		expected string
	}{
		{
			name:     "with field",
			field:    "model",
			message:  "is required",
			expected: `validation error on field "model": is required`,
		},
		{
			name:     "without field",
			field:    "",
			message:  "invalid request",
			expected: "validation error: invalid request",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := &ValidationError{Field: tt.field, Message: tt.message}
			if err.Error() != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, err.Error())
			}
		})
	}
}

func TestAgentError(t *testing.T) {
	inner := errors.New("LLM timeout")
	err := &AgentError{Err: inner}

	expected := "agent error: LLM timeout"
	if err.Error() != expected {
		t.Errorf("expected %q, got %q", expected, err.Error())
	}

	if !errors.Is(err, inner) {
		t.Error("Unwrap should return inner error")
	}
}

func TestSearchError(t *testing.T) {
	inner := errors.New("API rate limited")
	err := &SearchError{Query: "test query", Err: inner}

	expected := `search error for query "test query": API rate limited`
	if err.Error() != expected {
		t.Errorf("expected %q, got %q", expected, err.Error())
	}

	if !errors.Is(err, inner) {
		t.Error("Unwrap should return inner error")
	}
}

func TestResponderError(t *testing.T) {
	inner := errors.New("context cancelled")
	err := &ResponderError{Err: inner}

	expected := "responder error: context cancelled"
	if err.Error() != expected {
		t.Errorf("expected %q, got %q", expected, err.Error())
	}

	if !errors.Is(err, inner) {
		t.Error("Unwrap should return inner error")
	}
}

func TestStreamingError(t *testing.T) {
	inner := errors.New("connection closed")
	err := &StreamingError{Err: inner}

	expected := "streaming error: connection closed"
	if err.Error() != expected {
		t.Errorf("expected %q, got %q", expected, err.Error())
	}

	if !errors.Is(err, inner) {
		t.Error("Unwrap should return inner error")
	}
}

func TestErrorResponse(t *testing.T) {
	tests := []struct {
		name           string
		err            error
		expectedStatus int
		expectedType   string
	}{
		{
			name:           "validation error",
			err:            &ValidationError{Field: "model", Message: "is required"},
			expectedStatus: http.StatusBadRequest,
			expectedType:   "validation_error",
		},
		{
			name:           "agent error",
			err:            &AgentError{Err: errors.New("failed")},
			expectedStatus: http.StatusInternalServerError,
			expectedType:   "agent_error",
		},
		{
			name:           "search error",
			err:            &SearchError{Query: "test", Err: errors.New("failed")},
			expectedStatus: http.StatusInternalServerError,
			expectedType:   "search_error",
		},
		{
			name:           "responder error",
			err:            &ResponderError{Err: errors.New("failed")},
			expectedStatus: http.StatusInternalServerError,
			expectedType:   "responder_error",
		},
		{
			name:           "streaming error",
			err:            &StreamingError{Err: errors.New("failed")},
			expectedStatus: http.StatusInternalServerError,
			expectedType:   "streaming_error",
		},
		{
			name:           "unknown error",
			err:            errors.New("something went wrong"),
			expectedStatus: http.StatusInternalServerError,
			expectedType:   "api_error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status, body := ErrorResponse(tt.err)

			if status != tt.expectedStatus {
				t.Errorf("expected status %d, got %d", tt.expectedStatus, status)
			}

			errObj, ok := body["error"].(map[string]string)
			if !ok {
				t.Fatal("expected error object in response")
			}

			if errObj["type"] != tt.expectedType {
				t.Errorf("expected type %q, got %q", tt.expectedType, errObj["type"])
			}
		})
	}
}

func TestErrorResponseWithPipelineWrapper(t *testing.T) {
	// Validation error wrapped in pipeline error should still return 400
	validationErr := &ValidationError{Field: "model", Message: "is required"}
	pipelineErr := &PipelineError{Stage: "validate", Err: validationErr}

	status, body := ErrorResponse(pipelineErr)

	if status != http.StatusBadRequest {
		t.Errorf("expected status %d, got %d", http.StatusBadRequest, status)
	}

	errObj, ok := body["error"].(map[string]string)
	if !ok {
		t.Fatal("expected error object in response")
	}

	if errObj["type"] != "validation_error" {
		t.Errorf("expected type %q, got %q", "validation_error", errObj["type"])
	}
}

func TestErrorsAsChain(t *testing.T) {
	// Test that errors.As works through the chain
	inner := errors.New("root cause")
	agentErr := &AgentError{Err: inner}
	pipelineErr := &PipelineError{Stage: "agent", Err: agentErr}

	var target *AgentError
	if !errors.As(pipelineErr, &target) {
		t.Error("errors.As should find AgentError in chain")
	}
}
