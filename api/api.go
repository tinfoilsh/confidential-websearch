package api

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/google/uuid"
	"github.com/openai/openai-go/v3/option"
	log "github.com/sirupsen/logrus"

	"github.com/tinfoilsh/confidential-websearch/agent"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

// RecoveryMiddleware catches panics and returns 500 instead of crashing
func RecoveryMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				log.Errorf("panic recovered: %v", err)
				jsonError(w, "internal server error", http.StatusInternalServerError)
			}
		}()
		next(w, r)
	}
}

func jsonError(w http.ResponseWriter, message string, code int) {
	log.WithField("code", code).Warn(message)
	jsonErrorResponse(w, code, map[string]any{"error": message})
}

func jsonErrorResponse(w http.ResponseWriter, code int, body map[string]any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(body)
}

func parseRequestBody(r *http.Request, v any) error {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return fmt.Errorf("failed to read request: %w", err)
	}
	if err := json.Unmarshal(body, v); err != nil {
		return fmt.Errorf("failed to parse request: %w", err)
	}
	return nil
}

// convertMessages converts API messages to pipeline messages
func convertMessages(msgs []Message) []pipeline.Message {
	result := make([]pipeline.Message, len(msgs))
	for i, msg := range msgs {
		result[i] = pipeline.Message{
			Role:        msg.Role,
			Content:     msg.Content,
			Annotations: msg.Annotations,
		}
	}
	return result
}

// buildFlatAnnotations creates URL citations (Responses API format)
func buildFlatAnnotations(toolCalls []agent.ToolCall) []FlatAnnotation {
	var annotations []FlatAnnotation
	for _, tc := range toolCalls {
		for _, r := range tc.Results {
			annotations = append(annotations, FlatAnnotation{
				Type:  "url_citation",
				URL:   r.URL,
				Title: r.Title,
			})
		}
	}
	return annotations
}

// extractRequestOptions extracts request options from HTTP headers
func extractRequestOptions(r *http.Request) []option.RequestOption {
	var reqOpts []option.RequestOption
	if auth := r.Header.Get("Authorization"); auth != "" {
		reqOpts = append(reqOpts, option.WithHeader("Authorization", auth))
	}
	return reqOpts
}

func (s *Server) HandleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		jsonError(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, MaxRequestBodySize)

	var req IncomingRequest
	if err := parseRequestBody(r, &req); err != nil {
		jsonError(w, err.Error(), http.StatusBadRequest)
		return
	}

	reqOpts := extractRequestOptions(r)

	// Derive feature flags from options presence
	webSearchEnabled := req.WebSearchOptions != nil
	piiCheckEnabled := req.PIICheckOptions != nil
	injectionCheckEnabled := req.InjectionCheckOptions != nil
	log.Debugf("Request features: web_search=%v, pii_check=%v, injection_check=%v",
		webSearchEnabled, piiCheckEnabled, injectionCheckEnabled)

	pipelineReq := &pipeline.Request{
		Model:                 req.Model,
		Messages:              convertMessages(req.Messages),
		Stream:                req.Stream,
		Temperature:           req.Temperature,
		MaxTokens:             req.MaxTokens,
		Format:                pipeline.FormatChatCompletion,
		WebSearchEnabled:      webSearchEnabled,
		PIICheckEnabled:       piiCheckEnabled,
		InjectionCheckEnabled: injectionCheckEnabled,
	}

	if req.Stream {
		s.handleStreamingChatCompletion(w, r, pipelineReq, reqOpts)
	} else {
		s.handleNonStreamingChatCompletion(w, r, pipelineReq, reqOpts)
	}
}

func (s *Server) handleNonStreamingChatCompletion(w http.ResponseWriter, r *http.Request, req *pipeline.Request, reqOpts []option.RequestOption) {
	log.Infof("Processing query (model: %s)", req.Model)

	pctx, err := s.Pipeline.Execute(r.Context(), req, nil, reqOpts...)
	if pctx != nil && pctx.Cancel != nil {
		defer pctx.Cancel()
	}

	if err != nil {
		status, body := pipeline.ErrorResponse(err)
		jsonErrorResponse(w, status, body)
		return
	}

	log.Infof("Agent completed: %d searches", len(pctx.SearchResults))

	result := pctx.ResponderResult
	annotations := pipeline.BuildAnnotations(pctx.SearchResults)

	// Convert agent reasoning and blocked queries to API format
	var blockedSearches []BlockedSearch
	var agentReasoning string
	if pctx.AgentResult != nil {
		agentReasoning = pctx.AgentResult.SearchReasoning
		for _, bq := range pctx.AgentResult.BlockedQueries {
			blockedSearches = append(blockedSearches, BlockedSearch{
				ID:     bq.ID,
				Query:  bq.Query,
				Reason: bq.Reason,
			})
		}
	}

	response := ChatCompletionResponse{
		ID:      result.ID,
		Object:  result.Object,
		Created: result.Created,
		Model:   result.Model,
		Usage:   result.Usage,
		Choices: []ChatCompletionChoiceOutput{
			{
				Index:        0,
				FinishReason: "stop",
				Message: ChatCompletionMessageOutput{
					Role:            "assistant",
					Content:         result.Content,
					Annotations:     annotations,
					SearchReasoning: agentReasoning,
					BlockedSearches: blockedSearches,
				},
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) handleStreamingChatCompletion(w http.ResponseWriter, r *http.Request, req *pipeline.Request, reqOpts []option.RequestOption) {
	log.Infof("Processing streaming query (model: %s)", req.Model)

	emitter, err := NewSSEEmitter(w)
	if err != nil {
		jsonError(w, err.Error(), http.StatusInternalServerError)
		return
	}

	pctx, err := s.Pipeline.Execute(r.Context(), req, emitter, reqOpts...)
	if pctx != nil && pctx.Cancel != nil {
		defer pctx.Cancel()
	}

	if err != nil {
		emitter.EmitError(err)
		return
	}

	log.Infof("Streaming completed: %d searches", len(pctx.SearchResults))
}

func (s *Server) HandleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		jsonError(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, MaxRequestBodySize)

	var req ResponsesRequest
	if err := parseRequestBody(r, &req); err != nil {
		jsonError(w, err.Error(), http.StatusBadRequest)
		return
	}

	reqOpts := extractRequestOptions(r)

	// Derive feature flags from tools array and options
	webSearchEnabled := false
	for _, t := range req.Tools {
		if t.Type == "web_search" {
			webSearchEnabled = true
			break
		}
	}
	piiCheckEnabled := req.PIICheckOptions != nil
	injectionCheckEnabled := req.InjectionCheckOptions != nil
	log.Debugf("Responses request features: web_search=%v, pii_check=%v, injection_check=%v",
		webSearchEnabled, piiCheckEnabled, injectionCheckEnabled)

	pipelineReq := &pipeline.Request{
		Model:                 req.Model,
		Input:                 req.Input,
		Format:                pipeline.FormatResponses,
		WebSearchEnabled:      webSearchEnabled,
		PIICheckEnabled:       piiCheckEnabled,
		InjectionCheckEnabled: injectionCheckEnabled,
	}

	log.Infof("Processing responses request (model: %s)", req.Model)

	pctx, err := s.Pipeline.Execute(r.Context(), pipelineReq, nil, reqOpts...)
	if pctx != nil && pctx.Cancel != nil {
		defer pctx.Cancel()
	}

	if err != nil {
		status, body := pipeline.ErrorResponse(err)
		jsonErrorResponse(w, status, body)
		return
	}

	result := pctx.ResponderResult
	flatAnnotations := buildFlatAnnotations(pctx.SearchResults)

	// Extract agent reasoning
	var agentReasoning string
	if pctx.AgentResult != nil {
		agentReasoning = pctx.AgentResult.SearchReasoning
	}

	var output []ResponsesOutput

	// Add blocked searches first
	if pctx.AgentResult != nil {
		for _, bq := range pctx.AgentResult.BlockedQueries {
			output = append(output, ResponsesOutput{
				Type:   "web_search_call",
				ID:     "ws_" + bq.ID,
				Status: "blocked",
				Reason: bq.Reason,
				Action: &WebSearchAction{
					Type:  "search",
					Query: bq.Query,
				},
			})
		}
	}

	for _, tc := range pctx.SearchResults {
		output = append(output, ResponsesOutput{
			Type:   "web_search_call",
			ID:     "ws_" + tc.ID,
			Status: "completed",
			Action: &WebSearchAction{
				Type:  "search",
				Query: tc.Query,
			},
		})
	}

	output = append(output, ResponsesOutput{
		Type:   "message",
		ID:     "msg_" + uuid.New().String()[:8],
		Status: "completed",
		Role:   "assistant",
		Content: []ResponsesContent{
			{
				Type:            "output_text",
				Text:            result.Content,
				Annotations:     flatAnnotations,
				SearchReasoning: agentReasoning,
			},
		},
	})

	responsesResp := struct {
		ID        string            `json:"id"`
		Object    string            `json:"object"`
		CreatedAt int64             `json:"created_at"`
		Status    string            `json:"status"`
		Model     string            `json:"model"`
		Output    []ResponsesOutput `json:"output"`
		Usage     ResponsesUsage    `json:"usage"`
	}{
		ID:        "resp_" + uuid.New().String()[:8],
		Object:    "response",
		CreatedAt: result.Created,
		Status:    "completed",
		Model:     result.Model,
		Output:    output,
		Usage: ResponsesUsage{
			InputTokens:  result.Usage.PromptTokens,
			OutputTokens: result.Usage.CompletionTokens,
			TotalTokens:  result.Usage.TotalTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(responsesResp)
}

func (s *Server) HandleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok"}`))
}

func (s *Server) HandleRoot(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"service": "confidential-websearch", "status": "ok"})
}
