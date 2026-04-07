package api

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/google/uuid"
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
	errType := pipeline.ErrTypeInvalidRequest
	if code >= 500 {
		errType = pipeline.ErrTypeServer
	}
	jsonErrorResponse(w, code, map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    errType,
			"code":    nil,
			"param":   nil,
		},
	})
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
func buildFlatAnnotations(annotations []pipeline.Annotation) []FlatAnnotation {
	if len(annotations) == 0 {
		return nil
	}

	var flat []FlatAnnotation
	for _, annotation := range annotations {
		if annotation.Type != pipeline.AnnotationTypeURLCitation {
			continue
		}
		flat = append(flat, FlatAnnotation{
			Type:       ContentTypeURLCitation,
			URL:        annotation.URLCitation.URL,
			Title:      annotation.URLCitation.Title,
			StartIndex: annotation.URLCitation.StartIndex,
			EndIndex:   annotation.URLCitation.EndIndex,
		})
	}
	return flat
}

// buildLegacyFlatAnnotations creates URL citations from search results when
// exact marker locations are unavailable.
func buildLegacyFlatAnnotations(toolCalls []agent.ToolCall) []FlatAnnotation {
	var annotations []FlatAnnotation
	for _, tc := range toolCalls {
		for _, r := range tc.Results {
			annotations = append(annotations, FlatAnnotation{
				Type:  ContentTypeURLCitation,
				URL:   r.URL,
				Title: r.Title,
			})
		}
	}
	return annotations
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
		s.handleStreamingChatCompletion(w, r, pipelineReq)
	} else {
		s.handleNonStreamingChatCompletion(w, r, pipelineReq)
	}
}

func (s *Server) handleNonStreamingChatCompletion(w http.ResponseWriter, r *http.Request, req *pipeline.Request) {
	log.Infof("Processing query (model: %s)", req.Model)

	result, err := s.Runner.Run(r.Context(), req)
	if err != nil {
		status, body := pipeline.ErrorResponse(err)
		jsonErrorResponse(w, status, body)
		return
	}

	log.Infof("Web search completed: %d searches", len(result.SearchResults))

	annotations := result.Annotations

	// Convert agent reasoning and blocked queries to API format
	var blockedSearches []BlockedSearch
	for _, bq := range result.BlockedQueries {
		blockedSearches = append(blockedSearches, BlockedSearch{
			ID:     bq.ID,
			Query:  bq.Query,
			Reason: bq.Reason,
		})
	}

	var fetchCalls []FetchCall
	for _, fp := range result.FetchCalls {
		fetchCalls = append(fetchCalls, FetchCall{
			ID:     fp.ID,
			Status: StatusCompleted,
			Action: &WebSearchAction{
				Type: ActionTypeOpenPage,
				URL:  fp.URL,
			},
		})
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
				FinishReason: FinishReasonStop,
				Message: ChatCompletionMessageOutput{
					Role:            RoleAssistant,
					Content:         result.Content,
					Annotations:     annotations,
					SearchReasoning: result.SearchReasoning,
					BlockedSearches: blockedSearches,
					FetchCalls:      fetchCalls,
				},
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) handleStreamingChatCompletion(w http.ResponseWriter, r *http.Request, req *pipeline.Request) {
	log.Infof("Processing streaming query (model: %s)", req.Model)

	emitter, err := NewSSEEmitter(w)
	if err != nil {
		jsonError(w, err.Error(), http.StatusInternalServerError)
		return
	}

	defer func() {
		if r := recover(); r != nil {
			log.Errorf("panic recovered in streaming chat completion: %v", r)
			emitter.EmitError(fmt.Errorf("internal server error"))
		}
	}()

	result, err := s.Runner.Stream(r.Context(), req, emitter)
	if err != nil {
		emitter.EmitError(err)
		return
	}

	log.Infof("Streaming completed: %d searches", len(result.SearchResults))
}

func (s *Server) handleStreamingResponses(w http.ResponseWriter, r *http.Request, req *pipeline.Request) {
	log.Infof("Processing streaming responses request (model: %s)", req.Model)

	responseID := IDPrefixResponse + uuid.New().String()[:8]
	emitter, err := NewResponsesEmitter(w, responseID, req.Model)
	if err != nil {
		jsonError(w, err.Error(), http.StatusInternalServerError)
		return
	}

	defer func() {
		if r := recover(); r != nil {
			log.Errorf("panic recovered in streaming responses: %v", r)
			emitter.EmitError(fmt.Errorf("internal server error"))
		}
	}()

	// Emit response.created and response.in_progress at start
	if err := emitter.EmitResponseStart(); err != nil {
		emitter.EmitError(err)
		return
	}

	result, err := s.Runner.Stream(r.Context(), req, emitter)
	if err != nil {
		emitter.EmitError(err)
		return
	}

	log.Infof("Streaming responses completed: %d searches", len(result.SearchResults))
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
		Stream:                req.Stream,
		Format:                pipeline.FormatResponses,
		WebSearchEnabled:      webSearchEnabled,
		PIICheckEnabled:       piiCheckEnabled,
		InjectionCheckEnabled: injectionCheckEnabled,
	}

	if req.Stream {
		s.handleStreamingResponses(w, r, pipelineReq)
		return
	}

	log.Infof("Processing responses request (model: %s)", req.Model)

	result, err := s.Runner.Run(r.Context(), pipelineReq)
	if err != nil {
		status, body := pipeline.ErrorResponse(err)
		jsonErrorResponse(w, status, body)
		return
	}

	flatAnnotations := buildFlatAnnotations(result.Annotations)
	if len(flatAnnotations) == 0 {
		flatAnnotations = buildLegacyFlatAnnotations(result.SearchResults)
	}

	var output []ResponsesOutput

	// Add blocked searches first
	for _, bq := range result.BlockedQueries {
		output = append(output, ResponsesOutput{
			Type:   ItemTypeWebSearchCall,
			ID:     IDPrefixWebSearch + bq.ID,
			Status: StatusBlocked,
			Reason: bq.Reason,
			Action: &WebSearchAction{
				Type:  ActionTypeSearch,
				Query: bq.Query,
			},
		})
	}

	for _, fp := range result.FetchCalls {
		output = append(output, ResponsesOutput{
			Type:   ItemTypeWebSearchCall,
			ID:     IDPrefixWebSearch + fp.ID,
			Status: StatusCompleted,
			Action: &WebSearchAction{
				Type: ActionTypeOpenPage,
				URL:  fp.URL,
			},
		})
	}

	for _, tc := range result.SearchResults {
		output = append(output, ResponsesOutput{
			Type:   ItemTypeWebSearchCall,
			ID:     IDPrefixWebSearch + tc.ID,
			Status: StatusCompleted,
			Action: &WebSearchAction{
				Type:  ActionTypeSearch,
				Query: tc.Query,
			},
		})
	}

	output = append(output, ResponsesOutput{
		Type:   ItemTypeMessage,
		ID:     IDPrefixMessage + uuid.New().String()[:8],
		Status: StatusCompleted,
		Role:   RoleAssistant,
		Content: []ResponsesContent{
			{
				Type:            ContentTypeOutputText,
				Text:            result.Content,
				Annotations:     flatAnnotations,
				SearchReasoning: result.SearchReasoning,
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
		ID:        IDPrefixResponse + uuid.New().String()[:8],
		Object:    ObjectResponse,
		CreatedAt: result.Created,
		Status:    StatusCompleted,
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
	if r.URL.Path != "/" {
		jsonError(w, fmt.Sprintf("Invalid URL (%s %s)", r.Method, r.URL.Path), http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"service": "confidential-websearch", "status": "ok"})
}
