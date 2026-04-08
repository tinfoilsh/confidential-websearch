package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/google/uuid"
	log "github.com/sirupsen/logrus"

	"github.com/tinfoilsh/confidential-websearch/agent"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
)

type requestValidator interface {
	Validate(*pipeline.Request) error
}

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

func writeJSON(w http.ResponseWriter, v any) {
	data, err := json.Marshal(v)
	if err != nil {
		jsonError(w, "failed to encode response", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(data)))
	w.Write(data)
}

func sanitizeErrorMessage(err error) string {
	var validationErr *pipeline.ValidationError
	if errors.As(err, &validationErr) {
		return validationErr.Error()
	}
	log.WithError(err).Error("streaming request failed")
	return "internal server error"
}

func parseRequestBody(r *http.Request, v any) error {
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(v); err != nil {
		if err == io.EOF {
			return fmt.Errorf("failed to read request: empty body")
		}
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

func convertUserLocation(location *UserLocation) *pipeline.UserLocation {
	if location == nil || location.Approximate == nil {
		return nil
	}

	return &pipeline.UserLocation{
		Country: location.Approximate.Country,
		City:    location.Approximate.City,
		Region:  location.Approximate.Region,
	}
}

func extractResponsesWebSearchOptions(tools []ResponsesTool) (bool, pipeline.SearchContextSize, *pipeline.UserLocation, []string) {
	for _, tool := range tools {
		if tool.Type != "web_search" {
			continue
		}
		var allowedDomains []string
		if tool.Filters != nil {
			allowedDomains = tool.Filters.AllowedDomains
		}
		return true, pipeline.SearchContextSize(tool.SearchContextSize), convertUserLocation(tool.UserLocation), allowedDomains
	}

	return false, "", nil, nil
}

func newResponseContinuationStore() *responseContinuationStore {
	s := &responseContinuationStore{
		entries:   make(map[string]responseContinuationEntry),
		stopPrune: make(chan struct{}),
	}
	go s.backgroundPrune()
	return s
}

func (s *Server) continuationStore() *responseContinuationStore {
	s.responseStoreOnce.Do(func() {
		if s.responseStore == nil {
			s.responseStore = newResponseContinuationStore()
		}
	})
	return s.responseStore
}

func (s *Server) rememberResponseID(publicID, upstreamID string) {
	if publicID == "" || upstreamID == "" || publicID == upstreamID {
		return
	}
	s.continuationStore().Put(publicID, upstreamID)
}

func (s *Server) resolvePreviousResponseID(id string) string {
	if id == "" {
		return ""
	}
	if upstreamID, ok := s.continuationStore().Get(id); ok {
		return upstreamID
	}
	return id
}

func (s *responseContinuationStore) Put(publicID, upstreamID string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	if len(s.entries) >= responseContinuationMaxSize {
		s.pruneExpiredLocked(now)
	}
	if len(s.entries) >= responseContinuationMaxSize {
		s.evictOldestLocked()
	}
	s.entries[publicID] = responseContinuationEntry{
		upstreamID: upstreamID,
		expiresAt:  now.Add(responseContinuationTTL),
	}
}

func (s *responseContinuationStore) Get(publicID string) (string, bool) {
	s.mu.RLock()
	entry, ok := s.entries[publicID]
	s.mu.RUnlock()
	if !ok {
		return "", false
	}

	now := time.Now()
	if now.After(entry.expiresAt) {
		s.mu.Lock()
		entry, ok = s.entries[publicID]
		if ok && now.After(entry.expiresAt) {
			delete(s.entries, publicID)
		}
		s.mu.Unlock()
		return "", false
	}

	return entry.upstreamID, true
}

func (s *responseContinuationStore) pruneExpiredLocked(now time.Time) {
	for id, entry := range s.entries {
		if now.After(entry.expiresAt) {
			delete(s.entries, id)
		}
	}
}

func (s *responseContinuationStore) evictOldestLocked() {
	var oldestID string
	var oldestExpiry time.Time
	for id, entry := range s.entries {
		if oldestID == "" || entry.expiresAt.Before(oldestExpiry) {
			oldestID = id
			oldestExpiry = entry.expiresAt
		}
	}
	if oldestID != "" {
		delete(s.entries, oldestID)
	}
}

func (s *responseContinuationStore) backgroundPrune() {
	ticker := time.NewTicker(responseContinuationPruneInt)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			s.mu.Lock()
			s.pruneExpiredLocked(time.Now())
			s.mu.Unlock()
		case <-s.stopPrune:
			return
		}
	}
}

func (s *responseContinuationStore) stop() {
	close(s.stopPrune)
}

func (s *Server) validateForStreaming(req *pipeline.Request) error {
	validator, ok := s.Runner.(requestValidator)
	if !ok {
		return nil
	}
	return validator.Validate(req)
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
	piiCheckEnabled := s.DefaultPIICheckEnabled || req.PIICheckOptions != nil
	fetchInjectionCheckEnabled := s.DefaultFetchInjectionCheckEnabled || req.InjectionCheckOptions != nil
	var searchContextSize pipeline.SearchContextSize
	var userLocation *pipeline.UserLocation
	var allowedDomains []string
	if req.WebSearchOptions != nil {
		searchContextSize = pipeline.SearchContextSize(req.WebSearchOptions.SearchContextSize)
		userLocation = convertUserLocation(req.WebSearchOptions.UserLocation)
		if req.WebSearchOptions.Filters != nil {
			allowedDomains = req.WebSearchOptions.Filters.AllowedDomains
		}
	}
	log.Debugf("Request features: web_search=%v, pii_check=%v, fetch_injection_check=%v",
		webSearchEnabled, piiCheckEnabled, fetchInjectionCheckEnabled)

	pipelineReq := &pipeline.Request{
		Model:                      req.Model,
		Messages:                   convertMessages(req.Messages),
		Stream:                     req.Stream,
		Temperature:                req.Temperature,
		MaxTokens:                  req.MaxTokens,
		Format:                     pipeline.FormatChatCompletion,
		WebSearchEnabled:           webSearchEnabled,
		PIICheckEnabled:            piiCheckEnabled,
		FetchInjectionCheckEnabled: fetchInjectionCheckEnabled,
		SearchContextSize:          searchContextSize,
		UserLocation:               userLocation,
		AllowedDomains:             allowedDomains,
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
	if len(annotations) == 0 {
		annotations = pipeline.BuildAnnotations(result.SearchResults)
	}

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
			Status: fp.Status,
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

	writeJSON(w, response)
}

func (s *Server) handleStreamingChatCompletion(w http.ResponseWriter, r *http.Request, req *pipeline.Request) {
	log.Infof("Processing streaming query (model: %s)", req.Model)

	if err := s.validateForStreaming(req); err != nil {
		status, body := pipeline.ErrorResponse(err)
		jsonErrorResponse(w, status, body)
		return
	}

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

	if err := s.validateForStreaming(req); err != nil {
		status, body := pipeline.ErrorResponse(err)
		jsonErrorResponse(w, status, body)
		return
	}

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
	s.rememberResponseID(responseID, result.ID)

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
	webSearchEnabled, searchContextSize, userLocation, allowedDomains := extractResponsesWebSearchOptions(req.Tools)
	piiCheckEnabled := s.DefaultPIICheckEnabled || req.PIICheckOptions != nil
	fetchInjectionCheckEnabled := s.DefaultFetchInjectionCheckEnabled || req.InjectionCheckOptions != nil
	log.Debugf("Responses request features: web_search=%v, pii_check=%v, fetch_injection_check=%v",
		webSearchEnabled, piiCheckEnabled, fetchInjectionCheckEnabled)

	pipelineReq := &pipeline.Request{
		Model:                      req.Model,
		Input:                      req.Input,
		PreviousResponseID:         s.resolvePreviousResponseID(req.PreviousResponseID),
		Stream:                     req.Stream,
		Format:                     pipeline.FormatResponses,
		WebSearchEnabled:           webSearchEnabled,
		PIICheckEnabled:            piiCheckEnabled,
		FetchInjectionCheckEnabled: fetchInjectionCheckEnabled,
		SearchContextSize:          searchContextSize,
		UserLocation:               userLocation,
		AllowedDomains:             allowedDomains,
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

	responseID := result.ID
	if responseID == "" {
		responseID = IDPrefixResponse + uuid.New().String()[:8]
	}
	s.rememberResponseID(responseID, result.ID)

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
			Status: fp.Status,
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
		ID:        responseID,
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

	writeJSON(w, responsesResp)
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
