package api

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/google/uuid"
	"github.com/openai/openai-go/v2/option"
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
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": message})
}

func jsonErrorResponse(w http.ResponseWriter, code int, body map[string]interface{}) {
	log.WithField("code", code).Warn("error response")
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(body)
}

func parseRequestBody(r *http.Request, v interface{}) error {
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
			Role:            msg.Role,
			Content:         msg.Content,
			Annotations:     msg.Annotations, // Same type due to alias
			SearchReasoning: msg.SearchReasoning,
		}
		// Convert reasoning items from API format to pipeline format
		for _, ri := range msg.ReasoningItems {
			pRI := pipeline.ReasoningItem{
				ID:   ri.ID,
				Type: ri.Type,
			}
			for _, s := range ri.Summary {
				pRI.Summary = append(pRI.Summary, pipeline.ReasoningSummaryPart{
					Type: s.Type,
					Text: s.Text,
				})
			}
			result[i].ReasoningItems = append(result[i].ReasoningItems, pRI)
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
				Type:          "url_citation",
				URL:           r.URL,
				Title:         r.Title,
				Content:       r.Content,
				PublishedDate: r.PublishedDate,
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

// extractUserQuery returns the content of the last non-empty user message, or empty string if none
func extractUserQuery(messages []pipeline.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" && messages[i].Content != "" {
			return messages[i].Content
		}
	}
	return ""
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

	pipelineReq := &pipeline.Request{
		Model:       req.Model,
		Messages:    convertMessages(req.Messages),
		Stream:      req.Stream,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Format:      pipeline.FormatChatCompletion,
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

	result := pctx.ResponderResult.(*pipeline.ResponderResultData)
	annotations := pipeline.BuildAnnotations(pctx.SearchResults)

	// Convert agent reasoning items to API format
	var reasoningItems []ReasoningItem
	var agentReasoning string
	if pctx.AgentResult != nil {
		agentReasoning = pctx.AgentResult.AgentReasoning
		for _, ri := range pctx.AgentResult.ReasoningItems {
			apiRI := ReasoningItem{
				ID:   ri.ID,
				Type: ri.Type,
			}
			for _, s := range ri.Summary {
				apiRI.Summary = append(apiRI.Summary, ReasoningSummaryPart{
					Type: s.Type,
					Text: s.Text,
				})
			}
			reasoningItems = append(reasoningItems, apiRI)
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
					ReasoningItems:  reasoningItems,
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

	pipelineReq := &pipeline.Request{
		Model:  req.Model,
		Input:  req.Input,
		Format: pipeline.FormatResponses,
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

	result := pctx.ResponderResult.(*pipeline.ResponderResultData)
	flatAnnotations := buildFlatAnnotations(pctx.SearchResults)

	// Convert agent reasoning items to API format
	var reasoningItems []ReasoningItem
	var agentReasoning string
	if pctx.AgentResult != nil {
		agentReasoning = pctx.AgentResult.AgentReasoning
		for _, ri := range pctx.AgentResult.ReasoningItems {
			apiRI := ReasoningItem{
				ID:   ri.ID,
				Type: ri.Type,
			}
			for _, s := range ri.Summary {
				apiRI.Summary = append(apiRI.Summary, ReasoningSummaryPart{
					Type: s.Type,
					Text: s.Text,
				})
			}
			reasoningItems = append(reasoningItems, apiRI)
		}
	}

	var output []ResponsesOutput

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
				ReasoningItems:  reasoningItems,
			},
		},
	})

	responsesResp := struct {
		ID     string            `json:"id"`
		Object string            `json:"object"`
		Model  string            `json:"model"`
		Output []ResponsesOutput `json:"output"`
	}{
		ID:     "resp_" + uuid.New().String()[:8],
		Object: "response",
		Model:  result.Model,
		Output: output,
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
