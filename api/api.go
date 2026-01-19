package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/google/uuid"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/responses"
	log "github.com/sirupsen/logrus"

	"github.com/tinfoilsh/confidential-websearch/agent"
	"github.com/tinfoilsh/confidential-websearch/llm"
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
			Annotations:     convertAnnotationsToPipeline(msg.Annotations),
			SearchReasoning: msg.SearchReasoning,
		}
	}
	return result
}

// convertAnnotationsToPipeline converts API annotations to pipeline annotations
func convertAnnotationsToPipeline(anns []Annotation) []pipeline.Annotation {
	if anns == nil {
		return nil
	}
	result := make([]pipeline.Annotation, len(anns))
	for i, ann := range anns {
		result[i] = pipeline.Annotation{
			Type: ann.Type,
			URLCitation: pipeline.URLCitation{
				Title:         ann.URLCitation.Title,
				URL:           ann.URLCitation.URL,
				Content:       ann.URLCitation.Content,
				PublishedDate: ann.URLCitation.PublishedDate,
			},
		}
	}
	return result
}

// convertAnnotationsFromPipeline converts pipeline annotations to API annotations
func convertAnnotationsFromPipeline(anns []pipeline.Annotation) []Annotation {
	if anns == nil {
		return nil
	}
	result := make([]Annotation, len(anns))
	for i, ann := range anns {
		result[i] = Annotation{
			Type: ann.Type,
			URLCitation: URLCitation{
				Title:         ann.URLCitation.Title,
				URL:           ann.URLCitation.URL,
				Content:       ann.URLCitation.Content,
				PublishedDate: ann.URLCitation.PublishedDate,
			},
		}
	}
	return result
}

// buildAnnotationsFromToolCalls creates API annotations from agent tool calls
func buildAnnotationsFromToolCalls(toolCalls []agent.ToolCall) []Annotation {
	var annotations []Annotation
	for _, tc := range toolCalls {
		for _, r := range tc.Results {
			annotations = append(annotations, Annotation{
				Type: "url_citation",
				URLCitation: URLCitation{
					URL:           r.URL,
					Title:         r.Title,
					Content:       r.Content,
					PublishedDate: r.PublishedDate,
				},
			})
		}
	}
	return annotations
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

	log.Infof("Agent completed: %d tool calls", len(pctx.AgentResult.ToolCalls))

	result := pctx.ResponderResult.(*pipeline.ResponderResultData)
	annotations := buildAnnotationsFromToolCalls(pctx.AgentResult.ToolCalls)

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
					SearchReasoning: pctx.AgentResult.AgentReasoning,
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

	ctx, cancel := context.WithTimeout(r.Context(), RequestTimeout)
	defer cancel()

	// Run agent with streaming event callbacks
	searchCallsSent := make(map[string]bool)
	toolCallArgs := make(map[int]string)
	toolCallIDs := make(map[int]string)

	agentResult, agentErr := s.Agent.RunStreaming(ctx, req.Messages[len(req.Messages)-1].Content,
		func(event responses.ResponseStreamEventUnion) {
			switch event.Type {
			case "response.output_item.added":
				if event.Item.Type == "function_call" {
					fc := event.Item.AsFunctionCall()
					toolCallIDs[int(event.OutputIndex)] = fc.CallID
				}

			case "response.function_call_arguments.delta":
				idx := int(event.OutputIndex)
				args := event.Delta
				if args == "" {
					args = event.Arguments
				}
				toolCallArgs[idx] += args

				id := toolCallIDs[idx]
				if id != "" && !searchCallsSent[id] {
					var args struct {
						Query string `json:"query"`
					}
					if json.Unmarshal([]byte(toolCallArgs[idx]), &args) == nil && args.Query != "" {
						emitter.EmitSearchCall(id, "in_progress", args.Query)
						searchCallsSent[id] = true
					}
				}
			}
		})

	// Build set of successful search IDs and emit completion events
	successfulSearchIDs := make(map[string]bool)
	if agentResult != nil {
		for _, tc := range agentResult.ToolCalls {
			successfulSearchIDs[tc.ID] = true
		}
	}

	for id := range searchCallsSent {
		status := "completed"
		if !successfulSearchIDs[id] {
			status = "failed"
		}
		emitter.EmitSearchCall(id, status, "")
	}

	if agentErr != nil {
		emitter.EmitError(agentErr)
		return
	}

	log.Infof("Agent completed: %d tool calls", len(agentResult.ToolCalls))

	// Build messages and call responder using pipeline components
	messageBuilder := llm.NewMessageBuilder()
	responderMessages := messageBuilder.Build(req.Messages, agentResult)

	responder := llm.NewTinfoilResponder(&s.Client.Chat.Completions)
	params := pipeline.ResponderParams{
		Model:       req.Model,
		Messages:    responderMessages,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	}

	annotations := pipeline.BuildAnnotations(agentResult)
	if err := responder.Stream(ctx, params, annotations, agentResult.AgentReasoning, emitter, reqOpts...); err != nil {
		log.Errorf("Streaming error: %v", err)
	}
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
	flatAnnotations := buildFlatAnnotations(pctx.AgentResult.ToolCalls)

	var output []ResponsesOutput

	for _, tc := range pctx.AgentResult.ToolCalls {
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
				Type:        "output_text",
				Text:        result.Content,
				Annotations: flatAnnotations,
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
