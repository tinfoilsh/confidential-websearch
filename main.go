package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/shared"
	log "github.com/sirupsen/logrus"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/agent"
	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/search"
)

const (
	maxRequestBodySize = 200 << 20 // 200 MB
	requestTimeout     = 2 * time.Minute
)

var verbose = flag.Bool("v", false, "enable verbose logging")

// Server holds all dependencies for the HTTP handlers
type Server struct {
	cfg    *config.Config
	client *tinfoil.Client
	agent  *agent.Agent
}

// recoveryMiddleware catches panics and returns 500 instead of crashing
func recoveryMiddleware(next http.HandlerFunc) http.HandlerFunc {
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
	Title string `json:"title"`
	URL   string `json:"url"`
}

// Annotation represents a url_citation annotation
type Annotation struct {
	Type        string      `json:"type"`
	URLCitation URLCitation `json:"url_citation"`
}

// WebSearchCall represents a search operation in streaming output
type WebSearchCall struct {
	Type   string `json:"type"`
	ID     string `json:"id"`
	Status string `json:"status"`
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
	Type  string `json:"type"`
	Title string `json:"title"`
	URL   string `json:"url"`
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

// buildAnnotations creates URL citations from search results
func buildAnnotations(toolCalls []agent.ToolCall) []Annotation {
	var annotations []Annotation
	for _, tc := range toolCalls {
		for _, r := range tc.Results {
			annotations = append(annotations, Annotation{
				Type: "url_citation",
				URLCitation: URLCitation{
					URL:   r.URL,
					Title: r.Title,
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
				Type:  "url_citation",
				URL:   r.URL,
				Title: r.Title,
			})
		}
	}
	return annotations
}

func jsonError(w http.ResponseWriter, message string, code int) {
	log.WithField("code", code).Warn(message)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": message})
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	// Only allow POST
	if r.Method != http.MethodPost {
		jsonError(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Set request timeout
	ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
	defer cancel()

	// Limit request body size to prevent DoS
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodySize)

	// Forward the Authorization header from the incoming request
	authHeader := r.Header.Get("Authorization")
	var reqOpts []option.RequestOption
	if authHeader != "" {
		reqOpts = append(reqOpts, option.WithHeader("Authorization", authHeader))
	}

	// Parse incoming request
	var req IncomingRequest
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		jsonError(w, fmt.Sprintf("failed to read request: %v", err), http.StatusBadRequest)
		return
	}
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		jsonError(w, fmt.Sprintf("failed to parse request: %v", err), http.StatusBadRequest)
		return
	}
	if req.Model == "" {
		jsonError(w, "model parameter is required", http.StatusBadRequest)
		return
	}

	// Extract user query (last user message)
	var userQuery string
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" && req.Messages[i].Content != "" {
			userQuery = req.Messages[i].Content
			break
		}
	}
	if userQuery == "" {
		jsonError(w, "no user message found", http.StatusBadRequest)
		return
	}

	log.Infof("Processing query (model: %s)", req.Model)

	// Set up streaming if requested
	var flusher http.Flusher
	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		var ok bool
		flusher, ok = w.(http.Flusher)
		if !ok {
			jsonError(w, "streaming not supported", http.StatusInternalServerError)
			return
		}
	}

	// Step 1: Run agent to gather search results
	var agentResult *agent.Result
	var agentErr error

	if req.Stream {
		searchCallsSent := make(map[string]bool)

		agentResult, agentErr = s.agent.RunStreaming(ctx, userQuery,
			func(chunk openai.ChatCompletionChunk) {
				for _, choice := range chunk.Choices {
					for _, tc := range choice.Delta.ToolCalls {
						if tc.ID != "" && !searchCallsSent[tc.ID] {
							searchEvent := WebSearchCall{
								Type:   "web_search_call",
								ID:     tc.ID,
								Status: "in_progress",
							}
							data, _ := json.Marshal(searchEvent)
							fmt.Fprintf(w, "data: %s\n\n", data)
							flusher.Flush()
							searchCallsSent[tc.ID] = true
						}
					}
				}
			},
			reqOpts...)

		for id := range searchCallsSent {
			searchEvent := WebSearchCall{
				Type:   "web_search_call",
				ID:     id,
				Status: "completed",
			}
			data, _ := json.Marshal(searchEvent)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	} else {
		agentResult, agentErr = s.agent.Run(ctx, userQuery, reqOpts...)
	}

	if agentErr != nil {
		if req.Stream {
			errData, _ := json.Marshal(map[string]interface{}{"error": map[string]string{"message": agentErr.Error()}})
			fmt.Fprintf(w, "data: %s\n\n", errData)
			flusher.Flush()
		} else {
			jsonError(w, fmt.Sprintf("agent failed: %v", agentErr), http.StatusInternalServerError)
		}
		return
	}
	log.Infof("Agent completed: %d tool calls", len(agentResult.ToolCalls))

	// Step 2: Build messages for responder - preserve original context
	var messages []openai.ChatCompletionMessageParamUnion

	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			messages = append(messages, openai.SystemMessage(msg.Content))
		case "user":
			messages = append(messages, openai.UserMessage(msg.Content))
		case "assistant":
			messages = append(messages, openai.AssistantMessage(msg.Content))
		}
	}

	// If we have search results, append tool calls and tool results
	if len(agentResult.ToolCalls) > 0 {
		toolCalls := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(agentResult.ToolCalls))
		for _, tc := range agentResult.ToolCalls {
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: tc.ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      "search",
						Arguments: fmt.Sprintf(`{"query": %q}`, tc.Query),
					},
				},
			})
		}

		assistantMsg := openai.ChatCompletionAssistantMessageParam{ToolCalls: toolCalls}
		if agentResult.AgentReasoning != "" {
			assistantMsg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
				OfString: openai.Opt(agentResult.AgentReasoning),
			}
		}
		messages = append(messages, openai.ChatCompletionMessageParamUnion{OfAssistant: &assistantMsg})

		// Add tool results
		for _, tc := range agentResult.ToolCalls {
			var toolContent string
			for i, sr := range tc.Results {
				toolContent += fmt.Sprintf("[%d] %s\nURL: %s\n%s\n\n", i+1, sr.Title, sr.URL, sr.Content)
			}
			messages = append(messages, openai.ToolMessage(toolContent, tc.ID))
		}

		messages = append(messages, openai.UserMessage("Based on the search results above, please provide a helpful answer to my question. Do not attempt to search again or use other tools."))
	}

	// Step 3: Call responder
	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(req.Model),
		Messages: messages,
	}
	if req.Temperature != nil {
		params.Temperature = openai.Float(*req.Temperature)
	}
	if req.MaxTokens != nil {
		params.MaxTokens = openai.Int(*req.MaxTokens)
	}

	if req.Stream {
		stream := s.client.Chat.Completions.NewStreaming(ctx, params, reqOpts...)

		annotations := buildAnnotations(agentResult.ToolCalls)
		annotationsSent := false

		for stream.Next() {
			chunk := stream.Current()

			if !annotationsSent && len(annotations) > 0 && len(chunk.Choices) > 0 {
				annotationsChunk := StreamingChunk{
					ID:      chunk.ID,
					Object:  "chat.completion.chunk",
					Created: chunk.Created,
					Model:   chunk.Model,
					Choices: []StreamingChoice{
						{
							Index: 0,
							Delta: StreamingDelta{Annotations: annotations},
						},
					},
				}
				data, _ := json.Marshal(annotationsChunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
				annotationsSent = true
			}

			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
		if err := stream.Err(); err != nil {
			log.Errorf("Streaming error: %v", err)
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	} else {
		resp, err := s.client.Chat.Completions.New(ctx, params, reqOpts...)
		if err != nil {
			jsonError(w, fmt.Sprintf("responder failed: %v", err), http.StatusInternalServerError)
			return
		}

		annotations := buildAnnotations(agentResult.ToolCalls)

		respJSON, err := json.Marshal(resp)
		if err != nil {
			jsonError(w, fmt.Sprintf("failed to marshal response: %v", err), http.StatusInternalServerError)
			return
		}
		var respMap map[string]interface{}
		if err := json.Unmarshal(respJSON, &respMap); err != nil {
			jsonError(w, fmt.Sprintf("failed to unmarshal response: %v", err), http.StatusInternalServerError)
			return
		}

		if choices, ok := respMap["choices"].([]interface{}); ok {
			for _, c := range choices {
				if choice, ok := c.(map[string]interface{}); ok {
					if msg, ok := choice["message"].(map[string]interface{}); ok {
						msg["annotations"] = annotations
					}
				}
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(respMap)
	}
}

func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		jsonError(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
	defer cancel()

	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodySize)

	authHeader := r.Header.Get("Authorization")
	var reqOpts []option.RequestOption
	if authHeader != "" {
		reqOpts = append(reqOpts, option.WithHeader("Authorization", authHeader))
	}

	var req ResponsesRequest
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		jsonError(w, fmt.Sprintf("failed to read request: %v", err), http.StatusBadRequest)
		return
	}
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		jsonError(w, fmt.Sprintf("failed to parse request: %v", err), http.StatusBadRequest)
		return
	}
	if req.Model == "" {
		jsonError(w, "model parameter is required", http.StatusBadRequest)
		return
	}
	if req.Input == "" {
		jsonError(w, "input parameter is required", http.StatusBadRequest)
		return
	}

	log.Infof("Processing responses request (model: %s)", req.Model)

	agentResult, agentErr := s.agent.Run(ctx, req.Input, reqOpts...)
	if agentErr != nil {
		jsonError(w, fmt.Sprintf("agent failed: %v", agentErr), http.StatusInternalServerError)
		return
	}

	var messages []openai.ChatCompletionMessageParamUnion
	messages = append(messages, openai.UserMessage(req.Input))

	if len(agentResult.ToolCalls) > 0 {
		toolCalls := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(agentResult.ToolCalls))
		for _, tc := range agentResult.ToolCalls {
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: tc.ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      "search",
						Arguments: fmt.Sprintf(`{"query": %q}`, tc.Query),
					},
				},
			})
		}

		assistantMsg := openai.ChatCompletionAssistantMessageParam{ToolCalls: toolCalls}
		if agentResult.AgentReasoning != "" {
			assistantMsg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
				OfString: openai.Opt(agentResult.AgentReasoning),
			}
		}
		messages = append(messages, openai.ChatCompletionMessageParamUnion{OfAssistant: &assistantMsg})

		for _, tc := range agentResult.ToolCalls {
			var toolContent string
			for i, sr := range tc.Results {
				toolContent += fmt.Sprintf("[%d] %s\nURL: %s\n%s\n\n", i+1, sr.Title, sr.URL, sr.Content)
			}
			messages = append(messages, openai.ToolMessage(toolContent, tc.ID))
		}

		messages = append(messages, openai.UserMessage("Based on the search results above, please provide a helpful answer to my question. Do not attempt to search again or use other tools."))
	}

	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(req.Model),
		Messages: messages,
	}

	resp, err := s.client.Chat.Completions.New(ctx, params, reqOpts...)
	if err != nil {
		jsonError(w, fmt.Sprintf("responder failed: %v", err), http.StatusInternalServerError)
		return
	}

	flatAnnotations := buildFlatAnnotations(agentResult.ToolCalls)

	var output []ResponsesOutput

	for _, tc := range agentResult.ToolCalls {
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

	if len(resp.Choices) > 0 {
		output = append(output, ResponsesOutput{
			Type:   "message",
			ID:     "msg_" + uuid.New().String()[:8],
			Status: "completed",
			Role:   "assistant",
			Content: []ResponsesContent{
				{
					Type:        "output_text",
					Text:        resp.Choices[0].Message.Content,
					Annotations: flatAnnotations,
				},
			},
		})
	}

	responsesResp := struct {
		ID     string            `json:"id"`
		Object string            `json:"object"`
		Model  string            `json:"model"`
		Output []ResponsesOutput `json:"output"`
	}{
		ID:     "resp_" + uuid.New().String()[:8],
		Object: "response",
		Model:  resp.Model,
		Output: output,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(responsesResp)
}

func main() {
	flag.Parse()
	if *verbose {
		log.SetLevel(log.DebugLevel)
	}

	cfg := config.Load()

	client, err := tinfoil.NewClient()
	if err != nil {
		log.Fatalf("Failed to create Tinfoil client: %v", err)
	}

	searcher, err := search.NewProvider(search.Config{
		ExaAPIKey:  cfg.ExaAPIKey,
		BingAPIKey: cfg.BingAPIKey,
	})
	if err != nil {
		log.Fatalf("Failed to create search provider: %v", err)
	}

	srv := &Server{
		cfg:    cfg,
		client: client,
		agent:  agent.New(client, cfg.AgentModel, searcher),
	}

	http.HandleFunc("/v1/chat/completions", recoveryMiddleware(srv.handleChatCompletions))
	http.HandleFunc("/v1/responses", recoveryMiddleware(srv.handleResponses))
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok"}`))
	})
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"service": "confidential-websearch", "status": "ok"})
	})

	server := &http.Server{
		Addr:         cfg.ListenAddr,
		ReadTimeout:  5 * time.Minute,
		WriteTimeout: 0, // Disabled for streaming
	}

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		log.Infof("Starting on %s (agent: %s, search: %s, enclave: %s)",
			cfg.ListenAddr, cfg.AgentModel, searcher.Name(), client.Enclave())
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()

	<-sigChan
	log.Info("Shutting down...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	server.Shutdown(ctx)
}
