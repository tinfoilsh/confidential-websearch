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
		agentResult, agentErr = s.agent.RunStreaming(ctx, userQuery,
			func(chunk openai.ChatCompletionChunk) {
				data, _ := json.Marshal(chunk)
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
			},
			reqOpts...)
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
		for stream.Next() {
			data, _ := json.Marshal(stream.Current())
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
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
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
