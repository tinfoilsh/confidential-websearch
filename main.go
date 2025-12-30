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

var (
	verbose = flag.Bool("v", false, "enable verbose logging")
	cfg     *config.Config
	client  *tinfoil.Client
	ag      *agent.Agent
)

func jsonError(w http.ResponseWriter, message string, code int) {
	log.WithField("code", code).Warn(message)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": message})
}

// IncomingRequest represents the incoming chat request
type IncomingRequest struct {
	Model         string                   `json:"model"`
	Messages      []map[string]interface{} `json:"messages"`
	Stream        bool                     `json:"stream"`
	Temperature   *float64                 `json:"temperature,omitempty"`
	MaxTokens     *int64                   `json:"max_tokens,omitempty"`
	StreamOptions map[string]interface{}   `json:"stream_options,omitempty"`
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

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

	// Use model from request for the responder
	responderModel := req.Model
	if responderModel == "" {
		jsonError(w, "model parameter is required", http.StatusBadRequest)
		return
	}

	// Extract user query (last user message)
	var userQuery string
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if role, ok := req.Messages[i]["role"].(string); ok && role == "user" {
			if content, ok := req.Messages[i]["content"].(string); ok {
				userQuery = content
				break
			}
		}
	}

	if userQuery == "" {
		jsonError(w, "no user message found", http.StatusBadRequest)
		return
	}

	log.Infof("Processing query: %s (responder: %s)", truncate(userQuery, 100), responderModel)

	// For streaming requests, set up SSE headers early
	var flusher http.Flusher
	var streamingStarted bool
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
		streamingStarted = true
	}

	// Helper to write SSE data in standard OpenAI format
	writeChunk := func(data []byte) {
		if streamingStarted {
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}

	// Helper to create a simple content chunk for status messages
	writeStatus := func(status string) {
		if !streamingStarted {
			return
		}
		// Create a minimal chunk that looks like OpenAI format
		chunk := map[string]interface{}{
			"id":           "agent-status",
			"object":       "chat.completion.chunk",
			"model":        cfg.AgentModel,
			"choices":      []map[string]interface{}{},
			"agent_status": status, // Custom field for status
		}
		data, _ := json.Marshal(chunk)
		writeChunk(data)
	}

	// Step 1: Run agent to gather search results (forward auth header)
	var agentResult *agent.Result
	var agentErr error

	if req.Stream {
		// Stream the agent's LLM call and forward chunks
		agentResult, agentErr = ag.RunStreaming(ctx, userQuery,
			func(chunk openai.ChatCompletionChunk) {
				// Forward agent chunks directly
				data, _ := json.Marshal(chunk)
				writeChunk(data)
			},
			func(status string) {
				// Send search status updates
				writeStatus(status)
			},
			reqOpts...)
	} else {
		agentResult, agentErr = ag.Run(ctx, userQuery, reqOpts...)
	}

	if agentErr != nil {
		if streamingStarted {
			// Already started streaming, send error in stream
			errChunk := map[string]interface{}{
				"error": map[string]string{"message": agentErr.Error()},
			}
			data, _ := json.Marshal(errChunk)
			writeChunk(data)
			return
		}
		jsonError(w, fmt.Sprintf("agent failed: %v", agentErr), http.StatusInternalServerError)
		return
	}
	log.Infof("Agent completed: %d tool calls, %d results", len(agentResult.ToolCalls), len(agentResult.SearchResults))

	// Step 2: Build messages for responder - preserve original context
	var messages []openai.ChatCompletionMessageParamUnion

	// Convert original messages from the request
	for _, msg := range req.Messages {
		role, _ := msg["role"].(string)
		content, _ := msg["content"].(string)
		switch role {
		case "system":
			messages = append(messages, openai.SystemMessage(content))
		case "user":
			messages = append(messages, openai.UserMessage(content))
		case "assistant":
			messages = append(messages, openai.AssistantMessage(content))
		}
	}

	// If we have search results, append tool calls and tool results
	if len(agentResult.ToolCalls) > 0 {
		// Build tool calls for the assistant message
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

		// Add assistant message with tool calls AND reasoning content
		assistantMsg := openai.ChatCompletionAssistantMessageParam{
			ToolCalls: toolCalls,
		}

		// Include agent's reasoning if available (provides context for why search was done)
		if agentResult.AgentReasoning != "" {
			assistantMsg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
				OfString: openai.Opt(agentResult.AgentReasoning),
			}
			log.Debugf("Including agent reasoning in responder context: %s", agentResult.AgentReasoning)
		}

		messages = append(messages, openai.ChatCompletionMessageParamUnion{
			OfAssistant: &assistantMsg,
		})

		// Add tool results - format nicely with numbered citations
		for i, tc := range agentResult.ToolCalls {
			var toolContent string
			// Find results for this tool call
			endIdx := len(agentResult.SearchResults)
			if i+1 < len(agentResult.ToolCalls) {
				endIdx = agentResult.ToolCalls[i+1].ResultIdx
			}

			resultNum := 1
			for j := tc.ResultIdx; j < endIdx; j++ {
				sr := agentResult.SearchResults[j]
				toolContent += fmt.Sprintf("[%d] %s\nURL: %s\n%s\n\n",
					resultNum, sr.Title, sr.URL, sr.Content)
				resultNum++
			}

			messages = append(messages, openai.ToolMessage(toolContent, tc.ID))
		}

		// Add instruction to synthesize the search results
		messages = append(messages, openai.UserMessage("Based on the search results above, please provide a helpful answer to my question. Do not attempt to search again or use other tools."))
	}

	// Debug: print messages being sent to responder
	debugJSON, _ := json.MarshalIndent(messages, "", "  ")
	log.Debugf("Messages to responder:\n%s", string(debugJSON))

	// Step 3: Call responder with model from request (forward auth header)
	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(responderModel),
		Messages: messages,
	}

	if req.Temperature != nil {
		params.Temperature = openai.Float(*req.Temperature)
	}
	if req.MaxTokens != nil {
		params.MaxTokens = openai.Int(*req.MaxTokens)
	}

	if req.Stream {
		log.Debugf("Streaming response from responder (%s)", responderModel)

		stream := client.Chat.Completions.NewStreaming(ctx, params, reqOpts...)
		for stream.Next() {
			chunk := stream.Current()
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
		resp, err := client.Chat.Completions.New(ctx, params, reqOpts...)
		if err != nil {
			jsonError(w, fmt.Sprintf("responder failed: %v", err), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func main() {
	flag.Parse()
	if *verbose {
		log.SetLevel(log.DebugLevel)
	}

	// Load configuration
	cfg = config.Load()

	// Initialize Tinfoil client (no API key needed - forwarded from requests)
	// The Tinfoil client provides secure, attested communication with Tinfoil enclaves
	var err error
	client, err = tinfoil.NewClient()
	if err != nil {
		log.Fatalf("Failed to create Tinfoil client: %v", err)
	}

	// Initialize search provider (uses Exa or Bing based on available API key)
	searcher, err := search.NewProvider(search.Config{
		ExaAPIKey:  cfg.ExaAPIKey,
		BingAPIKey: cfg.BingAPIKey,
	})
	if err != nil {
		log.Fatalf("Failed to create search provider: %v", err)
	}

	// Initialize agent (uses same client, configured agent model)
	ag = agent.New(client, cfg.AgentModel, searcher)

	// Set up HTTP handlers
	http.HandleFunc("/v1/chat/completions", handleChatCompletions)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok"}`))
	})
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"service": "confidential-websearch",
			"status":  "ok",
		})
	})

	// Setup graceful shutdown
	server := &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      nil,
		ReadTimeout:  5 * time.Minute,
		WriteTimeout: 0, // Disabled for streaming
	}

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		log.Infof("Starting websearch proxy on %s", cfg.ListenAddr)
		log.Infof("Agent model: %s", cfg.AgentModel)
		log.Infof("Responder model: from request")
		log.Infof("Search provider: %s", searcher.Name())
		log.Infof("Using Tinfoil enclave: %s", client.Enclave())
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()

	<-sigChan
	log.Info("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.WithError(err).Error("Failed to gracefully shutdown server")
	}

	log.Info("Server stopped")
}
