package main

import (
	"context"
	"flag"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/openai/openai-go/v3/option"
	log "github.com/sirupsen/logrus"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/agent"
	"github.com/tinfoilsh/confidential-websearch/api"
	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/llm"
	"github.com/tinfoilsh/confidential-websearch/pipeline"
	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
)

var verbose = flag.Bool("v", false, "enable verbose logging")

func main() {
	flag.Parse()
	if *verbose {
		log.SetLevel(log.DebugLevel)
	}

	cfg := config.Load()

	client, err := tinfoil.NewClient(option.WithAPIKey(cfg.TinfoilAPIKey))
	if err != nil {
		log.Fatalf("Failed to create Tinfoil client: %v", err)
	}

	searcher, err := search.NewProvider(search.Config{
		ExaAPIKey: cfg.ExaAPIKey,
	})
	if err != nil {
		log.Fatalf("Failed to create search provider: %v", err)
	}

	baseAgent := agent.New(client, cfg.AgentModel)
	responder := llm.NewTinfoilResponder(&client.Chat.Completions)
	messageBuilder := llm.NewMessageBuilder()
	safeguardClient := safeguard.NewClient(client, cfg.SafeguardModel)

	// Wrap agent with SafeAgent to support per-request PII filtering via tools
	agentRunner := agent.NewSafeAgent(baseAgent, safeguardClient)

	p := pipeline.NewPipeline([]pipeline.Stage{
		&pipeline.ValidateStage{},
		&pipeline.AgentStage{Agent: agentRunner},
		&pipeline.SearchStage{Searcher: searcher},
		&pipeline.FilterResultsStage{
			Checker: safeguardClient,
			Policy:  safeguard.PromptInjectionPolicy,
			Enabled: cfg.EnableInjectionCheck,
		},
		&pipeline.BuildMessagesStage{Builder: messageBuilder},
		&pipeline.ResponderStage{Responder: responder},
	}, api.RequestTimeout)

	srv := &api.Server{
		Cfg:      cfg,
		Client:   client,
		Pipeline: p,
	}

	http.HandleFunc("/v1/chat/completions", api.RecoveryMiddleware(srv.HandleChatCompletions))
	http.HandleFunc("/v1/responses", api.RecoveryMiddleware(srv.HandleResponses))
	http.HandleFunc("/health", srv.HandleHealth)
	http.HandleFunc("/", srv.HandleRoot)

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
