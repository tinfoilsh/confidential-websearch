package main

import (
	"context"
	"flag"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	log "github.com/sirupsen/logrus"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/api"
	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/engine"
	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
)

var (
	verbose = flag.Bool("v", false, "enable verbose logging")
	version = "dev"
)

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

	if cfg.CloudflareAccountID == "" || cfg.CloudflareAPIToken == "" {
		log.Fatal("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set")
	}
	fetcher := fetch.NewFetcher(cfg.CloudflareAccountID, cfg.CloudflareAPIToken)

	safeguardClient := safeguard.NewClient(client, cfg.SafeguardModel)
	service := engine.NewService(
		responsesClient{inner: &client.Responses},
		searcher,
		fetcher,
		safeguardClient,
		engine.WithToolSummaryModel(cfg.ToolSummaryModel),
		engine.WithToolLoopMaxIter(cfg.ToolLoopMaxIter),
	)

	apiServer := &api.Server{
		Runner:                       service,
		DefaultPIICheckEnabled:       cfg.EnablePIICheck,
		DefaultInjectionCheckEnabled: cfg.EnableInjectionCheck,
	}

	server := mcp.NewServer(&mcp.Implementation{
		Name:    "confidential-websearch",
		Version: version,
	}, nil)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "search",
		Description: "Search the web using Exa AI. Returns titles, URLs, content snippets, and publication dates.",
	}, newSearchHandler(service, cfg))

	mcp.AddTool(server, &mcp.Tool{
		Name:        "fetch",
		Description: "Fetch web pages as markdown content via Cloudflare Browser Rendering.",
	}, newFetchHandler(service, cfg))

	handler := mcp.NewStreamableHTTPHandler(func(r *http.Request) *mcp.Server {
		return server
	}, nil)

	mux := http.NewServeMux()
	mux.Handle("/mcp", handler)
	mux.HandleFunc("/v1/chat/completions", api.RecoveryMiddleware(apiServer.HandleChatCompletions))
	mux.HandleFunc("/v1/responses", api.RecoveryMiddleware(apiServer.HandleResponses))
	mux.HandleFunc("/health", apiServer.HandleHealth)
	mux.HandleFunc("/", apiServer.HandleRoot)

	httpServer := &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      mux,
		ReadTimeout:  5 * time.Minute,
		WriteTimeout: 0,
	}

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		log.Infof("Starting websearch server on %s (search: %s)", cfg.ListenAddr, searcher.Name())
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()

	<-sigChan
	log.Info("Shutting down...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	httpServer.Shutdown(ctx)
}

type responsesClient struct {
	inner *responses.ResponseService
}

func (c responsesClient) New(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) (*responses.Response, error) {
	return c.inner.New(ctx, body, opts...)
}

func (c responsesClient) NewStreaming(ctx context.Context, body responses.ResponseNewParams, opts ...option.RequestOption) engine.ResponseStream {
	return c.inner.NewStreaming(ctx, body, opts...)
}
