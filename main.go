package main

import (
	"context"
	"flag"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go/v3/option"
	log "github.com/sirupsen/logrus"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/config"
	"github.com/tinfoilsh/confidential-websearch/fetch"
	"github.com/tinfoilsh/confidential-websearch/safeguard"
	"github.com/tinfoilsh/confidential-websearch/search"
	"github.com/tinfoilsh/confidential-websearch/tools"
	"github.com/tinfoilsh/confidential-websearch/usage"
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
	reporter, err := usage.NewReporter(
		strings.TrimRight(cfg.ControlPlaneURL, "/")+"/api/internal/usage-reports",
		cfg.UsageReporterID,
		cfg.UsageReporterSecret,
	)
	if err != nil {
		log.Fatalf("Failed to create usage reporter: %v", err)
	}
	defer reporter.Close(context.Background())

	var (
		svc          *tools.Service
		searcherName string
	)

	if isLocalTestMode() {
		svc = newLocalTestService()
		searcherName = "local-test"
	} else {
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
		svc = tools.NewService(searcher, fetcher, safeguardClient)
		searcherName = searcher.Name()
	}

	handler := mcp.NewStreamableHTTPHandler(func(r *http.Request) *mcp.Server {
		return newMCPServer(svc, cfg, reporter, r)
	}, &mcp.StreamableHTTPOptions{Stateless: true})

	mux := http.NewServeMux()
	mux.Handle("/mcp", handler)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})

	httpServer := &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      mux,
		ReadTimeout:  5 * time.Minute,
		WriteTimeout: 0,
	}

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		log.Infof("Starting websearch server on %s (search: %s)", cfg.ListenAddr, searcherName)
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
