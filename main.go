package main

import (
	"context"
	"encoding/json"
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
	"github.com/tinfoilsh/confidential-websearch/domainrank"
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
	toolDescriptions, err := config.LoadToolDescriptions()
	if err != nil {
		log.Fatalf("Failed to load tool descriptions: %v", err)
	}
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
		recorder     *LocalCallRecorder
	)

	if isLocalTestMode() {
		svc, recorder = newLocalTestService()
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

		fetcher := fetch.NewFetcher(cfg.ExaAPIKey)

		safeguardClient := safeguard.NewClient(client, cfg.SafeguardModel)

		var ranker domainrank.Ranker = domainrank.NopRanker{}
		if cfg.CloudflareAPIToken != "" {
			cfRanker, err := domainrank.NewCloudflareRanker(context.Background(), cfg.CloudflareAPIToken)
			if err != nil {
				log.WithError(err).Warn("domain ranker unavailable; injection checks will run on all fetched pages until first refresh succeeds")
			}
			if cfRanker != nil {
				go cfRanker.Run(context.Background())
				ranker = cfRanker
			}
		} else {
			log.Warn("CLOUDFLARE_API_TOKEN not set; injection checks will run on all fetched pages")
		}

		svc = tools.NewService(searcher, fetcher, safeguardClient, ranker)
		searcherName = searcher.Name()
	}

	handler := mcp.NewStreamableHTTPHandler(func(r *http.Request) *mcp.Server {
		return newMCPServer(svc, cfg, toolDescriptions, reporter, r)
	}, &mcp.StreamableHTTPOptions{Stateless: true})

	mux := http.NewServeMux()
	mux.Handle("/mcp", handler)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})

	// /debug/last-call exposes the most recent SearchArgs and fetch URL list
	// seen by the fixture-mode service. It is only mounted when
	// LOCAL_TEST_MODE=1 so production deployments never leak request shape.
	if recorder != nil {
		mux.HandleFunc("/debug/last-call", func(w http.ResponseWriter, r *http.Request) {
			searchAt, query, opts := recorder.LastSearch()
			fetchAt, urls := recorder.LastFetch()
			payload := map[string]any{
				"search": map[string]any{
					"observed_at":           searchAt.Format(time.RFC3339Nano),
					"query":                 query,
					"max_results":           opts.MaxResults,
					"max_content_chars":     opts.MaxContentCharacters,
					"content_mode":          string(opts.ContentMode),
					"user_location_country": opts.UserLocationCountry,
					"allowed_domains":       opts.AllowedDomains,
					"excluded_domains":      opts.ExcludedDomains,
					"category":              string(opts.Category),
					"start_published_date":  opts.StartPublishedDate,
					"end_published_date":    opts.EndPublishedDate,
					"max_age_hours":         opts.MaxAgeHours,
				},
				"fetch": map[string]any{
					"observed_at": fetchAt.Format(time.RFC3339Nano),
					"urls":        urls,
				},
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(payload)
		})
	}

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
