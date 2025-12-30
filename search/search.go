package search

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

// Result represents a single search result
type Result struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Content string `json:"content"`
}

// Provider defines the interface for search providers
type Provider interface {
	Search(ctx context.Context, query string, maxResults int) ([]Result, error)
	Name() string
}

// Config holds API keys for search providers
type Config struct {
	ExaAPIKey  string
	BingAPIKey string
}

// NewProvider creates a search provider based on available API keys
// Priority: Exa > Bing
func NewProvider(cfg Config) (Provider, error) {
	httpClient := &http.Client{
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 100,
			IdleConnTimeout:     90 * time.Second,
		},
		Timeout: 30 * time.Second,
	}

	if cfg.ExaAPIKey != "" {
		return &ExaProvider{
			apiKey:     cfg.ExaAPIKey,
			httpClient: httpClient,
		}, nil
	}

	if cfg.BingAPIKey != "" {
		return &BingProvider{
			apiKey:     cfg.BingAPIKey,
			httpClient: httpClient,
		}, nil
	}

	return nil, fmt.Errorf("no search API key configured (set EXA_API_KEY or BING_API_KEY)")
}
