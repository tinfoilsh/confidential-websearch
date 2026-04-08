package search

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

const (
	httpMaxIdleConns        = 100
	httpMaxIdleConnsPerHost = 100
	httpIdleConnTimeout     = 90 * time.Second
	httpClientTimeout       = 30 * time.Second
)

// Result represents a single search result
type Result struct {
	Title         string `json:"title"`
	URL           string `json:"url"`
	Content       string `json:"content"`
	Favicon       string `json:"favicon,omitempty"`
	PublishedDate string `json:"published_date,omitempty"`
}

type Options struct {
	MaxResults           int
	MaxContentCharacters int
	UserLocationCountry  string
	AllowedDomains       []string
}

// Provider defines the interface for search providers
type Provider interface {
	Search(ctx context.Context, query string, opts Options) ([]Result, error)
	Name() string
}

// Config holds API keys for search providers
type Config struct {
	ExaAPIKey string
}

// NewProvider creates a search provider based on available API keys
func NewProvider(cfg Config) (Provider, error) {
	if cfg.ExaAPIKey == "" {
		return nil, fmt.Errorf("no search API key configured (set EXA_API_KEY)")
	}

	httpClient := &http.Client{
		Transport: &http.Transport{
			MaxIdleConns:        httpMaxIdleConns,
			MaxIdleConnsPerHost: httpMaxIdleConnsPerHost,
			IdleConnTimeout:     httpIdleConnTimeout,
		},
		Timeout: httpClientTimeout,
	}

	return &ExaProvider{
		apiKey:     cfg.ExaAPIKey,
		httpClient: httpClient,
		baseURL:    defaultExaBaseURL,
	}, nil
}
