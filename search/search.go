package search

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

// HTTP client tuning for calls to the Exa API.
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

type ContentMode string

const (
	ContentModeText       ContentMode = "text"
	ContentModeHighlights ContentMode = "highlights"
)

// Category mirrors Exa's category enum. Blank means unfiltered.
type Category string

const (
	CategoryCompany         Category = "company"
	CategoryPeople          Category = "people"
	CategoryResearchPaper   Category = "research paper"
	CategoryNews            Category = "news"
	CategoryPersonalSite    Category = "personal site"
	CategoryFinancialReport Category = "financial report"
)

type Options struct {
	MaxResults           int
	MaxContentCharacters int
	ContentMode          ContentMode
	UserLocationCountry  string
	AllowedDomains       []string
	ExcludedDomains      []string
	Category             Category
	StartPublishedDate   string
	EndPublishedDate     string
	// MaxAgeHours controls Exa's livecrawl cache. Nil = Exa default (livecrawl
	// only when no cache exists); pointer to 0 = force livecrawl every time;
	// pointer to -1 = never livecrawl (cache-only).
	MaxAgeHours *int
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
