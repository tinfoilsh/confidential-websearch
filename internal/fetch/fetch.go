package fetch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	log "github.com/sirupsen/logrus"
)

const (
	// maxContentLength caps the per-page text returned to callers.
	maxContentLength = 50000
	// maxResponseBytes caps how much of the Exa API response we read.
	maxResponseBytes = 8 * 1024 * 1024
	// fetchTimeout is the HTTP deadline for the Exa contents call.
	fetchTimeout = 30 * time.Second
	// exaContentsURL is the Exa Contents endpoint.
	exaContentsURL = "https://api.exa.ai/contents"
	// exaMaxAgeHours caps cached content age; older results trigger a fresh
	// crawl. Exa now treats `maxAgeHours` as the single freshness control
	// (the legacy `livecrawl` flag was deprecated and rejects requests that
	// set both).
	exaMaxAgeHours     = 2
	fetchProviderError = "upstream content fetch failed"
)

// FetchedPage represents a fetched URL and its text content
type FetchedPage struct {
	URL     string `json:"url"`
	Content string `json:"content"`
	Metadata
}

const (
	FetchStatusCompleted = "completed"
	FetchStatusFailed    = "failed"
)

type URLResult struct {
	URL     string `json:"url"`
	Status  string `json:"status"`
	Content string `json:"content,omitempty"`
	Error   string `json:"error,omitempty"`
	Metadata
}

// Fetcher fetches URL contents via the Exa /contents endpoint.
type Fetcher struct {
	client *http.Client
	apiURL string
	apiKey string
}

// NewFetcher creates a new URL fetcher backed by Exa Contents.
func NewFetcher(apiKey string) *Fetcher {
	return &Fetcher{
		client: &http.Client{Timeout: fetchTimeout},
		apiURL: exaContentsURL,
		apiKey: apiKey,
	}
}

type exaContentsRequest struct {
	URLs        []string       `json:"urls"`
	Text        bool           `json:"text"`
	MaxAgeHours int            `json:"maxAgeHours"`
	Extras      *ExtrasOptions `json:"extras,omitempty"`
}

type exaContentsResponse struct {
	Results []exaContentsResult `json:"results"`
}

type exaContentsResult struct {
	URL           string  `json:"url"`
	Text          string  `json:"text"`
	Title         string  `json:"title"`
	Author        string  `json:"author"`
	PublishedDate string  `json:"publishedDate"`
	Image         string  `json:"image"`
	Favicon       string  `json:"favicon"`
	Extras        *Extras `json:"extras"`
}

// FetchURLs fetches the contents of the given URLs in a single batched call.
func (f *Fetcher) FetchURLs(ctx context.Context, urls []string, extras *ExtrasOptions) []FetchedPage {
	results := f.FetchURLResults(ctx, urls, extras)
	pages := make([]FetchedPage, 0, len(results))
	for _, result := range results {
		if result.Status != FetchStatusCompleted || result.Content == "" {
			continue
		}
		pages = append(pages, FetchedPage{
			URL:      result.URL,
			Content:  result.Content,
			Metadata: result.Metadata,
		})
	}
	return pages
}

func (f *Fetcher) FetchURLResults(ctx context.Context, urls []string, extras *ExtrasOptions) []URLResult {
	results := make([]URLResult, len(urls))
	for i, u := range urls {
		results[i] = URLResult{URL: u, Status: FetchStatusFailed}
	}
	if len(urls) == 0 {
		return results
	}

	contents, err := f.callExa(ctx, urls, extras)
	if err != nil {
		log.Debug("Exa contents call failed")
		for i := range results {
			results[i].Error = fetchProviderError
		}
		return results
	}

	for i, rawURL := range urls {
		item := contents[rawURL]
		text := strings.TrimSpace(item.Text)
		if text == "" {
			results[i].Error = "empty content returned"
			continue
		}
		results[i].Status = FetchStatusCompleted
		results[i].Content = truncate(text)
		results[i].Metadata = Metadata{
			Title:         item.Title,
			Author:        item.Author,
			PublishedDate: item.PublishedDate,
			Image:         item.Image,
			Favicon:       item.Favicon,
			Extras:        item.Extras,
		}
	}
	return results
}

func (f *Fetcher) callExa(ctx context.Context, urls []string, extras *ExtrasOptions) (map[string]exaContentsResult, error) {
	reqBody := exaContentsRequest{
		URLs:        urls,
		Text:        true,
		MaxAgeHours: exaMaxAgeHours,
		Extras:      extras,
	}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, f.apiURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", f.apiKey)

	resp, err := f.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("exa API status %d", resp.StatusCode)
	}

	var exaResp exaContentsResponse
	if err := json.Unmarshal(body, &exaResp); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	contents := make(map[string]exaContentsResult, len(exaResp.Results))
	for _, item := range exaResp.Results {
		contents[item.URL] = item
	}
	return contents, nil
}

func truncate(s string) string {
	runes := []rune(s)
	if len(runes) > maxContentLength {
		return string(runes[:maxContentLength]) + "\n\n[content truncated]"
	}
	return s
}
