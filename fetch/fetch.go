package fetch

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"

	htmltomarkdown "github.com/JohannesKaufmann/html-to-markdown/v2"
	log "github.com/sirupsen/logrus"
)

const (
	maxResponseBytes = 512 * 1024 // 512KB max response body
	maxContentLength = 50000      // 50K chars max output per page
	fetchTimeout     = 10 * time.Second
)

// FetchedPage represents a fetched URL and its text content
type FetchedPage struct {
	URL     string
	Content string
}

// Fetcher fetches URL contents from user messages
type Fetcher struct {
	client *http.Client
}

// NewFetcher creates a new URL fetcher
func NewFetcher() *Fetcher {
	return &Fetcher{
		client: &http.Client{
			Timeout: fetchTimeout,
			Transport: &http.Transport{
				MaxIdleConns:        20,
				MaxIdleConnsPerHost: 5,
				IdleConnTimeout:     30 * time.Second,
			},
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				if len(via) >= 5 {
					return fmt.Errorf("too many redirects")
				}
				return nil
			},
		},
	}
}

// urlPattern matches http/https URLs in text
var urlPattern = regexp.MustCompile(`https?://[^\s<>"'\)\]\}]+`)

// ExtractURLs finds all URLs in the given text
func ExtractURLs(text string) []string {
	matches := urlPattern.FindAllString(text, -1)

	// Deduplicate and clean
	seen := make(map[string]bool)
	var unique []string
	for _, u := range matches {
		// Trim trailing punctuation that's likely not part of the URL
		u = strings.TrimRight(u, ".,;:!?")
		if !seen[u] {
			seen[u] = true
			unique = append(unique, u)
		}
	}
	return unique
}

// FetchURLs fetches the contents of the given URLs in parallel
func (f *Fetcher) FetchURLs(ctx context.Context, urls []string) []FetchedPage {
	var wg sync.WaitGroup
	var mu sync.Mutex
	var pages []FetchedPage

	for _, u := range urls {
		wg.Add(1)
		go func(rawURL string) {
			defer wg.Done()

			content, err := f.fetchURL(ctx, rawURL)
			if err != nil {
				log.Debugf("Failed to fetch %s: %v", rawURL, err)
				return
			}
			if content == "" {
				return
			}

			mu.Lock()
			pages = append(pages, FetchedPage{
				URL:     rawURL,
				Content: content,
			})
			mu.Unlock()
		}(u)
	}

	wg.Wait()
	return pages
}

// fetchURL fetches a single URL and returns its content as clean text/markdown
func (f *Fetcher) fetchURL(ctx context.Context, url string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; WebSearchBot/1.0)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain,text/markdown")

	resp, err := f.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return "", err
	}

	ct := resp.Header.Get("Content-Type")

	// Plain text or markdown: use as-is
	if strings.HasPrefix(ct, "text/plain") || strings.HasPrefix(ct, "text/markdown") {
		return truncate(strings.TrimSpace(string(body))), nil
	}

	// HTML: convert to markdown
	if strings.Contains(ct, "html") || ct == "" {
		md, err := htmltomarkdown.ConvertString(string(body))
		if err != nil {
			return "", fmt.Errorf("html-to-markdown conversion failed: %w", err)
		}
		return truncate(strings.TrimSpace(md)), nil
	}

	return "", fmt.Errorf("unsupported content type: %s", ct)
}

func truncate(s string) string {
	runes := []rune(s)
	if len(runes) > maxContentLength {
		return string(runes[:maxContentLength]) + "\n\n[content truncated]"
	}
	return s
}
