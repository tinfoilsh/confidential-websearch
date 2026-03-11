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

	tld "github.com/bombsimon/tld-validator"
	htmltomarkdown "github.com/JohannesKaufmann/html-to-markdown/v2"
	"github.com/doyensec/safeurl"
	log "github.com/sirupsen/logrus"
)

const (
	maxResponseBytes  = 512 * 1024 // 512KB max response body
	maxContentLength  = 50000      // 50K chars max output per page
	fetchTimeout      = 10 * time.Second
	maxConcurrentURLs = 5
	maxURLsPerMessage = 10
)

// httpDoer is the interface for making HTTP requests (satisfied by both http.Client and safeurl.WrappedClient)
type httpDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

// FetchedPage represents a fetched URL and its text content
type FetchedPage struct {
	URL     string
	Content string
}

// Fetcher fetches URL contents from user messages
type Fetcher struct {
	client httpDoer
}

// NewFetcher creates a new URL fetcher with SSRF protection
func NewFetcher() *Fetcher {
	config := safeurl.GetConfigBuilder().
		SetTimeout(fetchTimeout).
		SetAllowedSchemes("http", "https").
		Build()
	return &Fetcher{
		client: safeurl.Client(config),
	}
}

// newUnsafeFetcher creates a fetcher without SSRF checks (for testing with httptest)
func newUnsafeFetcher() *Fetcher {
	return &Fetcher{
		client: &http.Client{
			Timeout: fetchTimeout,
		},
	}
}

// urlPattern matches http/https URLs in text
var urlPattern = regexp.MustCompile(`https?://[^\s<>"'\)\]\}]+`)

// bareDomainPattern matches bare domains like example.com or sub.example.com/path
// Must have a valid TLD (2+ alpha chars) and not be preceded by :// (already matched above)
var bareDomainPattern = regexp.MustCompile(`(?:^|[\s(])([a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}(?:/[^\s<>"'\)\]\}]*)?)`)

// ExtractURLs finds all URLs in the given text
func ExtractURLs(text string) []string {
	seen := make(map[string]bool)
	var unique []string

	add := func(u string) {
		u = strings.TrimRight(u, ".,;:!?")
		if !seen[u] {
			seen[u] = true
			unique = append(unique, u)
		}
	}

	// Match explicit http/https URLs first
	for _, u := range urlPattern.FindAllString(text, -1) {
		add(u)
	}

	// Match bare domains (e.g. example.com, example.com/path) and prepend https://
	for _, m := range bareDomainPattern.FindAllStringSubmatch(text, -1) {
		bare := strings.TrimRight(m[1], ".,;:!?")
		if !hasValidTLD(bare) {
			continue
		}
		full := "https://" + bare
		if seen[full] || seen["http://"+bare] {
			continue
		}
		add(full)
	}

	if len(unique) > maxURLsPerMessage {
		unique = unique[:maxURLsPerMessage]
	}
	return unique
}

// hasValidTLD checks if the domain portion of a bare domain (with optional path) has a valid IANA TLD.
func hasValidTLD(bare string) bool {
	domain := bare
	if idx := strings.Index(bare, "/"); idx != -1 {
		domain = bare[:idx]
	}
	lastDot := strings.LastIndex(domain, ".")
	if lastDot == -1 {
		return false
	}
	return tld.IsValid(domain[lastDot+1:])
}

// FetchURLs fetches the contents of the given URLs in parallel
func (f *Fetcher) FetchURLs(ctx context.Context, urls []string) []FetchedPage {
	var wg sync.WaitGroup
	var mu sync.Mutex
	var pages []FetchedPage
	sem := make(chan struct{}, maxConcurrentURLs)

	for _, u := range urls {
		// Respect context cancellation while waiting for a semaphore slot
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
		}
		if ctx.Err() != nil {
			break
		}
		wg.Add(1)
		go func(rawURL string) {
			defer wg.Done()
			defer func() { <-sem }()

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
func (f *Fetcher) fetchURL(ctx context.Context, rawURL string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
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
