package fetch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	tld "github.com/bombsimon/tld-validator"
	log "github.com/sirupsen/logrus"
)

const (
	maxContentLength   = 50000      // 50K chars max output per page
	maxResponseBytes   = 512 * 1024 // 512KB max API response body
	fetchTimeout       = 30 * time.Second
	maxConcurrentURLs  = 5
	maxURLsPerMessage  = 10
	allowedSchemeHTTP  = "http"
	allowedSchemeHTTPS = "https"
)

var blockedHostSuffixes = []string{
	".internal",
	".local",
	".localhost",
}

var blockedIPPrefixes = mustParseBlockedIPPrefixes([]string{
	"0.0.0.0/8",
	"10.0.0.0/8",
	"100.64.0.0/10",
	"127.0.0.0/8",
	"169.254.0.0/16",
	"172.16.0.0/12",
	"192.0.0.0/24",
	"192.0.2.0/24",
	"192.168.0.0/16",
	"198.18.0.0/15",
	"198.51.100.0/24",
	"203.0.113.0/24",
	"224.0.0.0/4",
	"240.0.0.0/4",
	"::/128",
	"::1/128",
	"2001:db8::/32",
	"fc00::/7",
	"fe80::/10",
	"ff00::/8",
})

// cloudflareAPIURLFormat is the URL template for the Cloudflare Browser Rendering markdown endpoint.
const cloudflareAPIURLFormat = "https://api.cloudflare.com/client/v4/accounts/%s/browser-rendering/markdown"

// FetchedPage represents a fetched URL and its text content
type FetchedPage struct {
	URL     string
	Content string
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
}

// Fetcher fetches URL contents via Cloudflare Browser Rendering API
type Fetcher struct {
	client   *http.Client
	apiURL   string
	apiToken string
}

// NewFetcher creates a new URL fetcher using Cloudflare Browser Rendering
func NewFetcher(accountID, apiToken string) *Fetcher {
	return &Fetcher{
		client: &http.Client{
			Timeout: fetchTimeout,
		},
		apiURL:   fmt.Sprintf(cloudflareAPIURLFormat, accountID),
		apiToken: apiToken,
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

func validateTargetURL(ctx context.Context, rawURL string) error {
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return fmt.Errorf("url is required")
	}

	parsed, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("parse url: %w", err)
	}
	if parsed.User != nil {
		return fmt.Errorf("embedded credentials are not allowed")
	}
	if parsed.Scheme != allowedSchemeHTTP && parsed.Scheme != allowedSchemeHTTPS {
		return fmt.Errorf("unsupported scheme %q", parsed.Scheme)
	}
	host := strings.ToLower(parsed.Hostname())
	if host == "" {
		return fmt.Errorf("url host is required")
	}
	if isBlockedHostname(host) {
		return fmt.Errorf("host %q is not allowed", host)
	}

	if addr, err := netip.ParseAddr(host); err == nil {
		if isBlockedAddr(addr) {
			return fmt.Errorf("ip address %q is not allowed", host)
		}
		return nil
	}

	addrs, err := net.DefaultResolver.LookupIPAddr(ctx, host)
	if err != nil {
		return fmt.Errorf("resolve host %q: %w", host, err)
	}
	if len(addrs) == 0 {
		return fmt.Errorf("host %q resolved to no addresses", host)
	}

	for _, addr := range addrs {
		if ip, ok := netip.AddrFromSlice(addr.IP); ok && isBlockedAddr(ip) {
			return fmt.Errorf("resolved address %q is not allowed", addr.IP.String())
		}
	}

	return nil
}

func isBlockedHostname(host string) bool {
	if host == "localhost" {
		return true
	}
	for _, suffix := range blockedHostSuffixes {
		if strings.HasSuffix(host, suffix) {
			return true
		}
	}
	return false
}

func isBlockedAddr(addr netip.Addr) bool {
	addr = addr.Unmap()
	if addr.IsPrivate() ||
		addr.IsLoopback() ||
		addr.IsMulticast() ||
		addr.IsLinkLocalMulticast() ||
		addr.IsLinkLocalUnicast() ||
		addr.IsInterfaceLocalMulticast() ||
		addr.IsUnspecified() {
		return true
	}

	for _, prefix := range blockedIPPrefixes {
		if prefix.Contains(addr) {
			return true
		}
	}

	return false
}

func mustParseBlockedIPPrefixes(prefixes []string) []netip.Prefix {
	parsed := make([]netip.Prefix, 0, len(prefixes))
	for _, prefix := range prefixes {
		parsedPrefix, err := netip.ParsePrefix(prefix)
		if err != nil {
			panic(fmt.Sprintf("invalid blocked IP prefix %q: %v", prefix, err))
		}
		parsed = append(parsed, parsedPrefix)
	}
	return parsed
}

// cloudflareRequest is the JSON body sent to the Cloudflare Browser Rendering API
type cloudflareRequest struct {
	URL                 string          `json:"url"`
	RejectResourceTypes []string        `json:"rejectResourceTypes"`
	GotoOptions         *cloudflareGoto `json:"gotoOptions,omitempty"`
}

type cloudflareGoto struct {
	WaitUntil string `json:"waitUntil"`
}

// cloudflareResponse is the JSON response from the Cloudflare Browser Rendering API
type cloudflareResponse struct {
	Success bool      `json:"success"`
	Result  string    `json:"result"`
	Errors  []cfError `json:"errors"`
}

type cfError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// FetchURLs fetches the contents of the given URLs in parallel
func (f *Fetcher) FetchURLs(ctx context.Context, urls []string) []FetchedPage {
	results := f.FetchURLResults(ctx, urls)
	pages := make([]FetchedPage, 0, len(results))
	for _, result := range results {
		if result.Status != FetchStatusCompleted || result.Content == "" {
			continue
		}
		pages = append(pages, FetchedPage{
			URL:     result.URL,
			Content: result.Content,
		})
	}
	return pages
}

func (f *Fetcher) FetchURLResults(ctx context.Context, urls []string) []URLResult {
	var wg sync.WaitGroup
	sem := make(chan struct{}, maxConcurrentURLs)
	results := make([]URLResult, len(urls))

	for i, u := range urls {
		results[i] = URLResult{
			URL:    u,
			Status: FetchStatusFailed,
		}

		// Respect context cancellation while waiting for a semaphore slot
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
		}
		if ctx.Err() != nil {
			results[i].Error = ctx.Err().Error()
			break
		}
		wg.Add(1)
		go func(index int, rawURL string) {
			defer wg.Done()
			defer func() { <-sem }()

			if err := validateTargetURL(ctx, rawURL); err != nil {
				log.Debugf("Rejected fetch URL %s: %v", rawURL, err)
				results[index].Error = err.Error()
				return
			}

			content, err := f.fetchURL(ctx, rawURL)
			if err != nil {
				log.Debugf("Failed to fetch %s: %v", rawURL, err)
				results[index].Error = err.Error()
				return
			}
			if content == "" {
				results[index].Error = "empty content returned"
				return
			}

			results[index].Status = FetchStatusCompleted
			results[index].Content = content
			results[index].Error = ""
		}(i, u)
	}

	wg.Wait()
	return results
}

// fetchURL fetches a single URL via Cloudflare Browser Rendering and returns markdown
func (f *Fetcher) fetchURL(ctx context.Context, rawURL string) (string, error) {
	reqBody := cloudflareRequest{
		URL:                 rawURL,
		RejectResourceTypes: []string{"image", "media", "font", "stylesheet"},
		GotoOptions:         &cloudflareGoto{WaitUntil: "networkidle2"},
	}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, f.apiURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+f.apiToken)
	req.Header.Set("Content-Type", "application/json")

	resp, err := f.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return "", err
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("cloudflare API status %d: %s", resp.StatusCode, string(body))
	}

	var cfResp cloudflareResponse
	if err := json.Unmarshal(body, &cfResp); err != nil {
		return "", fmt.Errorf("unmarshal response: %w", err)
	}

	if !cfResp.Success {
		if len(cfResp.Errors) > 0 {
			return "", fmt.Errorf("cloudflare API error: %s", cfResp.Errors[0].Message)
		}
		return "", fmt.Errorf("cloudflare API returned success=false")
	}

	return truncate(strings.TrimSpace(cfResp.Result)), nil
}

func truncate(s string) string {
	runes := []rune(s)
	if len(runes) > maxContentLength {
		return string(runes[:maxContentLength]) + "\n\n[content truncated]"
	}
	return s
}
