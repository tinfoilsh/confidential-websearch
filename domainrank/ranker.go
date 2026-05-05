package domainrank

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	log "github.com/sirupsen/logrus"
	"golang.org/x/net/publicsuffix"
)

const (
	cloudflareDatasetURL = "https://api.cloudflare.com/client/v4/radar/datasets/ranking_top_500000"

	refreshInterval = 24 * time.Hour
	requestTimeout  = 60 * time.Second

	maxResponseBytes = 64 * 1024 * 1024
)

// Ranker reports whether a hostname is in the top-500k popularity bucket.
type Ranker interface {
	InTopBucket(host string) bool
}

// NopRanker treats every domain as long-tail, preserving today's behavior
// when the Cloudflare Radar list is unavailable.
type NopRanker struct{}

func (NopRanker) InTopBucket(string) bool { return false }

// CloudflareRanker keeps an in-memory snapshot of the Cloudflare Radar
// top-500k domains list and refreshes it periodically.
type CloudflareRanker struct {
	apiToken string
	client   *http.Client
	url      string
	domains  atomic.Pointer[map[string]struct{}]
}

// NewCloudflareRanker constructs a ranker and performs an initial load. If
// the initial load fails the ranker is returned with an empty domain set so
// the caller can decide whether to start the refresh loop or fall back to
// NopRanker.
func NewCloudflareRanker(ctx context.Context, apiToken string) (*CloudflareRanker, error) {
	r := &CloudflareRanker{
		apiToken: apiToken,
		client:   &http.Client{Timeout: requestTimeout},
		url:      cloudflareDatasetURL,
	}
	empty := make(map[string]struct{})
	r.domains.Store(&empty)

	if err := r.Refresh(ctx); err != nil {
		return r, err
	}
	return r, nil
}

// Run blocks until ctx is cancelled, refreshing the dataset every 24h.
func (r *CloudflareRanker) Run(ctx context.Context) {
	ticker := time.NewTicker(refreshInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := r.Refresh(ctx); err != nil {
				log.WithError(err).Warn("domain ranker refresh failed; keeping previous snapshot")
			}
		}
	}
}

// Refresh fetches the latest dataset and atomically replaces the in-memory
// snapshot. Errors leave the previous snapshot in place.
func (r *CloudflareRanker) Refresh(ctx context.Context) error {
	domains, err := r.fetchDataset(ctx)
	if err != nil {
		return err
	}
	r.domains.Store(&domains)
	log.Infof("domain ranker loaded %d entries", len(domains))
	return nil
}

func (r *CloudflareRanker) fetchDataset(ctx context.Context) (map[string]struct{}, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+r.apiToken)
	req.Header.Set("Accept", "text/csv")

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("cloudflare radar status %d: %s", resp.StatusCode, string(body))
	}

	return parseDomainList(io.LimitReader(resp.Body, maxResponseBytes))
}

func parseDomainList(r io.Reader) (map[string]struct{}, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	domains := make(map[string]struct{}, 512_000)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.EqualFold(line, "domain") {
			continue
		}
		domains[strings.ToLower(line)] = struct{}{}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read dataset: %w", err)
	}
	if len(domains) == 0 {
		return nil, fmt.Errorf("dataset contained no domains")
	}
	return domains, nil
}

// InTopBucket reports whether host (or its registrable domain) is in the
// top-500k bucket.
func (r *CloudflareRanker) InTopBucket(host string) bool {
	host = normalizeHost(host)
	if host == "" {
		return false
	}
	domains := r.domains.Load()
	if domains == nil {
		return false
	}
	if _, ok := (*domains)[host]; ok {
		return true
	}
	registrable, err := publicsuffix.EffectiveTLDPlusOne(host)
	if err != nil || registrable == host {
		return false
	}
	_, ok := (*domains)[registrable]
	return ok
}

func normalizeHost(host string) string {
	host = strings.TrimSpace(strings.ToLower(host))
	if host == "" {
		return ""
	}
	if i := strings.IndexByte(host, ':'); i != -1 {
		host = host[:i]
	}
	host = strings.TrimPrefix(host, "www.")
	return host
}
