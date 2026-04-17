package usage

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"

	usageclient "github.com/tinfoilsh/usage-reporting-go/client"
	"github.com/tinfoilsh/usage-reporting-go/contract"
)

const sessionCacheTTL = 30 * time.Minute

type Reporter struct {
	client *usageclient.ReporterClient

	mu         sync.Mutex
	reportedAt map[string]time.Time
}

func NewReporter(endpoint, reporterID, secret string) (*Reporter, error) {
	if err := validateReporterEndpoint(endpoint); err != nil {
		return nil, err
	}
	return &Reporter{
		client: usageclient.New(usageclient.Config{
			Endpoint: endpoint,
			Reporter: contract.Reporter{
				ID:      reporterID,
				Service: "websearch",
			},
			Secret: secret,
		}),
		reportedAt: make(map[string]time.Time),
	}, nil
}

func validateReporterEndpoint(endpoint string) error {
	if endpoint == "" {
		return fmt.Errorf("usage reporter endpoint is empty")
	}
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return fmt.Errorf("invalid usage reporter endpoint %q: %w", endpoint, err)
	}
	if parsed.Scheme != "https" {
		return fmt.Errorf("usage reporter endpoint %q must use https scheme", endpoint)
	}
	if parsed.Host == "" {
		return fmt.Errorf("usage reporter endpoint %q is missing a host", endpoint)
	}
	return nil
}

// ReportSession records a single usage event for the caller identified by
// standard Tinfoil tool headers on the incoming MCP request.
func (r *Reporter) ReportSession(ctx context.Context, req *http.Request) {
	if r == nil {
		return
	}
	rc := contextFromRequest(req)
	if rc.RequestID == "" {
		return
	}
	now := time.Now().UTC()

	r.mu.Lock()
	for requestID, reportedAt := range r.reportedAt {
		if reportedAt.Before(now.Add(-sessionCacheTTL)) {
			delete(r.reportedAt, requestID)
		}
	}
	if _, ok := r.reportedAt[rc.RequestID]; ok {
		r.mu.Unlock()
		return
	}
	r.reportedAt[rc.RequestID] = now
	r.mu.Unlock()

	r.client.AddEvent(contract.Event{
		RequestID:  rc.RequestID,
		OccurredAt: now,
		APIKey:     bearerToken(rc.AuthHeader),
		Operation: contract.Operation{
			Service: "websearch",
			Name:    "session",
		},
		Meters: []contract.Meter{
			{Name: "requests", Quantity: 1},
		},
		Attributes: map[string]string{
			"model":     rc.Model,
			"route":     rc.Route,
			"streaming": map[bool]string{true: "true", false: "false"}[rc.Streaming],
		},
	})
}

func (r *Reporter) Close(ctx context.Context) error {
	if r == nil || r.client == nil {
		return nil
	}
	return r.client.Stop(ctx)
}
