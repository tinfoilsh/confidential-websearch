package usage

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"
	usagereporting "github.com/tinfoilsh/usage-reporting-go"
	usageclient "github.com/tinfoilsh/usage-reporting-go/client"
)

const (
	sessionCacheTTL      = 30 * time.Minute
	usageContextMaxSkew  = 10 * time.Minute
	usageContextMaxDepth = 8
)

type Reporter struct {
	client             *usageclient.ReporterClient
	usageContextSecret string

	mu         sync.Mutex
	reportedAt map[string]time.Time
}

// NewReporter constructs a usage reporter wired to the controlplane batch
// ingestion endpoint. The two HMAC secrets are deliberately separate:
// reporterSecret authenticates outbound usage batches to the controlplane,
// while usageContextSecret verifies inbound signed billing context attached
// by upstream services (e.g. the model router) so this service knows whether
// the call has already been counted as a customer-billable request.
func NewReporter(endpoint, reporterID, reporterSecret, usageContextSecret string) (*Reporter, error) {
	if err := validateReporterEndpoint(endpoint); err != nil {
		return nil, err
	}
	if usageContextSecret == "" {
		return nil, fmt.Errorf("usage context secret is required")
	}
	return &Reporter{
		client: usageclient.New(usageclient.Config{
			Endpoint:   endpoint,
			ReporterID: reporterID,
			Secret:     reporterSecret,
		}),
		usageContextSecret: usageContextSecret,
		reportedAt:         make(map[string]time.Time),
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
//
// A signed usage-context header is optional. When absent, the call is treated
// as a direct customer-facing MCP request and billed accordingly. When the
// header is present but fails verification (bad signature, stale, malformed),
// the request is rejected: an attacker tampering with that header must not be
// allowed to silently fall through to the direct-billing default.
//
// The billing dedup key is the ContextID inside a valid signed usage context
// (set by trusted upstreams such as the model router). Every other call gets a
// fresh server-generated ID; request-id headers are never trusted, since a
// direct caller could otherwise reuse an ID to collapse many billable calls
// into the dedup window.
func (r *Reporter) ReportSession(ctx context.Context, req *http.Request) error {
	if r == nil {
		return nil
	}
	if req == nil {
		return nil
	}
	rc := contextFromRequest(req)
	now := time.Now().UTC()

	usageCtx, attributes, err := r.verifiedContext(req, rc, now)
	if err != nil {
		return err
	}

	customerRequests := int64(1)
	requestID := uuid.NewString()
	if usageCtx != nil {
		if !usageCtx.BillCustomerRequest {
			customerRequests = 0
		}
		if usageCtx.ContextID != "" {
			requestID = usageCtx.ContextID
		}
	}

	r.mu.Lock()
	for id, reportedAt := range r.reportedAt {
		if reportedAt.Before(now.Add(-sessionCacheTTL)) {
			delete(r.reportedAt, id)
		}
	}
	if _, ok := r.reportedAt[requestID]; ok {
		r.mu.Unlock()
		return nil
	}
	r.reportedAt[requestID] = now
	r.mu.Unlock()

	r.client.AddEvent(usagereporting.Event{
		RequestID:  requestID,
		OccurredAt: now,
		APIKey:     bearerToken(rc.AuthHeader),
		Operation: usagereporting.Operation{
			Service: usagereporting.ServiceWebsearch,
			Name:    usagereporting.OperationWebsearchSession,
		},
		CustomerRequests: customerRequests,
		Attributes:       attributes,
	})
	return nil
}

// ReportPIICheck records one privacy filter run. Unlike the session event it
// is billed on every call regardless of the parent's BillCustomerRequest
// flag: the filter is priced independently of web search, and the parent
// request has not already paid for it. Each run gets a fresh event ID so
// several searches inside one session are each charged.
func (r *Reporter) ReportPIICheck(ctx context.Context, req *http.Request) error {
	if r == nil || req == nil {
		return nil
	}
	rc := contextFromRequest(req)
	now := time.Now().UTC()

	_, attributes, err := r.verifiedContext(req, rc, now)
	if err != nil {
		return err
	}

	r.client.AddEvent(usagereporting.Event{
		RequestID:  uuid.NewString(),
		OccurredAt: now,
		APIKey:     bearerToken(rc.AuthHeader),
		Operation: usagereporting.Operation{
			Service: usagereporting.ServicePIIFilter,
			Name:    usagereporting.OperationPIIFilterRedact,
		},
		CustomerRequests: 1,
		Attributes:       attributes,
	})
	return nil
}

// verifiedContext parses and verifies the optional signed usage-context
// header and returns it with the shared billing attributes. A header that is
// present but invalid is an error rather than a fallthrough to direct billing.
func (r *Reporter) verifiedContext(req *http.Request, rc requestContext, now time.Time) (*usagereporting.Context, map[string]string, error) {
	attributes := map[string]string{
		"model":     rc.Model,
		"route":     rc.Route,
		"streaming": map[bool]string{true: "true", false: "false"}[rc.Streaming],
	}
	if r.usageContextSecret == "" {
		return nil, attributes, nil
	}
	usageCtx, ok, err := usagereporting.FromHeaders(req.Header, r.usageContextSecret, now, usageContextMaxSkew)
	if err != nil {
		return nil, nil, fmt.Errorf("verify usage context: %w", err)
	}
	if !ok {
		return nil, attributes, nil
	}
	if !usagereporting.VerifyAPIKeyHash(bearerToken(rc.AuthHeader), usageCtx.APIKeyHash) {
		return nil, nil, fmt.Errorf("verify usage context api key: mismatch")
	}
	if usageCtx.Depth > usageContextMaxDepth {
		return nil, nil, fmt.Errorf("verify usage context depth: %d exceeds max %d", usageCtx.Depth, usageContextMaxDepth)
	}
	if usageCtx.ContextID != "" {
		attributes["context_id"] = usageCtx.ContextID
	}
	if usageCtx.RootRequestID != "" {
		attributes["root_request_id"] = usageCtx.RootRequestID
	}
	if usageCtx.ParentService != "" {
		attributes["parent_service"] = usageCtx.ParentService
	}
	if usageCtx.Depth > 0 {
		attributes["depth"] = strconv.Itoa(usageCtx.Depth)
	}
	return &usageCtx, attributes, nil
}

func (r *Reporter) Close(ctx context.Context) {
	if r == nil || r.client == nil {
		return
	}
	r.client.Stop(ctx)
}
