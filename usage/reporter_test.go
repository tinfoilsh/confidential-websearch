package usage

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	usageclient "github.com/tinfoilsh/usage-reporting-go/client"
	"github.com/tinfoilsh/usage-reporting-go/contract"
	"github.com/tinfoilsh/usage-reporting-go/usagecontext"
)

func TestReportSessionEmitsDirectCustomerRequestCounter(t *testing.T) {
	reporter, batches, closeServer := newTestReporter(t, "secret")
	defer closeServer()
	defer reporter.Close(context.Background())

	req := newUsageRequest("request-1")
	reporter.ReportSession(context.Background(), req)
	if err := reporter.client.Flush(context.Background()); err != nil {
		t.Fatalf("flush usage reporter: %v", err)
	}

	event := singleEvent(t, batches)
	if got := counterQuantity(event, contract.CounterCustomerRequests); got != 1 {
		t.Fatalf("customer request counter mismatch: got %d want 1", got)
	}
	if len(event.Meters) != 0 {
		t.Fatalf("expected no meters on websearch event, got %v", event.Meters)
	}
}

func TestReportSessionUsesSignedUsageContextCustomerRequestCount(t *testing.T) {
	reporter, batches, closeServer := newTestReporter(t, "secret")
	defer closeServer()
	defer reporter.Close(context.Background())

	req := newUsageRequest("request-1")
	customerRequests := int64(0)
	if err := usagecontext.SetHeaders(req.Header, usagecontext.Context{
		RootRequestID:        "root-request-1",
		ParentService:        "router",
		CustomerRequestCount: &customerRequests,
		IssuedAt:             time.Now().UTC(),
	}, "secret"); err != nil {
		t.Fatalf("set usage context headers: %v", err)
	}

	reporter.ReportSession(context.Background(), req)
	if err := reporter.client.Flush(context.Background()); err != nil {
		t.Fatalf("flush usage reporter: %v", err)
	}

	event := singleEvent(t, batches)
	if got := counterQuantity(event, contract.CounterCustomerRequests); got != 0 {
		t.Fatalf("customer request counter mismatch: got %d want 0", got)
	}
	if len(event.Meters) != 0 {
		t.Fatalf("expected no meters on websearch event, got %v", event.Meters)
	}
	if got := event.Attributes["root_request_id"]; got != "root-request-1" {
		t.Fatalf("root request attribute mismatch: got %q", got)
	}
	if got := event.Attributes["parent_service"]; got != "router" {
		t.Fatalf("parent service attribute mismatch: got %q", got)
	}
}

func TestReportSessionFallsBackOnInvalidUsageContext(t *testing.T) {
	reporter, batches, closeServer := newTestReporter(t, "secret")
	defer closeServer()
	defer reporter.Close(context.Background())

	req := newUsageRequest("request-1")
	customerRequests := int64(0)
	if err := usagecontext.SetHeaders(req.Header, usagecontext.Context{
		RootRequestID:        "root-request-1",
		ParentService:        "router",
		CustomerRequestCount: &customerRequests,
		IssuedAt:             time.Now().UTC(),
	}, "wrong-secret"); err != nil {
		t.Fatalf("set usage context headers: %v", err)
	}

	reporter.ReportSession(context.Background(), req)
	if err := reporter.client.Flush(context.Background()); err != nil {
		t.Fatalf("flush usage reporter: %v", err)
	}

	event := singleEvent(t, batches)
	if got := counterQuantity(event, contract.CounterCustomerRequests); got != 1 {
		t.Fatalf("customer request counter mismatch: got %d want fallback 1", got)
	}
	if _, ok := event.Attributes["parent_service"]; ok {
		t.Fatal("invalid usage context should not add parent_service attribute")
	}
}

func TestReportSessionDeduplicatesRequestID(t *testing.T) {
	reporter, batches, closeServer := newTestReporter(t, "secret")
	defer closeServer()
	defer reporter.Close(context.Background())

	req := newUsageRequest("request-1")
	reporter.ReportSession(context.Background(), req)
	reporter.ReportSession(context.Background(), req)
	if err := reporter.client.Flush(context.Background()); err != nil {
		t.Fatalf("flush usage reporter: %v", err)
	}

	batch := singleBatch(t, batches)
	if got := len(batch.Events); got != 1 {
		t.Fatalf("expected one deduplicated event, got %d", got)
	}
}

func newTestReporter(t *testing.T, secret string) (*Reporter, <-chan contract.Batch, func()) {
	t.Helper()
	batches := make(chan contract.Batch, 4)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var batch contract.Batch
		if err := json.NewDecoder(r.Body).Decode(&batch); err != nil {
			t.Errorf("decode usage batch: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		batches <- batch
		w.WriteHeader(http.StatusOK)
	}))

	reporter := &Reporter{
		client: usageclient.New(usageclient.Config{
			Endpoint:      server.URL,
			Reporter:      contract.Reporter{ID: "reporter", Service: "websearch"},
			Secret:        secret,
			FlushInterval: time.Hour,
		}),
		usageContextSecret: secret,
		reportedAt:         make(map[string]time.Time),
	}
	return reporter, batches, server.Close
}

func newUsageRequest(requestID string) *http.Request {
	req := httptest.NewRequest(http.MethodPost, "/mcp", nil)
	req.Header.Set(headerRequestID, requestID)
	req.Header.Set(headerModel, "gpt-oss-120b")
	req.Header.Set(headerRoute, "/v1/chat/completions")
	req.Header.Set(headerStreaming, "true")
	req.Header.Set("Authorization", "Bearer tk_test")
	return req
}

func singleEvent(t *testing.T, batches <-chan contract.Batch) contract.Event {
	t.Helper()
	batch := singleBatch(t, batches)
	if len(batch.Events) != 1 {
		t.Fatalf("expected one event, got %d", len(batch.Events))
	}
	return batch.Events[0]
}

func singleBatch(t *testing.T, batches <-chan contract.Batch) contract.Batch {
	t.Helper()
	select {
	case batch := <-batches:
		return batch
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for usage batch")
	}
	return contract.Batch{}
}

func counterQuantity(event contract.Event, name string) int64 {
	for _, counter := range event.Counters {
		if counter.Name == name {
			return counter.Quantity
		}
	}
	return 0
}

