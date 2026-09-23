package usage

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	usagereporting "github.com/tinfoilsh/usage-reporting-go"
	usageclient "github.com/tinfoilsh/usage-reporting-go/client"
)

func TestReportSessionEmitsDirectCustomerRequest(t *testing.T) {
	reporter, batches, closeServer := newTestReporter(t, "secret")
	defer closeServer()
	defer reporter.Close(context.Background())

	req := newUsageRequest("request-1")
	if err := reporter.ReportSession(context.Background(), req); err != nil {
		t.Fatalf("report session: %v", err)
	}
	reporter.client.Flush(context.Background())

	event := singleEvent(t, batches)
	if event.CustomerRequests != 1 {
		t.Fatalf("customer request mismatch: got %d want 1", event.CustomerRequests)
	}
	if len(event.Meters) != 0 {
		t.Fatalf("expected no meters on websearch event, got %v", event.Meters)
	}
}

func TestReportSessionEmitsDirectCustomerRequestWithoutToolRequestID(t *testing.T) {
	reporter, batches, closeServer := newTestReporter(t, "secret")
	defer closeServer()
	defer reporter.Close(context.Background())

	req := httptest.NewRequest(http.MethodPost, "/mcp", nil)
	req.Header.Set("Authorization", "Bearer tk_test")
	if err := reporter.ReportSession(context.Background(), req); err != nil {
		t.Fatalf("report session: %v", err)
	}
	reporter.client.Flush(context.Background())

	event := singleEvent(t, batches)
	if event.RequestID == "" {
		t.Fatal("expected generated request ID")
	}
	if event.CustomerRequests != 1 {
		t.Fatalf("customer request mismatch: got %d want 1", event.CustomerRequests)
	}
	if event.APIKey != "tk_test" {
		t.Fatalf("api key mismatch: got %q want %q", event.APIKey, "tk_test")
	}
}

func TestReportSessionDefersToParentWhenContextSetsBillCustomerRequestFalse(t *testing.T) {
	reporter, batches, closeServer := newTestReporter(t, "secret")
	defer closeServer()
	defer reporter.Close(context.Background())

	req := newUsageRequest("request-1")
	if err := usagereporting.SetHeaders(req.Header, usagereporting.Context{
		ContextID:           "context-1",
		RootRequestID:       "root-request-1",
		ParentService:       usagereporting.ServiceRouter,
		APIKeyHash:          usagereporting.HashAPIKey("tk_test"),
		Depth:               1,
		BillCustomerRequest: false,
		IssuedAt:            time.Now().UTC(),
	}, "secret"); err != nil {
		t.Fatalf("set usage context headers: %v", err)
	}

	if err := reporter.ReportSession(context.Background(), req); err != nil {
		t.Fatalf("report session: %v", err)
	}
	reporter.client.Flush(context.Background())

	event := singleEvent(t, batches)
	if event.CustomerRequests != 0 {
		t.Fatalf("customer request mismatch: got %d want 0", event.CustomerRequests)
	}
	if len(event.Meters) != 0 {
		t.Fatalf("expected no meters on websearch event, got %v", event.Meters)
	}
	if got := event.Attributes["root_request_id"]; got != "root-request-1" {
		t.Fatalf("root request attribute mismatch: got %q", got)
	}
	if got := event.Attributes["context_id"]; got != "context-1" {
		t.Fatalf("context id attribute mismatch: got %q", got)
	}
	if got := event.Attributes["parent_service"]; got != usagereporting.ServiceRouter {
		t.Fatalf("parent service attribute mismatch: got %q", got)
	}
	if got := event.Attributes["depth"]; got != "1" {
		t.Fatalf("depth attribute mismatch: got %q", got)
	}
}

func TestRepeatedSearchesOnlyReportWebsearchSession(t *testing.T) {
	reporter, batches, closeServer := newTestReporter(t, "secret")
	defer closeServer()
	defer reporter.Close(context.Background())

	req := newUsageRequest("request-1")
	if err := usagereporting.SetHeaders(req.Header, usagereporting.Context{
		ContextID:           "context-1",
		RootRequestID:       "root-request-1",
		ParentService:       usagereporting.ServiceRouter,
		APIKeyHash:          usagereporting.HashAPIKey("tk_test"),
		Depth:               1,
		BillCustomerRequest: false,
		IssuedAt:            time.Now().UTC(),
	}, "secret"); err != nil {
		t.Fatalf("set usage context headers: %v", err)
	}

	for i := 0; i < 2; i++ {
		if err := reporter.ReportSession(context.Background(), req); err != nil {
			t.Fatalf("report session %d: %v", i, err)
		}
	}
	reporter.client.Flush(context.Background())

	batch := singleBatch(t, batches)
	if len(batch.Events) != 1 {
		t.Fatalf("expected one websearch session event, got %d", len(batch.Events))
	}
	for _, event := range batch.Events {
		if event.Operation.Service != usagereporting.ServiceWebsearch || event.Operation.Name != usagereporting.OperationWebsearchSession {
			t.Fatalf("operation mismatch: got %+v", event.Operation)
		}
		if event.CustomerRequests != 0 {
			t.Fatalf("session must preserve parent billing, got %d", event.CustomerRequests)
		}
		if got := event.Attributes["root_request_id"]; got != "root-request-1" {
			t.Fatalf("root request attribute mismatch: got %q", got)
		}
		if got := event.Attributes["parent_service"]; got != usagereporting.ServiceRouter {
			t.Fatalf("parent service attribute mismatch: got %q", got)
		}
	}
}

func TestReportSessionRejectsInvalidUsageContext(t *testing.T) {
	reporter, batches, closeServer := newTestReporter(t, "secret")
	defer closeServer()
	defer reporter.Close(context.Background())

	req := newUsageRequest("request-1")
	if err := usagereporting.SetHeaders(req.Header, usagereporting.Context{
		RootRequestID:       "root-request-1",
		ParentService:       usagereporting.ServiceRouter,
		BillCustomerRequest: false,
		IssuedAt:            time.Now().UTC(),
	}, "wrong-secret"); err != nil {
		t.Fatalf("set usage context headers: %v", err)
	}

	err := reporter.ReportSession(context.Background(), req)
	if err == nil {
		t.Fatal("expected report session to reject tampered usage context")
	}
	reporter.client.Flush(context.Background())
	expectNoBatch(t, batches)
}

func TestReportSessionRejectsUsageContextAPIKeyMismatch(t *testing.T) {
	reporter, batches, closeServer := newTestReporter(t, "secret")
	defer closeServer()
	defer reporter.Close(context.Background())

	req := newUsageRequest("request-1")
	if err := usagereporting.SetHeaders(req.Header, usagereporting.Context{
		RootRequestID:       "root-request-1",
		ParentService:       usagereporting.ServiceRouter,
		APIKeyHash:          usagereporting.HashAPIKey("tk_other"),
		BillCustomerRequest: false,
		IssuedAt:            time.Now().UTC(),
	}, "secret"); err != nil {
		t.Fatalf("set usage context headers: %v", err)
	}

	err := reporter.ReportSession(context.Background(), req)
	if err == nil {
		t.Fatal("expected report session to reject mismatched api key hash")
	}
	reporter.client.Flush(context.Background())
	expectNoBatch(t, batches)
}

func TestReportSessionRejectsUsageContextDepthLimit(t *testing.T) {
	reporter, batches, closeServer := newTestReporter(t, "secret")
	defer closeServer()
	defer reporter.Close(context.Background())

	req := newUsageRequest("request-1")
	if err := usagereporting.SetHeaders(req.Header, usagereporting.Context{
		RootRequestID:       "root-request-1",
		ParentService:       usagereporting.ServiceRouter,
		APIKeyHash:          usagereporting.HashAPIKey("tk_test"),
		Depth:               usageContextMaxDepth + 1,
		BillCustomerRequest: false,
		IssuedAt:            time.Now().UTC(),
	}, "secret"); err != nil {
		t.Fatalf("set usage context headers: %v", err)
	}

	err := reporter.ReportSession(context.Background(), req)
	if err == nil {
		t.Fatal("expected report session to reject excessive usage context depth")
	}
	reporter.client.Flush(context.Background())
	expectNoBatch(t, batches)
}

func TestNewReporterRequiresUsageContextSecret(t *testing.T) {
	_, err := NewReporter("https://controlplane.example/api/internal/usage-reports", "reporter", "reporter-secret", "")
	if err == nil {
		t.Fatal("expected NewReporter to reject empty usage context secret")
	}
}

func TestReportSessionDeduplicatesOnSignedContextID(t *testing.T) {
	reporter, batches, closeServer := newTestReporter(t, "secret")
	defer closeServer()
	defer reporter.Close(context.Background())

	signedCtx := usagereporting.Context{
		ContextID:           "signed-context-1",
		RootRequestID:       "signed-context-1",
		ParentService:       usagereporting.ServiceRouter,
		APIKeyHash:          usagereporting.HashAPIKey("tk_test"),
		Depth:               1,
		BillCustomerRequest: true,
		IssuedAt:            time.Now().UTC(),
	}
	for _, headerID := range []string{"header-a", "header-b"} {
		req := newUsageRequest(headerID)
		if err := usagereporting.SetHeaders(req.Header, signedCtx, "secret"); err != nil {
			t.Fatalf("set usage context headers: %v", err)
		}
		if err := reporter.ReportSession(context.Background(), req); err != nil {
			t.Fatalf("report session: %v", err)
		}
	}
	reporter.client.Flush(context.Background())

	batch := singleBatch(t, batches)
	if got := len(batch.Events); got != 1 {
		t.Fatalf("expected one event keyed by signed context id, got %d", got)
	}
	if got := batch.Events[0].RequestID; got != "signed-context-1" {
		t.Fatalf("request id mismatch: got %q want %q", got, "signed-context-1")
	}
}

func TestReportSessionBillsDirectCallerPerRequestDespiteRequestID(t *testing.T) {
	reporter, batches, closeServer := newTestReporter(t, "secret")
	defer closeServer()
	defer reporter.Close(context.Background())

	req := newUsageRequest("request-1")
	if err := reporter.ReportSession(context.Background(), req); err != nil {
		t.Fatalf("report session: %v", err)
	}
	if err := reporter.ReportSession(context.Background(), req); err != nil {
		t.Fatalf("report session: %v", err)
	}
	reporter.client.Flush(context.Background())

	batch := singleBatch(t, batches)
	if got := len(batch.Events); got != 2 {
		t.Fatalf("expected two billed events for direct caller, got %d", got)
	}
	for _, event := range batch.Events {
		if event.RequestID == "request-1" {
			t.Fatal("expected generated request ID, kept client-supplied value")
		}
		if event.CustomerRequests != 1 {
			t.Fatalf("customer request mismatch: got %d want 1", event.CustomerRequests)
		}
	}
}

func newTestReporter(t *testing.T, secret string) (*Reporter, <-chan usagereporting.Batch, func()) {
	t.Helper()
	batches := make(chan usagereporting.Batch, 4)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var batch usagereporting.Batch
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
			ReporterID:    "reporter",
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
	req.Header.Set("X-Tinfoil-Tool-Request-Id", requestID)
	req.Header.Set(headerModel, "gpt-oss-120b")
	req.Header.Set(headerRoute, "/v1/chat/completions")
	req.Header.Set(headerStreaming, "true")
	req.Header.Set("Authorization", "Bearer tk_test")
	return req
}

func singleEvent(t *testing.T, batches <-chan usagereporting.Batch) usagereporting.Event {
	t.Helper()
	batch := singleBatch(t, batches)
	if len(batch.Events) != 1 {
		t.Fatalf("expected one event, got %d", len(batch.Events))
	}
	return batch.Events[0]
}

func singleBatch(t *testing.T, batches <-chan usagereporting.Batch) usagereporting.Batch {
	t.Helper()
	select {
	case batch := <-batches:
		return batch
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for usage batch")
	}
	return usagereporting.Batch{}
}

func expectNoBatch(t *testing.T, batches <-chan usagereporting.Batch) {
	t.Helper()
	select {
	case batch := <-batches:
		t.Fatalf("expected no usage batch, got %d events", len(batch.Events))
	case <-time.After(100 * time.Millisecond):
	}
}
