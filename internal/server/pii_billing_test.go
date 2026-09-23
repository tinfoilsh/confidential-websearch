package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/tinfoilsh/confidential-websearch/internal/config"
	"github.com/tinfoilsh/confidential-websearch/internal/safeguard"
	"github.com/tinfoilsh/confidential-websearch/internal/tools"
)

type receiptRedactor struct {
	result        safeguard.PIIRedactionResult
	err           error
	authorization string
}

func (r *receiptRedactor) Redact(_ context.Context, _, authorization string) (safeguard.PIIRedactionResult, error) {
	r.authorization = authorization
	return r.result, r.err
}

func TestMCPPreservesEndpointBillingOnSearchFailure(t *testing.T) {
	for _, tc := range []struct {
		name                                             string
		providerFails, inferenceFails, unknown, disabled bool
	}{
		{name: "clean query billed"},
		{name: "search fails after billed inference", providerFails: true},
		{name: "inference fails without charge", inferenceFails: true},
		{name: "inference response lost", inferenceFails: true, unknown: true},
		{name: "filter disabled", disabled: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			count := 1
			if tc.inferenceFails || tc.disabled {
				count = 0
			}
			filter := &receiptRedactor{result: safeguard.PIIRedactionResult{Text: "clean query", BillableRequests: &count}}
			if tc.unknown {
				filter.result.BillableRequests = nil
			}
			if tc.inferenceFails {
				filter.err = errors.New("private backend details")
			}
			searcher := &mockSearchProvider{}
			if tc.providerFails {
				searcher.err = errors.New("private provider details")
			}
			svc := tools.NewService(searcher, nil, nil, filter, nil)
			handler := mcp.NewStreamableHTTPHandler(func(r *http.Request) *mcp.Server {
				return NewMCPServer(svc, &config.Config{EnablePIICheck: !tc.disabled}, config.ToolDescriptions{}, nil, "test", r)
			}, &mcp.StreamableHTTPOptions{Stateless: true, JSONResponse: true})
			req := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search","arguments":{"query":"clean query"}}}`))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Accept", "application/json, text/event-stream")
			req.Header.Set("Authorization", "Bearer tk_customer")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, req)
			var envelope struct {
				Result mcp.CallToolResult `json:"result"`
			}
			if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
				t.Fatal(err)
			}
			if envelope.Result.IsError != (tc.inferenceFails || tc.providerFails) {
				t.Fatalf("wrong tool status: %s", response.Body.String())
			}
			structured, ok := envelope.Result.StructuredContent.(map[string]any)
			if !ok {
				t.Fatalf("missing structured billing: %s", response.Body.String())
			}
			value, present := structured["pii_filter_requests"]
			if envelope.Result.IsError {
				if _, present := structured["pii_checked"]; present {
					t.Fatal("error result must not imply the filter was unchecked")
				}
			}
			if !present || tc.unknown && value != nil || !tc.unknown && value != float64(count) {
				t.Fatalf("receipt lost: %v", structured)
			}
			if !tc.disabled && filter.authorization != "Bearer tk_customer" {
				t.Fatal("customer credential not forwarded")
			}
			if tc.disabled && filter.authorization != "" {
				t.Fatal("disabled privacy filter was invoked")
			}
			if strings.Contains(response.Body.String(), "private") || strings.Contains(response.Body.String(), "tk_customer") {
				t.Fatal("response leaked credentials or backend errors")
			}
		})
	}
}
