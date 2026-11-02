package safeguard

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	usagereporting "github.com/tinfoilsh/usage-reporting-go"
)

func TestPrivacyFilterCustomerAttributionAndReceipt(t *testing.T) {
	for _, tc := range []struct {
		name, header, body string
		want               int
		wantError          bool
		status             int
		unknown            bool
	}{
		{"clean query billed", "1", `{"detected_spans":[]}`, 1, false, http.StatusOK, false},
		{"legacy endpoint unbilled", "", `{"detected_spans":[]}`, 0, false, http.StatusOK, false},
		{"bad spans still billed", "1", `{"detected_spans":[{"label":"secret","start":0,"end":99,"text":"bad"}]}`, 1, true, http.StatusOK, false},
		{"bad JSON still billed", "1", `{`, 1, true, http.StatusOK, false},
		{"HTTP error preserves receipt", "1", "private backend error", 1, true, http.StatusBadGateway, false},
		{"HTTP error without receipt", "", "private backend error", 0, true, http.StatusBadRequest, false},
		{"invalid receipt rejected", "invalid", `{"detected_spans":[]}`, 0, true, http.StatusOK, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := &PrivacyFilterClient{enclave: "privacy.example.com"}
			client.httpClient.Store(&http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				if req.Header.Get("Authorization") != "Bearer tk_customer" {
					t.Fatal("customer credential not forwarded")
				}
				return &http.Response{StatusCode: tc.status, Header: http.Header{billableRequestsHeader: []string{tc.header}}, Body: io.NopCloser(strings.NewReader(tc.body))}, nil
			})})
			got, err := client.Redact(context.Background(), "hiking", "Bearer tk_customer", nil)
			if (err != nil) != tc.wantError {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.unknown {
				if got.BillableRequests != nil || err.Error() != "invalid privacy filter billing receipt" {
					t.Fatalf("invalid receipt accepted: %+v, %v", got, err)
				}
				return
			}
			if got.BillableRequests == nil || *got.BillableRequests != tc.want {
				t.Fatalf("result %+v, error %v", got, err)
			}
		})
	}
}

func TestPrivacyFilterSendsUsageContext(t *testing.T) {
	usageContext := http.Header{
		usagereporting.HeaderContext:               []string{"encoded-context"},
		usagereporting.HeaderUsageContextSignature: []string{"signature"},
	}
	for _, sent := range []http.Header{usageContext, nil} {
		var got http.Header
		client := &PrivacyFilterClient{enclave: "privacy.example.com"}
		client.httpClient.Store(&http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			got = req.Header.Clone()
			return &http.Response{StatusCode: http.StatusOK, Header: http.Header{billableRequestsHeader: []string{"1"}}, Body: io.NopCloser(strings.NewReader(`{"detected_spans":[]}`))}, nil
		})})
		if _, err := client.Redact(context.Background(), "hiking", "Bearer tk_customer", sent); err != nil {
			t.Fatal(err)
		}
		for key := range usageContext {
			if got.Get(key) != sent.Get(key) {
				t.Fatalf("%s = %q, want %q", key, got.Get(key), sent.Get(key))
			}
		}
	}
}

func TestPrivacyFilterDoesNotFallbackToServiceCredential(t *testing.T) {
	client := &PrivacyFilterClient{enclave: "privacy.example.com"}
	called := false
	client.httpClient.Store(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		called = true
		return nil, errors.New("connection lost")
	})})
	result, err := client.Redact(context.Background(), "query", "", nil)
	if err == nil || called || result.BillableRequests == nil || *result.BillableRequests != 0 {
		t.Fatal("missing credential was not rejected before inference")
	}
	result, err = client.Redact(context.Background(), "query", "Bearer tk_customer", nil)
	if err == nil || result.BillableRequests != nil {
		t.Fatal("network failure must leave billing unknown")
	}
}
