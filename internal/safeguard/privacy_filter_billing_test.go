package safeguard

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestPrivacyFilterCustomerAttributionAndReceipt(t *testing.T) {
	for _, tc := range []struct {
		name, header, body string
		want               int
		wantError          bool
	}{
		{"clean query billed", "1", `{"detected_spans":[]}`, 1, false},
		{"legacy endpoint unbilled", "", `{"detected_spans":[]}`, 0, false},
		{"bad spans still billed", "1", `{"detected_spans":[{"label":"secret","start":0,"end":99,"text":"bad"}]}`, 1, true},
		{"bad JSON still billed", "1", `{`, 1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := &PrivacyFilterClient{enclave: "privacy.example.com"}
			client.httpClient.Store(&http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				if req.Header.Get("Authorization") != "Bearer tk_customer" {
					t.Fatal("customer credential not forwarded")
				}
				return &http.Response{StatusCode: http.StatusOK, Header: http.Header{billableRequestsHeader: []string{tc.header}}, Body: io.NopCloser(strings.NewReader(tc.body))}, nil
			})})
			got, err := client.Redact(context.Background(), "hiking", "Bearer tk_customer")
			if (err != nil) != tc.wantError || got.BillableRequests == nil || *got.BillableRequests != tc.want {
				t.Fatalf("result %+v, error %v", got, err)
			}
		})
	}
}

func TestPrivacyFilterDoesNotFallbackToServiceCredential(t *testing.T) {
	client := &PrivacyFilterClient{enclave: "privacy.example.com"}
	called := false
	client.httpClient.Store(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		called = true
		return nil, errors.New("connection lost")
	})})
	result, err := client.Redact(context.Background(), "query", "")
	if err == nil || called || result.BillableRequests == nil || *result.BillableRequests != 0 {
		t.Fatal("missing credential was not rejected before inference")
	}
	result, err = client.Redact(context.Background(), "query", "Bearer tk_customer")
	if err == nil || result.BillableRequests != nil {
		t.Fatal("network failure must leave billing unknown")
	}
}
