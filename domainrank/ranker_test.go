package domainrank

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestNormalizeHost(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"example.com", "example.com"},
		{"  Example.COM ", "example.com"},
		{"www.example.com", "example.com"},
		{"example.com:8443", "example.com"},
		{"www.docs.python.org", "docs.python.org"},
		{"", ""},
	}
	for _, tc := range cases {
		if got := normalizeHost(tc.in); got != tc.want {
			t.Errorf("normalizeHost(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestParseDomainList_SkipsHeaderAndBlanks(t *testing.T) {
	csv := "domain\nexample.com\n\n  WIKIPEDIA.org \nexample.com\n"
	got, err := parseDomainList(strings.NewReader(csv))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := got["example.com"]; !ok {
		t.Error("expected example.com")
	}
	if _, ok := got["wikipedia.org"]; !ok {
		t.Error("expected wikipedia.org normalized to lowercase")
	}
	if len(got) != 2 {
		t.Errorf("expected 2 unique entries, got %d", len(got))
	}
}

func TestParseDomainList_ErrorsOnEmpty(t *testing.T) {
	if _, err := parseDomainList(strings.NewReader("domain\n\n")); err == nil {
		t.Fatal("expected error for empty dataset")
	}
}

func TestInTopBucket_ExactAndRegistrableMatches(t *testing.T) {
	r := newTestRanker(map[string]struct{}{
		"python.org":  {},
		"example.com": {},
	})

	cases := []struct {
		host string
		want bool
	}{
		{"python.org", true},
		{"www.python.org", true},
		{"docs.python.org", true},  // eTLD+1 fallback
		{"example.com:443", true},
		{"evil.example", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := r.InTopBucket(tc.host); got != tc.want {
			t.Errorf("InTopBucket(%q) = %v, want %v", tc.host, got, tc.want)
		}
	}
}

func TestRefresh_ReplacesSnapshot(t *testing.T) {
	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if got := req.Header.Get("Authorization"); got != "Bearer test-token" {
			t.Errorf("unexpected auth header: %q", got)
		}
		hits.Add(1)
		fmt.Fprintln(w, "domain")
		fmt.Fprintln(w, "example.com")
		if hits.Load() > 1 {
			fmt.Fprintln(w, "second-load.example")
		}
	}))
	defer server.Close()

	r := &CloudflareRanker{
		apiToken: "test-token",
		client:   server.Client(),
		url:      server.URL,
	}
	empty := make(map[string]struct{})
	r.domains.Store(&empty)

	if err := r.Refresh(context.Background()); err != nil {
		t.Fatalf("first refresh failed: %v", err)
	}
	if !r.InTopBucket("example.com") {
		t.Fatal("expected example.com after first refresh")
	}
	if r.InTopBucket("second-load.example") {
		t.Fatal("did not expect second-load.example after first refresh")
	}

	if err := r.Refresh(context.Background()); err != nil {
		t.Fatalf("second refresh failed: %v", err)
	}
	if !r.InTopBucket("second-load.example") {
		t.Fatal("expected second-load.example after second refresh")
	}
}

func TestRefresh_NonOKResponseLeavesSnapshotIntact(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("boom"))
	}))
	defer server.Close()

	r := &CloudflareRanker{
		apiToken: "test-token",
		client:   server.Client(),
		url:      server.URL,
	}
	preloaded := map[string]struct{}{"example.com": {}}
	r.domains.Store(&preloaded)

	if err := r.Refresh(context.Background()); err == nil {
		t.Fatal("expected error from 500 response")
	}
	if !r.InTopBucket("example.com") {
		t.Fatal("previous snapshot should still serve lookups after a failed refresh")
	}
}

func TestNopRanker(t *testing.T) {
	if (NopRanker{}).InTopBucket("google.com") {
		t.Fatal("NopRanker should always return false")
	}
}

func newTestRanker(domains map[string]struct{}) *CloudflareRanker {
	r := &CloudflareRanker{}
	r.domains.Store(&domains)
	return r
}
