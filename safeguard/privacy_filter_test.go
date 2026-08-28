package safeguard

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestPrivacyFilterRedactAppliesRequestTimeout(t *testing.T) {
	startedAt := time.Now()
	var deadline time.Time
	client := &PrivacyFilterClient{
		enclave: "privacy.example.com",
		httpClient: &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			deadline, _ = req.Context().Deadline()
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(`{"detected_spans":[]}`)),
				Header:     make(http.Header),
			}, nil
		})},
	}

	redacted, err := client.Redact(context.Background(), "public search")
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if redacted != "public search" {
		t.Fatalf("got %q, want unchanged content", redacted)
	}
	if deadline.IsZero() {
		t.Fatal("expected request context to have a deadline")
	}
	remaining := deadline.Sub(startedAt)
	if remaining < safeguardRequestTimeout-time.Second || remaining > safeguardRequestTimeout+time.Second {
		t.Fatalf("expected deadline near %v, got %v", safeguardRequestTimeout, remaining)
	}
}

func TestApplyPIIPolicy(t *testing.T) {
	tests := []struct {
		name    string
		content string
		spans   map[string]string
		want    string
	}{
		{
			name:    "hiking location without person remains searchable",
			content: "hiking trails near 742 Evergreen Terrace",
			spans:   map[string]string{"private_address": "742 Evergreen Terrace"},
			want:    "hiking trails near 742 Evergreen Terrace",
		},
		{
			name:    "address paired with person is masked",
			content: "hiking trails near 742 Evergreen Terrace for John Smith",
			spans: map[string]string{
				"private_address": "742 Evergreen Terrace",
				"private_person":  "John Smith",
			},
			want: "hiking trails near for John Smith",
		},
		{
			name:    "date paired with person is masked",
			content: "records for John Smith born March 15, 1985",
			spans: map[string]string{
				"private_person": "John Smith",
				"private_date":   "March 15, 1985",
			},
			want: "records for John Smith born",
		},
		{
			name:    "email is always masked",
			content: "email john@example.com about trail conditions",
			spans:   map[string]string{"private_email": "john@example.com"},
			want:    "email about trail conditions",
		},
		{
			name:    "leading PII and adjacent whitespace are removed",
			content: "john@example.com   hiking trails",
			spans:   map[string]string{"private_email": "john@example.com"},
			want:    "hiking trails",
		},
		{
			name:    "unrelated whitespace is preserved",
			content: "find  trails\tnear\nParis john@example.com",
			spans:   map[string]string{"private_email": "john@example.com"},
			want:    "find  trails\tnear\nParis",
		},
		{
			name:    "unicode offsets are handled",
			content: "écrivez à john@example.com",
			spans:   map[string]string{"private_email": "john@example.com"},
			want:    "écrivez à",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var spans []pfSpan
			for label, text := range tc.spans {
				byteStart := strings.Index(tc.content, text)
				if byteStart == -1 {
					t.Fatalf("test span %q not found", text)
				}
				start := utf8.RuneCountInString(tc.content[:byteStart])
				spans = append(spans, pfSpan{
					Label: label,
					Start: start,
					End:   start + utf8.RuneCountInString(text),
					Text:  text,
				})
			}

			got, err := applyPIIPolicy(tc.content, spans)
			if err != nil {
				t.Fatalf("applyPIIPolicy: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestApplyPIIPolicyRejectsInvalidSelectedSpan(t *testing.T) {
	_, err := applyPIIPolicy("email john@example.com", []pfSpan{{
		Label: "private_email",
		Start: 0,
		End:   4,
		Text:  "john@example.com",
	}})
	if err == nil {
		t.Fatal("expected invalid span to fail closed")
	}
}
