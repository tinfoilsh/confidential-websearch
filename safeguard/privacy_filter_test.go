package safeguard

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"slices"
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
	var hasDeadline bool
	var remaining time.Duration
	client := &PrivacyFilterClient{
		enclave: "privacy.example.com",
	}
	client.httpClient.Store(&http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		deadline, ok := req.Context().Deadline()
		hasDeadline = ok
		if ok {
			remaining = time.Until(deadline)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(`{"detected_spans":[]}`)),
			Header:     make(http.Header),
		}, nil
	})})

	redacted, err := client.Redact(context.Background(), "public search")
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if redacted.Text != "public search" {
		t.Fatalf("got %q, want unchanged content", redacted.Text)
	}
	if !hasDeadline {
		t.Fatal("expected request context to have a deadline")
	}
	assertRequestTimeout(t, remaining)
}

func TestPrivacyFilterRedactReturnsRemovedSpans(t *testing.T) {
	const query = "john@example.com hiking trails"
	client := &PrivacyFilterClient{enclave: "privacy.example.com"}
	client.httpClient.Store(&http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if req.Method != http.MethodPost || req.URL.Path != "/redact" {
			t.Fatalf("unexpected privacy filter request: %s %s", req.Method, req.URL)
		}
		var input pfRedactRequest
		if err := json.NewDecoder(req.Body).Decode(&input); err != nil {
			t.Fatal(err)
		}
		if input.Text != query {
			t.Fatalf("privacy filter received %q, want %q", input.Text, query)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(`{"detected_spans":[{"label":"private_email","start":0,"end":16,"text":"john@example.com"}]}`)),
			Header:     make(http.Header),
		}, nil
	})})
	got, err := client.Redact(context.Background(), query)
	if err != nil {
		t.Fatal(err)
	}
	want := []PIIRedaction{{Type: "private_email", Start: 0, End: 16}}
	if got.Text != "hiking trails" || !slices.Equal(got.Redactions, want) {
		t.Fatalf("unexpected redaction result: %+v", got)
	}
	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "john@example.com") {
		t.Fatalf("redaction result repeats removed text: %s", encoded)
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
			if got.Text != tc.want {
				t.Fatalf("got %q, want %q", got.Text, tc.want)
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

func TestApplyPIIPolicyRedactionMetadata(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		spans    []pfSpan
		wantText string
		want     []PIIRedaction
	}{
		{
			name:     "email with adjacent whitespace",
			content:  "john@example.com   hiking trails",
			spans:    []pfSpan{{Label: "private_email", Start: 0, End: 16, Text: "john@example.com"}},
			wantText: "hiking trails",
			want:     []PIIRedaction{{Type: "private_email", Start: 0, End: 16}},
		},
		{
			name:     "entire query removed",
			content:  "john@example.com",
			spans:    []pfSpan{{Label: "private_email", Start: 0, End: 16, Text: "john@example.com"}},
			wantText: "",
			want:     []PIIRedaction{{Type: "private_email", Start: 0, End: 16}},
		},
		{
			name:     "unicode code point offsets",
			content:  "𐐷 é john@example.com",
			spans:    []pfSpan{{Label: "private_email", Start: 4, End: 20, Text: "john@example.com"}},
			wantText: "𐐷 é",
			want:     []PIIRedaction{{Type: "private_email", Start: 4, End: 20}},
		},
		{
			name:     "byte offsets normalized to code points",
			content:  "𐐷 é john@example.com",
			spans:    []pfSpan{{Label: "private_email", Start: 8, End: 24, Text: "john@example.com"}},
			wantText: "𐐷 é",
			want:     []PIIRedaction{{Type: "private_email", Start: 4, End: 20}},
		},
		{
			name:    "multiple spans sorted by original position",
			content: "john@example.com jane@example.com trails",
			spans: []pfSpan{
				{Label: "private_email", Start: 17, End: 33, Text: "jane@example.com"},
				{Label: "private_email", Start: 0, End: 16, Text: "john@example.com"},
			},
			wantText: "trails",
			want: []PIIRedaction{
				{Type: "private_email", Start: 0, End: 16},
				{Type: "private_email", Start: 17, End: 33},
			},
		},
		{
			name:    "overlapping spans retain categories",
			content: "secret-token",
			spans: []pfSpan{
				{Label: "account_number", Start: 7, End: 12, Text: "token"},
				{Label: "secret", Start: 0, End: 12, Text: "secret-token"},
			},
			wantText: "",
			want: []PIIRedaction{
				{Type: "secret", Start: 0, End: 12},
				{Type: "account_number", Start: 7, End: 12},
			},
		},
		{
			name:     "names alone are not masked",
			content:  "John Smith",
			spans:    []pfSpan{{Label: "private_person", Start: 0, End: 10, Text: "John Smith"}},
			wantText: "John Smith",
		},
		{
			name:     "date alone is not masked",
			content:  "March 15, 1985",
			spans:    []pfSpan{{Label: "private_date", Start: 0, End: 14, Text: "March 15, 1985"}},
			wantText: "March 15, 1985",
		},
		{
			name:    "only date in identifying combination is reported",
			content: "John Smith March 15, 1985",
			spans: []pfSpan{
				{Label: "private_person", Start: 0, End: 10, Text: "John Smith"},
				{Label: "private_date", Start: 11, End: 25, Text: "March 15, 1985"},
			},
			wantText: "John Smith",
			want:     []PIIRedaction{{Type: "private_date", Start: 11, End: 25}},
		},
		{
			name:     "no detections",
			content:  "hiking trails",
			wantText: "hiking trails",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := applyPIIPolicy(tc.content, tc.spans)
			if err != nil {
				t.Fatal(err)
			}
			if got.Text != tc.wantText || !slices.Equal(got.Redactions, tc.want) {
				t.Fatalf("got %+v, want text %q and spans %+v", got, tc.wantText, tc.want)
			}
			if got.Redactions == nil {
				t.Fatal("expected an empty list, not null, when nothing is masked")
			}
		})
	}
}

func TestApplyPIIPolicyInvalidSpansReturnNoPartialResult(t *testing.T) {
	const content = "john@example.com"
	for _, span := range []pfSpan{
		{Label: "private_email", Start: -1, End: 16, Text: content},
		{Label: "private_email", Start: 0, End: 0, Text: content},
		{Label: "private_email", Start: 16, End: 0, Text: content},
		{Label: "private_email", Start: 0, End: 17, Text: content},
		{Label: "private_email", Start: 0, End: 16, Text: "different"},
	} {
		valid := pfSpan{Label: "private_email", Start: 0, End: len(content), Text: content}
		got, err := applyPIIPolicy(content, []pfSpan{valid, span})
		if err == nil || got.Text != "" || len(got.Redactions) != 0 {
			t.Fatalf("invalid span %+v returned result %+v, error %v", span, got, err)
		}
	}
}
