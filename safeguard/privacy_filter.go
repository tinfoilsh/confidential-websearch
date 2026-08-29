// Privacy Filter
// Implements the Checker interface using the OpenAI Privacy
// Filter token-classification model served from a Tinfoil enclave.
// Instead of an LLM prompt, it calls POST /redact and applies the PII
// policy as deterministic code on the returned spans.
//
// Full policy for reference @ bottom of file

package safeguard

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"unicode"
	"unicode/utf8"

	log "github.com/sirupsen/logrus"
	"github.com/tinfoilsh/tinfoil-go/verifier/client"
)

var pfAlwaysRedact = map[string]bool{
	"private_email":  true,
	"private_phone":  true,
	"account_number": true,
	"secret":         true,
}

var pfPersonPairRedact = map[string]bool{
	"private_address": true,
	"private_date":    true,
}

type PIIRedactor interface {
	Redact(ctx context.Context, content string) (string, error)
}

type PrivacyFilterClient struct {
	enclave      string
	apiKey       string
	httpClient   atomic.Pointer[http.Client]
	secureClient *client.SecureClient
	mu           sync.Mutex
}

// NewPrivacyFilterClient creates a client that calls the privacy filter enclave at the given
// domain. The repo is used for attestation verification (code measurement
// pinned to the GitHub repo's signed release).
func NewPrivacyFilterClient(enclave, repo, apiKey string) (*PrivacyFilterClient, error) {
	sc := client.NewSecureClient(enclave, repo)
	httpClient, err := sc.HTTPClient()
	if err != nil {
		return nil, fmt.Errorf("failed to verify privacy filter enclave: %w", err)
	}
	log.WithField("enclave", enclave).Info("privacy filter PII checker verified")
	privacyClient := &PrivacyFilterClient{
		enclave:      enclave,
		apiKey:       apiKey,
		secureClient: sc,
	}
	privacyClient.httpClient.Store(httpClient)
	return privacyClient, nil
}

type pfRedactRequest struct {
	Text string `json:"text"`
}

type pfSpan struct {
	Label string `json:"label"`
	Start int    `json:"start"`
	End   int    `json:"end"`
	Text  string `json:"text"`
}

type pfRedactResponse struct {
	DetectedSpans []pfSpan `json:"detected_spans"`
}

// reverify re-runs attestation verification and returns a fresh HTTP client
// bound to the new TLS public key. Serialized on mu so concurrent retries
// don't race while refreshing SecureClient ground truth.
func (c *PrivacyFilterClient) reverify() (*http.Client, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, err := c.secureClient.Verify(); err != nil {
		return nil, err
	}
	newClient, err := c.secureClient.HTTPClient()
	if err != nil {
		return nil, err
	}
	c.httpClient.Store(newClient)
	return newClient, nil
}

func (c *PrivacyFilterClient) currentHTTPClient() *http.Client {
	return c.httpClient.Load()
}

// Redact sends the content to /redact and masks spans selected by the
// deterministic PII policy in code.
func (c *PrivacyFilterClient) Redact(ctx context.Context, content string) (string, error) {
	spans, err := c.redact(ctx, content)
	if err != nil {
		return "", err
	}
	return applyPIIPolicy(content, spans)
}

// redact calls the privacy filter /redact endpoint and returns the detected spans.
func (c *PrivacyFilterClient) redact(ctx context.Context, text string) ([]pfSpan, error) {
	ctx, cancel := context.WithTimeout(ctx, safeguardRequestTimeout)
	defer cancel()

	body, err := json.Marshal(pfRedactRequest{Text: text})
	if err != nil {
		return nil, fmt.Errorf("marshal redact request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST",
		fmt.Sprintf("https://%s/redact", c.enclave), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.currentHTTPClient().Do(req)
	if err != nil {
		// Re-verify attestation on TLS errors (enclave may have rotated
		// its certificate after a restart) and retry once. Concurrent retries
		// serialize on mu while refreshing attestation state.
		if errors.Is(err, client.ErrCertMismatch) || errors.Is(err, client.ErrNoTLS) {
			log.WithError(err).Warn("privacy filter cert mismatch, re-verifying enclave")
			retryClient, rerr := c.reverify()
			if rerr != nil {
				return nil, fmt.Errorf("privacy filter re-verification failed: %w", rerr)
			}
			req, _ = http.NewRequestWithContext(ctx, "POST",
				fmt.Sprintf("https://%s/redact", c.enclave), bytes.NewReader(body))
			req.Header.Set("Authorization", "Bearer "+c.apiKey)
			req.Header.Set("Content-Type", "application/json")
			resp, err = retryClient.Do(req)
		}
		if err != nil {
			return nil, fmt.Errorf("privacy filter /redact request: %w", err)
		}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("privacy filter /redact returned %d: %s", resp.StatusCode, string(respBody))
	}

	var result pfRedactResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode privacy filter response: %w", err)
	}
	return result.DetectedSpans, nil
}

// applyPIIPolicy masks always-sensitive spans and masks dates or addresses
// only when the input also identifies a private person.
func applyPIIPolicy(content string, spans []pfSpan) (string, error) {
	hasPrivatePerson := false
	for _, s := range spans {
		if s.Label == "private_person" {
			hasPrivatePerson = true
			break
		}
	}

	selected := make([]pfSpan, 0, len(spans))
	for _, s := range spans {
		if pfAlwaysRedact[s.Label] || hasPrivatePerson && pfPersonPairRedact[s.Label] {
			selected = append(selected, s)
		}
	}

	runes := []rune(content)
	type redactionRange struct {
		start int
		end   int
	}
	ranges := make([]redactionRange, 0, len(selected))
	for _, s := range selected {
		start, end := s.Start, s.End
		if start < 0 || end <= start {
			return "", fmt.Errorf("invalid %s span bounds", s.Label)
		}
		if end <= len(runes) && string(runes[start:end]) == s.Text {
			ranges = append(ranges, redactionRange{start: start, end: end})
			continue
		}
		if end <= len(content) && content[start:end] == s.Text &&
			utf8.ValidString(content[:start]) && utf8.ValidString(content[:end]) {
			ranges = append(ranges, redactionRange{
				start: utf8.RuneCountInString(content[:start]),
				end:   utf8.RuneCountInString(content[:end]),
			})
			continue
		}
		return "", fmt.Errorf("could not locate %s span", s.Label)
	}

	if len(ranges) == 0 {
		return content, nil
	}

	for i := range ranges {
		if ranges[i].start == 0 {
			for ranges[i].end < len(runes) && unicode.IsSpace(runes[ranges[i].end]) {
				ranges[i].end++
			}
		}
		if ranges[i].end == len(runes) {
			for ranges[i].start > 0 && unicode.IsSpace(runes[ranges[i].start-1]) {
				ranges[i].start--
			}
		}
		if ranges[i].start > 0 && ranges[i].end < len(runes) &&
			unicode.IsSpace(runes[ranges[i].start-1]) && unicode.IsSpace(runes[ranges[i].end]) {
			for ranges[i].end < len(runes) && unicode.IsSpace(runes[ranges[i].end]) {
				ranges[i].end++
			}
		}
	}

	sort.Slice(ranges, func(i, j int) bool {
		return ranges[i].start < ranges[j].start
	})
	merged := ranges[:1]
	for _, current := range ranges[1:] {
		last := &merged[len(merged)-1]
		if current.start <= last.end {
			if current.end > last.end {
				last.end = current.end
			}
			continue
		}
		merged = append(merged, current)
	}

	var redacted strings.Builder
	redacted.Grow(len(content))
	cursor := 0
	for _, span := range merged {
		redacted.WriteString(string(runes[cursor:span.start]))
		cursor = span.end
	}
	redacted.WriteString(string(runes[cursor:]))
	return redacted.String(), nil
}

// Policy ---
// Mask text containing information that could identify, locate, or contact a specific private individual:
// - Government IDs: SSN, passport number, driver's license, voter ID, national ID
// - Financial: credit card numbers, bank account numbers, IBANs (any long numeric sequence that appears to be an account identifier)
// - Contact: personal email addresses, personal phone numbers (any format including non-standard delimiters), home addresses
// - Linkable identifiers: VINs, license plates, device serial numbers—IDs traceable to an owner
// - Identifying combinations: name + date of birth, name + address, or other combinations that together identify a specific person

// Allow: names alone, birthdates alone, business/corporate contact info, public figures' public info, generic descriptions.

// Examples ---
// Violations:
// - "SSN 123-45-6789"
// - "card number 4532-1234-5678-9012"
// - "account 259271629759774384609" (long numeric account identifier)
// - "john.doe@gmail.com"
// - "call me at 555-867-5309"
// - "phone: 009319186.0136" (non-standard format)
// - "I live at 742 Evergreen Terrace, Springfield"
// - "VIN 2HGBH41JXMN109186"
// - "John Smith, born March 15, 1985" (name + DOB combination)

// Safe:
// - "John Smith" (name alone)
// - "born March 15, 1985" (date alone)
// - "contact@acme.com" (business email)
// - "1-800-555-1234" (toll-free)
// - "CEO of Acme Corp"
// - "diagnosed with diabetes" (no identifying info attached)
