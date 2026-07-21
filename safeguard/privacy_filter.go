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
	"strings"
	"sync"

	log "github.com/sirupsen/logrus"
	"github.com/tinfoilsh/tinfoil-go/verifier/client"
)

var pfAlwaysBlock = map[string]bool{
	"private_email":   true,
	"private_phone":   true,
	"private_address": true,
	"account_number":  true,
	"secret":          true,
}

type PrivacyFilterClient struct {
	enclave      string
	apiKey       string
	httpClient   *http.Client
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
	return &PrivacyFilterClient{
		enclave:      enclave,
		apiKey:       apiKey,
		httpClient:   httpClient,
		secureClient: sc,
	}, nil
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
// don't race on SecureClient.groundTruth or c.httpClient.
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
	c.httpClient = newClient
	return newClient, nil
}

// Check implements the Checker interface. The content is sent to /redact
// and the returned spans are evaluated against the deterministic PII policy
// in code (see applyPIIPolicy), not an LLM prompt.
func (c *PrivacyFilterClient) Check(ctx context.Context, content string) (*CheckResult, error) {
	spans, err := c.redact(ctx, content)
	if err != nil {
		return nil, err
	}
	return applyPIIPolicy(spans), nil
}

// redact calls the privacy filter /redact endpoint and returns the detected spans.
func (c *PrivacyFilterClient) redact(ctx context.Context, text string) ([]pfSpan, error) {
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

	resp, err := c.httpClient.Do(req)
	if err != nil {
		// Re-verify attestation on TLS errors (enclave may have rotated
		// its certificate after a restart) and retry once. Concurrent retries
		// serialize on mu to avoid racing on groundTruth and httpClient.
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

// applyPIIPolicy evaluates detected spans against the deterministic PII policy.
// Block if any always-block category is present, or if person + (date or address).
// The rationale lists exactly which spans tripped the policy.
func applyPIIPolicy(spans []pfSpan) *CheckResult {
	labels := make(map[string]bool)
	var blocked []pfSpan
	for _, s := range spans {
		labels[s.Label] = true
		if pfAlwaysBlock[s.Label] {
			blocked = append(blocked, s)
		}
	}

	if len(blocked) > 0 {
		parts := make([]string, len(blocked))
		for i, s := range blocked {
			parts[i] = fmt.Sprintf("%s(%q)", s.Label, s.Text)
		}
		return &CheckResult{
			Violation: true,
			Rationale: "pii: " + strings.Join(parts, ", "),
		}
	}

	hasPerson := labels["private_person"]
	hasDate := labels["private_date"]
	hasAddress := labels["private_address"]

	if hasPerson && (hasDate || hasAddress) {
		var combo []string
		if hasDate {
			combo = append(combo, "date")
		}
		if hasAddress {
			combo = append(combo, "address")
		}
		return &CheckResult{
			Violation: true,
			Rationale: fmt.Sprintf("pii: person+%s combination", strings.Join(combo, "+")),
		}
	}

	return &CheckResult{Violation: false}
}

// Policy ---
// Block text containing information that could identify, locate, or contact a specific private individual:
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
