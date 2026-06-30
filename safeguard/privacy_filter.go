// Privacy Filter

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

	log "github.com/sirupsen/logrus"
	"github.com/tinfoilsh/verifier/client"
)

// OPF categories that always trigger a block, regardless of context.
// Matches the pii_leakage policy: contact info, financial IDs, and secrets
// are blocked on their own. Names, dates, and addresses are only blocked
// in combination (person + date, person + address).
var pfAlwaysBlock = map[string]bool{
	"private_email":  true,
	"private_phone":  true,
	"account_number": true,
	"secret":         true,
}

// PrivacyFilterChecker implements the Checker interface using the OpenAI Privacy
// Filter token-classification model served from a Tinfoil enclave.
// Instead of an LLM prompt, it calls POST /redact and applies the PII
// policy as deterministic code on the returned spans.
type PrivacyFilterChecker struct {
	enclave      string
	apiKey       string
	httpClient   *http.Client
	secureClient *client.SecureClient
}

// NewPrivacyFilterChecker creates a checker that calls the privacy filter enclave at the given
// domain. The repo is used for attestation verification (code measurement
// pinned to the GitHub repo's signed release).
func NewPrivacyFilterChecker(enclave, repo, apiKey string) (*PrivacyFilterChecker, error) {
	sc := client.NewSecureClient(enclave, repo)
	httpClient, err := sc.HTTPClient()
	if err != nil {
		return nil, fmt.Errorf("failed to verify privacy filter enclave: %w", err)
	}
	log.WithField("enclave", enclave).Info("privacy filter PII checker verified")
	return &PrivacyFilterChecker{
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

// Check implements the Checker interface. The policy parameter is ignored —
// the PII policy is code, not a prompt. The content is sent to /redact and
// the returned spans are evaluated against the deterministic policy.
func (c *PrivacyFilterChecker) Check(ctx context.Context, _ string, content string) (*CheckResult, error) {
	spans, err := c.redact(ctx, content)
	if err != nil {
		return nil, err
	}
	return applyPIIPolicy(spans), nil
}

// redact calls the privacy filter /redact endpoint and returns the detected spans.
func (c *PrivacyFilterChecker) redact(ctx context.Context, text string) ([]pfSpan, error) {
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
		// its certificate after a restart) and retry once.
		if errors.Is(err, client.ErrCertMismatch) || errors.Is(err, client.ErrNoTLS) {
			log.WithError(err).Warn("privacy filter cert mismatch, re-verifying enclave")
			newClient, rerr := c.secureClient.HTTPClient()
			if rerr != nil {
				return nil, fmt.Errorf("privacy filter re-verification failed: %w", rerr)
			}
			c.httpClient = newClient
			req, _ = http.NewRequestWithContext(ctx, "POST",
				fmt.Sprintf("https://%s/redact", c.enclave), bytes.NewReader(body))
			req.Header.Set("Authorization", "Bearer "+c.apiKey)
			req.Header.Set("Content-Type", "application/json")
			resp, err = c.httpClient.Do(req)
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
