package safeguard

import (
	_ "embed"
	"fmt"

	"gopkg.in/yaml.v3"
)

//go:embed policies.yml
var policiesYAML []byte

// policiesFile mirrors the on-disk shape of policies.yml.
type policiesFile struct {
	Version  int               `yaml:"version"`
	Policies map[string]string `yaml:"policies"`
}

// PolicyVersion is the loaded policy bundle's version stamp. It is exported so
// callers can include it in log lines or classification telemetry.
var PolicyVersion int

// PIILeakagePolicy detects if text contains private personal information.
var PIILeakagePolicy string

// PromptInjectionPolicy detects prompt injection attempts in web content.
var PromptInjectionPolicy string

func init() {
	var parsed policiesFile
	if err := yaml.Unmarshal(policiesYAML, &parsed); err != nil {
		panic(fmt.Sprintf("safeguard: failed to parse embedded policies.yml: %v", err))
	}

	pii, ok := parsed.Policies["pii_leakage"]
	if !ok || pii == "" {
		panic("safeguard: policies.yml missing 'pii_leakage' policy")
	}
	injection, ok := parsed.Policies["prompt_injection"]
	if !ok || injection == "" {
		panic("safeguard: policies.yml missing 'prompt_injection' policy")
	}

	PolicyVersion = parsed.Version
	PIILeakagePolicy = pii
	PromptInjectionPolicy = injection
}
