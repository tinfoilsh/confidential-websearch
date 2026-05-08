package main

import (
	"testing"

	"github.com/tinfoilsh/confidential-websearch/config"
)

func TestNewUsageReporterSkipsSecretsInLocalTestMode(t *testing.T) {
	reporter, err := newUsageReporter(&config.Config{}, true)
	if err != nil {
		t.Fatalf("new usage reporter: %v", err)
	}
	if reporter != nil {
		t.Fatal("expected local test mode to disable usage reporter")
	}
}

func TestNewUsageReporterRequiresReporterSecretOutsideLocalTestMode(t *testing.T) {
	_, err := newUsageReporter(&config.Config{}, false)
	if err == nil {
		t.Fatal("expected missing reporter secret to fail")
	}
	if got, want := err.Error(), "USAGE_REPORTER_SECRET is required"; got != want {
		t.Fatalf("error mismatch: got %q want %q", got, want)
	}
}

func TestNewUsageReporterRequiresContextSecretOutsideLocalTestMode(t *testing.T) {
	_, err := newUsageReporter(&config.Config{
		UsageReporterSecret: "reporter-secret",
	}, false)
	if err == nil {
		t.Fatal("expected missing usage context secret to fail")
	}
	if got, want := err.Error(), "USAGE_CONTEXT_SECRET is required"; got != want {
		t.Fatalf("error mismatch: got %q want %q", got, want)
	}
}
