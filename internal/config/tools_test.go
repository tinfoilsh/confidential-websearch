package config

import (
	"strings"
	"testing"
)

func TestLoadToolDescriptions(t *testing.T) {
	descriptions, err := LoadToolDescriptions()
	if err != nil {
		t.Fatalf("LoadToolDescriptions() returned error: %v", err)
	}

	if !strings.Contains(descriptions.Search, "query") {
		t.Errorf("search description missing 'query' argument hint, got %q", descriptions.Search)
	}
	if !strings.Contains(descriptions.Search, "max_age_hours") {
		t.Errorf("search description missing 'max_age_hours' argument hint")
	}
	if !strings.Contains(descriptions.Fetch, "urls") {
		t.Errorf("fetch description missing 'urls' argument hint, got %q", descriptions.Fetch)
	}
}

func TestParseToolDescriptionsRejectsEmptySearch(t *testing.T) {
	raw := []byte("tools:\n  search:\n    description: \"\"\n  fetch:\n    description: \"fetch description\"\n")
	_, err := parseToolDescriptions(raw)
	if err == nil {
		t.Fatal("expected error for empty search description, got nil")
	}
}

func TestParseToolDescriptionsRejectsEmptyFetch(t *testing.T) {
	raw := []byte("tools:\n  search:\n    description: \"search description\"\n  fetch:\n    description: \"\"\n")
	_, err := parseToolDescriptions(raw)
	if err == nil {
		t.Fatal("expected error for empty fetch description, got nil")
	}
}

func TestParseToolDescriptionsRejectsMalformedYAML(t *testing.T) {
	raw := []byte("not: [valid: yaml")
	_, err := parseToolDescriptions(raw)
	if err == nil {
		t.Fatal("expected error for malformed YAML, got nil")
	}
}
