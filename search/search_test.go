package search

import (
	"testing"
)

func TestNewProvider_MissingAPIKey(t *testing.T) {
	_, err := NewProvider(Config{ExaAPIKey: ""})
	if err == nil {
		t.Error("expected error when API key is empty")
	}
	if err.Error() != "no search API key configured (set EXA_API_KEY)" {
		t.Errorf("unexpected error message: %s", err.Error())
	}
}

func TestNewProvider_Success(t *testing.T) {
	provider, err := NewProvider(Config{ExaAPIKey: "test-key"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if provider == nil {
		t.Fatal("expected provider to be non-nil")
	}

	// Verify it's an ExaProvider
	exaProvider, ok := provider.(*ExaProvider)
	if !ok {
		t.Fatal("expected provider to be *ExaProvider")
	}
	if exaProvider.apiKey != "test-key" {
		t.Errorf("expected apiKey 'test-key', got '%s'", exaProvider.apiKey)
	}
}

func TestExaProvider_Name(t *testing.T) {
	provider, _ := NewProvider(Config{ExaAPIKey: "test-key"})
	if provider.Name() != "Exa" {
		t.Errorf("expected name 'Exa', got '%s'", provider.Name())
	}
}

func TestResult_Fields(t *testing.T) {
	r := Result{
		Title:         "Test Title",
		URL:           "https://example.com",
		Content:       "Test content",
		Favicon:       "https://example.com/favicon.ico",
		PublishedDate: "2024-01-01",
	}

	if r.Title != "Test Title" {
		t.Errorf("Title mismatch")
	}
	if r.URL != "https://example.com" {
		t.Errorf("URL mismatch")
	}
	if r.Content != "Test content" {
		t.Errorf("Content mismatch")
	}
	if r.Favicon != "https://example.com/favicon.ico" {
		t.Errorf("Favicon mismatch")
	}
	if r.PublishedDate != "2024-01-01" {
		t.Errorf("PublishedDate mismatch")
	}
}
