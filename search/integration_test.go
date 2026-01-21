//go:build integration

package search

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestExaProvider_Integration_RealSearch(t *testing.T) {
	apiKey := os.Getenv("EXA_API_KEY")
	if apiKey == "" {
		t.Skip("EXA_API_KEY not set, skipping integration test")
	}

	provider, err := NewProvider(Config{ExaAPIKey: apiKey})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	results, err := provider.Search(ctx, "golang testing best practices", 3)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) == 0 {
		t.Error("expected at least one result")
	}

	for i, r := range results {
		if r.URL == "" {
			t.Errorf("result %d: URL should not be empty", i)
		}
		if r.Title == "" {
			t.Errorf("result %d: Title should not be empty", i)
		}
	}
}

func TestExaProvider_Integration_MaxResults(t *testing.T) {
	apiKey := os.Getenv("EXA_API_KEY")
	if apiKey == "" {
		t.Skip("EXA_API_KEY not set, skipping integration test")
	}

	provider, err := NewProvider(Config{ExaAPIKey: apiKey})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	results, err := provider.Search(ctx, "weather forecast", 2)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(results) > 2 {
		t.Errorf("expected at most 2 results, got %d", len(results))
	}
}

func TestExaProvider_Integration_SpecialCharacters(t *testing.T) {
	apiKey := os.Getenv("EXA_API_KEY")
	if apiKey == "" {
		t.Skip("EXA_API_KEY not set, skipping integration test")
	}

	provider, err := NewProvider(Config{ExaAPIKey: apiKey})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Test query with special characters
	_, err = provider.Search(ctx, "what is 2+2?", 1)
	if err != nil {
		t.Fatalf("search with special characters failed: %v", err)
	}
}

func TestExaProvider_Integration_LongQuery(t *testing.T) {
	apiKey := os.Getenv("EXA_API_KEY")
	if apiKey == "" {
		t.Skip("EXA_API_KEY not set, skipping integration test")
	}

	provider, err := NewProvider(Config{ExaAPIKey: apiKey})
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	longQuery := "What are the best practices for writing unit tests in Go programming language with proper mocking and table-driven tests?"
	_, err = provider.Search(ctx, longQuery, 3)
	if err != nil {
		t.Fatalf("search with long query failed: %v", err)
	}
}
