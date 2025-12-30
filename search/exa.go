package search

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// ExaProvider handles web searches using Exa AI
type ExaProvider struct {
	apiKey     string
	httpClient *http.Client
}

func (p *ExaProvider) Name() string {
	return "Exa"
}

// exaRequest represents the Exa API request body
type exaRequest struct {
	Query      string      `json:"query"`
	NumResults int         `json:"numResults"`
	Contents   exaContents `json:"contents"`
}

type exaContents struct {
	Text bool `json:"text"`
}

// exaResponse represents the Exa API response structure
type exaResponse struct {
	Results []struct {
		Title string `json:"title"`
		URL   string `json:"url"`
		Text  string `json:"text"`
	} `json:"results"`
}

// Search performs an Exa web search
func (p *ExaProvider) Search(ctx context.Context, query string, maxResults int) ([]Result, error) {
	reqBody := exaRequest{
		Query:      query,
		NumResults: maxResults,
		Contents:   exaContents{Text: true},
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.exa.ai/search", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("exa API returned status %d", resp.StatusCode)
	}

	var data exaResponse
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}

	results := make([]Result, 0, len(data.Results))
	for _, item := range data.Results {
		// Truncate content to reasonable length for context
		content := item.Text
		if len(content) > 500 {
			content = content[:500] + "..."
		}
		results = append(results, Result{
			Title:   item.Title,
			URL:     item.URL,
			Content: content,
		})
	}

	return results, nil
}
