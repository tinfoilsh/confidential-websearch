package search

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const defaultExaBaseURL = "https://api.exa.ai"

// ExaProvider handles web searches using Exa AI
type ExaProvider struct {
	apiKey     string
	httpClient *http.Client
	baseURL    string
}

func (p *ExaProvider) Name() string {
	return "Exa"
}

// exaRequest represents the Exa API request body
// See: https://docs.exa.ai/reference/search
type exaRequest struct {
	Query      string            `json:"query"`
	Type       string            `json:"type,omitempty"`
	NumResults int               `json:"numResults,omitempty"`
	Contents   *exaContentsParam `json:"contents,omitempty"`
}

type exaContentsParam struct {
	Text *exaTextParam `json:"text,omitempty"`
}

type exaTextParam struct {
	MaxCharacters int `json:"maxCharacters,omitempty"`
}

// exaResponse represents the Exa API response structure
type exaResponse struct {
	Results []struct {
		Title         string `json:"title"`
		URL           string `json:"url"`
		Text          string `json:"text"`
		Favicon       string `json:"favicon"`
		PublishedDate string `json:"publishedDate"`
	} `json:"results"`
	Error string `json:"error,omitempty"`
}

// Search performs an Exa web search
func (p *ExaProvider) Search(ctx context.Context, query string, maxResults int) ([]Result, error) {
	reqBody := exaRequest{
		Query:      query,
		Type:       "fast", // Use fast type to guarantee ZDR (Exa maintains the index)
		NumResults: maxResults,
		Contents: &exaContentsParam{
			Text: &exaTextParam{},
		},
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/search", bytes.NewReader(jsonBody))
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

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("exa API returned status %d: %s", resp.StatusCode, string(body))
	}

	var data exaResponse
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if data.Error != "" {
		return nil, fmt.Errorf("exa API error: %s", data.Error)
	}

	results := make([]Result, 0, len(data.Results))
	for _, item := range data.Results {
		results = append(results, Result{
			Title:         item.Title,
			URL:           item.URL,
			Content:       item.Text,
			Favicon:       item.Favicon,
			PublishedDate: item.PublishedDate,
		})
	}

	return results, nil
}
