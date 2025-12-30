package search

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// BingProvider handles web searches using Bing API
type BingProvider struct {
	apiKey     string
	httpClient *http.Client
}

func (p *BingProvider) Name() string {
	return "Bing"
}

// bingResponse represents the Bing API response structure
type bingResponse struct {
	WebPages struct {
		Value []struct {
			Name    string `json:"name"`
			URL     string `json:"url"`
			Snippet string `json:"snippet"`
		} `json:"value"`
	} `json:"webPages"`
}

// Search performs a Bing web search
func (p *BingProvider) Search(ctx context.Context, query string, maxResults int) ([]Result, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", "https://api.bing.microsoft.com/v7.0/search", nil)
	if err != nil {
		return nil, err
	}

	q := req.URL.Query()
	q.Set("q", query)
	q.Set("count", fmt.Sprintf("%d", maxResults))
	req.URL.RawQuery = q.Encode()
	req.Header.Set("Ocp-Apim-Subscription-Key", p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bing API returned status %d", resp.StatusCode)
	}

	var data bingResponse
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}

	results := make([]Result, 0, len(data.WebPages.Value))
	for _, item := range data.WebPages.Value {
		results = append(results, Result{
			Title:   item.Name,
			URL:     item.URL,
			Content: item.Snippet,
		})
	}

	return results, nil
}
