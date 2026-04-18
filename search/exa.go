package search

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/tinfoilsh/confidential-websearch/config"
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
	Query              string            `json:"query"`
	Type               string            `json:"type,omitempty"`
	NumResults         int               `json:"numResults,omitempty"`
	UserLocation       string            `json:"userLocation,omitempty"`
	IncludeDomains     []string          `json:"includeDomains,omitempty"`
	ExcludeDomains     []string          `json:"excludeDomains,omitempty"`
	Category           string            `json:"category,omitempty"`
	StartPublishedDate string            `json:"startPublishedDate,omitempty"`
	EndPublishedDate   string            `json:"endPublishedDate,omitempty"`
	Contents           *exaContentsParam `json:"contents,omitempty"`
}

type exaContentsParam struct {
	Text        *exaTextParam       `json:"text,omitempty"`
	Highlights  *exaHighlightsParam `json:"highlights,omitempty"`
	MaxAgeHours *int                `json:"maxAgeHours,omitempty"`
}

type exaTextParam struct {
	MaxCharacters int `json:"maxCharacters,omitempty"`
}

type exaHighlightsParam struct {
	Query         string `json:"query,omitempty"`
	MaxCharacters int    `json:"maxCharacters,omitempty"`
}

// exaResponse represents the Exa API response structure
type exaResponse struct {
	Results []struct {
		Title         string `json:"title"`
		URL           string `json:"url"`
		Text          string `json:"text"`
		Highlights    any    `json:"highlights"`
		Favicon       string `json:"favicon"`
		PublishedDate string `json:"publishedDate"`
	} `json:"results"`
	Error string `json:"error,omitempty"`
}

// Search performs an Exa web search
func (p *ExaProvider) Search(ctx context.Context, query string, opts Options) ([]Result, error) {
	maxResults := opts.MaxResults
	if maxResults <= 0 {
		maxResults = config.DefaultMaxSearchResults
	}

	maxCharacters := opts.MaxContentCharacters
	if maxCharacters <= 0 {
		maxCharacters = config.MaxSearchContentLength
	}

	contents := &exaContentsParam{}
	if opts.ContentMode == ContentModeHighlights {
		contents.Highlights = &exaHighlightsParam{
			Query:         query,
			MaxCharacters: maxCharacters,
		}
	} else {
		contents.Text = &exaTextParam{
			MaxCharacters: maxCharacters,
		}
	}
	if opts.MaxAgeHours != nil {
		contents.MaxAgeHours = opts.MaxAgeHours
	}

	reqBody := exaRequest{
		Query:      query,
		Type:       "fast", // Use fast type to guarantee ZDR (Exa maintains the index)
		NumResults: maxResults,
		Contents:   contents,
	}
	if opts.UserLocationCountry != "" {
		reqBody.UserLocation = opts.UserLocationCountry
	}
	if len(opts.AllowedDomains) > 0 {
		reqBody.IncludeDomains = opts.AllowedDomains
	}
	if len(opts.ExcludedDomains) > 0 {
		reqBody.ExcludeDomains = opts.ExcludedDomains
	}
	if opts.Category != "" {
		reqBody.Category = string(opts.Category)
	}
	if opts.StartPublishedDate != "" {
		reqBody.StartPublishedDate = opts.StartPublishedDate
	}
	if opts.EndPublishedDate != "" {
		reqBody.EndPublishedDate = opts.EndPublishedDate
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
			Content:       exaResultContent(item.Text, item.Highlights),
			Favicon:       item.Favicon,
			PublishedDate: item.PublishedDate,
		})
	}

	return results, nil
}

func exaResultContent(text string, highlights any) string {
	if content := parseExaHighlights(highlights); content != "" {
		return content
	}
	return text
}

func parseExaHighlights(raw any) string {
	if raw == nil {
		return ""
	}

	switch highlights := raw.(type) {
	case []any:
		parts := make([]string, 0, len(highlights))
		for _, highlight := range highlights {
			switch value := highlight.(type) {
			case string:
				value = strings.TrimSpace(value)
				if value != "" {
					parts = append(parts, value)
				}
			case map[string]any:
				if text, ok := value["text"].(string); ok {
					text = strings.TrimSpace(text)
					if text != "" {
						parts = append(parts, text)
					}
				}
			}
		}
		return strings.Join(parts, "\n\n")
	case string:
		return strings.TrimSpace(highlights)
	default:
		return ""
	}
}
