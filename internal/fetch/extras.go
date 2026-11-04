package fetch

import "fmt"

// ExtrasOptions selects structured assets to extract from each page. Counts
// mirror Exa's contents extras; zero leaves the corresponding field disabled.
type ExtrasOptions struct {
	Links          int `json:"links,omitempty" jsonschema:"Number of page URLs to return (0-1000)."`
	ImageLinks     int `json:"imageLinks,omitempty" jsonschema:"Number of image URLs to return (0-1000)."`
	RichLinks      int `json:"richLinks,omitempty" jsonschema:"Number of links with anchor text to return (0-1000)."`
	RichImageLinks int `json:"richImageLinks,omitempty" jsonschema:"Number of images with alt text to return (0-1000)."`
	CodeBlocks     int `json:"codeBlocks,omitempty" jsonschema:"Number of code blocks with their annotated language to return (0-1000)."`
}

func (o *ExtrasOptions) Validate() error {
	if o == nil {
		return nil
	}
	for _, field := range []struct {
		name  string
		count int
	}{
		{"links", o.Links},
		{"imageLinks", o.ImageLinks},
		{"richLinks", o.RichLinks},
		{"richImageLinks", o.RichImageLinks},
		{"codeBlocks", o.CodeBlocks},
	} {
		if field.count < 0 || field.count > 1000 {
			return fmt.Errorf("'extras.%s' must be between 0 and 1000", field.name)
		}
	}
	return nil
}

type Link struct {
	URL    string `json:"url"`
	Anchor string `json:"anchor,omitempty"`
}

type ImageLink struct {
	URL string `json:"url"`
	Alt string `json:"alt,omitempty"`
}

type CodeBlock struct {
	Text   string `json:"text"`
	Source string `json:"source"`
}

type Extras struct {
	Links          []string    `json:"links,omitempty"`
	ImageLinks     []string    `json:"imageLinks,omitempty"`
	RichLinks      []Link      `json:"richLinks,omitempty"`
	RichImageLinks []ImageLink `json:"richImageLinks,omitempty"`
	CodeBlocks     []CodeBlock `json:"codeBlocks,omitempty"`
}

// Metadata contains optional page details and assets returned by Exa.
type Metadata struct {
	Title         string  `json:"title,omitempty"`
	Author        string  `json:"author,omitempty"`
	PublishedDate string  `json:"published_date,omitempty"`
	Image         string  `json:"image,omitempty"`
	Favicon       string  `json:"favicon,omitempty"`
	Extras        *Extras `json:"extras,omitempty"`
}
