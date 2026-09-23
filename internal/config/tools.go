package config

import (
	_ "embed"
	"fmt"
	"strings"

	"gopkg.in/yaml.v3"
)

//go:embed tools.yml
var defaultToolsYAML []byte

// ToolDescriptions holds the human-readable tool descriptions advertised
// over MCP. The descriptions enumerate the accepted arguments so models
// trained on other search-tool schemas (Tavily's recency_days, Brave's
// topn, etc.) have a strong hint about what this tool actually takes
// before they start hallucinating alternatives.
type ToolDescriptions struct {
	Search string
	Fetch  string
}

type toolsFile struct {
	Tools map[string]toolEntry `yaml:"tools"`
}

type toolEntry struct {
	Description string `yaml:"description"`
}

// LoadToolDescriptions parses the embedded tools.yml bundled with the
// binary. Returns an error rather than falling back to defaults so a
// malformed file during development is caught at startup rather than
// silently shipping stale copy.
func LoadToolDescriptions() (ToolDescriptions, error) {
	return parseToolDescriptions(defaultToolsYAML)
}

func parseToolDescriptions(raw []byte) (ToolDescriptions, error) {
	var parsed toolsFile
	if err := yaml.Unmarshal(raw, &parsed); err != nil {
		return ToolDescriptions{}, fmt.Errorf("parsing tools.yml: %w", err)
	}

	descriptions := ToolDescriptions{
		Search: strings.TrimSpace(parsed.Tools["search"].Description),
		Fetch:  strings.TrimSpace(parsed.Tools["fetch"].Description),
	}

	if descriptions.Search == "" {
		return ToolDescriptions{}, fmt.Errorf("tools.yml: tools.search.description is empty")
	}
	if descriptions.Fetch == "" {
		return ToolDescriptions{}, fmt.Errorf("tools.yml: tools.fetch.description is empty")
	}

	return descriptions, nil
}
