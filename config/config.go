package config

import (
	"os"
	"strconv"
)

// LLM parameter constants
const (
	AgentTemperature     = 0.1
	AgentMaxTokens     = 1024
	AgentMaxIterations = 5 // Max tool-calling loop iterations per request
	SafeguardTemperature = 0.0
)

// Search constants
const (
	DefaultMaxSearchResults = 10
	MaxSearchContentLength  = 2000
)

// Config holds the proxy configuration
type Config struct {
	// Agent model (small model for tool use decisions)
	AgentModel string

	// API keys
	TinfoilAPIKey string
	ExaAPIKey     string

	// Cloudflare Browser Rendering
	CloudflareAccountID string
	CloudflareAPIToken  string

	// Server settings
	ListenAddr string

	// Safeguard settings
	SafeguardModel       string
	EnableInjectionCheck bool
}

// Load creates a new config from environment variables
func Load() *Config {
	return &Config{
		AgentModel:           getEnv("AGENT_MODEL", "gpt-oss-120b"),
		TinfoilAPIKey:        os.Getenv("TINFOIL_API_KEY"),
		ExaAPIKey:            os.Getenv("EXA_API_KEY"),
		CloudflareAccountID: os.Getenv("CLOUDFLARE_ACCOUNT_ID"),
		CloudflareAPIToken:  os.Getenv("CLOUDFLARE_API_TOKEN"),
		ListenAddr:           getEnv("LISTEN_ADDR", ":8089"),
		SafeguardModel:       getEnv("SAFEGUARD_MODEL", "gpt-oss-safeguard-120b"),
		EnableInjectionCheck: getEnvBool("ENABLE_INJECTION_CHECK", false),
	}
}

func getEnv(key, fallback string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return fallback
}

func getEnvBool(key string, fallback bool) bool {
	val := os.Getenv(key)
	if val == "" {
		return fallback
	}
	parsed, err := strconv.ParseBool(val)
	if err != nil {
		return fallback
	}
	return parsed
}
