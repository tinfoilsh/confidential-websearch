package config

import (
	"os"
	"strconv"
)

// LLM parameter constants
const (
	AgentTemperature     = 0.3
	AgentMaxTokens       = 1024
	SafeguardTemperature = 0.0
	SafeguardMaxTokens   = 256
)

// Search constants
const (
	DefaultMaxSearchResults = 5
	MaxSearchContentLength  = 500
)

// Config holds the proxy configuration
type Config struct {
	// Agent model (small model for tool use decisions)
	AgentModel string

	// Search API key
	ExaAPIKey string

	// Server settings
	ListenAddr string

	// Safeguard settings
	SafeguardModel       string
	EnablePIICheck       bool
	EnableInjectionCheck bool
}

// Load creates a new config from environment variables
func Load() *Config {
	return &Config{
		AgentModel:           getEnv("AGENT_MODEL", "gpt-oss-120b-free"),
		ExaAPIKey:            os.Getenv("EXA_API_KEY"),
		ListenAddr:           getEnv("LISTEN_ADDR", ":8089"),
		SafeguardModel:       getEnv("SAFEGUARD_MODEL", "gpt-oss-safeguard-120b"),
		EnablePIICheck:       getEnvBool("ENABLE_PII_CHECK", true),
		EnableInjectionCheck: getEnvBool("ENABLE_INJECTION_CHECK", true),
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
