package config

import (
	"os"
)

// Config holds the proxy configuration
type Config struct {
	// Agent model (small model for tool use decisions)
	AgentModel string

	// Search API key
	ExaAPIKey string

	// Server settings
	ListenAddr string
}

// Load creates a new config from environment variables
func Load() *Config {
	return &Config{
		AgentModel: getEnv("AGENT_MODEL", "gpt-oss-120b-free"),
		ExaAPIKey:  os.Getenv("EXA_API_KEY"),
		ListenAddr: getEnv("LISTEN_ADDR", ":8089"),
	}
}

func getEnv(key, fallback string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return fallback
}
