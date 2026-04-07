package config

import (
	"os"
	"strconv"
)

const (
	SafeguardTemperature = 0.0

	DefaultMaxSearchResults = 10
	MaxSearchContentLength  = 2000
)

// Config holds the server configuration
type Config struct {
	TinfoilAPIKey        string
	ExaAPIKey            string
	CloudflareAccountID  string
	CloudflareAPIToken   string
	ListenAddr           string
	SafeguardModel       string
	ToolSummaryModel     string
	EnablePIICheck       bool
	EnableInjectionCheck bool
}

// Load creates a new config from environment variables
func Load() *Config {
	return &Config{
		TinfoilAPIKey:        os.Getenv("TINFOIL_API_KEY"),
		ExaAPIKey:            os.Getenv("EXA_API_KEY"),
		CloudflareAccountID:  os.Getenv("CLOUDFLARE_ACCOUNT_ID"),
		CloudflareAPIToken:   os.Getenv("CLOUDFLARE_API_TOKEN"),
		ListenAddr:           getEnv("LISTEN_ADDR", ":8089"),
		SafeguardModel:       getEnv("SAFEGUARD_MODEL", "gpt-oss-safeguard-120b"),
		ToolSummaryModel:     getEnv("TOOL_SUMMARY_MODEL", "llama3-3-70b"),
		EnablePIICheck:       getEnvBool("ENABLE_PII_CHECK", true),
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
