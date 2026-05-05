package config

import (
	"os"
	"strconv"
)

// Config holds the server configuration
type Config struct {
	TinfoilAPIKey              string
	ExaAPIKey                  string
	CloudflareAPIToken         string
	ControlPlaneURL            string
	UsageReporterID            string
	UsageReporterSecret        string
	ListenAddr                 string
	SafeguardModel             string
	EnablePIICheck             bool
	EnableFetchInjectionCheck  bool
	EnableSearchInjectionCheck bool
}

// Load creates a new config from environment variables
func Load() *Config {
	return &Config{
		TinfoilAPIKey:              os.Getenv("TINFOIL_API_KEY"),
		ExaAPIKey:                  os.Getenv("EXA_API_KEY"),
		CloudflareAPIToken:         os.Getenv("CLOUDFLARE_API_TOKEN"),
		ControlPlaneURL:            getEnv("CONTROL_PLANE_URL", "https://api.tinfoil.sh"),
		UsageReporterID:            getEnv("USAGE_REPORTER_ID", "websearch-mcp"),
		UsageReporterSecret:        os.Getenv("USAGE_REPORTER_SECRET"),
		ListenAddr:                 getEnv("LISTEN_ADDR", ":8089"),
		SafeguardModel:             getEnv("SAFEGUARD_MODEL", "gpt-oss-safeguard-120b"),
		EnablePIICheck:             getEnvBool("ENABLE_PII_CHECK", true),
		EnableFetchInjectionCheck:  getEnvBool("ENABLE_FETCH_INJECTION_CHECK", true),
		EnableSearchInjectionCheck: getEnvBool("ENABLE_SEARCH_INJECTION_CHECK", true),
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
