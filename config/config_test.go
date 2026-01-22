package config

import (
	"os"
	"testing"
)

func TestLoad_Defaults(t *testing.T) {
	// Clear relevant env vars
	envVars := []string{"AGENT_MODEL", "EXA_API_KEY", "LISTEN_ADDR", "SAFEGUARD_MODEL", "ENABLE_INJECTION_CHECK"}
	originalValues := make(map[string]string)
	for _, key := range envVars {
		originalValues[key] = os.Getenv(key)
		os.Unsetenv(key)
	}
	defer func() {
		for key, val := range originalValues {
			if val != "" {
				os.Setenv(key, val)
			}
		}
	}()

	cfg := Load()

	if cfg.AgentModel != "gpt-oss-120b-free" {
		t.Errorf("AgentModel: expected 'gpt-oss-120b-free', got '%s'", cfg.AgentModel)
	}
	if cfg.ListenAddr != ":8089" {
		t.Errorf("ListenAddr: expected ':8089', got '%s'", cfg.ListenAddr)
	}
	if cfg.SafeguardModel != "gpt-oss-safeguard-120b" {
		t.Errorf("SafeguardModel: expected 'gpt-oss-safeguard-120b', got '%s'", cfg.SafeguardModel)
	}
	if cfg.EnableInjectionCheck {
		t.Error("EnableInjectionCheck: expected false by default")
	}
	if cfg.ExaAPIKey != "" {
		t.Errorf("ExaAPIKey: expected empty, got '%s'", cfg.ExaAPIKey)
	}
}

func TestLoad_CustomValues(t *testing.T) {
	// Set custom env vars
	os.Setenv("AGENT_MODEL", "custom-model")
	os.Setenv("EXA_API_KEY", "test-api-key")
	os.Setenv("LISTEN_ADDR", ":9000")
	os.Setenv("SAFEGUARD_MODEL", "custom-safeguard")
	os.Setenv("ENABLE_INJECTION_CHECK", "false")
	defer func() {
		os.Unsetenv("AGENT_MODEL")
		os.Unsetenv("EXA_API_KEY")
		os.Unsetenv("LISTEN_ADDR")
		os.Unsetenv("SAFEGUARD_MODEL")
		os.Unsetenv("ENABLE_INJECTION_CHECK")
	}()

	cfg := Load()

	if cfg.AgentModel != "custom-model" {
		t.Errorf("AgentModel: expected 'custom-model', got '%s'", cfg.AgentModel)
	}
	if cfg.ExaAPIKey != "test-api-key" {
		t.Errorf("ExaAPIKey: expected 'test-api-key', got '%s'", cfg.ExaAPIKey)
	}
	if cfg.ListenAddr != ":9000" {
		t.Errorf("ListenAddr: expected ':9000', got '%s'", cfg.ListenAddr)
	}
	if cfg.SafeguardModel != "custom-safeguard" {
		t.Errorf("SafeguardModel: expected 'custom-safeguard', got '%s'", cfg.SafeguardModel)
	}
	if cfg.EnableInjectionCheck {
		t.Error("EnableInjectionCheck: expected false")
	}
}

func TestGetEnv_WithValue(t *testing.T) {
	os.Setenv("TEST_GET_ENV", "test-value")
	defer os.Unsetenv("TEST_GET_ENV")

	result := getEnv("TEST_GET_ENV", "fallback")
	if result != "test-value" {
		t.Errorf("expected 'test-value', got '%s'", result)
	}
}

func TestGetEnv_WithFallback(t *testing.T) {
	os.Unsetenv("TEST_GET_ENV_UNSET")

	result := getEnv("TEST_GET_ENV_UNSET", "fallback-value")
	if result != "fallback-value" {
		t.Errorf("expected 'fallback-value', got '%s'", result)
	}
}

func TestGetEnvBool_True(t *testing.T) {
	testCases := []string{"true", "True", "TRUE", "1"}
	for _, val := range testCases {
		os.Setenv("TEST_BOOL", val)
		result := getEnvBool("TEST_BOOL", false)
		if !result {
			t.Errorf("expected true for value '%s'", val)
		}
	}
	os.Unsetenv("TEST_BOOL")
}

func TestGetEnvBool_False(t *testing.T) {
	testCases := []string{"false", "False", "FALSE", "0"}
	for _, val := range testCases {
		os.Setenv("TEST_BOOL", val)
		result := getEnvBool("TEST_BOOL", true)
		if result {
			t.Errorf("expected false for value '%s'", val)
		}
	}
	os.Unsetenv("TEST_BOOL")
}

func TestGetEnvBool_Invalid(t *testing.T) {
	os.Setenv("TEST_BOOL", "invalid")
	defer os.Unsetenv("TEST_BOOL")

	// Should return fallback when value is invalid
	resultTrue := getEnvBool("TEST_BOOL", true)
	if !resultTrue {
		t.Error("expected true fallback for invalid value")
	}

	resultFalse := getEnvBool("TEST_BOOL", false)
	if resultFalse {
		t.Error("expected false fallback for invalid value")
	}
}

func TestGetEnvBool_Unset(t *testing.T) {
	os.Unsetenv("TEST_BOOL_UNSET")

	resultTrue := getEnvBool("TEST_BOOL_UNSET", true)
	if !resultTrue {
		t.Error("expected true fallback when unset")
	}

	resultFalse := getEnvBool("TEST_BOOL_UNSET", false)
	if resultFalse {
		t.Error("expected false fallback when unset")
	}
}

func TestConstants(t *testing.T) {
	if AgentTemperature != 0.3 {
		t.Errorf("AgentTemperature: expected 0.3, got %f", AgentTemperature)
	}
	if AgentMaxTokens != 1024 {
		t.Errorf("AgentMaxTokens: expected 1024, got %d", AgentMaxTokens)
	}
	if SafeguardTemperature != 0.0 {
		t.Errorf("SafeguardTemperature: expected 0.0, got %f", SafeguardTemperature)
	}
	if DefaultMaxSearchResults != 5 {
		t.Errorf("DefaultMaxSearchResults: expected 5, got %d", DefaultMaxSearchResults)
	}
	if MaxSearchContentLength != 500 {
		t.Errorf("MaxSearchContentLength: expected 500, got %d", MaxSearchContentLength)
	}
}
