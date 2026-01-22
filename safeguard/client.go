package safeguard

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/tinfoilsh/tinfoil-go"

	"github.com/tinfoilsh/confidential-websearch/config"
)

// DefaultModel is the default safeguard model
const DefaultModel = "gpt-oss-safeguard-120b"

// CheckResult contains the result of a safety check
type CheckResult struct {
	Violation bool   `json:"violation"`
	Rationale string `json:"rationale"`
}

// checkResultSchema is the JSON schema for structured output enforcement
var checkResultSchema = map[string]interface{}{
	"type": "object",
	"properties": map[string]interface{}{
		"violation": map[string]interface{}{
			"type":        "boolean",
			"description": "Whether the content violates the policy",
		},
		"rationale": map[string]interface{}{
			"type":        "string",
			"description": "Brief explanation of the classification decision",
		},
	},
	"required":             []string{"violation", "rationale"},
	"additionalProperties": false,
}

// Client wraps the Tinfoil client for safeguard model calls
type Client struct {
	tinfoil *tinfoil.Client
	model   string
}

// NewClient creates a new safeguard client
func NewClient(tinfoil *tinfoil.Client, model string) *Client {
	if model == "" {
		model = DefaultModel
	}
	return &Client{tinfoil: tinfoil, model: model}
}

// Check evaluates content against a policy and returns the result
func (c *Client) Check(ctx context.Context, policy, content string) (*CheckResult, error) {
	resp, err := c.tinfoil.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: shared.ChatModel(c.model),
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(policy),
			openai.UserMessage(content),
		},
		Temperature: openai.Float(config.SafeguardTemperature),
		MaxTokens:   openai.Int(config.SafeguardMaxTokens),
		ResponseFormat: openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
				JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
					Name:   "check_result",
					Schema: checkResultSchema,
					Strict: openai.Bool(true),
				},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("safeguard call failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("safeguard returned no response")
	}

	var result CheckResult
	if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &result); err != nil {
		return nil, fmt.Errorf("failed to parse safeguard response: %w", err)
	}

	return &result, nil
}
