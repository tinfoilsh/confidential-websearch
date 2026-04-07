# Confidential Web Search Proxy

A secure web search and URL fetch service for LLM applications, running inside a Tinfoil enclave.

The service exposes three surfaces:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /mcp`

It uses a single tool-capable model loop to decide whether to answer directly or call server-side `search` and `fetch` tools. Search and fetch outputs can be filtered for safety and compacted before the model produces a final answer with citations.

Uses the [Tinfoil Go SDK](https://github.com/tinfoilsh/tinfoil-go) for secure, attested communication with Tinfoil enclaves.

## Architecture

```text
User Request
  │ model: "<tool-capable-model>"
  ▼
┌──────────────────────────────────────────────┐
│         Compatibility / MCP Surface          │
│  - /v1/chat/completions                      │
│  - /v1/responses                             │
│  - /mcp                                      │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│          Single Model Orchestrator           │
│                                              │
│  One model decides whether to:               │
│  - answer directly                           │
│  - call search(query)                        │
│  - call fetch(url)                           │
│                                              │
│  The loop carries forward provider-side      │
│  state with previous_response_id.            │
└───────────────┬───────────────────────┬──────┘
                │                       │
                ▼                       ▼
        ┌───────────────┐      ┌───────────────────┐
        │ Exa Search    │      │ Cloudflare Render │
        └──────┬────────┘      └─────────┬─────────┘
               │                         │
               └──────────┬──────────────┘
                          ▼
        ┌──────────────────────────────────────┐
        │ Optional safeguard checks            │
        │ - PII filtering for search queries   │
        │ - Prompt injection filtering for     │
        │   search results and fetched pages   │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │ Optional tool-output compaction      │
        │ - Large search/fetch outputs are     │
        │   summarized by TOOL_SUMMARY_MODEL   │
        │ - Source markers like 【1】 are       │
        │   preserved for citation fidelity    │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │ Final model answer + annotations     │
        └──────────────────────────────────────┘
```

## Quick Start

```bash
export TINFOIL_API_KEY="your-tinfoil-api-key"
export EXA_API_KEY="your-exa-api-key"
export CLOUDFLARE_ACCOUNT_ID="your-cloudflare-account-id"
export CLOUDFLARE_API_TOKEN="your-cloudflare-api-token"

# optional
export TOOL_SUMMARY_MODEL="llama3-3-70b"

# run the proxy
go run .

# with verbose logging
go run . -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TINFOIL_API_KEY` | - | Tinfoil API key for enclave model access |
| `EXA_API_KEY` | - | Exa search API key |
| `CLOUDFLARE_ACCOUNT_ID` | - | Cloudflare account ID for Browser Rendering |
| `CLOUDFLARE_API_TOKEN` | - | Cloudflare API token for Browser Rendering |
| `SAFEGUARD_MODEL` | `gpt-oss-safeguard-120b` | Model used for safety filtering |
| `TOOL_SUMMARY_MODEL` | `llama3-3-70b` | Model used to compact oversized search/fetch outputs |
| `ENABLE_PII_CHECK` | `true` | Default for PII filtering on outgoing search queries |
| `ENABLE_INJECTION_CHECK` | `false` | Default for prompt injection filtering on search/fetch output |
| `LISTEN_ADDR` | `:8089` | Address to listen on |

## Request Flow

Each request follows a single-model server-side tool loop:

1. Validate the request and normalize web search options
2. Ask the model to answer directly or call `search` / `fetch`
3. Execute tool calls server-side
4. Apply optional PII and prompt-injection safeguards
5. Compact oversized tool output when needed while preserving source markers
6. Continue the loop with `previous_response_id`
7. Return the final answer with citations, reasoning, blocked searches, and fetch status

## API Endpoints

This server provides OpenAI-compatible compatibility endpoints plus MCP.

### Chat Completions

`POST /v1/chat/completions`

```bash
curl http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "What is the latest news about SpaceX?"}],
    "web_search_options": {"search_context_size": "medium"},
    "stream": true
  }'
```

Response includes standard OpenAI fields plus custom extensions:

- `choices[0].message.annotations` - URL citations
- `choices[0].message.fetch_calls` - fetched URLs with status
- `choices[0].message.search_reasoning` - search/fetch orchestration reasoning
- `choices[0].message.blocked_searches` - queries blocked by safety filters

`web_search_options.search_context_size` and `web_search_options.user_location` are optional.

### Responses API

`POST /v1/responses`

```bash
curl http://localhost:8089/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "input": "What is the latest news about SpaceX?",
    "tools": [{"type": "web_search", "search_context_size": "medium"}],
    "stream": true
  }'
```

Response includes structured `web_search_call` items for both searches and URL fetches, plus message content with annotations.

### MCP

`POST /mcp`

The MCP surface exposes `search` and `fetch` tools directly.

### Health Check

`GET /health` returns `{"status":"ok"}`.

## Safety Features

### PII Detection

Blocks outgoing search queries that would leak sensitive personally identifiable information.

### Prompt Injection Detection

Filters search results and fetched pages that contain prompt injection attempts before they are passed back into the answering loop.

### Fetch Target Validation

Rejects unsafe fetch targets before they reach Cloudflare Browser Rendering, including localhost, internal hostnames, private IP ranges, and unsupported URL schemes.

## Docker

```bash
docker build -t websearch-proxy .
docker run -p 8089:8089 \
  -e TINFOIL_API_KEY=$TINFOIL_API_KEY \
  -e EXA_API_KEY=$EXA_API_KEY \
  -e CLOUDFLARE_ACCOUNT_ID=$CLOUDFLARE_ACCOUNT_ID \
  -e CLOUDFLARE_API_TOKEN=$CLOUDFLARE_API_TOKEN \
  websearch-proxy
```

## Security

This proxy uses the Tinfoil Go SDK which provides:

- Automatic attestation validation to ensure enclave integrity
- TLS certificate pinning with attested certificates
- Direct-to-enclave encrypted communication
- Service-held credentials for model, search, and fetch providers inside the enclave

All processing occurs within secure enclaves, so search queries, results, and responses remain encrypted outside the trusted execution environment.

## Reporting Vulnerabilities

Please report security vulnerabilities by either:

- Emailing [security@tinfoil.sh](mailto:security@tinfoil.sh)
- Opening an issue on GitHub on this repository

We aim to respond to legitimate security reports within 24 hours.
