# Confidential Web Search MCP Server

A secure Model Context Protocol (MCP) server that exposes `search` and `fetch` tools backed by Exa and Cloudflare Browser Rendering, running inside a Tinfoil enclave.

The server exposes two surfaces:

- `POST /mcp` - MCP Streamable HTTP endpoint
- `GET /health` - health check

Clients (typically an upstream router that owns its own model and tool loop) call `search` to discover sources and `fetch` to read specific pages. Queries and results can be filtered by an in-enclave safeguard model before leaving or re-entering the trusted boundary.

Uses the [Tinfoil Go SDK](https://github.com/tinfoilsh/tinfoil-go) for secure, attested communication with Tinfoil enclaves.

## Architecture

```text
MCP Client (e.g. router, agent runtime)
  │  MCP tool call: search / fetch
  ▼
┌──────────────────────────────────────────────┐
│             MCP Streamable HTTP              │
│                     /mcp                     │
└──────────────────────┬───────────────────────┘
                       ▼
        ┌──────────────────────────────────────┐
        │ Optional safeguard checks            │
        │ - PII filtering on search queries    │
        │ - Prompt injection filtering on      │
        │   search results and fetched pages   │
        └──────────────────┬───────────────────┘
                           ▼
                ┌───────────────┐      ┌───────────────────┐
                │ Exa Search    │      │ Cloudflare Render │
                └───────────────┘      └───────────────────┘
```

The server also advertises one MCP prompt, `openai_web_search`, containing the system instructions a caller should hand its own model when wiring these tools up.

## Quick Start

```bash
export TINFOIL_API_KEY="your-tinfoil-api-key"
export EXA_API_KEY="your-exa-api-key"
export CLOUDFLARE_ACCOUNT_ID="your-cloudflare-account-id"
export CLOUDFLARE_API_TOKEN="your-cloudflare-api-token"
export USAGE_REPORTER_SECRET="your-usage-reporter-secret"

go run .

# with verbose logging
go run . -v
```

For local development without real upstream providers, set `LOCAL_TEST_MODE=1` to use built-in deterministic fixtures instead of Exa and Cloudflare.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TINFOIL_API_KEY` | - | Tinfoil API key for the in-enclave safeguard model |
| `EXA_API_KEY` | - | Exa search API key |
| `CLOUDFLARE_ACCOUNT_ID` | - | Cloudflare account ID for Browser Rendering |
| `CLOUDFLARE_API_TOKEN` | - | Cloudflare API token for Browser Rendering |
| `SAFEGUARD_MODEL` | `gpt-oss-safeguard-120b` | Model used for safety filtering |
| `ENABLE_PII_CHECK` | `true` | Run PII filtering on outgoing search queries |
| `ENABLE_INJECTION_CHECK` | `false` | Run prompt-injection filtering on search/fetch output |
| `LISTEN_ADDR` | `:8089` | Address to listen on |
| `CONTROL_PLANE_URL` | `https://api.tinfoil.sh` | Base URL for the usage reporter |
| `USAGE_REPORTER_ID` | `websearch-mcp` | Identifier reported with usage events |
| `USAGE_REPORTER_SECRET` | - | Shared secret for signing usage reports |
| `LOCAL_TEST_MODE` | - | Set to `1` to serve static fixtures instead of calling Exa/Cloudflare |

## Tools

### `search`

Search the web and return ranked results with titles, URLs, snippets, and publication dates.

Arguments:

- `query` (string, required) - natural language search query
- `max_results` (int, optional) - number of results to return; defaults to 8

### `fetch`

Fetch one or more web pages via Cloudflare Browser Rendering and return the rendered markdown.

Arguments:

- `urls` (string array, required) - one or more HTTP/HTTPS URLs; capped at 20 per request

The response contains a per-URL `results` list that preserves input order (including failures) plus a `pages` list with just the successfully fetched content.

## Safety Features

### PII Detection

Blocks outgoing search queries that would leak sensitive personally identifiable information.

### Prompt Injection Detection

Filters search results and fetched pages that contain prompt injection attempts before they are returned to the caller.

### Fetch Target Validation

Rejects unsafe fetch targets before they reach Cloudflare Browser Rendering, including localhost, internal hostnames, private IP ranges, and unsupported URL schemes.

## Docker

```bash
docker build -t websearch-mcp .
docker run -p 8089:8089 \
  -e TINFOIL_API_KEY=$TINFOIL_API_KEY \
  -e EXA_API_KEY=$EXA_API_KEY \
  -e CLOUDFLARE_ACCOUNT_ID=$CLOUDFLARE_ACCOUNT_ID \
  -e CLOUDFLARE_API_TOKEN=$CLOUDFLARE_API_TOKEN \
  -e USAGE_REPORTER_SECRET=$USAGE_REPORTER_SECRET \
  websearch-mcp
```

## Security

This service uses the Tinfoil Go SDK which provides:

- Automatic attestation validation to ensure enclave integrity
- TLS certificate pinning with attested certificates
- Direct-to-enclave encrypted communication
- Service-held credentials for the safeguard model, search, and fetch providers inside the enclave

All processing occurs within secure enclaves, so search queries, results, and fetched page content remain encrypted outside the trusted execution environment.

## Reporting Vulnerabilities

Please report security vulnerabilities by either:

- Emailing [security@tinfoil.sh](mailto:security@tinfoil.sh)
- Opening an issue on GitHub on this repository

We aim to respond to legitimate security reports within 24 hours.
