# Confidential Web Search MCP Server

A secure Model Context Protocol (MCP) server that exposes `search` and `fetch` tools backed by Exa, running inside a Tinfoil enclave.

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
                │ Exa Search    │      │ Exa Contents      │
                └───────────────┘      └───────────────────┘
```

## Quick Start

```bash
export TINFOIL_API_KEY="your-tinfoil-api-key"
export EXA_API_KEY="your-exa-api-key"
export USAGE_REPORTER_SECRET="your-usage-reporter-secret"

go run .

# with verbose logging
go run . -v
```

For local development without real upstream providers, set `LOCAL_TEST_MODE=1` to use built-in deterministic fixtures instead of Exa. In that mode the server also mounts `GET /debug/last-call`, which returns the arguments of the most recent MCP tool call and is consumed by the eval harness.

See [`local_testing.md`](./local_testing.md) for the full runbook, including how to exercise the server through the model router and the eval harness.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TINFOIL_API_KEY` | - | Tinfoil API key for the in-enclave safeguard model |
| `EXA_API_KEY` | - | Exa API key (used for both search and fetch) |
| `CLOUDFLARE_API_TOKEN` | - | Cloudflare Radar API token used to load the top-domain list that bypasses prompt-injection filtering. When unset, every fetched page and search result goes through the safeguard. |
| `SAFEGUARD_MODEL` | `gpt-oss-safeguard-120b` | Model used for safety filtering |
| `ENABLE_PII_CHECK` | `true` | Run PII filtering on outgoing search queries. A per-request `X-Tinfoil-Tool-PII-Check` header can override this (see [Router Integration](#router-integration)). |
| `ENABLE_FETCH_INJECTION_CHECK` | `true` | Run prompt-injection filtering on fetched pages by default (top-popularity domains are skipped unless the caller explicitly opts in via header). |
| `ENABLE_SEARCH_INJECTION_CHECK` | `true` | Run prompt-injection filtering on search results by default (same top-domain skip applies). |
| `LISTEN_ADDR` | `:8089` | Address to listen on |
| `CONTROL_PLANE_URL` | `https://api.tinfoil.sh` | Base URL for the usage reporter |
| `USAGE_REPORTER_ID` | `websearch-mcp` | Identifier reported with usage events |
| `USAGE_REPORTER_SECRET` | - | Shared secret for signing usage reports |
| `LOCAL_TEST_MODE` | - | Set to `1` to serve static fixtures instead of calling Exa |

## Tools

The server exposes two MCP tools. All arguments are passed as JSON in the standard MCP `tools/call` envelope. The full JSON Schema is also advertised via `tools/list`, so any MCP-compliant client can introspect the surface directly.

### `search`

Search the web and return ranked results with titles, URLs, snippets, and publication dates.

#### Arguments

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | yes | - | Natural language search query. Max ~400 characters. |
| `max_results` | int | no | `8` | Number of results to return. Any positive integer; non-positive values fall back to the default. The upstream search provider applies its own ceiling. |
| `content_mode` | string | no | `highlights` | Per-result content granularity. `highlights` returns key excerpts relevant to the query; `text` returns the full page text as markdown. |
| `max_content_chars` | int | no | `700` | Per-result character budget for the snippet or text returned in each hit. Higher values surface more context at a higher token cost. |
| `user_location_country` | string | no | - | ISO 3166-1 alpha-2 country code (e.g. `US`, `GB`, `DE`) used to bias results toward that locale. |
| `allowed_domains` | string[] | no | - | Only return results whose host matches one of these domains. |
| `excluded_domains` | string[] | no | - | Drop results from these domains. Useful for filtering aggregators or SEO farms. |
| `category` | string | no | - | Restrict to one of: `company`, `people`, `research paper`, `news`, `personal site`, `financial report`. Note: `company` and `people` are incompatible with date filters and `excluded_domains` and the server will reject such combinations. |
| `start_published_date` | string | no | - | ISO-8601 date (e.g. `2024-01-01` or `2024-01-01T00:00:00Z`). Only include results published at or after this instant. |
| `end_published_date` | string | no | - | ISO-8601 date. Only include results published at or before this instant. |
| `max_age_hours` | int | no | - | Cache freshness control. `0` forces a livecrawl on every result (freshest, slowest). `-1` disables livecrawl (cache-only, fastest). Omit for the upstream default (livecrawl only when uncached). |

#### Response

```json
{
  "results": [
    {
      "title": "string",
      "url": "string",
      "content": "string",
      "favicon": "string (optional)",
      "published_date": "string (optional, ISO-8601)"
    }
  ]
}
```

### `fetch`

Fetch one or more web pages via the Exa Contents API and return the page text.

#### Arguments

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `urls` | string[] | yes | - | One or more HTTP/HTTPS URLs. Capped at 20 per request. Each page is fetched via Exa Contents (livecrawl preferred, with a short cache horizon). |
| `allowed_domains` | string[] | no | - | If set, reject any URL whose host is not in this list before it is sent to the renderer. |

#### Response

The response contains a per-URL `results` list that preserves input order (including failures) plus a `pages` list with just the successfully fetched content.

```json
{
  "pages": [
    { "url": "string", "content": "string (text)" }
  ],
  "results": [
    {
      "url": "string",
      "status": "completed | failed",
      "content": "string (present when status=completed)",
      "error": "string (present when status=failed)"
    }
  ]
}
```

## Examples

The server speaks MCP Streamable HTTP on `POST /mcp`. The examples below use raw JSON-RPC envelopes against a locally running instance (`LISTEN_ADDR=:8089`); a real MCP client handles the envelope for you.

### Minimal `search`

```bash
curl -sS -X POST http://localhost:8089/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "search",
      "arguments": {
        "query": "confidential computing attestation 2026",
        "max_results": 5
      }
    }
  }'
```

### `search` with filters, recency, and locale bias

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {
      "query": "latest Python 3.13 release notes",
      "max_results": 8,
      "content_mode": "text",
      "max_content_chars": 1200,
      "allowed_domains": ["python.org", "docs.python.org"],
      "category": "news",
      "start_published_date": "2024-10-01",
      "user_location_country": "US",
      "max_age_hours": 0
    }
  }
}
```

### `fetch` with a domain allowlist

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "fetch",
    "arguments": {
      "urls": [
        "https://docs.python.org/3/whatsnew/3.13.html",
        "https://peps.python.org/pep-0703/"
      ],
      "allowed_domains": ["python.org", "docs.python.org", "peps.python.org"]
    }
  }
}
```

## Router Integration

When this server runs behind the Tinfoil model router, the router can override the server's env-configured safety defaults on a per-request basis by forwarding these HTTP headers on the `POST /mcp` call. Anything else leaves the server-side defaults in place.

| Header | Values | Effect |
|--------|--------|--------|
| `X-Tinfoil-Tool-PII-Check` | `true`, `false`, `1`, `0` | Overrides `ENABLE_PII_CHECK` for this request only. |
| `X-Tinfoil-Tool-Injection-Check` | `true`, `false`, `1`, `0` | Overrides the env defaults for fetch and search prompt-injection filtering for this request only. An explicit `true` also disables the popularity-based skip and runs the safeguard on every result. |

Missing, empty, or unparseable values fall back to the env default, so a malformed header can never silently weaken filtering below what the operator configured.

## Safety Features

### PII Detection

Blocks outgoing search queries that would leak sensitive personally identifiable information.

### Prompt Injection Detection

Filters fetched pages and search results that contain prompt injection attempts before they are returned to the caller. Pages hosted on the top ~500k most-visited domains (per the Cloudflare Radar list, refreshed every 24 hours) are treated as trusted and bypass the safeguard. Callers that want maximum coverage can pass `X-Tinfoil-Tool-Injection-Check: true` to force the safeguard on every result regardless of host popularity.

## Docker

```bash
docker build -t websearch-mcp .
docker run -p 8089:8089 \
  -e TINFOIL_API_KEY=$TINFOIL_API_KEY \
  -e EXA_API_KEY=$EXA_API_KEY \
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
