# Confidential Web Search

Tinfoil's web search server is an MCP server that exposes `search` and `fetch` tools backed by Exa, running inside a secure enclave. The model router calls it to power built-in web search; MCP clients can also call it directly.

## How it works

For each tool call the server:

1. Optionally masks PII in the outgoing search query using the in-enclave privacy filter
2. Runs the search or fetch against Exa
3. Optionally screens results and fetched pages for prompt injection with an in-enclave safeguard model
4. Returns the results over MCP Streamable HTTP on `POST /mcp`

Client-facing behavior (tool arguments, response shapes, safety headers, and limits) is documented at [docs.tinfoil.sh](https://docs.tinfoil.sh):

- [Web search in chat completions](https://docs.tinfoil.sh/guides/web-search)
- [Web search over MCP](https://docs.tinfoil.sh/guides/mcp-websearch)

## Running locally

```bash
export TINFOIL_API_KEY="your-tinfoil-api-key"
export EXA_API_KEY="your-exa-api-key"
export USAGE_REPORTER_SECRET="your-usage-reporter-secret"
export USAGE_CONTEXT_SECRET="your-usage-context-secret"

go run ./cmd/websearch-mcp
```

Set `LOCAL_TEST_MODE=1` to serve deterministic fixtures instead of calling Exa; add `-v` for debug logs. See [`local_testing.md`](./local_testing.md) for the full runbook, including the eval harness and running behind the model router.

## Architecture Overview

- **[mcp_server.go](mcp_server.go)**: MCP transport and tool registration
- **[config/](config/)**: Environment variables and tool definitions
- **[handlers.go](handlers.go)**: `search` and `fetch` tool handlers and per-request safety header handling
- **[search/](search/)**, **[fetch/](fetch/)**: Exa search and contents clients
- **[safeguard/](safeguard/)**: PII masking and prompt-injection filtering
- **[domainrank/](domainrank/)**: Cloudflare Radar top-domain list used to skip injection checks
- **[usage/](usage/)**: Usage reporting to the control plane
- **[evals/](evals/)**: Eval harness for tool behavior

## Reporting Vulnerabilities

Please report security vulnerabilities by either:

- Emailing [security@tinfoil.sh](mailto:security@tinfoil.sh)
- Opening an issue on GitHub on this repository

We aim to respond to (legitimate) security reports within 24 hours.
