# Confidential Web Search

Tinfoil's web search server is an MCP server that exposes `search` and `fetch` tools backed by Exa, running inside a secure enclave. The model router calls it to power built-in web search; MCP clients can also call it directly.

Client-facing behavior (tool arguments, response shapes, PII masking, prompt-injection screening, safety headers, and limits) is documented at [docs.tinfoil.sh](https://docs.tinfoil.sh):

- [Web search in chat completions](https://docs.tinfoil.sh/guides/web-search)
- [Web search over MCP](https://docs.tinfoil.sh/guides/mcp-websearch)

## Development

```bash
go test -race ./...
LOCAL_TEST_MODE=1 go run ./cmd/websearch-mcp
```

`LOCAL_TEST_MODE=1` serves deterministic fixtures instead of calling Exa. See [DEVELOPMENT.md](DEVELOPMENT.md) for running against Exa, probes, and the eval harness.

## Reporting Vulnerabilities

Please report security vulnerabilities by either:

- Emailing [security@tinfoil.sh](mailto:security@tinfoil.sh)
- Opening an issue on GitHub on this repository

We aim to respond to (legitimate) security reports within 24 hours.
