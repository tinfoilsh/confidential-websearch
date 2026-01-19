# Confidential Web Search Proxy

A proxy that augments LLM responses with real-time web search results. It uses a two-model architecture:

1. **Agent Model** (small, fast, configured): Decides whether to search and what to search for
2. **Responder Model** (from request): Generates the final response using search results

Uses the [Tinfoil Go SDK](https://github.com/tinfoilsh/tinfoil-go) for secure, attested communication with Tinfoil enclaves.

## Architecture

```
User Request
  │ model: "kimi-k2-thinking"
  │ Authorization: Bearer <api-key>
  ▼
┌─────────────────┐
│   Agent Model   │ ──► Decides: search needed?
│  (gpt-oss-120b) │     (configured, small/fast)
└─────────────────┘
         │
         ▼ web_search tool call
┌────────────────┐
│      Exa       │ ──► Returns search results
└────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         Responder Model             │
│     (from request: kimi-k2-thinking)│ ──► Generates response
└─────────────────────────────────────┘
         │
         ▼
   Final Response (streamed)
```

## Quick Start

```bash
# Set search API key
export EXA_API_KEY="your-exa-api-key"

# Run the proxy
go run .
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXA_API_KEY` | - | Exa AI Search API key (required) |
| `AGENT_MODEL` | `gpt-oss-120b` | Agent model for tool use decisions |
| `LISTEN_ADDR` | `:8089` | Address to listen on |

## API

The proxy exposes an OpenAI-compatible `/v1/chat/completions` endpoint:
- The `model` field specifies which model generates the final response
- The `Authorization` header is forwarded to backend models

```bash
curl http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TINFOIL_API_KEY" \
  -d '{
    "model": "kimi-k2-thinking",
    "messages": [{"role": "user", "content": "What is the latest news about SpaceX?"}],
    "stream": true
  }'
```

## How It Works

1. **Request arrives** with `model` and API key
2. **Agent phase**: Small agent model (`gpt-oss-120b`) decides if search is needed
   - If yes, generates search queries using the `web_search` tool
   - Searches are executed in parallel via Exa
3. **Response phase**: Search results are injected into the context as tool results
4. **Final response**: The model from the request generates the answer
5. Response is streamed back to the client

## Docker

```bash
docker build -t websearch-proxy .
docker run -p 8089:8089 \
  -e EXA_API_KEY=$EXA_API_KEY \
  websearch-proxy
```

## Security

This proxy uses the Tinfoil Go SDK which provides:
- Automatic attestation validation to ensure enclave integrity
- TLS certificate pinning with attested certificates
- Direct-to-enclave encrypted communication
- API key forwarding from client requests (no stored credentials)
