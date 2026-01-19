# Confidential Web Search Proxy

A proxy that augments LLM responses with real-time web search results, running inside a secure enclave. It uses a multi-model architecture with safety filtering:

1. **Agent Model** (small, fast): Decides whether to search and generates queries
2. **Safeguard Model**: Filters PII from queries and detects prompt injection in results
3. **Responder Model** (from request): Generates the final response using search results

Uses the [Tinfoil Go SDK](https://github.com/tinfoilsh/tinfoil-go) for secure, attested communication with Tinfoil enclaves.

## Architecture

```
User Request
  │ model: "kimi-k2"
  │ Authorization: Bearer <api-key>
  ▼
┌─────────────────────┐
│    Agent Model      │ ──► Decides: search needed?
│ (gpt-oss-120b-free) │     Generates search queries
└─────────────────────┘
          │
          ▼ search queries
┌─────────────────────┐
│   PII Filter        │ ──► Blocks queries with sensitive data
│ (Safeguard Model)   │     (SSN, bank accounts, medical IDs)
└─────────────────────┘
          │
          ▼ filtered queries
┌─────────────────────┐
│       Exa API       │ ──► Returns search results
└─────────────────────┘
          │
          ▼ search results
┌─────────────────────┐
│  Injection Filter   │ ──► Removes results with prompt injection
│ (Safeguard Model)   │     (instruction overrides, jailbreaks)
└─────────────────────┘
          │
          ▼ clean results
    ──► SSE: web_search_call events (query + status)
          │
          ▼
┌─────────────────────────────────────┐
│         Responder Model             │
│       (from request: kimi-k2)       │
└─────────────────────────────────────┘
          │
          ▼ (streaming)
    1. metadata chunk (annotations + reasoning)
    2. response content chunks...
```

## Quick Start

```bash
# Set required API key
export EXA_API_KEY="your-exa-api-key"

# Run the proxy
go run .

# With verbose logging
go run . -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXA_API_KEY` | - | Exa AI Search API key (required) |
| `AGENT_MODEL` | `gpt-oss-120b-free` | Agent model for search decisions |
| `SAFEGUARD_MODEL` | `gpt-oss-safeguard-120b` | Model for safety checks |
| `ENABLE_PII_CHECK` | `true` | Enable PII detection in search queries |
| `ENABLE_INJECTION_CHECK` | `true` | Enable prompt injection detection in results |
| `LISTEN_ADDR` | `:8089` | Address to listen on |

## API Endpoints

### Chat Completions (OpenAI-compatible)

`POST /v1/chat/completions`

```bash
curl http://localhost:8089/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TINFOIL_API_KEY" \
  -d '{
    "model": "kimi-k2",
    "messages": [{"role": "user", "content": "What is the latest news about SpaceX?"}],
    "stream": true
  }'
```

Response includes:

- `choices[0].message.content` - The generated response
- `choices[0].message.annotations` - URL citations from search results
- `choices[0].message.search_reasoning` - Agent's reasoning for search decisions

### Responses API

`POST /v1/responses`

```bash
curl http://localhost:8089/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TINFOIL_API_KEY" \
  -d '{
    "model": "kimi-k2",
    "input": "What is the latest news about SpaceX?"
  }'
```

Response includes structured output with `web_search_call` events and message content with annotations.

### Health Check

`GET /health` - Returns `{"status":"ok"}`

## Safety Features

### PII Detection

Blocks search queries that would leak sensitive personally identifiable information:

- Social Security Numbers, Tax IDs, Passport numbers
- Bank account numbers, Credit card numbers
- Medical record numbers, Health insurance IDs

Does NOT block: names, email addresses, phone numbers, addresses (these are commonly searched).

### Prompt Injection Detection

Filters search results that contain prompt injection attempts:

- Instruction overrides ("ignore previous instructions")
- Role manipulation ("you are now DAN")
- System prompt extraction attempts
- Jailbreak patterns

Results flagged as containing injection are removed before being passed to the responder model.

## Pipeline Stages

The request flows through four stages:

1. **ValidateStage** - Validates request format, extracts user query
2. **AgentStage** - Runs agent model with search tool, executes searches
3. **BuildMessagesStage** - Injects search results into conversation context
4. **ResponderStage** - Generates final response (streaming or non-streaming)

## Docker

```bash
docker build -t websearch-proxy .
docker run -p 8089:8089 \
  -e EXA_API_KEY=$EXA_API_KEY \
  websearch-proxy
```

To disable safety checks:

```bash
docker run -p 8089:8089 \
  -e EXA_API_KEY=$EXA_API_KEY \
  -e ENABLE_PII_CHECK=false \
  -e ENABLE_INJECTION_CHECK=false \
  websearch-proxy
```

## Security

This proxy uses the Tinfoil Go SDK which provides:

- Automatic attestation validation to ensure enclave integrity
- TLS certificate pinning with attested certificates
- Direct-to-enclave encrypted communication
- API key forwarding from client requests (no stored credentials)

All processing occurs within secure enclaves - search queries, results, and responses never leave the trusted execution environment unencrypted.

## Reporting Vulnerabilities

Please report security vulnerabilities by either:

- Emailing [security@tinfoil.sh](mailto:security@tinfoil.sh)

- Opening an issue on GitHub on this repository

We aim to respond to (legitimate) security reports within 24 hours.
