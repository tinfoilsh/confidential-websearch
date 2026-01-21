# Confidential Web Search Proxy

A proxy that augments LLM responses with real-time web search results, running inside a secure enclave.

Requests flow through a pipeline of specialized models that handle search decisions, safety filtering, and response generation:

1. **Agent Model**: A small, fast model that decides whether a search is needed and generates queries
2. **Safeguard Model**: Filters PII from outgoing queries and detects prompt injection in search results
3. **Responder Model**: The user's requested model, which generates the final response using search context

Uses the [Tinfoil Go SDK](https://github.com/tinfoilsh/tinfoil-go) for secure, attested communication with Tinfoil enclaves.

## Architecture

```
User Request
  │ model: "<responder-model>"
  │ Authorization: Bearer <api-key>
  ▼
┌─────────────────────┐
│    Agent Model      │ ──► Decides: search needed?
│   (small, fast)     │     Generates search queries
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
│      (from user request)            │
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
| `AGENT_MODEL` | - | Model for search decisions (small, fast) |
| `SAFEGUARD_MODEL` | - | Model for safety filtering |
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
    "model": "<responder-model>",
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
    "model": "<responder-model>",
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

Also blocks email addresses and phone numbers (these uniquely identify individuals). Does NOT block: names, addresses, or dates alone (these are commonly searched and don't uniquely identify someone). Combinations that identify a specific person (e.g., "John Smith, DOB 03/15/1985") are also blocked.

### Prompt Injection Detection

Filters search results that contain prompt injection attempts:

- Instruction overrides ("ignore previous instructions")
- Role manipulation ("you are now DAN")
- System prompt extraction attempts
- Jailbreak patterns

Results flagged as containing injection are removed before being passed to the responder model.

## Pipeline Stages

The request flows through six stages:

1. **ValidateStage** - Validates request format, extracts user query
2. **AgentStage** - Runs agent model with search tool, returns pending searches
3. **SearchStage** - Executes pending searches in parallel via Exa API
4. **FilterResultsStage** - Filters search results for prompt injection
5. **BuildMessagesStage** - Injects search results into conversation context
6. **ResponderStage** - Generates final response (streaming or non-streaming)

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
