# Confidential Web Search Proxy

A proxy that augments LLM responses with real-time web search results, running inside a secure enclave.

Requests flow through a pipeline of specialized models that handle search/fetch decisions, safety filtering, and response generation:

1. **Agent Model**: A small, fast model with `search` and `fetch` tools that decides what external data to retrieve
2. **Safeguard Model**: Filters PII from outgoing search queries and detects prompt injection in results
3. **Responder Model**: The user's requested model, which generates the final response using search/fetch context

Uses the [Tinfoil Go SDK](https://github.com/tinfoilsh/tinfoil-go) for secure, attested communication with Tinfoil enclaves.

## Architecture

```
User Request
  │ model: "<responder-model>"
  │ Authorization: Bearer <api-key>
  ▼
┌─────────────────────────────────────────────┐
│              Agent Model                    │
│            (small, fast)                    │
│                                             │
│  Tools: search(query), fetch(url)           │
│  Loops up to 5 iterations to collect        │
│  all needed tool calls, with reasoning      │
│  fed back between iterations.               │
└──────────┬──────────────────┬───────────────┘
           │                  │
   search queries        fetch URLs
           │                  │
           ▼                  │
┌─────────────────────┐       │
│   PII Filter        │       │
│ (Safeguard Model)   │       │
└─────────┬───────────┘       │
          │                   │
          ▼                   ▼
┌─────────────────┐  ┌───────────────────┐
│    Exa API      │  │ Cloudflare Render │
│  (web search)   │  │  (URL fetch)      │
└────────┬────────┘  └────────┬──────────┘
         │                    │
         └────────┬───────────┘
                  ▼
┌─────────────────────────────────────┐
│         Injection Filter            │
│        (Safeguard Model)            │
│  Removes results/pages with         │
│  prompt injection attempts          │
└──────────────────┬──────────────────┘
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
# Set required API keys
export EXA_API_KEY="your-exa-api-key"
export CLOUDFLARE_ACCOUNT_ID="your-cloudflare-account-id"
export CLOUDFLARE_API_TOKEN="your-cloudflare-api-token"

# Run the proxy
go run .

# With verbose logging
go run . -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXA_API_KEY` | - | Exa AI Search API key (required) |
| `CLOUDFLARE_ACCOUNT_ID` | - | Cloudflare account ID for Browser Rendering (required) |
| `CLOUDFLARE_API_TOKEN` | - | Cloudflare API token for Browser Rendering (required) |
| `AGENT_MODEL` | - | Model for search/fetch decisions (small, fast) |
| `SAFEGUARD_MODEL` | - | Model for safety filtering |
| `ENABLE_INJECTION_CHECK` | `false` | Default for prompt injection detection (can be overridden per-request via tools) |
| `LISTEN_ADDR` | `:8089` | Address to listen on |

## API Endpoints

This server provides an OpenAI-compatible API with custom search and safety tools. Standard OpenAI SDKs can make requests, but custom streaming events and response fields are extensions that require additional client handling.

### Chat Completions

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

Response includes standard OpenAI fields plus custom extensions:

- `choices[0].message.content` - The generated response
- `choices[0].message.annotations` - URL citations from search results
- `choices[0].message.fetch_calls` - Fetched URLs with status and content (extension)
- `choices[0].message.search_reasoning` - Agent's reasoning for search/fetch decisions (extension)
- `choices[0].message.blocked_searches` - Queries blocked by safety filters (extension)

**Streaming:** In addition to standard content chunks, streams custom `web_search_call` events for both search and fetch status. Search events have `action.type: "search"` with a query, fetch events have `action.type: "open_page"` with a URL. These use a `chat.completion.chunk` envelope so SDKs don't fail, but the content is custom.

### Responses API

`POST /v1/responses`

```bash
curl http://localhost:8089/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TINFOIL_API_KEY" \
  -d '{
    "model": "<responder-model>",
    "input": "What is the latest news about SpaceX?",
    "tools": [{"type": "web_search"}],
    "stream": true
  }'
```

Response includes structured output with `web_search_call` items (for both searches and URL fetches) and message content with annotations.

**Streaming:** When `stream: true`, emits OpenAI-conformant `response.*` events:

- `response.created`, `response.in_progress` - Lifecycle events
- `response.web_search_call.in_progress/completed` - Search status (`action.type: "search"`)
- `response.web_search_call.in_progress/completed` - Fetch status (`action.type: "open_page"`)
- `response.output_text.delta` - Content chunks
- `response.output_text.annotation.added` - URL citations
- `response.completed` - Final event

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

The request flows through seven stages:

1. **ValidateStage** - Validates request format, extracts user query
2. **AgentStage** - Runs agent model with search and fetch tools (up to 5 iterations)
3. **SearchStage** - Executes pending searches in parallel via Exa API
4. **FetchStage** - Fetches pending URLs in parallel via Cloudflare Browser Rendering
5. **FilterResultsStage** - Filters search results and fetched pages for prompt injection
6. **BuildMessagesStage** - Injects search results and fetched pages into conversation context
7. **ResponderStage** - Generates final response (streaming or non-streaming)

## Docker

```bash
docker build -t websearch-proxy .
docker run -p 8089:8089 \
  -e EXA_API_KEY=$EXA_API_KEY \
  -e CLOUDFLARE_ACCOUNT_ID=$CLOUDFLARE_ACCOUNT_ID \
  -e CLOUDFLARE_API_TOKEN=$CLOUDFLARE_API_TOKEN \
  websearch-proxy
```

Safety checks are controlled per-request via the `tools` array. Include `{ "type": "pii_check" }` to enable PII filtering on search queries, and `{ "type": "injection_check" }` to filter prompt injection from search results.

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
