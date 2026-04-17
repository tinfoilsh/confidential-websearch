# Local Testing

This is the websearch-centric runbook. Use it when you want to debug the
`confidential-websearch` MCP server itself.

For router-owned end-to-end model calls and the eval harness, pair this with
`../model-router/local_testing.md`. The two guides are intentionally split so
this file focuses on starting and probing the MCP server directly, while the
router guide focuses on model-facing behavior.

## 1. Choose a mode

### Fixture mode

Use this when you want deterministic local behavior without Exa or Cloudflare.

```bash
LOCAL_TEST_MODE=1 \
LISTEN_ADDR=127.0.0.1:8091 \
go run .
```

### Real-provider mode

Use this when you want to probe the real Exa search path and Cloudflare fetch
path. If this repo has a local `.env`, load it first.

```bash
set -a && . ./.env && set +a
LISTEN_ADDR=127.0.0.1:8091 \
go run .
```

## 2. Smoke test the HTTP surface

```bash
curl -sS http://127.0.0.1:8091/health
```

Expected response:

```text
OK
```

## 3. Smoke test MCP initialize

```bash
curl -i -sS -X POST http://127.0.0.1:8091/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  --data '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-06-18",
      "capabilities": {},
      "clientInfo": {
        "name": "local-probe",
        "version": "1.0"
      }
    }
  }'
```

Expect a `200 OK` response with `serverInfo.name` set to
`confidential-websearch`. The streamable SDK may also return an
`Mcp-Session-Id` header even though this server is configured in stateless mode.

## 4. Smoke test MCP tools directly

Capture the session header from `initialize`, then list tools:

```bash
SESSION_ID=$(
  curl -sS -D /tmp/websearch-mcp.headers -o /tmp/websearch-mcp.init.json \
    -X POST http://127.0.0.1:8091/mcp \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    --data '{
      "jsonrpc": "2.0",
      "id": 1,
      "method": "initialize",
      "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {
          "name": "local-probe",
          "version": "1.0"
        }
      }
    }' >/dev/null &&
  awk 'BEGIN{IGNORECASE=1} /^Mcp-Session-Id:/ {print $2}' /tmp/websearch-mcp.headers | tr -d '\r'
)

curl -sS -X POST http://127.0.0.1:8091/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  --data '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
  }'
```

Expected tools include `search` and `fetch`.

## 5. Fixture-mode tool probes

When `LOCAL_TEST_MODE=1` is enabled, the fixture URLs are:

- `https://local.test/cats/almanac`
- `https://local.test/cats/gazette`

Direct `search` probe:

```bash
curl -sS -X POST http://127.0.0.1:8091/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  --data '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "search",
      "arguments": {
        "query": "Nimbus breakfast",
        "max_results": 3
      }
    }
  }'
```

Direct `fetch` probe:

```bash
curl -sS -X POST http://127.0.0.1:8091/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  --data '{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
      "name": "fetch",
      "arguments": {
        "urls": ["https://local.test/cats/almanac"]
      }
    }
  }'
```

## 6. Hand off to the router flow

Once direct MCP probes are healthy, use `../model-router/local_testing.md` to:

- start a local router with `DEBUG=1`
- point `LOCAL_WEBSEARCH_MCP_ENDPOINT` at this server
- run model-facing smoke tests and the eval harness

## 7. Cleanup

If the server is still running in the background:

```bash
lsof -ti tcp:8091 | xargs kill
```
