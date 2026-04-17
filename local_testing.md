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

## 8. Run the automated eval matrix

Once the MCP server and a local model-router are both up (see section 6), the
repo ships an end-to-end eval harness under `evals/`. Use it instead of
hand-curling one-off prompts whenever you want a fresh signal on router +
websearch quality.

### 8.1 Quickstart

```bash
TINFOIL_API_KEY=... ./evals/run_websearch_eval.sh
```

Point it at a non-default router URL, or run against deterministic local
fixtures:

```bash
TINFOIL_API_KEY=... ./evals/run_websearch_eval.sh http://localhost:8090
TINFOIL_API_KEY=... USE_LOCAL_FIXTURES=1 ./evals/run_websearch_eval.sh
```

Useful environment overrides:

- `CONTEXT_SIZES` — space-separated retrieval-depth buckets to exercise.
  Defaults to `"low medium high"`; set to `"medium"` for a quick smoke run.
- `TASKS` — space-separated task list; defaults to the full set
  (`depth-low depth-medium depth-high search fetch location domains external-off`).
- `SKIP_LLM_JUDGE=1` — skip the LLM-as-judge grading step (useful when running
  offline or to keep costs down).
- `JUDGE_MODEL`, `JUDGE_ENDPOINT`, `JUDGE_CONCURRENCY`, `JUDGE_TIMEOUT` — tune
  the judge. Default model is `glm-5-1`, default endpoint is
  `https://inference.tinfoil.sh/v1/chat/completions`.

Results are written to `evals/results/<timestamp>/`:

- `raw/*.json` and `raw/*.sse` response captures
- per-case `.elapsed` timing files
- `results.json` with the analyzed matrix (heuristic grade, LLM verdict,
  parameter-assertion pass/fail, streaming parity)

Re-run the analyzer on a prior result directory:

```bash
python3 evals/analyze_websearch_eval.py evals/results/<timestamp>
```

### 8.2 What the matrix covers

Every `(model, endpoint, stream, context_size, task)` tuple is captured and
graded on three axes:

1. **Retrieval-depth buckets.** Three prompt classes — `depth-low`
   (single-fact), `depth-medium` (multi-source synthesis), `depth-high`
   (deep research) — exercised at `search_context_size` low / medium / high
   so regressions in how the router maps context size to tool-call breadth
   are visible.
2. **Parameter-assertion tasks.** Three dedicated prompts verify that
   OpenAI's web search options propagate end-to-end:
   - `location` — sets `user_location.approximate.country=GB`; passes if a
     search ran and UK context appears in the answer or annotations
     (soft-passes when the upstream search index ignores the hint).
   - `domains` — sets `filters.allowed_domains=["python.org"]`; every
     annotation URL must be under `python.org`.
   - `external-off` — sets `external_web_access=false`; the answer must
     admit the access is disabled or produce no annotations.
3. **Qualitative LLM-as-judge grading.** Each response is sent to Tinfoil
   (`glm-5-1` by default) with a compact rubric scoring correctness,
   cited-source support, and helpfulness (0–3 each) plus an overall
   `excellent|good|partial|poor` verdict. The aggregate is printed by
   retrieval-depth bucket so you can spot quality changes as depth grows.

The analyzer also cross-checks streaming vs non-streaming parity (content,
annotations, and `web_search_call` events should match) and flags mismatches
per `(model, endpoint, ctx, task)`.
