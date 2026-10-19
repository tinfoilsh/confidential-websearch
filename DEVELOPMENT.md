# Development

Run the unit tests:

```bash
go test -race ./...
```

Run the Exa integration tests:

```bash
EXA_API_KEY=... go test -tags=integration -timeout 10m ./...
```

Start the server with deterministic fixtures instead of Exa. Logs are enabled only in this mode; add `-v` for debug logs:

```bash
LOCAL_TEST_MODE=1 LISTEN_ADDR=127.0.0.1:8091 go run ./cmd/websearch-mcp
```

Start the server against Exa with secrets from `.env` (see [.env.example](.env.example)) and the production privacy filter:

```bash
set -a && . ./.env && set +a
LISTEN_ADDR=127.0.0.1:8091 go run ./cmd/websearch-mcp
```

Probe the health check and the `search` and `fetch` tools (fixture URLs live under `https://local.test/cats/`):

```bash
curl -sS http://127.0.0.1:8091/health

curl -sS -X POST http://127.0.0.1:8091/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  --data '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search","arguments":{"query":"Nimbus breakfast","max_results":3}}}'

curl -sS -X POST http://127.0.0.1:8091/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  --data '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fetch","arguments":{"urls":["https://local.test/cats/almanac"]}}}'
```

To test through the model router, start a local router with `DEBUG=1` and `LOCAL_MCP_ENDPOINT_WEBSEARCH=http://127.0.0.1:8091/mcp`, following the router's [local testing guide](https://github.com/tinfoilsh/confidential-model-router/blob/main/local_testing.md).

Run the end-to-end eval against a local router on port 8090 (`--base` selects another router; results go to `evals/results/<timestamp>/`):

```bash
TINFOIL_API_KEY=... python3 evals/run_websearch_eval.py
TINFOIL_API_KEY=... python3 evals/run_websearch_eval.py --local-fixtures --mcp-base http://127.0.0.1:8091
```

Set `CONTEXT_SIZES="medium"` for a quick run or `SKIP_LLM_JUDGE=1` to skip judge grading. Re-analyze a previous run:

```bash
python3 evals/analyze_websearch_eval.py evals/results/<timestamp>
```
