# Websearch Evaluation

This harness evaluates the current router-owned websearch flow, not the MCP server directly.

For manual local bring-up and one-off probes, use the `local_testing.md` files
in the two repos:

- `../local_testing.md` for standalone `confidential-websearch` MCP debugging
- `../../model-router/local_testing.md` for router-owned end-to-end testing

This file only covers the automated matrix driver.

## Requirements

- a running router endpoint that exposes `/v1/chat/completions` and `/v1/responses`
- `TINFOIL_API_KEY`
- optional: `USE_LOCAL_FIXTURES=1` when that router is pointed at a local `websearch` server running with `LOCAL_TEST_MODE=1`

## Run

```bash
# Default local router
TINFOIL_API_KEY=... ./evals/run_websearch_eval.sh

# Custom router URL
TINFOIL_API_KEY=... ./evals/run_websearch_eval.sh http://localhost:8090

# Deterministic local MCP fixtures
TINFOIL_API_KEY=... USE_LOCAL_FIXTURES=1 ./evals/run_websearch_eval.sh
```

## What it tests

The script queries `https://api.tinfoil.sh/v1/models` to discover which chat models support `/v1/chat/completions` and `/v1/responses`, then runs every valid combination of:

- **Endpoint**: chat completions, responses
- **Streaming**: true, false
- **Task**: search, fetch

## Output

Results are saved to `evals/results/<timestamp>/` and include:

- `raw/*.json` and `raw/*.sse` response captures
- per-case `.elapsed` timing files
- `results.json` with the analyzed matrix

The analyzer treats either source markers, direct URLs, or annotation objects as citations so local fixture runs and deployed runs are graded consistently.

## Re-analyze

```bash
python3 evals/analyze_websearch_eval.py evals/results/<timestamp>
```
