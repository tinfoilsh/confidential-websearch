# Websearch Evaluation

Tests all supported model/endpoint/stream/task combinations against a running websearch server.

## Setup

Start the websearch server locally:

```bash
source .env
LISTEN_ADDR=:8089 go run .
```

## Run

```bash
# Default (localhost:8089)
./evals/run_websearch_eval.sh

# Custom server URL
./evals/run_websearch_eval.sh http://localhost:8091

# Against a deployed server
./evals/run_websearch_eval.sh https://websearch.example.com
```

## What it tests

The script queries `https://api.tinfoil.sh/v1/models` to discover which chat models support `/v1/chat/completions` and `/v1/responses`, then runs every valid combination of:

- **Endpoint**: chat completions, responses (only for models that list it)
- **Streaming**: true, false
- **Task**: search (current news query), fetch (Wikipedia URL fetch)

## Output

Results are saved to `evals/results/<timestamp>/`:

```
results/
  20260415_103500/
    results.json          # Structured analysis of all tests
    raw/                  # Raw response files
      gpt-oss-120b__chat__stream-false__search.json
      gpt-oss-120b__chat__stream-true__search.sse
      gpt-oss-120b__chat__stream-false__search.elapsed
      ...
```

`results.json` contains an array of objects, one per test:

```json
{
  "model": "gpt-oss-120b",
  "endpoint": "chat",
  "stream": false,
  "task": "search",
  "elapsed_seconds": 12,
  "content_length": 899,
  "citation_markers": 3,
  "annotations": 3,
  "quality": "EXCELLENT",
  "error": null
}
```

Quality grades:
- **EXCELLENT**: citation markers and annotations present, counts match
- **GOOD**: both present but counts differ
- **PARTIAL_MARKERS**: markers in text but no annotation objects
- **PARTIAL_ANNOTATIONS**: annotation objects but no markers in text
- **NO_CITATIONS**: content returned but no citations
- **ERROR**: request failed
- **EMPTY**: no content returned

## Re-analyze

To re-run analysis on existing results without re-running the server:

```bash
python3 evals/analyze_websearch_eval.py evals/results/<timestamp>
```
