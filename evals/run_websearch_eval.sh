#!/bin/bash
#
# Websearch server evaluation matrix.
#
# Runs all supported model/endpoint/stream/task combinations and saves
# raw responses as JSON/SSE files plus a summary results.json.
#
# Usage:
#   ./evals/run_websearch_eval.sh                         # default: http://localhost:8089
#   ./evals/run_websearch_eval.sh http://localhost:8091    # custom base URL
#
# The model catalog at https://api.tinfoil.sh/v1/models is queried to
# determine which models support /v1/chat/completions vs /v1/responses.
#
# Results are saved to evals/results/<timestamp>/.

set -uo pipefail

BASE="${1:-http://localhost:8089}"
CATALOG_URL="https://api.tinfoil.sh/v1/models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${SCRIPT_DIR}/results/${TIMESTAMP}"
mkdir -p "$OUT/raw"

SEARCH_Q='What were the top 3 news headlines today? Cite every claim with source markers. Be thorough but concise.'
FETCH_Q='Fetch https://en.wikipedia.org/wiki/Python_(programming_language) and give me a 3-sentence summary of what Python is, citing the page.'

echo "Websearch eval: $BASE"
echo "Output: $OUT"
echo ""

# Fetch model catalog
CATALOG=$(curl -s "$CATALOG_URL")
if [ -z "$CATALOG" ]; then
  echo "ERROR: Could not fetch model catalog from $CATALOG_URL"
  exit 1
fi

CHAT_MODELS=$(echo "$CATALOG" | python3 -c "
import json,sys
data=json.load(sys.stdin)
for m in data['data']:
    if m['type']=='chat' and m.get('tool_calling') and '/v1/chat/completions' in m.get('endpoints',[]):
        print(m['id'])
" 2>/dev/null)

RESP_MODELS=$(echo "$CATALOG" | python3 -c "
import json,sys
data=json.load(sys.stdin)
for m in data['data']:
    if m['type']=='chat' and m.get('tool_calling') and '/v1/responses' in m.get('endpoints',[]):
        print(m['id'])
" 2>/dev/null)

if [ -z "$CHAT_MODELS" ] && [ -z "$RESP_MODELS" ]; then
  echo "ERROR: No supported models found in catalog (JSON parse failure or empty catalog)"
  exit 1
fi

echo "Chat completions models: $(echo $CHAT_MODELS | tr '\n' ', ')"
echo "Responses models: $(echo $RESP_MODELS | tr '\n' ', ')"
echo ""

do_curl() {
  local url=$1 fname=$2 stream=$3 body=$4
  local ext="json"
  [ "$stream" = "true" ] && ext="sse"
  local start=$(date +%s)
  curl -sS --max-time 300 "$url" \
    -H "Content-Type: application/json" \
    -d "$body" > "$OUT/raw/${fname}.${ext}" 2>"$OUT/raw/${fname}.err"
  local end=$(date +%s)
  echo $((end - start)) > "$OUT/raw/${fname}.elapsed"
}

make_body_chat() {
  local model=$1 stream_bool=$2 prompt=$3
  python3 << PYEOF
import json
stream_val = True if "$stream_bool" == "true" else False
print(json.dumps({
    "model": "$model",
    "messages": [{"role": "user", "content": "$prompt"}],
    "stream": stream_val,
    "web_search_options": {"enabled": True}
}))
PYEOF
}

make_body_responses() {
  local model=$1 stream_bool=$2 prompt=$3
  python3 << PYEOF
import json
stream_val = True if "$stream_bool" == "true" else False
print(json.dumps({
    "model": "$model",
    "input": "$prompt",
    "stream": stream_val,
    "tools": [{"type": "web_search"}]
}))
PYEOF
}

run_test() {
  local model=$1 endpoint=$2 stream=$3 task=$4
  local safe_model="${model//\//_}"
  local fname="${safe_model}__${endpoint}__stream-${stream}__${task}"
  echo "  $fname"

  local prompt="$SEARCH_Q"
  [ "$task" = "fetch" ] && prompt="$FETCH_Q"

  local body
  if [ "$endpoint" = "chat" ]; then
    body=$(make_body_chat "$model" "$stream" "$prompt")
    do_curl "$BASE/v1/chat/completions" "$fname" "$stream" "$body"
  else
    body=$(make_body_responses "$model" "$stream" "$prompt")
    do_curl "$BASE/v1/responses" "$fname" "$stream" "$body"
  fi
}

# Run the matrix
for task in search fetch; do
  echo "=== Chat Completions | Non-Streaming | ${task} ==="
  for m in $CHAT_MODELS; do run_test "$m" chat false "$task"; done

  echo "=== Chat Completions | Streaming | ${task} ==="
  for m in $CHAT_MODELS; do run_test "$m" chat true "$task"; done

  echo "=== Responses | Non-Streaming | ${task} ==="
  for m in $RESP_MODELS; do run_test "$m" responses false "$task"; done

  echo "=== Responses | Streaming | ${task} ==="
  for m in $RESP_MODELS; do run_test "$m" responses true "$task"; done
done

echo ""
echo "Raw results saved. Running analysis..."
echo ""

python3 "${SCRIPT_DIR}/analyze_websearch_eval.py" "$OUT"
