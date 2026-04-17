#!/bin/bash
#
# Websearch via router evaluation matrix.
#
# Runs all supported model/endpoint/stream/task combinations and saves
# raw responses as JSON/SSE files plus a summary results.json.
#
# Usage:
#   TINFOIL_API_KEY=... ./evals/run_websearch_eval.sh
#   TINFOIL_API_KEY=... ./evals/run_websearch_eval.sh http://localhost:8090
#   TINFOIL_API_KEY=... USE_LOCAL_FIXTURES=1 ./evals/run_websearch_eval.sh
#
# The model catalog at https://api.tinfoil.sh/v1/models is queried to
# determine which models support /v1/chat/completions vs /v1/responses.
#
# Results are saved to evals/results/<timestamp>/.

set -uo pipefail

BASE="${1:-http://localhost:8090}"
CATALOG_URL="https://api.tinfoil.sh/v1/models"
API_KEY="${TINFOIL_API_KEY:-}"
USE_LOCAL_FIXTURES="${USE_LOCAL_FIXTURES:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${SCRIPT_DIR}/results/${TIMESTAMP}"
mkdir -p "$OUT/raw"

if [ -z "$API_KEY" ]; then
  echo "ERROR: TINFOIL_API_KEY must be set"
  exit 1
fi

if [ "$USE_LOCAL_FIXTURES" = "1" ]; then
  SEARCH_Q='Search the web and answer in one sentence: According to the Neighborhood Cat Gazette, which cushions do the cats in the sunroom prefer? Cite the source.'
  FETCH_Q='Fetch https://local.test/cats/almanac and answer in one sentence: According to the page, what does Nimbus do after breakfast? Cite the source.'
  CHAT_MAX_TOKENS=120
  RESP_MAX_TOKENS=120
else
  SEARCH_Q='What were the top 3 news headlines today? Cite every claim with source markers. Be thorough but concise.'
  FETCH_Q='Fetch https://en.wikipedia.org/wiki/Python_(programming_language) and give me a 3-sentence summary of what Python is, citing the page.'
  CHAT_MAX_TOKENS=220
  RESP_MAX_TOKENS=220
fi

RESP_STREAM_MAX_TOKENS=400

echo "Websearch eval (router): $BASE"
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
    -H "Authorization: Bearer $API_KEY" \
    -d "$body" > "$OUT/raw/${fname}.${ext}" 2>"$OUT/raw/${fname}.err"
  local end=$(date +%s)
  echo $((end - start)) > "$OUT/raw/${fname}.elapsed"
}

make_body_chat() {
  local model=$1 stream_bool=$2 prompt=$3
  MODEL="$model" STREAM_BOOL="$stream_bool" PROMPT="$prompt" CHAT_MAX_TOKENS="$CHAT_MAX_TOKENS" python3 <<'PYEOF'
import json
import os

stream_val = os.environ["STREAM_BOOL"] == "true"
print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "stream": stream_val,
    "web_search_options": {},
    "temperature": 0,
    "max_tokens": int(os.environ["CHAT_MAX_TOKENS"])
}))
PYEOF
}

make_body_responses() {
  local model=$1 stream_bool=$2 prompt=$3
  MODEL="$model" STREAM_BOOL="$stream_bool" PROMPT="$prompt" RESP_MAX_TOKENS="$RESP_MAX_TOKENS" RESP_STREAM_MAX_TOKENS="$RESP_STREAM_MAX_TOKENS" python3 <<'PYEOF'
import json
import os

stream_val = os.environ["STREAM_BOOL"] == "true"
print(json.dumps({
    "model": os.environ["MODEL"],
    "input": os.environ["PROMPT"],
    "stream": stream_val,
    "tools": [{"type": "web_search"}],
    "temperature": 0,
    "max_output_tokens": int(os.environ["RESP_STREAM_MAX_TOKENS"] if stream_val else os.environ["RESP_MAX_TOKENS"])
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
