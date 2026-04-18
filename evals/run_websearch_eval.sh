#!/bin/bash
#
# Websearch via router evaluation matrix.
#
# Exercises the router-owned websearch flow across models, endpoints, streaming
# modes, retrieval-depth buckets (search_context_size low|medium|high), and
# dedicated parameter-assertion tasks that verify user_location and
# allowed_domains are honored end-to-end.
#
# Usage:
#   TINFOIL_API_KEY=... ./evals/run_websearch_eval.sh
#   TINFOIL_API_KEY=... ./evals/run_websearch_eval.sh http://localhost:8090
#   TINFOIL_API_KEY=... USE_LOCAL_FIXTURES=1 ./evals/run_websearch_eval.sh
#
# Environment overrides:
#   TINFOIL_API_KEY     required; bearer token the router accepts.
#   USE_LOCAL_FIXTURES  when 1, swap real-web prompts for the local-fixture set.
#   CONTEXT_SIZES       space-separated retrieval-depth buckets (default
#                       "low medium high"; set to "medium" for a quick smoke run).
#   TASKS               space-separated tasks to run; defaults to the full set
#                       (depth-low depth-medium depth-high search fetch
#                        location domains).
#   SKIP_LLM_JUDGE      when 1, skip the LLM-as-judge step in the analyzer.
#
# Results are saved to evals/results/<timestamp>/.

set -uo pipefail

BASE="${1:-http://localhost:8090}"
CATALOG_URL="https://api.tinfoil.sh/v1/models"
API_KEY="${TINFOIL_API_KEY:-}"
USE_LOCAL_FIXTURES="${USE_LOCAL_FIXTURES:-0}"
CONTEXT_SIZES="${CONTEXT_SIZES:-low medium high}"
TASKS="${TASKS:-depth-low depth-medium depth-high search fetch location domains}"
SKIP_LLM_JUDGE="${SKIP_LLM_JUDGE:-0}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${SCRIPT_DIR}/results/${TIMESTAMP}"
mkdir -p "$OUT/raw"

if [ -z "$API_KEY" ]; then
  echo "ERROR: TINFOIL_API_KEY must be set"
  exit 1
fi

CHAT_MAX_TOKENS=300
RESP_MAX_TOKENS=300
RESP_STREAM_MAX_TOKENS=500

prompt_for_task() {
  local task=$1
  if [ "$USE_LOCAL_FIXTURES" = "1" ]; then
    case "$task" in
      depth-low|search)
        echo 'Search the web and answer in one sentence: According to the Neighborhood Cat Gazette, which cushions do the cats in the sunroom prefer? Cite the source.'
        ;;
      depth-medium)
        echo 'Compare what the Neighborhood Cat Gazette and the Cat Almanac say about breakfast routines. Cite sources.'
        ;;
      depth-high)
        echo 'Produce a 3-bullet brief on the cats in the sunroom using any relevant fixture page. Cite every claim.'
        ;;
      fetch)
        echo 'Fetch https://local.test/cats/almanac and answer in one sentence: According to the page, what does Nimbus do after breakfast? Cite the source.'
        ;;
      location)
        echo 'What is the weather today? Cite sources.'
        ;;
      domains)
        echo 'Where should I learn Python? Cite only the official site.'
        ;;
    esac
    return
  fi
  case "$task" in
    depth-low)
      echo 'What is the current price of Bitcoin in USD? One sentence. Cite your source.'
      ;;
    depth-medium)
      echo 'Who won the 2025 Nobel Prize in Physics, and what prior discovery made their work possible? Cite at least three sources.'
      ;;
    depth-high)
      echo 'Produce a 5-bullet brief on the NIST post-quantum cryptography finalists, their intended use, and their status. Cite every claim and follow any links required to verify.'
      ;;
    search)
      echo 'What were the top 3 news headlines today? Cite every claim with source markers. Be thorough but concise.'
      ;;
    fetch)
      echo 'Fetch https://en.wikipedia.org/wiki/Python_(programming_language) and give me a 3-sentence summary of what Python is, citing the page.'
      ;;
    location)
      echo 'Search the web for "weather forecast today" and tell me what the first few results are reporting. Cite every result you mention.'
      ;;
    domains)
      echo 'What is Python? Cite only official sources.'
      ;;
  esac
}

echo "Websearch eval (router): $BASE"
echo "Output: $OUT"
echo "Context sizes: $CONTEXT_SIZES"
echo "Tasks: $TASKS"
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

# Persist the exact user prompt alongside each response so the analyzer grades
# answers against what was actually asked, independent of fixture vs real-web
# mode.
save_prompt() {
  local fname=$1 prompt=$2
  printf '%s' "$prompt" > "$OUT/raw/${fname}.prompt"
}

# Build a Chat Completions request body. Parameter forwarding happens via
# web_search_options: search_context_size always, plus task-specific fields for
# user_location and allowed_domains so we can verify that the router + MCP
# server honor each flag end-to-end.
make_body_chat() {
  local model=$1 stream_bool=$2 prompt=$3 ctx=$4 task=$5
  MODEL="$model" STREAM_BOOL="$stream_bool" PROMPT="$prompt" CTX="$ctx" TASK="$task" \
    CHAT_MAX_TOKENS="$CHAT_MAX_TOKENS" python3 <<'PYEOF'
import json
import os

stream_val = os.environ["STREAM_BOOL"] == "true"
task = os.environ["TASK"]
ctx = os.environ["CTX"]

web_opts = {"search_context_size": ctx}
if task == "location":
    web_opts["user_location"] = {
        "type": "approximate",
        "approximate": {"country": "GB", "city": "London"},
    }
if task == "domains":
    web_opts["filters"] = {"allowed_domains": ["python.org"]}

print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "stream": stream_val,
    "web_search_options": web_opts,
    "temperature": 0,
    "max_tokens": int(os.environ["CHAT_MAX_TOKENS"])
}))
PYEOF
}

make_body_responses() {
  local model=$1 stream_bool=$2 prompt=$3 ctx=$4 task=$5
  MODEL="$model" STREAM_BOOL="$stream_bool" PROMPT="$prompt" CTX="$ctx" TASK="$task" \
    RESP_MAX_TOKENS="$RESP_MAX_TOKENS" RESP_STREAM_MAX_TOKENS="$RESP_STREAM_MAX_TOKENS" python3 <<'PYEOF'
import json
import os

stream_val = os.environ["STREAM_BOOL"] == "true"
task = os.environ["TASK"]
ctx = os.environ["CTX"]

tool = {"type": "web_search", "search_context_size": ctx}
if task == "location":
    tool["user_location"] = {
        "type": "approximate",
        "approximate": {"country": "GB", "city": "London"},
    }
if task == "domains":
    tool["filters"] = {"allowed_domains": ["python.org"]}

print(json.dumps({
    "model": os.environ["MODEL"],
    "input": os.environ["PROMPT"],
    "stream": stream_val,
    "tools": [tool],
    "temperature": 0,
    "max_output_tokens": int(os.environ["RESP_STREAM_MAX_TOKENS"] if stream_val else os.environ["RESP_MAX_TOKENS"])
}))
PYEOF
}

run_test() {
  local model=$1 endpoint=$2 stream=$3 ctx=$4 task=$5
  local safe_model="${model//\//_}"
  local fname="${safe_model}__${endpoint}__stream-${stream}__ctx-${ctx}__${task}"
  echo "  $fname"

  local prompt
  prompt=$(prompt_for_task "$task")
  if [ -z "$prompt" ]; then
    echo "    (no prompt for task $task; skipping)"
    return
  fi

  save_prompt "$fname" "$prompt"

  local body
  if [ "$endpoint" = "chat" ]; then
    body=$(make_body_chat "$model" "$stream" "$prompt" "$ctx" "$task")
    do_curl "$BASE/v1/chat/completions" "$fname" "$stream" "$body"
  else
    body=$(make_body_responses "$model" "$stream" "$prompt" "$ctx" "$task")
    do_curl "$BASE/v1/responses" "$fname" "$stream" "$body"
  fi
}

# Assertion tasks only need a single context bucket to validate flag passthrough.
is_assertion_task() {
  case "$1" in
    location|domains) return 0 ;;
    *) return 1 ;;
  esac
}

for task in $TASKS; do
  if is_assertion_task "$task"; then
    contexts="medium"
  else
    contexts="$CONTEXT_SIZES"
  fi

  for ctx in $contexts; do
    echo "=== task=${task} ctx=${ctx} | chat | non-streaming ==="
    for m in $CHAT_MODELS; do run_test "$m" chat false "$ctx" "$task"; done

    echo "=== task=${task} ctx=${ctx} | chat | streaming ==="
    for m in $CHAT_MODELS; do run_test "$m" chat true "$ctx" "$task"; done

    echo "=== task=${task} ctx=${ctx} | responses | non-streaming ==="
    for m in $RESP_MODELS; do run_test "$m" responses false "$ctx" "$task"; done

    echo "=== task=${task} ctx=${ctx} | responses | streaming ==="
    for m in $RESP_MODELS; do run_test "$m" responses true "$ctx" "$task"; done
  done
done

echo ""
echo "Raw results saved. Running analysis..."
echo ""

SKIP_LLM_JUDGE="$SKIP_LLM_JUDGE" TINFOIL_API_KEY="$API_KEY" \
  python3 "${SCRIPT_DIR}/analyze_websearch_eval.py" "$OUT"
