#!/usr/bin/env python3
"""
Analyze websearch eval results and produce results.json + console summary.

Usage:
    python3 evals/analyze_websearch_eval.py evals/results/<timestamp>

Environment:
    TINFOIL_API_KEY     required for the LLM-as-judge grading step.
    SKIP_LLM_JUDGE=1    skip the LLM grading step (useful for offline runs).
    JUDGE_MODEL         override the judge model (default: glm-5-1).
    JUDGE_ENDPOINT      override the judge endpoint
                        (default: https://inference.tinfoil.sh/v1/chat/completions).
    JUDGE_CONCURRENCY   number of judge calls to run in parallel (default 4).
    JUDGE_TIMEOUT       seconds to wait for a judge response (default 60).
"""

import concurrent.futures
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

CITATION_PATTERN = re.compile(r"\u3010(\d+)\u3011")
URL_CITATION_PATTERN = re.compile(r"https?://[^\s\]\)】]+")

DEFAULT_JUDGE_MODEL = "glm-5-1"
DEFAULT_JUDGE_ENDPOINT = "https://inference.tinfoil.sh/v1/chat/completions"
DEFAULT_JUDGE_CONCURRENCY = 4
DEFAULT_JUDGE_TIMEOUT = 60

LOCATION_PATTERN = re.compile(r"\b(london|uk|united kingdom|england|britain)\b", re.IGNORECASE)
LOCATION_DOMAIN_PATTERN = re.compile(r"\.uk(/|$)|bbc\.|theguardian\.|metoffice\.|sky\.com", re.IGNORECASE)

# Parameter-assertion configuration: task name -> assertion description.
# Each task's assertion is graded by `run_assertion` below.
ASSERTION_TASKS = {
    "location": "response references London or the UK",
    "domains": "all annotations come from python.org",
}


def parse_nonstream(raw, endpoint):
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None, None, None, f"invalid JSON: {raw[:200]}"
    if not isinstance(data, dict):
        return None, None, None, f"unexpected JSON type: {type(data).__name__}"

    err = data.get("error")
    if isinstance(err, dict) and err.get("message"):
        return None, None, None, err["message"][:500]
    if isinstance(err, str) and err:
        return None, None, None, err[:500]

    content = ""
    annotations = []
    web_search_calls = []

    if "choices" in data:
        if not data["choices"]:
            return None, None, None, "empty choices"
        msg = data["choices"][0].get("message", {})
        content = msg.get("content", "") or ""
        for a in msg.get("annotations") or []:
            if a:
                annotations.append(a)
    elif "output" in data:
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    content += c.get("text", "") or ""
                    for a in c.get("annotations") or []:
                        if a:
                            annotations.append(a)
            elif item.get("type") == "web_search_call":
                web_search_calls.append(item)
    else:
        return None, None, None, f"unknown format: {list(data.keys())[:5]}"

    return content, annotations, web_search_calls, None


def parse_stream(raw, endpoint):
    if not raw.strip():
        return None, None, None, "empty response"

    # Non-SSE error response
    if not raw.strip().startswith("data:") and not raw.strip().startswith("event:"):
        try:
            obj = json.loads(raw.strip())
            err = obj.get("error")
            if isinstance(err, dict):
                return None, None, None, err.get("message", str(err))[:500]
            if err:
                return None, None, None, str(err)[:500]
        except (json.JSONDecodeError, ValueError):
            pass

    content = ""
    annotations = []
    web_search_calls = []

    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue

        etype = obj.get("type", "")

        # Top-level web_search_call SSE event (chat completions path).
        if etype == "web_search_call":
            web_search_calls.append(obj)

        # Chat completions delta
        if "choices" in obj:
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content"):
                    content += delta["content"]
                for a in delta.get("annotations") or []:
                    if a:
                        annotations.append(a)

        # Responses text delta
        if etype == "response.output_text.delta":
            content += obj.get("delta", "")

        # Responses completed -> annotations + web_search_calls
        if etype == "response.completed":
            resp = obj.get("response", {})
            for out in resp.get("output", []):
                if out.get("type") == "message":
                    for c in out.get("content", []):
                        for a in c.get("annotations") or []:
                            if a:
                                annotations.append(a)
                elif out.get("type") == "web_search_call":
                    web_search_calls.append(out)

        # Responses item.added (web_search_call events streamed inline).
        if etype == "response.output_item.added":
            item = obj.get("item", {})
            if item.get("type") == "web_search_call":
                web_search_calls.append(item)

        # Error in stream
        if "error" in obj and isinstance(obj["error"], dict):
            return None, None, None, obj["error"].get("message", str(obj["error"]))[:500]

    return content, annotations, web_search_calls, None


def parse_filename(base):
    """Return (model, endpoint, stream, ctx, task) or None if unparseable.

    Supports both the new ctx-aware encoding and the legacy encoding
    (no context bucket) so old result directories still analyze cleanly.
    """
    parts = base.split("__")
    if len(parts) == 5:
        model, endpoint, stream_part, ctx_part, task = parts
        stream = "true" in stream_part
        ctx = ctx_part.replace("ctx-", "", 1)
        return model, endpoint, stream, ctx, task
    if len(parts) == 4:
        model, endpoint, stream_part, task = parts
        stream = "true" in stream_part
        return model, endpoint, stream, "legacy", task
    return None


def analyze_file(results_dir, fname):
    path = os.path.join(results_dir, "raw", fname)
    base, ext = os.path.splitext(fname)
    # Skip sidecar files; they're consumed via explicit lookups below.
    if ext in (".err", ".elapsed", ".prompt") or base.endswith(".last_call"):
        return None

    parsed = parse_filename(base)
    if parsed is None:
        return None
    model, endpoint, stream, ctx, task = parsed

    with open(path, "r") as f:
        raw = f.read()

    elapsed_path = os.path.join(results_dir, "raw", f"{base}.elapsed")
    elapsed = 0
    if os.path.exists(elapsed_path):
        try:
            elapsed = int(open(elapsed_path).read().strip())
        except ValueError:
            pass

    prompt_path = os.path.join(results_dir, "raw", f"{base}.prompt")
    question = ""
    if os.path.exists(prompt_path):
        try:
            with open(prompt_path, "r") as fh:
                question = fh.read().strip()
        except OSError:
            question = ""

    # .err is written by the Python driver when the HTTP layer itself fails
    # (timeout, connection reset, non-2xx). Treat that as a distinct
    # `transport_error` so the analyzer can tell a broken network from a
    # broken model.
    err_path = os.path.join(results_dir, "raw", f"{base}.err")
    transport_error = ""
    if os.path.exists(err_path):
        try:
            transport_error = open(err_path).read().strip()[:300]
        except OSError:
            transport_error = ""

    # .last_call.json is only written by run_websearch_eval.py when
    # LOCAL_WEBSEARCH_MCP_BASE is set (fixture mode). When present it is the
    # ground truth for the parameter-assertion tasks.
    last_call_path = os.path.join(results_dir, "raw", f"{base}.last_call.json")
    last_call = None
    if os.path.exists(last_call_path):
        try:
            with open(last_call_path, "r") as fh:
                last_call = json.load(fh)
        except (OSError, json.JSONDecodeError):
            last_call = None

    if ext == ".json":
        content, annotations, web_search_calls, error = parse_nonstream(raw, endpoint)
    elif ext == ".sse":
        content, annotations, web_search_calls, error = parse_stream(raw, endpoint)
    else:
        return None

    content = content or ""
    annotations = annotations or []
    web_search_calls = web_search_calls or []
    markers = CITATION_PATTERN.findall(content)
    urls = URL_CITATION_PATTERN.findall(content)
    unique_references = len(set(markers)) + len(set(urls))

    # Heuristic quality grade; the LLM-as-judge score is attached separately
    # under the "judge" key so both signals are visible side by side.
    # Transport errors outrank model errors so we can separate infra from
    # model-quality regressions in the aggregate.
    if transport_error:
        heuristic = "TRANSPORT_ERROR"
    elif error:
        heuristic = "ERROR"
    elif not content.strip():
        heuristic = "EMPTY"
    elif unique_references and annotations:
        if unique_references == len(annotations):
            heuristic = "EXCELLENT"
        else:
            heuristic = "GOOD"
    elif unique_references:
        heuristic = "GOOD"
    elif annotations:
        heuristic = "PARTIAL_ANNOTATIONS"
    else:
        heuristic = "NO_CITATIONS"

    assertion = run_assertion(
        task, content, annotations, web_search_calls, error, last_call
    )

    return {
        "model": model,
        "endpoint": endpoint,
        "stream": stream,
        "ctx": ctx,
        "task": task,
        "question": question,
        "elapsed_seconds": elapsed,
        "content_length": len(content),
        "content_snippet": content[:300].replace("\n", " "),
        "content": content,
        "citation_markers": len(markers),
        "url_citations": len(set(urls)),
        "annotations": len(annotations),
        "annotation_details": [
            _flatten_annotation(a) for a in annotations[:10]
        ],
        "web_search_calls": len(web_search_calls),
        "heuristic_grade": heuristic,
        "assertion": assertion,
        "error": error,
        "transport_error": transport_error or None,
        "last_call": last_call,
    }


def _flatten_annotation(annotation):
    """Return a (url, title) dict regardless of nested vs flat shape."""
    if not isinstance(annotation, dict):
        return {"url": "", "title": ""}
    if "url_citation" in annotation and isinstance(annotation["url_citation"], dict):
        inner = annotation["url_citation"]
        return {"url": inner.get("url", ""), "title": inner.get("title", "")}
    return {"url": annotation.get("url", ""), "title": annotation.get("title", "")}


def run_assertion(task, content, annotations, web_search_calls, error, last_call):
    """Evaluate the task's parameter-assertion; return None for non-assertion tasks.

    When a /debug/last-call snapshot is available (fixture mode against a
    local MCP server), assertions grade directly against the SearchArgs the
    server observed. That makes the test fail fast when a request-shaping
    field is dropped between the router and the MCP tool call, independent
    of whether the upstream index actually honors the hint.

    When no snapshot is available (real-web runs), assertions fall back to
    heuristics on the answer text and annotations.
    """
    if task not in ASSERTION_TASKS:
        return None

    if task == "location":
        snapshot = _last_search_snapshot(last_call)
        if snapshot is not None:
            country = (snapshot.get("user_location_country") or "").upper()
            if country == "GB":
                return {
                    "pass": True,
                    "reason": "MCP search observed user_location_country=GB",
                }
            return {
                "pass": False,
                "reason": (
                    "MCP search did not observe user_location_country=GB "
                    f"(got {country!r})"
                ),
            }

        if error:
            return {
                "pass": False,
                "reason": f"request errored (location may have broken something): {error[:100]}",
            }
        if not web_search_calls and not annotations:
            return {
                "pass": False,
                "reason": "no search activity observed; cannot confirm user_location was plumbed",
            }
        if LOCATION_PATTERN.search(content or ""):
            return {"pass": True, "reason": "response mentions London / UK"}
        for ann in annotations:
            url = _flatten_annotation(ann).get("url", "")
            if LOCATION_DOMAIN_PATTERN.search(url):
                return {
                    "pass": True,
                    "reason": f"annotation from UK-biased domain: {url}",
                }
        return {
            "pass": True,
            "reason": (
                "search ran with user_location forwarded; upstream index did "
                "not bias results toward UK (soft pass)"
            ),
        }

    if task == "domains":
        snapshot = _last_search_snapshot(last_call)
        if snapshot is not None:
            observed = _normalize_domain_list(snapshot.get("allowed_domains"))
            expected = {"python.org"}
            if observed == expected:
                return {
                    "pass": True,
                    "reason": "MCP search observed allowed_domains=['python.org']",
                }
            return {
                "pass": False,
                "reason": (
                    "MCP search observed allowed_domains="
                    f"{sorted(observed)}, expected ['python.org']"
                ),
            }

        if error:
            return {"pass": False, "reason": f"request errored: {error[:100]}"}
        if not annotations:
            return {"pass": False, "reason": "no annotations produced; cannot verify domain filter"}
        off_domain = []
        for ann in annotations:
            flat = _flatten_annotation(ann)
            url = flat.get("url", "")
            host = urllib.parse.urlparse(url).hostname or ""
            host = host.lower()
            if host == "python.org" or host.endswith(".python.org"):
                continue
            off_domain.append(url)
        if off_domain:
            return {
                "pass": False,
                "reason": f"annotations outside python.org: {off_domain[:3]}",
            }
        return {"pass": True, "reason": "all annotations rooted at python.org"}

    return None


def _last_search_snapshot(last_call):
    """Return the `search` block of a /debug/last-call payload if it reflects
    a search that was actually observed (non-empty `observed_at`), else None.

    A zero observed_at marks "no search call has been recorded yet" and must
    not be treated as a passing assertion.
    """
    if not isinstance(last_call, dict):
        return None
    search_block = last_call.get("search")
    if not isinstance(search_block, dict):
        return None
    observed_at = (search_block.get("observed_at") or "").strip()
    if not observed_at or observed_at.startswith("0001-01-01"):
        return None
    return search_block


def _normalize_domain_list(raw):
    """Coerce an allowed/excluded domain field into a lowercase set."""
    if not raw:
        return set()
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        return set()
    return {str(item).strip().lower() for item in items if str(item).strip()}


def build_judge_prompt(result):
    """Compact rubric prompt for the LLM-as-judge call."""
    question = _question_for(result)
    annotation_lines = "\n".join(
        f"  - {a.get('title', '')[:80]} ({a.get('url', '')})"
        for a in result.get("annotation_details", [])
    ) or "  (none)"
    user = (
        f"Question: {question}\n\n"
        f"Answer:\n{result['content'][:2500]}\n\n"
        f"Annotations cited by the model:\n{annotation_lines}\n\n"
        "Grade the answer against the question. Output ONLY strict JSON with "
        'the following keys and nothing else: {"correctness":0-3,'
        '"cited_support":0-3,"helpfulness":0-3,'
        '"verdict":"excellent|good|partial|poor","notes":"<one sentence>"}.'
    )
    return user


def _question_for(result):
    """Return the exact prompt sent to the router, so the judge grades the
    actual answer against the actual question (works for real-web, fixture,
    and any future prompt variants). Old result dirs without a .prompt sidecar
    fall back to a generic task label."""
    question = (result.get("question") or "").strip()
    if question:
        return question
    return f"Task: {result['task']}"


def _judge_http_call(api_key, endpoint, model, prompt, timeout):
    payload = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an evaluator grading a web search answer. "
                    "Respond with STRICT JSON only: no prose, no markdown, no "
                    'extra keys. Shape: {"correctness":0-3,'
                    '"cited_support":0-3,"helpfulness":0-3,'
                    '"verdict":"excellent|good|partial|poor","notes":"<one sentence>"}.'
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 800,
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return body


def grade_with_llm(result, api_key, model, endpoint, timeout):
    if result.get("error") or not result.get("content", "").strip():
        return {
            "verdict": "poor",
            "correctness": 0,
            "cited_support": 0,
            "helpfulness": 0,
            "notes": "empty or errored response; not sent to judge",
            "skipped": True,
        }
    prompt = build_judge_prompt(result)
    try:
        body = _judge_http_call(api_key, endpoint, model, prompt, timeout)
        data = json.loads(body)
        content = data["choices"][0]["message"].get("content") or ""
        verdict_json = _extract_json_object(content)
        if verdict_json is None:
            return {
                "verdict": "poor",
                "correctness": 0,
                "cited_support": 0,
                "helpfulness": 0,
                "notes": f"judge did not return JSON: {content[:120]}",
                "skipped": True,
            }
        return {
            "verdict": verdict_json.get("verdict", "poor"),
            "correctness": int(verdict_json.get("correctness", 0)),
            "cited_support": int(verdict_json.get("cited_support", 0)),
            "helpfulness": int(verdict_json.get("helpfulness", 0)),
            "notes": str(verdict_json.get("notes", ""))[:200],
            "skipped": False,
        }
    except urllib.error.HTTPError as exc:
        return {
            "verdict": "poor",
            "correctness": 0,
            "cited_support": 0,
            "helpfulness": 0,
            "notes": f"judge HTTP error: {exc.code} {exc.reason}",
            "skipped": True,
        }
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return {
            "verdict": "poor",
            "correctness": 0,
            "cited_support": 0,
            "helpfulness": 0,
            "notes": f"judge network error: {exc}",
            "skipped": True,
        }
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        return {
            "verdict": "poor",
            "correctness": 0,
            "cited_support": 0,
            "helpfulness": 0,
            "notes": f"judge parse error: {exc}",
            "skipped": True,
        }


def _extract_json_object(raw):
    """Pull the first top-level JSON object out of the judge response."""
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def run_llm_judge(results, api_key, model, endpoint, concurrency, timeout):
    """Grade each result with the LLM-as-judge in parallel."""
    if not results:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(grade_with_llm, r, api_key, model, endpoint, timeout): r
            for r in results
        }
        done = 0
        total = len(futures)
        for future in concurrent.futures.as_completed(futures):
            r = futures[future]
            r["judge"] = future.result()
            done += 1
            if done % 5 == 0 or done == total:
                print(f"  judged {done}/{total}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    raw_dir = os.path.join(results_dir, "raw")
    if not os.path.isdir(raw_dir):
        print(f"ERROR: {raw_dir} not found")
        sys.exit(1)

    results = []
    for fname in sorted(os.listdir(raw_dir)):
        r = analyze_file(results_dir, fname)
        if r is not None:
            results.append(r)

    skip_llm = os.environ.get("SKIP_LLM_JUDGE", "0") == "1"
    api_key = os.environ.get("TINFOIL_API_KEY", "")
    if skip_llm or not api_key:
        for r in results:
            r["judge"] = {"skipped": True, "notes": "skipped"}
        if not api_key and not skip_llm:
            print("NOTE: TINFOIL_API_KEY not set, skipping LLM-as-judge step.")
    else:
        judge_model = os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        judge_endpoint = os.environ.get("JUDGE_ENDPOINT", DEFAULT_JUDGE_ENDPOINT)
        judge_concurrency = int(
            os.environ.get("JUDGE_CONCURRENCY", str(DEFAULT_JUDGE_CONCURRENCY))
        )
        judge_timeout = int(
            os.environ.get("JUDGE_TIMEOUT", str(DEFAULT_JUDGE_TIMEOUT))
        )
        print(
            f"\nRunning LLM-as-judge ({judge_model}, concurrency={judge_concurrency}) "
            f"on {len(results)} results..."
        )
        run_llm_judge(
            results, api_key, judge_model, judge_endpoint, judge_concurrency, judge_timeout
        )

    # Strip the full content from the serialized row to keep results.json small;
    # the snippet + judge notes are what post-hoc readers need.
    serializable = []
    for r in results:
        row = dict(r)
        row.pop("content", None)
        serializable.append(row)

    output_path = os.path.join(results_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    render_console_summary(results)

    print(f"\n  Results saved to: {output_path}")


def render_console_summary(results):
    print(f"{'=' * 160}")
    print(f"  WEBSEARCH EVAL RESULTS ({len(results)} tests)")
    print(f"{'=' * 160}")
    print(
        f"  {'Model':<16} {'API':<10} {'S':<3} {'Ctx':<7} {'Task':<12} "
        f"{'Time':>5} {'Len':>6} {'Mkrs':>5} {'URLs':>5} {'Anns':>5} {'WSC':>4} "
        f"{'Heuristic':<18} {'Verdict':<10} {'Assert':<7} Notes"
    )
    print(f"  {'-' * 160}")

    for r in results:
        stream_s = "y" if r["stream"] else "n"
        judge = r.get("judge") or {}
        verdict = "-" if judge.get("skipped") else judge.get("verdict", "-")
        assertion = r.get("assertion")
        if assertion is None:
            assert_s = "n/a"
        else:
            assert_s = "pass" if assertion["pass"] else "fail"
        notes = (judge.get("notes") or "")[:55]
        if not notes and r.get("error"):
            notes = r["error"][:55]
        elif not notes and assertion is not None:
            notes = assertion["reason"][:55]
        print(
            f"  {r['model']:<16} {r['endpoint']:<10} {stream_s:<3} {r['ctx']:<7} {r['task']:<12} "
            f"{r['elapsed_seconds']:>4}s {r['content_length']:>6} {r['citation_markers']:>5} "
            f"{r['url_citations']:>5} {r['annotations']:>5} {r['web_search_calls']:>4} "
            f"{r['heuristic_grade']:<18} {verdict:<10} {assert_s:<7} {notes}"
        )

    render_llm_summary(results)
    render_assertion_summary(results)
    render_parity_summary(results)
    render_overall_summary(results)


def render_llm_summary(results):
    rows = [r for r in results if not r.get("judge", {}).get("skipped")]
    if not rows:
        return
    print(f"\n  {'=' * 120}")
    print(f"  LLM-AS-JUDGE AGGREGATE ({len(rows)} judged)")
    print(f"  {'=' * 120}")

    def _avg(field):
        values = [r["judge"].get(field, 0) for r in rows]
        return sum(values) / max(len(values), 1)

    print(f"  Avg correctness  : {_avg('correctness'):.2f} / 3")
    print(f"  Avg cited_support: {_avg('cited_support'):.2f} / 3")
    print(f"  Avg helpfulness  : {_avg('helpfulness'):.2f} / 3")

    verdict_counts = {}
    for r in rows:
        v = r["judge"].get("verdict", "poor")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    counts = " ".join(f"{k}={v}" for k, v in sorted(verdict_counts.items()))
    print(f"  Verdicts: {counts}")

    # Per-context bucket breakdown to make retrieval-depth regressions obvious.
    buckets = {}
    for r in rows:
        buckets.setdefault(r["ctx"], []).append(r)
    for ctx, ctx_rows in sorted(buckets.items()):
        avg_correct = sum(r["judge"].get("correctness", 0) for r in ctx_rows) / len(
            ctx_rows
        )
        avg_helpful = sum(r["judge"].get("helpfulness", 0) for r in ctx_rows) / len(
            ctx_rows
        )
        print(
            f"  ctx={ctx:<8} n={len(ctx_rows):<3} correctness={avg_correct:.2f} "
            f"helpfulness={avg_helpful:.2f}"
        )


def render_assertion_summary(results):
    rows = [r for r in results if r.get("assertion")]
    if not rows:
        return
    print(f"\n  {'=' * 120}")
    print(f"  PARAMETER ASSERTIONS ({len(rows)} checks)")
    print(f"  {'=' * 120}")
    passed = sum(1 for r in rows if r["assertion"]["pass"])
    print(f"  Pass: {passed}/{len(rows)}")
    for r in rows:
        if r["assertion"]["pass"]:
            continue
        print(
            f"    FAIL  {r['model']:<16} {r['endpoint']:<10} task={r['task']:<14} "
            f"-> {r['assertion']['reason'][:100]}"
        )


def render_parity_summary(results):
    print(f"\n  {'=' * 120}")
    print(f"  STREAMING vs NON-STREAMING PARITY")
    print(f"  {'=' * 120}")
    parity_issues = False
    for model in sorted(set(r["model"] for r in results)):
        for endpoint in ["chat", "responses"]:
            for ctx in sorted(set(r["ctx"] for r in results)):
                for task in sorted(set(r["task"] for r in results)):
                    ns = [
                        r
                        for r in results
                        if r["model"] == model
                        and r["endpoint"] == endpoint
                        and not r["stream"]
                        and r["task"] == task
                        and r["ctx"] == ctx
                    ]
                    st = [
                        r
                        for r in results
                        if r["model"] == model
                        and r["endpoint"] == endpoint
                        and r["stream"]
                        and r["task"] == task
                        and r["ctx"] == ctx
                    ]
                    if not ns or not st:
                        continue
                    ns, st = ns[0], st[0]
                    issues = []
                    ns_ok = ns["content_length"] > 0 and not ns["error"]
                    st_ok = st["content_length"] > 0 and not st["error"]
                    if ns_ok != st_ok:
                        issues.append(
                            f"availability: ns={'OK' if ns_ok else 'FAIL'} "
                            f"stream={'OK' if st_ok else 'FAIL'}"
                        )
                    if ns_ok and st_ok:
                        if ns["annotations"] > 0 and st["annotations"] == 0:
                            issues.append(
                                f"annotations lost in stream: ns={ns['annotations']}, stream=0"
                            )
                        if ns["web_search_calls"] > 0 and st["web_search_calls"] == 0:
                            issues.append(
                                f"web_search_calls lost in stream: "
                                f"ns={ns['web_search_calls']}, stream=0"
                            )
                    if issues:
                        parity_issues = True
                        print(f"  {model}|{endpoint}|ctx={ctx}|task={task}:")
                        for i in issues:
                            print(f"    - {i}")
    if not parity_issues:
        print("  No parity issues found.")


def render_overall_summary(results):
    print(f"\n  {'=' * 120}")
    print(f"  SUMMARY")
    print(f"  {'=' * 120}")

    counts = _overall_counts(results)
    print(
        f"  Total: {counts['total']}  OK: {counts['ok']}  "
        f"ModelErrors: {counts['model_errors']}  "
        f"TransportErrors: {counts['transport_errors']}"
    )
    print(
        f"  Heuristic: EXCELLENT={counts['excellent']}  "
        f"GOOD={counts['good']}  PARTIAL={counts['partial']}  "
        f"NONE={counts['no_citations']}  "
        f"TRANSPORT_ERROR={counts['transport_error_grade']}"
    )
    if counts["assertion_total"]:
        print(
            f"  Assertions: {counts['assertion_passed']}/"
            f"{counts['assertion_total']} passed"
        )


def _overall_counts(results):
    """Single source of truth for top-line aggregate counts.

    Returned keys:
        total, ok, model_errors, transport_errors,
        excellent, good, partial, no_citations, transport_error_grade,
        assertion_total, assertion_passed
    """
    counts = {
        "total": len(results),
        "ok": 0,
        "model_errors": 0,
        "transport_errors": 0,
        "excellent": 0,
        "good": 0,
        "partial": 0,
        "no_citations": 0,
        "transport_error_grade": 0,
        "assertion_total": 0,
        "assertion_passed": 0,
    }
    grade_keys = {
        "EXCELLENT": "excellent",
        "GOOD": "good",
        "PARTIAL_ANNOTATIONS": "partial",
        "NO_CITATIONS": "no_citations",
        "TRANSPORT_ERROR": "transport_error_grade",
    }
    for r in results:
        if r.get("transport_error"):
            counts["transport_errors"] += 1
        elif r.get("error"):
            counts["model_errors"] += 1
        elif r.get("content_length", 0) > 0:
            counts["ok"] += 1

        key = grade_keys.get(r.get("heuristic_grade"))
        if key is not None:
            counts[key] += 1

        assertion = r.get("assertion")
        if assertion is not None:
            counts["assertion_total"] += 1
            if assertion.get("pass"):
                counts["assertion_passed"] += 1
    return counts


if __name__ == "__main__":
    main()
