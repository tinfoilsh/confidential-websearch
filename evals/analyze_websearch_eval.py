#!/usr/bin/env python3
"""
Analyze websearch eval results and produce results.json + console summary.

Usage:
    python3 evals/analyze_websearch_eval.py evals/results/<timestamp>
"""

import json
import os
import re
import sys
from collections import defaultdict

CITATION_PATTERN = re.compile(r"\u3010(\d+)\u3011")


def parse_nonstream(raw, endpoint):
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None, None, f"invalid JSON: {raw[:200]}"

    err = data.get("error")
    if isinstance(err, dict) and err.get("message"):
        return None, None, err["message"][:500]
    if isinstance(err, str) and err:
        return None, None, err[:500]

    content = ""
    annotations = []

    if "choices" in data:
        if not data["choices"]:
            return None, None, "empty choices"
        msg = data["choices"][0].get("message", {})
        content = msg.get("content", "") or ""
        for a in msg.get("annotations") or []:
            if a:
                annotations.append(a)
        for sc in msg.get("web_search_calls") or []:
            pass  # captured separately
    elif "output" in data:
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    content += c.get("text", "") or ""
                    for a in c.get("annotations") or []:
                        if a:
                            annotations.append(a)
    else:
        return None, None, f"unknown format: {list(data.keys())[:5]}"

    return content, annotations, None


def parse_stream(raw, endpoint):
    if not raw.strip():
        return None, None, "empty response"

    # Non-SSE error response
    if not raw.strip().startswith("data:") and not raw.strip().startswith("event:"):
        try:
            obj = json.loads(raw.strip())
            err = obj.get("error")
            if isinstance(err, dict):
                return None, None, err.get("message", str(err))[:500]
            if err:
                return None, None, str(err)[:500]
        except (json.JSONDecodeError, ValueError):
            pass

    content = ""
    annotations = []

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

        etype = obj.get("type", "")

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

        # Responses completed -> annotations
        if etype == "response.completed":
            resp = obj.get("response", {})
            for out in resp.get("output", []):
                if out.get("type") == "message":
                    for c in out.get("content", []):
                        for a in c.get("annotations") or []:
                            if a:
                                annotations.append(a)

        # Error in stream
        if "error" in obj and isinstance(obj["error"], dict):
            return None, None, obj["error"].get("message", str(obj["error"]))[:500]

    return content, annotations, None


def analyze_file(results_dir, fname):
    path = os.path.join(results_dir, "raw", fname)
    base, ext = os.path.splitext(fname)
    if ext == ".err" or ext == ".elapsed":
        return None

    parts = base.split("__")
    if len(parts) != 4:
        return None

    model, endpoint, stream_part, task = parts
    stream = "true" in stream_part

    with open(path, "r") as f:
        raw = f.read()

    elapsed_path = os.path.join(results_dir, "raw", f"{base}.elapsed")
    elapsed = 0
    if os.path.exists(elapsed_path):
        try:
            elapsed = int(open(elapsed_path).read().strip())
        except ValueError:
            pass

    if ext == ".json":
        content, annotations, error = parse_nonstream(raw, endpoint)
    elif ext == ".sse":
        content, annotations, error = parse_stream(raw, endpoint)
    else:
        return None

    content = content or ""
    annotations = annotations or []
    markers = CITATION_PATTERN.findall(content)

    # Quality grade
    if error:
        quality = "ERROR"
    elif not content.strip():
        quality = "EMPTY"
    elif markers and annotations:
        unique_markers = len(set(markers))
        if unique_markers == len(annotations):
            quality = "EXCELLENT"
        else:
            quality = "GOOD"
    elif markers:
        quality = "PARTIAL_MARKERS"
    elif annotations:
        quality = "PARTIAL_ANNOTATIONS"
    else:
        quality = "NO_CITATIONS"

    return {
        "model": model,
        "endpoint": endpoint,
        "stream": stream,
        "task": task,
        "elapsed_seconds": elapsed,
        "content_length": len(content),
        "content_snippet": content[:300].replace("\n", " "),
        "citation_markers": len(markers),
        "unique_markers": len(set(markers)),
        "annotations": len(annotations),
        "quality": quality,
        "error": error,
    }


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

    # Save results.json
    output_path = os.path.join(results_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Console summary
    print(f"{'=' * 130}")
    print(f"  WEBSEARCH EVAL RESULTS ({len(results)} tests)")
    print(f"{'=' * 130}")
    print(
        f"  {'Model':<16} {'API':<12} {'Stream':<8} {'Task':<7} "
        f"{'Time':>5} {'Len':>6} {'Mkrs':>5} {'Anns':>5} {'Quality':<20} {'Error'}"
    )
    print(f"  {'-' * 120}")

    for r in results:
        stream_s = "yes" if r["stream"] else "no"
        err_s = (r["error"] or "")[:50]
        print(
            f"  {r['model']:<16} {r['endpoint']:<12} {stream_s:<8} {r['task']:<7} "
            f"{r['elapsed_seconds']:>4}s {r['content_length']:>6} {r['citation_markers']:>5} "
            f"{r['annotations']:>5} {r['quality']:<20} {err_s}"
        )

    # Parity check
    print(f"\n  {'=' * 120}")
    print(f"  STREAMING vs NON-STREAMING PARITY")
    print(f"  {'=' * 120}")
    parity_issues = False
    for model in sorted(set(r["model"] for r in results)):
        for endpoint in ["chat", "responses"]:
            for task in ["search", "fetch"]:
                ns = [
                    r
                    for r in results
                    if r["model"] == model
                    and r["endpoint"] == endpoint
                    and not r["stream"]
                    and r["task"] == task
                ]
                st = [
                    r
                    for r in results
                    if r["model"] == model
                    and r["endpoint"] == endpoint
                    and r["stream"]
                    and r["task"] == task
                ]
                if not ns or not st:
                    continue
                ns, st = ns[0], st[0]
                issues = []
                ns_ok = ns["content_length"] > 0 and not ns["error"]
                st_ok = st["content_length"] > 0 and not st["error"]
                if ns_ok != st_ok:
                    issues.append(
                        f"availability: ns={'OK' if ns_ok else 'FAIL'} stream={'OK' if st_ok else 'FAIL'}"
                    )
                if ns_ok and st_ok:
                    if ns["annotations"] > 0 and st["annotations"] == 0:
                        issues.append(
                            f"annotations lost in stream: ns={ns['annotations']}, stream=0"
                        )
                if issues:
                    parity_issues = True
                    print(f"  {model}|{endpoint}|{task}:")
                    for i in issues:
                        print(f"    - {i}")
    if not parity_issues:
        print("  No parity issues found.")

    # Stats
    print(f"\n  {'=' * 120}")
    print(f"  SUMMARY")
    print(f"  {'=' * 120}")
    total = len(results)
    ok = sum(1 for r in results if r["content_length"] > 0 and not r["error"])
    errors = sum(1 for r in results if r["error"])
    excellent = sum(1 for r in results if r["quality"] == "EXCELLENT")
    good = sum(1 for r in results if r["quality"] == "GOOD")
    partial = sum(
        1
        for r in results
        if r["quality"] in ("PARTIAL_MARKERS", "PARTIAL_ANNOTATIONS")
    )
    no_cit = sum(1 for r in results if r["quality"] == "NO_CITATIONS")

    print(f"  Total: {total}  OK: {ok}  Errors: {errors}")
    print(
        f"  Citations: EXCELLENT={excellent}  GOOD={good}  PARTIAL={partial}  NONE={no_cit}"
    )

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
