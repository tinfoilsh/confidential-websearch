#!/usr/bin/env python3
"""Websearch via router evaluation matrix, Python driver.

Discovers tool-capable models from the router catalog, walks the
(model, endpoint, stream, ctx, task) matrix, POSTs each body with a pooled
httpx client, and captures `.json`/`.sse`/`.err`/`.elapsed`/`.prompt` side
files. Request bodies are built in Python so shell-escaping edge cases do
not apply, and transport failures (timeouts, connection resets, non-2xx)
are recorded distinctly from model-level failures.

When LOCAL_WEBSEARCH_MCP_BASE is set (typically http://127.0.0.1:8091), the
runner also snapshots the /debug/last-call endpoint after each call. That
snapshot becomes the ground truth for the `location` and `domains`
parameter-assertion tasks, replacing the answer-text heuristics that could
never actually fail.

Usage:
    TINFOIL_API_KEY=... python evals/run_websearch_eval.py
    TINFOIL_API_KEY=... python evals/run_websearch_eval.py --base http://localhost:8090
    TINFOIL_API_KEY=... python evals/run_websearch_eval.py --local-fixtures \\
        --mcp-base http://127.0.0.1:8091

Environment overrides mirror the bash runbook:
    TINFOIL_API_KEY         required; bearer token the router accepts.
    USE_LOCAL_FIXTURES=1    swap real-web prompts for local-fixture prompts.
    LOCAL_WEBSEARCH_MCP_BASE base URL of the fixture MCP server's non-/mcp
                            surface (e.g. http://127.0.0.1:8091); when set,
                            the driver captures /debug/last-call snapshots.
    CONTEXT_SIZES           space-separated retrieval-depth buckets
                            (default "low medium high").
    TASKS                   space-separated tasks to run; defaults to the
                            full set (depth-low depth-medium depth-high
                            search fetch location domains).
    SKIP_LLM_JUDGE=1        skip the LLM-as-judge step in the analyzer.
    CATALOG_URL             override the model catalog endpoint
                            (default https://api.tinfoil.sh/v1/models).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_BASE = "http://localhost:8090"
DEFAULT_CATALOG_URL = "https://api.tinfoil.sh/v1/models"
DEFAULT_CONTEXT_SIZES = ("low", "medium", "high")
DEFAULT_TASKS = (
    "depth-low",
    "depth-medium",
    "depth-high",
    "search",
    "fetch",
    "location",
    "domains",
)

CHAT_MAX_TOKENS = 300
RESP_MAX_TOKENS = 300
RESP_STREAM_MAX_TOKENS = 500

REQUEST_TIMEOUT_SECONDS = 300.0


# Prompt catalog -----------------------------------------------------------

_PROMPTS_REAL = {
    "depth-low": (
        "What is the current price of Bitcoin in USD? One sentence. "
        "Cite your source."
    ),
    "depth-medium": (
        "Who won the 2025 Nobel Prize in Physics, and what prior discovery "
        "made their work possible? Cite at least three sources."
    ),
    "depth-high": (
        "Produce a 5-bullet brief on the NIST post-quantum cryptography "
        "finalists, their intended use, and their status. Cite every "
        "claim and follow any links required to verify."
    ),
    "search": (
        "What were the top 3 news headlines today? Cite every claim with "
        "source markers. Be thorough but concise."
    ),
    "fetch": (
        "Fetch https://en.wikipedia.org/wiki/Python_(programming_language) "
        "and give me a 3-sentence summary of what Python is, citing the page."
    ),
    "location": (
        'Search the web for "weather forecast today" and tell me what the '
        "first few results are reporting. Cite every result you mention."
    ),
    "domains": "What is Python? Cite only official sources.",
}

_PROMPTS_FIXTURE = {
    "depth-low": (
        "Search the web and answer in one sentence: According to the "
        "Neighborhood Cat Gazette, which cushions do the cats in the "
        "sunroom prefer? Cite the source."
    ),
    "depth-medium": (
        "Compare what the Neighborhood Cat Gazette and the Cat Almanac "
        "say about breakfast routines. Cite sources."
    ),
    "depth-high": (
        "Produce a 3-bullet brief on the cats in the sunroom using any "
        "relevant fixture page. Cite every claim."
    ),
    "search": (
        "Search the web and answer in one sentence: According to the "
        "Neighborhood Cat Gazette, which cushions do the cats in the "
        "sunroom prefer? Cite the source."
    ),
    "fetch": (
        "Fetch https://local.test/cats/almanac and answer in one sentence: "
        "According to the page, what does Nimbus do after breakfast? Cite "
        "the source."
    ),
    "location": "What is the weather today? Cite sources.",
    "domains": "Where should I learn Python? Cite only the official site.",
}


def prompt_for_task(task: str, use_fixtures: bool) -> str | None:
    table = _PROMPTS_FIXTURE if use_fixtures else _PROMPTS_REAL
    return table.get(task)


# Body builders ------------------------------------------------------------

def _task_web_options(task: str, ctx: str) -> dict[str, Any]:
    opts: dict[str, Any] = {"search_context_size": ctx}
    if task == "location":
        opts["user_location"] = {
            "type": "approximate",
            "approximate": {"country": "GB", "city": "London"},
        }
    if task == "domains":
        opts["filters"] = {"allowed_domains": ["python.org"]}
    return opts


def build_chat_body(model: str, prompt: str, ctx: str, task: str, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "web_search_options": _task_web_options(task, ctx),
        "temperature": 0,
        "max_tokens": CHAT_MAX_TOKENS,
    }


def build_responses_body(model: str, prompt: str, ctx: str, task: str, stream: bool) -> dict[str, Any]:
    tool: dict[str, Any] = {"type": "web_search", "search_context_size": ctx}
    if task == "location":
        tool["user_location"] = {
            "type": "approximate",
            "approximate": {"country": "GB", "city": "London"},
        }
    if task == "domains":
        tool["filters"] = {"allowed_domains": ["python.org"]}
    return {
        "model": model,
        "input": prompt,
        "stream": stream,
        "tools": [tool],
        "temperature": 0,
        "max_output_tokens": RESP_STREAM_MAX_TOKENS if stream else RESP_MAX_TOKENS,
    }


# Runner -------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    model: str
    endpoint: str  # "chat" | "responses"
    stream: bool
    ctx: str
    task: str

    def filename(self) -> str:
        safe_model = self.model.replace("/", "_")
        stream_s = "true" if self.stream else "false"
        return (
            f"{safe_model}__{self.endpoint}__stream-{stream_s}__ctx-{self.ctx}__{self.task}"
        )


def fetch_catalog(client: httpx.Client, catalog_url: str) -> tuple[list[str], list[str]]:
    resp = client.get(catalog_url, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    chat_models: list[str] = []
    resp_models: list[str] = []
    for m in data.get("data", []):
        if m.get("type") != "chat":
            continue
        if not m.get("tool_calling"):
            continue
        endpoints = m.get("endpoints", []) or []
        if "/v1/chat/completions" in endpoints:
            chat_models.append(m["id"])
        if "/v1/responses" in endpoints:
            resp_models.append(m["id"])
    return chat_models, resp_models


def enumerate_cases(
    chat_models: list[str],
    resp_models: list[str],
    context_sizes: list[str],
    tasks: list[str],
) -> list[Case]:
    assertion_tasks = {"location", "domains"}
    cases: list[Case] = []
    for task in tasks:
        # Parameter-assertion tasks only need one context bucket; the goal is
        # to verify flag passthrough, not retrieval depth behavior.
        ctxs = ["medium"] if task in assertion_tasks else context_sizes
        for ctx in ctxs:
            for endpoint, models in (("chat", chat_models), ("responses", resp_models)):
                for stream in (False, True):
                    for model in models:
                        cases.append(Case(model, endpoint, stream, ctx, task))
    return cases


def _snapshot_last_call(client: httpx.Client, mcp_base: str | None) -> dict[str, Any] | None:
    if not mcp_base:
        return None
    url = mcp_base.rstrip("/") + "/debug/last-call"
    try:
        resp = client.get(url, timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, json.JSONDecodeError, ValueError):
        return None


def _snapshot_belongs_to_request(
    snapshot: dict[str, Any] | None, request_started_at: dt.datetime
) -> bool:
    """Return True only if the snapshot was observed after the request began.

    The fixture MCP server holds a single-slot recorder that is never cleared
    between requests, so reading `/debug/last-call` always returns something
    once any tool call has happened in the process. We treat a snapshot as
    belonging to the current request only when the server observed the call
    at or after the request's monotonic start time; any earlier snapshot is
    leftover state from a prior case and must not be persisted.
    """
    if not isinstance(snapshot, dict):
        return False
    search_block = snapshot.get("search") or {}
    fetch_block = snapshot.get("fetch") or {}
    for observed_at in (search_block.get("observed_at"), fetch_block.get("observed_at")):
        if not isinstance(observed_at, str) or not observed_at.strip():
            continue
        if observed_at.startswith("0001-01-01"):
            continue
        parsed = _parse_rfc3339(observed_at)
        if parsed is None:
            continue
        if parsed >= request_started_at:
            return True
    return False


_RFC3339_FRACTIONAL_RE = re.compile(r"\.(\d+)")


def _parse_rfc3339(raw: str) -> dt.datetime | None:
    """Parse an RFC 3339 timestamp, including the Go `time.RFC3339Nano` shape.

    Python 3.9's `datetime.fromisoformat` rejects trailing `Z` and fractional
    seconds with a digit count other than 3 or 6. The Go server emits
    `time.RFC3339Nano`, which can produce anything from no fractional seconds
    up to 9 digits, so we normalize both dimensions before parsing.
    """
    value = raw.strip().replace("Z", "+00:00")
    match = _RFC3339_FRACTIONAL_RE.search(value)
    if match:
        digits = match.group(1)
        if len(digits) > 6:
            digits = digits[:6]
        else:
            digits = digits.ljust(6, "0")
        value = value[: match.start()] + "." + digits + value[match.end():]
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def _url_for(endpoint: str, base: str) -> str:
    if endpoint == "chat":
        return base.rstrip("/") + "/v1/chat/completions"
    return base.rstrip("/") + "/v1/responses"


def run_case(
    client: httpx.Client,
    base: str,
    api_key: str,
    case: Case,
    prompt: str,
    out_dir: pathlib.Path,
    mcp_base: str | None,
) -> None:
    body: dict[str, Any]
    if case.endpoint == "chat":
        body = build_chat_body(case.model, prompt, case.ctx, case.task, case.stream)
    else:
        body = build_responses_body(case.model, prompt, case.ctx, case.task, case.stream)

    fname = case.filename()
    ext = "sse" if case.stream else "json"
    out_path = out_dir / f"{fname}.{ext}"
    err_path = out_dir / f"{fname}.err"
    elapsed_path = out_dir / f"{fname}.elapsed"
    prompt_path = out_dir / f"{fname}.prompt"
    last_call_path = out_dir / f"{fname}.last_call.json"

    prompt_path.write_text(prompt)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    request_started_at = dt.datetime.now(dt.timezone.utc)
    start = time.monotonic()
    transport_error: str | None = None
    response_text = ""
    try:
        resp = client.post(
            _url_for(case.endpoint, base),
            headers=headers,
            json=body,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response_text = resp.text
        if resp.status_code >= 400:
            transport_error = f"HTTP {resp.status_code}"
    except httpx.TimeoutException:
        transport_error = "timeout"
    except httpx.HTTPError as exc:
        transport_error = f"transport: {exc.__class__.__name__}: {exc}"
    elapsed = time.monotonic() - start

    out_path.write_text(response_text)
    elapsed_path.write_text(str(int(round(elapsed))))
    if transport_error:
        err_path.write_text(transport_error)
    elif err_path.exists():
        err_path.unlink()

    # Only persist a snapshot for this case when the HTTP call itself
    # succeeded AND the server's recorded observation is newer than our
    # request start. The fixture MCP's /debug/last-call is a single mutable
    # slot; without both checks a failed or short-circuited request would
    # inherit a stale snapshot from a prior case.
    if transport_error is None:
        snapshot = _snapshot_last_call(client, mcp_base)
        if snapshot is not None and _snapshot_belongs_to_request(
            snapshot, request_started_at
        ):
            last_call_path.write_text(json.dumps(snapshot, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=os.environ.get("WEBSEARCH_BASE", DEFAULT_BASE))
    parser.add_argument(
        "--catalog-url",
        default=os.environ.get("CATALOG_URL", DEFAULT_CATALOG_URL),
    )
    parser.add_argument(
        "--local-fixtures",
        action="store_true",
        default=os.environ.get("USE_LOCAL_FIXTURES") == "1",
    )
    parser.add_argument(
        "--mcp-base",
        default=os.environ.get("LOCAL_WEBSEARCH_MCP_BASE"),
        help="Base URL of the fixture MCP server (enables /debug/last-call capture).",
    )
    parser.add_argument(
        "--context-sizes",
        default=os.environ.get("CONTEXT_SIZES", " ".join(DEFAULT_CONTEXT_SIZES)),
        help='space-separated retrieval-depth buckets (default "low medium high")',
    )
    parser.add_argument(
        "--tasks",
        default=os.environ.get("TASKS", " ".join(DEFAULT_TASKS)),
        help="space-separated task list",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        default=os.environ.get("SKIP_LLM_JUDGE") == "1",
    )
    args = parser.parse_args()

    api_key = os.environ.get("TINFOIL_API_KEY", "")
    if not api_key:
        print("ERROR: TINFOIL_API_KEY must be set", file=sys.stderr)
        return 2

    context_sizes = [c for c in args.context_sizes.split() if c]
    tasks = [t for t in args.tasks.split() if t]

    script_dir = pathlib.Path(__file__).resolve().parent
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = script_dir / "results" / timestamp / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Websearch eval (router): {args.base}")
    print(f"Output: {out_dir.parent}")
    print(f"Context sizes: {context_sizes}")
    print(f"Tasks: {tasks}")
    if args.mcp_base:
        print(f"MCP /debug/last-call capture: {args.mcp_base}")
    print()

    with httpx.Client(
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
    ) as client:
        try:
            chat_models, resp_models = fetch_catalog(client, args.catalog_url)
        except httpx.HTTPError as exc:
            print(f"ERROR: could not fetch model catalog from {args.catalog_url}: {exc}")
            return 1
        if not chat_models and not resp_models:
            print("ERROR: no supported models found in catalog")
            return 1

        print(f"Chat completions models: {', '.join(chat_models) or '(none)'}")
        print(f"Responses models: {', '.join(resp_models) or '(none)'}")
        print()

        cases = enumerate_cases(chat_models, resp_models, context_sizes, tasks)
        for case in cases:
            prompt = prompt_for_task(case.task, args.local_fixtures)
            if prompt is None:
                print(f"  (no prompt for task {case.task}; skipping)")
                continue
            print(f"  {case.filename()}")
            run_case(client, args.base, api_key, case, prompt, out_dir, args.mcp_base)

    print()
    print("Raw results saved. Running analysis...")
    print()

    analyzer = script_dir / "analyze_websearch_eval.py"
    env = os.environ.copy()
    env["SKIP_LLM_JUDGE"] = "1" if args.skip_llm_judge else env.get("SKIP_LLM_JUDGE", "0")
    env["TINFOIL_API_KEY"] = api_key
    rc = subprocess.run(
        [sys.executable, str(analyzer), str(out_dir.parent)],
        env=env,
        check=False,
    ).returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
