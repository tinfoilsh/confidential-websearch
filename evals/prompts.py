"""Safety filter policies loaded from the single source of truth.

The Go server reads safeguard/policies.yml at startup via //go:embed. This
module loads the same file so every harness grades the classifier against
the policy text the server is actually running, eliminating drift between
the Python eval prompts and the production prompts.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_POLICIES_PATH = Path(__file__).resolve().parent.parent / "safeguard" / "policies.yml"


def _load_policies() -> dict[str, str]:
    with _POLICIES_PATH.open("r") as fh:
        data = yaml.safe_load(fh) or {}
    policies = data.get("policies") or {}
    if not isinstance(policies, dict):
        raise RuntimeError(
            f"{_POLICIES_PATH}: 'policies' must be a mapping of name -> text"
        )
    missing = [k for k in ("pii_leakage", "prompt_injection") if not policies.get(k)]
    if missing:
        raise RuntimeError(f"{_POLICIES_PATH}: missing policies {missing}")
    return policies


_POLICIES = _load_policies()

PII_LEAKAGE_POLICY: str = _POLICIES["pii_leakage"]
PROMPT_INJECTION_POLICY: str = _POLICIES["prompt_injection"]
