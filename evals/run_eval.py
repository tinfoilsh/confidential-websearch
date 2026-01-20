#!/usr/bin/env python3
"""
Evaluation runner for safety filter prompts.

Usage:
    # Run all evaluations
    python run_eval.py

    # Run specific evaluation
    python run_eval.py --eval prompt-injection
    python run_eval.py --eval pii

    # Limit samples for quick testing
    python run_eval.py --max-samples 50

    # Save detailed results
    python run_eval.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import random

from dotenv import load_dotenv
from tqdm import tqdm

from client import ArbiterClient, CheckResult, SafeguardClient
from data_loaders import load_pii_dataset, load_pint_dataset
from data_loaders.pint import load_deepset_injection_dataset
from metrics import EvalMetrics, calculate_metrics
from prompts import PII_LEAKAGE_POLICY, PROMPT_INJECTION_POLICY

load_dotenv()


def run_prompt_injection_eval(
    client: SafeguardClient,
    max_samples: int | None = None,
    use_deepset: bool = False,
    seed: int = 0,
    mimic_production_wrapper: bool = True,
    arbiter: ArbiterClient | None = None,
) -> tuple[EvalMetrics, list[dict]]:
    """
    Evaluate the prompt injection detection policy.

    Args:
        client: Safeguard client instance
        max_samples: Maximum samples to evaluate
        use_deepset: Use deepset dataset instead of PINT
        seed: Random seed for shuffling
        mimic_production_wrapper: Wrap content in Title/URL/Content format
        arbiter: Optional arbiter client to resolve disagreements

    Returns:
        Tuple of (metrics, detailed_results)
    """
    print("\n" + "=" * 60)
    print("PROMPT INJECTION DETECTION EVALUATION")
    print("=" * 60)

    if use_deepset:
        print("Loading deepset/prompt-injections dataset...")
        samples = load_deepset_injection_dataset(max_samples=max_samples)
    else:
        print("Loading PINT benchmark dataset...")
        samples = load_pint_dataset(max_samples=max_samples)

    print(f"Loaded {len(samples)} samples")
    if arbiter:
        print("Arbiter enabled: will resolve classifier vs dataset disagreements")

    if seed is not None:
        random.Random(seed).shuffle(samples)

    y_true = []
    y_pred = []
    detailed_results = []
    error_count = 0
    arbiter_calls = 0
    arbiter_overrides = 0

    for sample in tqdm(samples, desc="Evaluating"):
        try:
            content = sample.text
            if mimic_production_wrapper:
                # Match production formatting in `agent/safe_agent.go`:
                # fmt.Sprintf("Title: %s\nURL: %s\nContent: %s", ...)
                content = f"Title: (eval:{sample.source})\nURL: https://example.invalid/\nContent: {sample.text[:500]}"

            result = client.check(PROMPT_INJECTION_POLICY, content)

            expected = sample.is_injection
            predicted = result.violation
            arbiter_decision = None

            # If classifier says safe but dataset says violation, ask arbiter
            if arbiter and expected and not predicted:
                arbiter_calls += 1
                try:
                    classifier_correct, explanation = arbiter.arbitrate(
                        sample.text, result.rationale, eval_type="injection"
                    )
                    arbiter_decision = {
                        "classifier_correct": classifier_correct,
                        "explanation": explanation,
                    }
                    if classifier_correct:
                        # Arbiter agrees with classifier - override the expected label
                        expected = False
                        arbiter_overrides += 1
                except Exception as e:
                    arbiter_decision = {"error": str(e)}

            y_true.append(expected)
            y_pred.append(predicted)

            detailed_results.append({
                "text": sample.text[:200] + "..." if len(sample.text) > 200 else sample.text,
                "dataset_label": sample.is_injection,
                "expected": expected,
                "predicted": predicted,
                "correct": expected == predicted,
                "rationale": result.rationale,
                "source": sample.source,
                "arbiter": arbiter_decision,
            })

        except Exception as e:
            print(f"\nError evaluating sample: {e}")
            error_count += 1
            detailed_results.append({
                "text": sample.text[:200],
                "expected": sample.is_injection,
                "predicted": None,
                "correct": False,
                "error": str(e),
            })

    metrics = calculate_metrics(y_true, y_pred)
    print(metrics)
    if arbiter:
        print(f"\nArbiter stats: {arbiter_calls} calls, {arbiter_overrides} overrides (dataset label was wrong)")
    if error_count:
        print(f"Note: {error_count} samples failed to evaluate and were excluded from metrics.")

    return metrics, detailed_results


def run_pii_eval(
    client: SafeguardClient,
    max_samples: int | None = None,
    seed: int = 0,
    arbiter: ArbiterClient | None = None,
) -> tuple[EvalMetrics, list[dict]]:
    """
    Evaluate the PII leakage detection policy.

    Args:
        client: Safeguard client instance
        max_samples: Maximum samples to evaluate
        arbiter: Optional arbiter client to resolve disagreements

    Returns:
        Tuple of (metrics, detailed_results)
    """
    print("\n" + "=" * 60)
    print("PII LEAKAGE DETECTION EVALUATION")
    print("=" * 60)

    print("Loading PII dataset...")
    samples = load_pii_dataset(max_samples=max_samples)
    print(f"Loaded {len(samples)} samples")
    if arbiter:
        print("Arbiter enabled: will resolve classifier vs dataset disagreements")

    if seed is not None:
        random.Random(seed).shuffle(samples)

    y_true = []
    y_pred = []
    detailed_results = []
    error_count = 0
    arbiter_calls = 0
    arbiter_overrides = 0

    for sample in tqdm(samples, desc="Evaluating"):
        try:
            result = client.check(PII_LEAKAGE_POLICY, sample.text)

            expected = sample.has_sensitive_pii
            predicted = result.violation
            arbiter_decision = None

            # If classifier says safe but dataset says violation, ask arbiter
            if arbiter and expected and not predicted:
                arbiter_calls += 1
                try:
                    classifier_correct, explanation = arbiter.arbitrate(
                        sample.text, result.rationale
                    )
                    arbiter_decision = {
                        "classifier_correct": classifier_correct,
                        "explanation": explanation,
                    }
                    if classifier_correct:
                        # Arbiter agrees with classifier - override the expected label
                        expected = False
                        arbiter_overrides += 1
                except Exception as e:
                    arbiter_decision = {"error": str(e)}

            y_true.append(expected)
            y_pred.append(predicted)

            detailed_results.append({
                "text": sample.text[:200] + "..." if len(sample.text) > 200 else sample.text,
                "dataset_label": sample.has_sensitive_pii,
                "expected": expected,
                "expected_type": sample.pii_type,
                "predicted": predicted,
                "correct": expected == predicted,
                "rationale": result.rationale,
                "source": sample.source,
                "arbiter": arbiter_decision,
            })

        except Exception as e:
            print(f"\nError evaluating sample: {e}")
            error_count += 1
            detailed_results.append({
                "text": sample.text[:200],
                "expected": sample.has_sensitive_pii,
                "predicted": None,
                "correct": False,
                "error": str(e),
            })

    metrics = calculate_metrics(y_true, y_pred)
    print(metrics)
    if arbiter:
        print(f"\nArbiter stats: {arbiter_calls} calls, {arbiter_overrides} overrides (dataset label was wrong)")
    if error_count:
        print(f"Note: {error_count} samples failed to evaluate and were excluded from metrics.")

    return metrics, detailed_results


def print_error_analysis(results: list[dict], eval_type: str, limit: int = 10) -> None:
    """Print analysis of incorrect predictions."""
    errors = [r for r in results if not r.get("correct", True)]

    if not errors:
        print(f"\nNo errors in {eval_type} evaluation!")
        return

    print(f"\n{'=' * 60}")
    print(f"ERROR ANALYSIS - {eval_type.upper()} ({len(errors)} errors)")
    print("=" * 60)

    false_positives = [r for r in errors if not r["expected"] and r.get("predicted")]
    false_negatives = [r for r in errors if r["expected"] and not r.get("predicted")]

    if false_positives:
        print(f"\nFalse Positives ({len(false_positives)} - safe content flagged as violation):")
        print("-" * 40)
        for r in false_positives[:limit]:
            print(f"  Text: {r['text'][:100]}...")
            print(f"  Category: {r.get('category', 'N/A')}")
            print(f"  Rationale: {r.get('rationale', 'N/A')}")
            print()

    if false_negatives:
        print(f"\nFalse Negatives ({len(false_negatives)} - violations missed):")
        print("-" * 40)
        for r in false_negatives[:limit]:
            print(f"  Text: {r['text'][:100]}...")
            print(f"  Expected type: {r.get('expected_type', 'N/A')}")
            print(f"  Category: {r.get('category', 'N/A')}")
            print(f"  Rationale: {r.get('rationale', 'N/A')}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Run safety filter evaluations")
    parser.add_argument(
        "--eval",
        choices=["prompt-injection", "pii", "all"],
        default="all",
        help="Which evaluation to run",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for quick testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for detailed results (JSON)",
    )
    parser.add_argument(
        "--use-deepset",
        action="store_true",
        help="Use deepset dataset instead of PINT for prompt injection",
    )
    parser.add_argument(
        "--error-analysis",
        action="store_true",
        help="Print detailed error analysis",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed for deterministic sample ordering (set to -1 to disable shuffling)",
    )
    parser.add_argument(
        "--no-prod-wrapper",
        action="store_true",
        help="Do NOT mimic production Title/URL/Content wrapper for prompt-injection eval",
    )
    parser.add_argument(
        "--no-arbiter",
        action="store_true",
        help="Disable kimi-k2-thinking arbiter (arbiter is ON by default for both evals)",
    )
    args = parser.parse_args()

    try:
        client = SafeguardClient()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nSet the SAFEGUARD_API_KEY environment variable or create a .env file:")
        print('  export SAFEGUARD_API_KEY="your-api-key"')
        sys.exit(1)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "evaluations": {},
    }

    arbiter = None if args.no_arbiter else ArbiterClient()

    if args.eval in ("prompt-injection", "all"):
        metrics, results = run_prompt_injection_eval(
            client,
            max_samples=args.max_samples,
            use_deepset=args.use_deepset,
            seed=None if args.seed == -1 else args.seed,
            mimic_production_wrapper=not args.no_prod_wrapper,
            arbiter=arbiter,
        )
        all_results["evaluations"]["prompt_injection"] = {
            "metrics": asdict(metrics),
            "results": results,
        }
        if args.error_analysis:
            print_error_analysis(results, "prompt-injection")

    if args.eval in ("pii", "all"):
        metrics, results = run_pii_eval(
            client,
            max_samples=args.max_samples,
            seed=None if args.seed == -1 else args.seed,
            arbiter=arbiter,
        )
        all_results["evaluations"]["pii"] = {
            "metrics": asdict(metrics),
            "results": results,
        }
        if args.error_analysis:
            print_error_analysis(results, "pii")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
