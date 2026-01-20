"""PII detection dataset loader.

Loads datasets for evaluating PII leakage detection in search queries.
Uses ai4privacy for positive samples (text containing sensitive PII)
and MS MARCO for negative samples (real search queries without PII).
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class PIISample:
    """A single PII detection test sample."""

    text: str
    has_sensitive_pii: bool
    pii_type: str
    source: str


NAME_ONLY_LABELS = {"GIVENNAME", "SURNAME"}


def load_pii_dataset(
    max_samples: int | None = None,
    include_negatives: bool = True,
) -> list[PIISample]:
    """
    Load PII detection samples combining ai4privacy positives and MS MARCO negatives.

    Args:
        max_samples: Maximum number of positive samples to load (None for all)
        include_negatives: Whether to include MS MARCO negative samples

    Returns:
        List of PIISample objects
    """
    samples = _load_ai4privacy_samples(max_samples=max_samples or 500)

    if include_negatives:
        positive_count = len(samples)
        negative_samples = _load_msmarco_negatives(max_samples=positive_count)
        samples.extend(negative_samples)

    return samples


def _load_ai4privacy_samples(max_samples: int = 500) -> list[PIISample]:
    """
    Load samples from the ai4privacy dataset.

    Uses the dataset's native `privacy_mask` annotations.
    Labels based on policy:
    - Name only (GIVENNAME/SURNAME) → not a violation
    - Any other PII (or name + other PII) → violation
    """
    dataset = load_dataset("ai4privacy/pii-masking-400k", split="train")

    samples = []
    for row in dataset:
        if len(samples) >= max_samples:
            break

        text = row.get("source_text", "")
        privacy_mask = row.get("privacy_mask", [])

        if not text or not privacy_mask:
            continue

        labels = {entity["label"] for entity in privacy_mask}
        has_only_name = labels.issubset(NAME_ONLY_LABELS)
        pii_type = next(iter(labels - NAME_ONLY_LABELS), next(iter(labels)))

        samples.append(
            PIISample(
                text=text,
                has_sensitive_pii=not has_only_name,
                pii_type=pii_type,
                source="ai4privacy",
            )
        )

    return samples


def _load_msmarco_negatives(max_samples: int = 500) -> list[PIISample]:
    """
    Load negative samples from MS MARCO (real search queries).

    MS MARCO contains real Bing search queries which represent
    realistic negative examples for PII detection.

    Source: https://huggingface.co/datasets/ms_marco
    """
    dataset = load_dataset(
        "ms_marco",
        "v1.1",
        split="train",
        streaming=True,
    )

    samples = []
    for row in dataset:
        if len(samples) >= max_samples:
            break

        query = row.get("query", "")
        if not query:
            continue

        samples.append(
            PIISample(
                text=query,
                has_sensitive_pii=False,
                pii_type="none",
                source="msmarco",
            )
        )

    return samples
