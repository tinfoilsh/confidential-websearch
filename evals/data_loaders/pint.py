"""Prompt injection dataset loaders.

Supports multiple sources:
- Deepset prompt-injections: https://huggingface.co/datasets/deepset/prompt-injections
- Common Crawl (via C4): benign web content for negative samples
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class PromptInjectionSample:
    """A single prompt injection test sample."""

    text: str
    is_injection: bool
    source: str


def load_pint_dataset(
    max_samples: int | None = None,
    include_common_crawl: bool = True,
) -> list[PromptInjectionSample]:
    """
    Load prompt injection dataset combining deepset injections and Common Crawl negatives.

    Args:
        max_samples: Maximum number of samples to load (None for all)
        include_common_crawl: Whether to include Common Crawl benign samples

    Returns:
        List of PromptInjectionSample objects
    """
    samples = load_deepset_injection_dataset(max_samples=max_samples)

    if include_common_crawl:
        injection_count = sum(1 for s in samples if s.is_injection)
        if max_samples:
            remaining_quota = max_samples - len(samples)
            crawl_count = min(injection_count, remaining_quota)
        else:
            crawl_count = injection_count
        if crawl_count > 0:
            crawl_samples = load_common_crawl_negatives(max_samples=crawl_count)
            samples.extend(crawl_samples)

    return samples


def load_deepset_injection_dataset(
    max_samples: int | None = None,
) -> list[PromptInjectionSample]:
    """
    Load the deepset prompt injection dataset from HuggingFace.

    Contains 662 samples (546 train + 116 test) with balanced
    injection/benign examples.

    Source: https://huggingface.co/datasets/deepset/prompt-injections

    Args:
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of PromptInjectionSample objects
    """
    train = load_dataset("deepset/prompt-injections", split="train")
    test = load_dataset("deepset/prompt-injections", split="test")

    samples = []
    for row in train:
        if max_samples and len(samples) >= max_samples:
            break
        samples.append(
            PromptInjectionSample(
                text=row["text"],
                is_injection=row["label"] == 1,
                source="deepset",
            )
        )

    for row in test:
        if max_samples and len(samples) >= max_samples:
            break
        samples.append(
            PromptInjectionSample(
                text=row["text"],
                is_injection=row["label"] == 1,
                source="deepset",
            )
        )

    return samples


def load_common_crawl_negatives(
    max_samples: int = 500,
) -> list[PromptInjectionSample]:
    """
    Load benign web content from Common Crawl (via C4 dataset) as negative samples.

    C4 (Colossal Clean Crawled Corpus) is derived from Common Crawl and provides
    clean, representative web content for testing false positive rates.

    Source: https://huggingface.co/datasets/allenai/c4

    Args:
        max_samples: Maximum number of samples to load

    Returns:
        List of PromptInjectionSample objects (all marked as non-injection)
    """
    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True,
    )

    samples = []
    for row in dataset:
        if len(samples) >= max_samples:
            break

        text = row["text"]
        if not text:
            continue

        samples.append(
            PromptInjectionSample(
                text=text,
                is_injection=False,
                source="common_crawl",
            )
        )

    return samples
