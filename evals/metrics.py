"""Evaluation metrics for safety filter assessment."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EvalMetrics:
    """Evaluation metrics for a safety filter."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    total_samples: int

    def __str__(self) -> str:
        return f"""
Evaluation Results ({self.total_samples} samples)
{'=' * 50}
Accuracy:   {self.accuracy:.3f}
Precision:  {self.precision:.3f}
Recall:     {self.recall:.3f}
F1 Score:   {self.f1:.3f}
F2 Score:   {self.f2:.3f}

Confusion Matrix:
                 Predicted
              Safe    Violation
Actual Safe    {self.true_negatives:<6}  {self.false_positives:<6}  (FPR: {self.false_positive_rate:.3f})
     Violation {self.false_negatives:<6}  {self.true_positives:<6}  (FNR: {self.false_negative_rate:.3f})
"""

    @property
    def false_positive_rate(self) -> float:
        """Rate of safe content incorrectly flagged as violations."""
        total_negatives = self.true_negatives + self.false_positives
        if total_negatives == 0:
            return 0.0
        return self.false_positives / total_negatives

    @property
    def false_negative_rate(self) -> float:
        """Rate of violations incorrectly classified as safe."""
        total_positives = self.true_positives + self.false_negatives
        if total_positives == 0:
            return 0.0
        return self.false_negatives / total_positives


def calculate_metrics(
    y_true: list[bool],
    y_pred: list[bool],
) -> EvalMetrics:
    """
    Calculate evaluation metrics from predictions.

    Args:
        y_true: Ground truth labels (True = violation, False = safe)
        y_pred: Predicted labels (True = violation, False = safe)

    Returns:
        EvalMetrics object with all calculated metrics
    """
    y_true_int = [1 if y else 0 for y in y_true]
    y_pred_int = [1 if y else 0 for y in y_pred]

    accuracy = accuracy_score(y_true_int, y_pred_int)
    precision = precision_score(y_true_int, y_pred_int, zero_division=0)
    recall = recall_score(y_true_int, y_pred_int, zero_division=0)
    f1 = f1_score(y_true_int, y_pred_int, zero_division=0)
    f2 = fbeta_score(y_true_int, y_pred_int, beta=2)

    cm = confusion_matrix(y_true_int, y_pred_int, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return EvalMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        f2=f2,
        true_positives=int(tp),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        total_samples=len(y_true),
    )


def fbeta_score(y_true: list[int], y_pred: list[int], beta: float = 2) -> float:
    """
    Calculate F-beta score.

    F2 score weights recall higher than precision, which is important
    for safety filters where missing a violation (false negative) is
    worse than a false alarm (false positive).
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    if precision + recall == 0:
        return 0.0

    beta_squared = beta**2
    return (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
