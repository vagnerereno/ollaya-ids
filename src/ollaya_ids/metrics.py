from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .constants import NORMAL_LABEL


@dataclass(frozen=True)
class RuntimeStats:
    samples: int
    elapsed_seconds: float

    @property
    def latency_ms_per_sample(self) -> float:
        return 1000.0 * self.elapsed_seconds / max(self.samples, 1)

    @property
    def throughput_samples_per_second(self) -> float:
        return self.samples / max(self.elapsed_seconds, 1e-12)


def multiclass_brier(y_true: Sequence[str], probabilities: np.ndarray, labels: list[str]) -> float:
    label_to_index = {label: index for index, label in enumerate(labels)}
    target = np.zeros_like(probabilities, dtype=float)
    for row, label in enumerate(y_true):
        target[row, label_to_index[str(label)]] = 1.0
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def categorical_log_loss(
    y_true: Sequence[str], probabilities: np.ndarray, labels: list[str]
) -> float:
    label_to_index = {label: index for index, label in enumerate(labels)}
    true_probability = np.asarray(
        [probabilities[row, label_to_index[str(label)]] for row, label in enumerate(y_true)]
    )
    return float(-np.log(np.clip(true_probability, 1e-15, 1.0)).mean())


def expected_calibration_error(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: np.ndarray,
    bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    correct = np.asarray(y_true) == np.asarray(y_pred)
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for lower, upper in pairwise(edges):
        if upper == 1.0:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence >= lower) & (confidence < upper)
        if not selected.any():
            continue
        score += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(score)


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: np.ndarray,
    labels: list[str],
    runtime: RuntimeStats,
) -> dict[str, object]:
    y_true_array = np.asarray(y_true, dtype=str)
    y_pred_array = np.asarray(y_pred, dtype=str)
    probability_array = np.asarray(probabilities, dtype=float)
    if probability_array.shape != (len(y_true_array), len(labels)):
        raise ValueError("Probability matrix does not match the sample and label dimensions")

    normal_index = labels.index(NORMAL_LABEL)
    attack_true = (y_true_array != NORMAL_LABEL).astype(int)
    attack_pred = (y_pred_array != NORMAL_LABEL).astype(int)
    attack_probability = 1.0 - probability_array[:, normal_index]

    true_normal = attack_true == 0
    false_positive_rate = (
        float(np.mean(attack_pred[true_normal] == 1)) if true_normal.any() else None
    )
    true_attack = attack_true == 1
    false_negative_rate = (
        float(np.mean(attack_pred[true_attack] == 0)) if true_attack.any() else None
    )

    report = classification_report(
        y_true_array,
        y_pred_array,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "samples": len(y_true_array),
        "labels": labels,
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
        "macro_precision": float(
            precision_score(
                y_true_array,
                y_pred_array,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true_array,
                y_pred_array,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true_array,
                y_pred_array,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true_array,
                y_pred_array,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "log_loss": categorical_log_loss(y_true_array, probability_array, labels),
        "brier_multiclass": multiclass_brier(y_true_array, probability_array, labels),
        "expected_calibration_error": expected_calibration_error(
            y_true_array, y_pred_array, probability_array
        ),
        "binary_intrusion": {
            "precision": float(precision_score(attack_true, attack_pred, zero_division=0)),
            "recall": float(recall_score(attack_true, attack_pred, zero_division=0)),
            "f1": float(f1_score(attack_true, attack_pred, zero_division=0)),
            "roc_auc": (
                float(roc_auc_score(attack_true, attack_probability))
                if len(np.unique(attack_true)) == 2
                else None
            ),
            "pr_auc": (
                float(average_precision_score(attack_true, attack_probability))
                if len(np.unique(attack_true)) == 2
                else None
            ),
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        },
        "runtime": {
            "elapsed_seconds": runtime.elapsed_seconds,
            "latency_ms_per_sample": runtime.latency_ms_per_sample,
            "throughput_samples_per_second": runtime.throughput_samples_per_second,
        },
        "per_class": {label: report[label] for label in labels},
        "confusion_matrix": confusion_matrix(y_true_array, y_pred_array, labels=labels).tolist(),
    }
