import numpy as np

from ollaya_ids.metrics import RuntimeStats, evaluate_predictions


def test_perfect_predictions_have_perfect_scores() -> None:
    labels = ["normal", "attack"]
    y_true = ["normal", "attack", "attack"]
    probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8]])

    metrics = evaluate_predictions(
        y_true,
        y_true,
        probabilities,
        labels,
        RuntimeStats(samples=3, elapsed_seconds=0.3),
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["binary_intrusion"]["false_positive_rate"] == 0.0
    assert metrics["runtime"]["latency_ms_per_sample"] == 100.0
