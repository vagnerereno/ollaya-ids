from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class JsonEncoder(json.JSONEncoder):
    def default(self, value: Any) -> Any:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return super().default(value)


def environment_info(packages: list[str]) -> dict[str, object]:
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, cls=JsonEncoder)
        handle.write("\n")


def write_prediction_table(
    path: str | Path,
    frame: pd.DataFrame,
    y_pred: list[str] | np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    label_column: str,
) -> None:
    output = frame[["row_id", label_column]].copy()
    output.rename(columns={label_column: "true_label"}, inplace=True)
    output["predicted_label"] = y_pred
    for index, label in enumerate(labels):
        output[f"probability_{label}"] = probabilities[:, index]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)


def write_metric_tables(output_dir: str | Path, metrics: dict[str, Any]) -> None:
    destination = Path(output_dir)
    write_json(destination / "metrics.json", metrics)
    pd.DataFrame(
        metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"]
    ).to_csv(destination / "confusion_matrix.csv")
    per_class = pd.DataFrame.from_dict(metrics["per_class"], orient="index")
    per_class.index.name = "class"
    per_class.to_csv(destination / "per_class.csv")
