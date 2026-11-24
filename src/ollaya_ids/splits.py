from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(
    frame: pd.DataFrame,
    label_column: str,
    strategy: str,
    train_size: float,
    validation_size: float,
    seed: int,
) -> pd.DataFrame:
    test_size = 1.0 - train_size - validation_size
    if min(train_size, validation_size, test_size) <= 0:
        raise ValueError("Train, validation and test sizes must all be positive")

    if strategy == "stratified":
        train_index, remainder_index = train_test_split(
            frame.index,
            train_size=train_size,
            random_state=seed,
            stratify=frame[label_column],
        )
        validation_fraction = validation_size / (validation_size + test_size)
        validation_index, test_index = train_test_split(
            remainder_index,
            train_size=validation_fraction,
            random_state=seed,
            stratify=frame.loc[remainder_index, label_column],
        )
    elif strategy == "temporal":
        if "Time" not in frame:
            raise ValueError("Temporal splitting requires the Time column")
        ordered = frame.sort_values(["Time", "row_id"]).index
        train_end = int(len(ordered) * train_size)
        validation_end = train_end + int(len(ordered) * validation_size)
        train_index = ordered[:train_end]
        validation_index = ordered[train_end:validation_end]
        test_index = ordered[validation_end:]
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    assignments = pd.Series(index=frame.index, dtype="string")
    assignments.loc[train_index] = "train"
    assignments.loc[validation_index] = "validation"
    assignments.loc[test_index] = "test"

    manifest = frame[["row_id", label_column]].copy()
    manifest.insert(1, "split", assignments)
    return manifest.sort_values("row_id").reset_index(drop=True)


def save_manifest(manifest: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype={"row_id": "string", "split": "string"})
    required = {"row_id", "split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Split manifest must contain: {', '.join(sorted(required))}")
    return manifest


def apply_manifest(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    joined = frame.merge(
        manifest[["row_id", "split"]], on="row_id", how="inner", validate="one_to_one"
    )
    if len(joined) != len(manifest):
        raise ValueError("Split manifest does not match the loaded dataset")
    return joined
