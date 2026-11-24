from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .constants import DECISION_FEATURES, LABEL_COLUMN

DatasetSection = Literal["csv", "arff", "all"]


@dataclass(frozen=True)
class DatasetProfile:
    source: str
    selected_section: str
    rows_read: int
    rows_after_cleaning: int
    columns: int
    class_counts: dict[str, int]
    duplicate_rows_removed: int
    conflicting_vectors_removed: int
    malformed_lines_skipped: int
    fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _iter_ereno_rows(
    path: Path, section: DatasetSection
) -> tuple[list[str], Iterable[tuple[str, list[str]]], Counter[str]]:
    handle = path.open("r", encoding="utf-8-sig", newline="")
    reader = csv.reader(handle, skipinitialspace=True)
    header = [value.strip() for value in next(reader)]
    counters: Counter[str] = Counter()

    def rows() -> Iterable[tuple[str, list[str]]]:
        current = "csv"
        try:
            for row in reader:
                if row and row[0].strip().lower().startswith("@relation"):
                    current = "metadata"
                    continue
                if row and row[0].strip().lower() == "@data":
                    current = "arff"
                    continue
                if current == "metadata":
                    counters["metadata"] += 1
                    continue
                if len(row) != len(header):
                    counters["malformed"] += 1
                    continue
                if section != "all" and current != section:
                    counters[f"ignored_{current}"] += 1
                    continue
                counters[current] += 1
                yield current, [value.strip() for value in row]
        finally:
            handle.close()

    return header, rows(), counters


def _fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("\x1f".join(frame.columns).encode("utf-8"))
    for row in frame.itertuples(index=False, name=None):
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_ereno(
    path: str | Path,
    section: DatasetSection = "csv",
    label_column: str = LABEL_COLUMN,
    drop_conflicting: bool = True,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, DatasetProfile]:
    source = Path(path)
    header, rows, counters = _iter_ereno_rows(source, section)

    records: list[list[str]] = []
    origins: list[str] = []
    for origin, values in rows:
        records.append(values)
        origins.append(origin)

    frame = pd.DataFrame.from_records(records, columns=header)
    if frame.empty:
        raise ValueError(f"No rows found in section {section!r} of {source}")
    if label_column not in frame:
        raise ValueError(f"Label column {label_column!r} was not found")

    frame.insert(0, "row_id", [f"{origin}:{index}" for index, origin in enumerate(origins)])
    feature_columns = [column for column in header if column != label_column]
    rows_read = len(frame)

    conflicting_vectors_removed = 0
    if drop_conflicting:
        label_cardinality = frame.groupby(feature_columns, dropna=False)[label_column].transform(
            "nunique"
        )
        conflicts = label_cardinality > 1
        conflicting_vectors_removed = int(conflicts.sum())
        frame = frame.loc[~conflicts].copy()

    duplicate_rows_removed = 0
    if drop_duplicates:
        duplicate_mask = frame.duplicated(subset=header, keep="first")
        duplicate_rows_removed = int(duplicate_mask.sum())
        frame = frame.loc[~duplicate_mask].copy()

    for column in feature_columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().all():
            frame[column] = converted

    frame[label_column] = frame[label_column].astype("string").str.strip()
    frame.reset_index(drop=True, inplace=True)

    profile = DatasetProfile(
        source=str(source),
        selected_section=section,
        rows_read=rows_read,
        rows_after_cleaning=len(frame),
        columns=len(header),
        class_counts={
            str(label): int(count)
            for label, count in frame[label_column].value_counts().sort_index().items()
        },
        duplicate_rows_removed=duplicate_rows_removed,
        conflicting_vectors_removed=conflicting_vectors_removed,
        malformed_lines_skipped=int(counters["malformed"]),
        fingerprint=_fingerprint(frame[[*header]]),
    )
    return frame, profile


def select_features(
    frame: pd.DataFrame,
    profile: str,
    label_column: str = LABEL_COLUMN,
) -> list[str]:
    if profile == "decision":
        missing = [feature for feature in DECISION_FEATURES if feature not in frame]
        if missing:
            raise ValueError(f"Missing decision features: {', '.join(missing)}")
        return list(DECISION_FEATURES)
    if profile == "all":
        return [column for column in frame if column not in {"row_id", label_column, "Time"}]
    raise ValueError(f"Unknown feature profile: {profile}")
