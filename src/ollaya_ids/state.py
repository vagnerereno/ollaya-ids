from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .constants import QUESTION


def _clean_value(value: Any, precision: int) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return round(number, precision)
    return str(value).strip()


def row_to_state(row: pd.Series, features: Iterable[str], precision: int = 6) -> dict[str, Any]:
    return {feature: _clean_value(row[feature], precision) for feature in features}


def labeled_request(
    row: pd.Series,
    features: Iterable[str],
    label_column: str,
    precision: int = 6,
) -> dict[str, Any]:
    questions = {key: {**value, "label": str(row[label_column])} for key, value in QUESTION.items()}
    return {
        "state": row_to_state(row, features, precision),
        "questions": questions,
        "_meta": {"id": str(row["row_id"])},
    }
