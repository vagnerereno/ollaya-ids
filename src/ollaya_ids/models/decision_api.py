from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import requests

from ..constants import QUESTION
from ..metrics import RuntimeStats
from ..state import row_to_state


@dataclass(frozen=True)
class DecisionApiConfig:
    base_url: str
    endpoint: str
    api_key: str
    model: str
    timeout_seconds: float = 60.0
    state_precision: int = 6


class DecisionApiClassifier:
    def __init__(self, config: DecisionApiConfig, features: Iterable[str], labels: list[str]):
        self.config = config
        self.features = list(features)
        self.labels = labels
        self.session = requests.Session()
        if config.api_key:
            self.session.headers["Authorization"] = f"Bearer {config.api_key}"

    def _predict_one(self, row: pd.Series) -> tuple[str, list[float], dict[str, Any]]:
        payload = {
            "model": self.config.model,
            "state": row_to_state(row, self.features, self.config.state_precision),
            "questions": QUESTION,
        }
        response = self.session.post(
            f"{self.config.base_url.rstrip('/')}/{self.config.endpoint.lstrip('/')}",
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        answer = body["answers"]["attack_type"]
        probabilities = answer["probabilities"]
        values = [float(probabilities.get(label, 0.0)) for label in self.labels]
        total = sum(values)
        if total <= 0:
            raise ValueError("Decision API returned an empty probability distribution")
        values = [value / total for value in values]
        return str(answer["choice"]), values, body

    def predict(
        self, frame: pd.DataFrame
    ) -> tuple[list[str], np.ndarray, RuntimeStats, list[dict[str, Any]]]:
        predictions: list[str] = []
        probabilities: list[list[float]] = []
        raw_responses: list[dict[str, Any]] = []
        started = perf_counter()
        for _, row in frame.iterrows():
            prediction, distribution, response = self._predict_one(row)
            predictions.append(prediction)
            probabilities.append(distribution)
            raw_responses.append(response)
        elapsed = perf_counter() - started
        return (
            predictions,
            np.asarray(probabilities),
            RuntimeStats(len(frame), elapsed),
            raw_responses,
        )
