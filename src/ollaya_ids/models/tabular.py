from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..metrics import RuntimeStats


@dataclass(frozen=True)
class FittedTabularModel:
    pipeline: Pipeline
    labels: list[str]
    training_seconds: float
    encoded_target: bool = False

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, RuntimeStats]:
        started = perf_counter()
        raw_predictions = self.pipeline.predict(frame)
        model_probabilities = self.pipeline.predict_proba(frame)
        elapsed = perf_counter() - started

        model = self.pipeline.named_steps["model"]
        if self.encoded_target:
            predictions = np.asarray([self.labels[int(index)] for index in raw_predictions])
            model_labels = [self.labels[int(index)] for index in model.classes_]
        else:
            predictions = raw_predictions
            model_labels = [str(label) for label in model.classes_]
        probability_by_label = {
            label: model_probabilities[:, index] for index, label in enumerate(model_labels)
        }
        probabilities = np.column_stack(
            [probability_by_label.get(label, np.zeros(len(frame))) for label in self.labels]
        )
        return predictions, probabilities, RuntimeStats(len(frame), elapsed)


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = frame.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    numeric = [column for column in frame.columns if column not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def fit_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    labels: list[str],
    options: dict[str, Any],
    seed: int,
) -> FittedTabularModel:
    model = RandomForestClassifier(
        n_estimators=int(options.get("estimators", 500)),
        min_samples_leaf=int(options.get("min_samples_leaf", 1)),
        max_features=options.get("max_features", "sqrt"),
        n_jobs=int(options.get("jobs", -1)),
        class_weight="balanced_subsample",
        random_state=seed,
    )
    pipeline = Pipeline([("preprocess", _preprocessor(x_train)), ("model", model)])
    started = perf_counter()
    pipeline.fit(x_train, y_train)
    return FittedTabularModel(pipeline, labels, perf_counter() - started)


def fit_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    labels: list[str],
    options: dict[str, Any],
    seed: int,
) -> FittedTabularModel:
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError("Install the xgboost extra: pip install -e '.[xgboost]'") from error

    model = XGBClassifier(
        n_estimators=int(options.get("estimators", 700)),
        max_depth=int(options.get("max_depth", 8)),
        learning_rate=float(options.get("learning_rate", 0.08)),
        subsample=float(options.get("subsample", 0.85)),
        colsample_bytree=float(options.get("colsample_bytree", 0.85)),
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device=options.get("device", "cuda"),
        random_state=seed,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocess", _preprocessor(x_train)), ("model", model)])
    label_to_index = {label: index for index, label in enumerate(labels)}
    encoded_target = y_train.astype(str).map(label_to_index)
    if encoded_target.isna().any():
        unknown = sorted(set(y_train.astype(str)) - set(label_to_index))
        raise ValueError(f"Unknown training labels: {', '.join(unknown)}")
    started = perf_counter()
    pipeline.fit(x_train, encoded_target.astype(int))
    return FittedTabularModel(pipeline, labels, perf_counter() - started, encoded_target=True)
