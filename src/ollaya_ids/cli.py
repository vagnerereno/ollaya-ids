from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .artifacts import environment_info, write_json, write_metric_tables, write_prediction_table
from .config import load_config, with_overrides
from .constants import CLASS_DESCRIPTIONS
from .data import load_ereno, select_features
from .metrics import evaluate_predictions
from .models.decision_api import DecisionApiClassifier, DecisionApiConfig
from .models.tabular import fit_random_forest, fit_xgboost
from .splits import apply_manifest, load_manifest, make_splits, save_manifest
from .state import labeled_request


def _load_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, Any, list[str]]:
    dataset = config["dataset"]
    frame, profile = load_ereno(
        dataset["path"],
        section=dataset.get("section", "csv"),
        label_column=dataset.get("label_column", "class"),
        drop_conflicting=bool(dataset.get("drop_conflicting", True)),
        drop_duplicates=bool(dataset.get("drop_duplicates", True)),
    )
    features = select_features(
        frame,
        dataset.get("feature_profile", "decision"),
        dataset.get("label_column", "class"),
    )
    return frame, profile, features


def _paths(config: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(config.get("output_dir", "artifacts"))
    return root, root / "splits.csv"


def _split_signature(config: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    split = config["split"]
    dataset = config["dataset"]
    return {
        "dataset_fingerprint": fingerprint,
        "section": dataset.get("section", "csv"),
        "drop_conflicting": bool(dataset.get("drop_conflicting", True)),
        "drop_duplicates": bool(dataset.get("drop_duplicates", True)),
        "seed": int(config.get("seed", 42)),
        "strategy": split.get("strategy", "stratified"),
        "train_size": float(split.get("train_size", 0.7)),
        "validation_size": float(split.get("validation_size", 0.15)),
    }


def _prepare(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    frame, profile, features = _load_dataset(config)
    root, manifest_path = _paths(config)
    split = config["split"]
    label_column = config["dataset"].get("label_column", "class")
    manifest = make_splits(
        frame,
        label_column=label_column,
        strategy=split.get("strategy", "stratified"),
        train_size=float(split.get("train_size", 0.7)),
        validation_size=float(split.get("validation_size", 0.15)),
        seed=int(config.get("seed", 42)),
    )
    save_manifest(manifest, manifest_path)
    write_json(root / "dataset_profile.json", profile.to_dict())
    write_json(root / "split_config.json", _split_signature(config, profile.fingerprint))
    summary = (
        manifest.groupby(["split", label_column], observed=True)
        .size()
        .rename("samples")
        .reset_index()
    )
    summary.to_csv(root / "split_summary.csv", index=False)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return apply_manifest(frame, manifest), features


def _prepared_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    root, manifest_path = _paths(config)
    if not manifest_path.exists():
        return _prepare(config)
    frame, profile, features = _load_dataset(config)
    split_config_path = root / "split_config.json"
    if not split_config_path.exists():
        return _prepare(config)
    stored = json.loads(split_config_path.read_text(encoding="utf-8"))
    if stored != _split_signature(config, profile.fingerprint):
        return _prepare(config)
    return apply_manifest(frame, load_manifest(manifest_path)), features


def command_inspect(args: argparse.Namespace) -> None:
    config = with_overrides(
        load_config(args.config),
        {"dataset.section": args.section, "dataset.feature_profile": args.feature_profile},
    )
    frame, profile, features = _load_dataset(config)
    print(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))
    print(f"features ({len(features)}): {', '.join(features)}")
    print(frame[config["dataset"].get("label_column", "class")].value_counts().to_string())


def command_prepare(args: argparse.Namespace) -> None:
    config = with_overrides(
        load_config(args.config),
        {
            "dataset.section": args.section,
            "dataset.feature_profile": args.feature_profile,
            "split.strategy": args.strategy,
        },
    )
    joined, features = _prepare(config)
    print(f"Prepared {len(joined):,} rows with {len(features)} features")
    print(joined.groupby("split").size().to_string())


def command_baseline(args: argparse.Namespace) -> None:
    config = with_overrides(
        load_config(args.config),
        {
            "baseline.xgboost.device": args.device,
            "dataset.section": args.section,
            "dataset.feature_profile": args.feature_profile,
        },
    )
    frame, features = _prepared_dataset(config)
    label_column = config["dataset"].get("label_column", "class")
    labels = list(CLASS_DESCRIPTIONS)
    train = frame.loc[frame["split"] == "train"]
    test = frame.loc[frame["split"] == "test"]

    if args.model == "random-forest":
        fitted = fit_random_forest(
            train[features],
            train[label_column],
            labels,
            config["baseline"]["random_forest"],
            int(config.get("seed", 42)),
        )
    else:
        fitted = fit_xgboost(
            train[features],
            train[label_column],
            labels,
            config["baseline"]["xgboost"],
            int(config.get("seed", 42)),
        )

    predictions, probabilities, runtime = fitted.predict(test[features])
    metrics = evaluate_predictions(test[label_column], predictions, probabilities, labels, runtime)
    metrics["model"] = args.model
    metrics["training_seconds"] = fitted.training_seconds
    metrics["feature_profile"] = config["dataset"].get("feature_profile", "decision")
    metrics["device"] = (
        config["baseline"]["xgboost"].get("device", "cuda") if args.model == "xgboost" else "cpu"
    )
    metrics["environment"] = environment_info(
        ["numpy", "pandas", "scikit-learn", "xgboost", "ollaya-ids"]
    )

    feature_profile = config["dataset"].get("feature_profile", "decision")
    run_name = args.run_name or (
        args.model if feature_profile == "decision" else f"{args.model}-{feature_profile}"
    )
    output_dir = Path(config.get("output_dir", "artifacts")) / "results" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, output_dir / "model.joblib")
    write_prediction_table(
        output_dir / "predictions.csv",
        test,
        predictions,
        probabilities,
        labels,
        label_column,
    )
    write_metric_tables(output_dir, metrics)
    print(f"{args.model}: macro-F1={metrics['macro_f1']:.4f}")


def _stratified_sample(
    frame: pd.DataFrame, label_column: str, size: int, seed: int
) -> pd.DataFrame:
    if size <= 0 or size >= len(frame):
        return frame.copy()
    sampled, _ = train_test_split(
        frame,
        train_size=size,
        random_state=seed,
        stratify=frame[label_column],
    )
    return sampled.sort_values("row_id").reset_index(drop=True)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()


def command_evaluate_api(args: argparse.Namespace) -> None:
    config = with_overrides(
        load_config(args.config),
        {
            "decision_model.base_url": args.base_url,
            "decision_model.endpoint": args.endpoint,
            "decision_model.model": args.model,
            "decision_model.max_test_samples": args.max_samples,
            "dataset.section": args.section,
            "dataset.feature_profile": args.feature_profile,
        },
    )
    frame, features = _prepared_dataset(config)
    label_column = config["dataset"].get("label_column", "class")
    decision = config["decision_model"]
    test = frame.loc[frame["split"] == "test"]
    test = _stratified_sample(
        test,
        label_column,
        int(decision.get("max_test_samples", 2000)),
        int(config.get("seed", 42)),
    )
    labels = list(CLASS_DESCRIPTIONS)
    classifier = DecisionApiClassifier(
        DecisionApiConfig(
            base_url=decision["base_url"],
            endpoint=decision.get("endpoint", "/v1/systemone"),
            api_key=decision.get("api_key", ""),
            model=decision["model"],
            timeout_seconds=float(decision.get("timeout_seconds", 60)),
            state_precision=int(decision.get("state_precision", 6)),
        ),
        features,
        labels,
    )
    predictions, probabilities, runtime, _ = classifier.predict(test)
    metrics = evaluate_predictions(test[label_column], predictions, probabilities, labels, runtime)
    metrics["model"] = decision["model"]
    metrics["endpoint"] = f"{decision['base_url']}{decision.get('endpoint', '/v1/systemone')}"
    metrics["feature_profile"] = config["dataset"].get("feature_profile", "decision")
    metrics["environment"] = environment_info(
        ["numpy", "pandas", "scikit-learn", "requests", "ollaya-ids"]
    )

    run_name = args.run_name or _safe_name(decision["model"])
    output_dir = Path(config.get("output_dir", "artifacts")) / "results" / run_name
    write_prediction_table(
        output_dir / "predictions.csv",
        test,
        predictions,
        probabilities,
        labels,
        label_column,
    )
    write_metric_tables(output_dir, metrics)
    print(f"{decision['model']}: macro-F1={metrics['macro_f1']:.4f}")


def command_export_kev(args: argparse.Namespace) -> None:
    config = with_overrides(
        load_config(args.config),
        {
            "dataset.section": args.section,
            "dataset.feature_profile": args.feature_profile,
        },
    )
    frame, features = _prepared_dataset(config)
    label_column = config["dataset"].get("label_column", "class")
    precision = int(config["decision_model"].get("state_precision", 6))
    output_dir = Path(config.get("output_dir", "artifacts")) / "kev"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        subset = frame.loc[frame["split"] == split]
        destination = output_dir / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for _, row in subset.iterrows():
                record = labeled_request(row, features, label_column, precision)
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        print(f"{split}: {len(subset):,} records -> {destination}")


def command_compare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    results_dir = Path(config.get("output_dir", "artifacts")) / "results"
    rows = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": metrics_path.parent.name,
                "model": metrics.get("model"),
                "samples": metrics.get("samples"),
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "macro_recall": metrics.get("macro_recall"),
                "weighted_f1": metrics.get("weighted_f1"),
                "brier_multiclass": metrics.get("brier_multiclass"),
                "expected_calibration_error": metrics.get("expected_calibration_error"),
                "intrusion_f1": metrics.get("binary_intrusion", {}).get("f1"),
                "false_positive_rate": metrics.get("binary_intrusion", {}).get(
                    "false_positive_rate"
                ),
                "latency_ms": metrics.get("runtime", {}).get("latency_ms_per_sample"),
                "throughput": metrics.get("runtime", {}).get("throughput_samples_per_second"),
            }
        )
    if not rows:
        raise SystemExit(f"No metrics found under {results_dir}")
    comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    comparison.to_csv(results_dir / "comparison.csv", index=False)
    print(comparison.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ollaya-ids")
    parser.add_argument("--config", default="configs/experiment.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def dataset_options(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--section", choices=["csv", "arff", "all"])
        subparser.add_argument("--feature-profile", choices=["decision", "all"])

    inspect_parser = subparsers.add_parser("inspect")
    dataset_options(inspect_parser)
    inspect_parser.set_defaults(function=command_inspect)

    prepare_parser = subparsers.add_parser("prepare")
    dataset_options(prepare_parser)
    prepare_parser.add_argument("--strategy", choices=["stratified", "temporal"])
    prepare_parser.set_defaults(function=command_prepare)

    baseline_parser = subparsers.add_parser("baseline")
    dataset_options(baseline_parser)
    baseline_parser.add_argument("--model", choices=["random-forest", "xgboost"], required=True)
    baseline_parser.add_argument("--device", choices=["cpu", "cuda"])
    baseline_parser.add_argument("--run-name")
    baseline_parser.set_defaults(function=command_baseline)

    api_parser = subparsers.add_parser("evaluate-api")
    dataset_options(api_parser)
    api_parser.add_argument("--base-url")
    api_parser.add_argument("--endpoint")
    api_parser.add_argument("--model")
    api_parser.add_argument("--max-samples", type=int)
    api_parser.add_argument("--run-name")
    api_parser.set_defaults(function=command_evaluate_api)

    export_parser = subparsers.add_parser("export-kev")
    dataset_options(export_parser)
    export_parser.set_defaults(function=command_export_kev)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.set_defaults(function=command_compare)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
