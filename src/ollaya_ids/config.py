from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must contain a mapping: {config_path}")
    return config


def with_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        target = updated
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return updated
