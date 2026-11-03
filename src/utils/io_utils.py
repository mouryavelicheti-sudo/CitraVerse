"""Config loading, path resolution, and structured I/O (CSV/JSON) helpers."""
import json
import os

import pandas as pd
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(path: str) -> str:
    """Resolve a path relative to the project root if it isn't absolute.

    Using this everywhere is what makes the project runnable regardless of
    the working directory it's launched from (important on Windows where
    users often double-click scripts or run from arbitrary cwd).
    """
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def load_yaml(path: str) -> dict:
    full = resolve_path(path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"Config file not found: {full}")
    with open(full, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, path: str) -> None:
    full = resolve_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_json(data, path: str) -> None:
    full = resolve_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str):
    full = resolve_path(path)
    with open(full, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    full = resolve_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if full.endswith(".csv"):
        df.to_csv(full, index=False)
    elif full.endswith(".json"):
        df.to_json(full, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported dataframe export format: {full}")


def ensure_dir(path: str) -> str:
    full = resolve_path(path)
    os.makedirs(full, exist_ok=True)
    return full
