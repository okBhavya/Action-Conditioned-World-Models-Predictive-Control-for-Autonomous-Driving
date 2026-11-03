"""Configuration loading, merging, and hashing utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Load and parse a YAML file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of a configuration dictionary."""
    encoded = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]
