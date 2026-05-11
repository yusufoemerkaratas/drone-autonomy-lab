"""Configuration loading helpers for missions and environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return a mapping."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {config_path}")

    return data


def load_mission_config() -> dict[str, Any]:
    """Load the default mission profile."""
    return load_yaml_config(CONFIG_DIR / "mission.yaml")


def load_environment_config() -> dict[str, Any]:
    """Load the default simulation environment."""
    return load_yaml_config(CONFIG_DIR / "environment.yaml")
