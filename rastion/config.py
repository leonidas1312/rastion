"""Local JSON config helpers for CLI authentication state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CONFIG_DIRNAME = ".rastion"
_CONFIG_FILENAME = "config.json"
_TOKEN_KEY = "hub_token"


def _config_dir() -> Path:
    override = os.environ.get("RASTION_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / _CONFIG_DIRNAME).resolve()


def _config_path() -> Path:
    return _config_dir() / _CONFIG_FILENAME


def _read_config() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_config(payload: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_token() -> str | None:
    """Read the persisted hub token from ~/.rastion/config.json."""
    value = _read_config().get(_TOKEN_KEY)
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def save_token(token: str) -> None:
    """Persist the hub token in ~/.rastion/config.json."""
    cleaned = token.strip()
    if not cleaned:
        raise ValueError("token cannot be empty")
    data = _read_config()
    data[_TOKEN_KEY] = cleaned
    _write_config(data)


def clear_token() -> None:
    """Remove the persisted hub token from ~/.rastion/config.json."""
    data = _read_config()
    if _TOKEN_KEY in data:
        data.pop(_TOKEN_KEY, None)
        _write_config(data)
