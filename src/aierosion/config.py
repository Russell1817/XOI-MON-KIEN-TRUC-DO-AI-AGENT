"""Load settings.yaml and resolve project paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def settings() -> dict:
    with open(ROOT / "config" / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for a key under `paths:` in settings.yaml; created if missing."""
    p = ROOT / settings()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def github_token() -> str | None:
    load_dotenv(ROOT / ".env")
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    return token or None
