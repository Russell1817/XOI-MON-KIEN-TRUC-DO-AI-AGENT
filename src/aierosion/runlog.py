"""Append one JSON line per script run so every number can be traced to a run."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

from .config import ROOT, path


def code_commit() -> str:
    """Current git commit of the code, with '+dirty' if there are uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except subprocess.CalledProcessError:
        return "no-commit"


def log_run(script: str, params: dict, outputs: dict, valid: bool = True, note: str = "") -> dict:
    entry = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": script,
        "code_commit": code_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "params": params,
        "outputs": outputs,
        "valid": valid,
        "note": note,
    }
    with open(path("logs") / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
