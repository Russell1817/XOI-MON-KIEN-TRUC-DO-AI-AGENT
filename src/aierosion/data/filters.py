"""Path filters shared by the agent and human PR selection."""
from __future__ import annotations

import re
from functools import lru_cache

from aierosion.config import settings


@lru_cache(maxsize=1)
def _excluded() -> list[re.Pattern]:
    return [re.compile(p) for p in settings()["filter"]["excluded_path_patterns"]]


@lru_cache(maxsize=1)
def _bots() -> list[re.Pattern]:
    return [re.compile(p) for p in settings()["human_control"]["bot_author_patterns"]]


@lru_cache(maxsize=1)
def _traces() -> list[re.Pattern]:
    return [re.compile(p, re.I) for p in settings()["human_control"]["agent_trace_patterns"]]


def is_bot_pr(author: str, subject: str) -> bool:
    """True if the PR author, or the source branch named in a merge subject, is a bot or an agent account."""
    branch = subject.split(" from ", 1)[1] if subject.startswith("Merge pull request") and " from " in subject else ""
    return any(p.search(author) or (branch and p.search(branch)) for p in _bots())


def has_agent_trace(text: str) -> bool:
    return any(p.search(text) for p in _traces())


def is_source_py(filename: str) -> bool:
    """True for a .py file that is not a test, doc, example or setup file."""
    name = filename.replace("\\", "/")
    if not name.endswith(".py"):
        return False
    return not any(p.search(name) for p in _excluded())
