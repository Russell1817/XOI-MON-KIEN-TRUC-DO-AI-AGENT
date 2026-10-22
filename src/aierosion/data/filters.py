"""Path filters shared by the agent and human PR selection."""
from __future__ import annotations

import re
from functools import lru_cache

from aierosion.config import settings


@lru_cache(maxsize=1)
def _excluded() -> list[re.Pattern]:
    return [re.compile(p) for p in settings()["filter"]["excluded_path_patterns"]]


def is_source_py(filename: str) -> bool:
    """True for a .py file that is not a test, doc, example or setup file."""
    name = filename.replace("\\", "/")
    if not name.endswith(".py"):
        return False
    return not any(p.search(name) for p in _excluded())
