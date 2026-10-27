"""Step 00: check that the environment is ready before running the pipeline."""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys


from aierosion.config import ROOT, settings

LIBS = ["pandas", "pyarrow", "huggingface_hub", "networkx", "scipy", "scikit_posthocs", "matplotlib", "yaml"]
MIN_FREE_GB = 20


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main() -> int:
    results = []
    results.append(check("python", sys.version_info[:2] == (3, 12), sys.version.split()[0]))

    versions = []
    missing = []
    for lib in LIBS:
        try:
            mod = importlib.import_module(lib)
            versions.append(f"{lib} {getattr(mod, '__version__', '?')}")
        except ImportError:
            missing.append(lib)
    results.append(check("libraries", not missing, ", ".join(versions) if not missing else f"missing {missing}"))

    git = shutil.which("git")
    git_ver = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip() if git else "not found"
    results.append(check("git", git is not None, git_ver))

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    results.append(check("disk", free_gb >= MIN_FREE_GB, f"{free_gb:.1f} GB free (need {MIN_FREE_GB})"))

    results.append(check("settings", "aidev" in settings(), "config/settings.yaml loaded"))

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
