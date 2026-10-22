"""Export every draw.io source in results/figures/src/ to PNG (scale 3, ~300 dpi at print size) and SVG."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = Path(os.environ.get("DRAWIO_EXE", Path.home() / "AppData/Local/Programs/draw.io/draw.io.exe"))


def main() -> int:
    out = ROOT / "results" / "figures"
    for src in sorted((out / "src").glob("*.drawio")):
        for fmt, extra in (("png", ["-s", "3"]), ("svg", [])):
            target = out / f"{src.stem}.{fmt}"
            subprocess.run([str(EXE), "-x", "-f", fmt, *extra, "-b", "10", "-o", str(target), str(src)],
                           check=True, capture_output=True, timeout=120)
            print(f"{src.name} -> {target.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
