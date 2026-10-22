"""Step 01: download AIDev-pop tables at a pinned revision and record their schema."""
from __future__ import annotations

import argparse
import json
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from aierosion.config import path, settings
from aierosion.runlog import log_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallback", action="store_true", help="also download all_pull_request / all_repository")
    args = parser.parse_args()

    cfg = settings()["aidev"]
    tables = cfg["tables"] + (cfg["fallback_tables"] if args.fallback else [])
    out_dir = path("raw") / "aidev"
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = {}
    for table in tables:
        local = hf_hub_download(
            repo_id=cfg["repo_id"],
            repo_type="dataset",
            filename=f"{table}.parquet",
            revision=cfg["revision"],
            local_dir=out_dir,
        )
        meta = pq.ParquetFile(local)
        schema[table] = {
            "rows": meta.metadata.num_rows,
            "columns": {f.name: str(f.type) for f in meta.schema_arrow},
        }
        print(f"{table}: {meta.metadata.num_rows:,} rows, {len(schema[table]['columns'])} columns")

    schema_file = out_dir / "schema.json"
    schema_file.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    log_run(
        "01_download_aidev.py",
        {"repo_id": cfg["repo_id"], "revision": cfg["revision"], "tables": tables},
        {t: s["rows"] for t, s in schema.items()},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
