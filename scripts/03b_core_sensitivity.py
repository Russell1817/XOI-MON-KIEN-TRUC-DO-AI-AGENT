"""Step 03b: sensitivity of the core-package threshold (Δ I of the top 5%, 10% and 20% packages by Ca).

Re-measures only the PRs of the analysis sample, at the exact commits recorded in pr_metrics.csv
(sha_before, sha_after), so the main results are not touched. The 10% column must reproduce the
existing d_i_core values; the script reports how many PRs match. Results are cached per repository in
data/interim/core_sensitivity/, so the script can be stopped and resumed.

Output: results/tables/core_sensitivity.csv (one row per PR: d_i_core_05, d_i_core_10, d_i_core_20)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd

from aierosion.config import path, settings
from aierosion.data.filters import is_source_py
from aierosion.git import repo as gitrepo
from aierosion.graph.build import GraphBuilder
from aierosion.metrics.compute import compute
from aierosion.runlog import log_run

FRACTIONS = {"05": 0.05, "10": 0.10, "20": 0.20}


def _slug(full_name: str) -> str:
    return full_name.replace("/", "__")


def measure_repo(full_name: str, prs: list[dict], shallow_since: str) -> dict:
    mcfg = settings()["mining"]
    out = path("interim") / "core_sensitivity" / f"{_slug(full_name)}.csv"
    clone_dir = path("repos") / f"sens_{_slug(full_name)}.git"
    t0, rows, info = time.time(), [], {"repo": full_name, "status": "ok"}
    try:
        gitrepo.clone(full_name, clone_dir, shallow_since, mcfg["blob_limit"], mcfg["clone_timeout_s"])
        builder = GraphBuilder(clone_dir)
        try:
            for p in prs:
                row = {"repo": full_name, "number": p["number"], "group": p["group"]}
                try:
                    before, after = p["sha_before"], p["sha_after"]
                    changed = gitrepo.changed_files(clone_dir, before, after)
                    moved = gitrepo.renames(clone_dir, before, after)
                    py = {f for _, f in changed if is_source_py(f)}
                    g0, g1 = builder.build(before), builder.build(after)
                    for key, frac in FRACTIONS.items():
                        row[f"d_i_core_{key}"] = compute(g0, g1, py, frac, renames=moved)[0]["d_i_core"]
                except Exception as e:  # one bad PR must not stop the repository
                    row["error"] = str(e)[:200]
                rows.append(row)
        finally:
            builder.close()
    except Exception as e:
        info.update(status="error", error=str(e)[:300], trace=traceback.format_exc()[-800:])
    finally:
        if not mcfg["keep_clones"]:
            gitrepo.remove(clone_dir)
    if info["status"] == "ok":
        pd.DataFrame(rows).to_csv(out, index=False)
    info["seconds"] = round(time.time() - t0, 1)
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    parser.add_argument("--repos", nargs="*")
    args = parser.parse_args()
    cfg = settings()

    m = pd.read_csv(path("processed") / "pr_metrics.csv", low_memory=False)
    ok = m[m["status"] == "ok"]
    groups = ok.groupby("repo")["group"].agg(set)
    sample = ok[ok["repo"].isin(groups[groups.apply(lambda s: {"agent", "human"} <= s)].index)]
    agent = pd.read_csv(path("interim") / "agent_prs.csv")
    # Same shallow boundary as step 03: 200 days before the repository's first agent PR.
    first = agent.groupby("full_name")["merged_at"].min()

    out_dir = path("interim") / "core_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    repos = sorted(sample["repo"].unique())
    if args.repos:
        repos = [r for r in repos if r in args.repos]
    todo = [r for r in repos if not (out_dir / f"{_slug(r)}.csv").exists()]
    print(f"{len(repos)} repositories, {len(repos) - len(todo)} cached, {len(todo)} to measure", flush=True)

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    errors = []
    with ProcessPoolExecutor(max_workers=args.workers or cfg["mining"]["workers"]) as pool:
        futures = {}
        for r in todo:
            since = (datetime.fromisoformat(first[r].replace("Z", "+00:00"))
                     - timedelta(days=cfg["mining"]["shallow_margin_days"])).strftime("%Y-%m-%d")
            prs = sample[sample["repo"] == r][["number", "group", "sha_before", "sha_after"]].to_dict("records")
            futures[pool.submit(measure_repo, r, prs, since)] = r
        for i, fut in enumerate(as_completed(futures), 1):
            info = fut.result()
            if info["status"] != "ok":
                errors.append(info)
            print(f"[{i}/{len(todo)}] {info['repo']}: {info['status']} {info['seconds']}s {info.get('error', '')[:100]}",
                  flush=True)

    parts = [pd.read_csv(f) for f in sorted(out_dir.glob("*.csv"))]
    if not parts:
        print("nothing measured", flush=True)
        return 1
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(path("tables") / "core_sensitivity.csv", index=False)
    merged = res.merge(sample[["repo", "number", "group", "d_i_core"]], on=["repo", "number", "group"], how="left")
    both_nan = merged["d_i_core"].isna() & merged["d_i_core_10"].isna()
    close = (merged["d_i_core"] - merged["d_i_core_10"]).abs() < 1e-9
    match = int((both_nan | close).sum())
    summary = {"prs": len(res), "repos": int(res["repo"].nunique()) if len(res) else 0,
               "match_10pct_with_main": match, "errors": len(errors)}
    print(json.dumps(summary), flush=True)
    log_run("03b_core_sensitivity.py", {"fractions": list(FRACTIONS.values())}, summary, valid=not errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
