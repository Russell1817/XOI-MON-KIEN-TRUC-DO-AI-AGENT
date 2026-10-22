"""Step 03: per repository, find agent and human PRs in git history and compute erosion metrics.

One clone per repository (shallow, partial, bare), removed after processing. Results are cached
per repository in data/interim/mined/, so the script can be stopped and resumed.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd

from aierosion.config import path, settings
from aierosion.data.filters import is_source_py
from aierosion.git import repo as gitrepo
from aierosion.graph.build import GraphBuilder
from aierosion.metrics.compute import compute
from aierosion.runlog import log_run

LINTER_FILES = [".importlinter", "setup.cfg", "pyproject.toml", "tox.ini"]


def _slug(full_name: str) -> str:
    return full_name.replace("/", "__")


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _import_linter_config(repo) -> dict:
    found = {}
    for f in LINTER_FILES:
        text = gitrepo.git(repo, "show", f"HEAD:{f}", check=False)
        if "importlinter" in text:
            found[f] = text
    return found


def mine_repo(full_name: str, agent_prs: list[dict], aidev_numbers: list[int]) -> dict:
    cfg = settings()
    mcfg, hcfg = cfg["mining"], cfg["human_control"]
    out_dir = path("interim") / "mined"
    clone_dir = path("repos") / f"{_slug(full_name)}.git"
    t0 = time.time()
    info = {"repo": full_name, "status": "ok"}

    merged = [_parse_date(p["merged_at"]) for p in agent_prs]
    lo = min(merged) - timedelta(days=hcfg["window_days"])
    hi = max(merged) + timedelta(days=hcfg["window_days"])
    shallow_since = (min(merged) - timedelta(days=mcfg["shallow_margin_days"])).strftime("%Y-%m-%d")

    rows, edges = [], []
    try:
        gitrepo.clone(full_name, clone_dir, shallow_since, mcfg["blob_limit"], mcfg["clone_timeout_s"])
        info["clone_mb"] = round(sum(f.stat().st_size for f in clone_dir.rglob("*") if f.is_file()) / 1e6, 1)
        commits = gitrepo.first_parent_pr_commits(clone_dir)
        by_number = {}
        for c in commits:
            by_number.setdefault(c.number, c)
        info["pr_commits_on_default_branch"] = len(by_number)

        bot_re = [re.compile(p) for p in hcfg["bot_author_patterns"]]
        trace_re = [re.compile(p, re.I) for p in hcfg["agent_trace_patterns"]]
        aidev = set(aidev_numbers)

        selected = []
        for p in agent_prs:
            c = by_number.get(int(p["number"]))
            if c is None:
                rows.append({**_base(full_name, p["number"], "agent", p["agent"]), "status": "not_on_first_parent"})
            else:
                selected.append(("agent", p["agent"], c))

        human_pool, human_counts = [], {"in_window": 0, "bot": 0, "agent_trace": 0}
        for n, c in by_number.items():
            if n in aidev or not (lo <= _parse_date(c.date) <= hi):
                continue
            human_counts["in_window"] += 1
            if any(r.search(f"{c.author} <{c.email}>") for r in bot_re):
                human_counts["bot"] += 1
                continue
            text = c.message
            if c.style == "merge":
                text += "\n" + gitrepo.commit_messages(clone_dir, f"{c.parents[0]}..{c.parents[1]}")
            if any(r.search(text) for r in trace_re):
                human_counts["agent_trace"] += 1
                continue
            human_pool.append(c)
        info["human_candidates"] = human_counts

        # Changed source files decide eligibility for both groups the same way.
        eligible_h = []
        for c in human_pool:
            changed = gitrepo.changed_files(clone_dir, c.parents[0], c.sha)
            if any(is_source_py(f) for _, f in changed):
                eligible_h.append(c)
        info["human_eligible"] = len(eligible_h)
        # crc32, not hash(): str hashing is salted per process and would break reproducibility.
        rng = random.Random(cfg["seed"] + zlib.crc32(full_name.encode()))
        if len(eligible_h) > mcfg["max_human_per_repo"]:
            eligible_h = rng.sample(eligible_h, mcfg["max_human_per_repo"])
        selected += [("human", "Human", c) for c in eligible_h]
        selected.sort(key=lambda s: s[2].date)

        builder = GraphBuilder(clone_dir)
        try:
            for group, label, c in selected:
                row = _base(full_name, c.number, group, label)
                row.update(sha_before=c.parents[0], sha_after=c.sha, merge_style=c.style, merged_at=c.date,
                           author=c.author)
                try:
                    changed = gitrepo.changed_files(clone_dir, c.parents[0], c.sha)
                    py = {f for _, f in changed if is_source_py(f)}
                    row.update(files_changed=len(changed), py_files_changed=len(py),
                               loc_changed=gitrepo.loc_changed(clone_dir, c.parents[0], c.sha))
                    if not py:
                        row["status"] = "no_py_source_change"
                        rows.append(row)
                        continue
                    g0, g1 = builder.build(c.parents[0]), builder.build(c.sha)
                    moved = gitrepo.renames(clone_dir, c.parents[0], c.sha)
                    metrics, added, added_pkg = compute(g0, g1, py, cfg["graph"]["core_module_top_fraction"],
                                                        renames=moved)
                    row.update(metrics, status="ok", renamed_py=sum(f.endswith(".py") for f in (moved or {})),
                               rename_detection="ok" if moved is not None else "unavailable")
                    edges.append({"repo": full_name, "number": c.number, "group": group,
                                  "added_edges": added, "added_pkg_edges": added_pkg})
                except Exception as e:  # one bad PR must not stop the repository
                    row.update(status="error", error=str(e)[:300])
                rows.append(row)
        finally:
            builder.close()

        # Repository layout and import-linter config, for the layering rules (step 05).
        head_files = [p for p, _ in gitrepo.ls_py(clone_dir, "HEAD") if is_source_py(p)]
        (out_dir / f"{_slug(full_name)}.layout.json").write_text(
            json.dumps({"files": head_files, "import_linter": _import_linter_config(clone_dir)}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        info.update(status="error", error=str(e)[:500], trace=traceback.format_exc()[-1500:])
    finally:
        if not mcfg["keep_clones"]:
            gitrepo.remove(clone_dir)

    info["seconds"] = round(time.time() - t0, 1)
    pd.DataFrame(rows).to_csv(out_dir / f"{_slug(full_name)}.csv", index=False)
    with open(out_dir / f"{_slug(full_name)}.edges.jsonl", "w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e) + "\n")
    (out_dir / f"{_slug(full_name)}.info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def _base(full_name: str, number, group: str, label: str) -> dict:
    return {"repo": full_name, "number": int(number), "group": group, "agent": label}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="*", help="only these owner/name repositories")
    parser.add_argument("--limit", type=int, help="only the first N repositories")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--retry-errors", action="store_true", help="re-mine repositories that failed transiently")
    args = parser.parse_args()

    agent = pd.read_csv(path("interim") / "agent_prs.csv")
    allpr = pd.read_parquet(path("raw") / "aidev" / "pull_request.parquet", columns=["repo_id", "number"])
    numbers = allpr.groupby("repo_id")["number"].apply(list).to_dict()
    repos = sorted(agent["full_name"].unique())
    if args.repos:
        repos = [r for r in repos if r in args.repos]
    if args.limit:
        repos = repos[: args.limit]

    out_dir = path("interim") / "mined"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.retry_errors:
        # Transient failures (network, DNS) are retried; deleted repositories stay excluded.
        for f in out_dir.glob("*.info.json"):
            info = json.loads(f.read_text(encoding="utf-8"))
            if info["status"] == "error" and "Repository not found" not in info.get("error", ""):
                print(f"retrying {info['repo']}: {info['error'][-80:]!r}")
                f.unlink()
    todo =[r for r in repos if not (out_dir / f"{_slug(r)}.info.json").exists()]
    print(f"{len(repos)} repositories, {len(repos) - len(todo)} cached, {len(todo)} to mine")

    if sys.platform == "win32":
        # Keep the machine awake while mining (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); a sleep in the middle
        # of a run makes git subprocess timeouts fire with negative durations.
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    workers = args.workers or settings()["mining"]["workers"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for r in todo:
            sub = agent[agent["full_name"] == r]
            futures[pool.submit(mine_repo, r, sub.to_dict("records"), numbers[int(sub["repo_id"].iloc[0])])] = r
        for i, fut in enumerate(as_completed(futures), 1):
            r = futures[fut]
            try:
                info = fut.result()
                print(f"[{i}/{len(todo)}] {r}: {info['status']} {info.get('seconds')}s "
                      f"{info.get('error', '')[:120]}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(todo)}] {r}: worker crashed {e}", flush=True)

    # Combine every repository of the current selection (step 02 may have narrowed it since mining),
    # not only the ones mined in this call: --repos runs must not drop the others from the outputs.
    wanted = set(agent["full_name"].unique())
    frames = [pd.read_csv(out_dir / f"{_slug(r)}.csv") for r in sorted(wanted)
              if (out_dir / f"{_slug(r)}.csv").exists() and (out_dir / f"{_slug(r)}.csv").stat().st_size > 2]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(combined):
        # Agent PRs must belong to the current step-02 selection (it may have been refined after mining).
        keep = set(zip(agent["full_name"], agent["number"].astype(int)))
        is_agent = combined["group"] == "agent"
        in_sel = [(r, int(n)) in keep for r, n in zip(combined["repo"], combined["number"])]
        combined = combined[~is_agent | pd.Series(in_sel, index=combined.index)]
    combined.to_csv(path("interim") / "mined_prs.csv", index=False)
    infos = [json.loads((out_dir / f"{_slug(r)}.info.json").read_text(encoding="utf-8")) for r in wanted
             if (out_dir / f"{_slug(r)}.info.json").exists()]
    pd.DataFrame([{k: v for k, v in i.items() if k != "trace"} for i in infos]).to_csv(
        path("logs") / "mining_repos.csv", index=False)
    ok_repos = sum(i["status"] == "ok" for i in infos)
    status = combined["status"].value_counts().to_dict() if len(combined) else {}
    print(f"\nRepositories ok: {ok_repos}/{len(infos)}; PR status: {status}")
    log_run("03_mine_repos.py", {"mining": settings()["mining"], "human_control": settings()["human_control"],
                                 "repos_requested": len(repos)},
            {"repos_ok": ok_repos, "repos_total": len(infos), "pr_status": status})
    return 0


if __name__ == "__main__":
    sys.exit(main())
