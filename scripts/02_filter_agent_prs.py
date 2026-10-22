"""Step 02: select merged agent PRs that change Python source in Python repositories."""
from __future__ import annotations

import sys

import pandas as pd

from aierosion.config import path, settings
from aierosion.data.filters import is_source_py
from aierosion.runlog import log_run


def main() -> int:
    cfg = settings()["filter"]
    raw = path("raw") / "aidev"
    pr = pd.read_parquet(raw / "pull_request.parquet")
    repo = pd.read_parquet(raw / "repository.parquet")
    funnel = []

    def step(name: str, df: pd.DataFrame) -> None:
        funnel.append({"step": name, "agent_prs": len(df), "repos": df["repo_id"].nunique()})
        print(f"{name:<45} {len(df):>7} PRs {df['repo_id'].nunique():>6} repos")

    step("AIDev-pop", pr)

    py_repos = repo[(repo["language"] == cfg["language"]) & (~repo["is_forked"])]
    df = pr.merge(
        py_repos[["id", "full_name", "stars", "license"]].rename(columns={"id": "repo_id"}), on="repo_id", how="inner"
    )
    step("primary language Python, not a fork", df)

    df = df[df["license"].isin(cfg["allowed_licenses"])]
    step("open-source license allows reuse", df)

    df = df[df["merged_at"].notna()]
    step("merged", df)

    # Files changed by each PR, from the per-commit file list.
    details = pd.read_parquet(
        raw / "pr_commit_details.parquet", columns=["pr_id", "filename", "additions", "deletions"]
    )
    details = details[details["pr_id"].isin(df["id"]) & details["filename"].notna()]
    details["source_py"] = details["filename"].map(is_source_py)
    per_pr = details.groupby("pr_id").agg(
        files_changed=("filename", "nunique"),
        loc_changed=("additions", "sum"),
        deleted=("deletions", "sum"),
        source_py_files=("source_py", "sum"),
    )
    per_pr["loc_changed"] = per_pr["loc_changed"] + per_pr.pop("deleted")
    df = df.merge(per_pr, left_on="id", right_index=True, how="left")
    df = df[df["source_py_files"].fillna(0) > 0]
    step("changes Python source (not tests/docs/setup)", df)

    counts = df.groupby("repo_id")["id"].transform("count")
    df = df[counts >= cfg["min_agent_prs_per_repo"]]
    step(f"repo has >= {cfg['min_agent_prs_per_repo']} such agent PRs", df)

    out = df[
        ["id", "number", "agent", "user", "created_at", "merged_at", "repo_id", "full_name", "stars", "license",
         "html_url", "files_changed", "source_py_files", "loc_changed"]
    ].rename(columns={"id": "pr_id"}).sort_values(["full_name", "number"])
    out.to_csv(path("interim") / "agent_prs.csv", index=False)
    pd.DataFrame(funnel).to_csv(path("logs") / "filter_funnel.csv", index=False)

    by_agent = out["agent"].value_counts()
    print("\nPer agent:\n" + by_agent.to_string())
    ok = (
        len(out) >= cfg["min_total_agent_prs"]
        and out["repo_id"].nunique() >= cfg["min_repos"]
    )
    small = by_agent[by_agent < cfg["min_prs_per_agent"]].index.tolist()
    note = f"agents below {cfg['min_prs_per_agent']} PRs: {small}" if small else ""
    print(f"\nPass criteria met: {ok}. {note}")
    log_run(
        "02_filter_agent_prs.py",
        {"filter": cfg},
        {"agent_prs": len(out), "repos": int(out["repo_id"].nunique()), "per_agent": by_agent.to_dict()},
        valid=ok,
        note=note,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
