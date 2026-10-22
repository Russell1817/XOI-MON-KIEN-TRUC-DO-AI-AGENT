"""Step 06: statistical analysis for RQ1-RQ3 and robustness checks.

Analysis sample: PRs with status 'ok' in repositories that have at least one analysed agent PR
and one analysed human PR, so that every comparison is between PRs of the same repositories.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from aierosion.config import path, settings
from aierosion.runlog import log_run
from aierosion.stats.tests import compare_rates, compare_two, holm, kruskal_dunn, wilson

RQ1_CONTINUOUS = ["added_edges", "edges_per_file", "added_pkg_edges", "d_cyclic_pkgs", "d_ce_mean", "d_i_core"]


def analysis_sample(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    groups = ok.groupby("repo")["group"].agg(set)
    both = groups[groups.apply(lambda s: {"agent", "human"} <= s)].index
    ok = ok[ok["repo"].isin(both)]
    ok["edges_per_100_loc"] = 100 * ok["added_edges"] / ok["loc_changed"].where(ok["loc_changed"] > 0)
    return ok


def rq1(d: pd.DataFrame) -> pd.DataFrame:
    a, h = d[d["group"] == "agent"], d[d["group"] == "human"]
    rows = [{"metric": m, **compare_two(a[m], h[m])} for m in RQ1_CONTINUOUS + ["edges_per_100_loc"]]
    r = compare_rates(int(a["new_cycle"].sum()), len(a), int(h["new_cycle"].sum()), len(h))
    rows.append({"metric": "new_cycle (rate)", **r})
    out = pd.DataFrame(rows)
    # Holm over the RQ1 family (edges_per_100_loc is a robustness variant, kept outside the family).
    fam = out["metric"] != "edges_per_100_loc"
    out.loc[fam, "p_holm"] = holm(out.loc[fam, "p"].tolist())
    return out


def size_strata(d: pd.DataFrame) -> pd.DataFrame:
    s = d[d["loc_changed"].notna()].copy()
    s["size_q"] = pd.qcut(s["loc_changed"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    rows = []
    for q, g in s.groupby("size_q", observed=True):
        a, h = g[g["group"] == "agent"], g[g["group"] == "human"]
        rows.append({"stratum": q, "loc_range": f"{int(g['loc_changed'].min())}-{int(g['loc_changed'].max())}",
                     "metric": "added_edges", **compare_two(a["added_edges"], h["added_edges"])})
        rows.append({"stratum": q, "loc_range": f"{int(g['loc_changed'].min())}-{int(g['loc_changed'].max())}",
                     "metric": "new_cycle (rate)",
                     **compare_rates(int(a["new_cycle"].sum()), len(a), int(h["new_cycle"].sum()), len(h))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].tolist())
    return out


def repo_paired(d: pd.DataFrame, min_each: int = 3) -> pd.DataFrame:
    rows = []
    for m in ["added_edges", "edges_per_file", "new_cycle"]:
        per = d.groupby(["repo", "group"])[m].agg(["mean", "count"]).unstack("group")
        per = per[(per[("count", "agent")] >= min_each) & (per[("count", "human")] >= min_each)]
        diff = per[("mean", "agent")] - per[("mean", "human")]
        nonzero = diff[diff != 0]
        res = stats.wilcoxon(nonzero) if len(nonzero) >= 5 else None
        rows.append({"metric": m, "repos": len(per), "agent_higher": int((diff > 0).sum()),
                     "human_higher": int((diff < 0).sum()), "ties": int((diff == 0).sum()),
                     "median_diff": float(diff.median()) if len(diff) else np.nan,
                     "W": float(res.statistic) if res else np.nan, "p": float(res.pvalue) if res else np.nan})
    return pd.DataFrame(rows)


def without_largest(d: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    top = d[d["group"] == "agent"]["repo"].value_counts().index[0]
    rest = d[d["repo"] != top]
    a, h = rest[rest["group"] == "agent"], rest[rest["group"] == "human"]
    rows = [{"metric": "added_edges", **compare_two(a["added_edges"], h["added_edges"])},
            {"metric": "edges_per_file", **compare_two(a["edges_per_file"], h["edges_per_file"])},
            {"metric": "new_cycle (rate)",
             **compare_rates(int(a["new_cycle"].sum()), len(a), int(h["new_cycle"].sum()), len(h))}]
    return top, pd.DataFrame(rows)


def rq2(d: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    e = d[d["rq2_eligible"] == True]  # noqa: E712  (column holds True/False/NaN)
    groups = e.groupby("repo")["group"].agg(set)
    e = e[e["repo"].isin(groups[groups.apply(lambda s: {"agent", "human"} <= s)].index)]
    a, h = e[e["group"] == "agent"], e[e["group"] == "human"]
    info = {"repos": int(e["repo"].nunique()),
            "by_source": e.drop_duplicates("repo")["layer_source"].value_counts().to_dict()}
    rows = [{"metric": "has_violation (rate)",
             **compare_rates(int(a["has_violation"].sum()), len(a), int(h["has_violation"].sum()), len(h))},
            {"metric": "new_violations", **compare_two(a["new_violations"], h["new_violations"])}]
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].tolist())
    return info, out


def rq3(d: pd.DataFrame, min_prs: int) -> tuple[pd.DataFrame, dict, dict]:
    a = d[d["group"] == "agent"]
    counts = a["agent"].value_counts()
    kept = counts[counts >= min_prs].index
    a = a[a["agent"].isin(kept)].copy()
    rows = []
    for name, g in a.groupby("agent"):
        lo, hi = wilson(int(g["new_cycle"].sum()), len(g))
        v = g[g["rq2_eligible"] == True]  # noqa: E712
        vk, vn = int(v["has_violation"].sum()), len(v)
        rows.append({"agent": name, "prs": len(g), "repos": g["repo"].nunique(),
                     "median_loc": float(g["loc_changed"].median()),
                     "share_added_edges": float((g["added_edges"] > 0).mean()),
                     "median_added_edges": float(g["added_edges"].median()),
                     "mean_added_edges": float(g["added_edges"].mean()),
                     "mean_edges_per_file": float(g["edges_per_file"].mean()),
                     "new_cycle_k": int(g["new_cycle"].sum()), "new_cycle_rate": float(g["new_cycle"].mean()),
                     "new_cycle_ci_low": lo, "new_cycle_ci_high": hi,
                     "violation_k": vk, "violation_n": vn, "violation_rate": vk / vn if vn else np.nan})
    table = pd.DataFrame(rows).sort_values("prs", ascending=False)

    tests, dunn = {}, {}
    for m in ["added_edges", "edges_per_file", "edges_per_100_loc", "loc_changed"]:
        res, posthoc = kruskal_dunn(a.dropna(subset=[m]), m, "agent")
        tests[m] = res
        dunn[m] = posthoc.round(4).to_dict()
    ct = pd.crosstab(a["agent"], a["new_cycle"])
    if ct.shape[1] == 2:
        chi = stats.chi2_contingency(ct)
        tests["new_cycle (rate)"] = {"chi2": float(chi.statistic), "p": float(chi.pvalue), "dof": int(chi.dof),
                                     "min_expected": float(chi.expected_freq.min())}
    p = holm([tests[k]["p"] for k in ["added_edges", "edges_per_file"]] +
             ([tests["new_cycle (rate)"]["p"]] if "new_cycle (rate)" in tests else []))
    for k, pv in zip(["added_edges", "edges_per_file", "new_cycle (rate)"], p):
        tests[k]["p_holm"] = pv
    excluded = {k: int(v) for k, v in counts[counts < min_prs].items()}
    return table, tests, {"dunn_holm": dunn, "excluded_agents": excluded}


def describe(d: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grp, g in d.groupby("group"):
        rows.append({"group": grp, "repos": g["repo"].nunique(), "prs": len(g),
                     "median_loc": float(g["loc_changed"].median()), "median_files": float(g["files_changed"].median()),
                     "median_py_files": float(g["py_files_changed"].median()),
                     "median_modules_before": float(g["modules_before"].median()),
                     "share_merge_commit": float((g["merge_style"] == "merge").mean())})
    return pd.DataFrame(rows)


def main() -> int:
    cfg = settings()
    df = pd.read_csv(path("processed") / "pr_metrics.csv", low_memory=False)
    d = analysis_sample(df)
    tables = path("tables")

    status = df.groupby(["group", "status"]).size().unstack(fill_value=0)
    status.to_csv(tables / "pr_status_by_group.csv")
    t1 = describe(d, df)
    t1.to_csv(tables / "table1_data.csv", index=False)
    t2 = rq1(d)
    t2.to_csv(tables / "table2_rq1.csv", index=False)
    strata = size_strata(d)
    strata.to_csv(tables / "robust_size_strata.csv", index=False)
    paired = repo_paired(d)
    paired.to_csv(tables / "robust_repo_paired.csv", index=False)
    top, nolargest = without_largest(d)
    nolargest.to_csv(tables / "robust_without_largest_repo.csv", index=False)
    rq2_info, t_rq2 = rq2(d)
    t_rq2.to_csv(tables / "table_rq2.csv", index=False)
    t3, rq3_tests, rq3_extra = rq3(d, cfg["filter"]["min_prs_per_agent"])
    t3.to_csv(tables / "table3_rq3.csv", index=False)

    summary = {
        "analysis_sample": {"prs": len(d), "repos": int(d["repo"].nunique()),
                            "agent": int((d["group"] == "agent").sum()), "human": int((d["group"] == "human").sum())},
        "largest_agent_repo": top,
        "rq2": rq2_info,
        "rq3_tests": rq3_tests,
        **rq3_extra,
    }
    (tables / "stats_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for name, t in [("Table 1", t1), ("RQ1", t2), ("Size strata", strata), ("Repo paired", paired),
                    (f"Without {top}", nolargest), ("RQ2", t_rq2), ("RQ3", t3)]:
        print(f"\n== {name} ==\n{t.round(4).to_string(index=False)}")
    print(json.dumps({k: v for k, v in summary.items() if k != "dunn_holm"}, indent=1, default=str))
    log_run("06_statistics.py", {"alpha": cfg["stats"]["alpha"], "min_prs_per_agent": cfg["filter"]["min_prs_per_agent"]},
            summary["analysis_sample"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
