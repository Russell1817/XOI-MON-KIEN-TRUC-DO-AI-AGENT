"""Step 06: statistical analysis for RQ1-RQ3 and robustness checks.

Analysis sample: PRs with status 'ok' in repositories that have at least one analysed agent PR
and one analysed human PR, so that every comparison is between PRs of the same repositories.

Most PRs add no internal dependency, so RQ1 is reported in two parts: how often a PR adds an
edge (Fisher's exact test, odds ratio) and how many edges it adds when it does (Mann-Whitney U,
Cliff's delta). The cycle indicator is split into "a package joins a cycle" and "an edge is added
inside a cycle whose packages were already cyclic".
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from aierosion.config import path, settings
from aierosion.runlog import log_run
from aierosion.stats.tests import compare_rates, compare_two, holm, kruskal_dunn, wilson

# RQ1 family for the Holm correction: (name, column, kind). "rate" = share of PRs with value > 0.
RQ1_FAMILY = [
    ("adds_edge", "added_edges", "rate"),
    ("edges_if_any", "added_edges", "positive"),
    ("adds_pkg_edge", "added_pkg_edges", "rate"),
    ("cycle_new_pkgs", "cycle_new_pkgs", "rate"),
    ("cycle_existing_only", "cycle_existing_only", "rate"),
    ("d_ce_mean", "d_ce_mean", "continuous"),
    ("d_i_core", "d_i_core", "continuous"),
]
# Reported next to the family, not part of it.
RQ1_EXTRA = [
    ("new_cycle", "new_cycle", "rate"),
    ("added_edges", "added_edges", "continuous"),
    ("edges_per_100_loc", "edges_per_100_loc", "continuous"),
]


def analysis_sample(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    groups = ok.groupby("repo")["group"].agg(set)
    both = groups[groups.apply(lambda s: {"agent", "human"} <= s)].index
    ok = ok[ok["repo"].isin(both)]
    ok["edges_per_100_loc"] = 100 * ok["added_edges"] / ok["loc_changed"].where(ok["loc_changed"] > 0)
    return ok


def _compare(a: pd.DataFrame, h: pd.DataFrame, col: str, kind: str) -> dict:
    if kind == "rate":
        return compare_rates(int((a[col] > 0).sum()), len(a), int((h[col] > 0).sum()), len(h))
    if kind == "positive":
        return compare_two(a.loc[a[col] > 0, col], h.loc[h[col] > 0, col])
    return compare_two(a[col], h[col])


def rq1(d: pd.DataFrame) -> pd.DataFrame:
    a, h = d[d["group"] == "agent"], d[d["group"] == "human"]
    rows = [{"metric": name, "kind": kind, "family": True, **_compare(a, h, col, kind)}
            for name, col, kind in RQ1_FAMILY]
    rows += [{"metric": name, "kind": kind, "family": False, **_compare(a, h, col, kind)}
             for name, col, kind in RQ1_EXTRA]
    out = pd.DataFrame(rows)
    fam = out["family"]
    out.loc[fam, "p_holm"] = holm(out.loc[fam, "p"].tolist())
    return out


def _size_quartiles(d: pd.DataFrame) -> pd.DataFrame:
    s = d[d["loc_changed"].notna()].copy()
    # Quartile bounds are computed on the whole analysis sample (both groups together).
    s["size_q"] = pd.qcut(s["loc_changed"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return s


def size_strata(d: pd.DataFrame) -> pd.DataFrame:
    s = _size_quartiles(d)
    rows = []
    for q, g in s.groupby("size_q", observed=True):
        a, h = g[g["group"] == "agent"], g[g["group"] == "human"]
        rng = f"{int(g['loc_changed'].min())}-{int(g['loc_changed'].max())}"
        for name, col in (("adds_edge", "added_edges"), ("cycle_new_pkgs", "cycle_new_pkgs")):
            rows.append({"stratum": q, "loc_range": rng, "metric": name, **_compare(a, h, col, "rate")})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].tolist())
    return out


def repo_paired(d: pd.DataFrame, min_each: int = 3) -> pd.DataFrame:
    d = d.assign(adds_edge=(d["added_edges"] > 0).astype(float))
    rows = []
    for m in ["adds_edge", "added_edges", "cycle_new_pkgs"]:
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


def _subset_rates(d: pd.DataFrame, label: str) -> list[dict]:
    a, h = d[d["group"] == "agent"], d[d["group"] == "human"]
    return [{"subset": label, "metric": name, **_compare(a, h, col, "rate")}
            for name, col in (("adds_edge", "added_edges"), ("cycle_new_pkgs", "cycle_new_pkgs"),
                              ("cycle_existing_only", "cycle_existing_only"))]


def subsets(d: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Sensitivity: without the repository with most agent PRs, and without OpenAI Codex PRs."""
    top = d[d["group"] == "agent"]["repo"].value_counts().index[0]
    rows = _subset_rates(d[d["repo"] != top], "without_largest_repo")
    no_codex = d[~((d["group"] == "agent") & (d["agent"] == "OpenAI_Codex"))]
    groups = no_codex.groupby("repo")["group"].agg(set)
    no_codex = no_codex[no_codex["repo"].isin(groups[groups.apply(lambda s: {"agent", "human"} <= s)].index)]
    rows += _subset_rates(no_codex, "without_codex")
    return top, pd.DataFrame(rows)


def graph_variants(d: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity to which imports make an edge: all imports, or module-level imports only."""
    a, h = d[d["group"] == "agent"], d[d["group"] == "human"]
    rows = []
    for variant, suffix in (("runtime (main)", ""), ("all imports", "_all"), ("module level only", "_top")):
        for name, col in (("adds_edge", "added_edges"), ("cycle_new_pkgs", "cycle_new_pkgs"),
                          ("new_cycle", "new_cycle")):
            rows.append({"variant": variant, "metric": name, **_compare(a, h, col + suffix, "rate")})
    return pd.DataFrame(rows)


def contamination(d: pd.DataFrame) -> dict:
    """Bound the effect of agent PRs hidden in the human group (screened sample, step 05c).

    With a share c of the human group made with an agent, the observed human rate is
    (1 - c) r_h + c r_a, so r_h = (r_obs - c r_a) / (1 - c).
    """
    f = path("manual_check") / "human_screen.csv"
    if not f.exists():
        return {}
    s = pd.read_csv(f)
    s = s[s["fetched"] == True]  # noqa: E712
    k, n = int((s["agent_involved"] == "yes").sum()), len(s)
    lo, hi = wilson(k, n)
    a, h = d[d["group"] == "agent"], d[d["group"] == "human"]
    out = {"screened": n, "agent_involved": k, "share": k / n, "share_ci": [lo, hi], "metrics": {}}
    for name, col in (("adds_edge", "added_edges"), ("cycle_new_pkgs", "cycle_new_pkgs")):
        ra, rh = float((a[col] > 0).mean()), float((h[col] > 0).mean())
        out["metrics"][name] = {"rate_agent": ra, "rate_human_observed": rh,
                                **{f"rate_human_corrected_{lab}": (rh - c * ra) / (1 - c)
                                   for lab, c in (("point", k / n), ("upper", hi))}}
    return out


def regression(d: pd.DataFrame) -> pd.DataFrame:
    """Logistic GEE, PRs clustered by repository (exchangeable working correlation).

    Model 1 controls PR size, merge style, merge date and repository size at the time of the PR.
    Model 2 adds the number of packages the PR touches (its spread).
    For RQ3, model 1 is also fitted with the agent term split into Claude Code and the other agents,
    to see whether the high C_new rate of Claude Code remains once PR size is controlled.
    """
    m = d.copy()
    m["agent_pr"] = (m["group"] == "agent").astype(int)
    m["claude_code"] = (m["agent"] == "Claude_Code").astype(int) * m["agent_pr"]
    m["other_agent"] = m["agent_pr"] - m["claude_code"]
    m["adds_edge"] = (m["added_edges"] > 0).astype(int)
    m["log_loc"] = np.log1p(m["loc_changed"])
    m["log_files"] = np.log1p(m["files_changed"])
    m["merge_commit"] = (m["merge_style"] == "merge").astype(int)
    t = pd.to_datetime(m["merged_at"], utc=True, errors="coerce")
    m["years"] = (t - t.min()).dt.days / 365.25
    m["log_modules"] = np.log1p(m["modules_before"])
    m["log_pkgs_touched"] = np.log1p(m["pkgs_touched"])
    base = "agent_pr + log_loc + log_files + merge_commit + years + log_modules"
    by_agent = base.replace("agent_pr", "claude_code + other_agent")
    rows = []
    for outcome in ("adds_edge", "cycle_new_pkgs"):
        specs = [("model1", base), ("model2", base + " + log_pkgs_touched")]
        if outcome == "cycle_new_pkgs":
            specs.append(("model1_by_agent", by_agent))
        for model, rhs in specs:
            cols = ["repo", outcome] + [c.strip() for c in rhs.split("+")]
            data = m[cols].dropna()
            # Exchangeable working correlation; rare outcomes may not converge with it, then the
            # independence structure is used (standard errors stay cluster-robust either way).
            for cov in (sm.cov_struct.Exchangeable(), sm.cov_struct.Independence()):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = smf.gee(f"{outcome} ~ {rhs}", "repo", data, family=sm.families.Binomial(),
                                  cov_struct=cov).fit()
                if np.isfinite(fit.params).all() and np.isfinite(fit.bse).all():
                    break
            ci = fit.conf_int()
            for term in fit.params.index:
                if term == "Intercept":
                    continue
                rows.append({"outcome": outcome, "model": model, "term": term, "n": int(fit.nobs),
                             "clusters": int(data["repo"].nunique()), "cov_struct": type(cov).__name__,
                             "coef": float(fit.params[term]),
                             "odds_ratio": float(np.exp(fit.params[term])),
                             "or_low": float(np.exp(ci.loc[term, 0])), "or_high": float(np.exp(ci.loc[term, 1])),
                             "p": float(fit.pvalues[term])})
    return pd.DataFrame(rows)


def spread(d: pd.DataFrame) -> pd.DataFrame:
    """Packages touched per PR, overall and in the largest size quartile (discussion hypothesis)."""
    s = _size_quartiles(d)
    rows = []
    for label, g in (("all", s), ("Q4", s[s["size_q"] == "Q4"])):
        a, h = g[g["group"] == "agent"], g[g["group"] == "human"]
        rows.append({"subset": label, **compare_two(a["pkgs_touched"], h["pkgs_touched"])})
    return pd.DataFrame(rows)


def selection(df: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """Where the agent PRs of the step-02 selection end up, and their size as recorded in AIDev."""
    final = pd.read_csv(path("interim") / "agent_prs.csv")
    final = final.rename(columns={"full_name": "repo", "loc_changed": "aidev_loc"})
    mined = df[df["group"] == "agent"][["repo", "number", "status"]]
    x = final.merge(mined, on=["repo", "number"], how="left")
    analysed = set(zip(d.loc[d["group"] == "agent", "repo"], d.loc[d["group"] == "agent", "number"]))
    x["category"] = np.where([(r, n) in analysed for r, n in zip(x["repo"], x["number"])], "analysed",
                             np.where(x["status"] == "ok", "no_human_in_repo", x["status"].fillna("repo_unavailable")))
    rows = []
    for cat, g in x.groupby("category"):
        rows.append({"category": cat, "prs": len(g), "repos": g["repo"].nunique(),
                     "median_aidev_loc": float(g["aidev_loc"].median()),
                     "share_codex": float((g["agent"] == "OpenAI_Codex").mean())})
    out = pd.DataFrame(rows)
    inc, exc = x[x["category"] == "analysed"]["aidev_loc"].dropna(), x[x["category"] != "analysed"]["aidev_loc"].dropna()
    res = compare_two(inc, exc)
    out.attrs["test"] = {"median_analysed": res["median_agent"], "median_excluded": res["median_human"],
                         "p": res["p"], "cliffs_delta": res["cliffs_delta"]}
    by_agent = x.groupby("agent")["category"].apply(lambda c: (c == "analysed").mean()).to_dict()
    out.attrs["analysed_share_by_agent"] = by_agent
    return out


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
        k = int(g["cycle_new_pkgs"].sum())
        lo, hi = wilson(k, len(g))
        v = g[g["rq2_eligible"] == True]  # noqa: E712
        vk, vn = int(v["has_violation"].sum()), len(v)
        rows.append({"agent": name, "prs": len(g), "repos": g["repo"].nunique(),
                     "median_loc": float(g["loc_changed"].median()),
                     "share_added_edges": float((g["added_edges"] > 0).mean()),
                     "median_added_edges": float(g["added_edges"].median()),
                     "mean_added_edges": float(g["added_edges"].mean()),
                     "mean_edges_per_file": float(g["edges_per_file"].mean()),
                     "new_cycle_k": int(g["new_cycle"].sum()), "new_cycle_rate": float(g["new_cycle"].mean()),
                     "cycle_new_pkgs_k": k, "cycle_new_pkgs_rate": k / len(g),
                     "cycle_new_pkgs_ci_low": lo, "cycle_new_pkgs_ci_high": hi,
                     "violation_k": vk, "violation_n": vn, "violation_rate": vk / vn if vn else np.nan})
    table = pd.DataFrame(rows).sort_values("prs", ascending=False)

    tests, dunn = {}, {}
    for m in ["added_edges", "edges_per_file", "edges_per_100_loc", "loc_changed"]:
        res, posthoc = kruskal_dunn(a.dropna(subset=[m]), m, "agent")
        tests[m] = res
        dunn[m] = posthoc.round(4).to_dict()
    ct = pd.crosstab(a["agent"], a["cycle_new_pkgs"])
    if ct.shape[1] == 2:
        chi = stats.chi2_contingency(ct)
        tests["cycle_new_pkgs (rate)"] = {"chi2": float(chi.statistic), "p": float(chi.pvalue), "dof": int(chi.dof),
                                          "min_expected": float(chi.expected_freq.min())}
    fam = [k for k in ["added_edges", "edges_per_file", "cycle_new_pkgs (rate)"] if k in tests]
    for k, pv in zip(fam, holm([tests[k]["p"] for k in fam])):
        tests[k]["p_holm"] = pv
    excluded = {k: int(v) for k, v in counts[counts < min_prs].items()}
    return table, tests, {"dunn_holm": dunn, "excluded_agents": excluded}


def describe(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grp, g in d.groupby("group"):
        rows.append({"group": grp, "repos": g["repo"].nunique(), "prs": len(g),
                     "median_loc": float(g["loc_changed"].median()), "median_files": float(g["files_changed"].median()),
                     "median_py_files": float(g["py_files_changed"].median()),
                     "median_modules_before": float(g["modules_before"].median()),
                     "median_pkgs_touched": float(g["pkgs_touched"].median()),
                     "share_merge_commit": float((g["merge_style"] == "merge").mean())})
    return pd.DataFrame(rows)


def main() -> int:
    cfg = settings()
    df = pd.read_csv(path("processed") / "pr_metrics.csv", low_memory=False)
    d = analysis_sample(df)
    tables = path("tables")

    status = df.groupby(["group", "status"]).size().unstack(fill_value=0)
    status.to_csv(tables / "pr_status_by_group.csv")
    t1 = describe(d)
    t1.to_csv(tables / "table1_data.csv", index=False)
    t2 = rq1(d)
    t2.to_csv(tables / "table2_rq1.csv", index=False)
    strata = size_strata(d)
    strata.to_csv(tables / "robust_size_strata.csv", index=False)
    paired = repo_paired(d)
    paired.to_csv(tables / "robust_repo_paired.csv", index=False)
    top, subs = subsets(d)
    subs.to_csv(tables / "robust_subsets.csv", index=False)
    variants = graph_variants(d)
    variants.to_csv(tables / "robust_graph_variants.csv", index=False)
    reg = regression(d)
    reg.to_csv(tables / "regression_gee.csv", index=False)
    spr = spread(d)
    spr.to_csv(tables / "spread_pkgs_touched.csv", index=False)
    sel = selection(df, d)
    sel.to_csv(tables / "selection_agent_prs.csv", index=False)
    cont = contamination(d)
    rq2_info, t_rq2 = rq2(d)
    t_rq2.to_csv(tables / "table_rq2.csv", index=False)
    t3, rq3_tests, rq3_extra = rq3(d, cfg["filter"]["min_prs_per_agent"])
    t3.to_csv(tables / "table3_rq3.csv", index=False)

    summary = {
        "analysis_sample": {"prs": len(d), "repos": int(d["repo"].nunique()),
                            "agent": int((d["group"] == "agent").sum()), "human": int((d["group"] == "human").sum())},
        "largest_agent_repo": top,
        "rq1_holm_family": [name for name, _, _ in RQ1_FAMILY],
        "selection": {**sel.attrs["test"], "analysed_share_by_agent": sel.attrs["analysed_share_by_agent"]},
        "contamination": cont,
        "rq2": rq2_info,
        "rq3_tests": rq3_tests,
        **rq3_extra,
    }
    (tables / "stats_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    for name, t in [("Table 1", t1), ("RQ1", t2), ("Size strata", strata), ("Repo paired", paired),
                    ("Subsets", subs), ("Graph variants", variants), ("GEE (agent term)", reg[reg["term"] == "agent_pr"]),
                    ("Spread", spr), ("Selection", sel), ("RQ2", t_rq2), ("RQ3", t3)]:
        print(f"\n== {name} ==\n{t.round(4).to_string(index=False)}")
    print(json.dumps({k: v for k, v in summary.items() if k != "dunn_holm"}, indent=1, default=str))
    log_run("06_statistics.py", {"alpha": cfg["stats"]["alpha"], "min_prs_per_agent": cfg["filter"]["min_prs_per_agent"]},
            summary["analysis_sample"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
