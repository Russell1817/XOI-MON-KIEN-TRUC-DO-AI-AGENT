"""Step 08: collect every number quoted in the report and the paper into results/tables/numbers.json.

Values are formatted the Vietnamese way (decimal comma, dot as thousands separator). The Markdown
sources of the paper and the report use {{KEY}} placeholders filled from this file, so every number in
the documents comes from the same analysis run.
"""
from __future__ import annotations

import json
import math
import sys

import pandas as pd

from aierosion.config import path
from aierosion.runlog import log_run

AGENT_VI = {"OpenAI_Codex": "OpenAI Codex", "Copilot": "GitHub Copilot", "Claude_Code": "Claude Code",
            "Cursor": "Cursor", "Devin": "Devin", "Google_Jules": "Google Jules"}


def n(x: float) -> str:
    """Integer with dot thousands separator."""
    return f"{int(round(x)):,}".replace(",", ".")


def d(x: float, k: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{k}f}".replace(".", ",").replace("-", "−")  # typographic minus sign


# Percentages: one decimal place; two for the rare-outcome metrics (human rate below 10%: package edges, C_new,
# C_exist, layering violations) wherever they appear, including their confidence intervals.
def pct(x: float, k: int = 1) -> str:
    return d(100 * x, k) + "%"


def p(x: float) -> str:
    """p-value with its relation sign, for prose: "= 0,137" or "< 0,001"."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if x < 0.001:
        return "< 0,001"
    return "= " + d(x, 3)


def rng(text: str) -> str:
    """'286-186659' -> '286–186.659'."""
    a, b = text.split("-")
    return f"{n(float(a))}–{n(float(b))}"


def pt(x: float) -> str:
    """p-value for a table cell (no sign unless it is a bound)."""
    return p(x).removeprefix("= ")


def main() -> int:
    t = path("tables")
    s = json.loads((t / "stats_summary.json").read_text(encoding="utf-8"))
    t1 = pd.read_csv(t / "table1_data.csv").set_index("group")
    rq1 = pd.read_csv(t / "table2_rq1.csv").set_index("metric")
    strata = pd.read_csv(t / "robust_size_strata.csv")
    paired = pd.read_csv(t / "robust_repo_paired.csv").set_index("metric")
    subs = pd.read_csv(t / "robust_subsets.csv")
    variants = pd.read_csv(t / "robust_graph_variants.csv")
    reg = pd.read_csv(t / "regression_gee.csv")
    spread = pd.read_csv(t / "spread_pkgs_touched.csv")
    selection = pd.read_csv(t / "selection_agent_prs.csv")
    rq2 = pd.read_csv(t / "table_rq2.csv").set_index("metric")
    rq3 = pd.read_csv(t / "table3_rq3.csv")
    funnel = pd.read_csv(path("logs") / "filter_funnel.csv")
    mining = pd.read_csv(path("logs") / "mining_repos.csv")
    metrics = pd.read_csv(path("processed") / "pr_metrics.csv", low_memory=False)
    manual_file = path("manual_check") / "manual_check_team.csv"
    manual = (pd.read_csv(manual_file, dtype=str, keep_default_na=False) if manual_file.exists()
              else pd.DataFrame(columns=["rater1_match", "rater2_match", "consensus_match"]))
    review_file = path("manual_check") / "violations_review.csv"
    review = pd.read_csv(review_file) if review_file.exists() and review_file.stat().st_size > 2 else pd.DataFrame()
    if len(review):
        # verdicts are read back as booleans by pandas; normalise to "true"/"false"
        review["verdict"] = review["verdict"].astype(str).str.strip().str.lower()

    v: dict[str, str] = {}
    funnel_labels = ["AIDev-pop ban đầu", "Ngôn ngữ chính Python, không phải bản sao (fork) của kho khác",
                     "Giấy phép mã nguồn mở cho phép dùng lại", "PR đã merge",
                     "PR sửa ít nhất một file Python mã nguồn", "Kho có ít nhất 3 PR như trên"]
    assert len(funnel) == len(funnel_labels), "filter_funnel.csv steps changed; update the labels"
    rows = ["| Bước lọc | Số PR agent | Số kho |", "|---|---|---|"]
    rows += [f"| {lab} | {n(r['agent_prs'])} | {n(r['repos'])} |" for lab, (_, r) in zip(funnel_labels, funnel.iterrows())]
    v["TABLE_FUNNEL"] = "\n".join(rows)
    final = pd.read_csv(path("interim") / "agent_prs.csv")
    v["LICENSE_EXCLUDED_REPOS"] = n(funnel["repos"].iloc[1] - funnel["repos"].iloc[2])
    v["FUNNEL_BY_AGENT"] = "; ".join(f"{AGENT_VI.get(a, a)} {n(c)}" for a, c in final["agent"].value_counts().items())
    a_all = metrics[metrics["group"] == "agent"]
    ok = metrics[metrics["status"] == "ok"]
    v |= {
        "FUNNEL_FINAL_PR": n(funnel["agent_prs"].iloc[-1]), "FUNNEL_FINAL_REPO": n(funnel["repos"].iloc[-1]),
        "REPO_CLONED": n((mining["status"] == "ok").sum()),
        # Mining effort of the final run (results/logs/mining_repos.csv): per-repository wall time and clone size.
        "MINING_HOURS": d(mining["seconds"].sum() / 3600, 1), "MINING_MED_S": d(mining["seconds"].median(), 0),
        "MINING_MAX_S": n(round(mining["seconds"].max())), "MINING_MAX_REPO": mining.loc[mining["seconds"].idxmax(), "repo"],
        "CLONE_GB": d(mining["clone_mb"].sum() / 1024, 1),
        "REPO_DELETED": n(mining.get("error", pd.Series(dtype=str)).fillna("").astype(str).str.contains("Repository not found").sum()),
        "AGENT_IN_REPOS": n(len(a_all)),
        "AGENT_NOT_FOUND": n((a_all["status"] == "not_on_first_parent").sum()),
        "AGENT_NOT_FOUND_PCT": pct((a_all["status"] == "not_on_first_parent").mean()),
        "AGENT_NO_PY": n((a_all["status"] == "no_py_source_change").sum()),
        "AGENT_OK": n((a_all["status"] == "ok").sum()),
        "HUMAN_OK": n(((metrics["group"] == "human") & (metrics["status"] == "ok")).sum()),
        "N_PR": n(s["analysis_sample"]["prs"]), "N_AGENT": n(s["analysis_sample"]["agent"]),
        "N_HUMAN": n(s["analysis_sample"]["human"]), "N_REPO": n(s["analysis_sample"]["repos"]),
        "LARGEST_REPO": s["largest_agent_repo"],
        "PARSE_ERR_PRS": n((ok["parse_errors"] > 0).sum()), "MISSING_BLOB_PRS": n((ok["missing_blobs"] > 0).sum()),
        "PARSE_ERR_REPOS": n(ok[ok["parse_errors"] > 0]["repo"].nunique()),
        "MISSING_BLOB_REPOS": n(ok[ok["missing_blobs"] > 0]["repo"].nunique()),
        "RENAME_PRS": n((ok.get("renamed_py", pd.Series(0, index=ok.index)).fillna(0) > 0).sum()),
        "RENAME_UNAVAIL": n((ok.get("rename_detection", pd.Series("ok", index=ok.index)) == "unavailable").sum()),
    }
    # Share of agent PRs not found on the first-parent chain, per agent and in the two largest sources.
    nf = a_all[a_all["status"] == "not_on_first_parent"]["repo"].value_counts()
    top2 = [f"{r} ({n(c)})" for r, c in nf.head(2).items()]
    v["NOT_FOUND_TOP2"] = " và ".join(top2)
    v["NOT_FOUND_TOP2_EN_LIST"] = " and ".join(top2)
    found = a_all.groupby("agent")["status"].apply(lambda x: (x != "not_on_first_parent").mean())
    v["FOUND_RATE_BY_AGENT"] = "; ".join(f"{AGENT_VI.get(k, k)} {pct(x)}" for k, x in found.sort_values().items())

    largest_n = int((ok[(ok["group"] == "agent")]["repo"] == s["largest_agent_repo"]).sum())
    v["LARGEST_REPO_AGENT_PRS"] = n(largest_n)
    v["LARGEST_REPO_SHARE"] = pct(largest_n / s["analysis_sample"]["agent"])

    for g, key in (("agent", "A"), ("human", "H")):
        r = t1.loc[g]
        v[f"T1_{key}_REPOS"], v[f"T1_{key}_PRS"] = n(r["repos"]), n(r["prs"])
        v[f"T1_{key}_LOC"], v[f"T1_{key}_FILES"] = d(r["median_loc"], 0), d(r["median_files"], 0)
        v[f"T1_{key}_PYFILES"], v[f"T1_{key}_MODULES"] = d(r["median_py_files"], 0), d(r["median_modules_before"], 0)
        v[f"T1_{key}_MERGE"] = pct(r["share_merge_commit"])

    # RQ1 table: how often a PR adds structure (share of PRs, Fisher, odds ratio) and how much it adds
    # when it does (Mann-Whitney U, Cliff's delta).
    def k_of(r):
        return 2 if r["rate_human"] < 0.1 else 1

    rq1_rows = [("adds_edge", "% PR thêm ≥ 1 cạnh"), ("edges_if_any", "Số cạnh mới khi có thêm: trung vị; TB"),
                ("adds_pkg_edge", "% PR thêm ≥ 1 cạnh package"),
                ("cycle_new_pkgs", "% PR đưa package mới vào chu trình"),
                ("cycle_existing_only", "% PR chỉ thêm cạnh trong chu trình có sẵn"),
                ("d_ce_mean", "Δ Ce: TB"), ("d_i_core", "Δ I package lõi: TB")]
    rows = ["| Chỉ số | Agent | Người | Hiệu ứng | *p* (Holm) |", "|---|---|---|---|---|"]
    for m, lab in rq1_rows:
        r = rq1.loc[m]
        if r["kind"] == "rate":
            ca = f"{pct(r['rate_agent'], k_of(r))} ({n(r['k_agent'])}/{n(r['n_agent'])})"
            ch = f"{pct(r['rate_human'], k_of(r))} ({n(r['k_human'])}/{n(r['n_human'])})"
            eff = f"OR = {d(r['odds_ratio'])}"
        elif r["kind"] == "positive":
            ca = f"{d(r['median_agent'], 0)}; {d(r['mean_agent'], 2)}"
            ch = f"{d(r['median_human'], 0)}; {d(r['mean_human'], 2)}"
            eff = f"δ = {d(r['cliffs_delta'], 3)}"
        else:
            ca, ch, eff = d(r["mean_agent"], 3), d(r["mean_human"], 3), f"δ = {d(r['cliffs_delta'], 3)}"
        rows.append(f"| {lab} | {ca} | {ch} | {eff} | {pt(r['p_holm'])} |")
    v["TABLE_RQ1"] = "\n".join(rows)
    v["RQ1_FAMILY_N"] = n(len(s["rq1_holm_family"]))
    for m, key in (("adds_edge", "EDGES"), ("adds_pkg_edge", "PKGE"), ("cycle_new_pkgs", "CNP"),
                   ("cycle_existing_only", "CEX"), ("new_cycle", "CYC")):
        r = rq1.loc[m]
        v |= {f"RQ1_{key}_A": pct(r["rate_agent"], k_of(r)), f"RQ1_{key}_H": pct(r["rate_human"], k_of(r)),
              f"RQ1_{key}_KA": n(r["k_agent"]), f"RQ1_{key}_KH": n(r["k_human"]),
              f"RQ1_{key}_OR": d(r["odds_ratio"]), f"RQ1_{key}_P": p(r["p_holm"] if r["family"] else r["p"])}
    for m, key in (("edges_if_any", "EIA"), ("d_ce_mean", "DCE"), ("d_i_core", "DI"), ("added_edges", "ALL"),
                   ("edges_per_100_loc", "E100")):
        r = rq1.loc[m]
        v |= {f"RQ1_{key}_MED_A": d(r["median_agent"], 0), f"RQ1_{key}_MED_H": d(r["median_human"], 0),
              f"RQ1_{key}_A": d(r["mean_agent"], 2), f"RQ1_{key}_H": d(r["mean_human"], 2),
              f"RQ1_{key}_DELTA": d(r["cliffs_delta"], 3), f"RQ1_{key}_P": p(r["p_holm"] if r["family"] else r["p"])}
    # Of the PRs flagged by the combined indicator, the share in which a package joins a cycle.
    cyc, cnp = rq1.loc["new_cycle"], rq1.loc["cycle_new_pkgs"]
    v |= {"CYC_TOTAL": n(cyc["k_agent"] + cyc["k_human"]), "CNP_TOTAL": n(cnp["k_agent"] + cnp["k_human"]),
          "CNP_SHARE_OF_CYC": pct((cnp["k_agent"] + cnp["k_human"]) / (cyc["k_agent"] + cyc["k_human"]))}

    rows = ["| Nhóm quy mô (số dòng) | Chỉ số | Agent | Người | OR | *p* (Holm) |", "|---|---|---|---|---|---|"]
    for _, r in strata.iterrows():
        first = r["metric"] == "adds_edge"
        q = f"{r['stratum']} ({rng(r['loc_range'])})" if first else ""
        lab = "% PR thêm cạnh" if first else "% PR đưa package mới vào chu trình"
        k = 1 if first else 2
        na, nh = (f" (n = {n(r['n_agent'])})", f" (n = {n(r['n_human'])})") if first else ("", "")
        # With a zero count the Haldane-corrected OR can point the wrong way (0% vs 0.08% -> OR 1.95).
        orr = "–" if 0 in (r["k_agent"], r["k_human"]) else d(r["odds_ratio"])
        rows.append(f"| {q} | {lab} | {pct(r['rate_agent'], k)}{na} | {pct(r['rate_human'], k)}{nh} | "
                    f"{orr} | {pt(r['p_holm'])} |")
    v["TABLE_STRATA"] = "\n".join(rows)
    for q in ("Q1", "Q2", "Q3", "Q4"):
        sq = strata[strata["stratum"] == q].set_index("metric")
        e, c2 = sq.loc["adds_edge"], sq.loc["cycle_new_pkgs"]
        v |= {f"{q}_RANGE": rng(e["loc_range"]),
              f"{q}_EDGES_A": pct(e["rate_agent"]), f"{q}_EDGES_H": pct(e["rate_human"]),
              f"{q}_EDGES_OR": d(e["odds_ratio"]), f"{q}_EDGES_P": p(e["p_holm"]),
              f"{q}_CNP_A": pct(c2["rate_agent"], 2), f"{q}_CNP_H": pct(c2["rate_human"], 2),
              f"{q}_CNP_OR": d(c2["odds_ratio"]), f"{q}_CNP_P": p(c2["p_holm"])}
    edge_strata = strata[strata["metric"] == "adds_edge"]
    cnp_strata = strata[strata["metric"] == "cycle_new_pkgs"]
    sig_q = list(edge_strata[edge_strata["p_holm"] < 0.05]["stratum"].astype(str))
    v["STRATA_EDGES_SIG"] = ", ".join(sig_q) if sig_q else "không nhóm nào"
    v["STRATA_EDGES_NS_MIN_P"] = p(edge_strata[edge_strata["p_holm"] >= 0.05]["p_holm"].min()) \
        if (edge_strata["p_holm"] >= 0.05).any() else "–"
    v["STRATA_CNP_MIN_P"] = p(cnp_strata["p_holm"].min())

    for m, key in (("adds_edge", "EDGES"), ("added_edges", "NEDGES"), ("cycle_new_pkgs", "CNP")):
        r = paired.loc[m]
        v |= {f"PAIR_{key}_REPOS": n(r["repos"]), f"PAIR_{key}_AH": n(r["agent_higher"]),
              f"PAIR_{key}_HH": n(r["human_higher"]), f"PAIR_{key}_TIES": n(r["ties"]), f"PAIR_{key}_P": p(r["p"])}

    for sub, key in (("without_largest_repo", "NOLARGE"), ("without_codex", "NOCODEX")):
        ss = subs[subs["subset"] == sub].set_index("metric")
        v[f"{key}_N_A"] = n(ss.loc["adds_edge", "n_agent"])
        for m, mk in (("adds_edge", "EDGES"), ("cycle_new_pkgs", "CNP"), ("cycle_existing_only", "CEX")):
            r = ss.loc[m]
            k = 1 if m == "adds_edge" else 2
            v |= {f"{key}_{mk}_A": pct(r["rate_agent"], k), f"{key}_{mk}_H": pct(r["rate_human"], k),
                  f"{key}_{mk}_OR": d(r["odds_ratio"]), f"{key}_{mk}_P": p(r["p"])}

    vlab = {"runtime (main)": "Chính: bỏ import trong TYPE_CHECKING", "all imports": "Mọi import",
            "module level only": "Chỉ import ở mức module"}
    rows = ["| Đồ thị | % PR thêm cạnh (agent; người) | OR (*p*) | % PR đưa package mới vào chu trình (agent; người) | OR (*p*) |",
            "|---|---|---|---|---|"]
    for var, g in variants.groupby("variant", sort=False):
        e, c2 = g.set_index("metric").loc["adds_edge"], g.set_index("metric").loc["cycle_new_pkgs"]
        rows.append(f"| {vlab[var]} | {pct(e['rate_agent'])}; {pct(e['rate_human'])} | {d(e['odds_ratio'])} ({pt(e['p'])}) | "
                    f"{pct(c2['rate_agent'], 2)}; {pct(c2['rate_human'], 2)} | {d(c2['odds_ratio'])} ({pt(c2['p'])}) |")
        key = {"runtime (main)": "MAIN", "all imports": "ALL", "module level only": "TOP"}[var]
        for m, mk in (("adds_edge", "EDGES"), ("cycle_new_pkgs", "CNP"), ("new_cycle", "CYC")):
            r = g.set_index("metric").loc[m]
            kv = 1 if m == "adds_edge" else 2
            v |= {f"VAR_{key}_{mk}_A": pct(r["rate_agent"], kv), f"VAR_{key}_{mk}_H": pct(r["rate_human"], kv),
                  f"VAR_{key}_{mk}_OR": d(r["odds_ratio"]), f"VAR_{key}_{mk}_P": p(r["p"])}
    v["TABLE_VARIANTS"] = "\n".join(rows)
    # Core-package threshold (step 03b): Δ I at 5%, 10% and 20%; sensitivity analysis, so unadjusted p.
    core_file = t / "robust_core_threshold.csv"
    core = pd.read_csv(core_file).set_index("fraction") if core_file.exists() else None
    if core is not None:
        for frac, key in ((0.05, "05"), (0.10, "10"), (0.20, "20")):
            r = core.loc[frac]
            v |= {f"CORE{key}_P": p(r["p"]), f"CORE{key}_DELTA": d(r["cliffs_delta"], 3),
                  f"CORE{key}_N": n(r["n_agent"] + r["n_human"])}
        v["CORE_PHOLM_MIN"] = p(core["p_holm"].min())
        sens = pd.read_csv(t / "core_sensitivity.csv")
        main10 = metrics[["repo", "number", "group", "d_i_core"]].merge(sens, on=["repo", "number", "group"])
        same = ((main10["d_i_core"] - main10["d_i_core_10"]).abs() < 1e-9) | (main10["d_i_core"].isna() & main10["d_i_core_10"].isna())
        # PRs whose Δ I changed once ties in Ca were broken by name (step 04 used set order).
        v |= {"CORE_MEASURED": n(len(sens)), "CORE_MATCH10": n(int(same.sum())),
              "CORE_TIE_CHANGED": n(len(sens) - int(same.sum()))}
    # Share of PR-level edges that come only from TYPE_CHECKING imports (all minus runtime variant).
    tc_only = (metrics.loc[metrics["status"] == "ok", "added_edges_all"] - metrics.loc[metrics["status"] == "ok", "added_edges"])
    v["TC_ONLY_EDGES"] = n(tc_only.clip(lower=0).sum())
    v["TC_ONLY_SHARE"] = pct(tc_only.clip(lower=0).sum() / metrics.loc[metrics["status"] == "ok", "added_edges_all"].sum())

    tlab = {"agent_pr": "PR của agent", "log_loc": "log(1 + số dòng thay đổi)", "log_files": "log(1 + số file thay đổi)",
            "merge_commit": "Kiểu merge commit", "years": "Thời điểm merge (năm)",
            "log_modules": "log(1 + số module trước PR)", "log_pkgs_touched": "log(1 + số package PR sửa)"}
    rows = ["| Biến | Thêm ≥ 1 cạnh: OR [KTC 95%] | *p* | Đưa package mới vào chu trình: OR [KTC 95%] | *p* |",
            "|---|---|---|---|---|"]
    m2 = reg[reg["model"] == "model1"]  # table: model 1 (model 2 adds the spread, collinear with files changed)
    for term, lab in tlab.items():
        if term == "log_pkgs_touched":
            continue
        cells = []
        for outcome in ("adds_edge", "cycle_new_pkgs"):
            r = m2[(m2["outcome"] == outcome) & (m2["term"] == term)].iloc[0]
            cells += [f"{d(r['odds_ratio'])} [{d(r['or_low'])}; {d(r['or_high'])}]", pt(r["p"])]
        rows.append(f"| {lab} | " + " | ".join(cells) + " |")
    v["TABLE_REG"] = "\n".join(rows)
    for outcome, ok_ in (("adds_edge", "EDGES"), ("cycle_new_pkgs", "CNP")):
        for model, mk in (("model1", "M1"), ("model2", "M2")):
            r = reg[(reg["outcome"] == outcome) & (reg["model"] == model) & (reg["term"] == "agent_pr")].iloc[0]
            v |= {f"REG_{ok_}_{mk}_OR": d(r["odds_ratio"]), f"REG_{ok_}_{mk}_LO": d(r["or_low"]),
                  f"REG_{ok_}_{mk}_HI": d(r["or_high"]), f"REG_{ok_}_{mk}_P": p(r["p"])}
        r = reg[(reg["outcome"] == outcome) & (reg["model"] == "model2") & (reg["term"] == "log_pkgs_touched")].iloc[0]
        v |= {f"REG_{ok_}_SPREAD_OR": d(r["odds_ratio"]), f"REG_{ok_}_SPREAD_P": p(r["p"])}
        r = reg[(reg["outcome"] == outcome) & (reg["model"] == "model1") & (reg["term"] == "years")].iloc[0]
        v |= {f"REG_{ok_}_YEARS_OR": d(r["odds_ratio"]), f"REG_{ok_}_YEARS_P": p(r["p"])}
    r0 = reg.iloc[0]
    v |= {"REG_N": n(r0["n"]), "REG_CLUSTERS": n(r0["clusters"]),
          # PRs left out of the GEE models because loc_changed is missing (diff touched a file > 1 MB).
          "REG_DROPPED": n(s["analysis_sample"]["prs"] - r0["n"]),
          "REG_COV_CNP": "exchangeable" if (reg[reg["outcome"] == "cycle_new_pkgs"]["cov_struct"] == "Exchangeable").all()
          else "độc lập"}

    for sub, key in (("all", "ALL"), ("Q4", "Q4")):
        r = spread.set_index("subset").loc[sub]
        v |= {f"SPREAD_{key}_MED_A": d(r["median_agent"], 0), f"SPREAD_{key}_MED_H": d(r["median_human"], 0),
              f"SPREAD_{key}_A": d(r["mean_agent"], 2), f"SPREAD_{key}_H": d(r["mean_human"], 2),
              f"SPREAD_{key}_DELTA": d(r["cliffs_delta"], 3), f"SPREAD_{key}_P": p(r["p"])}

    sel = selection.set_index("category")
    st = s["selection"]
    v |= {"SEL_ANALYSED": n(sel.loc["analysed", "prs"]), "SEL_TOTAL": n(selection["prs"].sum()),
          "SEL_ANALYSED_SHARE": pct(sel.loc["analysed", "prs"] / selection["prs"].sum()),
          "SEL_MED_IN": d(st["median_analysed"], 0), "SEL_MED_OUT": d(st["median_excluded"], 0),
          "SEL_DELTA": d(st["cliffs_delta"], 3), "SEL_P": p(st["p"]),
          "SEL_SHARE_BY_AGENT": "; ".join(f"{AGENT_VI.get(a, a)} {pct(x)}" for a, x in
                                          sorted(st["analysed_share_by_agent"].items(), key=lambda kv: kv[1]))}
    for cat, key in (("no_human_in_repo", "NOHUMAN"), ("not_on_first_parent", "NOTFOUND"),
                     ("repo_unavailable", "UNAVAIL"), ("no_py_source_change", "NOPY")):
        if cat in sel.index:
            v |= {f"SEL_{key}_N": n(sel.loc[cat, "prs"]), f"SEL_{key}_MED": d(sel.loc[cat, "median_aidev_loc"], 0),
                  f"SEL_{key}_CODEX": pct(sel.loc[cat, "share_codex"])}

    cont = s.get("contamination") or {}
    if cont:
        v |= {"CONT_K": n(cont["agent_involved"]), "CONT_N": n(cont["screened"]), "CONT_PCT": pct(cont["share"]),
              "CONT_LO": pct(cont["share_ci"][0]), "CONT_HI": pct(cont["share_ci"][1])}
        screen = pd.read_csv(path("manual_check") / "human_screen.csv")
        v["CONT_FLAGGED"] = n(int(screen.loc[screen["fetched"] == True, "flagged"].sum()))  # noqa: E712
        for m, key in (("adds_edge", "EDGES"), ("cycle_new_pkgs", "CNP")):
            c2 = cont["metrics"][m]
            kc = 1 if m == "adds_edge" else 2
            v |= {f"CONT_{key}_A": pct(c2["rate_agent"], kc), f"CONT_{key}_H_OBS": pct(c2["rate_human_observed"], kc),
                  f"CONT_{key}_H_UP": pct(c2["rate_human_corrected_upper"], kc)}

    r2 = rq2.loc["has_violation (rate)"]
    v |= {"RQ2_REPOS": n(s["rq2"]["repos"]), "RQ2_KA": n(r2["k_agent"]), "RQ2_NA": n(r2["n_agent"]),
          "RQ2_KH": n(r2["k_human"]), "RQ2_NH": n(r2["n_human"]), "RQ2_RATE_A": pct(r2["rate_agent"], 2),
          "RQ2_RATE_H": pct(r2["rate_human"], 2), "RQ2_P": p(r2["p_holm"]),
          "RQ2_IL_REPOS": n(s["rq2"]["by_source"].get("import-linter", 0))}
    if len(review):
        tp = int((review["verdict"] == "true").sum())
        v |= {"VIOL_REVIEW_EDGES": n(len(review)), "VIOL_REVIEW_TRUE": n(tp),
              "VIOL_REVIEW_PREC": pct(tp / len(review)) if len(review) else "–",
              "VIOL_REVIEW_PRS": n(review[["repo", "number"]].drop_duplicates().shape[0])}

    rows = ["| Agent | Số PR | Số kho | Số dòng (trung vị) | % PR thêm cạnh | Cạnh mới (TB) | % PR đưa package mới vào chu trình [KTC 95%] |",
            "|---|---|---|---|---|---|---|"]
    for _, r in rq3.iterrows():
        rows.append(f"| {AGENT_VI.get(r['agent'], r['agent'])} | {n(r['prs'])} | {n(r['repos'])} | {d(r['median_loc'], 0)} | "
                    f"{pct(r['share_added_edges'])} | {d(r['mean_added_edges'], 2)} | {pct(r['cycle_new_pkgs_rate'], 2)} "
                    f"[{d(100 * r['cycle_new_pkgs_ci_low'], 2)}; {d(100 * r['cycle_new_pkgs_ci_high'], 2)}] |")
    v["TABLE_RQ3"] = "\n".join(rows)
    k = s["rq3_tests"]
    cc = rq3.set_index("agent").loc["Claude_Code"] if "Claude_Code" in set(rq3["agent"]) else None
    chi = k.get("cycle_new_pkgs (rate)", {})
    v |= {"RQ3_K": n(k["added_edges"]["k"]), "RQ3_N": n(k["added_edges"]["n"]),
          "RQ3_H_EDGES": d(k["added_edges"]["H"], 1), "RQ3_P_EDGES": p(k["added_edges"]["p_holm"]),
          "RQ3_EPS_EDGES": d(k["added_edges"]["epsilon_sq"], 3),
          "RQ3_H_LOC": d(k["loc_changed"]["H"], 1), "RQ3_P_LOC": p(k["loc_changed"]["p"]),
          "RQ3_EPS_LOC": d(k["loc_changed"]["epsilon_sq"], 3),
          "RQ3_H_E100": d(k["edges_per_100_loc"]["H"], 1), "RQ3_P_E100": p(k["edges_per_100_loc"]["p"]),
          "RQ3_EPS_E100": d(k["edges_per_100_loc"]["epsilon_sq"], 3),
          "RQ3_CHI": d(chi.get("chi2", math.nan), 1), "RQ3_CHI_P": p(chi.get("p_holm", math.nan)),
          "RQ3_CHI_MINEXP": d(chi.get("min_expected", math.nan), 1),
          "RQ3_EXCLUDED": ", ".join(f"{AGENT_VI.get(a, a)} ({n(c)} PR)" for a, c in s["excluded_agents"].items()) or "không có"}
    if cc is not None:
        v |= {"CC_PRS": n(cc["prs"]), "CC_LOC": d(cc["median_loc"], 0), "CC_CNP": pct(cc["cycle_new_pkgs_rate"], 2),
              "CC_CNP_K": n(cc["cycle_new_pkgs_k"]), "CC_CNP_LO": d(100 * cc["cycle_new_pkgs_ci_low"], 2),
              "CC_CNP_HI": d(100 * cc["cycle_new_pkgs_ci_high"], 2), "CC_EDGES": pct(cc["share_added_edges"])}
        # C_new of Claude Code vs human PRs once PR size, merge style, date and repository size are controlled.
        ra = reg[(reg["outcome"] == "cycle_new_pkgs") & (reg["model"] == "model1_by_agent")].set_index("term")
        v |= {"CC_ADJ_OR": d(ra.loc["claude_code", "odds_ratio"]), "CC_ADJ_LO": d(ra.loc["claude_code", "or_low"]),
              "CC_ADJ_HI": d(ra.loc["claude_code", "or_high"]), "CC_ADJ_P": p(ra.loc["claude_code", "p"]),
              "OTHER_ADJ_OR": d(ra.loc["other_agent", "odds_ratio"]), "OTHER_ADJ_P": p(ra.loc["other_agent", "p"])}
        dunn = pd.DataFrame(s["dunn_holm"]["edges_per_100_loc"])
        others = [a for a in dunn.columns if a != "Claude_Code"]
        sig = [AGENT_VI.get(a, a) for a in others if dunn.loc["Claude_Code", a] < 0.05]
        vi_list = lambda xs: ", ".join(xs[:-1]) + " và " + xs[-1] if len(xs) > 1 else "".join(xs)
        v["CC_E100_SIG_VS"] = vi_list(sig) if sig else "không agent nào"
        ns = [AGENT_VI.get(a, a) for a in others if dunn.loc["Claude_Code", a] >= 0.05]
        v["CC_E100_NS_VS"] = vi_list(ns) or "không agent nào"
        v["CC_E100_CLAUSE"] = (f"khác {v['CC_E100_SIG_VS']}, nhưng không khác {vi_list(ns)}" if ns and sig
                               else "khác tất cả các agent còn lại" if sig else "không khác agent nào")
        en_list = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else "".join(xs)  # noqa: E731
        cc_clause_en = (f"differs from {en_list(sig)} but not from {en_list(ns)}" if ns and sig
                        else "differs from all other agents" if sig else "does not differ from any other agent")

    # Manual check by the two authors (step 05): agreement and tool accuracy.
    yn = lambda col: manual[col].str.strip().str.lower()  # noqa: E731
    manual_done = bool(len(manual)) and all(yn(c).isin(["yes", "no"]).all()
                                             for c in ("rater1_match", "rater2_match", "consensus_match"))
    if manual_done:
        r1, r2 = yn("rater1_match") == "yes", yn("rater2_match") == "yes"
        po = float((r1 == r2).mean())
        pe = float(r1.mean() * r2.mean() + (1 - r1.mean()) * (1 - r2.mean()))
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        v |= {"MANUAL_N": n(len(manual)), "MANUAL_MATCH": n((yn("consensus_match") == "yes").sum()),
              "MANUAL_AGREE": pct(po), "MANUAL_KAPPA": d(kappa, 2)}
    else:
        v |= {"MANUAL_N": n(len(manual)) if len(manual) else "[chưa có]", "MANUAL_MATCH": "[chưa có]",
              "MANUAL_AGREE": "[chưa có]", "MANUAL_KAPPA": "[chưa có]"}

    # Figure 6 (ECDF of added edges among PRs that add edges), on the analysis sample.
    both = ok.groupby("repo")["group"].agg(set)
    sample = ok[ok["repo"].isin(both[both.apply(lambda g: {"agent", "human"} <= g)].index)]
    # Repositories with measured agent PRs but no eligible human PR drop out of the comparison.
    lost = ok[ok["repo"].isin(both[both.apply(lambda g: "human" not in g)].index) & (ok["group"] == "agent")]["repo"].value_counts()
    sa = sample[sample["group"] == "agent"]
    codex_share = (sa["agent"] == "OpenAI_Codex").mean()
    v |= {"CODEX_N": n((sa["agent"] == "OpenAI_Codex").sum()), "CODEX_SHARE": pct(codex_share)}
    v |= {"NOHUMAN_REPOS": n(len(lost)), "NOHUMAN_AGENT_PRS": n(lost.sum()),
          "NOHUMAN_TOP_REPO": lost.index[0] if len(lost) else "", "NOHUMAN_TOP_PRS": n(lost.iloc[0]) if len(lost) else "0"}
    # Cloned repositories in which no agent PR could be measured (not on the first-parent chain, or no source .py change).
    cloned = set(mining.loc[mining["status"] == "ok", "repo"])
    noagent = cloned - set(ok.loc[ok["group"] == "agent", "repo"])
    noagent_status = metrics[(metrics["group"] == "agent") & metrics["repo"].isin(noagent)]["status"].value_counts()
    v |= {"NOAGENT_REPOS": n(len(noagent)), "NOAGENT_NOTFOUND_PRS": n(noagent_status.get("not_on_first_parent", 0)),
          "NOAGENT_NOPY_PRS": n(noagent_status.get("no_py_source_change", 0))}
    repo_chain_ok = len(cloned) - len(noagent) - len(lost) == sample["repo"].nunique()
    pos = sample[sample["added_edges"] > 0]
    pos_a, pos_h = pos[pos["group"] == "agent"]["added_edges"], pos[pos["group"] == "human"]["added_edges"]
    v |= {"ECDF_MED_A": d(pos_a.median(), 0), "ECDF_MED_H": d(pos_h.median(), 0),
          "ECDF_P90_A": d(pos_a.quantile(0.9), 0), "ECDF_P90_H": d(pos_h.quantile(0.9), 0),
          "ECDF_MAX_A": n(pos_a.max()), "ECDF_MAX_H": n(pos_h.max())}
    # English-formatted variants for the English abstract (decimal point, comma thousands separator).
    import re as _re

    def en(text: str) -> str:
        return _re.sub(r"\d[\d.,]*\d|\d", lambda m: m.group(0).translate(str.maketrans({".": ",", ",": "."})), text)

    v |= {f"{k}_EN": en(val) for k, val in list(v.items()) if "\n" not in val}
    if cc is not None:
        v["CC_E100_CLAUSE_EN"] = cc_clause_en
    v["NOT_FOUND_TOP2_EN"] = en(v.pop("NOT_FOUND_TOP2_EN_LIST"))
    v.pop("NOT_FOUND_TOP2_EN_LIST_EN", None)
    v["RQ3_EXCLUDED_EN"] = ", ".join(f"{AGENT_VI.get(a, a)} ({n(c)} PR{'s' if c != 1 else ''})"
                                     for a, c in s["excluded_agents"].items()) or "none"
    # English versions of the generated tables (English manuscript): same numbers, English number format and labels.
    labels_en = [
        ("| Bước lọc | Số PR agent | Số kho |", "| Filtering step | Agent PRs | Repositories |"),
        ("AIDev-pop ban đầu", "AIDev-pop (initial)"),
        ("Ngôn ngữ chính Python, không phải bản sao (fork) của kho khác", "Primary language Python, not a fork"),
        ("Giấy phép mã nguồn mở cho phép dùng lại", "Open-source license allowing reuse"),
        ("PR đã merge", "Merged PRs"),
        ("PR sửa ít nhất một file Python mã nguồn", "PRs changing at least one Python source file"),
        ("Kho có ít nhất 3 PR như trên", "Repositories with at least 3 such PRs"),
        ("| Chỉ số | Agent | Người | Hiệu ứng | *p* (Holm) |", "| Metric | Agent | Human | Effect | *p* (Holm) |"),
        ("Số cạnh mới khi có thêm: trung vị; TB", "New edges when any are added: median; mean"),
        ("% PR thêm ≥ 1 cạnh package", "% PRs adding ≥ 1 package edge"),
        ("% PR thêm ≥ 1 cạnh", "% PRs adding ≥ 1 edge"),
        ("% PR chỉ thêm cạnh trong chu trình có sẵn", "% PRs adding edges only inside existing cycles"),
        ("% PR đưa package mới vào chu trình [KTC 95%]", "% PRs bringing a new package into a cycle [95% CI]"),
        ("% PR đưa package mới vào chu trình", "% PRs bringing a new package into a cycle"),
        ("Δ Ce: TB", "Δ Ce: mean"), ("Δ I package lõi: TB", "Δ I of core packages: mean"),
        ("| Biến | Thêm ≥ 1 cạnh: OR [KTC 95%] | *p* | Đưa package mới vào chu trình: OR [KTC 95%] | *p* |",
         "| Variable | Adds ≥ 1 edge: OR [95% CI] | *p* | Brings a new package into a cycle: OR [95% CI] | *p* |"),
        ("PR của agent", "Agent PR"), ("log(1 + số dòng thay đổi)", "log(1 + changed lines)"),
        ("log(1 + số file thay đổi)", "log(1 + changed files)"), ("Kiểu merge commit", "Merge-commit style"),
        ("Thời điểm merge (năm)", "Merge time (years)"), ("log(1 + số module trước PR)", "log(1 + modules before the PR)"),
        ("| Nhóm quy mô (số dòng) | Chỉ số | Agent | Người | OR | *p* (Holm) |",
         "| Size group (changed lines) | Metric | Agent | Human | OR | *p* (Holm) |"),
        ("% PR thêm cạnh", "% PRs adding edges"),
        ("| Agent | Số PR | Số kho | Số dòng (trung vị) | % PR thêm cạnh | Cạnh mới (TB) |",
         "| Agent | PRs | Repositories | Changed lines (median) | % PRs adding edges | New edges (mean) |"),
        ("% PRs adding edges | Cạnh mới (TB) |", "% PRs adding edges | New edges (mean) |"),
        ("| Agent | Số PR | Số kho | Số dòng (trung vị) |", "| Agent | PRs | Repositories | Changed lines (median) |"),
    ]
    for key in ("TABLE_FUNNEL", "TABLE_RQ1", "TABLE_REG", "TABLE_STRATA", "TABLE_RQ3"):
        t_en = _re.sub(r"\[([^\]\s]+); ([^\]\s]+)\]", r"[\1, \2]", en(v[key]))  # English CI style: [lo, hi]
        for vi_label, en_label in labels_en:
            t_en = t_en.replace(vi_label, en_label)
        v[f"{key}_EN"] = t_en
    out = t / "numbers.json"
    out.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(v)} values -> {out}")

    # Every qualitative statement in the paper and the report, checked against the numbers.
    # The documents are not built while any of these is false.
    alpha = 0.05
    alpha = 0.05
    r = rq1
    regi = reg.set_index(["outcome", "model", "term"])
    or_ = lambda o, m, t="agent_pr": regi.loc[(o, m, t), "odds_ratio"]  # noqa: E731
    pv = lambda o, m, t="agent_pr": regi.loc[(o, m, t), "p"]  # noqa: E731
    sig = lambda row: (row["p_holm"] if row["family"] else row["p"]) < alpha  # noqa: E731
    lower = lambda row: row["rate_agent"] < row["rate_human"]  # noqa: E731
    var = variants.set_index(["variant", "metric"])
    sub = subs.set_index(["subset", "metric"])
    q = strata.set_index(["stratum", "metric"])
    spr = spread.set_index("subset")
    by_agent = s["selection"]["analysed_share_by_agent"]
    cont = s.get("contamination") or {}
    rq3i = rq3.set_index("agent")
    okd = metrics[metrics["status"] == "ok"]
    claims = {
        # Abstract, RQ1
        "RQ1: agent PRs add an edge less often, significant after Holm": lower(r.loc["adds_edge"]) and sig(r.loc["adds_edge"]),
        "RQ1: agent PRs add a package edge less often, significant after Holm":
            lower(r.loc["adds_pkg_edge"]) and sig(r.loc["adds_pkg_edge"]),
        "RQ1: among PRs adding edges, same median and no significant difference":
            r.loc["edges_if_any", "median_agent"] == r.loc["edges_if_any", "median_human"] and not sig(r.loc["edges_if_any"]),
        "RQ1: Δ Ce lower for agents (δ < 0), significant after Holm":
            r.loc["d_ce_mean", "cliffs_delta"] < 0 and sig(r.loc["d_ce_mean"]),
        "RQ1: Δ I of core packages not significant": not sig(r.loc["d_i_core"]),
        # Core threshold: tiny effect at 5%, 10% and 20%; none significant once the three are Holm-adjusted.
        "Core threshold: |δ| < 0.05 and Holm p > α at 5%, 10%, 20%":
            core is None or bool((core["cliffs_delta"].abs() < 0.05).all() and (core["p_holm"] > alpha).all()),
        "RQ1: C_new not significant after Holm": not sig(r.loc["cycle_new_pkgs"]),
        "RQ1: C_exist lower for agents, significant after Holm":
            lower(r.loc["cycle_existing_only"]) and sig(r.loc["cycle_existing_only"]),
        "RQ1: fewer than 60% of PRs flagged by the combined indicator bring a new package into a cycle (\"chỉ\")":
            (r.loc["cycle_new_pkgs", "k_agent"] + r.loc["cycle_new_pkgs", "k_human"])
            / (r.loc["new_cycle", "k_agent"] + r.loc["new_cycle", "k_human"]) < 0.6,
        "RQ1: every effect is small (|δ| < 0.147, rate differences < 5 percentage points)":
            bool((r["cliffs_delta"].dropna().abs() < 0.147).all())
            and bool(((r["rate_agent"] - r["rate_human"]).dropna().abs() < 0.05).all()),
        # Regression
        "GEE model 1: agent PRs add an edge less often (OR < 1, p < 0.05)":
            or_("adds_edge", "model1") < 1 and pv("adds_edge", "model1") < alpha,
        "GEE model 1: no significant agent effect on C_new": pv("cycle_new_pkgs", "model1") >= alpha,
        "GEE: lines changed, files changed and repository size raise the odds of adding an edge":
            all(or_("adds_edge", "model1", t) > 1 and pv("adds_edge", "model1", t) < alpha
                for t in ("log_loc", "log_files", "log_modules")),
        "GEE model 1: later PRs add edges less often (years OR < 1, p < 0.05)":
            or_("adds_edge", "model1", "years") < 1 and pv("adds_edge", "model1", "years") < alpha,
        "GEE model 2: packages touched is the strongest predictor of adding an edge (largest OR)":
            or_("adds_edge", "model2", "log_pkgs_touched") == reg[(reg["outcome"] == "adds_edge") & (reg["model"] == "model2")]["odds_ratio"].max(),
        "GEE: adding the spread moves the agent OR only slightly towards 1 (\"chỉ đổi\")":
            or_("adds_edge", "model1") < or_("adds_edge", "model2") < 1
            and or_("adds_edge", "model2") - or_("adds_edge", "model1") < 0.1,
        # Size strata
        "Strata: agent share of PRs adding an edge lower in all four strata":
            all(q.loc[(k, "adds_edge"), "rate_agent"] < q.loc[(k, "adds_edge"), "rate_human"] for k in ("Q1", "Q2", "Q3", "Q4")),
        "Strata: adding an edge significant after Holm only in Q3":
            list(strata[(strata["metric"] == "adds_edge") & (strata["p_holm"] < alpha)]["stratum"].astype(str)) == ["Q3"],
        "Strata: C_new not significant in any stratum": bool((strata[strata["metric"] == "cycle_new_pkgs"]["p_holm"] >= alpha).all()),
        # Fig 6
        "Fig 6: agent 90th percentile lower": float(v["ECDF_P90_A"].replace(",", ".")) < float(v["ECDF_P90_H"].replace(",", ".")),
        "Fig 6: PRs with more than 100 new edges occur only in the human group": pos_a.max() <= 100 and pos_h.max() > 100,
        # Sensitivity
        "Sensitivity: adding an edge still lower and significant without the largest repository and without Codex":
            all(lower(sub.loc[(k, "adds_edge")]) and sub.loc[(k, "adds_edge"), "p"] < alpha
                for k in ("without_largest_repo", "without_codex")),
        "Sensitivity: adding an edge lower and significant with all imports and with module-level imports":
            all(lower(var.loc[(k, "adds_edge")]) and var.loc[(k, "adds_edge"), "p"] < alpha
                for k in ("all imports", "module level only")),
        "Sensitivity: C_new not significant with either alternative graph":
            all(var.loc[(k, "cycle_new_pkgs"), "p"] >= alpha for k in ("all imports", "module level only")),
        "Sensitivity: TYPE_CHECKING-only edges are a small share (< 5%) of new edges":
            (okd["added_edges_all"] - okd["added_edges"]).clip(lower=0).sum() / okd["added_edges_all"].sum() < 0.05,
        "Sensitivity: without Codex, agent C_new rate above the human rate but not significant":
            sub.loc[("without_codex", "cycle_new_pkgs"), "rate_agent"] > sub.loc[("without_codex", "cycle_new_pkgs"), "rate_human"]
            and sub.loc[("without_codex", "cycle_new_pkgs"), "p"] >= alpha,
        "Repo-paired: mean new edges higher for humans in more repositories, significant":
            paired.loc["added_edges", "human_higher"] > paired.loc["added_edges", "agent_higher"] and paired.loc["added_edges", "p"] < alpha,
        "Repo-paired: share of PRs adding an edge not significant": not (paired.loc["adds_edge", "p"] < alpha),
        "Repo-paired: C_new higher for humans in more repositories, significant":
            paired.loc["cycle_new_pkgs", "human_higher"] > paired.loc["cycle_new_pkgs", "agent_higher"]
            and paired.loc["cycle_new_pkgs", "p"] < alpha,
        # Data and selection
        "Data: agent PRs larger than human PRs (median LOC)": t1.loc["agent", "median_loc"] > t1.loc["human", "median_loc"],
        "Data: agent PRs sit in smaller repositories (median modules before)":
            t1.loc["agent", "median_modules_before"] < t1.loc["human", "median_modules_before"],
        "Data: most not-found agent PRs are in the two named repositories (\"phần lớn\")":
            nf.head(2).sum() / nf.sum() > 0.5,
        "Data: most agent PRs of repositories without humans belong to the named repository (\"phần lớn\")":
            lost.iloc[0] / lost.sum() > 0.5,
        "Selection: analysed agent PRs larger than excluded ones, significant":
            s["selection"]["median_analysed"] > s["selection"]["median_excluded"] and s["selection"]["p"] < alpha,
        "Data: cloned repos minus repos without measured agent PRs minus repos without human PRs = analysed repos":
            repo_chain_ok,
        "Selection: Codex excluded far more often (analysed share below half of any other agent's)":
            by_agent["OpenAI_Codex"] < 0.5 * min(x for a_, x in by_agent.items() if a_ != "OpenAI_Codex"),
        "Contamination: screened agent involvement found, and correction widens the gap":
            bool(cont) and cont["agent_involved"] > 0
            and cont["metrics"]["adds_edge"]["rate_human_corrected_upper"] >= cont["metrics"]["adds_edge"]["rate_human_observed"],
        "Discussion: C_exist PRs sit in larger repositories than other PRs (median modules before)":
            sample.loc[sample["cycle_existing_only"] == 1, "modules_before"].median() > sample["modules_before"].median(),
        "Discussion: agent PRs touch fewer packages (δ < 0, significant), clearest in Q4":
            spr.loc["all", "cliffs_delta"] < 0 and spr.loc["all", "p"] < alpha
            and spr.loc["Q4", "cliffs_delta"] < spr.loc["all", "cliffs_delta"]
            and spr.loc["Q4", "median_agent"] < spr.loc["Q4", "median_human"],
        "Limitations: Codex is close to half of analysed agent PRs (\"gần một nửa\": 40-50%)":
            0.40 <= codex_share < 0.50,
        "Discussion: more than half of the PRs flagged by the combined indicator are C_exist (\"hơn một nửa\")":
            (cnp["k_agent"] + cnp["k_human"]) / (cyc["k_agent"] + cyc["k_human"]) < 0.5,
        "Discussion: Codex has the smallest PRs and the lowest C_new rate among agents":
            rq3i["median_loc"].idxmin() == "OpenAI_Codex" and rq3i["cycle_new_pkgs_rate"].idxmin() == "OpenAI_Codex",
        # RQ2, RQ3
        "RQ2: violations rare (< 1% in both groups) and not significant": r2["rate_agent"] < 0.01 and r2["rate_human"] < 0.01
            and r2["p_holm"] >= alpha,
        "RQ3: added edges differ between agents (Kruskal-Wallis, Holm)": k["added_edges"]["p_holm"] < alpha,
        "RQ3: PR size differs clearly between agents (\"rõ rệt\": ε² larger than for added edges)":
            k["loc_changed"]["p"] < alpha and k["loc_changed"]["epsilon_sq"] > k["added_edges"]["epsilon_sq"],
        "RQ3: chi-square not used because the smallest expected count is below 5": chi.get("min_expected", 0) < 5,
    }
    if cc is not None:
        others = rq3i.drop("Claude_Code")
        claims |= {
            "RQ3: Claude Code has the largest median PR size": cc["median_loc"] == rq3["median_loc"].max(),
            "RQ3: Claude Code has the highest C_new rate": cc["cycle_new_pkgs_rate"] == rq3["cycle_new_pkgs_rate"].max(),
            "RQ3: Claude Code C_new interval does not overlap Codex and Copilot":
                all(cc["cycle_new_pkgs_ci_low"] > others.loc[a_, "cycle_new_pkgs_ci_high"] for a_ in ("OpenAI_Codex", "Copilot")),
            "RQ3: Claude Code is the smallest or second-smallest agent sample (\"mẫu còn nhỏ\")": cc["prs"] <= sorted(rq3["prs"])[1],
            "RQ3: after normalising by 100 LOC Claude Code still differs from at least one agent":
                v["CC_E100_SIG_VS"] != "không agent nào",
            "RQ3: Claude Code median PR size falls in the largest size quartile (Q4)":
                cc["median_loc"] >= int(strata[strata["stratum"] == "Q4"]["loc_range"].iloc[0].split("-")[0]),
            "RQ3: with PR size controlled (GEE), Claude Code C_new does not differ from human PRs":
                regi.loc[("cycle_new_pkgs", "model1_by_agent", "claude_code"), "p"] >= alpha,
        }
    if len(review):
        claims["Violation review completed (every flagged edge has a verdict)"] = bool(review["verdict"].isin(["true", "false"]).all())
    # The two-reviewer check (manual_check_team.csv) was not carried out and is not reported in the paper.
    pending = []
    (t / "pending.json").write_text(json.dumps(pending, ensure_ascii=False, indent=1), encoding="utf-8")
    result = {k_: bool(ok_) for k_, ok_ in claims.items()}
    (t / "claims.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    failed = [k_ for k_, ok_ in result.items() if not ok_]
    print(f"claims: {len(result) - len(failed)}/{len(result)} hold")
    for k_ in failed:
        print("  FALSE:", k_)
    log_run("08_numbers.py", {}, {"values": len(v), "claims_false": failed}, valid=not failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
