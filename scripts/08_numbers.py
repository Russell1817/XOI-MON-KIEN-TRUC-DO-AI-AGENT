"""Step 08: collect every number quoted in the report and the paper into results/tables/numbers.json.

Values are formatted the Vietnamese way (decimal comma, dot as thousands separator). The Markdown
sources use {{KEY}} placeholders that tools/md2docx.py fills from this file, so every number in the
documents comes from the same analysis run (writing rule 2.1).
"""
from __future__ import annotations

import json
import math
import sys

import pandas as pd

from aierosion.config import path
from aierosion.runlog import log_run
from aierosion.stats.tests import wilson

AGENT_VI = {"OpenAI_Codex": "OpenAI Codex", "Copilot": "GitHub Copilot", "Claude_Code": "Claude Code",
            "Cursor": "Cursor", "Devin": "Devin", "Google_Jules": "Google Jules"}


def n(x: float) -> str:
    """Integer with dot thousands separator."""
    return f"{int(round(x)):,}".replace(",", ".")


def d(x: float, k: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{k}f}".replace(".", ",")


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
    nolarge = pd.read_csv(t / "robust_without_largest_repo.csv").set_index("metric")
    rq2 = pd.read_csv(t / "table_rq2.csv").set_index("metric")
    rq3 = pd.read_csv(t / "table3_rq3.csv")
    funnel = pd.read_csv(path("logs") / "filter_funnel.csv")
    mining = pd.read_csv(path("logs") / "mining_repos.csv")
    metrics = pd.read_csv(path("processed") / "pr_metrics.csv", low_memory=False)
    manual_file = path("manual_check") / "manual_check.csv"
    manual = pd.read_csv(manual_file) if manual_file.exists() else pd.DataFrame(columns=["match"])
    review_file = path("manual_check") / "violations_review.csv"
    review = pd.read_csv(review_file) if review_file.exists() and review_file.stat().st_size > 2 else pd.DataFrame()
    if len(review):
        # verdicts are read back as booleans by pandas; normalise to "true"/"false"
        review["verdict"] = review["verdict"].astype(str).str.strip().str.lower()

    v: dict[str, str] = {}
    funnel_labels = ["AIDev-pop ban đầu", "Ngôn ngữ chính Python, không phải fork",
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
        "RENAME_PRS": n((ok.get("renamed_py", pd.Series(0, index=ok.index)).fillna(0) > 0).sum()),
        "RENAME_UNAVAIL": n((ok.get("rename_detection", pd.Series("ok", index=ok.index)) == "unavailable").sum()),
    }
    # Share of agent PRs not found on the first-parent chain, per agent and in the two largest sources.
    nf = a_all[a_all["status"] == "not_on_first_parent"]["repo"].value_counts()
    v["NOT_FOUND_TOP2"] = ", ".join(f"{r} ({n(c)})" for r, c in nf.head(2).items())
    found = a_all.groupby("agent")["status"].apply(lambda x: (x != "not_on_first_parent").mean())
    v["FOUND_RATE_BY_AGENT"] = "; ".join(f"{AGENT_VI.get(k, k)} {pct(x, 0)}" for k, x in found.sort_values().items())

    largest_n = int((ok[(ok["group"] == "agent")]["repo"] == s["largest_agent_repo"]).sum())
    v["LARGEST_REPO_AGENT_PRS"] = n(largest_n)
    v["LARGEST_REPO_SHARE"] = pct(largest_n / s["analysis_sample"]["agent"])

    for g, key in (("agent", "A"), ("human", "H")):
        r = t1.loc[g]
        v[f"T1_{key}_REPOS"], v[f"T1_{key}_PRS"] = n(r["repos"]), n(r["prs"])
        v[f"T1_{key}_LOC"], v[f"T1_{key}_FILES"] = d(r["median_loc"], 0), d(r["median_files"], 0)
        v[f"T1_{key}_PYFILES"], v[f"T1_{key}_MODULES"] = d(r["median_py_files"], 0), d(r["median_modules_before"], 0)
        v[f"T1_{key}_MERGE"] = pct(r["share_merge_commit"])

    labels = {"added_edges": "Cạnh mới", "edges_per_file": "Cạnh mới / file", "added_pkg_edges": "Cạnh package mới",
              "d_cyclic_pkgs": "Δ package trong chu trình", "d_ce_mean": "Δ Ce", "d_i_core": "Δ I package lõi"}
    rows = ["| Chỉ số | Agent: TB (% PR > 0) | Người: TB (% PR > 0) | Cliff's δ | p (Holm) |", "|---|---|---|---|---|"]
    for m, lab in labels.items():
        r = rq1.loc[m]
        rows.append(f"| {lab} | {d(r['mean_agent'], 3)} ({pct(r['share_pos_agent'])}) | "
                    f"{d(r['mean_human'], 3)} ({pct(r['share_pos_human'])}) | {d(r['cliffs_delta'], 3)} | {pt(r['p_holm'])} |")
    c = rq1.loc["new_cycle (rate)"]
    rows.append(f"| Chu trình mới (tỷ lệ PR) | {pct(c['rate_agent'], 2)} ({n(c['k_agent'])}/{n(c['n_agent'])}) | "
                f"{pct(c['rate_human'], 2)} ({n(c['k_human'])}/{n(c['n_human'])}) | OR = {d(c['odds_ratio'])} | {pt(c['p_holm'])} |")
    v["TABLE_RQ1"] = "\n".join(rows)
    for m, key in (("added_edges", "EDGES"), ("edges_per_file", "EPF"), ("d_cyclic_pkgs", "DCYC"),
                   ("d_ce_mean", "DCE"), ("d_i_core", "DI"), ("added_pkg_edges", "PKGE")):
        r = rq1.loc[m]
        v |= {f"RQ1_{key}_A": d(r["mean_agent"], 2), f"RQ1_{key}_H": d(r["mean_human"], 2),
              f"RQ1_{key}_POS_A": pct(r["share_pos_agent"]), f"RQ1_{key}_POS_H": pct(r["share_pos_human"]),
              f"RQ1_{key}_DELTA": d(r["cliffs_delta"], 3), f"RQ1_{key}_P": p(r["p_holm"])}
    v |= {"RQ1_CYC_A": pct(c["rate_agent"], 2), "RQ1_CYC_H": pct(c["rate_human"], 2),
          "RQ1_CYC_KA": n(c["k_agent"]), "RQ1_CYC_KH": n(c["k_human"]), "RQ1_CYC_OR": d(c["odds_ratio"]),
          "RQ1_CYC_P": p(c["p_holm"])}
    e100 = rq1.loc["edges_per_100_loc"]
    v |= {"E100_A": d(e100["mean_agent"], 2), "E100_H": d(e100["mean_human"], 2), "E100_DELTA": d(e100["cliffs_delta"], 3),
          "E100_P": p(e100["p"])}

    rows = ["| Nhóm quy mô (số dòng) | Chỉ số | Agent | Người | Hiệu ứng | p (Holm) |", "|---|---|---|---|---|---|"]
    for _, r in strata.iterrows():
        if r["metric"] == "added_edges":
            rows.append(f"| {r['stratum']} ({rng(r['loc_range'])}) | % PR thêm cạnh | {pct(r['share_pos_agent'])} "
                        f"(n = {n(r['n_agent'])}) | {pct(r['share_pos_human'])} (n = {n(r['n_human'])}) | "
                        f"δ = {d(r['cliffs_delta'], 3)} | {pt(r['p_holm'])} |")
        else:
            rows.append(f"| | % PR tạo chu trình | {pct(r['rate_agent'], 2)} | {pct(r['rate_human'], 2)} | "
                        f"OR = {d(r['odds_ratio'])} | {pt(r['p_holm'])} |")
    v["TABLE_STRATA"] = "\n".join(rows)
    q4 = strata[strata["stratum"] == "Q4"].set_index("metric")
    v |= {"Q4_RANGE": rng(q4.loc["added_edges", "loc_range"]),
          "Q4_EDGES_A": pct(q4.loc["added_edges", "share_pos_agent"]), "Q4_EDGES_H": pct(q4.loc["added_edges", "share_pos_human"]),
          "Q4_EDGES_DELTA": d(q4.loc["added_edges", "cliffs_delta"], 3), "Q4_EDGES_P": p(q4.loc["added_edges", "p_holm"]),
          "Q4_CYC_A": pct(q4.loc["new_cycle (rate)", "rate_agent"], 1), "Q4_CYC_H": pct(q4.loc["new_cycle (rate)", "rate_human"], 1),
          "Q4_CYC_P": p(q4.loc["new_cycle (rate)", "p_holm"])}
    low = strata[(strata["stratum"] != "Q4")]
    v["Q123_MIN_P"] = p(low["p_holm"].min())

    for m, key in (("added_edges", "EDGES"), ("edges_per_file", "EPF"), ("new_cycle", "CYC")):
        r = paired.loc[m]
        v |= {f"PAIR_{key}_REPOS": n(r["repos"]), f"PAIR_{key}_AH": n(r["agent_higher"]),
              f"PAIR_{key}_HH": n(r["human_higher"]), f"PAIR_{key}_TIES": n(r["ties"]), f"PAIR_{key}_P": p(r["p"])}
    nl_e, nl_c = nolarge.loc["added_edges"], nolarge.loc["new_cycle (rate)"]
    v |= {"NOLARGE_N_A": n(nl_e["n_agent"]), "NOLARGE_EDGES_POS_A": pct(nl_e["share_pos_agent"]),
          "NOLARGE_EDGES_POS_H": pct(nl_e["share_pos_human"]), "NOLARGE_EDGES_DELTA": d(nl_e["cliffs_delta"], 3),
          "NOLARGE_EDGES_P": p(nl_e["p"]), "NOLARGE_CYC_A": pct(nl_c["rate_agent"], 2),
          "NOLARGE_CYC_H": pct(nl_c["rate_human"], 2), "NOLARGE_CYC_P": p(nl_c["p"])}

    r2 = rq2.loc["has_violation (rate)"]
    v |= {"RQ2_REPOS": n(s["rq2"]["repos"]), "RQ2_KA": n(r2["k_agent"]), "RQ2_NA": n(r2["n_agent"]),
          "RQ2_KH": n(r2["k_human"]), "RQ2_NH": n(r2["n_human"]), "RQ2_RATE_A": pct(r2["rate_agent"], 2),
          "RQ2_RATE_H": pct(r2["rate_human"], 2), "RQ2_P": p(r2["p_holm"]),
          "RQ2_IL_REPOS": n(s["rq2"]["by_source"].get("import-linter", 0))}
    if len(review):
        tp = int((review["verdict"] == "true").sum())
        v |= {"VIOL_REVIEW_EDGES": n(len(review)), "VIOL_REVIEW_TRUE": n(tp),
              "VIOL_REVIEW_PREC": pct(tp / len(review), 0) if len(review) else "–",
              "VIOL_REVIEW_PRS": n(review[["repo", "number"]].drop_duplicates().shape[0])}

    rows = ["| Agent | Số PR | Số kho | Số dòng (trung vị) | % PR thêm cạnh | Cạnh mới (TB) | % PR tạo chu trình [KTC 95%] |",
            "|---|---|---|---|---|---|---|"]
    for _, r in rq3.iterrows():
        rows.append(f"| {AGENT_VI.get(r['agent'], r['agent'])} | {n(r['prs'])} | {n(r['repos'])} | {d(r['median_loc'], 0)} | "
                    f"{pct(r['share_added_edges'])} | {d(r['mean_added_edges'], 2)} | {pct(r['new_cycle_rate'])} "
                    f"[{d(100 * r['new_cycle_ci_low'], 1)}; {d(100 * r['new_cycle_ci_high'], 1)}] |")
    v["TABLE_RQ3"] = "\n".join(rows)
    k = s["rq3_tests"]
    cc = rq3.set_index("agent").loc["Claude_Code"] if "Claude_Code" in set(rq3["agent"]) else None
    v |= {"RQ3_K": n(k["added_edges"]["k"]), "RQ3_N": n(k["added_edges"]["n"]),
          "RQ3_H_EDGES": d(k["added_edges"]["H"], 1), "RQ3_P_EDGES": p(k["added_edges"]["p_holm"]),
          "RQ3_EPS_EDGES": d(k["added_edges"]["epsilon_sq"], 3),
          "RQ3_H_LOC": d(k["loc_changed"]["H"], 1), "RQ3_P_LOC": p(k["loc_changed"]["p"]),
          "RQ3_EPS_LOC": d(k["loc_changed"]["epsilon_sq"], 3),
          "RQ3_H_E100": d(k["edges_per_100_loc"]["H"], 1), "RQ3_P_E100": p(k["edges_per_100_loc"]["p"]),
          "RQ3_EPS_E100": d(k["edges_per_100_loc"]["epsilon_sq"], 3),
          "RQ3_CHI": d(k["new_cycle (rate)"]["chi2"], 1), "RQ3_CHI_P": p(k["new_cycle (rate)"]["p_holm"]),
          "RQ3_CHI_MINEXP": d(k["new_cycle (rate)"]["min_expected"], 1),
          "RQ3_EXCLUDED": ", ".join(f"{AGENT_VI.get(a, a)} ({n(c)} PR)" for a, c in s["excluded_agents"].items()) or "không có"}
    if cc is not None:
        v |= {"CC_PRS": n(cc["prs"]), "CC_LOC": d(cc["median_loc"], 0), "CC_CYC": pct(cc["new_cycle_rate"]),
              "CC_EDGES": pct(cc["share_added_edges"])}
        dunn = pd.DataFrame(s["dunn_holm"]["edges_per_100_loc"])
        others = [a for a in dunn.columns if a != "Claude_Code"]
        sig = [AGENT_VI.get(a, a) for a in others if dunn.loc["Claude_Code", a] < 0.05]
        v["CC_E100_SIG_VS"] = ", ".join(sig) if sig else "không agent nào"
        v["CC_E100_NS_VS"] = ", ".join(AGENT_VI.get(a, a) for a in others if dunn.loc["Claude_Code", a] >= 0.05) or "không agent nào"
        ns = [AGENT_VI.get(a, a) for a in others if dunn.loc["Claude_Code", a] >= 0.05]
        v["CC_E100_CLAUSE"] = (f"khác {v['CC_E100_SIG_VS']}, nhưng không khác {', '.join(ns)}" if ns and sig
                               else "khác tất cả các agent còn lại" if sig else "không khác agent nào")

    v |= {"MANUAL_N": n(len(manual)), "MANUAL_MATCH": n((manual["match"] == "yes").sum())}

    # Figure 6 (ECDF of added edges among PRs that add edges), on the analysis sample.
    both = ok.groupby("repo")["group"].agg(set)
    sample = ok[ok["repo"].isin(both[both.apply(lambda g: {"agent", "human"} <= g)].index)]
    pos = sample[sample["added_edges"] > 0]
    pos_a, pos_h = pos[pos["group"] == "agent"]["added_edges"], pos[pos["group"] == "human"]["added_edges"]
    v |= {"ECDF_MED_A": d(pos_a.median(), 0), "ECDF_MED_H": d(pos_h.median(), 0),
          "ECDF_MAX_A": n(pos_a.max()), "ECDF_MAX_H": n(pos_h.max())}
    # English-formatted variants for the English abstract (decimal point, comma thousands separator).
    import re as _re

    def en(text: str) -> str:
        return _re.sub(r"\d[\d.,]*\d|\d", lambda m: m.group(0).translate(str.maketrans({".": ",", ",": "."})), text)

    v |= {f"{k}_EN": en(val) for k, val in list(v.items()) if "\n" not in val}
    out = t / "numbers.json"
    out.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(v)} values -> {out}")

    # Every qualitative statement in the paper and the report, checked against the numbers.
    # The documents are not built while any of these is false (tools/build_docs.py reads claims.json).
    alpha = 0.05
    claims = {
        "RQ1: agent PRs add an edge less often than human PRs, significant after Holm":
            rq1.loc["added_edges", "share_pos_agent"] < rq1.loc["added_edges", "share_pos_human"]
            and rq1.loc["added_edges", "p_holm"] < alpha,
        "RQ1: agent PRs create a new cycle less often than human PRs, significant after Holm":
            c["rate_agent"] < c["rate_human"] and c["p_holm"] < alpha,
        "RQ1: Δ Ce and Δ I of core packages not significant after Holm":
            rq1.loc["d_ce_mean", "p_holm"] >= alpha and rq1.loc["d_i_core", "p_holm"] >= alpha,
        "RQ1: every |Cliff's δ| in Table 4 is below 0.07":
            bool((rq1.loc[list(labels), "cliffs_delta"].abs() < 0.07).all()),
        "Size strata: no significant difference in Q1-Q3 (both metrics)":
            bool((low["p_holm"] >= alpha).all()),
        "Size strata: Q4 new-cycle rate far above Q1 in both groups (\"PR lớn ... cao hơn hẳn PR nhỏ\")":
            bool(q4.loc["new_cycle (rate)", "rate_agent"] > 5 * strata[(strata["stratum"] == "Q1") & (strata["metric"] == "new_cycle (rate)")]["rate_agent"].iloc[0]
                 and q4.loc["new_cycle (rate)", "rate_human"] > 5 * strata[(strata["stratum"] == "Q1") & (strata["metric"] == "new_cycle (rate)")]["rate_human"].iloc[0]),
        "Data: agent PRs smaller than human PRs (median LOC)": t1.loc["agent", "median_loc"] < t1.loc["human", "median_loc"],
        "Size strata: Q4 added-edge share lower for agents and significant":
            q4.loc["added_edges", "share_pos_agent"] < q4.loc["added_edges", "share_pos_human"]
            and q4.loc["added_edges", "p_holm"] < alpha,
        "Repo-paired: humans higher in more repositories for added edges, significant":
            paired.loc["added_edges", "human_higher"] > paired.loc["added_edges", "agent_higher"]
            and paired.loc["added_edges", "p"] < alpha,
        "Repo-paired: new-cycle difference not significant":
            not (paired.loc["new_cycle", "p"] < alpha),
        "Without largest repo: agents still add edges less often":
            nl_e["share_pos_agent"] < nl_e["share_pos_human"],
        "Fig 6: median added edges (PRs with edges) lower for agents": pos_a.median() < pos_h.median(),
        "Fig 6: agent ECDF lies on or above the human ECDF for most values (agent stochastically smaller)":
            sum((pos_a <= x).mean() >= (pos_h <= x).mean() for x in range(1, 101)) >= 90,
        "Fig 6: PRs with more than 100 new edges occur only in the human group":
            pos_a.max() <= 100 and pos_h.max() > 100,
        "Fig 7: repositories on both sides of the diagonal":
            paired.loc["added_edges", "agent_higher"] > 0 and paired.loc["added_edges", "human_higher"] > 0,
        "RQ2: difference in violation rate not significant": r2["p_holm"] >= alpha,
        "RQ2: violations are rare (< 1% of PRs in both groups)":
            r2["rate_agent"] < 0.01 and r2["rate_human"] < 0.01,
        "RQ3: added edges differ between agents (Kruskal-Wallis, Holm)": k["added_edges"]["p_holm"] < alpha,
        "RQ3: PR size differs between agents": k["loc_changed"]["p"] < alpha,
    }
    if cc is not None:
        claims |= {
            "RQ3: Claude Code has the largest median PR size": cc["median_loc"] == rq3["median_loc"].max(),
            "RQ3: Claude Code has the highest share of PRs adding edges":
                cc["share_added_edges"] == rq3["share_added_edges"].max(),
            "RQ3: Claude Code has the highest new-cycle rate": cc["new_cycle_rate"] == rq3["new_cycle_rate"].max(),
            "RQ3: Claude Code is the smallest or second-smallest agent sample (\"mẫu còn nhỏ\")":
                cc["prs"] <= sorted(rq3["prs"])[1],
            "RQ3: after normalising by 100 LOC Claude Code still differs from at least one agent":
                v["CC_E100_SIG_VS"] != "không agent nào",
        }
    if len(review):
        claims["Violation review completed (every flagged edge has a verdict)"] = bool(review["verdict"].isin(["true", "false"]).all())
    claims["Manual check: sample reviewed and every sampled PR matches"] = bool(len(manual)) and bool((manual["match"] == "yes").all())
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
