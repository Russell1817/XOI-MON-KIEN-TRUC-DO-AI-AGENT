"""Step 07: data figures for the paper, drawn with matplotlib (SVG + 300 dpi PNG).

Diagrams (Figures 1-2) are draw.io sources in results/figures/src/, exported by tools/export_drawio.py.

Colours: slots 1-2 of the validated reference palette (blue = agent, orange = human). Every
series also has its own marker shape so the figures stay readable in black-and-white print.
"""
from __future__ import annotations

import json
import unicodedata
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from aierosion.config import path
from aierosion.stats.tests import wilson

AGENT, HUMAN = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


LANG = "en" if "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "en" else "vi"
EN = {'PR của agent': 'Agent PRs', 'PR của người': 'Human PRs', 'Thêm ≥1 cạnh\nphụ thuộc (file)': 'Adds ≥1 dependency\nedge (file)', 'Thêm ≥1 cạnh\ngiữa package': 'Adds ≥1 edge\nbetween packages', 'Đưa package mới\nvào chu trình': 'Brings a new package\ninto a cycle', 'Chỉ thêm cạnh trong\nchu trình có sẵn': 'Adds edges only inside\nexisting cycles', 'Có vi phạm\nphân tầng (RQ2)': 'Has a layering\nviolation (RQ2)', 'Tỷ lệ PR (%), thanh ngang: khoảng tin cậy Wilson 95%': 'Share of PRs (%), bars: 95% Wilson confidence interval', 'Thêm ≥1 cạnh phụ thuộc': 'Adds ≥1 dependency edge', 'Đưa package mới vào chu trình': 'Brings a new package into a cycle', 'Tỷ lệ PR (%)': 'Share of PRs (%)', 'Tứ phân vị số dòng thay đổi': 'Quartile of changed lines', 'Tỷ lệ PR (%), khoảng tin cậy Wilson 95%': 'Share of PRs (%), 95% Wilson confidence interval', 'Số cạnh phụ thuộc mới của một PR (thang log)': 'New dependency edges per PR (log scale)', 'Tỷ lệ tích lũy (%)': 'Cumulative share (%)', 'Cạnh mới trung bình – PR của người': 'Mean new edges – human PRs', 'Cạnh mới trung bình – PR của agent': 'Mean new edges – agent PRs'}


def L(text: str) -> str:
    """Figure text in the language of the run (Vietnamese by default, English with --lang en)."""
    if LANG != "en":
        return text
    key = unicodedata.normalize("NFC", text)
    return next((v for k, v in EN.items() if unicodedata.normalize("NFC", k) == key), text)


def dec(s: str) -> str:
    return s if LANG == "en" else s.replace(".", ",")


VI_NUM = FuncFormatter(lambda v, _: dec(f"{v:g}"))  # decimal comma in Vietnamese, point in English
STYLE = {"agent": (AGENT, "o", L("PR của agent")), "human": (HUMAN, "s", L("PR của người"))}


AGENT_NAMES = {"OpenAI_Codex": "OpenAI Codex", "Copilot": "GitHub Copilot", "Claude_Code": "Claude Code",
               "Cursor": "Cursor", "Devin": "Devin", "Google_Jules": "Google Jules"}


def vi_int(x: int) -> str:
    """Integer with dot thousands separator (Vietnamese convention)."""
    return f"{int(x):,}" if LANG == "en" else f"{int(x):,}".replace(",", ".")


def vi(x: float, digits: int = 1) -> str:
    return dec(f"{x:.{digits}f}")


def save(fig, name: str) -> None:
    # Decimal comma only on purely numeric axes (default ScalarFormatter); category labels set with
    # set_yticks(..., labels) or set_xticks(..., labels) must stay untouched.
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            if type(axis.get_major_formatter()) is matplotlib.ticker.ScalarFormatter:
                axis.set_major_formatter(VI_NUM)
    out = path("figures")
    name = f"{name}_en" if LANG == "en" else name
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



def _rate_rows(d: pd.DataFrame, col: str, label: str) -> list[dict]:
    rows = []
    for grp in ("agent", "human"):
        g = d[d["group"] == grp][col].dropna()
        k, n = int(g.sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({"metric": label, "group": grp, "rate": k / n, "lo": lo, "hi": hi, "k": k, "n": n})
    return rows


def rates_by_group(d: pd.DataFrame) -> None:
    d = d.assign(adds_edge=(d["added_edges"] > 0).astype(int), adds_pkg_edge=(d["added_pkg_edges"] > 0).astype(int))
    e = d[d["rq2_eligible"] == True]  # noqa: E712
    groups = e.groupby("repo")["group"].agg(set)
    e = e[e["repo"].isin(groups[groups.apply(lambda s: {"agent", "human"} <= s)].index)]
    rows = (_rate_rows(d, "adds_edge", L("Thêm ≥1 cạnh\nphụ thuộc (file)"))
            + _rate_rows(d, "adds_pkg_edge", L("Thêm ≥1 cạnh\ngiữa package"))
            + _rate_rows(d, "cycle_new_pkgs", L("Đưa package mới\nvào chu trình"))
            + _rate_rows(d, "cycle_existing_only", L("Chỉ thêm cạnh trong\nchu trình có sẵn"))
            + _rate_rows(e, "has_violation", L("Có vi phạm\nphân tầng (RQ2)")))
    r = pd.DataFrame(rows)
    labels = list(dict.fromkeys(r["metric"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for j, grp in enumerate(("agent", "human")):
        color, marker, name = STYLE[grp]
        s = r[r["group"] == grp]
        y = [labels.index(m) + (0.14 if j == 0 else -0.14) for m in s["metric"]]
        ax.errorbar(s["rate"] * 100, y, xerr=[(s["rate"] - s["lo"]) * 100, (s["hi"] - s["rate"]) * 100],
                    fmt=marker, color=color, ms=6, mec="white", mew=0.8, elinewidth=1.6, capsize=0, label=name)
        for hv, xv, yv, k, n in zip(s["hi"] * 100, s["rate"] * 100, y, s["k"], s["n"]):
            ax.text(hv + 0.6, yv, f"{vi(xv)}% ({vi_int(k)}/{vi_int(n)})", fontsize=7, color=MUTED, va="center")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(L("Tỷ lệ PR (%), thanh ngang: khoảng tin cậy Wilson 95%"))
    ax.set_xlim(0, r["hi"].max() * 100 * 1.4)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "fig3_rates_by_group")


def size_strata(d: pd.DataFrame) -> None:
    s = d[d["loc_changed"].notna()].copy()
    s["size_q"] = pd.qcut(s["loc_changed"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    bounds = s.groupby("size_q", observed=True)["loc_changed"].agg(["min", "max"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
    for ax, (col, title) in zip(axes, [("added_edges", L("Thêm ≥1 cạnh phụ thuộc")), ("cycle_new_pkgs", L("Đưa package mới vào chu trình"))]):
        for j, grp in enumerate(("agent", "human")):
            color, marker, name = STYLE[grp]
            pts = []
            for q, g in s[s["group"] == grp].groupby("size_q", observed=True):
                k, n = int((g[col] > 0).sum()), len(g)
                lo, hi = wilson(k, n)
                pts.append((list(bounds.index).index(q) + (j - 0.5) * 0.12, k / n * 100, lo * 100, hi * 100))
            x, y, lo, hi = zip(*pts)
            ax.errorbar(x, y, yerr=[[a - b for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)]], fmt=marker + "-",
                        color=color, ms=5, lw=1.6, elinewidth=1.2, mec="white", mew=0.7, label=name)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xticks(range(4), [f"{q}\n{vi_int(r['min'])}–{vi_int(r['max'])}" for q, r in bounds.iterrows()], fontsize=7.5)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(L("Tỷ lệ PR (%)"))
    axes[0].legend(loc="upper left")
    for ax in axes:
        ax.set_xlabel(L("Tứ phân vị số dòng thay đổi"), fontsize=8, color=INK, labelpad=6)
    save(fig, "fig4_size_strata")


def per_agent() -> None:
    t = pd.read_csv(path("tables") / "table3_rq3.csv").sort_values("prs", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.5 + 0.45 * len(t)), sharey=True)
    for ax, (k_col, title) in zip(axes, [("share_added_edges", L("Thêm ≥1 cạnh phụ thuộc")),
                                          ("cycle_new_pkgs_rate", L("Đưa package mới vào chu trình"))]):
        if k_col == "share_added_edges":
            ks = (t["share_added_edges"] * t["prs"]).round().astype(int)
        else:
            ks = t["cycle_new_pkgs_k"]
        ci = [wilson(int(k), int(n)) for k, n in zip(ks, t["prs"])]
        rate = ks / t["prs"] * 100
        ax.errorbar(rate, range(len(t)), xerr=[rate - [c[0] * 100 for c in ci], [c[1] * 100 for c in ci] - rate],
                    fmt="o", color=AGENT, ms=6, mec="white", mew=0.8, elinewidth=1.6)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(t)), [f"{AGENT_NAMES.get(a, a)} (n = {vi_int(n)})" for a, n in zip(t["agent"], t["prs"])])
    for ax in axes:
        ax.set_xlabel(L("Tỷ lệ PR (%), khoảng tin cậy Wilson 95%"), fontsize=8, color=INK, labelpad=6)
    save(fig, "fig5_per_agent")


def edges_distribution(d: pd.DataFrame) -> None:
    """ECDF of added edges among PRs that add at least one edge (log scale): shape of the tail."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for grp in ("agent", "human"):
        color, marker, name = STYLE[grp]
        x = d[(d["group"] == grp) & (d["added_edges"] > 0)]["added_edges"].sort_values().to_numpy()
        y = (pd.Series(range(1, len(x) + 1)) / len(x)).to_numpy()
        ax.step(x, y * 100, where="post", color=color, lw=2, label=f"{name} (n = {vi_int(len(x))})")
        med = float(pd.Series(x).median())
        ax.plot([med], [50], marker=marker, color=color, ms=7, mec="white", mew=0.8, ls="none")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(L("Số cạnh phụ thuộc mới của một PR (thang log)"))
    ax.set_ylabel(L("Tỷ lệ tích lũy (%)"))
    ax.set_ylim(0, 101)
    ax.legend(loc="lower right")
    save(fig, "fig6_edges_ecdf")


def repo_scatter(d: pd.DataFrame, min_each: int = 3) -> None:
    """Per-repository mean added edges, agent vs human: how consistent the comparison is across projects."""
    per = d.groupby(["repo", "group"])["added_edges"].agg(["mean", "count"]).unstack("group")
    per = per[(per[("count", "agent")] >= min_each) & (per[("count", "human")] >= min_each)]
    xa, xh = per[("mean", "human")], per[("mean", "agent")]
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    lim = max(xa.max(), xh.max()) * 1.3 + 0.1
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(xa, xh, s=26, color=AGENT, edgecolor="white", linewidth=0.6, zorder=2)
    above, below = int((xh > xa).sum()), int((xh < xa).sum())
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ticks = [t for t in (0, 0.1, 1, 10, 100) if t <= lim]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(lambda v, _: dec(f"{v:g}")))
    ax.text(0.04, 0.96, (f"Above the diagonal (agent higher): {above} repositories\nBelow the diagonal (human higher): {below} repositories" if LANG == "en" else f"Trên đường chéo (agent cao hơn): {above} kho\nDưới đường chéo (người cao hơn): {below} kho"),
            transform=ax.transAxes, va="top", fontsize=7.5, color=INK, linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9), zorder=3)
    ax.set_xlabel(L("Cạnh mới trung bình – PR của người"))
    ax.set_ylabel(L("Cạnh mới trung bình – PR của agent"))
    save(fig, "fig7_repo_scatter")


def main() -> int:
    df = pd.read_csv(path("processed") / "pr_metrics.csv", low_memory=False)
    ok = df[df["status"] == "ok"]
    groups = ok.groupby("repo")["group"].agg(set)
    d = ok[ok["repo"].isin(groups[groups.apply(lambda s: {"agent", "human"} <= s)].index)]
    rates_by_group(d)
    size_strata(d)
    per_agent()
    edges_distribution(d)
    repo_scatter(d)
    print(json.dumps(sorted(p.name for p in path("figures").glob("*.png"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
