"""Non-parametric tests and effect sizes used for RQ1-RQ3."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats


def cliffs_delta(x, y) -> float:
    """P(X > Y) - P(X < Y), computed from the Mann-Whitney U statistic of x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2 * u / (len(x) * len(y)) - 1


def compare_two(x, y) -> dict:
    x, y = pd.Series(x).dropna(), pd.Series(y).dropna()
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    return {
        "n_agent": len(x), "n_human": len(y),
        "median_agent": float(x.median()), "median_human": float(y.median()),
        "mean_agent": float(x.mean()), "mean_human": float(y.mean()),
        "share_pos_agent": float((x > 0).mean()), "share_pos_human": float((y > 0).mean()),
        "U": float(res.statistic), "p": float(res.pvalue), "cliffs_delta": cliffs_delta(x, y),
    }


def compare_rates(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Fisher's exact test on two proportions; odds ratio with the Haldane correction if a cell is 0."""
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    res = stats.fisher_exact(table)
    a, b, c, d = (v + 0.5 for v in (k1, n1 - k1, k2, n2 - k2)) if 0 in (k1, n1 - k1, k2, n2 - k2) else (
        k1, n1 - k1, k2, n2 - k2)
    return {"k_agent": k1, "n_agent": n1, "k_human": k2, "n_human": n2,
            "rate_agent": k1 / n1 if n1 else math.nan, "rate_human": k2 / n2 if n2 else math.nan,
            "odds_ratio": (a * d) / (b * c), "p": float(res.pvalue)}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def holm(pvalues: list[float]) -> list[float]:
    """Holm-adjusted p-values (step-down, monotone)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = min(1.0, running)
    return adj


def kruskal_dunn(df: pd.DataFrame, value: str, group: str) -> tuple[dict, pd.DataFrame]:
    groups = [g[value].dropna().values for _, g in df.groupby(group)]
    h = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    # Epsilon-squared effect size for Kruskal-Wallis: H / (n - 1).
    res = {"H": float(h.statistic), "p": float(h.pvalue), "k": len(groups), "n": n,
           "epsilon_sq": float(h.statistic) / (n - 1)}
    dunn = sp.posthoc_dunn(df, val_col=value, group_col=group, p_adjust="holm")
    return res, dunn
