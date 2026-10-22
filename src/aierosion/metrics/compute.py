"""Erosion metrics: difference between the dependency graph after and before a PR."""
from __future__ import annotations

import math

import networkx as nx

from aierosion.graph.build import DepGraph
from aierosion.graph.imports import package_of


def _cyclic(nodes: set[str], edges: set[tuple[str, str]]) -> tuple[set[str], set[tuple[str, str]]]:
    """Nodes in strongly connected components of size >= 2, and the edges inside them."""
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    comp = {}
    for i, scc in enumerate(nx.strongly_connected_components(g)):
        if len(scc) > 1:
            for n in scc:
                comp[n] = i
    cyc_edges = {(u, v) for u, v in edges if u in comp and comp.get(u) == comp.get(v)}
    return set(comp), cyc_edges


def package_graph(g: DepGraph) -> tuple[set[str], set[tuple[str, str]]]:
    nodes = {package_of(f) for f in g.files}
    edges = {(package_of(u), package_of(v)) for u, v in g.edges}
    return nodes, {(u, v) for u, v in edges if u != v}


def _coupling(nodes: set[str], edges: set[tuple[str, str]]) -> tuple[dict, dict]:
    ce = {n: 0 for n in nodes}
    ca = {n: 0 for n in nodes}
    for u, v in edges:
        ce[u] = ce.get(u, 0) + 1
        ca[v] = ca.get(v, 0) + 1
    return ce, ca


def _instability(ce: int, ca: int) -> float:
    return ce / (ce + ca) if ce + ca else math.nan


def _mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def apply_renames(g: DepGraph, renames: dict[str, str]) -> DepGraph:
    """The 'before' graph with renamed files under their new path, so a move is not a new edge."""
    if not renames:
        return g
    r = lambda f: renames.get(f, f)  # noqa: E731
    return DepGraph(files={r(f) for f in g.files}, edges={(r(u), r(v)) for u, v in g.edges},
                    missing_blobs=g.missing_blobs, parse_errors=g.parse_errors)


def compute(
    before: DepGraph, after: DepGraph, changed_py: set[str], core_fraction: float,
    renames: dict[str, str] | None = None,
) -> tuple[dict, list, list]:
    before = apply_renames(before, renames or {})
    added = after.edges - before.edges
    removed = before.edges - after.edges
    cyc_mod0, _ = _cyclic(before.files, before.edges)
    cyc_mod1, _ = _cyclic(after.files, after.edges)

    pn0, pe0 = package_graph(before)
    pn1, pe1 = package_graph(after)
    cyc_pkg0, cyc_pe0 = _cyclic(pn0, pe0)
    cyc_pkg1, cyc_pe1 = _cyclic(pn1, pe1)

    ce0, ca0 = _coupling(pn0, pe0)
    ce1, ca1 = _coupling(pn1, pe1)
    touched = {package_of(f) for f in changed_py} & pn0 & pn1
    d_ce = [ce1[p] - ce0[p] for p in touched]

    ranked = sorted((p for p in pn0 if ca0[p] > 0), key=lambda p: ca0[p], reverse=True)
    core = ranked[: max(1, math.ceil(core_fraction * len(ranked)))] if ranked else []
    d_i = [
        _instability(ce1[p], ca1[p]) - _instability(ce0[p], ca0[p]) for p in core if p in pn1
    ]

    added_pkg = pe1 - pe0
    new_cycle = bool(cyc_pe1 - cyc_pe0)
    # Split of new_cycle: a package that was not in any cycle now is (a cycle formed or grew),
    # versus a new edge inside a cycle whose packages were all cyclic already.
    cycle_new_pkgs = bool(cyc_pkg1 - cyc_pkg0)
    return {
        "modules_before": len(before.files),
        "packages_before": len(pn0),
        "added_edges": len(added),
        "removed_edges": len(removed),
        "edges_per_file": len(added) / max(1, len(changed_py)),
        "added_pkg_edges": len(added_pkg),
        "cyclic_modules_before": len(cyc_mod0),
        "d_cyclic_modules": len(cyc_mod1) - len(cyc_mod0),
        "cyclic_pkgs_before": len(cyc_pkg0),
        "d_cyclic_pkgs": len(cyc_pkg1) - len(cyc_pkg0),
        "new_cycle": int(new_cycle),
        "cycle_new_pkgs": int(cycle_new_pkgs),
        "cycle_existing_only": int(new_cycle and not cycle_new_pkgs),
        "pkgs_touched": len({package_of(f) for f in changed_py}),
        "d_ce_mean": _mean([float(x) for x in d_ce]),
        "d_i_core": _mean(d_i),
        "missing_blobs": before.missing_blobs + after.missing_blobs,
        "parse_errors": before.parse_errors + after.parse_errors,
    }, sorted(added), sorted(added_pkg)


def variant_metrics(before: DepGraph, after: DepGraph, renames: dict[str, str] | None, suffix: str) -> dict:
    """The main indicators on another graph variant, for the sensitivity analysis."""
    before = apply_renames(before, renames or {})
    pn0, pe0 = package_graph(before)
    pn1, pe1 = package_graph(after)
    cyc_pkg0, cyc_pe0 = _cyclic(pn0, pe0)
    cyc_pkg1, cyc_pe1 = _cyclic(pn1, pe1)
    return {f"added_edges_{suffix}": len(after.edges - before.edges),
            f"added_pkg_edges_{suffix}": len(pe1 - pe0),
            f"new_cycle_{suffix}": int(bool(cyc_pe1 - cyc_pe0)),
            f"cycle_new_pkgs_{suffix}": int(bool(cyc_pkg1 - cyc_pkg0))}
