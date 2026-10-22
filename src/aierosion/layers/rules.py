"""Layer assignment: import-linter `layers` contracts when present, else directory names."""
from __future__ import annotations

import configparser
import re
import tomllib

from aierosion.graph.imports import module_parts

# Rank 0 is the top layer. Only unambiguous directory names are used: words such as
# endpoints, handlers, commands, web, persistence or storage also name non-layer concepts
# (API clients, event handlers, runtime commands, security "persistence") and were dropped
# after the manual review of flagged violations.
HEURISTIC_LAYERS = {
    "presentation": (0, {"views", "api", "apis", "routes", "routers", "cli", "ui", "controllers", "frontend"}),
    "domain": (1, {"services", "service", "domain", "usecases", "use_cases", "business"}),
    "data": (2, {"models", "db", "database", "dal", "repositories", "orm"}),
}


def _package_dirs(files: list[str]) -> set[str]:
    return {f.rsplit("/", 1)[0] for f in files if f.endswith("/__init__.py")}


def _heuristic_layer(path: str, pkg_dirs: set[str]) -> tuple[str, int] | None:
    """Layer from directory names below the top-level package (file names are not used)."""
    dirs = path.split("/")[:-1]
    if dirs and dirs[0] == "src":
        dirs = dirs[1:]
        prefix = "src/"
    else:
        prefix = ""
    top = next((i for i in range(len(dirs)) if prefix + "/".join(dirs[: i + 1]) in pkg_dirs), 0)
    # The top-level package (or project folder) name is never a layer, e.g. a backend package named "api".
    for comp in reversed(dirs[top + 1:]):
        for name, (rank, words) in HEURISTIC_LAYERS.items():
            if comp.lower() in words:
                return name, rank
    return None


def _layer_list(raw) -> list[str]:
    if isinstance(raw, str):
        raw = [line for line in raw.splitlines()]
    return [x.strip() for x in raw if x and x.strip()]


def parse_import_linter(configs: dict[str, str]) -> list[list[list[str]]]:
    """Layer contracts, each a list of ranks, each rank a list of dotted module names."""
    contracts = []

    def add(layers_raw, containers_raw):
        containers = _layer_list(containers_raw) if containers_raw else [""]
        ranks = []
        for entry in _layer_list(layers_raw):
            names = [n.strip().strip("()") for n in re.split(r"[|:]", entry) if n.strip()]
            ranks.append([f"{c}.{n}" if c else n for c in containers for n in names])
        if len(ranks) >= 2:
            contracts.append(ranks)

    for fname, text in configs.items():
        if fname.endswith(".toml"):
            try:
                data = tomllib.loads(text)
            except tomllib.TOMLDecodeError:
                continue
            for c in data.get("tool", {}).get("importlinter", {}).get("contracts", []):
                if c.get("type") == "layers":
                    add(c.get("layers", []), c.get("containers"))
        else:
            cp = configparser.ConfigParser(interpolation=None)
            try:
                cp.read_string(text)
            except configparser.Error:
                continue
            for sec in cp.sections():
                if sec.startswith("importlinter:contract") and cp[sec].get("type") == "layers":
                    add(cp[sec].get("layers", ""), cp[sec].get("containers"))
    return contracts


class LayerRules:
    def __init__(self, files: list[str], import_linter: dict[str, str]):
        self.pkg_dirs = _package_dirs(files)
        self.contracts = parse_import_linter(import_linter)
        self.source = "import-linter" if self.contracts else "directory-heuristic"
        layers_present = {self.layer(f) for f in files} - {None}
        if self.source == "import-linter":
            self.eligible = bool(layers_present)
        else:
            self.eligible = len({name for name, _ in layers_present}) >= 2
        self.layers_present = sorted(str(x) for x in layers_present)

    def layer(self, path: str):
        if self.contracts:
            mod = ".".join(module_parts(path))
            for ci, ranks in enumerate(self.contracts):
                for ri, names in enumerate(ranks):
                    if any(mod == n or mod.startswith(n + ".") for n in names):
                        return (f"contract{ci}", ri)
            return None
        return _heuristic_layer(path, self.pkg_dirs)

    def is_violation(self, src: str, dst: str) -> bool:
        a, b = self.layer(src), self.layer(dst)
        if a is None or b is None:
            return False
        if self.contracts:
            if a[0] != b[0]:  # layers of different contracts are not ordered
                return False
            return a[1] > b[1]
        # Directory heuristic: only an import *into* the presentation layer from below is a violation.
        # Data -> domain is not flagged: repositories commonly depend on domain models
        # (dependency inversion in DDD / clean architecture), so that direction is not a reliable signal.
        return b[1] == 0 and a[1] > 0
