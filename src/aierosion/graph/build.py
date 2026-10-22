"""Build the internal dependency graph of a commit, caching parsed files by blob sha."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aierosion.data.filters import is_source_py
from aierosion.git.repo import BlobReader, ls_py

from .imports import ImportSpec, ModuleIndex, parse_import_kinds

# Graph variants, by the import kinds whose edges they keep.
# "runtime" is the main analysis: imports under `if TYPE_CHECKING:` never run, so they are left out.
VARIANTS = {"runtime": {"top", "func"}, "all": {"top", "func", "typecheck"}, "top": {"top"}}
MAIN_VARIANT = "runtime"


@dataclass
class DepGraph:
    files: set[str]
    edges: set[tuple[str, str]]  # file -> file, main variant
    missing_blobs: int = 0
    parse_errors: int = 0
    unresolved: dict = field(default_factory=dict)
    edge_kinds: dict[tuple[str, str], set[str]] = field(default_factory=dict)

    def variant(self, name: str) -> "DepGraph":
        """The same commit with only the edges kept by a graph variant (see VARIANTS)."""
        if not self.edge_kinds:  # graphs built by hand in tests carry no kinds
            return self
        keep = VARIANTS[name]
        return DepGraph(files=self.files, edges={e for e, k in self.edge_kinds.items() if k & keep},
                        missing_blobs=self.missing_blobs, parse_errors=self.parse_errors)


class GraphBuilder:
    def __init__(self, repo: Path):
        self.reader = BlobReader(repo)
        self.repo = repo
        self.parsed: dict[str, list[tuple[ImportSpec, str]] | None | str] = {}
        self._last: tuple[str, DepGraph] | None = None

    def _imports(self, blob: str) -> list[tuple[ImportSpec, str]] | None | str:
        if blob not in self.parsed:
            data = self.reader.read(blob)
            self.parsed[blob] = "missing" if data is None else parse_import_kinds(data)
        return self.parsed[blob]

    def build(self, commit: str) -> DepGraph:
        # Consecutive PRs on the first-parent chain often share a commit.
        if self._last and self._last[0] == commit:
            return self._last[1]
        files = [(p, b) for p, b in ls_py(self.repo, commit) if is_source_py(p)]
        index = ModuleIndex([p for p, _ in files])
        g = DepGraph(files={p for p, _ in files}, edges=set())
        for p, blob in files:
            specs = self._imports(blob)
            if specs == "missing":
                g.missing_blobs += 1
                continue
            if specs is None:
                g.parse_errors += 1
                continue
            for spec, kind in specs:
                for target in index.resolve(p, spec):
                    g.edge_kinds.setdefault((p, target), set()).add(kind)
        main = VARIANTS[MAIN_VARIANT]
        g.edges = {e for e, k in g.edge_kinds.items() if k & main}
        self._last = (commit, g)
        return g

    def close(self) -> None:
        self.reader.close()
