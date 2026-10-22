"""Static import extraction and resolution to internal modules."""
from __future__ import annotations

import ast
import sys
import warnings
from collections import defaultdict

# (level, module, imported names); level 0 = absolute import.
ImportSpec = tuple[int, str, tuple[str, ...]]

STDLIB = frozenset(sys.stdlib_module_names)


# Where an import statement sits: module level ("top"), inside a function or lambda ("func"),
# or under `if TYPE_CHECKING:` ("typecheck", never executed at run time).
IMPORT_KINDS = ("top", "func", "typecheck")


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.found: list[tuple[ImportSpec, str]] = []
        self.kind = "top"

    def _inside(self, node, kind):
        outer, self.kind = self.kind, ("typecheck" if "typecheck" in (self.kind, kind) else kind)
        self.generic_visit(node)
        self.kind = outer

    def visit_FunctionDef(self, node):
        self._inside(node, "func")

    visit_AsyncFunctionDef = visit_Lambda = visit_FunctionDef

    def visit_If(self, node):
        if _is_type_checking(node.test):
            for child in node.body:
                outer, self.kind = self.kind, "typecheck"
                self.visit(child)
                self.kind = outer
            for child in node.orelse:
                self.visit(child)
        else:
            self.generic_visit(node)

    def visit_Import(self, node):
        self.found.extend(((0, a.name, ()), self.kind) for a in node.names)

    def visit_ImportFrom(self, node):
        self.found.append(((node.level, node.module or "", tuple(a.name for a in node.names)), self.kind))


def parse_import_kinds(source: bytes) -> list[tuple[ImportSpec, str]] | None:
    """Every import statement in a file with its kind (see IMPORT_KINDS). None on syntax error."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    v = _ImportVisitor()
    v.visit(tree)
    return v.found


def parse_imports(source: bytes) -> list[ImportSpec] | None:
    """All import statements in a file, including those inside functions. None on syntax error."""
    found = parse_import_kinds(source)
    return None if found is None else [spec for spec, _ in found]


def module_parts(path: str) -> tuple[str, ...]:
    """'src/pkg/sub/mod.py' -> ('pkg', 'sub', 'mod'); '__init__.py' names its package."""
    parts = path[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if len(parts) > 1 and parts[0] == "src":
        parts = parts[1:]
    return tuple(parts)


def package_of(path: str) -> str:
    """Package (directory) of a file, as a dotted name; '<root>' for top-level modules."""
    parts = module_parts(path)
    if not path.endswith("__init__.py"):
        parts = parts[:-1]
    return ".".join(parts) or "<root>"


class ModuleIndex:
    """Maps dotted import names to internal files at one commit."""

    def __init__(self, paths: list[str]):
        self.by_name: dict[str, str] = {}
        # suffix -> root directory prefix -> full dotted names
        suffix: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for p in paths:
            parts = module_parts(p)
            if not parts:
                continue
            name = ".".join(parts)
            # Prefer the package __init__ over a same-named module if both exist.
            if name not in self.by_name or p.endswith("__init__.py"):
                self.by_name[name] = p
            if p.startswith("src/"):
                # Some projects use src/ as a real package (`from src.monitoring import metrics`).
                alias = "src." + name
                if alias not in self.by_name or p.endswith("__init__.py"):
                    self.by_name[alias] = p
            for i in range(1, len(parts)):
                suffix[".".join(parts[i:])]["/".join(parts[:i])].add(name)
        self.by_suffix = suffix
        # Directories that are packages (contain __init__.py), in the same src-stripped namespace.
        self.pkg_dirs = {"/".join(module_parts(p)) for p in paths if p.endswith("__init__.py")}

    def _lookup(self, name: str, importer: str | None) -> str | None:
        if name in self.by_name:
            return self.by_name[name]
        if importer is None or name not in self.by_suffix:
            return None
        # Code under a sub-directory root (e.g. backend/app/...) imports "app.x" with backend/ on
        # sys.path. Accept a suffix match only when the importing file lives under that same root,
        # so that an external library sharing a name with some nested module is not linked.
        # The deepest enclosing root wins; an ambiguous match at that root is rejected.
        # A root is a plausible sys.path entry only if it is not itself a package, or if it is the
        # importing script's own directory (sys.path[0] when a script is run directly).
        roots = self.by_suffix[name]
        dirs = (importer[4:] if importer.startswith("src/") else importer).split("/")[:-1]
        own_dir = "/".join(dirs)
        for k in range(len(dirs), 0, -1):
            root = "/".join(dirs[:k])
            hits = roots.get(root)
            if hits and (root not in self.pkg_dirs or root == own_dir):
                return self.by_name[next(iter(hits))] if len(hits) == 1 else None
        return None

    def _longest_prefix(self, parts: list[str], importer: str | None) -> str | None:
        for k in range(len(parts), 0, -1):
            hit = self._lookup(".".join(parts[:k]), importer)
            if hit:
                return hit
        return None

    def resolve(self, importer: str, spec: ImportSpec) -> set[str]:
        """Internal files that one import statement depends on."""
        level, module, names = spec
        if level == 0:
            base = module.split(".")
            suffix_from = importer if base[0] not in STDLIB else None
        else:
            pkg = list(module_parts(importer))
            if not importer.endswith("__init__.py"):
                pkg = pkg[:-1]
            if level - 1 > len(pkg):
                return set()
            pkg = pkg[: len(pkg) - (level - 1)]
            base = pkg + (module.split(".") if module else [])
            suffix_from = None
        targets = set()
        if names:
            for n in names:
                if n == "*":
                    continue
                hit = self._lookup(".".join(base + [n]), suffix_from) if base or n else None
                if hit:
                    targets.add(hit)
        if not targets and base:
            hit = self._longest_prefix(base, suffix_from)
            if hit:
                targets.add(hit)
        targets.discard(importer)
        return targets
