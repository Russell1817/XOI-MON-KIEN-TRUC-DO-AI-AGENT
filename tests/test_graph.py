"""Unit tests for import resolution, graph building and metrics on a small git repository."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest

from aierosion.data.filters import is_source_py
from aierosion.git import repo as gitrepo
from aierosion.graph.build import DepGraph, GraphBuilder
from aierosion.graph.imports import ModuleIndex, module_parts, package_of, parse_import_kinds, parse_imports
from aierosion.metrics.compute import compute, variant_metrics


def test_module_parts_and_package():
    assert module_parts("src/pkg/sub/mod.py") == ("pkg", "sub", "mod")
    assert module_parts("pkg/__init__.py") == ("pkg",)
    assert package_of("pkg/sub/mod.py") == "pkg.sub"
    assert package_of("pkg/__init__.py") == "pkg"
    assert package_of("main.py") == "<root>"


def test_source_filter():
    assert is_source_py("pkg/mod.py")
    assert not is_source_py("tests/test_mod.py")
    assert not is_source_py("pkg/test_mod.py")
    assert not is_source_py("pkg/chatterbox_tests.py")  # pylabrobot naming
    assert not is_source_py("docs/conf.py")
    assert not is_source_py("setup.py")
    assert not is_source_py("pkg/mod.txt")


def test_parse_imports_and_syntax_error():
    specs = parse_imports(b"import os\nfrom . import a\nfrom ..b import c, d\ndef f():\n    import pkg.x\n")
    assert (0, "os", ()) in specs
    assert (1, "", ("a",)) in specs
    assert (2, "b", ("c", "d")) in specs
    assert (0, "pkg.x", ()) in specs  # imports inside functions are kept
    assert parse_imports(b"def broken(:\n") is None


@pytest.fixture
def index():
    return ModuleIndex([
        "pkg/__init__.py", "pkg/a.py", "pkg/sub/__init__.py", "pkg/sub/b.py",
        "src/lib/c.py", "backend/app/core.py", "pkg/logging.py",
    ])


def test_resolve_absolute(index):
    assert index.resolve("pkg/a.py", (0, "pkg.sub.b", ())) == {"pkg/sub/b.py"}
    # attribute access past a module resolves to the module
    assert index.resolve("pkg/a.py", (0, "pkg.sub.b.func", ())) == {"pkg/sub/b.py"}
    assert index.resolve("pkg/a.py", (0, "pkg.sub", ("b",))) == {"pkg/sub/b.py"}
    assert index.resolve("pkg/a.py", (0, "pkg.sub", ("name",))) == {"pkg/sub/__init__.py"}
    assert index.resolve("pkg/a.py", (0, "lib.c", ())) == {"src/lib/c.py"}  # src/ layout
    assert index.resolve("pkg/a.py", (0, "requests", ())) == set()  # third party


def test_resolve_suffix_and_stdlib_guard(index):
    # code under a sub-directory root resolves when the importer shares that root
    assert index.resolve("backend/app/core.py", (0, "app.core", ())) == set()  # self-import dropped
    assert index.resolve("backend/main.py", (0, "app.core", ())) == {"backend/app/core.py"}
    # ... but not from outside the root (likely an external library with the same name)
    assert index.resolve("pkg/a.py", (0, "app.core", ())) == set()
    # stdlib names never resolve by suffix to an internal module with the same name
    assert index.resolve("pkg/a.py", (0, "logging", ())) == set()
    assert index.resolve("pkg/a.py", (0, "pkg.logging", ())) == {"pkg/logging.py"}


def test_external_library_sharing_nested_name():
    # autogen case: `import docker` must not resolve to an internal sub-package named docker,
    # sqlmesh case: `import sqlglot` must not resolve to web/client/.../sqlglot/sqlglot.py.
    idx = ModuleIndex([
        "autogen_ext/__init__.py", "autogen_ext/code_executors/docker/__init__.py",
        "autogen_ext/teams/magentic_one.py", "web/client/src/workers/sqlglot/sqlglot.py", "core/model.py",
    ])
    assert idx.resolve("autogen_ext/teams/magentic_one.py", (0, "docker", ())) == set()
    assert idx.resolve("core/model.py", (0, "sqlglot", ())) == set()
    assert idx.resolve("autogen_ext/teams/magentic_one.py", (0, "autogen_ext.code_executors.docker", ())) == {
        "autogen_ext/code_executors/docker/__init__.py"}


def test_external_package_named_like_a_subpackage():
    # crewAI case: `from agents import Agent` (OpenAI Agents SDK) must not resolve to src/crewai/agents,
    # because crewai/ is a package, not a sys.path root.
    idx = ModuleIndex([
        "src/crewai/__init__.py", "src/crewai/agents/__init__.py",
        "src/crewai/agents/agent_adapters/__init__.py", "src/crewai/agents/agent_adapters/openai/adapter.py",
    ])
    assert idx.resolve("src/crewai/agents/agent_adapters/openai/adapter.py", (0, "agents", ("Agent",))) == set()
    assert idx.resolve("src/crewai/agents/agent_adapters/openai/adapter.py", (0, "crewai.agents", ())) == {
        "src/crewai/agents/__init__.py"}
    # A script importing a sibling folder of its own directory (sys.path[0]) still resolves.
    idx2 = ModuleIndex(["tool/__init__.py", "tool/manual_build.py", "tool/build/__init__.py", "tool/build/common.py"])
    assert idx2.resolve("tool/manual_build.py", (0, "build.common", ("sha",))) == {"tool/build/common.py"}


def test_src_used_as_a_real_package():
    # AGI-Alpha-Agent case: `from src.monitoring import metrics` with src/__init__.py present.
    idx = ModuleIndex(["src/__init__.py", "src/monitoring/__init__.py", "src/monitoring/metrics.py", "app/x.py"])
    assert idx.resolve("app/x.py", (0, "src.monitoring", ("metrics",))) == {"src/monitoring/metrics.py"}
    assert idx.resolve("app/x.py", (0, "monitoring.metrics", ())) == {"src/monitoring/metrics.py"}


def test_resolve_relative(index):
    assert index.resolve("pkg/sub/b.py", (2, "", ("a",))) == {"pkg/a.py"}
    assert index.resolve("pkg/sub/b.py", (1, "", ("missing",))) == {"pkg/sub/__init__.py"}
    assert index.resolve("pkg/__init__.py", (1, "sub", ("b",))) == {"pkg/sub/b.py"}
    assert index.resolve("pkg/a.py", (5, "", ("x",))) == set()  # beyond top level


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo: Path, files: dict[str, str], msg: str) -> str:
    for name, text in files.items():
        p = repo / name
        if text is None:
            p.unlink()
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def test_graph_and_metrics_on_git_repo(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    c0 = _commit(repo, {
        "app/__init__.py": "",
        "app/views/__init__.py": "",
        "app/views/page.py": "from app.services import logic\n",
        "app/services/__init__.py": "",
        "app/services/logic.py": "from app.models import user\n",
        "app/models/__init__.py": "",
        "app/models/user.py": "import os\n",
        "tests/test_x.py": "from app.views import page\n",
    }, "init")
    # The PR adds a back edge models -> views (a package cycle and a layering violation).
    c1 = _commit(repo, {"app/models/user.py": "import os\nfrom app.views.page import render\n"}, "Add render (#7)")

    commits = gitrepo.first_parent_pr_commits(repo)
    assert [(c.number, c.sha, c.style) for c in commits] == [(7, c1, "squash")]
    assert commits[0].parents == [c0]

    b = GraphBuilder(repo)
    g0, g1 = b.build(c0), b.build(c1)
    b.close()
    assert "tests/test_x.py" not in g0.files
    assert ("app/views/page.py", "app/services/logic.py") in g0.edges
    assert ("app/services/logic.py", "app/models/user.py") in g0.edges

    changed = {f for _, f in gitrepo.changed_files(repo, c0, c1)}
    m, added, added_pkg = compute(g0, g1, changed, 0.10)
    assert added == [("app/models/user.py", "app/views/page.py")]
    assert added_pkg == [("app.models", "app.views")]
    assert m["added_edges"] == 1 and m["removed_edges"] == 0
    assert m["cyclic_pkgs_before"] == 0 and m["d_cyclic_pkgs"] == 3 and m["new_cycle"] == 1
    assert m["d_cyclic_modules"] == 3
    assert m["cycle_new_pkgs"] == 1 and m["cycle_existing_only"] == 0  # the cycle is new
    assert m["pkgs_touched"] == 1
    assert m["d_ce_mean"] == 1.0  # app.models gains one efferent package
    assert not math.isnan(m["d_i_core"])
    assert gitrepo.loc_changed(repo, c0, c1) == 1


def test_core_ties_broken_by_name():
    # a, b and c all have Ca = 1; with 10% the core is one package, and it must be "a" on every run.
    files = {"x/m.py", "a/m.py", "b/m.py", "c/m.py"}
    g0 = DepGraph(files=files, edges={("x/m.py", "a/m.py"), ("x/m.py", "b/m.py"), ("x/m.py", "c/m.py")})
    g1 = DepGraph(files=files, edges=g0.edges | {("a/m.py", "c/m.py")})
    m, _, _ = compute(g0, g1, {"a/m.py"}, 0.10)
    assert m["d_i_core"] == 0.5  # I(a) goes from 0 to 1/2


def test_moved_file_adds_no_edges(tmp_path):
    # marimo case: a file moved to another folder keeps its imports; they are not new edges.
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    body = "from app import core\n" + "\n".join(f"x{i} = {i}" for i in range(30)) + "\n"
    c0 = _commit(repo, {"app/__init__.py": "", "app/core.py": "", "app/old/__init__.py": "", "app/old/tool.py": body}, "init")
    (repo / "app/new").mkdir()
    _git(repo, "mv", "app/old/tool.py", "app/new/tool.py")
    c1 = _commit(repo, {"app/new/__init__.py": ""}, "Move tool (#3)")
    moved = gitrepo.renames(repo, c0, c1)
    assert moved == {"app/old/tool.py": "app/new/tool.py"}
    b = GraphBuilder(repo)
    g0, g1 = b.build(c0), b.build(c1)
    b.close()
    changed = {f for _, f in gitrepo.changed_files(repo, c0, c1)}
    m, added, _ = compute(g0, g1, changed, 0.10, renames=moved)
    assert added == [] and m["added_edges"] == 0
    m_naive, added_naive, _ = compute(g0, g1, changed, 0.10)
    assert added_naive == [("app/new/tool.py", "app/core.py")]  # what the old method counted


def test_bot_pr_merged_by_a_human_is_detected(tmp_path):
    # A merge commit is authored by whoever merged; the PR author is the first commit on the branch.
    from aierosion.data.filters import has_agent_trace, is_bot_pr

    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, {"app/__init__.py": ""}, "init")
    _git(repo, "checkout", "-q", "-b", "pre-commit-ci-update-config")
    (repo / "app/x.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=pre-commit-ci[bot]", "-c", "user.email=66853113+pre-commit-ci[bot]@users.noreply.github.com",
         "commit", "-q", "-m", "[pre-commit.ci] auto fixes")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "-c", "user.name=Maintainer", "-c", "user.email=m@example.org", "merge", "-q", "--no-ff",
         "-m", "Merge pull request #9 from org/pre-commit-ci-update-config", "pre-commit-ci-update-config")
    (c,) = gitrepo.first_parent_pr_commits(repo)
    assert c.style == "merge" and c.author == "Maintainer"
    author = gitrepo.pr_author(repo, c)
    assert author.startswith("pre-commit-ci[bot]")
    assert is_bot_pr(author, c.message.split("\n", 1)[0])
    assert not is_bot_pr("Jane Doe <jane@example.org>", "Fix parser (#10)")
    assert has_agent_trace("Co-authored-by: Claude <noreply@anthropic.com>")
    assert not has_agent_trace("Refactor the parser")


def test_import_kinds():
    src = (b"import a\n"
           b"from typing import TYPE_CHECKING\n"
           b"if TYPE_CHECKING:\n    import b\n    def f():\n        import c\nelse:\n    import d\n"
           b"if typing.TYPE_CHECKING:\n    import e\n"
           b"def g():\n    import h\n"
           b"class K:\n    import m\n")
    kinds = {spec[1]: kind for spec, kind in parse_import_kinds(src)}
    assert kinds == {"a": "top", "typing": "top", "b": "typecheck", "c": "typecheck", "d": "top",
                     "e": "typecheck", "h": "func", "m": "top"}
    assert parse_import_kinds(b"def broken(:\n") is None


def test_graph_variants_and_existing_cycle(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    c0 = _commit(repo, {
        "app/__init__.py": "",
        "app/a/__init__.py": "", "app/a/x.py": "from app.b import y\n",
        "app/b/__init__.py": "", "app/b/y.py": "from app.d import u\n",
        "app/d/__init__.py": "", "app/d/u.py": "from app.a import x\n",
        "app/c/__init__.py": "", "app/c/z.py": "",
    }, "init")
    # Packages a -> b -> d -> a already form a cycle. The PR adds b -> a inside it, a type-only
    # import c -> a, and a function-level import a -> c.
    c1 = _commit(repo, {
        "app/b/y.py": "from app.d import u\nfrom app.a import x\n",
        "app/c/z.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from app.a import x\n",
        "app/a/x.py": "from app.b import y\ndef f():\n    from app.c import z\n",
    }, "Edges (#3)")
    b = GraphBuilder(repo)
    g0, g1 = b.build(c0), b.build(c1)
    b.close()
    changed = {f for _, f in gitrepo.changed_files(repo, c0, c1)}
    m, added, _ = compute(g0, g1, changed, 0.10)
    # main (runtime) variant: y -> x and x -> z; the type-only z -> x is left out
    assert set(added) == {("app/b/y.py", "app/a/x.py"), ("app/a/x.py", "app/c/z.py")}
    assert m["new_cycle"] == 1 and m["cycle_new_pkgs"] == 0 and m["cycle_existing_only"] == 1
    v_all = variant_metrics(g0.variant("all"), g1.variant("all"), None, "all")
    assert v_all["added_edges_all"] == 3 and v_all["cycle_new_pkgs_all"] == 1  # c joins the cycle
    v_top = variant_metrics(g0.variant("top"), g1.variant("top"), None, "top")
    assert v_top["added_edges_top"] == 1 and v_top["cycle_new_pkgs_top"] == 0
