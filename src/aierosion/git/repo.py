"""Git access without a working tree: clone, first-parent PR commits, trees, blobs, diffs."""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Never fetch missing blobs on demand; files above the clone size limit are reported as missing.
_ENV = {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"}

MERGE_RE = re.compile(r"^Merge pull request #(\d+) from ")
SQUASH_RE = re.compile(r"\(#(\d+)\)\s*$")


def git(repo: Path, *args: str, check: bool = True, timeout: int = 600) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, env=_ENV, timeout=timeout
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {r.stderr.decode('utf-8', 'replace')[:300]}")
    return r.stdout.decode("utf-8", "replace")


def clone(full_name: str, dest: Path, shallow_since: str, blob_limit: str, timeout: int) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git", "clone", "--bare", "--single-branch", "--no-tags",
        f"--filter=blob:limit={blob_limit}", f"--shallow-since={shallow_since}",
        f"https://github.com/{full_name}.git", str(dest),
    ]
    r = subprocess.run(cmd, capture_output=True, env=_ENV, timeout=timeout)
    if r.returncode != 0:
        remove(dest)
        raise RuntimeError(f"clone failed: {r.stderr.decode('utf-8', 'replace')[:300]}")


def remove(path: Path) -> None:
    def _onerror(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if path.exists():
        shutil.rmtree(path, onexc=_onerror)


@dataclass
class PRCommit:
    number: int
    sha: str
    parents: list[str]
    author: str
    email: str
    date: str
    message: str

    @property
    def style(self) -> str:
        return "merge" if len(self.parents) > 1 else "squash"


def first_parent_pr_commits(repo: Path) -> list[PRCommit]:
    """Commits on the default branch's first-parent chain that close a PR (merge or squash)."""
    out = git(repo, "log", "--first-parent", "--format=%H%x00%P%x00%an%x00%ae%x00%cI%x00%B%x1e", "HEAD")
    result = []
    for rec in out.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, parents, author, email, date, body = rec.split("\x00", 5)
        subject = body.split("\n", 1)[0].strip()
        m = MERGE_RE.match(subject) or SQUASH_RE.search(subject)
        if not m or not parents:  # no parent = shallow boundary, cannot diff
            continue
        result.append(PRCommit(int(m.group(1)), sha, parents.split(), author, email, date, body))
    return result


def is_shallow_boundary(repo: Path, sha: str) -> bool:
    shallow = repo / "shallow"
    return shallow.exists() and sha in shallow.read_text().split()


def commit_messages(repo: Path, rev_range: str) -> str:
    return git(repo, "log", "--format=%an <%ae>%n%B", rev_range, check=False)


def pr_author(repo: Path, c: "PRCommit") -> str:
    """'name <email>' of the PR's author.

    A squash commit is authored by the PR author. A merge commit is authored by whoever merged, so the
    author of the first commit on the PR branch (M^1..M^2, oldest first) is used instead.
    """
    if c.style != "merge":
        return f"{c.author} <{c.email}>"
    out = git(repo, "log", "--reverse", "--format=%an <%ae>", f"{c.parents[0]}..{c.parents[1]}", check=False)
    first = out.splitlines()[0] if out.strip() else ""
    return first or f"{c.author} <{c.email}>"


def ls_py(repo: Path, commit: str) -> list[tuple[str, str]]:
    """(path, blob sha) of every .py file in a commit."""
    out = git(repo, "ls-tree", "-r", "-z", commit)
    files = []
    for entry in out.split("\x00"):
        if not entry:
            continue
        meta, p = entry.split("\t", 1)
        mode, typ, sha = meta.split()
        if typ == "blob" and p.endswith(".py"):
            files.append((p, sha))
    return files


def changed_files(repo: Path, before: str, after: str) -> list[tuple[str, str]]:
    """(status, path) of files changed between two commits, renames as delete + add."""
    out = git(repo, "diff", "--name-status", "--no-renames", "-z", before, after)
    parts = [p for p in out.split("\x00") if p]
    return [(parts[i], parts[i + 1]) for i in range(0, len(parts) - 1, 2)]


def renames(repo: Path, before: str, after: str) -> dict[str, str] | None:
    """{old path: new path} for files git detects as renamed (similarity >= 50%).

    None if rename detection is impossible because a blob is missing from the partial clone.
    """
    try:
        out = git(repo, "diff", "--name-status", "-M", "-z", before, after)
    except RuntimeError:
        return None
    parts = [p for p in out.split("\x00") if p]
    result, i = {}, 0
    while i < len(parts):
        status = parts[i]
        if status.startswith(("R", "C")):
            if status.startswith("R"):
                result[parts[i + 1]] = parts[i + 2]
            i += 3
        else:
            i += 2
    return result


def loc_changed(repo: Path, before: str, after: str) -> int | None:
    """Added + deleted lines (renames detected, so a pure move counts 0); None if a blob is missing."""
    try:
        out = git(repo, "diff", "--numstat", "-M", before, after)
    except RuntimeError:
        return None
    total = 0
    for line in out.splitlines():
        a, d, _ = line.split("\t", 2)
        if a != "-":
            total += int(a) + int(d)
    return total


class BlobReader:
    """Persistent `git cat-file --batch` process."""

    def __init__(self, repo: Path):
        self.proc = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_ENV,
        )

    def read(self, sha: str) -> bytes | None:
        self.proc.stdin.write(sha.encode() + b"\n")
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().split()
        if len(header) < 3 or header[1] == b"missing":
            return None
        data = self.proc.stdout.read(int(header[2]))
        self.proc.stdout.read(1)
        return data

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=30)
