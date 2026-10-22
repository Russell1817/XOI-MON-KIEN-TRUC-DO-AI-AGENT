# Architecture erosion in AI coding agent pull requests

Code and data for the study *Architecture erosion introduced by AI coding agent pull requests: an empirical
study on the AIDev dataset* (Võ Hoàng Anh, HUFLIT, 2026).

> **Tóm tắt (VI).** Kho mã này chứa toàn bộ quy trình đo, dữ liệu đã xử lý và các bảng, hình dùng trong bài báo.
> Với mỗi pull request (PR) đã merge trong các kho Python lấy từ bộ dữ liệu AIDev, công cụ dựng đồ thị import
> giữa các module tại commit trước và sau PR, rồi so sánh PR của AI coding agent với PR của người trong cùng kho
> về cạnh phụ thuộc mới, chu trình giữa các package, độ ghép nối (Ce, I) và vi phạm phân tầng.

## Status

The measurement pipeline and its validation are public. Results (tables, result figures and the per-PR metrics)
will be added to this repository when the paper is published.

![Measurement pipeline](results/figures/fig1_pipeline.png)

*Figure 1. Measurement pipeline: from the AIDev agent PRs to the statistical analysis.*

## Research questions

- **RQ1.** Do agent PRs add dependency cycles and coupling more than human PRs in the same repositories?
- **RQ2.** Do agent PRs violate layering more often?
- **RQ3.** How do the agents (OpenAI Codex, GitHub Copilot, Cursor, Devin, Claude Code) differ?

## Method in brief

1. **Agent PRs.** AIDev-pop (repositories with more than 100 stars, Hugging Face `hao-li/AIDev`, revision
   `7840b6e`) filtered to Python repositories that are not forks, have a reuse-permitting open-source license,
   and have at least 3 merged agent PRs that change Python source code.
2. **Finding PRs in git, no GitHub API.** Each repository is cloned (bare, shallow from 200 days before its
   first agent PR, blobs larger than 1 MB skipped). On the default branch's first-parent history, a commit titled
   `Merge pull request #N ...` (merge) or ending with `(#N)` (squash) closes PR *N*. The state before the PR is
   the commit's first parent `M^1`, the state after is `M`.
   ![Identifying PRs in git history](results/figures/fig_pr_identification.png)

   *Figure 2. A merge commit or a squash commit on the first-parent chain closes a PR; the PR's change is
   the diff between `M^1` and `M` in both cases.*

3. **Two groups.** PR numbers listed in AIDev form the agent group. Other PRs merged within ±182 days of the
   repository's agent PRs form the human group, after removing bots, agent accounts and PRs with agent traces
   (agent co-authors, generated-by messages, agent branch names). At most 300 human PRs per repository are
   sampled (seed `42 + crc32(repository)`).
4. **Dependency graph.** Every Python source file (tests, docs, examples and `setup.py` excluded) is parsed with
   `ast`; each import is resolved to the longest matching internal module (relative imports, `src/` layouts,
   sub-directory roots). Renamed files are detected with `git diff -M` so that moving a file is not counted as
   new dependencies. Graphs are built at file and package (directory) level.
5. **Metrics (after − before).** New file-level edges, new package edges, new package cycles (strongly
   connected components), Δ Ce, Δ Instability of the core packages, and new layering violations
   (import-linter `layers` contracts when present, otherwise a conservative directory-name heuristic).
6. **Statistics.** Mann–Whitney U with Cliff's δ, Fisher's exact test, Kruskal–Wallis with Dunn's post-hoc test,
   Holm correction, α = 0.05. Robustness: comparison within PR-size quartiles, repository-paired Wilcoxon test,
   and re-analysis without the repository with the most agent PRs.

## Requirements

- Windows, macOS or Linux with `git` 2.45 or newer (`GIT_NO_LAZY_FETCH` is used for partial clones)
- [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the pinned dependencies from `uv.lock`
- About 20 GB of free disk space during mining (each clone is removed after its repository is processed)
- No GitHub token: everything goes through `git` and public github.com URLs

## Reproduce the results

```bash
uv python install 3.12
uv sync
uv run python scripts/00_check_env.py        # environment report
uv run pytest -q                             # 16 unit tests

uv run python scripts/01_download_aidev.py   # ~1.4 GB, pinned dataset revision
uv run python scripts/02_filter_agent_prs.py
uv run python scripts/03_mine_repos.py --workers 8   # several hours; resumable
uv run python scripts/04_layer_violations.py
uv run python scripts/05_manual_check.py     # draws the validation sample (frozen once reviewed)
uv run python scripts/05b_review_violations.py
uv run python scripts/06_statistics.py
uv run python scripts/07_figures.py
uv run python scripts/08_numbers.py          # numbers.json and claims.json
```

`03_mine_repos.py` caches each repository in `data/interim/mined/` and can be stopped and restarted.
Useful options: `--repos owner/name ...` to mine selected repositories, `--retry-errors` to re-mine
repositories that failed for transient reasons (network, machine sleep), `--limit N` for a trial run.
Repositories deleted from GitHub are reported and excluded.

`05_manual_check.py` and `05b_review_violations.py` prepare files for **manual** review: the reviewer reads
each diff or source file on GitHub and fills the `match` / `verdict` and `note` / `reason` columns. A reviewed
sample is kept as is unless `--resample` is passed; review verdicts are kept for edges flagged again.

## Pipeline

| Step | Script | Main outputs |
|---|---|---|
| 00 | `00_check_env.py` | environment report |
| 01 | `01_download_aidev.py` | `data/raw/aidev/*.parquet`, `schema.json` |
| 02 | `02_filter_agent_prs.py` | `data/interim/agent_prs.csv`, `results/logs/filter_funnel.csv` |
| 03 | `03_mine_repos.py` | `data/interim/mined/`, `data/interim/mined_prs.csv`, `results/logs/mining_repos.csv` |
| 04 | `04_layer_violations.py` | `data/processed/pr_metrics.csv`, `data/processed/violations.jsonl`, `config/layer_rules/*.json` |
| 05 | `05_manual_check.py` | `results/manual_check/manual_check.csv`, `manual_check_details.json` |
| 05b | `05b_review_violations.py` | `results/manual_check/violations_review.csv` |
| 06 | `06_statistics.py` | `results/tables/*.csv`, `stats_summary.json` |
| 07 | `07_figures.py` | `results/figures/fig3`–`fig7` (PNG 300 dpi and SVG) |
| 08 | `08_numbers.py` | `results/tables/numbers.json`, `claims.json` |

Every script appends a record (code commit, time, parameters, outputs) to `results/logs/runs.jsonl`.
All thresholds, seeds and paths are in `config/settings.yaml`.

## Main output: `data/processed/pr_metrics.csv`

One row per PR. Key columns:

| Column | Meaning |
|---|---|
| `repo`, `number`, `group`, `agent` | repository, PR number, `agent` / `human`, agent name or `Human` |
| `sha_before`, `sha_after`, `merge_style` | commits `M^1` and `M`; `merge` or `squash` |
| `status` | `ok`, `not_on_first_parent`, `no_py_source_change` or `error` |
| `loc_changed`, `files_changed`, `py_files_changed` | PR size |
| `added_edges`, `removed_edges`, `edges_per_file`, `added_pkg_edges` | dependency changes |
| `new_cycle`, `d_cyclic_pkgs`, `d_cyclic_modules` | cycle changes |
| `d_ce_mean`, `d_i_core` | coupling and instability changes |
| `new_violations`, `has_violation`, `rq2_eligible`, `layer_source` | layering |
| `renamed_py`, `rename_detection`, `missing_blobs`, `parse_errors` | data-quality flags |

Commit author names and e-mail addresses are removed from this file.

## Validation

- **Unit tests** (`tests/`): import resolution (absolute, relative, `src/` used as a layout or as a package,
  sub-directory roots, external libraries sharing a name with an internal module), the test-file filter,
  renamed files, graph metrics on a small git repository, and layer rules (heuristic and import-linter).
- **Manual check**: a random sample of 20 PRs compared with their diffs on GitHub
  (`results/manual_check/manual_check.csv`).
- **Layering violations**: every flagged edge read in the source code (`results/manual_check/violations_review.csv`).
- **Claims**: every qualitative statement of the paper is checked against the results in
  `results/tables/claims.json`.

Problems found by these checks during the study, and how they were fixed, are listed in the project log.

## Figures

| Figure | Source |
|---|---|
| `fig1_pipeline`, `fig_pr_identification`, `fig2_metric_example` | draw.io sources in `results/figures/src/`; export with `uv run python tools/export_drawio.py` (needs draw.io desktop) |
| `fig3_rates_by_group` to `fig7_repo_scatter` | generated from the data by `scripts/07_figures.py` |

## Repository layout

```
config/          settings.yaml and generated per-repository layer rules
src/aierosion/   library: data filters, git access, import graph, metrics, layer rules, statistics
scripts/         numbered pipeline steps 00-08
tests/           unit tests
tools/           draw.io export; manuscript checks and Word build (author tooling, see below)
data/processed/  final per-PR metrics (data/raw and data/interim are re-created by the scripts)
results/         tables, figures, manual checks, run logs
```

`tools/md2docx.py`, `tools/build_docs.py`, `tools/word_finalize.py` and `tools/render_docx.py` build the
manuscript and report as Word/PDF files with Microsoft Word. They expect the manuscript sources, which are kept
outside this repository, and are not needed to reproduce the results. `tools/check_paper.py` checks a Markdown
manuscript (citation order, table and figure references, wording rules).

## License and data

- Code: MIT (`LICENSE`).
- Derived data and figures: CC BY 4.0 (`DATA_LICENSE.md`).
- Input data: AIDev is released under CC BY 4.0 by its authors. Only repositories with a reuse-permitting
  open-source license are analysed. This project publishes computed metrics only, never source code from
  those repositories, and removes commit author names and e-mail addresses.
- How to cite: `CITATION.cff`.

## Contact

Võ Hoàng Anh, Faculty of Information Technology, Ho Chi Minh City University of Foreign Languages –
Information Technology (HUFLIT). Questions and issues: please open an issue in this repository.
