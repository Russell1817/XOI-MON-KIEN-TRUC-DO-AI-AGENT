"""Step 04: count new layering violations per PR and write the final metrics table."""
from __future__ import annotations

import json
import sys

import pandas as pd

from aierosion.config import ROOT, path
from aierosion.layers.rules import LayerRules
from aierosion.runlog import log_run


def main() -> int:
    mined_dir = path("interim") / "mined"
    rules_dir = ROOT / "config" / "layer_rules"
    df = pd.read_csv(path("interim") / "mined_prs.csv")

    repo_rows, viol = [], {}
    selected = set(pd.read_csv(path("interim") / "agent_prs.csv")["full_name"])
    for layout_file in sorted(mined_dir.glob("*.layout.json")):
        slug = layout_file.name.removesuffix(".layout.json")
        repo = slug.replace("__", "/", 1)
        if repo not in selected:
            continue
        layout = json.loads(layout_file.read_text(encoding="utf-8"))
        rules = LayerRules(layout["files"], layout["import_linter"])
        counts = {}
        for f in layout["files"]:
            lay = rules.layer(f)
            key = "unassigned" if lay is None else str(lay[0])
            counts[key] = counts.get(key, 0) + 1
        repo_rows.append({"repo": repo, "layer_source": rules.source, "rq2_eligible": rules.eligible,
                          "contracts": len(rules.contracts), **{f"files_{k}": v for k, v in counts.items()}})
        (rules_dir / f"{slug}.json").write_text(json.dumps({
            "repo": repo, "source": rules.source, "eligible": rules.eligible,
            "contracts": rules.contracts, "files_per_layer": counts,
        }, indent=1), encoding="utf-8")

        edges_file = mined_dir / f"{slug}.edges.jsonl"
        if rules.eligible and edges_file.exists():
            for line in edges_file.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                bad = [(s, d) for s, d in e["added_edges"] if rules.is_violation(s, d)]
                viol[(repo, e["number"], e["group"])] = bad

    repos = pd.DataFrame(repo_rows)
    repos.to_csv(path("tables") / "layer_rules_summary.csv", index=False)
    df = df.merge(repos[["repo", "layer_source", "rq2_eligible"]], on="repo", how="left")
    keys = list(zip(df["repo"], df["number"], df["group"]))
    df["new_violations"] = [len(viol[k]) if k in viol else None for k in keys]
    df["has_violation"] = [int(bool(viol[k])) if k in viol else None for k in keys]
    # Commit author names are personal data: used only to exclude bots in step 03, not published.
    df.drop(columns=["author"], errors="ignore").to_csv(path("processed") / "pr_metrics.csv", index=False)

    with open(path("processed") / "violations.jsonl", "w", encoding="utf-8") as f:
        for (repo, number, group), bad in viol.items():
            if bad:
                f.write(json.dumps({"repo": repo, "number": number, "group": group, "edges": bad}) + "\n")

    eligible = repos[repos["rq2_eligible"]]
    summary = {
        "repos": len(repos),
        "rq2_eligible": len(eligible),
        "by_source": eligible["layer_source"].value_counts().to_dict(),
        "import_linter_repos": int((repos["layer_source"] == "import-linter").sum()),
        "prs_with_violation": int(df["has_violation"].fillna(0).sum()),
    }
    print(summary)
    log_run("04_layer_violations.py", {}, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
