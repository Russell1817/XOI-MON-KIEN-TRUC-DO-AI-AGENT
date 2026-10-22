"""Step 05b: list every flagged layering violation for manual review (precision of the heuristic).

The reviewer fills `verdict` (true / false) and `reason` in results/manual_check/violations_review.csv.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from aierosion.config import ROOT, path
from aierosion.layers.rules import LayerRules
from aierosion.runlog import log_run


def main() -> int:
    metrics = pd.read_csv(path("processed") / "pr_metrics.csv", low_memory=False)
    sha = {(r.repo, int(r.number), r.group): r.sha_after for r in metrics.itertuples() if isinstance(r.sha_after, str)}
    rows = []
    for line in (path("processed") / "violations.jsonl").read_text(encoding="utf-8").splitlines():
        v = json.loads(line)
        slug = v["repo"].replace("/", "__")
        layout = json.loads((path("interim") / "mined" / f"{slug}.layout.json").read_text(encoding="utf-8"))
        rules = LayerRules(layout["files"], layout["import_linter"])
        for src, dst in v["edges"]:
            rows.append({
                "repo": v["repo"], "number": v["number"], "group": v["group"],
                "commit_url": f"https://github.com/{v['repo']}/commit/{sha.get((v['repo'], v['number'], v['group']), '')}",
                "src": src, "src_layer": rules.layer(src)[0], "dst": dst, "dst_layer": rules.layer(dst)[0],
                "verdict": "", "reason": "",
            })
    out = path("manual_check") / "violations_review.csv"
    new = pd.DataFrame(rows)
    if out.exists() and out.stat().st_size > 2 and len(new):
        # Keep verdicts already given for the same edge.
        old = pd.read_csv(out)
        key = ["repo", "number", "group", "src", "dst"]
        old = old[key + ["verdict", "reason"]].dropna(subset=["verdict"])
        new = new.drop(columns=["verdict", "reason"]).merge(old, on=key, how="left")
    new.to_csv(out, index=False)
    print(f"{len(rows)} flagged edges in {len({(r['repo'], r['number']) for r in rows})} PRs -> {out.relative_to(ROOT)}")
    log_run("05b_review_violations.py", {}, {"edges": len(rows)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
