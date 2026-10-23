"""Step 05c: estimate how many "human" PRs were in fact made with an AI agent.

AIDev labels miss some agent PRs (all candidate human PRs of MontrealAI/AGI-Alpha-Agent-v0 came
from `codex/...` branches). This step draws a fixed random sample of human PRs from the analysis
sample, downloads each PR page from github.com (no API token needed) and searches the page text
for agent traces: agent names in the source branch, "Generated with ..." footers, agent bot
accounts and co-author trailers. It is an automated screen; flagged PRs are listed with the
matching text so the authors can confirm them by hand.

Output: results/manual_check/human_screen.csv
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
import time

import pandas as pd

from aierosion.config import path, settings
from aierosion.runlog import log_run

N = 50
AGENTS = r"(codex|copilot|cursor|devin|claude|jules|chatgpt|openhands|sweep|aider|gemini)"
PATTERNS = {
    "branch": re.compile(r"\bfrom\s+[\w.-]+:(?:[\w./-]*/)?[\w./-]*" + AGENTS + r"[\w./-]*", re.I),
    "generated": re.compile(r"generated (?:with|by) \[?" + AGENTS, re.I),
    "coauthor": re.compile(r"co-authored-by:[^\n<]*" + AGENTS, re.I),
    "agent_link": re.compile(r"(chatgpt\.com/codex|app\.devin\.ai|cursor\.com/agents|claude\.ai/code|jules\.google)", re.I),
    "bot_author": re.compile(r"\b(copilot-swe-agent|devin-ai-integration|cursor\[bot\]|claude\[bot\]|google-labs-jules)\b", re.I),
}


def page_text(repo: str, number: int) -> str | None:
    url = f"https://github.com/{repo}/pull/{number}"
    for _ in range(3):
        r = subprocess.run(["curl", "-sL", "--max-time", "60", "-A", "Mozilla/5.0", url],
                           capture_output=True, timeout=90)
        if r.returncode == 0 and len(r.stdout) > 10000:
            raw = r.stdout.decode("utf-8", "replace")
            return html.unescape(re.sub(r"<[^>]+>", " ", raw))
        time.sleep(5)
    return None


def main() -> int:
    seed = settings()["seed"]
    d = pd.read_csv(path("processed") / "pr_metrics.csv", low_memory=False)
    ok = d[d["status"] == "ok"]
    both = ok.groupby("repo")["group"].agg(set)
    sample = ok[ok["repo"].isin(both[both.apply(lambda g: {"agent", "human"} <= g)].index)]
    human = sample[sample["group"] == "human"].sort_values(["repo", "number"]).sample(N, random_state=seed)
    rows = []
    for _, r in human.iterrows():
        text = page_text(r["repo"], int(r["number"]))
        hits = {k: p.search(text).group(0)[:120] for k, p in PATTERNS.items() if text and p.search(text)}
        rows.append({"repo": r["repo"], "number": int(r["number"]),
                     "url": f"https://github.com/{r['repo']}/pull/{int(r['number'])}",
                     "fetched": text is not None, "flagged": bool(hits),
                     "evidence": "; ".join(f"{k}: {v}" for k, v in hits.items()),
                     "confirmed_by_authors": ""})
        time.sleep(2)
    out = path("manual_check") / "human_screen.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    res = pd.DataFrame(rows)
    print(f"fetched {res['fetched'].sum()}/{len(res)}; flagged {res['flagged'].sum()} -> {out}")
    log_run("05c_screen_human_prs.py", {"seed": seed, "n": N},
            {"fetched": int(res["fetched"].sum()), "flagged": int(res["flagged"].sum())})
    return 0


if __name__ == "__main__":
    sys.exit(main())
