#!/usr/bin/env python3
"""Baseline vs. SA table of one run (all / normal / challenge), from the per-task result files.

Columns:
- SR: reward 1.0 / tasks.
- avg: mean task score (fraction of unit tests passed).
- s: agent wall time per task (baseline total_wall_time, SA wall_time).
- ratio: SA time / baseline time per task.
  - Median and Geomean are taken over tasks.
  - Total = sum of SA time / sum of baseline time.
- SA faster: tasks with ratio < 1.
- 4B pred acc: speculated Actor steps where a Speculator guess equals the Actor's action exactly (guess_match / guess_steps).

Usage, with the run tag of run_serial.sh / run_pairs.sh:
    summarize_sa_runs.py --out-dir runs/appworld_sa --tag test_scen1_serial1doff --mode off \
        --appworld-root ROOT --tasks test_scen1_tasks_le128k.txt
or with explicit run directories: --baseline-dir DIR --sa-dir DIR (instead of --out-dir / --tag / --mode).
Fails if a task has no result in either arm.
"""

import argparse
import json
import math
import statistics as st
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--mode", choices=["on", "off"])
    parser.add_argument("--spec-breadth", type=int, default=5)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--sa-dir", type=Path)
    parser.add_argument("--appworld-root", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    args = parser.parse_args()
    if args.baseline_dir and args.sa_dir:
        base_dir, sa_dir = args.baseline_dir, args.sa_dir
    elif args.out_dir and args.tag and args.mode:
        base_dir = args.out_dir / f"{args.tag}_{args.mode}_baseline"
        sa_dir = args.out_dir / f"{args.tag}_sa_k{args.spec_breadth}_spec4b_sa"
    else:
        parser.error("give --baseline-dir and --sa-dir, or --out-dir, --tag and --mode")

    tasks = args.tasks.read_text().split()
    normal = set((args.appworld_root / "data" / "datasets" / "test_normal.txt").read_text().split())
    rows = []
    for task in tasks:
        b = json.loads((base_dir / "results" / f"{task}.json").read_text())
        s = json.loads((sa_dir / "results" / f"{task}.json").read_text())
        rows.append({"split": "normal" if task in normal else "challenge",
                     "b_ok": b["reward"] == 1.0, "s_ok": s["reward"] == 1.0, "b_r": b["reward"], "s_r": s["reward"],
                     "b_s": b["total_wall_time"], "s_s": s["wall_time"],
                     "hit": s.get("guess_match") or 0, "steps": s.get("guess_steps") or 0})

    pct = lambda n, d: f"{100 * n / d:.1f}% ({n}/{d})" if d else "-"
    head = ["subset", "Baseline SR", "SA SR", "Base avg", "SA avg", "Base median s", "SA median s", "Median ratio",
            "Geomean ratio", "Total ratio", "SA faster", "4B pred acc"]
    table = []
    for group in ("all", "normal", "challenge"):
        rs = [r for r in rows if group == "all" or r["split"] == group]
        if not rs:
            continue
        n = len(rs)
        q = [r["s_s"] / r["b_s"] for r in rs]
        table.append([group, pct(sum(r["b_ok"] for r in rs), n), pct(sum(r["s_ok"] for r in rs), n),
                      f"{st.mean(r['b_r'] for r in rs):.3f}", f"{st.mean(r['s_r'] for r in rs):.3f}",
                      f"{st.median(r['b_s'] for r in rs):.0f}", f"{st.median(r['s_s'] for r in rs):.0f}",
                      f"{st.median(q):.3f}", f"{math.exp(st.mean(map(math.log, q))):.3f}",
                      f"{sum(r['s_s'] for r in rs) / sum(r['b_s'] for r in rs):.3f}",
                      pct(sum(x < 1 for x in q), n), pct(sum(r["hit"] for r in rs), sum(r["steps"] for r in rs))])
    widths = [max(len(row[i]) for row in table + [head]) for i in range(len(head))]
    line = lambda row: "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |"
    print(line(head))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in table:
        print(line(row))


if __name__ == "__main__":
    main()
