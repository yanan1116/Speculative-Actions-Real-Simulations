#!/usr/bin/env python3
"""Write the AppWorld test task list: the test_scen1 subset (default, 194 tasks) or the full test set (--full, 585 tasks).

Default (test_scen1): one task per scenario of test_normal and test_challenge. Within each split the scenarios (task id
without the trailing _<n>) are taken in sorted order, and one task per scenario is drawn with
random.Random(f"20261010-{split}"). This gives 56 + 139 = 195 tasks; 0de03ad_2 is excluded by default, because its
context exceeds 128K tokens in every arm, which leaves 194 tasks (the list behind the README results).

--full: every task of test_normal (168) and test_challenge (417), in file order, 585 tasks; nothing is excluded by
default.

--exclude ID ... replaces the default exclusion in either mode (--exclude with no ID excludes nothing).

Usage: make_test_scen1_tasks.py --appworld-root ROOT [--full] [--exclude ID ...] > tasks.txt
"""

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--appworld-root", required=True, type=Path)
    parser.add_argument("--full", action="store_true",
                        help="all 585 tasks of test_normal + test_challenge instead of the 194-task test_scen1 subset")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="task ids to leave out (default: 0de03ad_2 for test_scen1, none for --full)")
    args = parser.parse_args()
    exclude = set(args.exclude if args.exclude is not None else ([] if args.full else ["0de03ad_2"]))
    for split in ("test_normal", "test_challenge"):
        tasks = (args.appworld_root / "data" / "datasets" / f"{split}.txt").read_text().split()
        if args.full:
            for task in tasks:
                if task not in exclude:
                    print(task)
            continue
        scenarios: dict[str, list[str]] = {}
        for task in tasks:
            scenarios.setdefault(task.rsplit("_", 1)[0], []).append(task)
        rng = random.Random(f"20261010-{split}")
        for scenario in sorted(scenarios):
            task = rng.choice(sorted(scenarios[scenario]))
            if task not in exclude:
                print(task)


if __name__ == "__main__":
    main()
