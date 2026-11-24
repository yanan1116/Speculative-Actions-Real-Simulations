#!/usr/bin/env python3
"""Offline isolation audit of a finished SA run (run.py --mode sa).

For each task of the run, the actions the agent executed for real (its raw trajectory) are replayed, in order, in a
fresh AppWorld environment with no pre-execution at all. If no isolated pre-execution of the run left a trace, the
replay ends in exactly the state the run ended in. Checks per task:

1. outputs_equal: every replayed call returns what the run's real call returned (whole, or the 2000-character
   prefix older trajectory logs kept);
2. dbs_equal: the saved database changes and model counters (<outputs>/tasks/<id>/dbs/*) are byte-identical;
3. logs_equal: the saved API-call log and environment I/O log (<outputs>/tasks/<id>/logs/*) are byte-identical;
4. evaluation_equal: the replay's evaluation (success, passed / total tests) equals the run's result.

The replay writes only to its own experiment directory (<appworld_root>/experiments/outputs/<--experiment>).
Usage: audit_sa_isolation.py --run-dir RUNS/<run_id>_sa [--appworld-root ROOT] [--out FILE] [--tasks ...]
"""

import argparse
import filecmp
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def real_steps(trajectory):
    """(tool, args, recorded output) of each executed step.

    The executed call is the step's api_name and api_params (kept whole); model_output keeps the same call cut to
    2000 characters, and must equal it, or the step raises ValueError."""
    steps = []
    for step in trajectory["steps"]:
        if step["api_name"] == "_error":
            continue  # the Actor call failed: nothing was executed, as in the baseline
        tool, args = step["api_name"], step["api_params"]
        logged = json.dumps({"tool": tool, "args": args})
        # --repair-access-token: a step whose token was replaced logs the original under "token_repaired_from"; the
        # executed call is still tool + args.
        if '"token_repaired_from"' in step["model_output"]:
            repaired = json.loads(step["model_output"])
            logged = json.dumps({"tool": tool, "args": args, "token_repaired_from": repaired.get("token_repaired_from")})
        # Older logs keep 2000 characters of model_output / api_result, newer logs keep them whole.
        if not isinstance(args, dict) or step["model_output"] not in (logged, logged[:2000]):
            raise ValueError(f"step {step['step_id']}: api_name/api_params disagree with model_output")
        steps.append((tool, args, step["api_result"]))
    return steps


def files_equal(left, right):
    """Names of the files that differ or exist on one side only (empty: equal)."""
    names = sorted(set(os.listdir(left) if os.path.isdir(left) else []) |
                   set(os.listdir(right) if os.path.isdir(right) else []))
    return [name for name in names
            if not (os.path.isfile(os.path.join(left, name)) and os.path.isfile(os.path.join(right, name))
                    and filecmp.cmp(os.path.join(left, name), os.path.join(right, name), shallow=False))]


def audit_task(task_id, run_dir, run_outputs, experiment, max_interactions):
    from appworld import AppWorld
    from sa_appworld.runtime.agent import execute_tool

    trajectory = json.load(open(run_dir / "raw_trajectories" / f"{task_id}.json"))
    result = json.load(open(run_dir / "results" / f"{task_id}.json"))
    record = {"task": task_id, "run_error_type": result.get("error_type")}
    run_task_dir = run_outputs / "tasks" / task_id
    try:
        steps = real_steps(trajectory)
    except (ValueError, KeyError) as error:
        record["unreplayable"] = repr(error)
        return record
    env = AppWorld(task_id=task_id, experiment_name=experiment, raise_on_failure=False,
                   max_interactions=max_interactions)
    mismatched = []
    try:
        for index, (tool, args, recorded) in enumerate(steps):
            output = execute_tool(env, tool, args)
            if output != recorded and output[:2000] != recorded:
                mismatched.append(index + 1)
        evaluation = env.evaluate(suppress_errors=True)
    finally:
        env.close()
    replay_dir = Path(env.output_db_home_path_on_disk).parent  # <outputs>/<experiment>/tasks/<id>
    record.update({
        "steps": len(steps),
        "output_mismatch_steps": mismatched,
        "dbs_diff": files_equal(run_task_dir / "dbs", replay_dir / "dbs"),
        "logs_diff": files_equal(run_task_dir / "logs", replay_dir / "logs"),
        "replay_eval": [evaluation.success, evaluation.pass_count, evaluation.total_count],
        "run_reward": result.get("reward"),
    })
    replay_reward = evaluation.pass_count / evaluation.total_count if evaluation.total_count else 0.0
    # The run's result keeps the reward rounded to 3 decimals (5/6 -> 0.833).
    record["evaluation_equal"] = round(replay_reward, 3) == round(result.get("reward") or 0.0, 3)
    record["isolation_ok"] = (not mismatched and not record["dbs_diff"] and not record["logs_diff"]
                              and record["evaluation_equal"])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--appworld-root", default="/mnt/disk1t/smc-appworld/root")
    parser.add_argument("--out", type=Path, help="JSONL, one line per task (default: <run-dir>/isolation_audit.jsonl)")
    parser.add_argument("--tasks", nargs="*", help="default: every task with a result; skips tasks already in --out")
    args = parser.parse_args()

    from appworld import update_root

    update_root(args.appworld_root)
    config = json.load(open(args.run_dir / "config.json"))
    assert config.get("mode") == "sa", "not an SA run (--mode sa)"
    run_outputs = Path(args.appworld_root) / "experiments" / "outputs" / f"smc_{args.run_dir.name}_{config['mode']}"
    assert run_outputs.is_dir(), run_outputs
    experiment = f"audit_{args.run_dir.name}_plain"
    out = args.out or args.run_dir / "isolation_audit.jsonl"
    done = {json.loads(line)["task"] for line in open(out)} if out.exists() else set()
    tasks = args.tasks or sorted(p.stem for p in (args.run_dir / "results").glob("*.json"))
    with open(out, "a") as handle:
        for task_id in tasks:
            if task_id in done:
                continue
            record = audit_task(task_id, args.run_dir, run_outputs, experiment, config["max_steps"] * 3)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
