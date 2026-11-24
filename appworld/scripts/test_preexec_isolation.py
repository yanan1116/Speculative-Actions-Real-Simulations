#!/usr/bin/env python3
"""Per-API test of isolated pre-execution on AppWorld, driven by ground-truth API calls.

For every task given (default: train + dev, the splits that ship ground_truth/api_calls.json) the script replays the
recorded API calls through the agent's own tool path (agent.execute_tool with execute_api). Before each real call it
runs the same call through env_isolation.isolated_execute and checks:

1. clean: the environment fingerprint (env_isolation.environment_fingerprint) is identical before and after the
   isolated call;
2. same_output: the isolated call's output equals the real call's output (what a hit's replay compares).

It then replays the same calls again in a fresh environment without any isolated calls and checks:

3. end state: both replays end with the same database contents, change logs and model counters, and the same
   evaluation result (passed and failed tests).

Results: <out>/calls.jsonl (one line per call), <out>/tasks.jsonl (one line per task), <out>/apis.json (per API).
Run several shards in parallel with --shard i --num-shards n, then --merge.
"""

import argparse
import collections
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Production (sa_breadth_agent) runs every AppWorld operation in the main thread with AppWorld's timeouts on; so does
# every check here.


def load_routes(root):
    """(method, compiled path regex, app, api, path parameter names), literal paths first."""
    routes = []
    for path in glob.glob(os.path.join(root, "data", "api_docs", "standard", "*.json")):
        for api_name, doc in json.load(open(path)).items():
            names = re.findall(r"{(\w+)}", doc["path"])
            pattern = re.sub(r"\\{(\w+)\\}", r"(?P<\1>[^/]+)", re.escape(doc["path"]))
            routes.append((doc["method"].lower(), re.compile(f"^{pattern}$"), doc["app_name"], api_name, names))
    routes.sort(key=lambda route: len(route[4]))
    return routes


def resolve(routes, method, url):
    for route_method, regex, app, api, names in routes:
        if route_method != method:
            continue
        match = regex.match(url)
        if match:
            params = {}
            for name in names:
                value = match.group(name)
                params[name] = int(value) if value.isdigit() else value
            return app, api, params
    return None


def end_state_of(env):
    """The state a run ends with, keyed independently of the experiment name: memory and disk."""
    from sa_appworld.runtime.env_isolation import environment_fingerprint

    return {f"{key.split(':', 1)[0]}:{os.path.basename(key.split(':', 1)[1])}" if ":" in key else key: value
            for key, value in environment_fingerprint(env).items()
            if key.startswith(("db:", "changes:", "disk_")) or key == "model_hashes"}


def isolate_candidates(isolation, env, candidates, task_id, index):
    """Isolate the candidates one after another, each checked against the fingerprint taken just before it."""
    from sa_appworld.runtime.agent import perf_counter
    from sa_appworld.runtime.env_isolation import environment_fingerprint, fingerprint_differences

    records, outputs = [], []
    for role, (tool, tool_args, api) in candidates:
        before = environment_fingerprint(env)
        start = perf_counter()
        output = isolation.execute(tool, tool_args)  # IsolationError propagates
        seconds = perf_counter() - start
        dirty = fingerprint_differences(before, environment_fingerprint(env))
        records.append({"task": task_id, "index": index, "api": api, "role": role, "dirty": dirty,
                        "isolated_seconds": round(seconds, 4)})
        outputs.append(output)
    return records, outputs


def replay(task_id, calls, routes, experiment_name, isolate):
    """Each step: isolate the current call and the next two calls (wrong candidates for this step), then execute
    the current call for real."""
    from appworld import AppWorld
    from sa_appworld.runtime.agent import execute_tool
    from sa_appworld.runtime.env_isolation import EnvIsolation, environment_fingerprint, fingerprint_differences

    env = AppWorld(task_id=task_id, experiment_name=experiment_name, raise_on_failure=False,
                   max_interactions=100000)
    isolation = EnvIsolation(env, execute_tool) if isolate else None
    tokens = {}
    records = []

    def args_of(index):
        resolved = resolve(routes, calls[index]["method"], calls[index]["url"])
        if resolved is None:
            return None
        app, api, path_params = resolved
        params = dict(calls[index].get("data") or {})
        params.update(path_params)
        if "access_token" in params and app in tokens:
            params["access_token"] = tokens[app]
        return "execute_api", {"app_name": app, "api_name": api, "parameters": params}, f"{app}.{api}"

    try:
        for index, call in enumerate(calls):
            current = args_of(index)
            if current is None:
                records.append({"task": task_id, "index": index, "url": call["url"], "unresolved": True})
                continue
            tool, tool_args, api = current
            record = {"task": task_id, "index": index, "api": api, "method": call["method"], "role": "real"}
            if isolate:
                candidates = [("current", current)] + [
                    ("wrong", c) for c in (args_of(j) for j in range(index + 1, min(index + 3, len(calls)))) if c]
                candidate_records, candidate_outputs = isolate_candidates(isolation, env, candidates,
                                                                          task_id, index)
                records.extend(candidate_records)
                after = environment_fingerprint(env)
            output = execute_tool(env, tool, tool_args)
            if isolate:
                written = isolation.commit_point()
                record["same_output"] = candidate_outputs[0] == output
                record["real_writes"] = [p.rsplit("/", 1)[-1] for p in written]
                # The change logs must name exactly the databases whose bytes changed.
                bytes_changed = [key.rsplit("/", 1)[-1] for key in fingerprint_differences(
                    after, environment_fingerprint(env)) if key.startswith("db:")]
                record["write_detection_ok"] = sorted(bytes_changed) == sorted(record["real_writes"])
            record["error"] = output.startswith("Execution failed") or "Traceback" in output
            if api.endswith(".login") and not record["error"]:
                try:
                    tokens[api.split(".")[0]] = json.loads(output)["access_token"]
                except (ValueError, KeyError, TypeError):
                    pass
            records.append(record)
        end_state = end_state_of(env)
        evaluation = env.evaluate(suppress_errors=True)
        result = {"success": evaluation.success, "pass_count": evaluation.pass_count,
                  "total_count": evaluation.total_count,
                  "failures": sorted(str(f.get("requirement", f)) for f in evaluation.to_dict().get("failures", []))}
    finally:
        env.close()
    return records, end_state, result


def replay_steps(task_id, steps, experiment_name, isolate, evaluate=True):
    """Replay tool calls (tool, args, api) through agent.execute_tool. With isolate, each step first isolates the
    current call and the next two calls (wrong candidates for this step), as replay() does."""
    from appworld import AppWorld
    from sa_appworld.runtime.agent import execute_tool
    from sa_appworld.runtime.env_isolation import EnvIsolation, environment_fingerprint, fingerprint_differences

    env = AppWorld(task_id=task_id, experiment_name=experiment_name, raise_on_failure=False, max_interactions=100000)
    isolation = EnvIsolation(env, execute_tool) if isolate else None
    records = []
    result = None
    try:
        for index, (tool, tool_args, api) in enumerate(steps):
            record = {"task": task_id, "index": index, "api": api, "tool": tool, "role": "real"}
            if isolate:
                candidates = [("current", steps[index])] + [("wrong", steps[j])
                                                            for j in range(index + 1, min(index + 3, len(steps)))]
                candidate_records, candidate_outputs = isolate_candidates(isolation, env, candidates,
                                                                          task_id, index)
                records.extend(candidate_records)
                after = environment_fingerprint(env)
            output = execute_tool(env, tool, tool_args)
            if isolate:
                written = isolation.commit_point()
                record["same_output"] = candidate_outputs[0] == output
                record["real_writes"] = [p.rsplit("/", 1)[-1] for p in written]
                bytes_changed = [key.rsplit("/", 1)[-1] for key in fingerprint_differences(
                    after, environment_fingerprint(env)) if key.startswith("db:")]
                record["write_detection_ok"] = sorted(bytes_changed) == sorted(record["real_writes"])
            record["error"] = output.startswith(("Execution failed", "Error")) or "Traceback" in output
            records.append(record)
        end_state = end_state_of(env)
        if evaluate:
            evaluation = env.evaluate(suppress_errors=True)
            result = {"success": evaluation.success, "pass_count": evaluation.pass_count,
                      "total_count": evaluation.total_count}
    finally:
        env.close()
    return records, end_state, result


def replay_tools(task_id, root, experiment_name):
    """The agent's dedicated tools (api_docs lookups and supervisor shortcuts), which ground-truth calls never use:
    every app's API list, every API's doc, a few doc searches and each supervisor shortcut, called by tool name."""
    from sa_appworld.runtime.sa_breadth_agent import DIRECT_TOOL_APIS

    docs = {}
    for path in glob.glob(os.path.join(root, "data", "api_docs", "standard", "*.json")):
        docs[os.path.basename(path)[:-5]] = sorted(json.load(open(path)))
    calls = [("show_app_descriptions", {})]
    for app, apis in sorted(docs.items()):
        calls.append(("show_api_descriptions", {"app_name": app}))
        calls += [("show_api_doc", {"app_name": app, "api_name": api}) for api in apis]
    calls += [("search_api_docs", {"query": query}) for query in ("send money", "playlist songs", "delete a note")]
    calls += [(tool, {}) for tool in ("show_account_passwords", "show_active_task", "show_profile", "show_addresses",
                                      "show_payment_cards")]
    steps = [(tool, tool_args, DIRECT_TOOL_APIS[tool]) for tool, tool_args in calls]
    records, _, _ = replay_steps(task_id, steps, experiment_name, True, evaluate=False)
    return records


def run_trajectories(args):
    """--trajectories DIR: replay every <task>.json raw trajectory in DIR (TrajectoryLogger format)."""
    from appworld import update_root
    from sa_appworld.runtime.agent import perf_counter
    from sa_appworld.runtime.sa_breadth_agent import api_of

    update_root(args.appworld_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"\W+", "_", args.trajectories.rstrip("/"))[-60:]
    files = sorted(glob.glob(os.path.join(args.trajectories, "*.json")))[args.shard::args.num_shards]
    with open(out / f"calls.traj{args.shard}_{tag}.jsonl", "w") as calls_file, \
            open(out / f"tasks.traj{args.shard}_{tag}.jsonl", "w") as tasks_file:
        for path in files:
            start = perf_counter()
            trajectory = json.load(open(path))
            task_id = trajectory["task_id"]
            steps = []
            for step in trajectory.get("steps", []):
                try:
                    call = json.loads(step.get("model_output") or "")
                except ValueError:
                    continue  # a text reply, not a tool call
                if isinstance(call, dict) and call.get("tool") and not call["tool"].startswith("_"):
                    steps.append((call["tool"], call.get("args") or {}, api_of(call["tool"], call.get("args") or {})))
            records, end_isolated, result_isolated = replay_steps(
                task_id, steps, f"isotest_traj_iso_{args.shard}", True)
            _, end_plain, result_plain = replay_steps(task_id, steps, f"isotest_traj_plain_{args.shard}", False)
            for record in records:
                calls_file.write(json.dumps(record) + "\n")
            task = {
                "task": f"{task_id}@{tag}", "calls": len(steps),
                "dirty_calls": sum(bool(r.get("dirty")) for r in records),
                "output_mismatch": sum(r.get("same_output") is False for r in records),
                "unresolved": 0, "errors": sum(bool(r.get("error")) for r in records),
                "end_state_equal": end_isolated == end_plain,
                "end_state_diff": sorted(k for k in set(end_isolated) | set(end_plain)
                                         if end_isolated.get(k) != end_plain.get(k)),
                "evaluation_equal": result_isolated == result_plain,
                "write_detection_failures": sum(r.get("write_detection_ok") is False for r in records),
                "isolated": result_isolated, "plain": result_plain, "seconds": round(perf_counter() - start, 1),
            }
            tasks_file.write(json.dumps(task) + "\n")
            tasks_file.flush()
            calls_file.flush()
            print(f"[traj {args.shard}] {task_id} calls={len(steps)} dirty={task['dirty_calls']} "
                  f"mismatch={task['output_mismatch']} end_equal={task['end_state_equal']} "
                  f"eval_equal={task['evaluation_equal']} success={result_plain['success']}", flush=True)


def run_faults(args):
    """Failure paths, in the main thread as in production, on one dev task, with complete_task as the writing call
    (it writes supervisor.db and needs no login):
    1. the call writes, then raises: the original exception propagates and the fingerprint is unchanged;
    2. restoring a database fails: IsolationError, and the other state (interaction counter, random state) is still
       put back;
    3. after (2) the environment is dirty: the next isolated call is refused with IsolationError."""
    import random as random_module

    from appworld import AppWorld, update_root
    from sa_appworld.runtime.agent import execute_tool
    from sa_appworld.runtime.env_isolation import (EnvIsolation, IsolationError, environment_fingerprint,
                                                   fingerprint_differences)

    update_root(args.appworld_root)
    task_id = open(os.path.join(args.appworld_root, "data", "datasets", "dev.txt")).read().split()[0]
    results = {}
    env = AppWorld(task_id=task_id, experiment_name="isotest_faults", raise_on_failure=False, max_interactions=100000)
    try:
        def write_then_raise(environment, name, tool_args):
            execute_tool(environment, name, tool_args)
            raise RuntimeError("injected failure after a write")

        isolation = EnvIsolation(env, write_then_raise)
        before = environment_fingerprint(env)
        try:
            isolation.execute("complete_task", {"output": "x"})
            results["1_exception_propagates"] = False
        except RuntimeError as error:
            results["1_exception_propagates"] = "injected" in str(error) and not isinstance(error, IsolationError)
        results["1_fingerprint_unchanged"] = fingerprint_differences(before, environment_fingerprint(env)) == []

        isolation = EnvIsolation(env, execute_tool)
        supervisor_db = next(p for p in isolation.clean if p.endswith("supervisor.db"))
        isolation.clean[supervisor_db].close()  # restoring it from the closed copy must fail
        interactions_before = env.num_interactions
        random_before = random_module.getstate()
        try:
            isolation.execute("complete_task", {"output": "x"})
            results["2_isolation_error"] = False
        except IsolationError as error:
            results["2_isolation_error"] = "supervisor.db" in str(error)
        results["2_other_state_restored"] = (env.num_interactions == interactions_before
                                            and random_module.getstate() == random_before)
        try:
            isolation.execute("show_profile", {})
            results["3_dirty_env_refused"] = False
        except IsolationError:
            results["3_dirty_env_refused"] = True
    finally:
        env.close()
    print(json.dumps(results, indent=1))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "faults.json").write_text(json.dumps(results, indent=1) + "\n")
    if not all(results.values()):
        raise SystemExit("fault-path test failed")


def run_tools(args):
    from appworld import update_root

    update_root(args.appworld_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task_ids = open(os.path.join(args.appworld_root, "data", "datasets", "dev.txt")).read().split()[:args.tool_tasks]
    with open(out / "calls.tools.jsonl", "w") as calls_file:
        for task_id in task_ids:
            records = replay_tools(task_id, args.appworld_root, "isotest_tools")
            for record in records:
                calls_file.write(json.dumps(record) + "\n")
            print(f"[tools] {task_id} records={len(records)} dirty={sum(bool(r.get('dirty')) for r in records)} "
                  f"mismatch={sum(r.get('same_output') is False for r in records)} "
                  f"errors={sum(bool(r.get('error')) for r in records)}", flush=True)


def run_shard(args):
    from appworld import update_root
    from sa_appworld.runtime.agent import perf_counter  # time.time() is frozen inside AppWorld

    update_root(args.appworld_root)
    routes = load_routes(args.appworld_root)
    task_ids = []
    for split in args.splits:
        task_ids += open(os.path.join(args.appworld_root, "data", "datasets", f"{split}.txt")).read().split()
    task_ids = task_ids[args.shard::args.num_shards]
    if args.limit:
        task_ids = task_ids[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"calls.{args.shard}.jsonl", "w") as calls_file, open(out / f"tasks.{args.shard}.jsonl", "w") as tasks_file:
        for number, task_id in enumerate(task_ids, 1):
            start = perf_counter()
            calls_path = os.path.join(args.appworld_root, "data", "tasks", task_id, "ground_truth", "api_calls.json")
            calls = json.load(open(calls_path))
            records, end_isolated, result_isolated = replay(task_id, calls, routes, f"isotest_iso_{args.shard}", True)
            _, end_plain, result_plain = replay(task_id, calls, routes, f"isotest_plain_{args.shard}", False)
            for record in records:
                calls_file.write(json.dumps(record) + "\n")
            task = {
                "task": task_id, "calls": len(calls),
                "dirty_calls": sum(bool(r.get("dirty")) for r in records),
                "output_mismatch": sum(r.get("same_output") is False for r in records),
                "unresolved": sum(bool(r.get("unresolved")) for r in records),
                "errors": sum(bool(r.get("error")) for r in records),
                "end_state_equal": end_isolated == end_plain,
                "end_state_diff": sorted(k for k in set(end_isolated) | set(end_plain)
                                         if end_isolated.get(k) != end_plain.get(k)),
                "evaluation_equal": result_isolated == result_plain,
                "write_detection_failures": sum(r.get("write_detection_ok") is False for r in records),
                "isolated": result_isolated, "plain": result_plain, "seconds": round(perf_counter() - start, 1),
            }
            tasks_file.write(json.dumps(task) + "\n")
            tasks_file.flush()
            calls_file.flush()
            print(f"[{args.shard}] {number}/{len(task_ids)} {task_id} calls={len(calls)} dirty={task['dirty_calls']} "
                  f"mismatch={task['output_mismatch']} end_equal={task['end_state_equal']} "
                  f"eval_equal={task['evaluation_equal']} success={result_plain['success']} "
                  f"{task['seconds']}s", flush=True)


def merge(args):
    out = Path(args.out)
    per_api = collections.defaultdict(lambda: collections.Counter())
    for path in sorted(out.glob("calls.*.jsonl")):
        for line in open(path):
            record = json.loads(line)
            if record.get("unresolved"):
                continue
            counter = per_api[record["api"]]
            if record.get("role") in ("current", "wrong"):  # an isolated candidate
                counter["isolated_calls"] += 1
                counter["isolated_clean"] += not record["dirty"]
                counter["isolated_as_wrong"] += record["role"] == "wrong"
                continue
            counter["calls"] += 1  # a real call
            counter["same_output"] += record["same_output"]
            counter["real_writes"] += bool(record["real_writes"])
            counter["write_detection_ok"] += record["write_detection_ok"]
            counter["errors"] += record["error"]
            counter["successful_calls"] += not record["error"]
    tasks = [json.loads(line) for path in sorted(out.glob("tasks.*.jsonl")) for line in open(path)]
    apis = {api: dict(counts) for api, counts in sorted(per_api.items())}
    json.dump({"tasks": len(tasks), "apis": apis}, open(out / "apis.json", "w"), indent=1)
    print(f"tasks {len(tasks)}: end_state_equal {sum(t['end_state_equal'] for t in tasks)}, "
          f"evaluation_equal {sum(t['evaluation_equal'] for t in tasks)}, "
          f"plain success {sum(t['plain']['success'] for t in tasks)}, "
          f"calls {sum(t['calls'] for t in tasks)}, dirty {sum(t['dirty_calls'] for t in tasks)}, "
          f"output mismatch {sum(t['output_mismatch'] for t in tasks)}, unresolved {sum(t['unresolved'] for t in tasks)}")
    print(f"APIs exercised {len(apis)}; with a successful call {sum(a.get('successful_calls', 0) > 0 for a in apis.values())}; "
          f"all isolated calls clean {sum(a.get('isolated_clean', 0) == a.get('isolated_calls', 0) for a in apis.values())}; "
          f"all same output {sum(a.get('same_output', 0) == a.get('calls', 0) for a in apis.values())}; "
          f"writes observed {sum(a.get('real_writes', 0) > 0 for a in apis.values())}")
    # Verified: every call left the environment clean, gave the real call's output, and the change logs named exactly
    # the databases a real call changed; and at least one call succeeded, so the API's own code path ran.
    verified = sorted(api for api, a in apis.items()
                      if a.get("successful_calls", 0) > 0 and a.get("isolated_calls", 0) > 0
                      and a.get("isolated_clean", 0) == a["isolated_calls"]
                      and a.get("same_output", 0) == a["calls"] and a.get("write_detection_ok", 0) == a["calls"])
    failed_tasks = [t["task"] for t in tasks if not (t["end_state_equal"] and t["evaluation_equal"])]
    print(f"verified APIs {len(verified)}; tasks failing the end-state or evaluation check: {failed_tasks}")
    if args.write_verified:
        if failed_tasks:
            raise SystemExit("not writing the verified list: some tasks failed the end-state or evaluation check")
        payload = {"source": str(out), "tasks": len(tasks), "rule": "all calls clean, same output, change logs exact; "
                   ">= 1 successful call", "apis": verified}
        Path(args.write_verified).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.write_verified}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appworld-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--write-verified", default="", help="with --merge: write the verified API list here")
    parser.add_argument("--tools", action="store_true", help="test the agent's dedicated tools instead of GT calls")
    parser.add_argument("--tool-tasks", type=int, default=3, help="dev tasks to run the dedicated-tool test on")
    parser.add_argument("--faults", action="store_true", help="test the failure paths (raise, restore failure)")
    parser.add_argument("--trajectories", default="", help="replay the agent trajectories (*.json) in this directory")
    args = parser.parse_args()
    if args.merge:
        merge(args)
    elif args.trajectories:
        run_trajectories(args)
    elif args.faults:
        run_faults(args)
    elif args.tools:
        run_tools(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
