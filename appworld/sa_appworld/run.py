#!/usr/bin/env python3
"""Run AppWorld tasks with the sequential baseline agent (--mode baseline) or the original Speculative Actions agent
(--mode sa: Actor + Speculator with --spec-breadth guesses per step and isolated pre-execution,
runtime/sa_breadth_agent.py).

Compact result files deliberately exclude prompts, messages, API arguments,
and tool responses. Raw trajectories are opt-in because AppWorld tasks can
contain simulated credentials and other benchmark-sensitive values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sa_appworld.result_summary import build_run_summary, run_exit_code  # noqa: E402


SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    payload["_config_dir"] = str(config_path.parent)
    return payload


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    known, _ = pre_parser.parse_known_args()
    config = load_config(known.config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=known.config)
    parser.add_argument(
        "--mode",
        choices=["baseline", "sa"],
        default=config.get("mode", "baseline"),
    )
    parser.add_argument("--split", default=config.get("split", "test_normal"))
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--task-ids-file", default=config.get("task_ids_file"))
    parser.add_argument("--limit", type=int, default=config.get("limit"))
    parser.add_argument("--offset", type=int, default=config.get("offset", 0))
    parser.add_argument("--run-id", default=config.get("run_id"))
    parser.add_argument("--out-dir", default=config.get("out_dir", "runs/appworld"))
    parser.add_argument("--appworld-root", default=config.get("appworld_root"))
    parser.add_argument("--actor-model", default=config.get("actor_model", "Qwen/Qwen3.5-27B"))
    parser.add_argument("--speculator-model", default=config.get("speculator_model", "Qwen/Qwen3.5-4B"))
    parser.add_argument("--actor-url", default=config.get("actor_url", "http://localhost:8004/v1"))
    parser.add_argument("--speculator-url", default=config.get("speculator_url", "http://localhost:8003/v1"))
    # --mode sa only: endpoint of the pre-launched Actor calls (H_t + guess + its observation). Unset (default)
    # sends them to --actor-url, as before. It must serve --actor-model with the same max_model_len as --actor-url.
    parser.add_argument("--prelaunch-url", default=config.get("prelaunch_url"))
    parser.add_argument("--max-steps", type=int, default=config.get("max_steps", 80))
    parser.add_argument("--thinking-budget", type=int, default=config.get("thinking_budget", 4096))
    parser.add_argument("--owner-max-tokens", type=int, default=config.get("owner_max_tokens", 8192))
    # Actor thinking, sent explicitly as chat_template_kwargs.enable_thinking on every Actor request (both modes).
    parser.add_argument("--actor-thinking", choices=["on", "off"], default=config.get("actor_thinking", "off"))
    parser.add_argument("--preexec-verified-apis", default=config.get(
        "preexec_verified_apis", str(REPO_ROOT / "sa_appworld" / "runtime" / "preexec_verified_apis_v4.json")))
    parser.add_argument("--spec-breadth", type=int, default=config.get("spec_breadth", 1))
    # Sampling as a JSON object of temperature / top_p / top_k / min_p / presence_penalty / frequency_penalty /
    # repetition_penalty (agent.SAMPLING_FIELDS). --actor-sampling applies to every Actor call (both modes,
    # pre-launched calls included); default greedy. --spec-sampling is required with --mode sa.
    parser.add_argument("--actor-sampling", type=json.loads, default=config.get("actor_sampling", {"temperature": 0.0}))
    parser.add_argument("--spec-sampling", type=json.loads, default=config.get("spec_sampling"))
    parser.add_argument("--spec-max-tokens", type=int, default=config.get("spec_max_tokens", 1024))
    # Actor request timeout in seconds (the release used 300; thinking-on requests under load can need longer), and
    # retries of server/connection failures within a step without using up a step (0 = off, the release behaviour).
    parser.add_argument("--actor-request-timeout", type=float, default=config.get("actor_request_timeout", 300.0))
    parser.add_argument("--infra-retries", type=int, default=config.get("infra_retries", 0))
    # Simulated slower Actor: every Actor call (both modes, including SA's pre-launched calls) hands
    # its response back SEC seconds after the server returned it. 0 (default) = off, behaviour unchanged.
    parser.add_argument("--actor-extra-latency", type=float, default=config.get("actor_extra_latency", 0.0),
                        metavar="SEC")
    # Replace an execute_api access_token that is not verbatim a token from an earlier tool result with the latest token
    # of the same app (runtime/agent.repair_token_args); both modes. Off (default) = unchanged.
    parser.add_argument("--repair-access-token", action="store_true", default=config.get("repair_access_token", False))
    # End the task at the first Actor request refused for exceeding the context window (the history only grows, so
    # every later request would fail the same way); both modes. Off (default) = unchanged: _error
    # steps until --max-steps.
    parser.add_argument("--stop-on-context-overflow", action="store_true",
                        default=config.get("stop_on_context_overflow", False))
    parser.add_argument("--save-raw-trajectories", action=argparse.BooleanOptionalAction, default=config.get("save_raw_trajectories", False))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=config.get("resume", True))
    return parser.parse_args()


def load_requested_task_ids(args: argparse.Namespace) -> list[str]:
    if args.task_id:
        task_ids = list(args.task_id)
    elif args.task_ids_file:
        task_ids_path = Path(args.task_ids_file).expanduser()
        if not task_ids_path.is_absolute():
            task_ids_path = REPO_ROOT / task_ids_path
        text = task_ids_path.read_text(encoding="utf-8").strip()
        if text.startswith("[") or text.startswith("{"):
            payload = json.loads(text)
            if isinstance(payload, dict):
                payload = payload.get("task_ids", [])
            task_ids = [str(item) for item in payload]
        else:
            task_ids = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        from appworld import load_task_ids

        task_ids = list(load_task_ids(args.split))
    task_ids = task_ids[args.offset :]
    if args.limit is not None:
        task_ids = task_ids[: args.limit]
    invalid = [task_id for task_id in task_ids if not SAFE_TASK_ID.fullmatch(task_id)]
    if invalid:
        raise ValueError(f"unsafe AppWorld task ID: {invalid[0]!r}")
    return task_ids


def compact_result(mode: str, task_id: str, result: dict[str, Any]) -> dict[str, Any]:
    common = ["reward", "n_steps", "task_completed"]
    baseline = ["total_input_tokens", "total_output_tokens", "avg_step_time", "total_wall_time", "error_steps",
                "infra_retries_used", "token_repairs", "stop_reason", "context_overflow_step"]
    breadth = [
        "env_steps", "wall_time", "spec_breadth", "actor_extra_latency", "hit", "post_hit", "hit_replay_mismatch",
        "hit_not_launched",
        "miss", "no_spec", "no_tool_call_count", "guess_match", "guess_steps", "actor_calls",
        "speculative_actor_calls", "speculative_actor_cancelled", "preexec_total", "preexec_not_verified",
        "preexec_skipped", "avg_guesses", "avg_speculator_seconds", "max_speculator_queue_seconds",
        "speculator_errors", "speculator_cancelled", "actor_tokens_finished", "error_steps", "infra_retries_used",
        "token_repairs", "stop_reason", "context_overflow_step",
        "step_trace",
    ]
    fields = common + (baseline if mode == "baseline" else breadth)
    compact = {"task_id": task_id, "mode": mode}
    compact.update({key: result[key] for key in fields if key in result})
    return compact


def run_one(args: argparse.Namespace, task_id: str, run_dir: Path) -> dict[str, Any]:
    from appworld import AppWorld
    from sa_appworld.runtime.agent import run_agent
    from sa_appworld.runtime.trajectory_logger import TrajectoryLogger

    env = AppWorld(
        task_id=task_id,
        experiment_name=f"smc_{run_dir.name}_{args.mode}",  # AppWorld output dir name; audit_sa_isolation.py reads it
        raise_on_failure=False,
        max_interactions=args.max_steps * 3,
    )
    logger = None
    if args.save_raw_trajectories:
        logger = TrajectoryLogger(str(run_dir / "raw_trajectories"), task_id)
    try:
        if args.mode == "baseline":
            result = run_agent(
                env=env,
                task_instruction=env.task.instruction,
                model=args.actor_model,
                api_base=args.actor_url,
                thinking=args.actor_thinking == "on",
                thinking_budget=args.thinking_budget,
                max_steps=args.max_steps,
                sampling=args.actor_sampling,
                logger=logger,
                max_tokens_override=args.owner_max_tokens,
                actor_extra_latency=args.actor_extra_latency,
                request_timeout=args.actor_request_timeout,
                infra_retries=args.infra_retries,
                repair_access_token=args.repair_access_token,
                stop_on_context_overflow=args.stop_on_context_overflow,
            )
        else:
            from sa_appworld.runtime.sa_breadth_agent import run_sa_breadth_agent

            result = run_sa_breadth_agent(
                env=env,
                task_instruction=env.task.instruction,
                actor_model=args.actor_model,
                speculator_model=args.speculator_model,
                actor_url=args.actor_url,
                speculator_url=args.speculator_url,
                breadth=args.spec_breadth,
                actor_thinking=args.actor_thinking == "on",
                thinking_budget=args.thinking_budget,
                actor_max_tokens=args.owner_max_tokens,
                max_steps=args.max_steps,
                actor_sampling=args.actor_sampling,
                spec_sampling=args.spec_sampling,
                spec_max_tokens=args.spec_max_tokens,
                verified_apis=args.verified_apis,
                actor_extra_latency=args.actor_extra_latency,
                request_timeout=args.actor_request_timeout,
                infra_retries=args.infra_retries,
                prelaunch_url=args.prelaunch_url,
                logger=logger,
                repair_access_token=args.repair_access_token,
                stop_on_context_overflow=args.stop_on_context_overflow,
            )
        return compact_result(args.mode, task_id, result)
    finally:
        env.close()


def served_models(url: str) -> dict[str, Any]:
    """{model id: max_model_len} from GET <url>/models; raises SystemExit when the endpoint does not answer."""
    import urllib.request

    request = urllib.request.Request(url.rstrip("/") + "/models",
                                     headers={"Authorization": f"Bearer {os.environ.get('SMC_API_KEY', 'dummy')}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)["data"]
    except Exception as error:
        raise SystemExit(f"endpoint check: {url}/models failed: {error!r}")
    return {model["id"]: model.get("max_model_len") for model in data}


def check_endpoints(args: argparse.Namespace) -> None:
    """Before any task: every endpoint the run will call must serve the model it is used for, else exit.

    baseline: --actor-url. sa: --actor-url, --prelaunch-url (when set) and --speculator-url; the pre-launch
    endpoint must also report the same max_model_len as the Actor endpoint."""
    expected = [("--actor-url", args.actor_url, args.actor_model)]
    if args.mode == "sa":
        if args.prelaunch_url:
            expected.append(("--prelaunch-url", args.prelaunch_url, args.actor_model))
        expected.append(("--speculator-url", args.speculator_url, args.speculator_model))
    lengths = {}
    for flag, url, model in expected:
        served = served_models(url)
        if model not in served:
            raise SystemExit(f"endpoint check: {flag} {url} serves {sorted(served)}, not {model}")
        lengths[flag] = served[model]
        print(f"endpoint check: {flag} {url} serves {model} (max_model_len {served[model]})", flush=True)
    if "--prelaunch-url" in lengths and lengths["--prelaunch-url"] != lengths["--actor-url"]:
        raise SystemExit(f"endpoint check: max_model_len differs: --actor-url {lengths['--actor-url']}, "
                         f"--prelaunch-url {lengths['--prelaunch-url']}")


def main() -> int:
    args = parse_args()
    if args.prelaunch_url and args.mode != "sa":
        raise SystemExit("--prelaunch-url needs --mode sa")
    if args.spec_breadth < 1:
        raise SystemExit(f"--spec-breadth must be >= 1, got {args.spec_breadth}")
    if args.actor_extra_latency < 0:
        raise SystemExit(f"--actor-extra-latency must be >= 0, got {args.actor_extra_latency}")
    if args.spec_breadth > 1 and args.mode != "sa":
        raise SystemExit("--spec-breadth > 1 needs --mode sa")
    from sa_appworld.runtime.agent import sampling_request

    for flag, sampling in (("--actor-sampling", args.actor_sampling), ("--spec-sampling", args.spec_sampling or {})):
        try:
            sampling_request(sampling)
        except (ValueError, AttributeError) as error:
            raise SystemExit(f"{flag}: {error}")
    args.verified_apis = frozenset()
    if args.mode == "sa":
        if not args.spec_sampling:
            raise SystemExit("--mode sa needs explicit Speculator sampling: --spec-sampling")
        verified_path = Path(args.preexec_verified_apis)
        if not verified_path.is_file():
            raise SystemExit(f"verified API list not found: {verified_path} (scripts/test_preexec_isolation.py)")
        args.verified_apis = frozenset(json.loads(verified_path.read_text(encoding="utf-8"))["apis"])
        if not args.verified_apis:
            raise SystemExit(f"verified API list is empty: {verified_path}")
    check_endpoints(args)
    from appworld import update_root

    update_root(args.appworld_root)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir).expanduser()
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    run_dir = out_root / f"{run_id}_{args.mode}"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    task_ids = load_requested_task_ids(args)
    config = vars(args).copy()
    config["verified_apis"] = sorted(args.verified_apis)
    config["task_count"] = len(task_ids)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = []
    summary = build_run_summary(args.mode, len(task_ids), results)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for index, task_id in enumerate(task_ids, start=1):
        result_path = results_dir / f"{task_id}.json"
        if args.resume and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"[{index}/{len(task_ids)}] {task_id}: resume", flush=True)
        else:
            print(f"[{index}/{len(task_ids)}] {task_id}: {args.mode}", flush=True)
            try:
                result = run_one(args, task_id, run_dir)
            except Exception as exc:
                traceback.print_exc()
                result = {"task_id": task_id, "mode": args.mode, "error_type": type(exc).__name__}
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        summary = build_run_summary(args.mode, len(task_ids), results)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(f"summary={run_dir / 'summary.json'}", flush=True)
    return run_exit_code(summary)


if __name__ == "__main__":
    sys.exit(main())
