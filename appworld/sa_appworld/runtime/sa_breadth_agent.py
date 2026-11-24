"""Speculative Actions (SA) with k guesses per step on AppWorld (run.py --mode sa --spec-breadth k).

Two roles, as in the original Speculative Actions method:
- Actor: the slow, authoritative model. Every committed action is an Actor action, executed on the live environment.
- Speculator: the fast model. It guesses the Actor's next action.

One step t:
1. The Actor generates step t from the committed history H.
2. At the same time the Speculator samples k responses (n=k, --spec-sampling) and keeps the distinct tool
   calls as guesses.
3. While the Actor is still running, the main thread takes the guesses one by one (checking the Actor before each)
   and, for each guess g_i whose API passed the per-API isolation test (preexec_verified_apis_v4.json), executes it
   in isolation (env_isolation.EnvIsolation): the call runs on the current state and every change it makes is
   undone, so the live environment is untouched whatever the API does. With its output o_i, the runner pre-launches
   the Actor's next call on H + (g_i, o_i): a speculative Actor call for step t+1. Unverified APIs and complete_task
   are never executed early.
4. When the Actor returns a_t, a_t is executed on the live environment (always), giving o_t. Then:
   - a_t equals guess g_i exactly (name and all arguments), g_i's speculative call exists and o_t == o_i: commit
     (a_t, o_t), and that call becomes the Actor call for step t+1 ("hit");
   - the same, but o_t != o_i: the speculative call saw a wrong observation and is cancelled ("hit_replay_mismatch");
   - a_t equals a guess that has no speculative call (unverified API, or the Actor returned first): "hit_not_launched";
   - no guess equals a_t: "miss"; the guesses were not ready when the Actor returned: "no_spec".
   All other speculative calls are cancelled: requests are streamed, so a cancel closes the connection and the server
   stops generating.
5. After a hit, step t+1 only waits for the pre-launched Actor call and executes its action; it runs no speculation
   ("post_hit"), as in Algorithm 1 of the paper (a cache hit awaits the cached call and continues) and its Appendix A
   ("no speculation window immediately after a correct speculation"). Step t+2 speculates again.

Threads: model requests run off the main thread (Actor streams in worker threads, Speculator requests as tasks of
an asyncio loop in one background thread); every AppWorld operation runs in the main thread, so there is no shared
environment across threads and AppWorld's own timeouts stay on.

Speculator requests never queue: each runs as its own asyncio task as soon as it is started. When the Actor's action
for step t arrives, step t's Speculator request, if still running, is cancelled at once (its guesses can no longer be
used): cancelling the task closes the HTTP connection, and vLLM aborts a request whose client disconnects.

A hit's speculative call saw exactly the history the Actor would see (same action, same real observation), and no
speculative execution leaves a trace, so the trajectory is the sequential Actor trajectory (up to server-side
numerical nondeterminism).
"""

import asyncio
import concurrent.futures
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

from .agent import (_build_canonical_message, _get_client, _is_infra_error, _thinking_body, initial_messages,
                    is_context_overflow, repair_token_args,
                    perf_counter, sampling_request, stream_chat)
from .agent import _parse_qwen_tool_call
from .agent import _parse_tool_call as _agent_parse_tool_call
from .agent import execute_tool as _raw_execute_tool
from .env_isolation import EnvIsolation, IsolationError
from .tools import TOOLS
from .trajectory_logger import TrajectoryLogger


def execute_tool(env, tool_name: str, tool_args: dict) -> str:
    """Wrapper around agent.execute_tool that GUARANTEES the AppWorld
    safety_guard is disabled after the call.

    AppWorld's env.execute() enables a process-wide safety_guard (monkey-
    patches builtins.open). If env.execute hits an exception path that
    bypasses its normal cleanup (e.g., IPython internal state issue under
    threading), the guard is left enabled and subsequent file I/O in the
    main thread (including logger writes) fails with PermissionError.
    This wrapper ensures the guard is always reset before returning.
    """
    try:
        return _raw_execute_tool(env, tool_name, tool_args)
    finally:
        try:
            env.safety_guard.disable()
        except Exception:
            pass


def _parse_tool_call(message) -> tuple:
    """Extract (tool_name, tool_args, raw_msg_dict) from response message.

    Delegates native tool_call + Qwen XML fallback to agent.py; the extra
    msg_dict return is what the main loop needs for history.
    """
    msg_dict = {"role": message.role, "content": message.content}
    name, args = _agent_parse_tool_call(message)
    if name and message.tool_calls:
        tc = message.tool_calls[0]
        msg_dict["tool_calls"] = [{
            "id": tc.id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]
        return name, args, msg_dict
    if message.content:
        name, args = _parse_qwen_tool_call(message.content)
        if name:
            return name, args, msg_dict
    return "", {}, msg_dict


def _action_norm(name: str, args: dict) -> str:
    """Normalize action to 'app.api' for printing and whitelist lookup."""
    if name == "execute_api":
        return f"{args.get('app_name','?')}.{args.get('api_name','?')}"
    return name

# The agent's dedicated tools and the AppWorld API each one calls (agent.execute_tool).
DIRECT_TOOL_APIS = {
    "show_app_descriptions": "api_docs.show_app_descriptions",
    "show_api_descriptions": "api_docs.show_api_descriptions",
    "show_api_doc": "api_docs.show_api_doc",
    "search_api_docs": "api_docs.search_api_docs",
    "show_account_passwords": "supervisor.show_account_passwords",
    "show_active_task": "supervisor.show_active_task",
    "show_profile": "supervisor.show_profile",
    "show_addresses": "supervisor.show_addresses",
    "show_payment_cards": "supervisor.show_payment_cards",
    "complete_task": "supervisor.complete_task",
}
# Never executed early: it ends the task.
NEVER_PREEXECUTED = frozenset({"supervisor.complete_task"})


def api_of(name: str, args: Dict[str, Any]) -> str:
    """The AppWorld API ("app.api") a tool call invokes."""
    if name == "execute_api":
        return _action_norm("execute_api", args)
    return DIRECT_TOOL_APIS.get(name, name)


def action_key(name: str, args: Dict[str, Any]) -> str:
    """Exact identity of a tool call: name plus the canonical JSON of all arguments."""
    return name + "\x00" + json.dumps(args or {}, sort_keys=True)


class ActorCall:
    """One streamed Actor request that can be cancelled while it runs."""

    def __init__(self, pool, client, model, messages, max_tokens, sampling, extra_body, extra_latency=0.0,
                 tools=TOOLS):
        self.cancel_event = threading.Event()
        self.submit_time = perf_counter()
        self.future = pool.submit(
            stream_chat, client, model, messages, max_tokens, sampling, extra_body, self.cancel_event, extra_latency,
            tools)

    def cancel(self) -> None:
        self.cancel_event.set()


class SpeculatorRequests:
    """Speculator requests as tasks of an asyncio loop in one background thread: no worker pool, so no queue.

    submit() returns a concurrent.futures.Future; cancelling it cancels the task, which closes the HTTP connection
    (the server then aborts the request). A cancelled request's future raises CancelledError."""

    def __init__(self, url: str):
        from openai import AsyncOpenAI

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="speculator-loop", daemon=True)
        self.thread.start()
        # Same key, timeout and retries as agent._get_client.
        self.client = AsyncOpenAI(base_url=url, api_key=os.environ.get("SMC_API_KEY", "dummy"), timeout=300.0,
                                  max_retries=2)

    def submit(self, request_kwargs: Dict[str, Any]) -> "concurrent.futures.Future":
        submitted = perf_counter()

        async def run():
            started = perf_counter()
            response = await self.client.chat.completions.create(**request_kwargs)
            return response, perf_counter() - started, started - submitted

        return asyncio.run_coroutine_threadsafe(run(), self.loop)

    def close(self) -> None:
        async def shutdown():
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()
            await self.client.close()

        try:
            asyncio.run_coroutine_threadsafe(shutdown(), self.loop).result(timeout=30)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=30)


@dataclass
class Guess:
    name: str
    args: Dict[str, Any]
    key: str
    api: str
    sample_count: int
    # pending, isolated, not_verified, never, skipped (the Actor returned first)
    preexec: str = "pending"
    observation: Optional[str] = None
    speculative_call: Optional[ActorCall] = None


@dataclass
class Speculation:
    """The Speculator's work for one step. Its request runs in a thread; everything else happens in the main thread:
    `guesses` are filled from the response, then pre-executed one by one while the Actor is still running."""

    step: int
    history: List[Dict[str, Any]]
    future: "concurrent.futures.Future"
    guesses: List[Guess] = field(default_factory=list)
    parsed: bool = False
    next_guess: int = 0
    speculator_seconds: float = 0.0
    # Time between submit and the request actually starting (no queue: should be ~0).
    speculator_queue_seconds: Optional[float] = None
    speculator_error: Optional[str] = None
    # The Actor's action arrived first and the still-running request was cancelled (not an error).
    cancelled: bool = False
    actor_key: Optional[str] = None

    @property
    def drained(self) -> bool:
        return self.parsed and self.next_guess >= len(self.guesses)


def run_sa_breadth_agent(
    env,
    task_instruction: str,
    *,
    actor_model: str,
    speculator_model: str,
    actor_url: str,
    speculator_url: str,
    breadth: int,
    actor_thinking: bool,
    thinking_budget: int,
    actor_max_tokens: int,
    max_steps: int,
    actor_sampling: Dict[str, Any],
    spec_sampling: Dict[str, Any],
    spec_max_tokens: int,
    verified_apis: FrozenSet[str],
    actor_extra_latency: float = 0.0,
    request_timeout: float = 300.0,
    infra_retries: int = 0,
    prelaunch_url: Optional[str] = None,
    logger: Optional[TrajectoryLogger] = None,
    repair_access_token: bool = False,
    stop_on_context_overflow: bool = False,
) -> dict:
    """Threads: only model requests run in worker threads (the Actor's streams and the Speculator's request). Every
    AppWorld operation (real calls, isolated calls, evaluation) runs in the calling thread, so AppWorld's
    signal-based timeouts stay on, as in the baseline.

    request_timeout / infra_retries (run.py --actor-request-timeout / --infra-retries, as in the baseline): the Actor
    client's timeout, and how many times per step an Actor call that failed on the server or the connection (timeout,
    connection error, 5xx) is relaunched on the same history without using up a step.

    Everything the Actor side does matches the baseline (agent.run_agent): the same request (agent.stream_chat), and
    an Actor call that still fails (retries used up, or a model-side error such as context too long) turns the step
    into an _error step that uses up a step and leaves the history unchanged. There is no cap on consecutive replies
    without a tool call.

    prelaunch_url (run.py --prelaunch-url): the endpoint of the pre-launched Actor calls F_i (H_t + g_i + o_i); None
    (default) sends them to actor_url, as before. It must serve the same model with the same settings as actor_url
    (run.py checks the model name and max_model_len), so a kept F_i has the distribution of a normal Actor call.
    Every normal Actor call, including an infra-retry of a kept F_i, goes to actor_url.

    repair_access_token (run.py --repair-access-token, default off): agent.repair_token_args on the Actor's action
    (committed history) and on every guess (its speculation's history, the same one), before matching, pre-execution
    and the real call; so a guess and the Actor's action that differ only by a miscopied token count as equal.

    stop_on_context_overflow (run.py --stop-on-context-overflow, default off): as in the baseline (agent.run_agent),
    an Actor call refused because the prompt exceeds the context window ends the task after that one _error step
    (stop_reason "context_overflow"), instead of resending the same history up to max_steps."""
    if breadth < 1:
        raise ValueError(f"run_sa_breadth_agent needs breadth >= 1, got {breadth}")
    isolation = EnvIsolation(env, execute_tool)

    actor_client = _get_client(actor_url, timeout=request_timeout)
    prelaunch_client = (actor_client if prelaunch_url in (None, actor_url)
                        else _get_client(prelaunch_url, timeout=request_timeout))
    # Every Actor call, speculative or not, sends the same request body (the baseline's: thinking switch and
    # sampling); only the history differs. Thinking is explicit either way (chat_template_kwargs.enable_thinking).
    actor_kwargs, actor_sampling_extra = sampling_request(actor_sampling)
    actor_extra = {**_thinking_body(actor_thinking, thinking_budget), **actor_sampling_extra}
    spec_kwargs, spec_sampling_extra = sampling_request(spec_sampling)
    speculator_sampling = {**spec_kwargs, "n": breadth}
    speculator_extra = {"chat_template_kwargs": {"enable_thinking": False}, **spec_sampling_extra}

    actor_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4 * breadth + 4)
    speculator_requests = SpeculatorRequests(speculator_url)

    # Same first messages as the baseline (agent.run_agent).
    messages: List[Dict[str, Any]] = initial_messages(task_instruction, env.task.supervisor, env.task.datetime)
    tools = TOOLS
    speculations: List[Speculation] = []
    open_calls: List[ActorCall] = []
    counts = {"actor_calls": 0, "speculative_actor_calls": 0, "speculative_actor_cancelled": 0, "hit": 0,
              "hit_replay_mismatch": 0, "hit_not_launched": 0, "miss": 0, "no_spec": 0, "post_hit": 0,
              "no_tool_call": 0, "error": 0, "preexec_total": 0, "preexec_not_verified": 0, "preexec_skipped": 0,
              "speculator_cancelled": 0, "token_repairs": 0}
    step_trace: List[Dict[str, Any]] = []
    actor_tokens_finished = 0

    def launch_actor(history: List[Dict[str, Any]], client=actor_client) -> ActorCall:
        # Every Actor call, normal or pre-launched, gets the same simulated extra latency (0 = off).
        call = ActorCall(actor_pool, client, actor_model, list(history), actor_max_tokens,
                         actor_kwargs, actor_extra, actor_extra_latency, tools)
        open_calls.append(call)
        return call

    def start_speculation(step: int) -> Speculation:
        history = list(messages)
        future = speculator_requests.submit(dict(
            model=speculator_model, messages=history, tools=tools, max_tokens=spec_max_tokens,
            extra_body=speculator_extra, **speculator_sampling))
        speculation = Speculation(step=step, history=history, future=future)
        speculations.append(speculation)
        return speculation

    def parse_guesses(speculation: Speculation) -> None:
        speculation.parsed = True
        try:
            response, seconds, queue_seconds = speculation.future.result()
        except concurrent.futures.CancelledError:
            speculation.cancelled = True
            return
        except Exception as error:
            speculation.speculator_error = str(error)[:300]
            return
        speculation.speculator_seconds = seconds
        speculation.speculator_queue_seconds = queue_seconds
        by_key: Dict[str, Guess] = {}
        for choice in response.choices:
            name, args, _ = _parse_tool_call(choice.message)
            if not name:
                continue
            if repair_access_token:
                args, _ = repair_token_args(name, args, speculation.history)
            key = action_key(name, args)
            if key in by_key:
                by_key[key].sample_count += 1
                continue
            by_key[key] = Guess(name=name, args=args, key=key, api=api_of(name, args), sample_count=1)
        # Most-sampled guesses first: they get their observation and speculative Actor call earliest.
        speculation.guesses = sorted(by_key.values(), key=lambda guess: -guess.sample_count)

    def preexecute_next(speculation: Speculation) -> None:
        """Isolate the next guess and pre-launch its Actor call. Runs in the main thread only."""
        guess = speculation.guesses[speculation.next_guess]
        speculation.next_guess += 1
        if guess.api in NEVER_PREEXECUTED:
            guess.preexec = "never"
            return
        if guess.api not in verified_apis:
            guess.preexec = "not_verified"
            counts["preexec_not_verified"] += 1
            return
        # IsolationError propagates: the environment may be dirty, so the task stops here and is not evaluated.
        observation = isolation.execute(guess.name, guess.args)
        counts["preexec_total"] += 1
        guess.preexec = "isolated"
        guess.observation = observation
        assistant_message, tool_message = _build_canonical_message(
            speculation.step, guess.name, guess.args, observation)
        guess.speculative_call = launch_actor(speculation.history + [assistant_message, tool_message],
                                             client=prelaunch_client)
        counts["speculative_actor_calls"] += 1

    def wait_for_actor(actor: ActorCall, speculation: Optional[Speculation]):
        """Return the Actor's result; while it runs, work through the guesses, checking the Actor before each one."""
        while True:
            if actor.future.done() or speculation is None or speculation.drained:
                return actor.future.result()
            if speculation.parsed:
                preexecute_next(speculation)
                continue
            if speculation.future.done():
                parse_guesses(speculation)
                continue
            concurrent.futures.wait([actor.future, speculation.future],
                                    return_when=concurrent.futures.FIRST_COMPLETED)

    def close_speculation(speculation: Speculation, keep: Optional[ActorCall]) -> None:
        # The Actor's action is here: a Speculator request still running for this step is useless, cancel it now.
        if not speculation.future.done() and speculation.future.cancel():
            speculation.cancelled = True
            counts["speculator_cancelled"] += 1
        for guess in speculation.guesses[speculation.next_guess:]:
            guess.preexec = "skipped"
            counts["preexec_skipped"] += 1
        speculation.next_guess = len(speculation.guesses)
        for guess in speculation.guesses:
            if guess.speculative_call is not None and guess.speculative_call is not keep:
                guess.speculative_call.cancel()
                counts["speculative_actor_cancelled"] += 1

    def execute_real(name: str, args: Dict[str, Any]) -> str:
        output = execute_tool(env, name, args)
        isolation.commit_point()
        return output

    start_time = perf_counter()
    actor = launch_actor(messages)
    counts["actor_calls"] += 1
    # None: this step has no speculation, because the previous step was a hit.
    speculation: Optional[Speculation] = start_speculation(1)
    committed_step = 0
    task_completed = False
    error_steps = 0
    infra_retries_used = 0
    step_retries = 0  # infra retries spent on the current step
    context_overflow_step = None  # --stop-on-context-overflow

    try:
        while committed_step < max_steps:
            wait_start = perf_counter()
            try:
                result = wait_for_actor(actor, speculation)
            except IsolationError:
                raise
            except Exception as error:
                if step_retries < infra_retries and _is_infra_error(error):
                    # Relaunch the step's Actor call on the same committed history. A post-hit step (no
                    # speculation) stays without one; a speculated step gets a fresh speculation.
                    step_retries += 1
                    infra_retries_used += 1
                    print(f"  infra retry {step_retries}/{infra_retries} at step {committed_step + 1}: "
                          f"{type(error).__name__}", flush=True)
                    if speculation is not None:
                        close_speculation(speculation, keep=None)
                    actor = launch_actor(messages)
                    counts["actor_calls"] += 1
                    if speculation is not None:
                        speculation = start_speculation(committed_step + 1)
                    continue
                # As in the baseline: the step becomes an _error step. It uses up a step, nothing is executed, the
                # history is unchanged, and the next step asks the Actor again (with a fresh speculation).
                step = committed_step + 1
                if speculation is not None:
                    close_speculation(speculation, keep=None)
                error_steps += 1
                counts["error"] += 1
                committed_step = step
                step_retries = 0
                step_trace.append({"step": step, "path": "error", "actor_wait": round(perf_counter() - wait_start, 3),
                                   "error": f"{type(error).__name__}: {str(error)[:200]}"})
                if logger:
                    logger.log_step(
                        step_id=step, instruction=task_instruction[:500], model_output="", api_name="_error",
                        api_params={}, api_result=f"Exception: {error!r}"[:2000],
                        wall_clock_time=perf_counter() - wait_start, input_tokens=0, output_tokens=0, tier="error")
                print(f"  step {step}: [error] {type(error).__name__}: {str(error)[:200]}", flush=True)
                if stop_on_context_overflow and is_context_overflow(error):
                    context_overflow_step = step
                    print(f"  step {step}: context window exceeded, task stopped (--stop-on-context-overflow)",
                          flush=True)
                    break
                if committed_step >= max_steps:
                    break
                actor = launch_actor(messages)
                counts["actor_calls"] += 1
                speculation = start_speculation(committed_step + 1)
                continue
            step_retries = 0
            actor_wait = perf_counter() - wait_start
            message, _, completion_tokens = result
            actor_tokens_finished += completion_tokens
            step = committed_step + 1
            name, args, raw_message = _parse_tool_call(message)
            token_original = None  # --repair-access-token: the token the Actor wrote, when it was replaced
            if repair_access_token and name:
                args, token_original = repair_token_args(name, args, messages)
                counts["token_repairs"] += token_original is not None
            match: Optional[Guess] = None
            guesses: List[Guess] = []
            speculator_seconds = 0.0
            if speculation is not None:
                if not speculation.parsed and speculation.future.done():
                    parse_guesses(speculation)  # arrived together with the Actor: still counts for matching
                speculation.actor_key = action_key(name, args) if name else None
                guesses = list(speculation.guesses)
                match = next((guess for guess in guesses if name and guess.key == speculation.actor_key), None)
                speculator_seconds = speculation.speculator_seconds
            if not name:
                path = "no_tool_call"
            elif speculation is None:
                path = "post_hit"
            elif match is not None and match.speculative_call is not None:
                path = "hit"  # confirmed below by the replay
            elif match is not None:
                path = "hit_not_launched"
            elif speculation.parsed:
                path = "miss"
            else:
                path = "no_spec"
            if speculation is not None:
                close_speculation(speculation, keep=match.speculative_call if path == "hit" else None)
            guess_trace = [{"api": g.api, "n": g.sample_count, "pre": g.preexec,
                            "launched": g.speculative_call is not None} for g in guesses]

            if not name:
                counts["no_tool_call"] += 1
                content = raw_message.get("content") or ""
                if content.strip():
                    messages.append({"role": "assistant", "content": content})
                else:
                    feedback = execute_real("_no_tool_call", {})
                    messages.extend(_build_canonical_message(step, "_no_tool_call", {}, feedback))
                committed_step = step
                step_trace.append({"step": step, "path": path, "actor_wait": round(actor_wait, 3),
                                   "guesses": guess_trace})
                if committed_step >= max_steps:
                    break
                actor = launch_actor(messages)
                counts["actor_calls"] += 1
                speculation = start_speculation(committed_step + 1)
                continue

            env_start = perf_counter()
            tool_result = execute_real(name, args)
            env_seconds = perf_counter() - env_start
            next_actor: Optional[ActorCall] = None
            if path == "hit":
                if tool_result == match.observation:
                    next_actor = match.speculative_call
                else:
                    match.speculative_call.cancel()
                    counts["speculative_actor_cancelled"] += 1
                    path = "hit_replay_mismatch"
            counts[path] += 1
            committed_step = step
            messages.extend(_build_canonical_message(step, name, args, tool_result))
            step_trace.append({
                "step": step, "path": path, "tool": api_of(name, args), "actor_wait": round(actor_wait, 3),
                "env_seconds": round(env_seconds, 3), "speculator_seconds": round(speculator_seconds, 3),
                "guesses": guess_trace,
            })
            if logger:
                logger.log_step(
                    step_id=step, instruction=task_instruction[:500],
                    model_output=json.dumps({"tool": name, "args": args, **(
                        {"token_repaired_from": token_original} if token_original is not None else {})}),
                    api_name=name, api_params=args,
                    api_result=tool_result if isinstance(tool_result, str) else str(tool_result),
                    wall_clock_time=actor_wait, input_tokens=0, output_tokens=0, tier=path)
            print(f"  step {step}: [{path}] {api_of(name, args)} actor_wait={actor_wait:.2f}s "
                  f"guesses={len(guesses)} speculator={speculator_seconds:.2f}s", flush=True)
            if name == "complete_task":
                task_completed = True
                break
            if committed_step >= max_steps:
                break
            if next_actor is not None:
                # Hit: the pre-launched call is step t+1's Actor call, and step t+1 runs no speculation.
                actor = next_actor
                speculation = None
            else:
                actor = launch_actor(messages)
                counts["actor_calls"] += 1
                speculation = start_speculation(committed_step + 1)
    finally:
        # Also on an exception (IsolationError, Actor failure): stop every Actor stream and Speculator request.
        wall_time = perf_counter() - start_time
        for call in open_calls:
            call.cancel()
        for speculation_record in speculations:
            speculation_record.future.cancel()
        speculator_requests.close()
        actor_pool.shutdown(wait=False, cancel_futures=True)

    try:
        evaluation = env.evaluate(suppress_errors=True)
        reward = evaluation.pass_percentage / 100.0
        print(f"\n  Eval: success={evaluation.success} passed={evaluation.pass_count}/{evaluation.total_count}")
    except Exception as error:
        print(f"\n  Eval error: {error}")
        reward = 0.0
    if logger:
        logger.finalize(task_success=(reward > 0.99), evaluation_score=reward)

    # Guess accuracy over the speculated Actor steps whose guesses arrived (cancelled requests have none).
    for speculation_record in speculations:
        if not speculation_record.parsed and speculation_record.future.done():
            parse_guesses(speculation_record)
    verified = [s for s in speculations if s.actor_key is not None and s.parsed and not s.cancelled
                and s.speculator_error is None]
    guess_match = sum(any(g.key == s.actor_key for g in s.guesses) for s in verified)
    speculator_times = [s.speculator_seconds for s in speculations if s.speculator_seconds > 0]
    queue_times = [s.speculator_queue_seconds for s in speculations if s.speculator_queue_seconds is not None]
    guess_counts = [len(s.guesses) for s in speculations if s.parsed]
    env_steps = sum(counts[key] for key in ("hit", "hit_replay_mismatch", "hit_not_launched", "miss", "no_spec",
                                            "post_hit"))
    print(f"  Paths: hit={counts['hit']} post_hit={counts['post_hit']} replay_mismatch={counts['hit_replay_mismatch']} "
          f"hit_not_launched={counts['hit_not_launched']} miss={counts['miss']} no_spec={counts['no_spec']} "
          f"no_tool_call={counts['no_tool_call']} error={counts['error']}  guess_match={guess_match}/{len(verified)}  "
          f"speculative_actor_calls={counts['speculative_actor_calls']} "
          f"cancelled={counts['speculative_actor_cancelled']}  wall={wall_time:.0f}s")
    return {
        "reward": reward,
        "n_steps": committed_step,
        "env_steps": env_steps,
        "task_completed": task_completed,
        "wall_time": round(wall_time, 1),
        "spec_breadth": breadth,
        "actor_extra_latency": actor_extra_latency,
        "hit": counts["hit"],
        "hit_replay_mismatch": counts["hit_replay_mismatch"],
        "hit_not_launched": counts["hit_not_launched"],
        "miss": counts["miss"],
        "no_spec": counts["no_spec"],
        "post_hit": counts["post_hit"],
        "no_tool_call_count": counts["no_tool_call"],
        "guess_match": guess_match,
        "guess_steps": len(verified),
        "actor_calls": counts["actor_calls"],
        "speculative_actor_calls": counts["speculative_actor_calls"],
        "speculative_actor_cancelled": counts["speculative_actor_cancelled"],
        "preexec_total": counts["preexec_total"],
        "preexec_not_verified": counts["preexec_not_verified"],
        "preexec_skipped": counts["preexec_skipped"],
        "avg_guesses": round(sum(guess_counts) / max(len(guess_counts), 1), 2),
        "avg_speculator_seconds": round(sum(speculator_times) / max(len(speculator_times), 1), 3),
        "max_speculator_queue_seconds": round(max(queue_times, default=0.0), 3),
        # Real request failures only; cancellations (the Actor arrived first) are counted separately.
        "speculator_errors": sum(s.speculator_error is not None for s in speculations),
        "speculator_cancelled": counts["speculator_cancelled"],
        "actor_tokens_finished": actor_tokens_finished,
        "error_steps": error_steps,
        "infra_retries_used": infra_retries_used,
        **({"token_repairs": counts["token_repairs"]} if repair_access_token else {}),
        **({"stop_reason": "context_overflow" if context_overflow_step is not None else
            ("completed" if task_completed else "max_steps"), "context_overflow_step": context_overflow_step}
           if stop_on_context_overflow else {}),
        "step_trace": step_trace,
    }
