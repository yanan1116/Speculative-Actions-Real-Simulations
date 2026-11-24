"""Baseline AppWorld agent using tool-calling format via vLLM.

Uses a generic execute_api tool to call any AppWorld API,
plus discovery tools (show_api_descriptions, show_api_doc, etc.)
and supervisor shortcuts.
"""

import ctypes
import ctypes.util
import json
import os
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any, Optional

# AppWorld uses freezegun which freezes time.time() AND time.perf_counter().
# Use ctypes clock_gettime to bypass the freeze.
_CLOCK_MONOTONIC = 1
class _Timespec(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]
_librt = ctypes.CDLL(ctypes.util.find_library('rt') or 'librt.so.1')
_ts = _Timespec()

def perf_counter():
    _librt.clock_gettime(_CLOCK_MONOTONIC, ctypes.byref(_ts))
    return _ts.tv_sec + _ts.tv_nsec / 1e9

from openai import OpenAI

from .tools import TOOLS
from .trajectory_logger import TrajectoryLogger


DISABLE_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

# Baseline and --mode sa prompt: AppWorld's official ReAct instructions (the "Key instructions" of the official
# prompt, as used by the SkillOpt AppWorld line), rewritten for this agent's tool-call action space (one tool call
# per step instead of Python code blocks). The official prompt's worked example is code and is not included.
APPWORLD_SYSTEM_PROMPT = """I am your supervisor, and you are an AI Assistant whose job is to complete my day-to-day tasks fully autonomously.

To do this, you will interact with app(s) (e.g., spotify, venmo etc) using their APIs on my behalf, in a multi-step conversation: in each step you make exactly one tool call, I execute it and show you the result, and based on it you make the next call, until the task is done.

Tools:
- show_app_descriptions: list the available apps.
- show_api_descriptions(app_name): list the APIs of an app.
- show_api_doc(app_name, api_name): the specification of an API (input arguments and output JSON format).
- search_api_docs(query): search the API documentation.
- execute_api(app_name, api_name, parameters): call any app API with the given parameters.
- show_account_passwords, show_profile, show_addresses, show_payment_cards, show_active_task: shortcuts to the Supervisor app.
- complete_task(answer, status): finish the task (see D below).

**Key instructions**:

A. General instructions:

- Act fully on your own. You must make all decisions yourself and never ask me or anyone else to confirm or clarify. Your role is to solve the task, not to bounce questions back, or provide me directions to follow.
- You have full access -- complete permission to operate across my connected accounts and services.
- Never invent or guess values. For example, if I ask you to play a song, do not assume the ID is 123. Instead, look it up properly through the right API.
- Never leave placeholders; don't output things like "your_username". Always fill in the real value by retrieving it via APIs (e.g., Supervisor app for credentials).
- When I omit details, choose any valid value. For example, if I ask you to buy something but don't specify which payment card to use, you may pick any one of my available cards.
- Avoid collateral damage. Only perform what I explicitly ask for. Example: if I ask you to buy something, do not delete emails, return the order, or perform unrelated account operations.

B. App-specific instructions:

- All my personal information (biographical details, credentials, addresses, cards) is stored in the Supervisor app, accessible via its APIs.
- Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
- Always obtain the current date or time from the phone app's get_current_date_and_time API (or the date and time given with the task), never from your internal clock.
- All requests are concerning a single, default (no) time zone.
- For temporal requests, use proper time boundaries, e.g., when asked about periods like "yesterday", use complete ranges: 00:00:00 to 23:59:59.
- References to "file system" mean the file system app, not the machine's OS.
- Paginated APIs: Always process all results, calling the API again with page_index = 0, 1, 2, ... until a page comes back empty. Don't stop at the first page. Use the largest page_limit the API allows to need fewer calls.

C. Tool-call instructions:

- Always look at the API specification (show_api_doc) before calling an API for the first time.
- Make exactly one tool call per step, and make sure everything is working correctly before making any irreversible changes.
- Use the exact parameter names and types from the API specification. Access tokens come from the app's login API.
- The API documentation has both the input arguments and the output JSON format. Use this information when making API calls and reading their outputs.

D. Task-completion instructions:

You must call the complete_task tool after completing the task.
- If an answer is needed, e.g., for "How many songs are in the Spotify queue?", call it with the appropriate answer argument value.
- If no answer is required, e.g., for "Start my Spotify music player.", omit the answer argument.
- The task is doable, but if you cannot find a way, you can call it with status="fail" to exit with failure.

When the answer is given:
- Keep answers minimal. Return only the entity, number, or direct value requested - not full sentences.
  E.g., for the song title of the current playing track, return just the title.
- Numbers must be numeric and not in words.
  E.g., for the number of songs in the queue, return "10", not "ten".
"""


def initial_messages(task_instruction: str, supervisor: Any, task_datetime: Any) -> list:
    """The first two messages of the baseline and --mode sa agents: the official task header (supervisor's name,
    email, phone; task; current date and time) after APPWORLD_SYSTEM_PROMPT. supervisor: anything with first_name,
    last_name, email, phone_number (env.task.supervisor, or a task's specs.json "supervisor" dict)."""
    get = (lambda key: supervisor[key]) if isinstance(supervisor, dict) else (lambda key: getattr(supervisor, key))
    when = task_datetime.isoformat() if hasattr(task_datetime, "isoformat") else str(task_datetime)
    user = (f"My name is: {get('first_name')} {get('last_name')}. My personal email is {get('email')} and phone "
            f"number is {get('phone_number')}.\nTask: {task_instruction}\n\nThe current date and time is {when}.")
    system = APPWORLD_SYSTEM_PROMPT
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _get_client(api_base: str, timeout: float = 300.0) -> OpenAI:
    # SMC_API_KEY: key for endpoints that require one; unset keeps the original "dummy".
    return OpenAI(base_url=api_base, api_key=os.environ.get("SMC_API_KEY", "dummy"), timeout=timeout, max_retries=2)


def _is_infra_error(error: Exception) -> bool:
    """A failure of the model server or the connection (timeout, connection error, 5xx), not of the model."""
    import openai
    if isinstance(error, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    return isinstance(error, openai.APIStatusError) and getattr(error, "status_code", 0) >= 500


def is_context_overflow(error: Exception) -> bool:
    """The server refused the request because the prompt alone exceeds the model's context window (vLLM 400
    "maximum context length"). The history only grows, so every later request with it fails the same way."""
    import openai
    return isinstance(error, openai.BadRequestError) and "maximum context length" in str(error)


def _thinking_body(thinking: bool, thinking_budget: int = 4096):
    if not thinking:
        return DISABLE_THINKING
    body = {"chat_template_kwargs": {"enable_thinking": True}}
    if thinking_budget > 0:
        body["chat_template_kwargs"]["thinking_budget"] = thinking_budget
    return body


# Sampling fields the OpenAI client has no keyword for: vLLM reads them from the request body (extra_body).
_EXTRA_BODY_SAMPLING = ("top_k", "min_p", "repetition_penalty")
SAMPLING_FIELDS = ("temperature", "top_p", "presence_penalty", "frequency_penalty") + _EXTRA_BODY_SAMPLING


def sampling_request(sampling: dict) -> tuple[dict, dict]:
    """Split a sampling dict (run.py --actor-sampling / --spec-sampling) into the OpenAI client's keyword arguments
    and the vLLM fields that go in extra_body."""
    unknown = sorted(set(sampling) - set(SAMPLING_FIELDS))
    if unknown:
        raise ValueError(f"unknown sampling fields {unknown}; allowed: {list(SAMPLING_FIELDS)}")
    kwargs = {key: value for key, value in sampling.items() if key not in _EXTRA_BODY_SAMPLING}
    extra = {key: value for key, value in sampling.items() if key in _EXTRA_BODY_SAMPLING}
    return kwargs, extra


def _parse_tool_call(message) -> tuple[str, dict]:
    """Extract tool name and arguments from response message."""
    if message.tool_calls and len(message.tool_calls) > 0:
        tc = message.tool_calls[0]
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
        except (json.JSONDecodeError, TypeError):
            args = {}
        return name, args
    return "", {}


def _parse_qwen_tool_call(content: str) -> tuple[str, dict]:
    """Parse Qwen3.5's XML-style tool calls from content text."""
    import re
    if not content or "<tool_call>" not in content:
        return "", {}
    match = re.search(
        r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>',
        content, re.DOTALL
    )
    if not match:
        return "", {}
    func_name = match.group(1)
    params_text = match.group(2)
    params = {}
    for param_match in re.finditer(
        r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', params_text, re.DOTALL
    ):
        key = param_match.group(1)
        value = param_match.group(2).strip()
        try:
            value = json.loads(value)
        except (ValueError, json.JSONDecodeError):
            pass
        params[key] = value
    return func_name, params


_JWT_PATTERN = r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"


def _jwt_app(token: str) -> Optional[str]:
    """The app of an AppWorld access token: its JWT subject is "<app>+<username>"."""
    import base64
    try:
        payload = token.split(".")[1]
        subject = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))).get("sub")
    except Exception:
        return None
    return subject.split("+", 1)[0] if isinstance(subject, str) else None


def repair_token_args(tool_name: str, tool_args: dict, history: list) -> tuple[dict, Optional[dict]]:
    """--repair-access-token (run.py, default off; baseline and --mode sa alike).

    The 27B Actor copies the ~150-character JWT access token back character by character and sometimes changes one
    character of the base64 username (kat_simp -> Kat_simp / tat_simp, kevin.blac -> Kevin.blac), or writes a token
    that was never issued (a file_system_access_token pieced together from another app's token). The signature then
    fails, AppWorld answers "not authorized", the model logs in again, miscopies the new token the same way, and loops
    to the step cap (test_scen1 parv4/parv4off: 6 of 194 tasks looped in at least one arm). AppWorld's own agent
    keeps the token in a Python variable and never retypes it; this switch gives the tool-call agent the same.

    Token parameters of an execute_api call: `access_token` (a token of the called app) and `<app>_access_token`
    (e.g. file_system_access_token of gmail.send_email: a token of <app>). A value that is not, verbatim, a token from
    an earlier tool result in the history is replaced by the most recent such token of that app (JWT subject
    "<app>+..."). A verbatim token (right or wrong app), a missing parameter, and an app without an earlier token are
    left unchanged. The result depends only on (tool call, history), so in SA the Actor's action and the Speculator's
    guesses for a step (same history) are repaired identically and exact matching still works.

    Returns (args to execute and record, {parameter: original value} of the replaced ones, or None)."""
    import re
    if tool_name != "execute_api" or not isinstance(tool_args, dict):
        return tool_args, None
    params = tool_args.get("parameters")
    if not isinstance(params, dict):
        return tool_args, None
    targets = {}
    for key, value in params.items():
        if not isinstance(value, str) or not value:
            continue
        if key == "access_token":
            targets[key] = tool_args.get("app_name")
        elif key.endswith("_access_token"):
            targets[key] = key[:-len("_access_token")]
    if not targets:
        return tool_args, None
    issued = [found for message in history if message.get("role") == "tool" and isinstance(message.get("content"), str)
              for found in re.findall(_JWT_PATTERN, message["content"])]
    repaired, originals = dict(params), {}
    for key, app in targets.items():
        if params[key] in issued:
            continue
        same_app = [found for found in issued if _jwt_app(found) == app]
        if same_app:
            repaired[key] = same_app[-1]
            originals[key] = params[key]
    if not originals:
        return tool_args, None
    return {**tool_args, "parameters": repaired}, originals


def execute_tool(env, tool_name: str, tool_args: dict) -> str:
    """Execute a tool call on the AppWorld environment.

    Returns the string result to send back to the LLM.
    """
    try:
        if tool_name == "show_app_descriptions":
            result = env.execute("import json; print(json.dumps(apis.api_docs.show_app_descriptions(), indent=2))")
        elif tool_name == "show_api_descriptions":
            app = tool_args.get("app_name", "")
            result = env.execute(f'import json; print(json.dumps(apis.api_docs.show_api_descriptions(app_name="{app}"), indent=2))')
        elif tool_name == "show_api_doc":
            app = tool_args.get("app_name", "")
            api = tool_args.get("api_name", "")
            result = env.execute(f'import json; print(json.dumps(apis.api_docs.show_api_doc(app_name="{app}", api_name="{api}"), indent=2))')
        elif tool_name == "search_api_docs":
            query = tool_args.get("query", "")
            result = env.execute(f'import json; print(json.dumps(apis.api_docs.search_api_docs(query="{query}"), indent=2))')
        elif tool_name == "show_account_passwords":
            result = env.execute("import json; print(json.dumps(apis.supervisor.show_account_passwords(), indent=2))")
        elif tool_name == "show_active_task":
            result = env.execute("import json; print(json.dumps(apis.supervisor.show_active_task(), indent=2))")
        elif tool_name == "show_profile":
            result = env.execute("import json; print(json.dumps(apis.supervisor.show_profile(), indent=2))")
        elif tool_name == "show_addresses":
            result = env.execute("import json; print(json.dumps(apis.supervisor.show_addresses(), indent=2))")
        elif tool_name == "show_payment_cards":
            result = env.execute("import json; print(json.dumps(apis.supervisor.show_payment_cards(), indent=2))")
        elif tool_name == "complete_task":
            # AppWorld's complete_task takes `answer` (and `status`). The release passed `output=`, which AppWorld
            # drops, so every answer was lost. No answer / empty answer -> None (required for non-question tasks).
            kwargs = []
            answer = tool_args.get("answer")
            if isinstance(answer, str):
                answer = answer.strip()
            if answer not in (None, ""):
                kwargs.append(f"answer={answer!r}")
            if tool_args.get("status") in ("success", "fail"):
                kwargs.append(f"status={tool_args['status']!r}")
            result = env.execute(
                f"import json; print(json.dumps(apis.supervisor.complete_task({', '.join(kwargs)}), indent=2))")
        elif tool_name == "execute_api":
            app = tool_args.get("app_name", "")
            api = tool_args.get("api_name", "")
            params = tool_args.get("parameters", {})
            # Build kwargs string
            kwargs_parts = []
            for k, v in params.items():
                if isinstance(v, str):
                    v_escaped = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                    kwargs_parts.append(f'{k}="{v_escaped}"')
                elif isinstance(v, bool):
                    kwargs_parts.append(f'{k}={v}')
                elif v is None:
                    kwargs_parts.append(f'{k}=None')
                else:
                    kwargs_parts.append(f'{k}={json.dumps(v)}')
            kwargs_str = ", ".join(kwargs_parts)
            code = f'import json; print(json.dumps(apis.{app}.{api}({kwargs_str}), indent=2))'
            result = env.execute(code)
        else:
            result = f"Unknown tool: {tool_name}"
    except Exception as e:
        result = f"Error executing {tool_name}: {e}"
    return result


def _build_canonical_message(step, tool_name, tool_args, tool_result):
    """Build canonical (assistant, tool) message pair for committed history.

    Single source of truth for the message shape so baseline and pipeline
    grow `messages` bit-identically — divergence here causes greedy decoding
    to walk different trajectories on subsequent steps.

    sort_keys=True normalizes JSON key order so that semantically equal args
    from different sources (owner 27B vs speculator 4B) produce byte-identical
    strings. Without this, peer spec_ctx (built from s1's key order) can
    diverge from committed messages (built from owner's key order).
    """
    tc_id = f"call_{step}"
    canon_args = json.dumps(tool_args, sort_keys=True)
    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tc_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": canon_args},
        }],
    }
    # The full tool result (the release cut it to 4000 characters: a 20-item page or a long API doc was then cut, the
    # model could not see the whole page or the end of a listing, and at T=0 repeated the identical call). AppWorld's
    # own protocol (and the SkillOpt line) shows the full output.
    tool_msg = {
        "role": "tool",
        "tool_call_id": tc_id,
        "name": tool_name,
        "content": tool_result if isinstance(tool_result, str) else str(tool_result),
    }
    return assistant_msg, tool_msg


def stream_chat(client, model, messages, max_tokens, sampling, extra_body, cancel_event=None, extra_latency=0.0,
                tools=TOOLS):
    """One Actor request, streamed; the only way the baseline and the --mode sa runner call the Actor, so both
    receive and parse its reply the same way. Returns (message, prompt_tokens, completion_tokens), or None when
    cancel_event is set (SA's pre-launched calls: closing the stream makes the server abort the request).

    Errors (server, connection, context too long) propagate to the caller unchanged.
    extra_latency > 0 (--actor-extra-latency): the result is handed back that many seconds after the stream ended,
    simulating a slower Actor; a cancel during the wait returns None at once.
    """
    cancel_event = cancel_event or threading.Event()
    stream = client.chat.completions.create(
        model=model, messages=messages, tools=tools, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True}, extra_body=extra_body, **sampling)
    content_parts = []
    tool_calls = {}
    prompt_tokens = completion_tokens = 0
    try:
        for chunk in stream:
            if cancel_event.is_set():
                return None
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
            for call in delta.tool_calls or []:
                slot = tool_calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                if call.id:
                    slot["id"] = call.id
                if call.function is not None:
                    if call.function.name:
                        slot["name"] += call.function.name
                    if call.function.arguments:
                        slot["arguments"] += call.function.arguments
    finally:
        stream.close()
    if extra_latency > 0 and cancel_event.wait(extra_latency):
        return None
    parsed_calls = [
        SimpleNamespace(id=slot["id"] or f"call_{index}", type="function",
                        function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"]))
        for index, slot in sorted(tool_calls.items())
    ]
    message = SimpleNamespace(role="assistant", content="".join(content_parts) or None,
                              tool_calls=parsed_calls or None)
    return message, prompt_tokens, completion_tokens


def run_agent(
    env,
    task_instruction: str,
    model: str,
    api_base: str,
    thinking: bool = False,
    thinking_budget: int = 4096,
    max_steps: int = 20,
    sampling: Optional[dict] = None,
    logger: Optional[TrajectoryLogger] = None,
    tracer=None,
    max_tokens_override: Optional[int] = None,
    actor_extra_latency: float = 0.0,
    request_timeout: float = 300.0,
    infra_retries: int = 0,
    repair_access_token: bool = False,
    stop_on_context_overflow: bool = False,
) -> dict:
    """Run the baseline agent on one AppWorld task.

    actor_extra_latency: simulated slower Actor, seconds added after every Actor response (0 = off).
    repair_access_token: --repair-access-token, see repair_token_args (False = off, behaviour unchanged).
    stop_on_context_overflow: --stop-on-context-overflow. Off (default): a context-length 400 is an _error step and the
    next step resends the same (still too long) history, so the task burns _error steps up to max_steps. On: that
    _error step is logged once and the task ends there (stop_reason "context_overflow"), then is evaluated as usual;
    nothing was executed after it either way, so the evaluation is the same, only the wasted steps are gone.
    sampling: the Actor's sampling fields (sampling_request); None = greedy, {"temperature": 0.0}.

    Returns dict with reward, n_steps, messages, timing info.
    """
    from .event_trace import NULL_TRACER, messages_fingerprint
    tracer = tracer if tracer is not None else NULL_TRACER
    client = _get_client(api_base, timeout=request_timeout)
    infra_retries_used = 0
    error_steps = 0  # steps that ended in an exception (logged as _error)
    sampling_kwargs, sampling_extra = sampling_request(sampling if sampling is not None else {"temperature": 0.0})
    extra_body = {**_thinking_body(thinking, thinking_budget), **sampling_extra}
    # max_tokens_override: the same output limit as the SA Actor (run.py passes --owner-max-tokens).
    max_tokens = max_tokens_override or ((thinking_budget + 4096) if thinking else 1024)

    messages = initial_messages(task_instruction, env.task.supervisor, env.task.datetime)
    tools = TOOLS

    total_input_tokens = 0
    total_output_tokens = 0
    step_times = []
    task_completed = False
    token_repairs = 0
    context_overflow_step = None  # --stop-on-context-overflow: the step whose request exceeded the context window

    for step in range(1, max_steps + 1):
        t_start = perf_counter()
        schema_error = False
        execution_error = False
        thinking_truncated = False
        tool_name = ""
        tool_args = {}
        tool_result = ""
        token_original = None  # --repair-access-token: the token the model wrote, when it was replaced

        try:
            tracer.emit("owner", "llm_submit",
                        step_idx=step,
                        prefix_len=len(messages),
                        prefix_hash=messages_fingerprint(messages))
            with tracer.span("owner", "llm_call", step_idx=step):
                # --infra-retries: a server/connection failure is retried within the step (it does not use up a
                # step); a model-side error (e.g. 400 context too long) is not retried.
                for attempt in range(infra_retries + 1):
                    try:
                        msg, input_tokens, output_tokens = stream_chat(
                            client, model, messages, max_tokens, sampling_kwargs, extra_body,
                            extra_latency=actor_extra_latency, tools=tools)
                        break
                    except Exception as error:
                        if attempt >= infra_retries or not _is_infra_error(error):
                            raise
                        infra_retries_used += 1
                        print(f"  infra retry {attempt + 1}/{infra_retries} at step {step}: {type(error).__name__}",
                              flush=True)
            tracer.emit("owner", "llm_end", step_idx=step)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            # Parse tool call
            tool_name, tool_args = _parse_tool_call(msg)

            # Fallback: parse Qwen XML format from content
            if not tool_name and msg.content:
                tool_name, tool_args = _parse_qwen_tool_call(msg.content)
                if tool_name:
                    # Check if thinking consumed all tokens
                    if not tool_name and "<think>" in (msg.content or ""):
                        thinking_truncated = True

            if not tool_name:
                # No tool call — model may be responding with text
                content = msg.content or ""
                if not content.strip():
                    thinking_truncated = True
                    schema_error = True
                    tool_name = "_no_tool_call"
                    tool_result = "Error: No tool call produced. Please call a tool."
                else:
                    # Model produced text without tool call — treat as respond
                    tool_name = "_text_response"
                    tool_result = content
                    # Add as assistant message and continue
                    messages.append({"role": "assistant", "content": content})
                    t_end = perf_counter()
                    if logger:
                        logger.log_step(
                            step_id=step,
                            instruction=task_instruction[:500],
                            model_output=content,
                            api_name=tool_name,
                            api_params={},
                            api_result=tool_result,
                            wall_clock_time=t_end - t_start,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            schema_error=schema_error,
                            thinking_truncated=thinking_truncated,
                        )
                    continue

            if repair_access_token:
                tool_args, token_original = repair_token_args(tool_name, tool_args, messages)
                token_repairs += token_original is not None

            # Execute tool
            with tracer.span("env", "env_exec", who="main", tool=tool_name):
                tool_result = execute_tool(env, tool_name, tool_args)
            if "Error" in tool_result and "Traceback" in tool_result:
                execution_error = True
            tracer.emit("main", "commit", step_idx=step, tool=tool_name)

            assistant_msg, tool_response_msg = _build_canonical_message(
                step, tool_name, tool_args, tool_result)
            messages.append(assistant_msg)
            messages.append(tool_response_msg)

            # Check if task completed
            if tool_name == "complete_task":
                task_completed = True

        except Exception as e:
            tool_name = "_error"
            error_steps += 1
            tool_result = f"Exception: {traceback.format_exc()}"
            if stop_on_context_overflow and is_context_overflow(e):
                context_overflow_step = step
            execution_error = True
            input_tokens = 0
            output_tokens = 0

        t_end = perf_counter()
        step_time = t_end - t_start
        step_times.append(step_time)

        if logger:
            logger.log_step(
                step_id=step,
                instruction=task_instruction[:500],
                model_output=json.dumps({"tool": tool_name, "args": tool_args, **(
                    {"token_repaired_from": token_original} if token_original is not None else {})}),
                api_name=tool_name,
                api_params=tool_args,
                api_result=tool_result,
                wall_clock_time=step_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                schema_error=schema_error,
                execution_error=execution_error,
                thinking_truncated=thinking_truncated,
            )

        print(f"  Step {step}: {tool_name}({json.dumps(tool_args)[:80]}) -> {tool_result[:100]}... [{step_time:.2f}s]")

        if task_completed:
            break
        if context_overflow_step is not None:
            print(f"  Step {step}: context window exceeded, task stopped (--stop-on-context-overflow)", flush=True)
            break

    # Evaluate using AppWorld's TestTracker
    try:
        eval_result = env.evaluate(suppress_errors=True)
        reward = 1.0 if eval_result.success else 0.0
        pass_rate = eval_result.pass_percentage / 100.0 if eval_result.total_count > 0 else 0.0
        print(f"  Eval: success={eval_result.success} "
              f"passed={eval_result.pass_count}/{eval_result.total_count} "
              f"completed={eval_result.task_completed}")
        # Use pass_rate as the score for partial credit
        reward = pass_rate
    except Exception as e:
        print(f"  Evaluation error: {e}")
        reward = 0.0

    if logger:
        logger.finalize(task_success=(reward > 0), evaluation_score=reward)

    return {
        "reward": reward,
        "n_steps": len(step_times),
        "task_completed": task_completed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_step_time": sum(step_times) / len(step_times) if step_times else 0,
        "total_wall_time": sum(step_times),
        "error_steps": error_steps,
        "infra_retries_used": infra_retries_used,
        **({"token_repairs": token_repairs} if repair_access_token else {}),
        **({"stop_reason": "context_overflow" if context_overflow_step is not None else
            ("completed" if task_completed else "max_steps"), "context_overflow_step": context_overflow_step}
           if stop_on_context_overflow else {}),
        "messages": messages,
    }
