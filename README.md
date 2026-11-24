# Speculative Actions on AppWorld

This repository evaluates **Speculative Actions (SA)** in AppWorld, a stateful simulator. We run the same tasks with a sequential baseline and with SA to measure task latency and success. SA spends extra inference on guesses so the Actor can start its next step early. The simulator lets us check whether discarded guesses leave side effects. [SA paper](https://arxiv.org/abs/2510.04371) · [AppWorld](https://github.com/StonyBrookNLP/appworld)

## How SA works

At step `t`, the Actor and Speculator receive the same committed history `H_t`. The runner deduplicates the Speculator's samples and processes eligible guesses one at a time while the Actor is still generating. Each isolated tool result lets the runner start another Actor request for step `t+1`. That inference can overlap the remaining step-`t` generation and its live tool call.

The example uses eight samples. The experiments use five. Here `g1`–`g3` are allowlisted read-only calls and `g4` is a write call outside the allowlist. Allowlisted write calls can also run in isolation.

```text
Actor A       Actor(H_t) ───────────────────────────────────► a_t
Speculator C  Speculator(H_t)
                └─ 8 samples → deduplicate → g1, g2, g3, g4
                   g1 → isolated call → o1 → Actor_B(H_t + g1 + o1) → F1
                   g2 → isolated call → o2 → Actor_B(H_t + g2 + o2) → F2
                   g3 → isolated call → o3 → Actor_B(H_t + g3 + o3) → F3
                   g4 → defer to live execution if chosen

When a_t arrives
  Find an exact match on tool name and all arguments.
  Request cancellation of unmatched branches (F1 and F3 if a_t == g2).
  Execute a_t on live AppWorld → o_t.

  a_t == g2, F2 exists, and o_t == o2 → keep F2 → F2.result() for step t+1
  Otherwise → cancel any remaining branch → start Actor_A(H_t + a_t + o_t)
                                              if the task continues
```

`F1`, `F2`, and `F3` are Futures for streaming requests submitted to the runner's thread pool. `F2.result()` returns the Actor response immediately if it is ready, or waits for the remaining generation. After a hit, the runner uses that response for step `t+1` and skips speculation on that step. Cancellation closes the client stream when it handles the signal. Server work can continue until it detects the disconnect, and discarded branches still cost the tokens they already generated. [SA runner](appworld/sa_appworld/runtime/sa_breadth_agent.py)

### How tool calls are isolated

The runner uses a 456-API allowlist and reserves `supervisor.complete_task` for live execution. Model requests run concurrently. AppWorld calls run one at a time on the main thread. [API allowlist](appworld/sa_appworld/runtime/preexec_verified_apis_v4.json)

1. **Keep a clean copy.** The runner holds an in-memory SQLite copy of each app database. After a live call, it refreshes the copies of databases whose change logs grew.
2. **Run and roll back the guess.** The runner snapshots model write counters, the interaction count, I/O and request logs, token blacklists, and Python's global RNG. It disables on-disk saves during the call. A `finally` block identifies writes from the change logs, restores those databases from their clean copies, truncates their change logs, and restores the remaining state. A restore failure aborts the task before evaluation.
3. **Check the live observation.** The runner executes the Actor's actual call on the restored environment. It reuses a pre-launched response only when the complete call and the live output match the guessed call and output. The replay audit checks tool outputs, saved databases and logs, and the task reward.

[Isolation implementation](appworld/sa_appworld/runtime/env_isolation.py) · [Replay audit](appworld/scripts/audit_sa_isolation.py)

## Results

The four runs below predate the refactor. Post-refactor checks were offline. Each run used the same 194 test tasks, with up to 200 steps per task. Baseline used one Qwen3.8-27B-FP8 GPU. SA used two of those Actor GPUs and a Qwen3.5-4B Speculator on a third GPU. All GPUs were RTX 6000 Ada.

| Actor thinking | In flight | Baseline success | SA success | Median ratio | Total ratio | Replay passed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| On | 1 task per arm | 82.5% | 88.1% | 0.874 | 0.750 | 194/194 |
| Off | 1 task per arm | 78.9% | 76.8% | 0.891 | 0.901 | 194/194 |
| Off | 2 task pairs | 78.4% | 76.3% | 0.888 | 0.926 | 194/194 |
| Off | 4 task pairs | 77.3% | 75.3% | 0.940 | 1.310 | 193/194 |

Ratios are SA time divided by baseline time. The median uses paired task ratios. The total uses summed task times. Baseline sums timed steps and SA times its full loop, including in-loop logging. Both exclude environment setup and evaluation. Success means all unit tests passed. Both arms sample independently.

At four task pairs, SA still sped up most tasks, but summed task time rose by 31%. The sole replay mismatch affected two tool outputs and the I/O log because of a login-token expiry difference. Saved database changes and reward matched. An extra RNG draw reproduced the original outputs, and the source of that draw remains unconfirmed. [Latency distributions](docs/appworld_sa_latency_distribution_grid.png)

## Run from `appworld/`

Use Linux with Python 3.11, Bash, `curl`, and `bc`.

```bash
cd appworld
python3.11 -m venv .venv
.venv/bin/python -m pip install appworld==0.1.3 openai==1.109.1 httpx==0.27.2
.venv/bin/appworld install
export APPWORLD_ROOT="$PWD"
.venv/bin/appworld download data --root "$APPWORLD_ROOT"  # data 0.1.0
```

For the historical setup, start one vLLM 0.33.0 process per GPU. Use `MODEL=Qwen/Qwen3.8-27B-FP8` with `MAX_LEN=128000` for A, B, and D. Use `MODEL=Qwen/Qwen3.5-4B` with `MAX_LEN=262144` for C. Set `GPU`, `PORT`, and the shared `SMC_API_KEY` for each process. [vLLM serve reference](https://docs.vllm.ai/en/v0.33.0/cli/serve/)

```bash
CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" --port "$PORT" --api-key "$SMC_API_KEY" \
    --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.97 --enable-prefix-caching \
    --trust-remote-code --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

On the client, set the same key and the four endpoint URLs. The task generator selects one task per scenario with a fixed seed and excludes `0de03ad_2` for exceeding the Actor context window.

```bash
export SMC_API_KEY='<shared server key>'
export BASELINE_URL='http://host-d:port/v1'
export SA_ACTOR_URL='http://host-a:port/v1'
export SA_PRELAUNCH_URL='http://host-b:port/v1'
export SA_SPECULATOR_URL='http://host-c:port/v1'
export OUT_DIR="$PWD/runs"
mkdir -p "$OUT_DIR"
export TASKS="$OUT_DIR/test_scen1_tasks_le128k.txt"
.venv/bin/python scripts/make_test_scen1_tasks.py --appworld-root "$APPWORLD_ROOT" > "$TASKS"

scripts/run_serial.sh on    # test_scen1_serial1don
scripts/run_serial.sh off   # test_scen1_serial1doff
scripts/run_pairs.sh off 2  # test_scen1_stress2doff
scripts/run_pairs.sh off 4  # test_scen1_stress4doff
```

The serial driver runs one task per arm independently. The paired driver waits for both arms before reusing a slot. To run all 585 test tasks, generate a second list with `--full`, point `TASKS` to it, and use a separate `OUT_DIR`. On resume, the drivers skip existing result files, including files with `error_type`.

Summarize and replay-audit a finished run by substituting its tag and thinking mode. The audit reads the SA run's raw trajectories and the corresponding files in `$APPWORLD_ROOT/experiments/outputs/`.

```bash
TAG=test_scen1_serial1doff
.venv/bin/python scripts/summarize_sa_runs.py --out-dir "$OUT_DIR" --tag "$TAG" --mode off \
    --appworld-root "$APPWORLD_ROOT" --tasks "$TASKS"
.venv/bin/python scripts/audit_sa_isolation.py --run-dir "$OUT_DIR/${TAG}_sa_k5_spec4b_sa" \
    --appworld-root "$APPWORLD_ROOT"
```

Licensed under Apache 2.0. [License](LICENSE) · [Upstream attribution](https://github.com/zeyuliu1037/speculative-macro-commit)
