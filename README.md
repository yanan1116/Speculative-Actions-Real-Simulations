# Speculative Actions on AppWorld

This repository evaluates **Speculative Actions (SA)** in AppWorld, a stateful simulator. We run the same tasks with a sequential baseline and with SA to measure task latency and success. SA spends extra inference on guesses so the Actor can start its next step early. The simulator lets us check whether discarded guesses leave side effects. [Speculative Actions: A Lossless Framework for Faster Agentic Systems(arxiv: 2510.04371)](https://arxiv.org/abs/2510.04371) · [AppWorld](https://github.com/StonyBrookNLP/appworld)

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

The allowlist covers all 457 AppWorld APIs except `supervisor.complete_task` (456 APIs), which always runs live. Model requests run concurrently. AppWorld calls run one at a time on the main thread. [API allowlist](appworld/sa_appworld/runtime/preexec_verified_apis_v4.json)

1. **Keep a clean copy.** The runner holds an in-memory SQLite copy of each app database. After a live call, it refreshes the copies of databases whose change logs grew.
2. **Run and roll back the guess.** The runner snapshots model write counters, the interaction count, I/O and request logs, token blacklists, and Python's global RNG. It disables on-disk saves during the call. A `finally` block identifies writes from the change logs, restores those databases from their clean copies, truncates their change logs, and restores the remaining state. A restore failure aborts the task before evaluation.
3. **Check the live observation.** The runner executes the Actor's actual call on the restored environment. It reuses a pre-launched response only when the complete call and the live output match the guessed call and output. The replay audit checks tool outputs, saved databases and logs, and the task reward.

[Isolation implementation](appworld/sa_appworld/runtime/env_isolation.py) · [Replay audit](appworld/scripts/audit_sa_isolation.py)

## Benchmark selection

### AppWorld

AppWorld simulates nine everyday apps, such as Amazon, Spotify, Venmo, and a file system, behind 457 APIs backed by per-app databases. The agent solves a task through a sequence of API calls, and unit tests check the final database state. It suits SA for three reasons:

- **Tasks are long.** With thinking on, the baseline Actor takes a median of 27 steps per test task (90th percentile 52), so per-step latency adds up and there are many steps to speculate on.
- **Calls have side effects.** Writes such as payments and messages change the databases, so a guessed call must run in isolation and be rolled back when the Actor does not choose it. Hits are measured on live execution rather than on replayed logs, and the replay audit checks that discarded guesses leave no trace.
- **Success depends on state.** The unit tests show whether SA changes task outcomes, not only latency.

The test set has two splits: `test_normal` (168 tasks) and `test_challenge` (417 tasks, each needing at least one app absent from the training split). Tasks come in scenarios of three variants that share an instruction template and differ in details, with ids `<scenario>_<n>`. To cut cost about threefold while covering every scenario, we draw one task per scenario with a fixed seed, `random.Random(f"20261010-{split}")`. That gives 56 + 139 = 195 tasks. We exclude `0de03ad_2` because its context exceeds the Actor's 128K window in every arm, which leaves 194. [scripts/make_test_scen1_tasks.py](appworld/scripts/make_test_scen1_tasks.py) writes this list, and `--full` writes all 585 test tasks.

## Results

The four runs below predate the refactor. Post-refactor checks were offline. Each run used the same 194 test tasks, with up to 200 steps per task. Baseline used one Qwen3.8-27B-FP8 GPU. SA used two of those Actor GPUs and a Qwen3.5-4B Speculator on a third GPU. All GPUs were RTX 6000 Ada.

| Actor thinking | In flight | Baseline success | SA success | Baseline avg score | SA avg score | Replay passed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| On | 1 task per arm | 82.5% (160/194) | 88.1% (171/194) | 0.955 | 0.964 | 194/194 |
| Off | 1 task per arm | 78.9% (153/194) | 76.8% (149/194) | 0.922 | 0.928 | 194/194 |
| Off | 2 task pairs | 78.4% (152/194) | 76.3% (148/194) | 0.923 | 0.914 | 194/194 |
| Off | 4 task pairs | 77.3% (150/194) | 75.3% (146/194) | 0.927 | 0.922 | 193/194 |

| Actor thinking | In flight | Median ratio | Geomean ratio | Total ratio | SA faster |
| --- | --- | ---: | ---: | ---: | ---: |
| On | 1 task per arm | 0.874 | 0.879 | 0.750 | 71.1% (138/194) |
| Off | 1 task per arm | 0.891 | 0.908 | 0.901 | 76.8% (149/194) |
| Off | 2 task pairs | 0.888 | 0.908 | 0.926 | 73.2% (142/194) |
| Off | 4 task pairs | 0.940 | 1.107 | 1.310 | 59.8% (116/194) |

The sole replay mismatch affected two tool outputs and the I/O log because of a login-token expiry difference. Saved database changes and reward matched. An extra RNG draw reproduced the original outputs, and the source of that draw remains unconfirmed. [Latency distributions](docs/appworld_sa_latency_distribution_grid.png)

### Metrics

- **In flight.** With "1 task per arm", each arm runs one task at a time, and the two arms run side by side on their own GPUs. With "N task pairs", N task pairs run at once, and a slot takes the next pair only after both arms finish the current one.
- **Success** is the share of tasks whose unit tests all pass. **Avg score** is the mean, over tasks, of the fraction of unit tests passed.
- **Task time.** Baseline sums its timed steps. SA times its full loop, including in-loop logging. Both exclude environment setup and evaluation.
- **Replay passed** counts SA tasks whose replay audit matched tool outputs, saved databases, logs, and reward.

The latency metrics compare the two arms on the same task, with r = SA time / baseline time per task. Values below 1 favor SA.

| Metric | How it is computed | Sensitive to |
| --- | --- | --- |
| Median ratio | median of the 194 per-task ratios | the typical task only |
| Geomean ratio | geometric mean of the per-task ratios | every task, with equal weight |
| Total ratio | summed SA time / summed baseline time | long tasks, which dominate |
| SA faster | share of tasks with r < 1 | direction only, not size |

- **Median ratio** shows how much faster SA is on a typical task and ignores extreme values.
- **Geomean ratio** is exp(mean(log r)) = (r1 × r2 × … × rn)^(1/n). Ratios need the geometric mean because they are not symmetric. A task twice as fast has r = 0.5 and a task twice as slow has r = 2.0. Their arithmetic mean is 1.25, as if SA were 25% slower. Their geometric mean is √(0.5 × 2.0) = 1.0, so the two cancel out, as they should. Every task has the same weight whether it ran for 30 seconds or an hour, and a few extreme ratios move it only to a limited extent.
- **Total ratio** weights each task by its running time. A few long tasks, such as those that hit the 200-step cap, can decide its value.

### Analysis

Each SA step ends on one path. On a **hit**, the Actor's call matched a guess whose pre-launched Actor request was running on B, and the next step reuses that request (**post_hit**). On a **miss**, no guess matched. On **no_spec**, the Actor returned before the Speculator's guesses were ready. The waits below are the median time a step waited for its Actor response.

| Actor thinking | In flight | hit | post_hit | miss | no_spec | Wait on miss | Wait on post_hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| On | 1 task per arm | 35.1% (2173/6192) | 35.0% (2170/6192) | 27.3% (1689/6192) | 1.3% (80/6192) | 8.70 s | 2.63 s |
| Off | 1 task per arm | 36.7% (2150/5864) | 36.7% (2150/5864) | 16.0% (940/5864) | 10.4% (607/5864) | 3.25 s | 2.71 s |
| Off | 2 task pairs | 36.6% (2190/5980) | 36.6% (2186/5980) | 18.1% (1084/5980) | 8.4% (502/5980) | 4.66 s | 3.03 s |
| Off | 4 task pairs | 37.0% (2147/5798) | 37.0% (2147/5798) | 17.8% (1032/5798) | 7.8% (451/5798) | 4.63 s | 4.85 s |

**Thinking on gains the most.** With thinking on, the Actor spends longer on each step: a miss step waits a median 8.70 s, against 3.25 s with thinking off. That leaves the Speculator and the pre-launched request on B time to finish. Only 1.3% of steps had no guess ready, against 10.4% with thinking off, and a post-hit step waited 2.63 s instead of 8.70 s. The total ratio (0.750) is lower than the median (0.874), so long tasks gained more than short ones. On `test_challenge` tasks the total ratio is 0.666.

**The gain shrinks as more tasks run at once.** The hit rate stays at 35–37% in every run, so the guesses do not get worse. What changes is the cost of using them:

- **B becomes the bottleneck.** After each isolated guess, the runner sends a full-context Actor request to B, up to five per step. With four task pairs, B serves the pre-launched requests of four SA tasks at once. Its KV cache reached 97% usage and up to 21 requests waited in its queue (sampled every 30 s). With one task per arm, B peaked at 52% and 5 waiting requests.
- **Hits stop saving time.** When B is busy, the reused request is often still running when the Actor's call is confirmed. With four task pairs, a post-hit step waited a median 4.85 s, longer than the 4.63 s of a miss step.
- **Both arms slow down, SA more.** From one task per arm to four task pairs, the median time per step rose from 4.0 s to 6.0 s for baseline (+50%) and from 3.7 s to 6.1 s for SA (+63%).
- **Challenge tasks suffer most.** They take more steps with longer contexts, so each pre-launched request costs more. With four task pairs, `test_normal` tasks still favor SA (median 0.832, total 0.823, SA faster on 49/56), while `test_challenge` tasks do not (median 1.020, total 1.521, SA faster on 67/138).
- **A few long tasks decide the total.** For example, `b6d1f70_2` rose from 177 s to 1,926 s. Without the ten tasks where SA added the most time, the total ratio falls from 1.310 to 1.038. The median (0.940) shows that most tasks were still faster, but the geomean (1.107) shows that the slowdowns were larger than the speedups.

**Success rates.** SA does not change what the Actor generates. A reused response was computed from the same history the Actor would have seen, with the same call and the same tool output. Both arms sample independently, so the differences in success (−2.1 to +5.7 points) are expected to come from sampling, not from SA.

## Repository layout

Each environment is self-contained. All setup, config, and commands for an environment are run from that environment's directory.

```text
appworld/                AppWorld environment
  configs/               baseline.json and sa.json
  sa_appworld/run.py     runs one task with --mode baseline or --mode sa
  sa_appworld/runtime/   Actor, SA runner, isolation, API allowlist
  scripts/               task list, run drivers, summary, replay audit
docs/                    figures
```

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

For the single-GPU experiment, all roles share one GPU: the Actor (A and B) and the Speculator (C) run as two vLLM servers, each capped with NVIDIA MPS. `scripts/run_sequential.sh MODE` runs one task at a time. Before each arm, it waits until every endpoint reports no running or waiting request, so arms never overlap. With `ARMS=sa`, it runs only the SA arm and reuses an earlier baseline run.

Summarize and replay-audit a finished run by substituting its tag and thinking mode. The audit reads the SA run's raw trajectories and the corresponding files in `$APPWORLD_ROOT/experiments/outputs/`.

```bash
TAG=test_scen1_serial1doff
.venv/bin/python scripts/summarize_sa_runs.py --out-dir "$OUT_DIR" --tag "$TAG" --mode off \
    --appworld-root "$APPWORLD_ROOT" --tasks "$TASKS"
.venv/bin/python scripts/audit_sa_isolation.py --run-dir "$OUT_DIR/${TAG}_sa_k5_spec4b_sa" \
    --appworld-root "$APPWORLD_ROOT"
```

Licensed under Apache 2.0. [License](LICENSE) · [Upstream attribution: Speculative Macro Commit for Faster Tool-Using Agents (arxiv: 2609.03236)](https://github.com/zeyuliu1037/speculative-macro-commit)
