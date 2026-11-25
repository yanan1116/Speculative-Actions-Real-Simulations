#!/bin/bash
# Shared settings of the SA-vs-baseline AppWorld drivers (run_serial.sh, run_pairs.sh, run_pair.sh). Sourced, not run.
#
# Required environment (the run stops at once if one is missing):
#   APPWORLD_ROOT       AppWorld root with data 0.1.0 (appworld download data --root ...)
#   TASKS               task list, one id per line (scripts/make_test_scen1_tasks.py)
#   SMC_API_KEY         API key of the vLLM endpoints
#   BASELINE_URL        D: baseline Actor, used by the baseline arm only        (Qwen3.8-27B-FP8)
#   SA_ACTOR_URL        A: SA Actor, normal calls                                (same model and max_model_len as D)
#   SA_PRELAUNCH_URL    B: SA pre-launched Actor calls                           (same model and max_model_len as D)
#   SA_SPECULATOR_URL   C: SA Speculator                                         (Qwen3.5-4B)
# Optional:
#   OUT_DIR             run directory (default: appworld/runs)
#   SPEC_BREADTH        Speculator samples per step (default 5)
#   SPECULATOR_MODEL    model served at SA_SPECULATOR_URL (default Qwen/Qwen3.5-4B)
#   SPEC_NAME           Speculator label in the SA run id (default spec4b)
# URLs end in /v1, e.g. http://host:port/v1.

: "${APPWORLD_ROOT:?set APPWORLD_ROOT}"
: "${TASKS:?set TASKS (task list file)}"
: "${SMC_API_KEY:?set SMC_API_KEY}"
: "${BASELINE_URL:?set BASELINE_URL (endpoint D)}"
: "${SA_ACTOR_URL:?set SA_ACTOR_URL (endpoint A)}"
: "${SA_PRELAUNCH_URL:?set SA_PRELAUNCH_URL (endpoint B)}"
: "${SA_SPECULATOR_URL:?set SA_SPECULATOR_URL (endpoint C)}"
[ -f "$TASKS" ] || { echo "TASKS file not found: $TASKS" >&2; exit 2; }
[ -d "$APPWORLD_ROOT/data" ] || { echo "no data/ under APPWORLD_ROOT: $APPWORLD_ROOT" >&2; exit 2; }

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR=${OUT_DIR:-$REPO/runs}
SPEC_BREADTH=${SPEC_BREADTH:-5}
SPECULATOR_MODEL=${SPECULATOR_MODEL:-Qwen/Qwen3.5-4B}
SPEC_NAME=${SPEC_NAME:-spec4b}
PYTHON=$REPO/.venv/bin/python
mkdir -p "$OUT_DIR"
# Absolute paths, exported, so the per-task processes (xargs, after cd to the repo) see the same settings.
TASKS=$(realpath "$TASKS"); APPWORLD_ROOT=$(realpath "$APPWORLD_ROOT"); OUT_DIR=$(realpath "$OUT_DIR")
export APPWORLD_ROOT TASKS SMC_API_KEY BASELINE_URL SA_ACTOR_URL SA_PRELAUNCH_URL SA_SPECULATOR_URL OUT_DIR SPEC_BREADTH \
  SPECULATOR_MODEL SPEC_NAME

# Actor thinking mode -> ACTOR array. on: the configs' defaults (T 1.0, top_p 0.95, top_k 20, presence 0.0).
actor_args() {
  case $1 in
    on)  ACTOR=() ;;
    off) ACTOR=(--actor-thinking off --actor-sampling
                '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}') ;;
    *)   echo "MODE must be on or off, got '$1'" >&2; exit 2 ;;
  esac
}

# Run ids for TAG and MODE. run.py writes <OUT_DIR>/<run-id>_<mode>, so the result directories are
# BASE_DIR = <TAG>_<MODE>_baseline and SA_DIR = <TAG>_sa_k<n>_<SPEC_NAME>_sa.
run_ids() {  # TAG MODE
  BASE_RUN=$1_$2
  SA_RUN=$1_sa_k${SPEC_BREADTH}_${SPEC_NAME}
  BASE_DIR=$OUT_DIR/${BASE_RUN}_baseline
  SA_DIR=$OUT_DIR/${SA_RUN}_sa
}

# Arguments of both arms (needs ACTOR and the run ids). Both arms: --repair-access-token, --stop-on-context-overflow.
arm_args() {
  local common=(--split test_normal_scen1+test_challenge_scen1 --appworld-root "$APPWORLD_ROOT" --out-dir "$OUT_DIR"
                --repair-access-token --stop-on-context-overflow)
  BASE_ARGS=(--config configs/baseline.json --run-id "$BASE_RUN" --actor-url "$BASELINE_URL"
             "${ACTOR[@]}" "${common[@]}")
  SA_ARGS=(--config configs/sa.json --run-id "$SA_RUN"
           --spec-breadth "$SPEC_BREADTH" --actor-url "$SA_ACTOR_URL" --prelaunch-url "$SA_PRELAUNCH_URL"
           --speculator-url "$SA_SPECULATOR_URL" --speculator-model "$SPECULATOR_MODEL"
           --preexec-verified-apis sa_appworld/runtime/preexec_verified_apis_v4.json "${ACTOR[@]}" "${common[@]}")
}

log() { echo "$(date '+%F %T %Z') $*" >> "$LOG"; }

# Sum of one vLLM Prometheus metric over its label sets (0 if the endpoint does not answer).
metric() {  # URL(/v1) NAME
  curl -s -m 5 "${1%/v1}/metrics" | awk -v m="^vllm:$2\\\\{" '$0 ~ m {s += $NF} END {print s + 0}'
}

# Preflight: A, B and D must serve the same model with the same max_model_len (fail fast otherwise).
preflight() {
  sig() {
    curl -s -m 10 -H "Authorization: Bearer $SMC_API_KEY" "$1/models" |
      python3 -c 'import sys, json; d = json.load(sys.stdin)["data"][0]; print(d["id"], d["max_model_len"])' 2>/dev/null
  }
  local a b d
  a=$(sig "$SA_ACTOR_URL"); b=$(sig "$SA_PRELAUNCH_URL"); d=$(sig "$BASELINE_URL")
  if [ -z "$a" ] || [ "$a" != "$b" ] || [ "$a" != "$d" ]; then
    log "PREFLIGHT FAILED: A=[$a] B=[$b] D=[$d]"; echo "preflight failed, see $LOG" >&2; exit 2
  fi
  log "preflight ok: A/B/D = $a"
}

# Load sampler: every 30 s one JSON line with running / waiting / KV usage / preemptions of A, B, C, D
# (plus the number of tasks in flight when INFLIGHT_DIR is set). Started in the background; sets SAMPLER.
start_sampler() {
  ( while true; do
      local row="{\"t\": \"$(date '+%F %T')\""
      [ -n "${INFLIGHT_DIR-}" ] && row="$row, \"tasks_in_flight\": $(ls "$INFLIGHT_DIR" | wc -l)"
      local e u
      for e in A="$SA_ACTOR_URL" B="$SA_PRELAUNCH_URL" C="$SA_SPECULATOR_URL" D="$BASELINE_URL"; do
        u=${e#*=}
        row="$row, \"${e%%=*}\": {\"running\": $(metric "$u" num_requests_running), \"waiting\": $(metric "$u" num_requests_waiting), \"kv\": $(metric "$u" kv_cache_usage_perc), \"preemptions\": $(metric "$u" num_preemptions_total)}"
      done
      echo "$row}" >> "$LOAD"
      sleep 30
    done ) &
  SAMPLER=$!
}

# Exit 0 only if every task has a result file for each arm in ARMS (default: both).
finish() {
  local n=0 total t
  total=$(wc -l < "$TASKS")
  for t in $(cat "$TASKS"); do
    case " ${ARMS:-baseline sa} " in *" baseline "*) [ -f "$BASE_DIR/results/$t.json" ] || continue ;; esac
    case " ${ARMS:-baseline sa} " in *" sa "*) [ -f "$SA_DIR/results/$t.json" ] || continue ;; esac
    n=$((n + 1))
  done
  if [ "$n" -eq "$total" ]; then log "all done ($n pairs)"; touch "$OUT_DIR/$TAG.done"; exit 0; fi
  log "finished with $n/$total complete pairs"; exit 1
}
