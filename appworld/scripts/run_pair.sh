#!/bin/bash
# One task of run_pairs.sh (called through xargs): the task's baseline and SA arms run in parallel; returns when both
# have finished. Appends one JSON line per arm to <OUT_DIR>/<tag>.jsonl (rc, process seconds, tasks in flight and the
# endpoints' running requests at start).
# Usage: run_pair.sh MODE N INDEX TASK
MODE=$1; N=$2; index=$3; task=$4
source "$(dirname "$0")/common.sh"
cd "$REPO" || exit 1
TAG=test_scen1_stress${N}d$MODE
LOG=$OUT_DIR/$TAG.log; PAIRS=$OUT_DIR/$TAG.jsonl; FAILS=$OUT_DIR/$TAG.failures
INFLIGHT_DIR=$OUT_DIR/$TAG.inflight.d
actor_args "$MODE"; run_ids "$TAG" "$MODE"; arm_args
COMMON=(--task-id "$task")

# Safety stop: after 5 tasks with a missing result (endpoint down or similar) no new task starts.
if [ -f "$FAILS" ] && [ "$(wc -l < "$FAILS")" -ge 5 ]; then log "[$index] $task skipped: 5 tasks already missing a result"; exit 1; fi
[ -f "$BASE_DIR/results/$task.json" ] && [ -f "$SA_DIR/results/$task.json" ] && exit 0

load="{\"A\": $(metric "$SA_ACTOR_URL" num_requests_running), \"B\": $(metric "$SA_PRELAUNCH_URL" num_requests_running), \"C\": $(metric "$SA_SPECULATOR_URL" num_requests_running), \"D\": $(metric "$BASELINE_URL" num_requests_running)}"
touch "$INFLIGHT_DIR/$task"
start=$(date +%s.%N)
arm() {  # name, result dir, stdout file, args...
  local name=$1 dir=$2 out=$3; shift 3
  [ -f "$dir/results/$task.json" ] && return
  local s n rc
  s=$(date +%s.%N); n=$(ls "$INFLIGHT_DIR" | wc -l)
  "$PYTHON" -u sa_appworld/run.py "$@" >> "$out" 2>&1; rc=$?
  echo "{\"index\": $index, \"task\": \"$task\", \"arm\": \"$name\", \"rc\": $rc, \"process_seconds\": $(echo "$(date +%s.%N) - $s" | bc), \"tasks_in_flight_at_start\": $n, \"running_before\": $load, \"finished\": \"$(date '+%F %T %Z')\"}" >> "$PAIRS"
}
arm baseline "$BASE_DIR" "$OUT_DIR/${TAG}_baseline.out" "${BASE_ARGS[@]}" "${COMMON[@]}" &
arm sa "$SA_DIR" "$OUT_DIR/${TAG}_sa.out" "${SA_ARGS[@]}" "${COMMON[@]}" &
wait
rm -f "$INFLIGHT_DIR/$task"
if [ -f "$BASE_DIR/results/$task.json" ] && [ -f "$SA_DIR/results/$task.json" ]; then
  log "[$index] $task done ($(echo "$(date +%s.%N) - $start" | bc | cut -d. -f1) s)"
else
  echo "$task" >> "$FAILS"
  log "[$index] $task: missing result (baseline $([ -f "$BASE_DIR/results/$task.json" ] && echo ok || echo none), sa $([ -f "$SA_DIR/results/$task.json" ] && echo ok || echo none))"
  exit 1
fi
