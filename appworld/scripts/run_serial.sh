#!/bin/bash
# "1 task in flight" run: each arm runs the task list one task at a time, in list order, independently of the other arm
# (no pairing barrier). The arms use disjoint endpoints, so they do not compete:
#   baseline: D (BASELINE_URL) only
#   SA:       A (SA_ACTOR_URL) normal calls + B (SA_PRELAUNCH_URL) pre-launched calls + C (SA_SPECULATOR_URL), n=SPEC_BREADTH
#
# Usage: scripts/run_serial.sh MODE        MODE = on | off (Actor thinking); environment: see common.sh
# Run tag test_scen1_serial1d<MODE>; writes <OUT_DIR>/<tag>.{log,jsonl,load.jsonl} and the two run directories.
# Restartable: a task that already has a result is skipped. An arm stops after 3 consecutive tasks without a result.
# Exit 0 only if both arms have a result for every task.
MODE=${1:?usage: run_serial.sh on|off}
source "$(dirname "$0")/common.sh"
cd "$REPO" || exit 1
TAG=test_scen1_serial1d$MODE
LOG=$OUT_DIR/$TAG.log; RECORDS=$OUT_DIR/$TAG.jsonl; LOAD=$OUT_DIR/$TAG.load.jsonl
actor_args "$MODE"; run_ids "$TAG" "$MODE"; arm_args
preflight

arm_loop() {  # name, result dir, stdout file, args...
  local name=$1 dir=$2 out=$3; shift 3
  local index=0 fails=0 task s rc
  for task in $(cat "$TASKS"); do
    index=$((index + 1))
    [ -f "$dir/results/$task.json" ] && continue
    s=$(date +%s.%N)
    "$PYTHON" -u sa_appworld/run.py "$@" --task-id "$task" >> "$out" 2>&1; rc=$?
    echo "{\"index\": $index, \"task\": \"$task\", \"arm\": \"$name\", \"rc\": $rc, \"process_seconds\": $(echo "$(date +%s.%N) - $s" | bc), \"finished\": \"$(date '+%F %T %Z')\"}" >> "$RECORDS"
    if [ -f "$dir/results/$task.json" ]; then fails=0; else fails=$((fails + 1)); log "[$name $index] $task: no result (rc=$rc)"; fi
    if [ $fails -ge 3 ]; then log "[$name] stop: 3 consecutive tasks without a result"; return 1; fi
  done
  log "[$name] arm done"
}

start_sampler
log "start: $(wc -l < "$TASKS") tasks per arm, each arm one task at a time"
arm_loop baseline "$BASE_DIR" "$OUT_DIR/${TAG}_baseline.out" "${BASE_ARGS[@]}" &
P1=$!
arm_loop sa "$SA_DIR" "$OUT_DIR/${TAG}_sa.out" "${SA_ARGS[@]}" &
P2=$!
wait $P1 $P2
kill $SAMPLER
finish
