#!/bin/bash
# "Sequential" run: one task at a time over the task list, and for each task the baseline arm first, then the SA arm,
# never concurrently. Meant for a single shared endpoint (all of BASELINE_URL, SA_ACTOR_URL, SA_PRELAUNCH_URL and
# SA_SPECULATOR_URL may point to the same server), so each arm has the server to itself while it runs. Before each arm
# starts, the driver waits until every endpoint reports 0 running and 0 waiting requests (two checks 2 s apart), so
# requests the previous arm left on the server (e.g. cancelled speculative calls still being aborted) never overlap it.
#
# Usage: scripts/run_sequential.sh MODE        MODE = on | off (Actor thinking); environment: see common.sh
# Run tag RUN_TAG (default test_scen1_seq<MODE>); writes <OUT_DIR>/<tag>.{log,jsonl,load.jsonl} and the two run
# directories. Restartable: an arm that already has a result for the task is skipped. Stops after 3 consecutive tasks
# with a missing result. Exit 0 only if both arms have a result for every task.
# ARMS (default "baseline sa") selects the arms to run, e.g. ARMS=sa reuses an earlier baseline run; the completeness
# checks then cover only the selected arms.
MODE=${1:?usage: run_sequential.sh on|off}
ARMS=${ARMS:-baseline sa}
for a in $ARMS; do case $a in baseline|sa) ;; *) echo "ARMS must list baseline and/or sa, got '$ARMS'" >&2; exit 2 ;; esac; done
source "$(dirname "$0")/common.sh"
cd "$REPO" || exit 1
TAG=${RUN_TAG:-test_scen1_seq$MODE}
LOG=$OUT_DIR/$TAG.log; RECORDS=$OUT_DIR/$TAG.jsonl; LOAD=$OUT_DIR/$TAG.load.jsonl
actor_args "$MODE"; run_ids "$TAG" "$MODE"; arm_args
preflight

ENDPOINTS=$(printf '%s\n' "$BASELINE_URL" "$SA_ACTOR_URL" "$SA_PRELAUNCH_URL" "$SA_SPECULATOR_URL" | sort -u)

# 0 if every endpoint answers /metrics with no running and no waiting request; 1 otherwise (busy or not answering).
endpoints_idle() {
  local u m
  for u in $ENDPOINTS; do
    m=$(curl -s -m 5 "${u%/v1}/metrics") || return 1
    [ -n "$m" ] || return 1
    echo "$m" | awk '/^vllm:num_requests_(running|waiting)\{/ {s += $NF} END {exit !(s == 0)}' || return 1
  done
}

# Drain gate: wait until the endpoints are idle twice in a row, 2 s apart. Sets DRAIN_SECONDS.
drain() {
  local start last_log
  start=$(date +%s); last_log=$start
  while :; do
    if endpoints_idle; then sleep 2; endpoints_idle && break; fi
    if [ $(( $(date +%s) - last_log )) -ge 60 ]; then
      log "waiting for idle endpoints ($(( $(date +%s) - start )) s)"; last_log=$(date +%s)
    fi
    sleep 1
  done
  DRAIN_SECONDS=$(( $(date +%s) - start ))
}

arm() {  # index, task, name, result dir, stdout file, args...
  local index=$1 task=$2 name=$3 dir=$4 out=$5; shift 5
  [ -f "$dir/results/$task.json" ] && return 0
  local s rc
  drain
  s=$(date +%s.%N)
  "$PYTHON" -u sa_appworld/run.py "$@" --task-id "$task" >> "$out" 2>&1; rc=$?
  echo "{\"index\": $index, \"task\": \"$task\", \"arm\": \"$name\", \"rc\": $rc, \"drain_seconds\": $DRAIN_SECONDS, \"process_seconds\": $(echo "$(date +%s.%N) - $s" | bc), \"finished\": \"$(date '+%F %T %Z')\"}" >> "$RECORDS"
  [ -f "$dir/results/$task.json" ]
}

has_arm() { case " $ARMS " in *" $1 "*) return 0 ;; esac; return 1; }

start_sampler
log "start: $(wc -l < "$TASKS") tasks, one at a time, arms: $ARMS"
index=0; fails=0
for task in $(cat "$TASKS"); do
  index=$((index + 1))
  ok=1
  if has_arm baseline; then arm "$index" "$task" baseline "$BASE_DIR" "$OUT_DIR/${TAG}_baseline.out" "${BASE_ARGS[@]}" || ok=0; fi
  if has_arm sa; then arm "$index" "$task" sa "$SA_DIR" "$OUT_DIR/${TAG}_sa.out" "${SA_ARGS[@]}" || ok=0; fi
  if [ $ok -eq 1 ]; then
    fails=0; log "[$index] $task done"
  else
    fails=$((fails + 1))
    log "[$index] $task: missing result (baseline $([ -f "$BASE_DIR/results/$task.json" ] && echo ok || echo none), sa $([ -f "$SA_DIR/results/$task.json" ] && echo ok || echo none))"
    if [ $fails -ge 3 ]; then log "stop: 3 consecutive tasks with a missing result"; break; fi
  fi
done
kill $SAMPLER
finish
