#!/bin/bash
# "N task pairs in flight" run: N worker slots over the task list (xargs -P N). Each slot takes one task and runs its
# baseline and SA arms in parallel (run_pair.sh); the slot frees when both have finished, so at most 2N agents run.
# Endpoints as in run_serial.sh: baseline on D only, SA on A + B + C.
#
# Usage: scripts/run_pairs.sh MODE N      MODE = on | off (Actor thinking), N = pairs in flight (2, 4, ...)
# Run tag test_scen1_stress<N>d<MODE>; writes <OUT_DIR>/<tag>.{log,jsonl,load.jsonl}, <tag>.inflight.d and the two run
# directories. Restartable (a task with both results is skipped). After 5 tasks with a missing result no new task
# starts. Exit 0 only if every task has both results.
MODE=${1:?usage: run_pairs.sh on|off N}
N=${2:?usage: run_pairs.sh on|off N}
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/common.sh"
cd "$REPO" || exit 1
TAG=test_scen1_stress${N}d$MODE
LOG=$OUT_DIR/$TAG.log; LOAD=$OUT_DIR/$TAG.load.jsonl
INFLIGHT_DIR=$OUT_DIR/$TAG.inflight.d
run_ids "$TAG" "$MODE"
preflight
mkdir -p "$INFLIGHT_DIR"
start_sampler
log "start: $(wc -l < "$TASKS") tasks, $N in flight"
awk '{print NR, $1}' "$TASKS" | xargs -P "$N" -L 1 bash "$HERE/run_pair.sh" "$MODE" "$N"
kill $SAMPLER
finish
