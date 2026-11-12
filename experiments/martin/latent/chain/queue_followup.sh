#!/usr/bin/env bash
# Latent-width follow-up runs (Martin's OK on 2026-11-12 for the four next steps: 1000-step runs,
# half LR at 4096, profiling, second seed). Submits RUNS in order from the remote dir wg-dev-check,
# at most MAX_TRAIN training jobs queued/running at once (each submission also queues a cleanup
# job), each under the shared lock with the budget check, and starts watch_job.sh (node/GPU
# memory sampler) for each run except the profiling ones.
# Stops after two failed submissions or two runs that did not complete.
# Restart-safe: runs with a <name>.jobid file are not resubmitted.
set -u
R=/e/scratch/weatherai/maranonpaliz1
W=$R/wg-dev-check
D=$R/latent_chain
LOCK=$R/.submit.lock.d
MAX_TRAIN=3
# name=time=overlays after config/optimizations.yml (comma-separated, without config/ and .yml)
RUNS="long2048=01:30:00=exp_latent_1000steps,exp_latent_base2048
longg4096=02:00:00=exp_latent_1000steps,exp_latent_global4096
longg4096lr=02:00:00=exp_latent_1000steps,exp_latent_global4096,exp_latent_lrhalf
profg4096=00:30:00=exp_latent_torchprof,exp_latent_global4096
prof2048=00:30:00=exp_latent_torchprof,exp_latent_base2048
seed2g4096=00:45:00=exp_latent_seed2,exp_latent_global4096
seed2b2048=00:30:00=exp_latent_seed2,exp_latent_base2048"
NAMES=$(echo "$RUNS" | cut -d= -f1)

log() { echo "$(date +%FT%T) $*" >> "$D/status.txt"; }
ntrain() { squeue --me -h -o %j | grep -c '_train$'; }
release() { rm -f "$LOCK/owner"; rmdir "$LOCK" 2>/dev/null; }

failed_runs() {  # runs of this batch that ended in a state other than COMPLETED
  local n=0 job run st
  for nm in $NAMES; do
    [[ -s "$D/$nm.jobid" ]] || continue
    read -r job run < "$D/$nm.jobid"
    [[ -n "$(squeue -h -j "$job" 2>/dev/null)" ]] && continue
    st=$(sacct -X -n -P -j "$job" -o State | head -1)
    [[ -n "$st" && "$st" != COMPLETED* ]] && n=$((n + 1))
  done
  echo "$n"
}

submit() {  # $1 name, $2 time, $3 overlays; 0 ok, 1 retry later, 2 failed
  local name=$1 time=$2 ovl=$3 sublog=$D/$1.submit.log args b1 b2 job run rc
  until mkdir "$LOCK" 2>/dev/null; do log "lock held: $(cat "$LOCK/owner" 2>/dev/null)"; sleep 120; done
  echo "$(hostname) queue_followup pid=$$ $(date -u +%FT%TZ) $name" > "$LOCK/owner"
  cd "$W" || { release; log "ABORT no $W"; exit 6; }
  b1=$(make check 2>&1 | sed -n 's/^BUDGET_ACCOUNTS=//p' | tail -1)
  b2=$(grep -h '^export BUDGET_ACCOUNTS=' ~/.bashrc | tail -1 | sed 's/.*=//')
  if [[ "$b1" != e-ext-2025e01-128 || "$b2" != e-ext-2025e01-128 ]]; then
    release; log "ABORT BUDGET_ACCOUNTS make='$b1' bashrc='$b2'"; exit 4
  fi
  if (( $(ntrain) >= MAX_TRAIN )); then release; return 1; fi
  args="config/optimizations.yml"
  for o in ${ovl//,/ }; do
    [[ -s config/$o.yml ]] || { release; log "ABORT missing config/$o.yml"; exit 5; }
    args="$args config/$o.yml"
  done
  log "submit $name: $args, time $time (budget ok, $(ntrain) training jobs queued)"
  make train CONFIG=config/dphpc_memory_baseline_100steps.yml NODES=1 TIME="$time" \
    ARGS="$args" > "$sublog" 2>&1 < /dev/null
  rc=$?
  release
  job=$(grep -oP 'Slurm job launched with id: \K[0-9]+' "$sublog" | head -1)
  run=$(grep -oP 'Run id: \K[a-z0-9]+' "$sublog" | head -1)
  if (( rc != 0 )) || [[ -z "$job" ]]; then log "submit $name FAILED rc=$rc (see $sublog)"; return 2; fi
  echo "$job $run" > "$D/$name.jobid"
  log "submitted $name: run $run job $job"
}

sub_failures=0
mapfile -t LINES <<< "$RUNS"
for line in "${LINES[@]}"; do
  IFS='=' read -r name time ovl <<< "$line"
  if [[ -s "$D/$name.jobid" ]]; then log "$name already submitted: $(cat "$D/$name.jobid")"; continue; fi
  while :; do
    nf=$(failed_runs)
    if (( nf >= 2 )); then log "STOP: $nf runs of this batch did not complete"; exit 8; fi
    while (( $(ntrain) >= MAX_TRAIN )); do sleep 120; done
    submit "$name" "$time" "$ovl"; rc=$?
    (( rc == 0 )) && break
    (( rc == 1 )) && { sleep 60; continue; }
    sub_failures=$((sub_failures + 1))
    if (( sub_failures >= 2 )); then log "STOP: two failed submissions"; exit 7; fi
    sleep 300
  done
  if [[ $name == prof* ]]; then
    read -r job run < "$D/$name.jobid"
    ( while [[ -n "$(squeue -h -j "$job" 2>/dev/null)" ]]; do sleep 60; done
      log "$name finished: run $run job $job $(sacct -X -n -P -j "$job" -o State,Elapsed | head -1)" ) \
      > /dev/null 2>&1 < /dev/null &
  else
    nohup bash "$D/watch_job.sh" "$name" > "$D/watch_$name.out" 2>&1 < /dev/null &
  fi
done
log "queue_followup: all submitted"
wait
log "queue_followup: all finished"
