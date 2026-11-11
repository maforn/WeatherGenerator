#!/usr/bin/env bash
# Latent-width study (Martin, 2026-11-11): submits the three L5 runs one after another from
# the remote dir wg-dev-check (dev 4f11a5be + config/exp_latent_*.yml), each under the shared
# submission lock with the budget and queue checks of scripts/martin_submit_jupiter.sh, and
# samples node/GPU memory (sample.sh) while each training job runs.
# Limits (AGENTS.md): 1 node, <= 3 queued/running jobs (each submission = train + cleanup),
# stop after two failures. Restart-safe: runs with a <name>.jobid file are not resubmitted.
set -u
R=/e/scratch/weatherai/maranonpaliz1
W=$R/wg-dev-check
D=$R/latent_chain
LOCK=$R/.submit.lock.d
STATUS=$D/status.txt
RUNS="base2048=00:30:00 global4096=01:00:00 both4096=01:00:00"
log() { echo "$(date +%FT%T) $*" >> "$STATUS"; }
release() { rm -f "$LOCK/owner"; rmdir "$LOCK" 2>/dev/null; }
njobs() { squeue --me -h | wc -l; }
failures=0

submit() {  # $1 name, $2 time; prints nothing, sets JOB and RUN
  local name=$1 time=$2 sublog=$D/$1.submit.log
  while (( $(njobs) > 1 )); do sleep 60; done
  until mkdir "$LOCK" 2>/dev/null; do log "lock held: $(cat "$LOCK/owner" 2>/dev/null)"; sleep 120; done
  echo "$(hostname) latent_chain pid=$$ $(date -u +%FT%TZ) exp_latent_$name" > "$LOCK/owner"
  cd "$W" || { release; log "ABORT no $W"; exit 6; }
  local b1 b2 n
  b1=$(make check 2>&1 | sed -n 's/^BUDGET_ACCOUNTS=//p' | tail -1)
  b2=$(grep -h '^export BUDGET_ACCOUNTS=' ~/.bashrc | tail -1 | sed 's/.*=//')
  if [[ "$b1" != e-ext-2025e01-128 || "$b2" != e-ext-2025e01-128 ]]; then
    release; log "ABORT BUDGET_ACCOUNTS make='$b1' bashrc='$b2'"; exit 4
  fi
  n=$(njobs)
  if (( n + 2 > 3 )); then release; log "queue has $n jobs; waiting"; return 1; fi
  log "submit $name (budget ok, queue $n)"
  make train CONFIG=config/dphpc_memory_baseline_100steps.yml NODES=1 TIME="$time" \
    ARGS="config/optimizations.yml config/exp_latent_$name.yml" > "$sublog" 2>&1
  local rc=$?
  release
  JOB=$(grep -oP 'Slurm job launched with id: \K[0-9]+' "$sublog" | head -1)
  RUN=$(grep -oP 'Run id: \K[a-z0-9]+' "$sublog" | head -1)
  if (( rc != 0 )) || [[ -z "$JOB" ]]; then log "submit $name FAILED rc=$rc (see $sublog)"; return 2; fi
  echo "$JOB $RUN" > "$D/$name.jobid"
  log "submitted $name: run $RUN job $JOB"
}

for item in $RUNS; do
  name=${item%%=*}; time=${item#*=}
  if [[ -s "$D/$name.jobid" ]]; then
    read -r JOB RUN < "$D/$name.jobid"; log "$name already submitted ($RUN, job $JOB)"
  else
    tries=0
    while :; do
      submit "$name" "$time"; rc=$?
      (( rc == 0 )) && break
      (( rc == 1 )) && { sleep 60; continue; }
      tries=$((tries + 1)); failures=$((failures + 1))
      if (( tries >= 2 || failures >= 2 )); then log "STOP: $name failed to submit twice"; exit 7; fi
      sleep 300
    done
  fi
  # wait until the training job runs, then start the memory sampler inside it
  while st=$(squeue -h -j "$JOB" -o %T 2>/dev/null); [[ "$st" == PENDING || "$st" == CONFIGURING ]]; do sleep 30; done
  if [[ "$st" == RUNNING ]]; then
    csv=$D/$name.$JOB.mem.csv
    nohup srun --jobid="$JOB" --overlap -N1 -n1 --gpus-per-node=4 bash "$D/sample.sh" "$csv" \
      > "$D/$name.$JOB.sampler.log" 2>&1 < /dev/null &
    sleep 45
    if ! kill -0 $! 2>/dev/null; then
      log "$name sampler with GPUs exited ($(tail -2 "$D/$name.$JOB.sampler.log" | tr '\n' ' ')); retry without"
      nohup srun --jobid="$JOB" --overlap -N1 -n1 bash "$D/sample.sh" "$csv" \
        >> "$D/$name.$JOB.sampler.log" 2>&1 < /dev/null &
    fi
    log "$name running (job $JOB), sampler -> $csv"
  fi
  while [[ -n "$(squeue -h -j "$JOB" 2>/dev/null)" ]]; do sleep 60; done
  state=$(sacct -X -n -P -j "$JOB" -o State,Elapsed | head -1)
  log "$name finished: run $RUN job $JOB $state"
  if [[ "$state" != COMPLETED* ]]; then
    failures=$((failures + 1))
    if (( failures >= 2 )); then log "STOP: two failed runs"; exit 8; fi
  fi
done
log "chain done"
