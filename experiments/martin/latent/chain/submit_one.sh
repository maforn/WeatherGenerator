#!/usr/bin/env bash
# Submit one latent-width run now (Martin, 2026-11-11 23:1x: "queue all runs at once"; this
# lifts the 3-job limit to 6 for these three runs only). Lock, budget check, make train.
#   submit_one.sh <name> <time>
set -u
name=$1; time=$2
R=/e/scratch/weatherai/maranonpaliz1; W=$R/wg-dev-check; D=$R/latent_chain; LOCK=$R/.submit.lock.d
log() { echo "$(date +%FT%T) $*" >> "$D/status.txt"; }
[[ -s "$D/$name.jobid" ]] && { log "$name already submitted: $(cat "$D/$name.jobid")"; exit 0; }
until mkdir "$LOCK" 2>/dev/null; do sleep 20; done
trap 'rm -f "$LOCK/owner"; rmdir "$LOCK" 2>/dev/null' EXIT
echo "$(hostname) submit_one pid=$$ $(date -u +%FT%TZ) exp_latent_$name" > "$LOCK/owner"
cd "$W" || exit 6
b1=$(make check 2>&1 | sed -n 's/^BUDGET_ACCOUNTS=//p' | tail -1)
b2=$(grep -h '^export BUDGET_ACCOUNTS=' ~/.bashrc | tail -1 | sed 's/.*=//')
if [[ "$b1" != e-ext-2025e01-128 || "$b2" != e-ext-2025e01-128 ]]; then log "ABORT budget '$b1' '$b2'"; exit 4; fi
n=$(squeue --me -h | wc -l)
if (( n + 2 > 6 )); then log "ABORT $name: $n jobs queued, limit 6"; exit 5; fi
log "submit $name (budget ok, queue $n)"
make train CONFIG=config/dphpc_memory_baseline_100steps.yml NODES=1 TIME="$time" \
  ARGS="config/optimizations.yml config/exp_latent_$name.yml" > "$D/$name.submit.log" 2>&1
rc=$?
job=$(grep -oP 'Slurm job launched with id: \K[0-9]+' "$D/$name.submit.log" | head -1)
run=$(grep -oP 'Run id: \K[a-z0-9]+' "$D/$name.submit.log" | head -1)
if (( rc != 0 )) || [[ -z "$job" ]]; then log "submit $name FAILED rc=$rc"; exit 2; fi
echo "$job $run" > "$D/$name.jobid"
log "submitted $name: run $run job $job"
