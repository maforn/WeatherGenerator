#!/usr/bin/env bash
# Wait for one latent-width job to run, start the memory sampler inside it, log when it ends.
#   watch_job.sh <name>   (reads <name>.jobid)
set -u
name=$1
D=/e/scratch/weatherai/maranonpaliz1/latent_chain
log() { echo "$(date +%FT%T) $*" >> "$D/status.txt"; }
read -r JOB RUN < "$D/$name.jobid"
while st=$(squeue -h -j "$JOB" -o %T 2>/dev/null); [[ "$st" == PENDING || "$st" == CONFIGURING ]]; do sleep 30; done
if [[ "$st" == RUNNING ]]; then
  csv=$D/$name.$JOB.mem.csv
  srun --jobid="$JOB" --overlap -N1 -n1 --gpus-per-node=4 bash "$D/sample.sh" "$csv" \
    > "$D/$name.$JOB.sampler.log" 2>&1 < /dev/null &
  pid=$!; sleep 45
  if ! kill -0 $pid 2>/dev/null; then
    log "$name sampler with GPUs exited ($(tail -2 "$D/$name.$JOB.sampler.log" | tr '\n' ' ')); retry without"
    srun --jobid="$JOB" --overlap -N1 -n1 bash "$D/sample.sh" "$csv" \
      >> "$D/$name.$JOB.sampler.log" 2>&1 < /dev/null &
  fi
  log "$name running (job $JOB, node $(squeue -h -j "$JOB" -o %N)), sampler -> $csv"
fi
while [[ -n "$(squeue -h -j "$JOB" 2>/dev/null)" ]]; do sleep 60; done
log "$name finished: run $RUN job $JOB $(sacct -X -n -P -j "$JOB" -o State,Elapsed | head -1)"
