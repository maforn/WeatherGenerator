#!/usr/bin/env bash
# Fetch one finished latent-width run from JUPITER into experiments/martin/latent/<run>/.
#   scripts/fetch_latent_run.sh <name> <run_id> <train_job_id>
# <name> is the overlay suffix (base2048, global4096, both4096). Fetches the training metrics,
# logs and model config, the memory sampler CSV and submit log of the chain
# ($R/latent_chain), Slurm accounting (sacct, MaxRSS per step), and the parameter count of the
# last saved checkpoint (count_params.py, run on the login node). Needs an open `ssh jupiter`.
set -euo pipefail
name=$1; run=$2; job=$3
R=/e/scratch/weatherai/maranonpaliz1
S=/e/scratch/weatherai/shared_work
C=$R/latent_chain
OUT="$(cd "$(dirname "$0")/.." && pwd)/experiments/martin/latent/$run"
mkdir -p "$OUT"
rsync -a "jupiter:$S/results/$run/${run}_train_metrics.json" "jupiter:$S/models/$run/model_$run.json" "$OUT/"
rsync -a --include "log.txt" --include "output.*.txt" --include "weathergen.part1.*" \
  --include "cleanup_*" --include "error.txt" --exclude "*" "jupiter:$S/logs/$run/" "$OUT/"
rsync -a "jupiter:$C/$name.submit.log" "$OUT/" || true
rsync -a "jupiter:$C/$name.$job.mem.csv" "jupiter:$C/$name.$job.sampler.log" "$OUT/" || true
ssh -o BatchMode=yes jupiter "sacct -j $job -P --format=JobID,JobName%30,State,Start,End,Elapsed,NNodes,NTasks,MaxRSS,MaxRSSTask,AveRSS,MaxVMSize,NodeList" > "$OUT/sacct.txt"
ssh -o BatchMode=yes jupiter "cd $C && { type module >/dev/null 2>&1 || . \"\$LMOD_ROOT/lmod/init/bash\"; } \
  && module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3 2>/dev/null; \
  f=\$(ls $S/models/$run/${run}_chkpt*.chkpt | sort | tail -1); echo \"checkpoint: \$f\" >&2; \
  $R/wg-dev-check/.venv/bin/python count_params.py \"\$f\"; rm -f core.*" > "$OUT/params.json"
printf '{"name": "%s", "run_id": "%s", "train_job": "%s"}\n' "$name" "$run" "$job" > "$OUT/run.json"
ls -la "$OUT"
