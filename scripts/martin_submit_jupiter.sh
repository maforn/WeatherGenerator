#!/usr/bin/env bash
# Serialized JUPITER submission: lock -> sync -> budget check -> queue check -> make train -> unlock.
# Usage: scripts/martin_submit_jupiter.sh <config> [TIME] [extra launcher/sbatch args...]
#   e.g. ... config/exp_x.yml 00:30:00 --dependency=afterok:<jobid>
# <config> is make's CONFIG: comma-separated overlays on BASE.
# Environment:
#   SRC_DIR=<local checkout>   default $LOCAL/WeatherGenerator (a worktree for other branches)
#   REMOTE_DIR=<dir under $R>  default WeatherGenerator; give each branch its own directory,
#                              since rsync without --delete would mix two branches' files
#   TARGET=train|scalene-data  default train; scalene-data runs `make scalene-data` (one srun
#                              job, no cleanup job) in the background on the login node
#   MAKEVARS="K=V ..."         extra make variables, e.g. "OPTIONS=healpix_level=6
#                              SCALENE_DIR=profiling/scalene/l6 BATCHES=3"
# Needs an open SSH session (ETH VPN, ssh-add, `ssh jupiter exit` for MFA). Run from anywhere.
set -euo pipefail
CONFIG="$1"; TIME="${2:-00:30:00}"; shift $(( $# >= 2 ? 2 : $# )); EXTRA="$*"
# Each submission adds a training job plus a cleanup job. launch-slurm.py ignores
# `--cleanup-scripts no` for plain train submissions (observed 2026-11-01), so always count 2.
TARGET="${TARGET:-train}"
MAKEVARS="${MAKEVARS:-}"
case "$TARGET" in
  train) NEW_JOBS=2 ;;
  scalene-data) NEW_JOBS=1 ;;
  *) echo "ABORT: unknown TARGET '$TARGET'"; exit 2 ;;
esac
R=/e/scratch/weatherai/maranonpaliz1
LOCK="$R/.submit.lock.d"
LOCAL=/Users/martinmaranon2/DPHPC
SRC_DIR="${SRC_DIR:-$LOCAL/WeatherGenerator}"
REMOTE_DIR="${REMOTE_DIR:-WeatherGenerator}"
W="$R/$REMOTE_DIR"
SSH="ssh -o BatchMode=yes jupiter"

# mkdir is atomic: exactly one submitter can create the lock directory.
if ! $SSH "mkdir '$LOCK' 2>/dev/null && echo \"$(hostname) pid=$$ $(date -u +%FT%TZ) $CONFIG\" > '$LOCK/owner'"; then
  echo "LOCK HELD by: $($SSH "cat '$LOCK/owner' 2>/dev/null")"; exit 3
fi
trap '$SSH "rm -f \"$LOCK/owner\"; rmdir \"$LOCK\"" && echo "lock released"' EXIT
echo "lock acquired"

# Same exclusions as .minion.json, plus local caches and experiments/ (results are pulled, not
# pushed; launch-slurm.py would copy them into every job dir). No --delete: never remove remote files.
# /profiling and /experiments are anchored to the repo root: an unanchored "profiling" also skipped
# config/profiling/annotations.json, which nsys_profiling.sh needs (nsys exits with a usage error).
rsync -az --checksum --exclude .git --exclude .venv --exclude /profiling --exclude .pytest_cache \
  --exclude .ruff_cache --exclude __pycache__ --exclude .DS_Store --exclude /experiments \
  "$SRC_DIR/" "jupiter:$W/"
echo "synced $SRC_DIR -> $W"
# make train/scalene need this directory's own .venv (editable install of its src/)
if ! $SSH "test -x '$W/.venv/bin/python'"; then
  echo "ABORT: no $W/.venv; run 'make init' there first (builds an environment, ask Martin)"; exit 6
fi

CHECK=$($SSH "cd $W && make check 2>&1")
echo "$CHECK"
BUDGET=$(echo "$CHECK" | sed -n 's/^BUDGET_ACCOUNTS=//p' | tail -1)
BASHRC=$($SSH "bash -lc 'grep -h \"^export BUDGET_ACCOUNTS=\" ~/.bashrc | tail -1'" | sed 's/.*=//')
if [[ "$BUDGET" != "e-ext-2025e01-128" || "$BASHRC" != "e-ext-2025e01-128" ]]; then
  echo "ABORT: BUDGET_ACCOUNTS make='$BUDGET' bashrc='$BASHRC'"; exit 4
fi
echo "budget ok"

NJOBS=$($SSH "squeue --me -h | wc -l")
# AGENTS.md limit is 3 queued/running jobs; raise MAX_JOBS only with Martin's explicit approval.
MAX_JOBS="${MAX_JOBS:-3}"
if (( NJOBS + NEW_JOBS > MAX_JOBS )); then
  echo "ABORT: $NJOBS jobs queued/running + $NEW_JOBS new would exceed $MAX_JOBS"; exit 5
fi
echo "queue ok ($NJOBS jobs)"

if [[ "$TARGET" == train ]]; then
  $SSH "cd $W && make train CONFIG=$CONFIG NODES=1 TIME=$TIME ARGS=\"$EXTRA\" $MAKEVARS 2>&1"
else
  LOG="profiling/scalene/$(date -u +%Y%m%dT%H%M%SZ).log"
  # The ( ... & ) subshell exits at once and every descriptor is redirected, so ssh returns
  # (a bare `cd && ... &` backgrounds the whole chain, which keeps the channel open and the
  # lock held until the job ends; seen 2026-11-06 with job 2718460).
  $SSH "cd $W && mkdir -p profiling/scalene && (nohup make scalene-data CONFIG=$CONFIG \
    $MAKEVARS > $LOG 2>&1 < /dev/null &) > /dev/null 2>&1 < /dev/null"
  echo "scalene-data started in the background; log: $W/$LOG"
fi
$SSH "squeue --me"
