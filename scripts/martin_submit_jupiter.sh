#!/usr/bin/env bash
# Serialized JUPITER submission: lock -> sync -> budget check -> queue check -> make train -> unlock.
# Usage: scripts/martin_submit_jupiter.sh <config> [TIME] [extra launcher/sbatch args...]
#   e.g. ... config/exp_x.yml 00:30:00 --dependency=afterok:<jobid>
# Needs an open SSH session (ETH VPN, ssh-add, `ssh jupiter exit` for MFA). Run from anywhere.
set -euo pipefail
CONFIG="$1"; TIME="${2:-00:30:00}"; shift $(( $# >= 2 ? 2 : $# )); EXTRA="$*"
# Each submission adds a training job plus a cleanup job. launch-slurm.py ignores
# `--cleanup-scripts no` for plain train submissions (observed 2026-11-01), so always count 2.
NEW_JOBS=2
R=/e/scratch/weatherai/maranonpaliz1
LOCK="$R/.submit.lock.d"
LOCAL=/Users/martinmaranon2/DPHPC
SSH="ssh -o BatchMode=yes jupiter"

# mkdir is atomic: exactly one submitter can create the lock directory.
if ! $SSH "mkdir '$LOCK' 2>/dev/null && echo \"$(hostname) pid=$$ $(date -u +%FT%TZ) $CONFIG\" > '$LOCK/owner'"; then
  echo "LOCK HELD by: $($SSH "cat '$LOCK/owner' 2>/dev/null")"; exit 3
fi
trap '$SSH "rm -f \"$LOCK/owner\"; rmdir \"$LOCK\"" && echo "lock released"' EXIT
echo "lock acquired"

# Same exclusions as .minion.json, plus local caches. No --delete: never remove remote files.
rsync -az --checksum --exclude .git --exclude .venv --exclude profiling --exclude .pytest_cache \
  --exclude .ruff_cache --exclude __pycache__ --exclude .DS_Store \
  "$LOCAL/WeatherGenerator/" "jupiter:$R/WeatherGenerator/"
echo "synced WeatherGenerator"

CHECK=$($SSH "cd $R/WeatherGenerator && make check 2>&1")
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

$SSH "cd $R/WeatherGenerator && make train CONFIG=$CONFIG NODES=1 TIME=$TIME ARGS=\"$EXTRA\" 2>&1"
$SSH "squeue --me"
