#!/usr/bin/env bash
# Runs inside a training job (srun --overlap) and appends one line every 10 s to $1:
# node memory from `free -b` (used excludes page cache and /dev/shm; shared is mostly /dev/shm),
# RSS of the training ranks and of their loader workers (process tree of this user's python
# processes; RSS counts shared pages once per process), and nvidia-smi memory/utilisation per GPU.
out="$1"
[[ -s "$out" ]] || echo "unix_s,mem_total_b,mem_used_b,mem_shared_b,mem_buffcache_b,mem_available_b,rank_rss_kb,worker_rss_kb,n_ranks,n_workers,gpu_mem_used_mib,gpu_util_pct" > "$out"
while true; do
  t=$(date +%s)
  m=$(free -b | awk '/^Mem:/{print $2","$3","$5","$6","$7}')
  p=$(ps -u "$USER" -o pid=,ppid=,rss=,comm= | awk '
      {pid[NR]=$1; pp[NR]=$2; rss[NR]=$3; c[NR]=$4; comm[$1]=$4}
      END {r=0; w=0; nr=0; nw=0
        for (i in pid) { if (c[i] !~ /^(python|pt_)/) continue
          if (comm[pp[i]] ~ /^(python|pt_)/) {w+=rss[i]; nw++} else {r+=rss[i]; nr++} }
        printf "%d,%d,%d,%d", r, w, nr, nw}')
  g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd';' -)
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | paste -sd';' -)
  echo "$t,$m,$p,$g,$u" >> "$out"
  sleep 10
done
