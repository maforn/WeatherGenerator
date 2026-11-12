#!/usr/bin/env bash
# Runs inside a training job (srun --overlap) and appends one line every 10 s to $1:
# node memory from `free -b` (used excludes page cache and /dev/shm; shared is mostly /dev/shm),
# RSS of the training ranks and of their loader workers (process tree of this user's python
# processes; RSS counts shared pages once per process), and nvidia-smi memory/utilisation per GPU.
# On GH200 nodes `free` also counts the GPUs' HBM (it is onlined as CPU-less NUMA nodes), so the
# last six columns split memory by NUMA node type from /sys/devices/system/node/node*/meminfo:
# nodes with CPUs (LPDDR) and nodes without (HBM); used without cache = total - free - file pages.
out="$1"
[[ -s "$out" ]] || echo "unix_s,mem_total_b,mem_used_b,mem_shared_b,mem_buffcache_b,mem_available_b,rank_rss_kb,worker_rss_kb,n_ranks,n_workers,gpu_mem_used_mib,gpu_util_pct,cpu_numa_total_b,cpu_numa_free_b,cpu_numa_file_b,hbm_numa_total_b,hbm_numa_free_b,hbm_numa_file_b" > "$out"
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
  n=$(for d in /sys/devices/system/node/node[0-9]*; do
        if [[ -n "$(tr -d '\n' < "$d/cpulist")" ]]; then k=c; else k=h; fi
        awk -v k=$k '/MemTotal|MemFree|FilePages/ {print k, $3, $4}' "$d/meminfo"
      done | awk '{s[$1 $2] += $3 * 1024}
        END {printf "%d,%d,%d,%d,%d,%d", s["cMemTotal:"], s["cMemFree:"], s["cFilePages:"],
                                         s["hMemTotal:"], s["hMemFree:"], s["hFilePages:"]}')
  echo "$t,$m,$p,$g,$u,$n" >> "$out"
  sleep 10
done
