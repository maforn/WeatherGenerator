# Bottleneck study (week of 2026-11-06)

Team task: on `andres/scalene`, go to higher HEALPix levels (and spatial parallelism) and find
what breaks first: CPU, GPU, host memory, or storage → CPU. Then decide whether storage → CPU
needs optimizing. Results feed Alejandro's plots for the 20 Nov meeting.

## Who does what
- **Martin:** normal (data-parallel) runs at levels 5, 6 and 7, scalene and loader profiling. This file.
- **Jonnas:** spatial parallelism (`config/spatial_parallel_4.yml`). He checks the config and
  submits those runs himself.
  - Today every rank of a spatial group reads and normalizes the full sample and tokenizes the
    full-sphere target. Only source tokenization is cut to the rank's cells.
- **Shared:** the weekly budget, about 12 node-hours left until Thu 2026-11-12 16:00. Martin's
  part is about 4.5–5.

## What we already know (no new runs)
- **Read per sample doesn't depend on concurrency or level:**
  - ~7.2–8.7 s (flags off) and ~4.4–4.9 s (flags on), at 8 to 64 concurrent readers, levels 5
    and 6.
  - So storage bandwidth is not saturated on 1 node.
- **Tokenize and build grow with the level:** ×3.3 and ×3.5 from level 5 to 6 (svxaue1j → oq8xiir6).
- **Andres's loader commits:** 41.45 → 10.78 s per batch in one process at level 5, mostly tokenize.
  After them, read becomes a larger share of the remaining time.
- **Host memory at level 6:**
  - /dev/shm ≈ ranks × workers × prefetch × sample size. Samples are 9.2 GiB in fp32.
  - fp32 allows 5 workers at prefetch 1; worker-side bf16 allows 8 (oq8xiir6), which is merged
    in here (see Setup).
  - Level 7 would be ~37 GiB per sample in fp32 (~18 in bf16), so only ~1–2 workers per rank
    fit. That is the likely cause of Yuejiang's CPU OOM.
  - The 7 GiB GPU peak from Andres's options frees no host memory. Some of them use more:
    `source_tokens_on_cpu`, `ema_on_cpu` and `fsdp_cpu_offload` keep tensors in pinned host
    memory.

## Setup (laptop, done)
- **Branch:** `martin/exp/bottleneck` (local only), worktree `wg-martin-bottleneck`. It is
  `andres/scalene` plus two merges:
  - `martin/main`: profiling and loader flags;
  - `alejandro/bf16-worker`: workers yield bf16 source tokens when
    `source_tokens_in_mixed_precision` is on, which `config/optimizations.yml` does. The model
    input is unchanged (njkjo655).
  - Tokenizer output is bit-identical to Andres's base in 5184/5184 synthetic cases.
  - Unit tests: only the 6 seed-test failures that already exist upstream.
- **Overlays:**
  - `config/exp_bn_loader.yml`: all loader flags.
  - `config/exp_bn_profile.yml`: loader profiling with full memory and payload bytes.
  - `config/exp_bn_level6.yml`: 8 workers, prefetch 1.
  - `config/exp_bn_level7.yml`: 2 workers, prefetch 1, 10 steps.
  - `config/exp_bn_workers{16,24}.yml`: level-5 worker sweep.
- **`scripts/martin_submit_jupiter.sh`:**
  - `SRC_DIR`/`REMOTE_DIR` sync this worktree into its own JUPITER directory.
  - `TARGET=scalene-data` runs scalene.

## Setup (JUPITER, needs Martin: VPN, ssh-add, MFA)
1. First submission creates `/e/scratch/weatherai/maranonpaliz1/wg-bottleneck` through the sync.
2. Then run `make init` once there, to build its own `.venv`. The editable install must point at
   this branch's `src/`, so the existing venv cannot be reused. The script aborts until it exists.

Shorthand for the commands below:
```
cd /Users/martinmaranon2/DPHPC/wg-martin-bottleneck
export SRC_DIR=$PWD REMOTE_DIR=wg-bottleneck
BASE_CFG=config/dphpc_memory_baseline_100steps.yml
TRAIN_CFG=$BASE_CFG,config/optimizations.yml,config/exp_bn_loader.yml,config/exp_bn_profile.yml
```

## Runs, in order
| # | What | Command | Wall | Answers |
|---|---|---|---|---|
| S5 | scalene-data, level 5 | `TARGET=scalene-data MAKEVARS="SCALENE_DIR=profiling/scalene/l5" scripts/martin_submit_jupiter.sh $BASE_CFG,config/exp_bn_loader.yml` | ≤40 min | Line-level CPU after Andres's changes; read = I/O wait or CPU (zarr decompression)? |
| S6 | scalene-data, level 6 | same, `+config/exp_bn_level6.yml`, `SCALENE_DIR=profiling/scalene/l6` | ≤40 min | What grows with the level |
| S7 | scalene-data, level 7 | same, `+config/exp_bn_level7.yml`, `SCALENE_DIR=profiling/scalene/l7` | ≤40 min | Seconds and peak memory per level-7 sample; sizes D |
| A | train, level 5, 100 steps | `scripts/martin_submit_jupiter.sh $TRAIN_CFG 00:30:00` | 30 min | New reference: GPU- or loader-bound with Andres's loader and GPU options? |
| A16 | train, level 5, 16 workers | `... $TRAIN_CFG,config/exp_bn_workers16.yml 00:30:00` | 30 min | Do more workers still help with Andres's ~3.8× faster loader? Host-memory cost per worker |
| A24 | train, level 5, 24 workers (conditional) | `... $TRAIN_CFG,config/exp_bn_workers24.yml 00:30:00` | 30 min | Only if A16 still waits on the loader and leaves ≥ ~100 GiB MemAvailable |
| C | train, level 6, 100 steps | `... $TRAIN_CFG,config/exp_bn_level6.yml 00:45:00` | 45 min | Same at 3× tokenize/build; /dev/shm and node memory |
| D | train, level 7, 10 steps | `... $TRAIN_CFG,config/exp_bn_level7.yml 01:00:00` | ≤60 min | Does it run? Host OOM, /dev/shm, GPU OOM, or just slow? |

- The S runs are one job each, the trains two (with cleanup). Keep ≤3 queued/running.
- Before D, check S7. If one sample takes more than 15 min, drop D to 5 steps. If the peak is
  above about 200 GiB, stop and discuss.
- If A is already GPU-bound (no loader wait), skip A24: more workers cannot help.

## How each run is judged
- **GPU-bound:** loader wait per step ≈ 0 (`loader_wait_rank*.jsonl`), and step time = GPU step.
- **CPU-bound (loader):** waits > 0, and sample time is dominated by tokenize/build. More workers
  would help if memory allows.
- **Memory-bound:** workers are capped by /dev/shm or node MemAvailable before the loader keeps up.
  Fixes: bf16 in the workers (Alejandro's PR #3), a smaller payload.
- **Storage-bound:** read time per sample rises with concurrent readers, or scalene puts the read
  in system/I/O wait.
  - Only this case justifies rechunking.
  - Read MB/s comes from the per-sample `payload` bytes against `read_seconds`.

## Expected answer (to confirm)
Storage → CPU is not the limit on 1 node in data-parallel mode. The levels hit host memory
(level 7) and CPU (tokenize/build) first.
- With spatial parallelism (Jonnas), the issue is 4× duplicated per-rank work, not the storage
  bandwidth itself.
- Rechunking pays off only if reads become bandwidth-bound (multi-node, higher-resolution
  dataset).

## After the runs
- Fetch: rsync `shared_work/results/<run>/dataloader_profile` and the scalene JSON (`profiling/scalene/l*/data.json`).
- Save with `experiments/save.py <run>`, add LOG.md rows, make the memory and timeline plots.
- One summary table per level: step time, loader wait, stage split, host/shm peak, GPU peak.
