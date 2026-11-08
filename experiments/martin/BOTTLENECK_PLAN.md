# Bottleneck study (week of 2026-11-06)

Team task: on `andres/scalene`, go to higher HEALPix levels (and spatial parallelism) and find
what breaks first: CPU, GPU, host memory, or storage → CPU. Then decide whether storage → CPU
needs optimizing. Results feed Alejandro's plots for the 20 Nov meeting.

## Who does what
- **Martin:** normal (data-parallel) runs at levels 5, 6 and 7, scalene and loader profiling. This file.
- **Jonah:** spatial parallelism (`config/spatial_parallel_4.yml`). Jonah checks the config and
  submits those runs.
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
| A16 | train, level 5, 16 workers | `... $TRAIN_CFG,config/exp_bn_workers16.yml 00:30:00` | 30 min | **Skipped:** A was GPU-bound with 3.6× loader headroom (LOG.md `ykm2kcxf`) |
| A24 | train, level 5, 24 workers (conditional) | `... $TRAIN_CFG,config/exp_bn_workers24.yml 00:30:00` | 30 min | **Skipped** (same reason) |
| C | train, level 6, 100 steps | `... $TRAIN_CFG,config/exp_bn_level6.yml 00:45:00` | 45 min | Same at 3× tokenize/build; /dev/shm and node memory |
| D | train, level 7, 10 steps | `... $TRAIN_CFG,config/exp_bn_level7.yml 01:00:00` | ≤60 min | Does it run? Host OOM, /dev/shm, GPU OOM, or just slow? |

- The S runs are one job each, the trains two (with cleanup). Keep ≤3 queued/running.
- Before D, check S7. If one sample takes more than 15 min, drop D to 5 steps. If the peak is
  above about 200 GiB, stop and discuss.
- If A is already GPU-bound (no loader wait), skip A24: more workers cannot help.

## Results so far (2026-11-06)
- **S5:** the read is CPU work (obs reader 12.8% native vs 2.3% system time), not waiting on storage.
- **A (`ykm2kcxf`):** GPU-bound at level 5 (loader wait 1.3%), 4.33 s/step against 2.84–2.88 s
  before Andres's GPU options. The loader has 3.6× headroom with 12 workers.
- **S6/S7:** read is flat across levels; source tokenization grows ×3 (L6) and ×11 (L7). The
  process peak is 1.6 / 12.5 / **72 GiB** at L5/6/7, in source tokenization (`tokenizer_utils.py`
  :365 cat, :332 gather, :485 per-point rotation matrices). 113 s per level-7 sample.
  - **D as planned (2 workers) would need ~575 GiB of host RAM**, above the plan's 200 GiB stop
    line. Run with 1 worker instead (Martin's OK).
- **Added (Martin's OK, 2026-11-06), queued after C:**
  - **NOOPT:** `$B,config/exp_bn_loader.yml,config/exp_bn_profile.yml,config/exp_bn_nogpuopts.yml`,
    30 min. A without `optimizations.yml`. It keeps only the bf16 worker tokens and the per-step
    gc, so the loader payload stays the same. Answers whether the GPU options cause the +50%
    step time.
  - **W4:** `$TRAIN_CFG,config/exp_bn_workers4.yml`, 30 min. A with 4 workers. Answers how much
    host memory 8 fewer workers free, and whether the GPU stays fed.

## Per-level summary (2026-11-07)
Same code and options (A's setup: all loader flags, Andres's GPU options, bf16 worker tokens),
1 node, 4 GH200. Step = mean interval between batches after step 0.

| | L5 `ykm2kcxf` (A) | L6 `pv39uted` (C) | L7 `ywaqq3tw` (D) |
|---|---|---|---|
| Workers per GPU | 12 | 8 | 1 |
| Step | 4.1 s | 15.9 s | **162 s** |
| Loader wait share | 1.4% | 0.3% | 0.0% |
| Prep per sample (one worker) | 13.7 s | 32.4 s | 92.1 s |
| read / tokenize / build | 4.6 / 2.9 / 5.7 | 5.2 / 6.2 / 20.5 | 5.0 / 14.3 / 72.1 |
| Workers needed to keep up | 4 (checked by W4) | ~3 | 1 |
| Payload per sample (bf16) | 1.5 GiB | 4.8 GiB | 17 GiB |
| Rank RSS / workers RSS per rank | 8.6 / 64 GiB | 13.2 / 70 GiB | 30.7 / 47 GiB |
| GPU allocated peak (of 96) | 7.3 GiB | 17.1 GiB | 59.6 GiB |
| Node used peak (CPU+HBM, of 858) | 409 GiB | 486 GiB | 630 GiB |

- **GPU step bound at all three levels.** From L6 to L7 the step grows ×10 and the prep ×2.8,
  so the loader needs fewer workers as the level rises.
- **Read is flat** (~5 s); **build** (batch assembly) is the part that grows: 42% → 63% → 78%.
- **Host memory** sets the worker cap at L7: one worker is 47 GiB, a second per GPU would exceed
  the 477 GiB of CPU RAM. One is also enough.
- **W4 `kdi7cfy9` (L5, 4 workers instead of 12) keeps the GPU fed:** step 3.99 vs 4.19 s, loader
  wait 1.4% in both, node used peak **246 vs 409 GiB** (workers RSS 23–24 vs 64–66 GiB per rank).
  4 workers give 0.28 samples/s/rank for 0.25 used, so 4 is the minimum at L5. Size workers as
  prep per sample ÷ step, rounded up: each extra worker costs ~6 GiB at L5.
- **The flat stretches at L7 are GPU time** (SNAP7 `ft98r6re`, SNAP6 `k8eytats`). The main thread
  makes no GPU allocation for 91% of a L7 step (41–44% at L6): 2 pauses of 12–15 s in forward
  just before the decoders, and 2 of 34–41 s in backward after a checkpoint recompute. Nothing on
  the CPU at those places takes that long, and they grow ~15× for 4× the cells, which fits the
  16 global-attention forecast-engine blocks (`forecast_att_dense_rate: 1.0`, quadratic in the
  number of cells). Plot: `ft98r6re/plots/step_pauses_l6_l7.png`.
- **PROF7 `q4ag4rlz` confirms it at the kernel level** (PyTorch profiler, step 2 on GPU 0, the
  same sample as SNAP7; nsys could not reach a L7 step, see LOG.md NSYS7). Plot:
  `q4ag4rlz/plots/step_gpu_timeline_l7.png`.
  - The GPU is busy 99% of the 160 s step.
  - FlashAttention over the full sphere (196,608 tokens) is 86% of kernel time: 0.69 s per
    forward and 1.84 s per backward call, at ~450 TFLOP/s (~45% of the bf16 peak, normal for
    64-dim heads). The cost is the quadratic amount of work, not a slow kernel.
  - The forecast engine is 82% of the step. SNAP7's forward pauses are its 2 forecast-step
    forwards (2 × 12.6 s), and the backward pauses are its 2 backward stretches (42 and 41 s:
    8 × (rerun 2 blocks + their backward)).
  - Each forecast-engine attention runs forward 3 times per step. The whole-step rerun from
    `fe_checkpoint_per_step` (Andres's option in `optimizations.yml`) is 24 s (15%) of it.
    Turning it off would keep each step's 2-block group inputs from the forward to the
    backward: 3 × 0.8 GB (bf16) + 4 × 1.6 GB (fp32, after the LayerNorm at block 7) ≈ 9 GB per
    forecast step. Estimate: +9–18 GB, so the 60 GiB peak would stay under 96. Not tested:
    model options belong to the teammates.
- **So at L7 the bottleneck is GPU compute in the model**, not storage → CPU: the loader needs
  1 worker, and the training main thread is idle 91% of the step.

## Next steps (proposed 2026-11-08, not submitted; need Martin's OK and Jonah's share of the budget)
Every data-parallel run so far is GPU-bound. These tests look for what breaks next. Costs are
for 1 node.
1. **SP7: spatial parallelism at L7** (Jonah's run, with our loader profiling and the torch
   profiler). Per `docs/{encoder,decoder}_spatial_parallelism.md`, every spatial rank still runs
   the forecast engine and global assimilation on the full sphere. That is 82% of the L7 step,
   so we expect ~140–150 s per sample on 4 GPUs: about 4× less throughput than data-parallel.
   The 4 ranks also read the same sample. Source tokenization (where the 72 GiB peak is) is
   split, so the worker peak may drop. ~0.5–1 node-hour.
2. **READ: storage stress on 1 node.** A new read-only script (Martin's area) with 16/64/128/256
   processes doing only the read stage; record per-sample read time and aggregate GB/s. So far
   only 8–64 concurrent readers have been seen, and the time was flat. One node can only show the
   node's link ceiling, not the file-system aggregate: that needs 2–4 nodes (approval).
   ~0.5 node-hour.
   - **Done (job 2766316, LOG.md READ).** 1/16/64/128/192 readers: 0.19 / 2.8 / 9.9 / **13.6** /
     12.9 samples/s. The read is not storage-bound: off-CPU time stays ~1 s per sample at every
     level and iowait ≤ 0.5%. The ceiling is CPU-side contention: normalize (numpy, no I/O) slows
     ×9 and the read ×2.3 while the node is only 40–60% busy (memory bandwidth/cache). L5 training
     needs ~1 sample/s per node, so the read stage has ~14× headroom. Plot
     `read_stress/plots/read_stress.png`. Next if needed: record user vs system CPU per sample, and
     2–4 nodes for the file-system aggregate.
3. **TOK: cut the tokenizer peak at L7** (Martin's area): rotate per cell (`tokenizer_utils.py`
   :485) and avoid the full-size cat and gather (:365, :332). Check bit-identity with the
   synthetic harness, then rerun S7. Goal: 72 → <35 GiB, so 2 workers per GPU fit at L7.
   ~0.7 node-hour.
4. **S8: loader at L8** (scalene-data, 1 sample, no GPU). Prediction ~4× L7: ~290 GiB peak and
   ~6 min per sample, so 4 workers (1 per GPU) would not fit in 478 GiB, and host memory breaks
   first. L8 training is not worth a run: 16× the attention and ~4× the activations of L7.
   ~0.7 node-hour.
5. **CKPT7: L7 without `fe_checkpoint_per_step`** (Andres's option; measure only, 3-step
   snapshot as SNAP7). Expect −24 s per step and +9–18 GB of GPU peak. ~0.3 node-hour.
6. **W2-L6: L6 with 2 workers per GPU.** Prep 32 s against a 16 s step, so it should be on the
   edge: it confirms "workers = prep ÷ step" from the failing side. ~0.5 node-hour.

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
- With spatial parallelism (Jonah), the issue is 4× duplicated per-rank work, not the storage
  bandwidth itself.
- Rechunking pays off only if reads become bandwidth-bound (multi-node, higher-resolution
  dataset).

## After the runs
- Fetch: rsync `shared_work/results/<run>/dataloader_profile` and the scalene JSON (`profiling/scalene/l*/data.json`).
- Save with `experiments/save.py <run>`, add LOG.md rows, make the memory and timeline plots.
- One summary table per level: step time, loader wait, stage split, host/shm peak, GPU peak.
