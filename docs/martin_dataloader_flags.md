# Martin's data-loader experiments

## Goal and current status

Prepare training batches faster while using less CPU RAM. Keep the model's input
values and learning behaviour intact. A **worker** is a CPU process preparing
one batch while the GPU trains on another. A **token** groups data points for
the model. **Loss** measures prediction error; lower is generally better.

As of 2026-11-01 every flag below is implemented, default-off, CPU-tested and
benchmarked on JUPITER (level 5, 100 steps, seed 1). Flags were measured in
groups (reader / tokenizer); per-flag attribution inside a group is untested.
**Combined result** (`config/dphpc_dataloader_all.yml`, run `svxaue1j`): mean
step 4.67 → 2.88 s (−38%), zero stalls, job 12:51 → 9:20, host RAM unchanged,
loss within noise of the same-sample-assignment reference. See the run log.
Record successful and failed experiments in [the run log](../experiments/martin/LOG.md).

## What we measured so far

Source: `experiments/2026-10-20_nyue0k4y/train_metrics.json` and `summary.json`.
This existing run used HEALPix level 5, 12 workers per rank, seed 1, and 100 steps.

| Measurement | Baseline result | Plain-language meaning |
| --- | --- | --- |
| Median step interval | 2.905 s | A typical training step |
| Mean interval after the first 10 steps | 4.641 s | Waiting makes the overall run slower |
| Final three intervals | 37.82, 37.06, 38.01 s | Most workers have finished; the last worker still has work |
| Peak allocated GPU memory | 29.31 GiB | Saved training GPU peak |
| Training loss, first → last | 0.720 → 0.509 | Learning progressed, with fluctuations |
| Reported host memory | 90.8 GiB, accounting scope unverified | Do not yet treat this as a measured node total |

Step intervals are differences between consecutive training elapsed-time records,
so they include waiting and other work between log entries. The periodic stalls
support a loader bottleneck but do not directly measure individual worker time.
The 90.8 GiB value comes from Slurm `MaxRSS`, which reports the maximum across
tasks; confirm the site's accounting before interpreting it as whole-node usage.
See [Slurm's definition](https://slurm.schedmd.com/sacct.html).

## Flag reference

Every optimization defaults to `False` under `data_loading.optimizations`.
Read new settings with `.get(...)` defaults so existing configurations retain
their original behaviour. Each experiment gets `config/exp_<name>.yml`.

| Flag | What it changes and why it might work | Loss requirement | Measured effect |
| --- | --- | --- | --- |
| `target_requested_fields_only` | Build only the target fields the caller uses; currently the values path builds and discards coordinate encodings. Less calculation and temporary storage. | Almost identical | Part of tokenizer group: prep −11%, worker RSS +6% (`em2wu8xf`) |
| `skip_train_inverse_indices` | Skip sorting indices that training immediately discards. Keep them for evaluation. | Almost identical | Part of tokenizer group (`em2wu8xf`) |
| `source_tokens_bf16` | Cast source tokens to bf16 in the worker, after NaN masking and validity checks. `EmbeddingEngine.forward` already casts them to the bf16 mixed-precision dtype, so the model input is bit-identical while each sample in `/dev/shm` halves (level 6: 9.2 GiB, 98% source tokens, `pi3kavlj`). Only valid while `mixed_precision_dtype: bf16`. Throughput metrics that count source bytes (`utils/performance.py`) will report half. | Same (checked by `exp_level6_bf16_workers5`) | Loss check PASS (`njkjo655` vs `k6i6mdyp`: per-step mean \|Δ\| 0.0001). /dev/shm 185 → 96 GiB at 5 workers; enables 9 workers at level 6 (`k2rsep95`): 22.1 → 13.8 s/step (−37%), ~GPU-bound. |
| `anemoi_avoid_dtype_copy` | Avoid copying an already-float32 array just to request float32 again. Verify ownership before sharing storage. | Almost identical | Part of reader group: read 7.75→4.16 s, worker RSS −3% (`kawac1qm`) |
| `skip_all_valid_filter_copy` | When every row passes a filter, reuse the array instead of copying every row. Keep filtering when any row fails. | Almost identical | Part of reader group (`kawac1qm`) |
| `tokenize_coords_only` | During space-time grouping, copy only coordinates when the next operation uses only coordinates. | Almost identical | Part of tokenizer group (`em2wu8xf`) |
| `even_workset_split` | Share leftover samples across workers: four workers get 9 and eight get 8, instead of one getting 12 and eleven getting 8. Reduces uneven finishing times. | Similar and decreasing; sampled dates change | Mean step −28%, tail stall 39→10 s (`azmuvlzx`) |
| `source_direct_build` | Reduce padding/gathering/split-and-stack copies while preserving token order and padding. | Almost identical | Part of tokenizer group: build −2.1 s/sample (`em2wu8xf`) |
| `obs_single_read` | Read the union of observation columns once. May avoid repeated decompression; an extra combined buffer could instead raise memory. Inspect storage chunks first. | Almost identical | Part of reader group (`kawac1qm`) |
| `anemoi_select_vars` | Select needed variables before materializing data, in both ordinary and Operan readers. Disk savings depend on storage chunks and reader support. | Almost identical | Part of reader group (`kawac1qm`) |

### Profiling settings (`data_loading`, all default-off)

| Setting | Output in `<run_dir>/dataloader_profile/` |
| --- | --- |
| `profile_timing: true` | `samples_rank*_worker*.jsonl`: per sample read, normalize, tokenize, build and total seconds, source MB/tokens, worker RSS before/after. `loader_wait_rank*.jsonl`: how long the training loop waited for each batch. |
| `profile_memory_timeline: true` | `memory_rank*.csv`: rank + loader-worker RSS and GPU allocated/reserved, sampled every `profile_memory_interval_seconds` (default 0.5). Also `/dev/shm` used/total bytes per sample row. |
| `profile_payload_bytes: true` (needs `profile_timing`) | Adds `payload` to each sample record: bytes of torch tensors (what a worker moves into `/dev/shm` when handing the sample over) and numpy arrays reachable from the yielded batch, plus the largest attribute groups. Shared views counted once. Added after level-6 runs exhausted `/dev/shm` (`w0a1jb0w`, `es253qrq`). |
| `profile_memory_full: true` (needs `profile_memory_timeline`) | Adds honest host-memory columns to `memory_rank*.csv`: `rank_uss/pss_bytes`, `workers_uss/pss_bytes` (from `/proc/<pid>/smaps_rollup`; PSS splits shared pages between processes, so summing PSS over all ranks does not double-count like summed RSS), and `node_used/total_bytes` (`MemTotal − MemAvailable`, node-wide, includes `/dev/shm`; take the max across ranks, never the sum). Slower per sample than RSS (smaps walk), so off by default. |
| `profile_output_dir` | Optional override of the output directory. |

Summarize or compare runs with
`python scripts/analyze_dataloader_profile.py --run <dir> --baseline <dir> --output <plots>`.

### Plain loader settings

`num_workers` already existed (baseline 12). `prefetch_factor` and
`persistent_workers` are now forwarded to the PyTorch `DataLoader` only when
set, so old configurations keep PyTorch's defaults (2 and False). Prefetch means
preparing batches ahead of use; less prefetch may save memory but increase
waiting. More workers may improve speed while increasing memory. With
`persistent_workers: true`, the parent's mini-epoch is shared with workers
through a shared-memory counter so reused workers do not repeat the previous
epoch's samples; verify this on a short multi-epoch run before relying on it.

## Prepared overlays and when to run them

All are `config/exp_*.yml` on the same base, seed and level-5 data unless noted. The first
three are queued; the rest are prepared but not submitted.

| Overlay | Purpose | Run when |
| --- | --- | --- |
| `exp_profile_baseline` | Profiled baseline (`jnsctk9d`) | queued |
| `exp_even_workset_split` | Even worker split (`azmuvlzx`) | queued |
| `exp_readproc_all` | All 8 read/process flags (`dy8638jm`) | queued |
| `exp_profile_baseline_repeat` | Run-to-run noise | first free slot |
| `exp_readers_only` / `exp_tokenizer_only` | Bisect `exp_readproc_all` | if it was faster or changed loss |
| `exp_workers16` / `exp_prefetch4` | Less waiting, more RAM | if the training loop waits for batches |
| `exp_prefetch1` | Less RAM, maybe more waiting | if host RAM is the constraint |
| `exp_two_epochs_control` + `exp_persistent_two_epochs` | Persistent workers must reproduce the per-epoch sample ids | before using `persistent_workers` |
| `exp_gpu_snapshot_baseline` | CUDA allocator snapshot (pytorch.org/memory_viz), 3 steps | paired with a snapshot of the combined config |
| `exp_level6_feasibility` | HEALPix level 6, 20 steps | after the combined level-5 config |

## Plots

- `scripts/plot_dataloader_memory_stack.py --run <dir> [--baseline <dir>] --output <dir>` draws two figures:
  - Host memory stacked per process (training ranks in blue, loader workers in orange), with CUDA memory and per-batch waits on the same time axis.
  - A per-sample stage-time breakdown (read, normalize, tokenize, build), compared against the baseline.
- `scripts/analyze_dataloader_profile.py` prints summary numbers and draws the line plots: step time, loss, and peak comparisons.

## How we judge an experiment

1. Measure an instrumented baseline before optimizing. Check profiling overhead.
2. Enable one optimization at a time, keeping seed, streams and workload fixed.
3. Check synthetic outputs for unchanged values, shapes, order and padding.
4. Compare median step time, stalls, total duration, host memory, GPU peaks and loss.
5. Plot host/GPU GiB against elapsed seconds, marking peaks; also plot step time
   and loss per step. Save plots in `experiments/martin/<run_id>/`.
6. Explain the outcome in ordinary language, including unsuccessful changes.
   Combine only verified improvements in `config/dphpc_dataloader_all.yml`.

RSS means memory resident in RAM. Summed process RSS can count shared pages more
than once; label it honestly and, where available, supplement it with node/job
memory and proportional memory measurements. Before/after RSS and a 0.5-second
sampler can miss brief allocation spikes. Do not claim exact transient peaks.

## HEALPix level 6 comparison

HEALPix divides the globe into cells. The code uses `12 * 4**healpix_level`:
level 5 has **12,288 cells**, and level 6 has **49,152 cells**. Each cell therefore
covers one quarter of the area. This gives finer spatial grouping, not four times
as many source observations or a guaranteed improvement in forecast accuracy.
See the [HEALPix cell-count definition](https://healpix.sourceforge.io/html/idl_nside2npix.htm).

Level 6 can change token padding, CPU preparation, model work and GPU memory.
Four times as many cells does not imply exactly four times the memory or time.
Treat this as a separate resolution experiment, retaining level 5 as a reference.

Planned comparisons:

| Resolution | Baseline | Optimized |
| --- | --- | --- |
| Level 5 | Profiling on, optimizations off | Same setup plus verified flags |
| Level 6 | Same seed/streams/steps, `healpix_level: 6`, optimizations off | Same level-6 setup plus verified flags |

First run a short level-6 feasibility check on one node; if it fits, use a
100-step performance comparison. Use the existing baseline overlay plus the
resolution override, and record the complete effective configuration. Profiling
must be implemented first.

Status (2026-11-03): level 6 at 8+ workers exhausts the booster's 239 GiB `/dev/shm`
(9.2 GiB fp32 per sample). Recommended level-6 setup is `exp_level6_bf16_workers9.yml`
(all flags + `source_tokens_bf16`, 9 workers, prefetch 1): 13.8 s/step (`k2rsep95`),
/dev/shm peak 174 GiB, about GPU-bound. See `experiments/martin/LOG.md`.

Compare optimization correctness and GPU peaks **within the same resolution**.
Do not require level 6 to match level 5's loss or 29.31 GiB GPU peak. A short run
can establish feasibility and speed, but accuracy needs a separate, adequately
trained comparison on the same held-out weather dates with the same metrics.
Agree that larger study with Martin and the mentor before spending its budget.

## Explanation to add after every experiment

- **What changed:** code, commit, flag, default and overlay.
- **Why we tried it:** the unnecessary work or waiting we expected to remove.
- **What we measured:** baseline → experiment numbers, run IDs and plot links.
- **Did the data stay correct?** Output checks, loss behaviour and GPU peak.
- **Why it helped or failed:** evidence first; label untested explanations.
- **Decision:** keep, revise, or reject, and what to test next.
