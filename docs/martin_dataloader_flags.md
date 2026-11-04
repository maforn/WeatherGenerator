# Martin's data-loader optimizations

Faster batch preparation (disk → `yield batch`) with no change to model inputs or learning
behaviour. Everything is opt-in: every optimization is a default-off
`data_loading.optimizations.<flag>` read with `.get(...)`, so existing configurations keep the
original code path. Model, GPU processing, pinning and `to_device` are untouched.

## Results (JUPITER, 1 node, 4 × GH200, seed 1)

| | Before | After | Run IDs |
| --- | --- | --- | --- |
| **Level 5**, 100 steps, 12 workers | 4.67 s/step, 10 stalls (largest 39 s), job 12:51 | **2.88 s/step (−38%)**, 0 stalls, job 9:20 | `jnsctk9d` → `svxaue1j` |
| Level 5 host memory (summed RSS peak) | 390 GiB | 386 GiB (335 GiB with `source_tokens_bf16`) | `va54ajr5` |
| **Level 6**, 8 workers + bf16 | 27.5 s/step (4 workers, flags off; 8+ workers crash: `/dev/shm` full) | **13.9 s/step (−50%)**, ≥ 38 GiB node memory free | `a5z733w1` → `oq8xiir6` |

- Loss stays within same-seed run-to-run noise; CUDA allocator snapshots of the level-5 flags
  match the reference on all 67,797 allocations (same GPU peak).
- After the change, training is GPU-bound at both levels: the loader no longer sets the pace.
- Per-run details, failures and rejected settings (16 workers, `prefetch_factor: 1` at level 5)
  are in [the run log](../experiments/martin/LOG.md). The experiment overlays (`config/exp_*.yml`)
  and raw profiles live on branch `martin/backup-dump`.

## Recommended configurations

Overlays on `config/config_operan_georing_avhrr_forecasting_lowres.yml`:

- `config/dphpc_dataloader_all.yml`: level 5. Even worker split + 4 reader + 4 tokenizer flags,
  12 workers. Adding `source_tokens_bf16: true` is verified too (`va54ajr5`).
- `config/dphpc_dataloader_level6.yml`: level 6. All flags + `source_tokens_bf16`, 8 workers,
  `prefetch_factor: 1`.

Both enable profiling (it costs nothing measurable); drop the `profile_*` keys to disable it.

## Flag reference

Every optimization defaults to `False` under `data_loading.optimizations`.
Read new settings with `.get(...)` defaults so existing configurations retain
their original behaviour.

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

`source_tokens_bf16` is only valid while `mixed_precision_dtype: bf16`, because the model casts
source tokens to that dtype before use. Byte metrics in `utils/performance.py` then report half.

### Loader setting

`data_loading.prefetch_factor` is forwarded to the PyTorch `DataLoader` only when set, so older
configurations keep PyTorch's default (2). Use 1 at level 6, where each sample occupies
4–9 GiB of `/dev/shm`; at level 5 it made steps 7% slower.

### Profiling settings (`data_loading`, all default-off)

| Setting | Output in `<run_dir>/dataloader_profile/` |
| --- | --- |
| `profile_timing: true` | `samples_rank*_worker*.jsonl`: per sample read, normalize, tokenize, build and total seconds, source MB/tokens, worker RSS before/after. `loader_wait_rank*.jsonl`: how long the training loop waited for each batch. |
| `profile_memory_timeline: true` | `memory_rank*.csv`: rank + loader-worker RSS and GPU allocated/reserved, sampled every `profile_memory_interval_seconds` (default 0.5). Also `/dev/shm` used/total bytes per sample row. |
| `profile_payload_bytes: true` (needs `profile_timing`) | Adds `payload` to each sample record: bytes of torch tensors (what a worker moves into `/dev/shm` when handing the sample over) and numpy arrays reachable from the yielded batch, plus the largest attribute groups. Shared views counted once. Added after level-6 runs exhausted `/dev/shm` (`w0a1jb0w`, `es253qrq`). |
| `profile_memory_full: true` (needs `profile_memory_timeline`) | Adds honest host-memory columns to `memory_rank*.csv`: `rank_uss/pss_bytes`, `workers_uss/pss_bytes` (from `/proc/<pid>/smaps_rollup`; PSS splits shared pages between processes, so summing PSS over all ranks does not double-count like summed RSS), and `node_used/total_bytes` (`MemTotal − MemAvailable`, node-wide, includes `/dev/shm`; take the max across ranks, never the sum). Also `cpu_mem_used/total_bytes` and `hbm_mem_used/total_bytes` from `/sys/devices/system/node`: NUMA nodes with CPUs = host LPDDR, memory-only nodes = GH200 GPU HBM; used = MemTotal − MemFree − FilePages + Shmem (file cache excluded, `/dev/shm` included). Slower per sample than RSS (smaps walk), so off by default. |
| `profile_output_dir` | Optional override of the output directory. |

Summarize or compare runs with
`python scripts/analyze_dataloader_profile.py --run <dir> --baseline <dir> --output <plots>`.

## Results page

`docs/martin_dataloader_results.html` is a standalone page for the team (open it in a browser):
headline numbers, interactive "what is in host memory over time" views for level 5 before/after
and level 6, and every run behind the numbers.

## Plots

- `scripts/plot_memory_view.py --run <dir>`: what is in host memory over time, stacked by
  training ranks, loader workers and `/dev/shm` (samples waiting for training), with GPU memory
  and training steps on the same time axis. Uses PSS with `profile_memory_full`, else RSS.
- `scripts/plot_dataloader_memory_stack.py --run <dir> [--baseline <dir>]`: host memory per
  process, CUDA memory and per-batch waits; plus a per-sample stage-time breakdown.
- `scripts/analyze_dataloader_profile.py --run <dir> [--baseline <dir>] --output <dir>`:
  summary numbers and step-time, loss and peak comparisons.

Summed RSS counts shared pages once per process and overstates RAM (about 10% at level 6);
`profile_memory_full` gives PSS and NUMA totals. A 0.5-second sampler can miss brief spikes.

## HEALPix level 6 comparison

HEALPix divides the globe into cells. The code uses `12 * 4**healpix_level`:
level 5 has **12,288 cells**, and level 6 has **49,152 cells**. Each cell therefore
covers one quarter of the area. This gives finer spatial grouping, not four times
as many source observations or a guaranteed improvement in forecast accuracy.
See the [HEALPix cell-count definition](https://healpix.sourceforge.io/html/idl_nside2npix.htm).

Level 6 can change token padding, CPU preparation, model work and GPU memory.
Four times as many cells does not imply exactly four times the memory or time.
Treat this as a separate resolution experiment, retaining level 5 as a reference.

Status (2026-11-04): without bf16, level 6 at 8+ workers exhausts the booster's 239 GiB
`/dev/shm` (9.2 GiB fp32 per sample). Recommended level-6 setup: all flags + `source_tokens_bf16`,
**8 workers**, prefetch 1 (`config/dphpc_dataloader_level6.yml`, run `oq8xiir6`): 13.9 s/step,
about GPU-bound, /dev/shm peak 154 GiB, node memory never below 38 GiB available. 9 workers is
1.7% faster (within noise) but left 0.02 GiB available (`azjgvr5m`). See `experiments/martin/LOG.md`.
On GH200 nodes `node_used` (MemTotal − MemAvailable) counts host LPDDR (478 GiB) plus GPU HBM
(380 GiB); the NUMA columns separate the two.

Compare optimization correctness and GPU peaks **within the same resolution**.
Do not require level 6 to match level 5's loss or 29.31 GiB GPU peak. A short run
can establish feasibility and speed, but accuracy needs a separate, adequately
trained comparison on the same held-out weather dates with the same metrics.
Agree that larger study with Martin and the mentor before spending its budget.
