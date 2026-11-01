# Martin's data-loader run log

Plain-language explanations and flag definitions:
[data-loader experiment guide](../../docs/martin_dataloader_flags.md).

## Runs

The row below references an existing team baseline, not a new submission.
No optimization or HEALPix level-6 experiment has run yet; the profiled baseline is below. Record the resolution
in every row's notes; compare optimized runs against the same-resolution baseline.

| date | branch / commit | flags on | overlay | run_id | job_id | state | median s/step | stalls | peak GPU GiB | peak host RSS GiB | loss start→end | plots | verdict / notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-20 | Existing team baseline; commit unverified | No new loader flags | `dphpc_memory_baseline_100steps.yml` | `nyue0k4y` | `2418878` | COMPLETED | 2.905 | 11 intervals >5 s; includes 3 tail intervals of 37–38 s | 29.31 | 90.8 reported by Slurm; node total unverified | 0.720→0.509 | Not generated | Level 5. Mean after first 10 steps: 4.641 s. Source: `../2026-10-20_nyue0k4y/`. |
| 2026-11-01 | `martin/backup-dump` `919dd6bc` | None (profiling only) | `exp_profile_baseline.yml` | `jnsctk9d` | `2634852` | COMPLETED (12:51) | 2.854 | 10 intervals >5 s; largest 39.2 s | 29.31 | 390.3 summed RSS (overcounts shared pages); ~6 GiB per worker | 0.7198→0.5104 | [plots](jnsctk9d/plots/) | Level 5. **Profiling overhead: none measurable** vs `nyue0k4y` (2.905 s, 11 stalls, 29.31 GiB, 0.7198→0.5089). Per sample ~40 s per worker: tokenize 18.8, build 12.9, read 7.75, normalize 0.5 s (median, summed over streams); ERA5 build alone 4.0 s. Last worker per rank gets 12 samples vs 8 → ~160 s tail with 25–36 s waits; ~40 s startup wait for first samples. Verdict: tail stall cause confirmed (→ `even_workset_split`); tokenize+build dominate (79%) so tokenizer flags matter more than reader flags. |
| 2026-11-01 | `martin/backup-dump` `41bbd8a3` | `even_workset_split` (+ profiling) | `exp_even_workset_split.yml` | `azmuvlzx` | `2634926` | COMPLETED (10:28) | 2.878 | 9 intervals >5 s; largest **10.3 s** (was 39.2) | 29.33 | 405.3 summed RSS | 0.7198→0.4923 | [plots](azmuvlzx/plots/) | Level 5 vs `jnsctk9d`: **mean step after 10 steps 4.67→3.35 s (−28%)**, job 12:51→10:28. Workers now get 9/8 samples; the ~160 s tail is gone. Loss differs per step (mean abs 0.047; sampled dates change by design) but starts identical and decreases similarly. Verdict: **keep**. Cleanup job 2634927 was cancelled (venv left on scratch). |
| 2026-11-01 | `martin/backup-dump` `532cb7f6` | All 8 read/process flags (+ profiling) | `exp_readproc_all.yml` | `dy8638jm` | `2635004` | COMPLETED (10:54) | 2.792 | **3** intervals >5 s; largest 32.4 s | **32.11** ⚠ (steps 8–11 only; median step 29.236 vs 29.234) | 384.0 summed RSS; worker peak ~5.7 vs 6.2 GiB | 0.7198→0.5107 | [plots](dy8638jm/plots/) | Level 5 vs `jnsctk9d`: per-sample prep **40.0→32.9 s (−18%)**: read 7.75→4.38, tokenize 18.8→17.5, build 12.9→10.7. Mean step after 10 steps 4.67→3.78 s (−19%). Loss within run-to-run noise (mean abs Δ 0.0062 vs 0.0048 between the two flag-free baselines). Cumulative input MB identical at every step. **Unexplained +2.8 GiB GPU allocated in steps 8–11** (reserved 32.5→35.4). Verdict: promising, not kept until bisection explains the GPU bump. |
| 2026-11-01 | `martin/backup-dump` `4da76a8f` | None (profiling only) | `exp_profile_baseline_repeat.yml` | `t5dc8t3i` | `2636101` | COMPLETED (12:22) | 2.890 | 9 intervals >5 s; largest 39.8 s | 29.31 | 390.5 summed RSS | 0.7198→0.5144 | [plots](t5dc8t3i/plots/) | Level 5 repeat of `jnsctk9d`. **Noise floor:** mean step after 10 steps 4.54 vs 4.67 s (±3%); loss mean abs Δ 0.0058; GPU peak identical, no step >29.5 GiB; summed RSS identical. So `dy8638jm`'s loss is within noise and its +2.8 GiB GPU bump is real. |
| 2026-11-01 | `martin/backup-dump` `2f6a9104` | 4 tokenizer flags (+ profiling) | `exp_tokenizer_only.yml` | `em2wu8xf` | `2636357` | COMPLETED (11:13) | 2.897 | 5 intervals >5 s; largest 34.6 s | 29.31 (no bump) | 433.7 summed RSS; worker peak median **9.22 GiB (+6%)** | 0.7198→0.5100 | [plots](em2wu8xf/plots/) | Level 5 vs `jnsctk9d`: prep 40.0→35.5 s (build −2.1, tokenize −0.8; read 6.1 s is filesystem noise), mean step after 10 4.67→4.06 s (−13%). Loss within noise (0.0075). Verdict: faster but **more host RAM**; trade-off, not a free win. |
| 2026-11-01 | `martin/backup-dump` `5d8ba2f9` | 4 reader flags (+ profiling) | `exp_readers_only.yml` | `kawac1qm` | `2636415` | COMPLETED (11:13) | 2.839 | 4 intervals >5 s; largest 36.4 s | 29.31 (no bump) | 398.9 summed RSS; worker peak median **8.46 GiB (−3%)** | 0.7198→0.5056 | [plots](kawac1qm/plots/) | Level 5 vs `jnsctk9d`: prep 40.0→35.5 s (read 7.75→4.16), mean step after 10 4.67→4.07 s (−13%). Loss within noise (0.0050). Verdict: **keep**: faster and less RAM. |
| 2026-11-01 | `martin/backup-dump` `01f7d041` | None (profiling only), `prefetch_factor: 1` | `exp_prefetch1.yml` | `i24jcqoe` | `2636520` | COMPLETED (13:29) | 2.936 | 13 intervals >5 s; largest 39.6 s | 29.31 | node summed RSS peak **363 (−7%)**, mean 214 (−4%); worker peak unchanged 8.66 | 0.7198→0.5198 | [plots](i24jcqoe/plots/) | Level 5 vs `jnsctk9d`: mean step after 10 **4.67→5.01 s (+7%, beyond ±3% noise)**, total wait 622→776 s. The loader is already the bottleneck, so a smaller prefetch buffer starves the GPU more. Verdict: **reject** for now; retry only once the loader outpaces the GPU. |
| 2026-11-01 | `martin/backup-dump` `17a72ec6` | `even_workset_split` + `num_workers: 16` (+ profiling) | `exp_workers16_even.yml` | `c9cfzjqg` | `2637414` | COMPLETED (9:59) | 2.969 | **0** intervals >5 s; largest 4.7 s | 29.31 | node summed RSS peak **504 (+24% vs azmuvlzx 405)**, mean 338 (+23%); worker peak 8.50 | 0.7198→0.484 | [plots](c9cfzjqg/plots/) | Level 5 vs `azmuvlzx`: mean step after 10 **3.35→3.05 s (−9%)** ≈ median, so now GPU-bound; total wait 259→184 s. Loss similar and decreasing (sample-to-worker assignment changes with worker count). Verdict: works but costs +24% host RAM; prefer faster samples (flags) at 12 workers if that is enough. |
| 2026-11-01 | `martin/backup-dump` `07488497` | All 8 read/process flags (+ profiling) | `exp_readproc_all_repeat.yml` | `ppxxfvr2` | `2637453` | COMPLETED (10:49) | 2.792 | 3 intervals >5 s; largest 32.8 s | **29.31 (no bump)** | node summed RSS peak 389; worker peak 8.36 | 0.7198→0.5167 | [plots](ppxxfvr2/plots/) | Repeat of `dy8638jm`: mean step after 10 3.83 s (vs 3.78), prep 33.1 s (vs 32.9), loss within noise (0.0065). **GPU bump did not reproduce** → it was a one-off allocator event, not caused by the flags. Verdict: all 8 flags **keep**. |
| 2026-11-01 | `martin/backup-dump` `8308d927` | None; CUDA allocator snapshot (3 steps) | `exp_gpu_snapshot_baseline.yml` | `d8j13w9v` | `2637787` | COMPLETED (5:12) | | | | | | `d8j13w9v/d8j13w9v_memory_viz.html` (local, 28 MB, gitignored) | Baseline GPU memory_viz picture (rank 0). Snapshot pickle in `d8j13w9v/profiling_traces/` (local) and on JUPITER `shared_work/logs/d8j13w9v/profiling_traces/`. |
| 2026-11-01 | `martin/backup-dump` `d05657b0` | **Combined**: even split + all 8 flags, 12 workers (+ profiling) | `dphpc_dataloader_all.yml` | `svxaue1j` | COMPLETED (**9:20**, was 12:51) | 2.84 | **0** intervals >5 s; largest **3.6 s** | 29.33 | node summed RSS peak **386 (−1% vs 390)**; worker peak 8.32 (−4%) | 0.7198→0.4904 | [plots](svxaue1j/plots/) | Level 5. **Mean step after 10 steps 4.67→2.88 s (−38%)** ≈ median → GPU-bound; total wait 622→162 s; prep 40.0→33.8 s/sample. Matches 16 workers' speed (3.05 s) at baseline host RAM (16 workers: 504 GiB). Loss vs `azmuvlzx` (same sample assignment): mean abs Δ 0.0090, end 0.4904 vs 0.4923; vs baseline 0.046, the even-split effect. Verdict: **adopt**. |
| 2026-11-01 | `martin/backup-dump` `d05657b0` | Combined; CUDA allocator snapshot (3 steps) | `exp_gpu_snapshot_combined.yml` | `annepcty` | `2638484` | COMPLETED (4:26) | | | trace peak 27.47 (baseline 27.41) | | | `annepcty/annepcty_memory_viz.html` (local) | vs `d8j13w9v`: same event count (196,302) and allocation count (67,797); 91.8% of allocations identical in order and size; the rest are big activations differing ~0.1% because the even split serves different samples. Not an exact pair → exact reference `oxkz1yfl` (even split only) queued. |
| 2026-11-01 | `martin/backup-dump` `a04dca75` | `even_workset_split` only; CUDA allocator snapshot (3 steps) | `exp_gpu_snapshot_even.yml` | `oxkz1yfl` | `2638712` | COMPLETED (5:18) | | | trace peak 27.47 | | | `oxkz1yfl/oxkz1yfl_memory_viz.html` (local) | **vs `annepcty` (combined): 67,797/67,797 allocations identical in order and size (100%), same peak.** The 8 read/process flags leave GPU inputs bit-for-bit the same shape; the 8% difference against `d8j13w9v` comes only from the even split serving different samples. |

## Work completed on 2026-11-01

- Inspected sampler, readers, tokenization and training-loader setup.
- Recomputed timing and loss summaries from the saved baseline metrics.
- Identified avoidable target encoding, sorting and array copies; savings unmeasured.
- Created the flag/explanation guide and planned separate level-5/level-6 comparisons.
- No training code changed, jobs submitted, or node-hours consumed by this review.

## Implementation completed on 2026-11-01 (no runs yet)

- Reader and tokenizer flags: commits `6522a925`, `9a0da224`; wired into the
  sampler in `4ad990d4`.
- Profiling (timing, loader wait, memory timeline, analysis script): `ff838054`.
- `prefetch_factor` / `persistent_workers` support: `07ecf231`.
- `even_workset_split`: `77a3496d`.
- Fixed an undefined `TRAIN` name in the sampler that would have crashed target
  building; dropped an undocumented `free_raw_after_tokenize` flag that only
  deleted locals about to be rebound.
- 39 CPU tests pass locally. No jobs submitted, no node-hours used.

## Next steps

Implement default-off profiling, measure a baseline, then test isolated flags.
Check level-6 feasibility before committing to full comparisons. Append a row
immediately after each submission with its run/job IDs, then update results and
link plots when available. Record failures and the reason for the final verdict.
