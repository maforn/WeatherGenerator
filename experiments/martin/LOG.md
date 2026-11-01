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
| 2026-11-01 | `martin/backup-dump` `919dd6bc` | None (profiling only: `profile_timing`, `profile_memory_timeline`) | `exp_profile_baseline.yml` | `jnsctk9d` | `2634852` | PENDING | | | | | | | Level 5. Profiled baseline, same seed/steps as `nyue0k4y`; also measures profiling overhead. Submitted under lock `/e/scratch/weatherai/maranonpaliz1/.submit.lock.d`. |
| 2026-11-01 | `martin/backup-dump` `41bbd8a3` | `even_workset_split` (+ profiling) | `exp_even_workset_split.yml` | `azmuvlzx` | `2634926` | PENDING (afterok:2634852) | | | | | | | Level 5. Compare against `jnsctk9d`; expect fewer/shorter tail stalls. Loss: similar and decreasing (sampled dates change). Its cleanup job 2634927 was cancelled to stay within 3 jobs, so `slurm_weathergen_azmuvlzx_dir/.venv` stays on scratch. |
| 2026-11-01 | `martin/backup-dump` `532cb7f6` | All 8 read/process flags (+ profiling); `even_workset_split` off | `exp_readproc_all.yml` | `dy8638jm` | `2635004` | PENDING (afterok:2634852) | | | | | | | Level 5. Screen: loss must be almost identical to `jnsctk9d`. If faster, bisect per flag; if loss differs, bisect to find the culprit. Queued with Martin's approval above the 3-job limit (3 train + 2 cleanup jobs). |

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
