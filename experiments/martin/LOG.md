# Martin's data-loader run log

Plain-language explanations and flag definitions:
[data-loader experiment guide](../../docs/martin_dataloader_flags.md).

## Runs

The row below references an existing team baseline, not a new submission.
No optimization or HEALPix level-6 experiment has run yet. Record the resolution
in every row's notes; compare optimized runs against the same-resolution baseline.

| date | branch / commit | flags on | overlay | run_id | job_id | state | median s/step | stalls | peak GPU GiB | peak host RSS GiB | loss start→end | plots | verdict / notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-20 | Existing team baseline; commit unverified | No new loader flags | `dphpc_memory_baseline_100steps.yml` | `nyue0k4y` | `2418878` | COMPLETED | 2.905 | 11 intervals >5 s; includes 3 tail intervals of 37–38 s | 29.31 | 90.8 reported by Slurm; node total unverified | 0.720→0.509 | Not generated | Level 5. Mean after first 10 steps: 4.641 s. Source: `../2026-10-20_nyue0k4y/`. |

## Work completed on 2026-11-01

- Inspected sampler, readers, tokenization and training-loader setup.
- Recomputed timing and loss summaries from the saved baseline metrics.
- Identified avoidable target encoding, sorting and array copies; savings unmeasured.
- Created the flag/explanation guide and planned separate level-5/level-6 comparisons.
- No training code changed, jobs submitted, or node-hours consumed by this review.

## Next steps

Implement default-off profiling, measure a baseline, then test isolated flags.
Check level-6 feasibility before committing to full comparisons. Append a row
immediately after each submission with its run/job IDs, then update results and
link plots when available. Record failures and the reason for the final verdict.
