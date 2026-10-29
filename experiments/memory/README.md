# Memory scaling study: tools and runs

Peak GPU memory per training step, per GPU, from CUDA allocator snapshots (3-step runs with
`profiling.memory_profiling`). Snapshots are on JUPITER in
`/e/scratch/weatherai/shared_work/logs/<run>/profiling_traces/*_rank_0.pickle`.

## Tools

| File | What it does | Needs |
|---|---|---|
| `wg_memory_model.py` | Analytical model of peak A (decoder backward) and peak B (embedding recompute) from level, width, forecast steps, spatial parallel size and the memory options. Prints its validation against the runs below and projections. | stdlib |
| `snapdiff.py` | Live memory at the peak of two snapshots, grouped by WeatherGenerator call site, and the difference. `python -I snapdiff.py a.pickle b.pickle [--top N] [--depth K]` | stdlib |
| `memtimeline.py` | Stacked active-memory timeline per model part (like pytorch.org/memory_viz), several runs on one scale. `python -I memtimeline.py out.png title "label=a.pickle" ...` | numpy, matplotlib |

`memtimeline.py` maps frames to model parts by line ranges in `model/engines.py` of this
branch; on other code the ranges need checking. Backward allocations have no Python frame and
are assigned to the last part seen, which is a heuristic.

## Runs

GiB allocated at the peak (reserved in brackets where it matters). Baseline code is this branch.

**Scaling points, baseline code.** Config: `config/dphpc_level6_snapshot.yml` with
`healpix_level` and `training_config: forecast: num_steps` changed as listed (huw6isc4,
jmo7u7d6 and vkyfae3p are Matteo's runs with `config_memory_profiling.yml`, 12 workers).

| Run | Level | Forecast steps | Peak | Where |
|---|---|---|---|---|
| ua4l2vmq | 4 | 1 | 15.03 | B |
| ijubn2gz | 4 | 2 | 17.00 | A |
| jmo7u7d6 | 5 | 1 | 26.52 | B |
| huw6isc4 | 5 | 2 | 29.26 | A |
| vkyfae3p | 5 | 4 | 39.32 | A |
| szok2vom | 5 | 6 | 54.12 | A |
| xgd5y6vp | 6 | 1 | 73.67 (84.36) | B |
| u1xasv71 | 6 | 2 | 82.16 (87.45) | A |

At level 6 use at most 2 data workers: 12 workers x 4 ranks filled the node's 239 GiB
`/dev/shm` before step 1 (js5hwonm).

**Level 6, forecast steps 2, Andres's options** (`andres/backup-dump`, options off by default):

| Run | Options | Peak | Where |
|---|---|---|---|
| n624ql8q | none (control) | 82.16 | A |
| f1mgt06z | `decoder_gather_after_kv_projection` | 74.20 | B |
| jdg1lutw | that + `fe_checkpoint_per_block`, `ae_local_checkpoint_per_block`, `source_tokens_in_mixed_precision` ("4 options") | 69.78 | B |

**With a prototype of chunked embedding** (Alejandro, on top of `andres/backup-dump`, not
pushed; same idea as Matteo's change E): `embed_checkpoint_per_stream` (one checkpoint per
stream instead of one around all streams) and `embed_checkpoint_chunks: k` (each stream's
tokens embedded in k chunks).

| Run | Setup | Peak | Where |
|---|---|---|---|
| rliwt2zh | level 6, 1 step, per-stream only | 70.36 | B |
| g2z5vwaj | 4 options + per-stream | 66.52 | B |
| v2sceodc | 4 options + per-stream + 2 chunks | 53.45 (55.31) | A |
| qjbmx499 | 4 options + per-stream + 4 chunks | 53.45 (55.25) | A |
| xokjem39 | same, 4 forecast steps | 70.93 (72.85) | A |
| tdaw4mhu | same, 6 forecast steps | 88.68 (90.95) | A |

At level 6 the embedding recompute is dominated by ERA5_in (30.3 GiB of it in g2z5vwaj, every
other stream at most 6.6).

Speed at level 5 (Andres's `dphpc_measure.yml`, 50 steps, median GPU step): control 3.233 s;
4 options 3.225 s; + per-stream 3.120 s; + 2 chunks 3.387 s (+4.8%); + 4 chunks 3.953 s (+22%).
