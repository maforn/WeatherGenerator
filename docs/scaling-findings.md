# Scaling to higher HEALPix levels

How the operan config scales from HEALPix level 5 to 7 with the GPU memory optimizations
([memory-optimizations.md](memory-optimizations.md)) and the faster data loading
([profiling-findings.md](profiling-findings.md)). 1 node (4 GH200), 1 sample per GPU, 30 steps (10 at level 7, 7vgrhdp9),
`dphpc_memory_baseline_100steps.yml`, `healpix_level=<level>`; levels 6 and 7 started with `make train-direct`,
level 5 from xatp1z1w and a2ojap23 (`make train`, 100 steps) and zm0tjco5 (`make train-direct`).

## Results

| Level | Cells | Peak per GPU, all options | Peak per GPU, no options | Step, all options | Step, no options | Workers per rank that fit |
|---|---|---|---|---|---|---|
| 5 | 12,288 | 7.48 GiB | 29.31 GiB | 3.7 s | 2.8 s | 12 |
| 6 | 49,152 | 21.02 GiB | 82.23 GiB | 15.6 s | 13.6 s | 3 |
| 7 | 196,608 | 59.61 GiB | does not fit | 161 s | | 1 |

Data loading, one batch built in one process (`scripts/profile_data_loading.py`, float32 tokens):

| Level | Batch | s per batch | Peak memory of the process |
|---|---|---|---|
| 5 | 2.7 GiB | 15.6 | 8.2 GiB |
| 6 | 9.4 GiB | 27.3 | 21.9 GiB |
| 7 | 34.2 GiB | 60.6 | 76.8 GiB |

## Bottlenecks

1. Padding in the source tokens. Each occupied cell's last token is padded to the stream's
   `token_size` (1,024 points for the geostationary satellites). The real data stays at 0.24 GiB
   per batch, so the padding share grows with the number of cells: 89.8% of the source tokens at
   level 5, 97.4% at level 6. Every DataLoader worker builds, keeps in shared memory and hands
   over this padding, so host memory, not the GPU, limits the workers: 12 per rank at level 5,
   3 at level 6, 1 at level 7 (12 workers at level 6 were killed).
2. Dense global attention. The forecast engine and the global assimilation attend over all
   cells (`*_att_dense_rate: 1.0`): compute grows 16x per level, memory 4x (flash-attention).
   A step takes 4x longer at level 6 than at level 5, and 10x longer at level 7 than at level 6
   (161 s, with the 60 s batch of the single worker in time).
3. Data loading time. A batch takes 15.6, 27.3 and 60.6 s at levels 5, 6 and 7, and fewer
   workers fit at the same time: with 1 worker at level 7 a step cannot be faster than 60 s.
4. GPU memory. With all options, level 6 needs 21 GiB per GPU; without them 82 of 96 GiB, so
   larger models at level 6 need the options.

## Changes made

- Source tokens in bf16 already in the DataLoader workers (`source_tokens_in_mixed_precision`):
  half the shared memory and hand-off per batch. Host memory per rank at level 5 with 12
  workers: 76.9 -> 41.6 GiB. The values are identical: all 621,228,864 source token values of a
  batch equal the float32 tokens converted to bf16.
- `make train-direct`: training from the folder's `.venv`, for when `launch-slurm.py` cannot build
  a new venv (GitHub release downloads timed out with 504 for hours).

Early `make train-direct` runs gave each rank too few CPUs (`srun` does not pass on
`--cpus-per-task` unless told, the JUPITER job script sets `SRUN_CPUS_PER_TASK`): slower steps
(6.27 instead of 4.09 s at level 5) and a different step-0 loss (0.71876 instead of 0.71779).
Fixed; the level 5 to 7 rows above come from those runs, so their step times are too high.

## Unpadded source tokens and torch.compile (`config/efficiency.yml`)

All options, 1 sample per GPU, `make train-direct` with the CPU fix (s per step from step 10):

| Run | Level | Workers | s per step | Peak per GPU | Host memory per rank | Source data per sample |
|---|---|---|---|---|---|---|
| qtpiqnva, control | 5 | 12 | 4.09 | 7.49 GiB | 63.1 GiB | 1.21 GB |
| tz30c2cz, `source_tokens_unpadded` | 5 | 12 | 4.06 | 7.60 GiB | 38.7 GiB | 0.13 GB |
| 0gath1pq, `compile_blocks` | 5 | 12 | 5.10 | 7.49 GiB | 82.9 GiB | 1.21 GB |
| f0lkhiju, both | 5 | 12 | 4.92 | 7.60 GiB | 60.0 GiB | 0.13 GB |
| h4tgdnkt, control | 6 | 3 | 15.50 | 21.02 GiB | 43.9 GiB | 4.7 GB |
| yi4knvey, both | 6 | 3 | 16.44 | 20.06 GiB | 35.7 GiB | 0.14 GB |
| 160m47a2, both | 6 | 12 | 16.34 | 20.05 GiB | 55.9 GiB | 0.14 GB |

- `source_tokens_unpadded`: the workers send only the real points; the embedding pads them on
  the GPU inside each chunk checkpoint. Same batches (identical checksums at levels 5 and 6),
  9x (level 5) to 34x (level 6) less source data, 39% less host memory at level 5, 1 GiB less
  GPU memory at level 6, and 12 workers fit at level 6 (3 before). The step time does not change:
  training no longer waited on data.
- `compile_blocks` (off): `torch.compile` of each MLP and attention module, with
  flash-attention 3 run eagerly in between (it has no fake kernels). 25% slower at level 5, 6%
  at level 6: the step is bound by the CPU launching kernels, and each compiled call adds
  wrapper work (twice per attention module, split at flash-attention), also in every recompute.
  Compiled processes sometimes segfault at exit, after the model is saved.

## Bigger models at level 6

All options + `efficiency.yml`, 6 workers, 30 steps; parameters estimated from the checkpoint
size (float32):

| Run | Change | Parameters | s per step | Peak per GPU |
|---|---|---|---|---|
| j0lrc26f | none | ~0.92 B | 15.6 | 20.0 GiB |
| oguq1ja3 | `fe_num_blocks=32` | ~1.46 B | 25.0 | 21.4 GiB |
| sxdacewj | `ae_global_dim_embed=3072` | ~1.80 B | 20.3 | 25.1 GiB |
| 6l1ncv26 | 2 samples per GPU | ~0.92 B | 30.1 (15.05 per sample) | 39.2 GiB |

With the memory options, model size at level 6 costs time, not memory: twice the parameters fit
in 25 GiB of the 96 GB per GPU. A second sample per GPU adds only 3.5% throughput, as the GPU is
already busy at level 6.

## Next

- Bigger compiled regions: register flash-attention 3 as a torch custom op (with fake kernels),
  so whole checkpointed groups of blocks compile into one graph each.
- Scale `token_size` with the level (4x smaller per level). This changes the model's
  embedding; at level 6 it makes a batch 3x smaller (1.5 GiB in bf16) and the worker peak
  6.0 GiB instead of 21.9 GiB, with the same time per batch.
- Local or windowed attention in the forecast engine and global assimilation at level 6 and
  above (`forecast_att_dense_rate`, `ae_global_att_dense_rate` < 1), a modeling decision.
