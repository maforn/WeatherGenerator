# Scaling to higher HEALPix levels

How the operan config scales from HEALPix level 5 to 7 with the GPU memory optimizations
([memory-optimizations.md](memory-optimizations.md)) and the faster data loading
([profiling-findings.md](profiling-findings.md)). 1 node (4 GH200), 1 sample per GPU, 30 steps,
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

Runs started with `make train-direct` give a different step-0 loss than the same setup started with
`make train` (0.71876 against 0.71779 at level 5, with the same first batch and the same model
input); the cause is not found yet, so compare runs only within one way of starting them.

## Next

- Ship only the real points to the GPU and pad there (inside the embedding chunk checkpoints),
  instead of padding in the workers: 10x (level 5) to 40x (level 6) less data in the workers,
  shared memory, hand-off and copies, with the same model.
- Or scale `token_size` with the level (4x smaller per level). This changes the model's
  embedding; at level 6 it makes a batch 3x smaller (1.5 GiB in bf16) and the worker peak
  6.0 GiB instead of 21.9 GiB, with the same time per batch.
- Local or windowed attention in the forecast engine and global assimilation at level 6 and
  above (`forecast_att_dense_rate`, `ae_global_att_dense_rate` < 1), a modeling decision.
