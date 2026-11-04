# Data-loader optimizations

Opt-in options that make the loader workers build each sample faster (disk → `yield batch`)
without changing what the model receives. All are off by default and read with `.get(...)`, so
existing configs keep the old code path. `config/dataloader_optimizations.yml` turns them on;
`config/dataloader_optimizations_level6.yml` adds worker settings for HEALPix level 6.
Measured on `config_operan_georing_avhrr_forecasting_lowres.yml`, 1 node (4 GH200), seed 1.

| | Before | After | Runs |
| --- | --- | --- | --- |
| Level 5, 100 steps, 12 workers | 4.67 s/step mean, 10 waits > 5 s (largest 39 s), job 12:51 | **2.88 s/step (−38%)**, no wait > 5 s, job 9:20 | `jnsctk9d` → `svxaue1j` |
| Level 6, 100 steps | 27.5 s/step (options off, 4 workers) | **22.1 s/step (−20%)** (all nine options, 5 workers) | `a5z733w1` → `k6i6mdyp` |

```
minion run train --arg "CONFIG=config/dataloader_optimizations.yml"
# 100-step test as measured above, on a login node:
make train CONFIG="config/dphpc_memory_baseline_100steps.yml config/dataloader_optimizations.yml"
```

## Why it was slow

Each GPU rank has its own DataLoader workers, and each worker builds whole samples (read, filter
NaNs, normalize, tokenize, build). At level 5 the GPU step is ~2.85 s and a sample takes ~33 s
to build, so 12 workers keep up only on average. Two things made training wait:

- **Uneven split.** Each worker got `len // workers` samples and the last worker also took the
  remainder (100 samples, 12 workers: eleven get 8, the last gets 12). At the end of every
  mini-epoch the GPU waited ~39 s for that last worker.
- **Copies in the readers and tokenizer.** Arrays were copied when nothing changed (float32 →
  float32, a NaN filter that keeps every row), whole datasets were read before selecting
  variables, and tokens were built as lists of small tensors and stacked.

## The options (`data_loading.optimizations`)

| Option | What it does | Model input | Measured |
| --- | --- | --- | --- |
| `even_workset_split` | leftover batches go one each to the first workers: 100 over 12 → four get 9, eight get 8 | other dates sampled (each worker draws from its own range) | mean step −28%, largest wait 39 → 10 s (`azmuvlzx`) |
| `anemoi_select_vars` | select the needed variables before loading the array (Anemoi and Operan readers) | identical | reader group: read 7.75 → 4.16 s per sample, worker RSS −3% (`kawac1qm`) |
| `anemoi_avoid_dtype_copy` | no copy when the array is already float32 (Anemoi and Operan readers) | identical | reader group |
| `obs_single_read` | obs reader: one read of the needed columns instead of three (coordinates, geoinfos, channels) | identical | reader group |
| `skip_all_valid_filter_copy` | NaN filter: reuse the arrays when every row is valid | identical | reader group |
| `tokenize_coords_only` | space-time tokenization: copy only coordinates where only coordinates are used | identical | tokenizer group: build −11% (`em2wu8xf`) |
| `source_direct_build` | build each source token tensor in place instead of a list of pieces and `torch.stack` | identical | tokenizer group: −2.1 s per sample |
| `target_requested_fields_only` | targets: build only the fields the caller uses | identical | tokenizer group |
| `skip_train_inverse_indices` | training: skip sorting indices that only evaluation needs | identical | tokenizer group |

The Operan reader has its own copy of the Anemoi `_get`, so the two Anemoi options are applied in
both readers.

`data_loading.prefetch_factor` (DataLoader setting, not under `optimizations`) is passed to
PyTorch only when set; otherwise PyTorch's default (2) stays. Use 1 at level 6, where each queued
sample holds 4–9 GiB of `/dev/shm`; at level 5 it made steps 7% slower.

## Checks

- With the options off the code path is the old one. Unit tests on synthetic data check that each
  option gives the same output as the old path (and that the split is balanced): `src/weathergen/datasets/tokenizer_optimizations_test.py`,
  `tests/test_dataloader_reader_optimizations.py`, `tests/test_dataloader_loader_options.py`.
- Loss with all options on stays within same-seed run-to-run noise; GPU peak unchanged (29.33 vs
  29.31 GiB at level 5).
- `even_workset_split` changes which dates each worker samples, so losses are not step-for-step
  equal; they stay similar and decrease the same way.

## Level 6

Level 6 uses the same nine options as level 5; none is specific to one level.
`config/dataloader_optimizations_level6.yml` adds only `healpix_level: 6`, 5 workers and
`prefetch_factor: 1`. Both level-6 runs above use `prefetch_factor: 1`; they differ in the options
and in 4 vs 5 workers.

Level 6 has 4× the cells (49,152) and each fp32 sample takes 9.2 GiB of `/dev/shm`, 98% of it
source tokens. With 8 or more workers the node runs out of memory, so the level-6 config uses 5
workers and `prefetch_factor: 1`. Casting the source tokens to bf16 in the loader workers halves
each sample: then 8 workers fit and a step takes 13.9 s (`oq8xiir6`, measured with a prototype of
that cast; the team version is `source_tokens_in_mixed_precision` on `alejandro/bf16-worker`).
