# Findings: data loading of the operan config

Setup: `config_operan_georing_avhrr_forecasting_lowres.yml` + `dphpc_memory_baseline_100steps.yml`, 1 node (4 GH200),
1 sample per GPU, 100 steps, 12 DataLoader workers per rank unless stated. Runs and raw metrics are in the [table](README.md).

## 1. Time per step (run e3392v33)

Mean after warm-up, per rank (the stages of one rank add up; the max over ranks used before did not).

| Stage | Rank 0 | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|---|
| training loop waits for the next batch | 1.19 | 1.51 | 1.02 | 1.04 |
| step (forward, backward, optimizer, incl. copy to GPU 0.05) | 3.11 | 2.79 | 3.28 | 3.26 |
| a finished batch waits before it is used (`queued`) | 26.0 | 20.8 | 28.3 | 21.9 |
| one worker building one batch | 40.0 | 40.8 | 39.0 | 40.5 |
| of which read (zarr) | 7.2 | 7.3 | 7.2 | 7.3 |
| preprocess | 1.0 | 1.0 | 1.0 | 1.0 |
| tokenize | 18.6 | 19.2 | 17.9 | 18.9 |
| assemble | 13.1 | 13.4 | 12.9 | 13.3 |

- Wall time 4.55 s per step; wait + step per rank is 4.1 to 4.6 s.
- Handing one batch from a worker to the main process takes 0.78 s (`profile-data`, outside training).
- The first batch takes about 45 s, paid again at every DataLoader start (each mini-epoch and validation).
- Finished batches wait about 25 s on average, yet the loop still waits 1 to 1.5 s per step. Likely cause: the
  DataLoader returns batches strictly in worker order, so finished batches of other workers sit in memory while the
  loop waits for the next one in order.

## 2. Host memory (run e3392v33, peak)

| | Per rank |
|---|---|
| main training process | 2.1 GiB |
| 12 DataLoader workers | 89 GiB |
| of which shared memory (`/dev/shm`): finished batches waiting for the main process | about 34 GiB (node 135 GiB) |
| of which anonymous: working memory while building batches | about 53 GiB (node 211 GiB) |

- Per worker: about 2.8 GiB of finished batches plus 4.4 GiB working memory. A batch is about 2.4 GB of source
  tokens, so about 13 batches per rank are in shared memory at the peak.
- The cgroup on JUPITER covers the job step on a node (all 4 ranks), so the cgroup values are node totals.

## 3. Hot functions ([cProfile](2026-10-20_cprofile_data-loading.txt), 3 batches in one process)

| Function | s per batch | Why |
|---|---|---|
| `tokenizer_utils.hpy_splits` | 18.8 | per-cell Python loop: 12,288 cells x 27 calls per batch, each with `argsort`, `cat`, `zeros`, `split` |
| zarr reads (mostly the obs readers) | about 8 to 13 | ERA5 reader is only 0.6 |
| `tokenize_apply_mask_target` | 7.8 | incl. `get_target_coords_local` (5.0) |
| `tokenize_apply_mask_source` | 6.4 | per-token list comprehensions |

## 4. Worker sweep (memory vs speed)

| Workers | s/step | Mean wait per step (slowest rank) | Host RSS per rank (Slurm) |
|---|---|---|---|
| 4 | 11.1 | 8.5 | 32.5 GiB |
| 6 | 8.5 | 5.9 | n/a |
| 8 | 7.0 | 4.4 | 59.8 GiB |
| 12 | 4.75 | 2.0 | 91.3 GiB |

- Step time is about 45 s / workers: building batches limits training at every tested worker count.
- Keeping up with the GPU (about 3 s per step) would need about 15 workers, more host memory than one GH200 has.
  More workers is not a solution; cheaper batches are.

## 5. GPU memory ([snapshot report](2026-10-18_gpu-memory-snapshot_huw6isc4.txt), team run huw6isc4, rank 0, 3 steps)

The peak, 29.2 GiB per GPU, is reached in the backward pass (flash-attention backward), while the activations saved
by the forward pass are still alive. Made with `scripts/analyze_memory_snapshot.py`.

| Live at the peak | GiB |
|---|---|
| activations saved by the forecast engine | 5.7 (+1.0 FSDP buffers) |
| activations saved by the decoder (`predict_decoders`) | 4.8 |
| activations saved by the encoder | 4.2 |
| activations from checkpoint recomputation (`MultiSelfAttentionHeadVarlen`) | 2.3 |
| other forward activations | 1.5 |
| backward temporaries (gradients of activations) | 3.1 |
| input batch | 2.65 |
| weights (FSDP shard) + EMA copy | 1.8 |
| AdamW states | 1.7 |

- About 75% is activations; weights, gradients and optimizer states are about 4.5 GiB, because FSDP shards them over 4 GPUs.
- For bigger models, activations grow with width x depth x tokens and are the limit, not the weights.

## 6. GPU memory scaling (2026-10-25, 50 steps each, peak allocated per GPU, max over ranks)

| Test | Run | Peak GiB | vs baseline | s/step | samples/s (4 GPUs) |
|---|---|---|---|---|---|
| baseline | nyue0k4y | 29.31 | | 4.64 | 0.86 |
| `distributed.spatial_parallel.size=2` | myphnyfz | 20.28 | -31% | 3.35 | 0.60 |
| `distributed.spatial_parallel.size=4` | umwsh7wk | 15.77 | -46% | 3.71 | 0.27 |
| `fe_num_blocks=32` (forecast engine 2x deep) | ibiegv9d | 38.30 | +9.0 GiB | 4.34 | 0.92 |
| `ae_local/global_dim_embed=3072` (1.5x wide) | nayvfdr1 | 40.47 | +11.2 GiB (+38%) | 4.38 | 0.91 |

- Spatial parallelism saves less than 1/size: the forecast engine and global assimilation are still computed
  in full on every GPU (about 15.8 GiB remain at size 4). It also lowers throughput, since fewer samples run in parallel.
- Each extra forecast-engine block costs about 0.56 GiB per GPU, of which about 0.17 GiB weights and optimizer states.
- Width grows memory less than linearly here (x1.5 width, +38%), because input data and embeddings do not grow.

## 7. GPU memory reference on the unchanged config (2026-10-26)

Sections 5 and 6 used the team overlay, which also sets RMSNorm and a later start date. The reference from now on is the
operan config unchanged plus `dphpc_measure.yml` (50 steps, fixed data order, logging only):

- run dw1fgpmo: peak 31.69 GiB allocated per GPU; step 0 29.8 GiB, every later step about 31.6 GiB.
- snapshot lvzfwv01 ([report](2026-10-26_gpu-memory-snapshot_lvzfwv01.txt)): the replay finds 29.9 GiB at the peak; the
  about 1.7 GiB that later steps add are not in it (open question).

What the peak consists of (from the snapshot):

| Part | GiB | What it is |
|---|---|---|
| forecast engine | 5.7 (+1.0 FSDP) | 52 x 96 MiB + 18 x 48 MiB: the saved input of each checkpointed layer (12,288 cells x 2048, float32 and bf16) |
| decoder | 4.8 | incl. 2 x 864 MiB copies of each cell's 9-cell neighbourhood (`select_healpix_neighborhood_shard`) |
| encoder | 4.4 | token-level and cell-level saved inputs |
| recomputed attention (`MultiSelfAttentionHeadVarlen`) | 3.5 | layer-norm outputs of a large varlen attention (4 x 472.5 MiB) |
| backward | 3.1 | incl. one 2.34 GiB flash-attention backward buffer |
| input batch | 2.0 | source tokens copied to the GPU |
| weights + EMA, optimizer, FSDP | 4.7 | |

The peak is reached inside a flash-attention varlen backward, while the forecast engine and encoder inputs are still
alive; most likely the decoder's self-attention over the target points (not proven).

## 8. GPU memory options, each on its own (2026-10-26, 50 steps, against dw1fgpmo)

Each option is off by default and was turned on alone (see [docs](../docs/jupiter.md#gpu-memory-options)).
Peak = max over steps of the allocated GPU memory per GPU (max over ranks). Loss difference = per-step relative
difference to the reference, which is at the level of run-to-run noise for options with the same math.

| Option | Run | Peak GiB | Saved GiB | s/step | Loss diff mean / max |
|---|---|---|---|---|---|
| reference | dw1fgpmo | 31.69 | | 4.48 | |
| `fe_checkpoint_per_block` | cpg9zlnw | 30.03 | 1.66 | 4.29 | 0.01% / 0.09% |
| `decoder_gather_after_kv_projection` | u4x9cfxf | 29.99 | 1.69 | 4.31 | 0.03% / 0.20% |
| `ae_local_checkpoint_per_block` | au2md54b | 30.91 | 0.78 | 4.44 | 0.04% / 0.23% |
| `pred_checkpoint_per_layer` | ko1p095r | 34.48 | -2.79 | 4.47 | 0.02% / 0.14% |
| `source_tokens_in_mixed_precision` | py4htb4q | 30.53 | 1.16 | 4.30 | 0.05% / 0.24% |
| `fe_checkpoint_bf16` | oe2t8gs7 | 29.99 | 1.69 | 4.32 | 0.06% / 0.58% |

- Grouping several FSDP modules in one checkpoint needs their weights gathered at the start of the recompute
  (`_unshard` in `engines.py`); the first attempts (z7frx7e3, a5g9twto) failed without it.
- `pred_checkpoint_per_layer` raises the peak: the peak is in the decoder backward, where a coarser checkpoint
  recomputes more at once.
- Three options end at about 30.0 GiB, the step-0 level of the reference: each removes at least 1.7 GiB from the
  phase that set the peak, and another phase (about 29.9 GiB) becomes the highest. Their combined saving needs a
  combined run.

## 9. Options combined (2026-10-26, 50 steps, against dw1fgpmo)

| Run | Peak GiB | Saved GiB | Step 0 | s/step | Loss diff mean / max |
|---|---|---|---|---|---|
| reference dw1fgpmo | 31.69 | | 29.81 | 4.48 | |
| 4 exact options (zjif1x9c) | 28.93 | 2.76 (8.7%) | 27.06 | 4.27 | 0.05% / 0.17% |
| 4 exact options + `fe_checkpoint_bf16` (pbpl25ww) | 28.93 | 2.76 | 27.06 | 4.28 | 0.03% / 0.12% |

The 4 exact options are `fe_checkpoint_per_block`, `decoder_gather_after_kv_projection`,
`ae_local_checkpoint_per_block` and `source_tokens_in_mixed_precision`.

- Step 0 is about 1.7 GiB lower in every run because AdamW creates its states (1.72 GiB per GPU) only at the end of
  step 0, after the first backward pass.
- The peak moved ([snapshot report](2026-10-26_gpu-memory-snapshot_bnayqygv.txt)): it is now in the backward pass of
  the stream embedding, which runs in one checkpoint over all 9 streams, so all of them are recomputed at once
  (recomputed embedding attention 10.1 GiB, backward buffers 7.1 GiB, embedding outputs and positional encoding
  2.7 GiB). Forecast engine (0.2 GiB) and decoder (0.07 GiB) are small at that moment, which is also why adding
  `fe_checkpoint_bf16` saves nothing more.

## Next

1. Vectorize `hpy_splits` (one sort by cell and latitude, then split at cell boundaries): about 45% of the worker time.
2. Vectorize `tokenize_apply_mask_source/target` and `get_target_coords_local`: about 30%.
3. Memory without code changes: `prefetch_factor=1` (fewer finished batches in shared memory) and persistent workers.
4. Reads: obs readers do three separate zarr reads (coords, geoinfos, data) per window.
