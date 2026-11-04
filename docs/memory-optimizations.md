# GPU memory optimizations

Opt-in options that lower the peak GPU memory of training without changing the model.
All are off by default; `config/optimizations.yml` turns them on and states which options
need which. Measured on `config_operan_georing_avhrr_forecasting_lowres.yml` with
`dphpc_memory_baseline_100steps.yml`, 1 node (4 GH200), 1 sample per GPU, 100 steps.

| | Peak per GPU (max / median) | Run |
|---|---|---|
| no options | 29.31 GiB | [nyue0k4y](../experiments/2026-10-20_nyue0k4y/) |
| options of #1 and #2 | 17.14 / 17.11 GiB | [gbrjlgo3](../experiments/2026-11-04_gbrjlgo3/) |
| all options (`config/optimizations.yml`) | 7.49 / 7.45 GiB (-75%) | [jmbpdvci](../experiments/2026-11-04_jmbpdvci/) |

```
minion run train    --arg "CONFIG=config/dphpc_memory_baseline_100steps.yml,config/optimizations.yml"
minion run snapshot --arg "CONFIG=config/dphpc_memory_baseline_100steps.yml,config/optimizations.yml"
```

## What uses GPU memory

During one step the GPU holds:

- fixed state: the weight shard, gradients, AdamW states, the EMA copy, and weights FSDP has
  gathered for computing;
- the input batch;
- activations the forward pass keeps for the backward pass (with checkpointing: only the input
  of each checkpointed block);
- short-lived buffers of the backward pass (e.g. flash-attention scratch).

The peak is the highest point over the step. Several phases (decoder backward, embedding
recompute, forecast engine) reach similar heights, so lowering one phase only helps until the
next one is the highest. The highs per phase are printed by `scripts/analyze_memory_snapshot.py`.

## The options

PR: where the option was added (#1, #2, or this branch).

| Option | PR | What it does | Why it saves memory | Cost | Math |
|---|---|---|---|---|---|
| Recompute instead of keeping | | | | | |
| `checkpoint_per_block` | #1 | one checkpoint per attention + MLP block (forecast engine, local assimilation) | half as many checkpoint inputs are kept | none | identical |
| `fe_blocks_per_checkpoint: 2` | this | 2 blocks per forecast-engine checkpoint | half again | 2 blocks rebuilt at once | identical |
| `fe_checkpoint_per_step` | this | one checkpoint around each forecast step | only the step input is kept until the backward, instead of every block input of every step; flat in rollout length | one more forecast-engine forward | identical |
| `ae_adapter_checkpoint` | #2 | checkpoint each local-to-global adapter block | its activations (2.4 GiB) are not kept | one more adapter forward | identical |
| `ae_local_checkpoint_outer` | this | one checkpoint around each local-assimilation chunk | only the chunk input is kept | one more local forward | identical |
| `embed_checkpoint_per_chunk` | this | one checkpoint per chunk of patches, patch embedding included | the backward rebuilds one chunk at a time; nothing of the size of a stream is kept | one more embedding forward | identical |
| Smaller pieces | | | | | |
| `embed_num_chunks: 4` | #2 | embedding layers over 4 chunks of patches | smaller recompute and backward burst | more, smaller kernels | same, other dropout masks |
| `decoder_num_chunks: 4` | #2 | decoder over 4 groups of cells | smaller flash-attention backward scratch (-0.7 GiB against 2) | ~2% time against 2 | same, other dropout masks |
| No copies, narrower dtypes | | | | | |
| `decoder_gather_after_kv_projection` | #1 | project keys/values once per cell, then pick the 9 neighbours by index | no 9x copy of the cell tokens (2 x 864 MiB) | less compute | forward identical; gradients summed in bf16 |
| `source_tokens_in_mixed_precision` | #1 | source tokens on the GPU in bf16 | the embedding converts them to bf16 anyway | none | identical |
| `embed_index_copy` | #2 | `index_copy_` with a 1D index instead of `scatter_` | no int64 index the size of all tokens (0.78 GiB) | none | identical |
| State off the GPU | | | | | |
| `fe_reshard_after_forward` | this | free the gathered forecast-engine weights after each use | 1.0 GiB of bf16 weights are not kept through the step | more all-gathers (~1%) | identical |
| `source_tokens_on_cpu` | this | source tokens in pinned CPU memory, copied per stream by the embedding | the input batch is mostly not on the GPU (-1.1 GiB) | copies over NVLink-C2C | identical |
| `ema_on_cpu` | this | EMA weights in pinned CPU memory (FSDP CPU offload) | -0.86 GiB now, 4 bytes per parameter / GPUs | copies per step | identical |
| `fsdp_cpu_offload` | this | weights, gradients and AdamW states in pinned CPU memory; optimizer step on the Grace CPU | -2.6 GiB now, 12 to 16 bytes per parameter / GPUs | CPU step (~7%) | CPU rounding |
| Fix | | | | | |
| `train_gc_collect_every_step` | this | `gc.collect()` after each step | a reference cycle kept the previous batch alive into the next step: +1.7 GiB on random steps in about half of the runs | none measured | identical |

"Identical": same values (the checkpoint recompute replays dropout with the same random
state). Step-0 loss is bit-identical in those runs.

## How the savings added up

Median peak per GPU, each row on top of the previous one:

| Change | GiB | Run |
|---|---|---|
| options of #1 and #2 | 17.11 | gbrjlgo3 |
| + forecast engine (reshard, per step, 2 blocks), local outer checkpoint, 4 decoder chunks | 15.34 | emzyl1tp |
| + embedding checkpoint per chunk, EMA on CPU | 12.74 | kb0b39xi |
| + `fsdp_cpu_offload` | 9.34 | sjr8t2bc |
| + `train_gc_collect_every_step` (max 11.11 -> 9.36) | 9.34 | nrurj3mb |
| + patch embedding inside the chunk checkpoints | 8.80 | c0a906se |
| + `source_tokens_on_cpu` (this branch: jmbpdvci) | 7.45 | tn6df0qr |

Runs other than those in `experiments/` are recorded on the `andres/combined` branch.

## Trade-offs

- Time. The GPU work per step (median step time) went from 2.90 s (no options) to 4.06 s.
  Training waits on data loading (~4.5 s per batch), which hides most of it: 89 steps took
  461 s against 396 s with #1 and #2 and 415 s with no options (runs vary by about 3%).
  Once data loading is faster, the full recompute cost shows.
- Results. Losses after step 0 differ only at run-to-run noise: last 20 steps 0.5458
  (0.5529 with #1 and #2), ERA5 validation 0.5874 (0.5967). 100 steps show the computation is
  unchanged; they do not show the accuracy of a full training.
- `fsdp_cpu_offload`: about +4 GiB host memory per rank (data-loader workers already use about
  80 GiB per rank); more CPU threads than 16 contend with the workers; uses gloo for CPU
  collectives; skips the GradScaler, so fp16 would break (bf16 is fine); resuming from a
  checkpoint with it is untested.
- `ema_on_cpu`: validation copies the EMA weights from the CPU in every forward.
- `fe_reshard_after_forward` must stay off with the pushforward trick.
- Tested only on this config, 1 node, 1 sample per GPU, without spatial parallelism.

## Tried and dropped

| Idea | Result |
|---|---|
| `pred_checkpoint_per_layer` (decoder, coarser checkpoints) | raised the peak by 2.8 GiB: the decoder backward then rebuilds more at once |
| `fe_checkpoint_bf16` (bf16 checkpoint inputs) | nothing on top of the other options; rounds the residual |
| bf16 residual in the forecast engine | the LayerNorm after block 7 returns float32 under autocast and widens all later blocks to float32; keeping bf16 saved only 0.1 GiB here and rounds |
| one checkpoint per stream embedding | redundant with `embed_checkpoint_per_chunk` (+0.2 GiB, one more recompute) |
| `decoder_num_chunks: 2` | 2% faster, +0.7 GiB |

## Open

- The peak (7.46 GiB) is now in the local-assimilation backward: 4.3 GiB of flash-attention
  backward buffers. Smaller local-assimilation chunks are the next lever.
- Find and break the reference cycle that keeps a batch alive, instead of `gc.collect()`.
- Check with longer training, other configs, several nodes and spatial parallelism.
- Use the room: a deeper or wider forecast engine. Its gathered bf16 weights (2 bytes per
  forecast-engine parameter, not divided by the GPUs) become the next fixed cost.

## Measuring

`minion run snapshot` records a CUDA allocator snapshot of 3 steps;
`scripts/analyze_memory_snapshot.py <pickle>` (on JUPITER, with the `.venv`) prints what is
alive at the peak and the highest memory of each phase. To look at it,
`python -m torch.cuda._memory_viz trace_plot <pickle> -o out.html` and open the file in a
browser. Reports: [all options](../experiments/2026-11-04_gpu-memory-snapshot_r51jrmu4.txt),
[options of #1 and #2](../experiments/2026-11-04_gpu-memory-snapshot_su2lrjpi.txt).
