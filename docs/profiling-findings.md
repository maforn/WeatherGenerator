# Profiling findings: data loading and GPU use

What profiling the operan config showed after the GPU memory optimizations
([memory-optimizations.md](memory-optimizations.md)), and what was changed.
Setup: `config_operan_georing_avhrr_forecasting_lowres.yml` + `dphpc_memory_baseline_100steps.yml`,
1 node (4 GH200), 1 sample per GPU; times are for steps 10 to 99 (89 steps).

## Tools

| Tool | For | Command |
|---|---|---|
| Scalene, CPU + memory | data loading, line by line (one process, no workers) | `minion run scalene-data` |
| Scalene, CPU only | training process of rank 0 | `minion run scalene-train` |
| Nsight Systems | GPU kernels, CUDA calls, NCCL, per step | `minion run profile --arg "OPTIONS=profiling.nvtx_annotate=True"`, then `scripts/analyze_nsys.py` |

`scripts/profile_data_loading.py --checksum` prints a hash of every batch, to check that a
data-loading change gives the same batches.

Scalene's GPU mode does not work for the training: it runs `torch.profiler` from its signal
handler, which crashes the model setup; started late (`scripts/scalene_launch.py`) it still corrupts
the heap of rank 0 (with `fsdp_cpu_offload`) or hangs it (without). Use Nsight Systems for the GPU.

## Data loading: 3x faster, same batches

Scalene found 54% of the data-loading CPU time in `tokenizer_utils.py`, mostly small torch calls
repeated for each of the 12,288 HEALPix cells of every stream and window.

| Change | s per batch |
|---|---|
| before | 41.45 |
| `hpy_splits`: one stable sort by cell and latitude, one padding and one split for all points; per-cell counts with one `bincount` | 22.05 |
| target local coordinates on the flat points (each of 13 rotations re-concatenated and re-split 12,288 per-cell tensors) | 13.01 |
| one zarr read per window in the observation reader instead of three; vertex coordinates gathered directly | 10.78 |

Back to back on the same node: 34.9 / 40.5 s before, 10.5 / 13.9 s after (about 3.1x).
The checksums of all batches are unchanged.

## Training time with the new data loader

| Options | 89 steps | Peak per GPU | Host memory per rank | With the old loader |
|---|---|---|---|---|
| none | 283 s | 29.31 GiB | | 415 s |
| options of #1 and #2 | 320 s | 18.91 GiB | | 396 s |
| all 16 (`config/optimizations.yml`), 12 workers | 378 s | 7.49 GiB | 76.9 GiB | 461 s |
| all 16, 6 workers | 383 s | 7.47 GiB | 44.2 GiB | |

The data loader alone makes training 1.47x faster. Training no longer waits on data, so 6 workers
are as fast as 12, with half the host memory. With the waiting gone, the time cost of the memory
options shows: all 16 take 34% longer than no options.

## Where the step time goes (Nsight Systems, all options, new loader)

| | No options, old loader | All options, new loader |
|---|---|---|
| GPU running compute kernels | 57% | 40% |
| only NCCL running | 14% | 9% |
| GPU idle | 30% | 51% |

Per step (with Nsight overhead, 6.06 s): 35,500 kernel launches and 627 stream synchronizations
(0.13 s waiting). The GPU is idle 1.43 s per step during `torch.autograd.backward` and 0.43 s
outside any annotated range (optimizer step, EMA). GPU kernel time: GEMM 20%, NCCL all-gather 18%,
flash-attention 29%, elementwise 13%, norms 12%.

So the cost of the memory options is CPU work, not GPU arithmetic: the backward pass reruns the
Python forward code of every checkpointed block, and the GPU waits while thousands of small
kernels are launched.

## Using the freed memory: 2 samples per GPU

Same number of kernel launches, twice the work per kernel (`training_config.model_input.source_masking.num_samples=2`,
6 workers, 50 steps):

| Options | Samples per GPU | Step | Samples/s | Peak per GPU | Host memory per rank |
|---|---|---|---|---|---|
| none | 1 | 3.16 s | 1.27 | 29.31 GiB | 72.5 GiB |
| none | 2 | 5.17 s | 1.55 (+22%) | 53.81 GiB | 72.1 GiB |
| all 16 | 1 | 4.29 s | 0.93 | 7.47 GiB | 44.2 GiB |
| all 16 | 2 | 6.30 s | 1.27 (+37%) | 14.10 GiB | 34.4 GiB |

All options with 2 samples per GPU give the throughput of the original setup with 2.1x less GPU memory
and half the host memory. More samples per GPU are limited by the host, not the GPU: with 8 or 12
workers the finished batches overflow `/dev/shm` (DataLoader workers die with a bus error).

## Next

- Halve the host shared memory per batch (source tokens in bf16 already in the workers), to allow
  3 or 4 samples per GPU (about 28 GiB for 4 with all options).
- Fewer, larger kernels: norms and elementwise kernels are 25% of the GPU kernel time and most of
  the launches; fusing them (e.g. `torch.compile` on the blocks) cuts both.
- Less nested recompute where memory allows (each level reruns the Python forward).
