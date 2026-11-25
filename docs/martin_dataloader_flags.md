# Data-loader flags on the hackathon base

Six `data_loading` flags on top of the hackathon base (Florian Scheidl's
`exp/sync-barriers-perf-gc-full`, `2416cf631`). Each is off by default and read with
`cf.data_loading.get(...)`, so old configs run the original path. Each has a test next to the code
that compares the batch with the flag off and on.

**Two flags are new. The other four port options the DPHPC fork already had** (on `dphpc/dev`,
`martin/main` or the authors' branches) to the hackathon code, which is laid out differently. The
table names the earlier versions.

| Flag | Commit | Status | Earlier version |
| --- | --- | --- | --- |
| `release_yielded_batch` | `e02b4459` | new | none (see below) |
| `fast_time_encoding` | `295354c7` | new | none: every version encodes each point |
| `fast_read_copies` | `b125bb77` | partly new | see below |
| `fast_source_build` | `c0ce7b42` | port | `source_direct_build` (Martin, `martin/main`), `masked_counts_bincount` (Andres, #20) |
| `fast_target_build` | `c52ca8f7` | port | `target_requested_fields_only` (Martin, #8), `skip_train_inverse_indices` (Martin, `martin/main`), `masked_counts_bincount` (Andres, #20) |
| `source_tokens_bf16` | `07935694` | port | `source_tokens_in_mixed_precision`, bf16 source tokens in the workers (Alejandro, #3) |

PR numbers refer to maforn/WeatherGenerator; #3, #8 and #20 are merged into `dphpc/dev`.

## The two new flags

`release_yielded_batch`: the sampler is a generator, and after `yield batch` it keeps `batch`
alive while it builds the next one. By then the DataLoader worker has put the batch in its queue,
which moved its tensors to shared memory, so every worker holds one extra batch. The flag deletes
the reference after the yield (`release_yielded_batch_test.py`).
- Closest earlier work: `train_gc_collect_every_step` (Andres, #16, `dphpc/dev`) also drops a batch
  reference, but in the training process, against GPU memory spikes from a reference cycle. It
  cannot free the copy a worker holds, and this flag does not touch the GPU.
- Related: `source_tokens_unpadded` (Andres, `andres/efficiency`, no PR) makes each batch smaller;
  the two would add up.

`fast_time_encoding`: encodes each run of equal consecutive times once and repeats the rows,
instead of encoding every point (a geostationary window has about 5 M points and 6 times). It falls
back to the old path when there are more runs than half the points
(`fast_time_encoding_test.py`). The tokenizer speedups #19–#21 (Andres) changed other parts of the
tokenizer; the time encoding is the same on `dphpc/dev`, every fork branch and PR, the hackathon
base and ECMWF `develop`.

## What fast_read_copies takes from earlier work

- No copy when the time mask or the NaN filter keeps every point: `skip_all_valid_filter_copy`
  (Martin, #9).
- One gather of the needed obs columns, with coordinates, geoinfos and data as views of it: Andres's
  `b180dc36` (`andres/efficiency`, no PR). `obs_single_read` (Martin, #7) also reads the columns
  once, but copies each part out.
- Normalising all channels at once: upstream ECMWF #2793 (Javad Kasravi) does it for the data
  channels and is in the hackathon base; this flag does the same for the geoinfo channels.
- New: the row selection inside the single gather, no concatenate when only one reader has points,
  and `torch.from_numpy` in `readerdata_to_torch`.

## Measured on JUPITER (1 node, 4 GH200, 40 steps, seed 1794328803)

| Change | Runs | Result |
| --- | --- | --- |
| `release_yielded_batch`, 4 workers | `d19mnc19` → `hu0kystd` | host memory (PSS) peak 255.0 → 158.7 GiB, loss the same to all printed digits |
| `source_tokens_bf16`, 6 workers + release | `us41y657` → `em1oknyw` | `/dev/shm` peak 115.7 → 65.7 GiB, GPU memory 44.46 → 41.84 GiB, same loss |
| the four `fast_*` flags together, 4 workers + release | `hu0kystd` → `c1rxuuoz` | worker time 32.1 → 21.8 s per batch, mean step 8.06 → 5.36 s, same loss |
| hackathon fast GPU stack: 4 workers → 8 workers + all six flags | `f1aw45ai` → `tidlk85o` | mean step 8.11 → 2.91 s, no step waits for data |

The four `fast_*` flags only ran together, so the share of `fast_time_encoding` on JUPITER is not
measured. That training waits for the loader and more workers help was shown on the fork before
(Andres's worker sweep of 2026-10-20, runs `ne1uy549`, `jdadmo6b`, `aw3t8coe`, `adals25u`). The
fast GPU stack's O1 is Matteo's `embed_readout_batched` (#5).

## Enabling

Pass them as config overrides, as the runs did (`ARGS+=(...)` in the launch commands of the run
notes):

```
data_loading.release_yielded_batch=True data_loading.source_tokens_bf16=True
data_loading.fast_read_copies=True data_loading.fast_source_build=True
data_loading.fast_time_encoding=True data_loading.fast_target_build=True
```

Run notes: weathergenerator-experiments, `experiments/performance/2026-11-perf-dphpc/`
(data-loading 001–007; on Martin's branch `dphpc/2026-11-perf`, not on GitLab yet).
