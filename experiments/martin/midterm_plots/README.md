# Midterm plots: data-loader bottleneck study (made 2026-11-11)

Slide-ready PNGs (2200×1120 px, white background) for Matteo's shared presentation for the 18 Nov midterm.
**Every value comes from a measured run.** There are no estimates, projections or what-ifs; no SP7, no L8.
`numbers.json` holds every plotted value.

Regenerate from the raw run folders (all local, untracked) with:

    WeatherGenerator/.venv/bin/python scripts/plot_midterm_loader.py
    WeatherGenerator/.venv/bin/python scripts/plot_loader_options.py   # figure 10 only

| File | Message | Source runs |
|---|---|---|
| `01_loader_keeps_up.png` | Step time 4.1 / 15.9 / 162 s at L5/6/7; waiting for data 1.3 / 0.3 / 0.0 % of the step | A `ykm2kcxf`, C `pv39uted`, D `ywaqq3tw` (data parallel, 1 node, 4 GPUs) |
| `02_loader_stage_time.png` | Loader time per sample 14 / 32 / 92 s; read ~5 s at every level, build 6 → 20 → 72 s | same |
| `03_source_tokens_padding.png` | Source tokens 1.3 / 4.6 / 16.7 GiB per sample; 89 / 97 / 99.3 % of rows are padding | same + PAD (`experiments/martin/padding`) |
| `04_host_memory.png` | Per GPU: training process 9 / 13 / 31 GiB, its workers 66 / 71 / 47 GiB; node CPU memory peak 318–331 of 478 GiB | A, C, D |
| `05_read_throughput.png` | Read-only throughput 0.19 → 13.6 samples/s (128 readers); L5 training uses 0.95 samples/s (14×); I/O wait ≤ 0.5 % | READ job 2766316 + A |
| `06_workers_l5.png` | 4 vs 12 workers per GPU at L5: step 3.98 vs 4.11 s; node CPU memory 155 vs 318 GiB | W4 `kdi7cfy9`, A |
| `07_spatial_parallel_loader.png` | Loader per sample, data parallel vs one spatial rank: L5 12 vs 11 s, 8 vs 3 GiB; L7 91 vs 48 s, 77 vs 23 GiB; 4 ranks together 42 s (L5) and 173 s (L7) | SPLOAD job 2801802 |
| `08_spatial_vs_data_parallel_l5.png` | Alejandro's L5 training: samples/s 0.38–0.39 data parallel vs 0.13–0.14 spatial; GPU peak 28–29 vs 15.5–15.8 GiB. All four runs were loader-bound (65–74 % waiting). | `pi2vsvu1`, `n3tagpzt`, `qb0vmkdr`, `cnkr2ooi` |
| `09_gpu_busy.png` | GPU busy 64 % and 60 % of the step at L5, 99 % at L7; at L5, 1.44 
| `10_loader_options_l5.png` | My nine loader options, each alone at L5 against the runs that leave its stage alone: obs read −37 %, build −18 %, NaN filter −45 %, ERA5 read −25 %, step time with the even split −24 %; the four unmerged options stay inside the noise (−2 to +6 %). All nine together: step 4.61 → 2.88 s, loader wait per GPU 115 → 7 s. Values in `numbers_options.json` | single-flag runs 6 Nov: baseline `spcoicgc`, `vdhwy1yz`, `zi2h7d7a`, `hq52fbnw`, `fm6x4ski`, `ko1zio6z`, `cv2iljv2`, `t1te97ff`, `dom75441`, `mit3pxf0`; together: `jnsctk9d` / `svxaue1j` (1 Nov) |s idle in 21,397 gaps, waiting for data 0.16 ms | Jonah's `i3f3hngb`, PROF7 `q4ag4rlz` |

## How the values are computed
- **Options (figure 10):** each option's stage is the median over its 400 samples of the seconds that stage takes (summed over the streams it touches), compared with the median and range of the other single-option runs that leave that stage alone (the baseline node read ~15 % slower). `even_workset_split` changes no stage, so its row is the step time: mean interval between batch arrivals from step 10 on (loader wait log timestamps), mean of the 4 GPUs.
- **Step time:** median interval of `elapsed_training_time_seconds` in `<run>_train_metrics.json`.
- **Waiting for data:** mean `wait_seconds` from `dataloader_profile/loader_wait_rank*.jsonl` over steps ≥ 1, divided by the mean step, averaged over the 4 ranks. Step 0, which waits while the workers start, is excluded.
- **Loader stages:** per sample from `samples_rank*_worker*.jsonl`, summed over streams, median over samples. Read includes normalize; "other" is the total minus the stages.
- **Source tokens:** the `source_samples.samples` bytes in the worker payload, median over samples.
- **Host memory:** the maximum over time of `rank_rss_bytes` and `workers_rss_bytes` per rank, worst rank shown. The node value is `cpu_mem_used_bytes`, CPU only; GPU memory isn't included.
- **Padding share:** 1 − real points / padded rows, summed over streams, from `padding_summary.json`. These are counts on 4 real samples, checked exact against the real tokenizer at L5. This is the one panel that comes from a data measurement rather than a training or loader run.
- **L5 training samples/s:** `num_samples` / elapsed time between step 10 and step 99 of A.
- **Spatial vs data parallel samples/s:** samples per step / mean step, with 4 samples per step for data parallel and 1 for spatial parallel, from `experiments/others/sp_vs_dp_l5.json`.

## Notes
- **L6 GPU peak:** PyTorch's own peak counter (`max_allocated`) reports 21.0 GiB in C. The 17.1 GiB in `meeting_numbers.csv` and the deck is our profiler's sampled value, which misses short spikes. At L5 and L7 the two sources agree (7.5 and 59.6 GiB). None of these plots shows the L6 GPU peak.
- **Node memory:** the earlier "node used" figures (409 / 486 / 630 GiB) counted CPU plus GPU memory. CPU memory alone peaked at 318–331 GiB.
- **Not measured, so not plotted:**
  - whether more than one worker per GPU fits at L7;
  - the SP7 step time;
  - anything at L8.
