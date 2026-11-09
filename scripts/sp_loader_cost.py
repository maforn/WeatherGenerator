#!/usr/bin/env python3
"""Loader work per sample for each rank of a spatial-parallel group against data parallel.

For every level in --levels, runs one process as data parallel (spatial size 1) and --size
processes as the ranks of one spatial group, all at once on this node, each building --batches
batches (after one warm-up batch) in-process: no DataLoader workers, no GPU, one torch thread
(as a DataLoader worker). The ranks of a spatial group draw the same samples as the data-parallel
process (same seed, data-parallel rank 0). The sampler's own per-sample profiling
(data_loading.profile_timing) records read / normalize / tokenize / build seconds per stream and
the payload bytes; each process also records seconds per batch and its peak RSS.

Results: OUT/l<level>/sp<size>_rank<r>/{samples_rank0000_worker-01.jsonl,summary.json} and
OUT/summary.json (per level and process: median seconds per sample and per stage, peak RSS).

  python scripts/sp_loader_cost.py --base-config BASE [--config C ...] --out DIR \
      [--levels 5 7] [--size 4] [--batches 3]
"""

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

GIB = 2**30


def run_one(args) -> None:
    """One process: data parallel (size 1) or rank `rank` of a spatial group of `size`."""
    import torch

    import weathergen.utils.distributed as distributed
    from weathergen.common import config
    from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
    from weathergen.train.trainer import Trainer
    from weathergen.train.utils import TRAIN, cfg_keys_to_filter, filter_config_by_enabled

    torch.set_num_threads(1)
    out = args.out / f"l{args.level}" / f"sp{args.size}_rank{args.rank}"
    out.mkdir(parents=True, exist_ok=True)
    options = [
        *args.options,
        f"healpix_level={args.level}",
        f"distributed.spatial_parallel.size={args.size}",
        "data_loading.profile_timing=True",
        "data_loading.profile_payload_bytes=True",
        f"data_loading.profile_output_dir={out}",
    ]
    cf = config.load_merge_configs(
        None, None, None, args.base_config, *args.config, config.from_cli_arglist(options)
    )
    cf = Trainer.init_ddp(cf)
    # one data-parallel group of `size` ranks; this process is spatial rank `rank`. Without a
    # process group get_world_size() is 1, which the spatial-size check would reject.
    cf.rank = args.rank
    cf.world_size = args.size
    distributed.get_world_size = lambda: args.size
    cf.streams = config.load_streams(Path(cf.streams_directory))
    mode_cfg = filter_config_by_enabled(cf.training_config, cfg_keys_to_filter)
    sampler = MultiStreamDataSampler(cf, mode_cfg, stage=TRAIN)
    sp = sampler.spatial_parallel
    batches = iter(sampler)
    seconds = []
    for i in range(args.batches + 1):
        started = time.perf_counter()
        next(batches)
        seconds.append(time.perf_counter() - started)
        print(  # noqa: T201
            f"l{args.level} sp{args.size} rank{args.rank} batch {i}: {seconds[-1]:.1f} s",
            flush=True,
        )
    summary = {
        "level": args.level,
        "size": args.size,
        "rank": args.rank,
        "cell_start": sp.cell_start,
        "cell_end": sp.cell_end,
        "seconds_per_batch": seconds,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


def collect(out: Path) -> list:
    """Median stage seconds per sample (warm-up sample left out) for every process."""
    rows = []
    for summary_path in sorted(out.glob("l*/sp*_rank*/summary.json")):
        summary = json.loads(summary_path.read_text())
        records = []
        for path in summary_path.parent.glob("samples_rank*_worker*.jsonl"):
            records += [json.loads(line) for line in path.read_text().splitlines() if line]
        records = sorted(records, key=lambda r: r["timestamp"])[1:]
        stages = defaultdict(list)
        for record in records:
            per_stage = defaultdict(float)
            for values in record["streams"].values():
                for key, value in values.items():
                    if key.endswith("_seconds"):
                        per_stage[key.removesuffix("_seconds")] += value
            for stage, value in per_stage.items():
                stages[stage].append(value)
            stages["total"].append(record["total_seconds"])
            payload = record.get("payload") or {}
            if payload:
                nbytes = payload.get("torch_bytes", 0) + payload.get("numpy_bytes", 0)
                stages["payload_gib"].append(nbytes / GIB)
        rows.append(
            {
                **{k: summary[k] for k in ("level", "size", "rank", "cell_start", "cell_end")},
                "samples": [r["sample_id"] for r in records],
                "peak_rss_gib": summary["peak_rss_gib"],
                **{f"{k}_median": statistics.median(v) for k, v in stages.items() if v},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--config", type=Path, nargs="*", default=[])
    parser.add_argument("--options", nargs="*", default=[], help="config overrides, key=value")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[5, 7])
    parser.add_argument("--size", type=int, default=4, help="spatial-parallel group size")
    parser.add_argument("--batches", type=int, default=3, help="batches after one warm-up")
    parser.add_argument("--level", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.rank is not None:
        run_one(args)
        return

    common = [sys.executable, "-u", __file__, "--out", str(args.out)]
    common += ["--batches", str(args.batches)]
    if args.base_config:
        common += ["--base-config", str(args.base_config)]
    if args.config:
        common += ["--config", *map(str, args.config)]
    if args.options:
        common += ["--options", *args.options]
    procs = []
    for level in args.levels:
        for size, rank in [(1, 0)] + [(args.size, r) for r in range(args.size)]:
            cmd = [*common, "--level", str(level), "--size", str(size), "--rank", str(rank)]
            procs.append(subprocess.Popen(cmd, env=os.environ.copy()))
    codes = [p.wait() for p in procs]
    rows = collect(args.out)
    (args.out / "summary.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(  # noqa: T201
            f"l{r['level']} sp{r['size']} rank{r['rank']} cells [{r['cell_start']}, "
            f"{r['cell_end']}): total {r.get('total_median', 0):.1f} s, "
            + ", ".join(
                f"{k.removesuffix('_median')} {v:.2f}"
                for k, v in r.items()
                if k.endswith("_median") and k != "total_median"
            )
            + f", peak RSS {r['peak_rss_gib']:.1f} GiB, samples {r['samples']}"
        )
    if any(codes):
        raise SystemExit(f"exit codes {codes}")


if __name__ == "__main__":
    main()
