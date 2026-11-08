#!/usr/bin/env python3
"""Storage stress test of the data loader's read stage: P processes read samples as training
does (the sampler's source and target windows of every stream, with shuffle, NaN filter and
normalization), but do no tokenization and build no batches. One level per value of --procs,
each for --seconds, so one job shows how the read time per sample changes with concurrency.

Per sample it records wall seconds, CPU seconds of the process (all its threads), decoded
bytes, and /proc/self/io rchar and read_bytes. The node's CPU use and used memory are sampled
every 0.5 s. Wall seconds rising with P while CPU seconds stay flat means waiting on storage;
CPU seconds rising, or the node CPU near 100%, means CPU contention.

Every sample is a distinct time window (no window is read twice in one run), spaced by
--stride time steps so that windows do not overlap. A level is skipped if its estimated memory
(per-process cost measured in the earlier levels) would exceed --mem-fraction of the node's
available memory.

  python scripts/read_stress.py --base-config BASE [--config C ...] --out DIR \
      --procs 1 16 64 128 192 256 --seconds 90
"""

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from weathergen.common import config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.train.trainer import Trainer
from weathergen.train.utils import TRAIN, cfg_keys_to_filter, filter_config_by_enabled

GIB = 2**30


class StageTimer:
    """Stands in for SampleProfiler in collect_datasources: seconds per stage."""

    def __init__(self):
        self.seconds = defaultdict(float)

    @contextlib.contextmanager
    def measure(self, stream, stage):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[stage] += time.perf_counter() - start


def proc_io() -> dict[str, int]:
    """rchar (all read syscalls, page cache included) and read_bytes (block layer)."""
    values = {}
    with open("/proc/self/io") as f:
        for line in f:
            key, value = line.split(":")
            values[key] = int(value)
    return values


def private_bytes() -> int:
    """Memory this process does not share with its parent (pages it wrote or loaded)."""
    total = 0
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            if line.startswith(("Private_Clean:", "Private_Dirty:")):
                total += int(line.split()[1]) * 1024
    return total


def nbytes(rdatas) -> int:
    total = 0
    for rdata in rdatas:
        for value in vars(rdata).values():
            if isinstance(value, np.ndarray):  # noqa: TID251 (runtime type check)
                total += value.nbytes
    return total


def reader(sampler, indices, counter, limit, deadline, out_file, seed, forecast_steps, steps_in):
    """One stress process: read distinct samples until the deadline or until the shared counter
    reaches limit, one JSON line each."""
    sampler.rng = np.random.default_rng(seed)
    with open(out_file, "w") as out:
        while time.monotonic() < deadline:
            with counter.get_lock():
                k = counter.value
                if k >= limit:
                    break
                counter.value += 1
            idx = int(indices[k])
            timer = StageTimer()
            io_before, cpu_before = proc_io(), time.process_time()
            start = time.perf_counter()
            windows = []  # hold the whole sample, as a loader worker does
            for name, stream in sampler.streams_datasets.items():
                inputs, outputs = sampler._get_data_windows(
                    idx, forecast_steps, steps_in, stream.readers, timer, name
                )
                windows += inputs + outputs
            end = time.perf_counter()
            read_bytes, private = nbytes(windows), private_bytes()
            del windows
            io_after = proc_io()
            record = {
                "idx": idx,
                "start": start,
                "end": end,
                "wall_s": end - start,
                "cpu_s": time.process_time() - cpu_before,
                "read_s": timer.seconds["read"],
                "normalize_s": timer.seconds["normalize"],
                "decoded_bytes": read_bytes,
                "rchar": io_after["rchar"] - io_before["rchar"],
                "read_bytes": io_after["read_bytes"] - io_before["read_bytes"],
                "private_bytes": private,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()


def node_sampler(stop: threading.Event, samples: list) -> None:
    """Node CPU busy and iowait fractions and used memory, every 0.5 s."""

    def cpu_times():
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        idle, iowait = v[3], v[4]
        return sum(v), idle, iowait

    def used_bytes():
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, value = line.split(":")
                info[key] = int(value.split()[0]) * 1024
        return info["MemTotal"] - info["MemAvailable"]

    previous = cpu_times()
    while not stop.wait(0.5):
        now = cpu_times()
        total = max(now[0] - previous[0], 1)
        samples.append(
            {
                "t": time.perf_counter(),
                "busy": 1 - (now[1] - previous[1] + now[2] - previous[2]) / total,
                "iowait": (now[2] - previous[2]) / total,
                "used": used_bytes(),
            }
        )
        previous = now


def mem_available() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def summarize(procs, records, node, level_start, level_end, baseline_used) -> dict:
    done = records
    elapsed = max(r["end"] for r in done) - level_start if done else 0.0
    window = [s for s in node if level_start <= s["t"] <= level_end]

    def med(key):
        return statistics.median(r[key] for r in done) if done else None

    def p90(key):
        return float(np.percentile([r[key] for r in done], 90)) if done else None

    peak_used = max((s["used"] for s in window), default=baseline_used)
    return {
        "procs": procs,
        "samples": len(done),
        "elapsed_s": elapsed,
        "samples_per_s": len(done) / elapsed if elapsed else None,
        "decoded_gb_per_s": sum(r["decoded_bytes"] for r in done) / elapsed / 1e9
        if elapsed
        else None,
        "rchar_gb_per_s": sum(r["rchar"] for r in done) / elapsed / 1e9 if elapsed else None,
        "read_bytes_gb_per_s": sum(r["read_bytes"] for r in done) / elapsed / 1e9
        if elapsed
        else None,
        "decoded_gb_per_sample": med("decoded_bytes") / 1e9 if done else None,
        "wall_s_median": med("wall_s"),
        "wall_s_p90": p90("wall_s"),
        "read_s_median": med("read_s"),
        "cpu_s_median": med("cpu_s"),
        "wall_over_cpu": med("wall_s") / med("cpu_s") if done and med("cpu_s") else None,
        "node_cpu_busy_mean": statistics.mean(s["busy"] for s in window) if window else None,
        "node_iowait_mean": statistics.mean(s["iowait"] for s in window) if window else None,
        "node_used_peak_gib": peak_used / GIB,
        "mem_per_proc_gib": (peak_used - baseline_used) / procs / GIB,
        "private_gib_p95": float(np.percentile([r["private_bytes"] for r in done], 95)) / GIB
        if done
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--config", type=Path, nargs="*", default=[])
    parser.add_argument("--options", nargs="*", default=[], help="config overrides, key=value")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--procs", type=int, nargs="+", default=[1, 16, 64, 128, 192, 256])
    parser.add_argument("--seconds", type=float, default=90.0, help="per level")
    parser.add_argument("--stride", type=int, default=6, help="time steps between windows")
    parser.add_argument("--max-samples", type=int, default=1500, help="per level")
    parser.add_argument("--mem-fraction", type=float, default=0.7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cf = config.load_merge_configs(
        None, None, None, args.base_config, *args.config, config.from_cli_arglist(args.options)
    )
    cf = Trainer.init_ddp(cf)
    cf.streams = config.load_streams(Path(cf.streams_directory))
    mode_cfg = filter_config_by_enabled(cf.training_config, cfg_keys_to_filter)
    sampler = MultiStreamDataSampler(cf, mode_cfg, stage=TRAIN)
    perms, forecast_steps = sampler.reset()
    num_forecast_steps = int(forecast_steps[0])
    steps_in = max(sc.get("num_steps_input", 1) for sc in mode_cfg.get("model_input").values())
    # distinct, non-overlapping windows in random order
    indices = np.unique(np.asarray(perms))
    indices = indices[indices % args.stride == 0]
    indices = np.random.default_rng(0).permutation(indices)
    print(  # noqa: T201
        f"{len(indices)} windows (stride {args.stride}), {num_forecast_steps} forecast steps, "
        f"{steps_in} input steps, {len(os.sched_getaffinity(0))} CPUs, "
        f"{mem_available() / GIB:.0f} GiB available",
        flush=True,
    )

    ctx = mp.get_context("fork")
    counter = ctx.Value("q", 0)

    def run_level(procs, limit, seconds, level_dir):
        """Fork procs readers (as the DataLoader forks its workers) and wait for them."""
        level_dir.mkdir(exist_ok=True)
        deadline = time.monotonic() + seconds
        workers = [
            ctx.Process(
                target=reader,
                args=(
                    sampler,
                    indices,
                    counter,
                    limit,
                    deadline,
                    level_dir / f"proc{i:03d}.jsonl",
                    1000 * procs + i,
                    num_forecast_steps,
                    steps_in,
                ),
            )
            for i in range(procs)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        records = []
        for f in level_dir.glob("proc*.jsonl"):
            records += [json.loads(line) for line in f.read_text().splitlines() if line]
        return records, sorted({w.exitcode for w in workers})

    # warm-up: one sample in one forked reader, kept out of every level
    warm, _ = run_level(1, 1, 1e9, args.out / "warmup")
    print(f"warm-up sample: {warm[0]['wall_s']:.1f} s", flush=True)  # noqa: T201

    node, stop = [], threading.Event()
    monitor = threading.Thread(target=node_sampler, args=(stop, node), daemon=True)
    monitor.start()
    results, mem_per_proc = [], None
    for procs in args.procs:
        time.sleep(3)
        baseline_used = node[-1]["used"] if node else 0
        if mem_per_proc is not None:
            need = procs * mem_per_proc * 1.3
            if need > args.mem_fraction * mem_available():
                print(  # noqa: T201
                    f"skip {procs} procs: ~{need / GIB:.0f} GiB needed, "
                    f"{mem_available() / GIB:.0f} GiB available",
                    flush=True,
                )
                results.append({"procs": procs, "skipped": "memory"})
                continue
        if counter.value >= len(indices):
            print("all windows read; stopping", flush=True)  # noqa: T201
            break
        limit = min(counter.value + args.max_samples, len(indices))
        level_start = time.perf_counter()
        records, exitcodes = run_level(procs, limit, args.seconds, args.out / f"p{procs:03d}")
        summary = summarize(procs, records, node, level_start, time.perf_counter(), baseline_used)
        summary["exitcodes"] = exitcodes
        results.append(summary)
        if summary["samples"]:
            mem_per_proc = max(mem_per_proc or 0.0, summary["private_gib_p95"] * GIB)
        print(json.dumps(summary), flush=True)  # noqa: T201
    stop.set()
    (args.out / "summary.json").write_text(json.dumps(results, indent=1))
    (args.out / "node.jsonl").write_text("\n".join(json.dumps(s) for s in node))
    header = (
        "procs samples/s decodedGB/s rcharGB/s wall_med wall_p90 cpu_med wall/cpu cpu% private_GiB"
    )
    print(header)  # noqa: T201
    for r in results:
        if "skipped" in r:
            print(f"{r['procs']:5d} skipped ({r['skipped']})")  # noqa: T201
            continue
        print(  # noqa: T201
            f"{r['procs']:5d} {r['samples_per_s'] or 0:9.2f} {r['decoded_gb_per_s'] or 0:11.2f} "
            f"{r['rchar_gb_per_s'] or 0:9.2f} {r['wall_s_median'] or 0:8.2f} "
            f"{r['wall_s_p90'] or 0:8.2f} {r['cpu_s_median'] or 0:7.2f} "
            f"{r['wall_over_cpu'] or 0:8.2f} {100 * (r['node_cpu_busy_mean'] or 0):4.0f} "
            f"{r['private_gib_p95'] or 0:8.2f}"
        )


if __name__ == "__main__":
    main()
