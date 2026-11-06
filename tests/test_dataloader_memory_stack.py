"""Synthetic-profile checks for the stacked host-memory plot."""

import csv
import importlib.util
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "plot_stack", ROOT / "scripts/plot_dataloader_memory_stack.py"
)
plot_stack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_stack)

GIB = 1024**3
FIELDS = (
    "schema_version",
    "timestamp",
    "elapsed_seconds",
    "hostname",
    "rank",
    "pid",
    "stage",
    "mini_epoch",
    "step",
    "rank_rss_bytes",
    "worker_rss_bytes",
    "workers_rss_bytes",
    "total_rss_bytes",
    "gpu_allocated_bytes",
    "gpu_reserved_bytes",
)


def write_profile(directory, ranks=4, workers=12, steps=100, seconds_per_step=3.0, seed=0):
    """Profile files shaped like dataloader_profile.py output, with sawtooth worker memory."""
    rng = random.Random(seed)
    profile = Path(directory) / "dataloader_profile"
    profile.mkdir(parents=True)
    t0 = 1_700_000_000.0
    duration = steps * seconds_per_step
    for rank in range(ranks):
        with (profile / f"memory_rank{rank:04d}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            t = 0.0
            while t < duration:
                step = int(t / seconds_per_step)
                pids = {
                    str(1000 * (rank + 1) + w): (1.2 + 0.8 * math.sin(t / 7 + w)) * GIB
                    for w in range(workers)
                    if t < duration - (w * 4)
                }
                rank_rss = (6 + 0.5 * math.sin(t / 30)) * GIB
                gpu = (5 + 20 * abs(math.sin(t * math.pi / seconds_per_step))) * GIB
                writer.writerow(
                    {
                        "schema_version": 1,
                        "timestamp": t0 + t + rank * 0.07,
                        "elapsed_seconds": t,
                        "hostname": "jpbo-001",
                        "rank": rank,
                        "pid": 10 + rank,
                        "stage": "train",
                        "mini_epoch": 0,
                        "step": step,
                        "rank_rss_bytes": int(rank_rss),
                        "worker_rss_bytes": json.dumps({k: int(v) for k, v in pids.items()}),
                        "workers_rss_bytes": int(sum(pids.values())),
                        "total_rss_bytes": int(rank_rss + sum(pids.values())),
                        "gpu_allocated_bytes": int(gpu),
                        "gpu_reserved_bytes": int(gpu * 1.15),
                    }
                )
                t += 0.5
        with (profile / f"loader_wait_rank{rank:04d}.jsonl").open("w") as handle:
            for step in range(steps):
                wait = 37.0 if step >= steps - 3 else rng.expovariate(1 / 0.2)
                handle.write(
                    json.dumps(
                        {
                            "timestamp": t0 + step * seconds_per_step,
                            "rank": rank,
                            "stage": "train",
                            "mini_epoch": 0,
                            "step": step,
                            "wait_seconds": wait,
                        }
                    )
                    + "\n"
                )
        for worker in range(workers):
            path = profile / f"samples_rank{rank:04d}_worker{worker:03d}.jsonl"
            with path.open("w") as handle:
                for sample in range(8):
                    streams = {
                        name: {
                            "read_seconds": rng.uniform(0.5, 1.5),
                            "normalize_seconds": rng.uniform(0.05, 0.1),
                            "tokenize_seconds": rng.uniform(0.3, 0.8),
                            "build_seconds": rng.uniform(0.2, 0.5),
                        }
                        for name in ("ERA5", "SurfaceCombined")
                    }
                    handle.write(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "stage": "train",
                                "sample_id": str(sample),
                                "rank": rank,
                                "worker_id": worker,
                                "valid": True,
                                "error": None,
                                "total_seconds": 3.0,
                                "streams": streams,
                            }
                        )
                        + "\n"
                    )


def test_layers_cover_every_process_and_drop_exited_workers(tmp_path):
    write_profile(tmp_path, ranks=2, workers=3, steps=10)
    run = plot_stack.analyze.load_run(tmp_path)
    times, layers, steps, _origin = plot_stack.node_layers(run["memory"], "jpbo-001")
    assert [kind for _, kind, _ in layers] == ["rank"] * 2 + ["worker"] * 6
    assert all(len(values) == len(times) for _, _, values in layers)
    # worker 2 exits 8 s early: its band must fall to zero at the end, not stay frozen
    assert layers[-1][2][-1] == 0.0 and max(layers[-1][2]) > 0
    assert max(steps.values()) == 9


def test_plots_written_with_baseline(tmp_path):
    write_profile(tmp_path / "run")
    write_profile(tmp_path / "base", seed=1)
    out = tmp_path / "plots"
    plot_stack.main(
        ["--run", str(tmp_path / "run"), "--baseline", str(tmp_path / "base"), "--output", str(out)]
    )
    assert (out / "run_jpbo-001_memory_stack.png").stat().st_size > 10_000
    assert (out / "stage_breakdown.png").stat().st_size > 5_000


def test_stage_medians_sum_streams():
    samples = [{"streams": {"a": {"read_seconds": 1.0}, "b": {"read_seconds": 2.0}}}]
    assert plot_stack.stage_medians(samples)["read"] == 3.0
