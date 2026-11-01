#!/usr/bin/env python3
"""Summarize loader profiles and compare runs without requiring the WG environment.

Example:
    python scripts/analyze_dataloader_profile.py --run experiments/martin/newrun \
        --baseline experiments/martin/baseline --output experiments/martin/newrun/plots

Only PNG creation needs matplotlib. ``--no-plots`` needs Python's standard library.
"""

import argparse
import bisect
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GIB = 1024**3


def read_json_lines(path):
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                sys.stderr.write(f"Warning: ignoring incomplete last record in {path}\n")
            else:
                raise


def load_run(directory, stage="train"):
    directory = Path(directory)
    samples = []
    for path in sorted(directory.rglob("samples_rank*_worker*.jsonl")):
        samples.extend(
            record
            for record in read_json_lines(path)
            if record.get("stage", "").lower().split(".")[-1] == stage
        )
    memory = defaultdict(list)
    for path in sorted(directory.rglob("memory_rank*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("timestamp"):
                    continue
                parsed = dict(row)
                for field in (
                    "timestamp",
                    "rank_rss_bytes",
                    "workers_rss_bytes",
                    "total_rss_bytes",
                    "gpu_allocated_bytes",
                    "gpu_reserved_bytes",
                ):
                    parsed[field] = float(row[field]) if row.get(field) else None
                parsed["step"] = int(row["step"])
                key = (row.get("hostname", "unknown"), int(row["rank"]))
                memory[key].append(parsed)
    for rows in memory.values():
        rows.sort(key=lambda row: row["timestamp"])
    metrics_paths = sorted(directory.glob("*train_metrics.json"))
    if len(metrics_paths) > 1:
        raise ValueError(f"Multiple training metric files in {directory}; select one run directory")
    metrics = []
    if metrics_paths:
        metrics = [
            row
            for row in read_json_lines(metrics_paths[0])
            if row.get("stage") == stage and "elapsed_training_time_seconds" in row
        ]
    return {"name": directory.name, "samples": samples, "memory": dict(memory), "metrics": metrics}


def training_summary(metrics, stall_seconds=5.0):
    elapsed = [float(row["elapsed_training_time_seconds"]) for row in metrics]
    intervals = [b - a for a, b in zip(elapsed, elapsed[1:], strict=False)]
    losses = [float(row["loss_avg_mean"]) for row in metrics if "loss_avg_mean" in row]
    peaks = [
        float(row["performance.memory.train.max_allocated_gib"])
        for row in metrics
        if "performance.memory.train.max_allocated_gib" in row
    ]
    return {
        "steps": len(metrics),
        "median_step_seconds": statistics.median(intervals) if intervals else None,
        "mean_step_seconds": statistics.mean(intervals) if intervals else None,
        "mean_after_10_steps_seconds": statistics.mean(intervals[9:])
        if len(intervals) > 9
        else None,
        "stall_threshold_seconds": stall_seconds,
        "stalls": sum(value > stall_seconds for value in intervals),
        "largest_interval_seconds": max(intervals, default=None),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "training_gpu_peak_gib": max(peaks, default=None),
    }


def align_memory(memory, max_gap_seconds=2.0):
    """Hold the most recent sample only while fresh, requiring every rank.

    Never sum each rank's independent peak or interpolate across an unsampled
    gap. Returns synchronized samples; totals remain approximate, sampled RSS.
    """
    if not memory:
        return []
    sequences = [rows for rows in memory.values() if rows]
    if len(sequences) != len(memory):
        return []
    timestamps = [[row["timestamp"] for row in rows] for rows in sequences]
    grid = sorted({value for times in timestamps for value in times})
    result = []
    for timestamp in grid:
        selected = []
        for rows, times in zip(sequences, timestamps, strict=True):
            index = bisect.bisect_right(times, timestamp) - 1
            if index < 0 or timestamp - times[index] > max_gap_seconds:
                break
            selected.append(rows[index])
        if len(selected) != len(sequences):
            continue
        gpu_allocated = [row["gpu_allocated_bytes"] for row in selected]
        gpu_reserved = [row["gpu_reserved_bytes"] for row in selected]
        result.append(
            {
                "timestamp": timestamp,
                "rank_rss_bytes": sum(row["rank_rss_bytes"] for row in selected),
                "workers_rss_bytes": sum(row["workers_rss_bytes"] for row in selected),
                "total_rss_bytes": sum(row["total_rss_bytes"] for row in selected),
                "gpu_allocated_bytes": max(gpu_allocated) if None not in gpu_allocated else None,
                "gpu_reserved_bytes": max(gpu_reserved) if None not in gpu_reserved else None,
            }
        )
    return result


def summarize(run, stall_seconds=5.0, max_gap_seconds=2.0):
    streams = defaultdict(lambda: defaultdict(list))
    workers = defaultdict(list)
    for sample in run["samples"]:
        workers[
            f"{sample.get('hostname', 'unknown')}/rank{sample['rank']}/worker{sample['worker_id']}"
        ].append(sample)
        for stream, values in sample.get("streams", {}).items():
            for key, value in values.items():
                if isinstance(value, int | float):
                    streams[stream][key].append(value)
    stream_summary = {
        name: {key: statistics.mean(values) for key, values in fields.items()}
        for name, fields in streams.items()
    }
    slowest = max(
        stream_summary,
        key=lambda name: sum(
            value for key, value in stream_summary[name].items() if key.endswith("_seconds")
        ),
        default=None,
    )
    aligned = align_memory(run["memory"], max_gap_seconds)
    result = {
        "run": run["name"],
        "attempted_samples": len(run["samples"]),
        "invalid_samples": sum(sample.get("valid") is False for sample in run["samples"]),
        "slowest_stream_by_mean_stage_time": slowest,
        "mean_per_stream": stream_summary,
        "workers": {
            key: {
                "samples": len(samples),
                "mean_total_seconds": statistics.mean(s["total_seconds"] for s in samples),
                "max_observed_rss_gib": max(
                    max(s["rss_before_bytes"], s["rss_after_bytes"]) for s in samples
                )
                / GIB,
            }
            for key, samples in workers.items()
        },
        "training": training_summary(run["metrics"], stall_seconds),
        "sampled_sum_rss_peak_gib": max(
            (row["total_rss_bytes"] / GIB for row in aligned), default=None
        ),
        "memory_rank_count": len(run["memory"]),
        "memory_note": (
            "Sum of process RSS; shared pages may be counted repeatedly. Sampled, not exact peak."
        ),
    }
    return result


def print_summary(summary):
    sys.stdout.write(f"\nRun: {summary['run']}\n")
    sys.stdout.write(
        f"Attempted samples: {summary['attempted_samples']}; "
        f"invalid: {summary['invalid_samples']}\n"
    )
    sys.stdout.write("Per-stream means per attempted sample (seconds; MB is decimal):\n")
    for name, fields in summary["mean_per_stream"].items():
        values = ", ".join(f"{key}={value:.3f}" for key, value in sorted(fields.items()))
        sys.stdout.write(f"  {name}: {values}\n")
    sys.stdout.write(f"Slowest stream: {summary['slowest_stream_by_mean_stage_time']}\n")
    for worker, fields in summary["workers"].items():
        sys.stdout.write(
            f"  {worker}: {fields['samples']} samples, "
            f"{fields['mean_total_seconds']:.3f} s/sample, "
            f"observed RSS up to {fields['max_observed_rss_gib']:.3f} GiB\n"
        )
    sys.stdout.write(f"Training: {json.dumps(summary['training'])}\n")
    sys.stdout.write(f"Sampled summed RSS peak: {summary['sampled_sum_rss_peak_gib']} GiB\n")
    sys.stdout.write(summary["memory_note"] + "\n")


def make_plots(run, baseline, output, max_gap_seconds=2.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = [run] if baseline is None else [baseline, run]
    colours = ["#0072B2", "#D55E00"]

    def save(fig, filename):
        for ax in fig.axes:
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)

    def memory_line(ax, rows, field, origin, label, **kwargs):
        rows = [row for row in rows if row.get(field) is not None]
        if not rows:
            return
        x = [row["timestamp"] - origin for row in rows]
        y = [row[field] / GIB for row in rows]
        ax.plot(x, y, label=label, **kwargs)
        peak = max(range(len(y)), key=y.__getitem__)
        ax.scatter([x[peak]], [y[peak]], s=20, color=kwargs.get("color"))
        ax.annotate(
            f"peak {y[peak]:.2f} GiB",
            (x[peak], y[peak]),
            xytext=(4, 7),
            textcoords="offset points",
            fontsize=8,
        )

    for current in runs:
        memory = current["memory"]
        if not memory:
            continue
        origin = min(rows[0]["timestamp"] for rows in memory.values() if rows)
        aligned = align_memory(memory, max_gap_seconds)
        fig, ax = plt.subplots(figsize=(10, 5))
        for field, label, colour in (
            ("total_rss_bytes", "Ranks + workers (summed RSS)", "#0072B2"),
            ("rank_rss_bytes", "Ranks", "#009E73"),
            ("workers_rss_bytes", "Workers", "#D55E00"),
        ):
            memory_line(ax, aligned, field, origin, label, color=colour)
        for (host, rank), rows in memory.items():
            ax.plot(
                [r["timestamp"] - origin for r in rows],
                [r["rank_rss_bytes"] / GIB for r in rows],
                linewidth=0.8,
                alpha=0.6,
                label=f"{host} rank {rank}",
            )
        ax.set(
            xlabel="Time since first memory sample (s)",
            ylabel="Host memory (GiB)",
            title=f"{current['name']}: summed RSS (shared pages may be counted twice)",
        )
        ax.legend(fontsize=8)
        save(fig, f"{current['name']}_host_memory.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        for (host, rank), rows in memory.items():
            for field, style, label in (
                ("gpu_allocated_bytes", "-", "allocated"),
                ("gpu_reserved_bytes", "--", "reserved"),
            ):
                memory_line(ax, rows, field, origin, f"{host} rank {rank} {label}", linestyle=style)
        ax.set(
            xlabel="Time since first memory sample (s)",
            ylabel="GPU memory (GiB)",
            title=f"{current['name']}: sampled CUDA allocator memory",
        )
        if ax.lines:
            ax.legend(fontsize=8)
        save(fig, f"{current['name']}_gpu_memory.png")

    if baseline is not None:
        for field, filename, label in (
            ("total_rss_bytes", "comparison_host_memory.png", "Summed host RSS (GiB)"),
            ("gpu_allocated_bytes", "comparison_gpu_memory.png", "Maximum rank GPU memory (GiB)"),
        ):
            fig, ax = plt.subplots(figsize=(10, 5))
            peaks = []
            for current, colour in zip(runs, colours, strict=True):
                aligned = align_memory(current["memory"], max_gap_seconds)
                if not aligned:
                    continue
                origin = min(rows[0]["timestamp"] for rows in current["memory"].values())
                memory_line(ax, aligned, field, origin, current["name"], color=colour)
                valid = [row[field] / GIB for row in aligned if row[field] is not None]
                peaks.append(max(valid) if valid else math.nan)
                if field == "gpu_allocated_bytes":
                    memory_line(
                        ax,
                        aligned,
                        "gpu_reserved_bytes",
                        origin,
                        current["name"] + " reserved",
                        color=colour,
                        linestyle="--",
                    )
            title = "Baseline versus candidate (sampled peaks)"
            if len(peaks) == 2:
                title += f": peak {peaks[0]:.2f} → {peaks[1]:.2f} GiB"
            ax.set(
                xlabel="Time since each run's first memory sample (s)", ylabel=label, title=title
            )
            if ax.lines:
                ax.legend()
            save(fig, filename)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for current, colour in zip(runs, colours, strict=False):
        metrics = current["metrics"]
        elapsed = [row["elapsed_training_time_seconds"] for row in metrics]
        axes[0].plot(
            range(1, len(elapsed)),
            [b - a for a, b in zip(elapsed, elapsed[1:], strict=False)],
            label=current["name"],
            color=colour,
        )
        losses = [
            (i, row["loss_avg_mean"]) for i, row in enumerate(metrics) if "loss_avg_mean" in row
        ]
        axes[1].plot(
            [i for i, _ in losses],
            [loss for _, loss in losses],
            label=current["name"],
            color=colour,
        )
    axes[0].set(ylabel="Step interval (s)", title="Training step intervals and loss")
    axes[1].set(xlabel="Training step (zero-based)", ylabel="Training loss")
    axes[0].legend()
    save(fig, "step_time_and_loss.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="One copied run-results directory")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, help="Default: --run/plots")
    parser.add_argument("--stage", default="train", choices=("train", "val", "test"))
    parser.add_argument("--stall-seconds", type=float, default=5.0)
    parser.add_argument("--max-memory-gap-seconds", type=float, default=2.0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if not args.run.is_dir() or (args.baseline is not None and not args.baseline.is_dir()):
        parser.error("run and baseline must be existing directories")
    run = load_run(args.run, args.stage)
    baseline = load_run(args.baseline, args.stage) if args.baseline else None
    summaries = [
        summarize(item, args.stall_seconds, args.max_memory_gap_seconds)
        for item in ([baseline, run] if baseline else [run])
    ]
    output = args.output or args.run / "plots"
    output.mkdir(parents=True, exist_ok=True)
    for summary in summaries:
        print_summary(summary)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    if not args.no_plots:
        make_plots(run, baseline, output, args.max_memory_gap_seconds)


if __name__ == "__main__":
    main()
