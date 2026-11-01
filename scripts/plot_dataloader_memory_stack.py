#!/usr/bin/env python3
"""Stacked host-memory timeline per node, in the spirit of the PyTorch memory_viz view.

Each band is one process: training ranks (blue) and their DataLoader workers (orange), so
the top edge is the node's summed RSS. Below it, on the same time axis: CUDA memory and the
time the training loop waited for each batch. A second figure splits per-sample preparation
time into read / normalize / tokenize / build, optionally against a baseline run.

Example:
    python scripts/plot_dataloader_memory_stack.py --run <run_dir> [--baseline <run_dir>] \
        --output experiments/martin/<run_id>/plots

<run_dir> is any directory containing the ``dataloader_profile`` files (searched recursively).
Summed RSS counts shared pages once per process, so it overstates physical RAM.
"""

import argparse
import bisect
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GIB = 1024**3
RANK_COLOUR = "#2a78d6"
WORKER_COLOURS = ("#eb6834", "#f08a5d")  # one hue, two steps so adjacent workers separate
STAGE_COLOURS = {
    "read": "#2a78d6",
    "normalize": "#eb6834",
    "tokenize": "#1baf7a",
    "build": "#eda100",
}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _load_analyze():
    path = Path(__file__).with_name("analyze_dataloader_profile.py")
    spec = importlib.util.spec_from_file_location("analyze_dataloader_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze = _load_analyze()


def load_waits(directory, stage="train"):
    waits = []
    for path in sorted(Path(directory).rglob("loader_wait_rank*.jsonl")):
        for record in analyze.read_json_lines(path):
            if str(record.get("stage", "")).lower().split(".")[-1] == stage:
                waits.append(record)
    return waits


def node_layers(memory, host, interval=0.5, max_gap=2.0):
    """Resample every process of one node onto a shared time grid.

    Returns (times, layers, steps, origin): times are seconds since origin, layers is a list of
    (label, kind, GiB values) and steps maps absolute grid time to the first rank's step.
    A process contributes 0 outside its sampled span or across gaps longer than ``max_gap``
    (worker exited / not yet started).
    """
    series = {}  # (kind, rank, pid) -> (timestamps, bytes)
    rank_rows = {rank: rows for (h, rank), rows in memory.items() if h == host and rows}
    for rank, rows in sorted(rank_rows.items()):
        stamps = [row["timestamp"] for row in rows]
        series[("rank", rank, "")] = (stamps, [row["rank_rss_bytes"] or 0.0 for row in rows])
        per_worker = defaultdict(lambda: ([], []))
        for row in rows:
            for pid, rss in json.loads(row.get("worker_rss_bytes") or "{}").items():
                per_worker[pid][0].append(row["timestamp"])
                per_worker[pid][1].append(float(rss))
        for pid, values in sorted(per_worker.items(), key=lambda item: int(item[0])):
            series[("worker", rank, pid)] = values
    if not series:
        return [], [], {}, 0.0
    start = min(stamps[0] for stamps, _ in series.values())
    end = max(stamps[-1] for stamps, _ in series.values())
    count = int((end - start) / interval) + 1
    times = [start + i * interval for i in range(count)]

    def resample(stamps, values):
        out = []
        for t in times:
            i = bisect.bisect_right(stamps, t) - 1
            ok = i >= 0 and t - stamps[i] <= max_gap
            out.append(values[i] / GIB if ok else 0.0)
        return out

    layers = []
    for (kind, rank, pid), (stamps, values) in sorted(
        series.items(), key=lambda item: (item[0][0] != "rank", item[0][1], item[0][2])
    ):
        label = f"rank {rank}" if kind == "rank" else f"rank {rank} worker {pid}"
        layers.append((label, kind, resample(stamps, values)))
    steps = {}
    first_rank = min(rank_rows)
    rows = rank_rows[first_rank]
    stamps = [row["timestamp"] for row in rows]
    for t in times:
        i = bisect.bisect_right(stamps, t) - 1
        steps[t] = rows[i]["step"] if i >= 0 else -1
    return [t - start for t in times], layers, steps, start


def _style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_timeline(run, waits, host, output, interval=0.5, max_gap=2.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, layers, steps, origin = node_layers(run["memory"], host, interval, max_gap)
    if not times:
        return None
    fig, (ax_mem, ax_gpu, ax_wait) = plt.subplots(
        3, 1, figsize=(14, 8.5), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.2, 1.0]}
    )
    for ax in (ax_mem, ax_gpu, ax_wait):
        _style(ax)

    worker_index = 0
    colours = []
    for _, kind, _ in layers:
        if kind == "rank":
            colours.append(RANK_COLOUR)
        else:
            colours.append(WORKER_COLOURS[worker_index % 2])
            worker_index += 1
    ax_mem.stackplot(
        times,
        [values for _, _, values in layers],
        colors=colours,
        edgecolor="white",
        linewidth=0.35,
    )
    totals = [sum(values[i] for _, _, values in layers) for i in range(len(times))]
    peak = max(range(len(totals)), key=totals.__getitem__)
    ax_mem.annotate(
        f"peak {totals[peak]:.1f} GiB (step {steps[times[peak] + origin]})",
        (times[peak], totals[peak]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=9,
        color=INK,
    )
    ax_mem.scatter([times[peak]], [totals[peak]], s=24, color=INK, zorder=5)
    ranks = sum(kind == "rank" for _, kind, _ in layers)
    workers = len(layers) - ranks
    from matplotlib.patches import Patch

    ax_mem.legend(
        handles=[
            Patch(color=RANK_COLOUR, label=f"training ranks ({ranks})"),
            Patch(color=WORKER_COLOURS[0], label=f"loader workers ({workers})"),
        ],
        loc="upper left",
        fontsize=8,
        frameon=False,
    )
    ax_mem.set_ylabel("Host RSS (GiB)", color=INK)
    ax_mem.set_title(
        f"{run['name']} on {host}: host memory per process, stacked "
        "(summed RSS; shared pages counted per process, sampled every "
        f"{interval:g} s)",
        fontsize=10,
        color=INK,
        loc="left",
        pad=16,
    )
    ax_mem.text(
        1.0,
        1.0,
        "training step",
        transform=ax_mem.transAxes,
        fontsize=7,
        color=INK_MUTED,
        ha="right",
        va="bottom",
    )
    # step ticks along the top edge, every 10 steps
    last = None
    for t in times:
        step = steps[t + origin]
        if step >= 0 and step != last and step % 10 == 0:
            ax_mem.axvline(t, color=GRID, linewidth=0.6, zorder=0)
            ax_mem.text(
                t,
                1.0,
                f"{step}",
                transform=ax_mem.get_xaxis_transform(),
                fontsize=7,
                color=INK_MUTED,
                ha="center",
                va="bottom",
            )
        last = step

    # GPU: per grid time, the maximum across this node's ranks (allocated vs reserved)
    for field, style in (("allocated", "-"), ("reserved", "--")):
        key = f"gpu_{field}_bytes"
        per_rank = []
        for (h, _rank), rows in run["memory"].items():
            rows = [row for row in rows if h == host and row.get(key) is not None]
            if rows:
                per_rank.append(([row["timestamp"] for row in rows], [row[key] for row in rows]))
        if not per_rank:
            continue
        ys = []
        for t in times:
            values = []
            for stamps, vals in per_rank:
                i = bisect.bisect_right(stamps, t + origin) - 1
                if i >= 0 and t + origin - stamps[i] <= max_gap:
                    values.append(vals[i] / GIB)
            ys.append(max(values) if values else float("nan"))
        ax_gpu.plot(
            times, ys, style, color=RANK_COLOUR, linewidth=1.0, label=f"{field} (max over ranks)"
        )
    if ax_gpu.lines:
        ax_gpu.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            ncols=2,
            fontsize=8,
            frameon=False,
            borderaxespad=0,
        )
    else:
        ax_gpu.text(
            0.5,
            0.5,
            "no CUDA samples",
            transform=ax_gpu.transAxes,
            ha="center",
            color=INK_MUTED,
            fontsize=8,
        )
    ax_gpu.set_ylabel("GPU (GiB)", color=INK)

    host_waits = [w for w in waits if w.get("timestamp")]
    if host_waits:
        xs = [w["timestamp"] - origin for w in host_waits]
        ys = [w["wait_seconds"] for w in host_waits]
        ax_wait.vlines(xs, 0, ys, color=WORKER_COLOURS[0], linewidth=1.4)
        worst = max(range(len(ys)), key=ys.__getitem__)
        ax_wait.annotate(
            f"{ys[worst]:.1f} s",
            (xs[worst], ys[worst]),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
        total = sum(ys)
        ax_wait.set_title(
            f"training loop waiting for a batch (all ranks): total {total:.0f} s, "
            f"median {statistics.median(ys):.2f} s",
            fontsize=9,
            color=INK_MUTED,
            loc="left",
        )
    ax_wait.set_ylabel("wait (s)", color=INK)
    ax_wait.set_xlabel("seconds since first memory sample", color=INK)

    fig.tight_layout()
    path = Path(output) / f"{run['name']}_{host}_memory_stack.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def stage_medians(samples):
    """Median per-sample seconds for each stage, summed over streams."""
    per_stage = defaultdict(list)
    for sample in samples:
        if sample.get("error"):
            continue
        totals = defaultdict(float)
        for values in sample.get("streams", {}).values():
            for stage in STAGE_COLOURS:
                totals[stage] += values.get(f"{stage}_seconds", 0.0)
        for stage in STAGE_COLOURS:
            per_stage[stage].append(totals[stage])
    return {stage: statistics.median(v) if v else 0.0 for stage, v in per_stage.items()}


def plot_stages(runs, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(run["name"], stage_medians(run["samples"]), len(run["samples"])) for run in runs]
    rows = [row for row in rows if row[2]]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.7 * len(rows)))
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    for y, (_name, medians, count) in enumerate(rows):
        left = 0.0
        for stage, colour in STAGE_COLOURS.items():
            width = medians.get(stage, 0.0)
            ax.barh(
                y,
                width,
                left=left,
                height=0.5,
                color=colour,
                edgecolor="white",
                linewidth=2,
                label=stage if y == 0 else None,
            )
            if width > 0.06 * max(sum(m.values()) for _, m, _ in rows):
                ax.text(
                    left + width / 2,
                    y,
                    f"{width:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=INK,
                )
            left += width
        ax.text(left, y, f"  {left:.2f} s  (n={count})", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [name for name, _, _ in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("median seconds per sample (summed over streams)", color=INK)
    ax.set_title("Where batch preparation time goes", fontsize=10, color=INK, loc="left")
    ax.legend(ncols=4, loc="lower right", bbox_to_anchor=(1, 1.02), fontsize=8, frameon=False)
    fig.tight_layout()
    path = Path(output) / "stage_breakdown.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, help="Default: --run/plots")
    parser.add_argument("--stage", default="train", choices=("train", "val", "test"))
    parser.add_argument("--interval", type=float, default=0.5, help="Resampling grid (s)")
    args = parser.parse_args(argv)
    output = args.output or args.run / "plots"
    output.mkdir(parents=True, exist_ok=True)
    run = analyze.load_run(args.run, args.stage)
    waits = load_waits(args.run, args.stage)
    written = []
    for host in sorted({host for host, _ in run["memory"]}):
        written.append(plot_timeline(run, waits, host, output, args.interval))
    runs = [run] if args.baseline is None else [analyze.load_run(args.baseline, args.stage), run]
    written.append(plot_stages(runs, output))
    for path in written:
        if path:
            sys.stdout.write(f"{path}\n")


if __name__ == "__main__":
    main()
