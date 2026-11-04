#!/usr/bin/env python3
"""What is in a node's host memory over time: one stacked chart per run, by category.

Host-memory counterpart of the PyTorch ``memory_viz`` view (time on x, stacked contents on y).
Categories, bottom to top:

* training ranks: the 4 training processes;
* loader workers: the DataLoader worker processes, excluding /dev/shm;
* /dev/shm: finished samples handed from workers to the ranks, waiting for training.

Which memory number is used depends on what the run recorded:

* ``profile_memory_full`` runs: PSS (shared pages split between processes, no double counting).
  /dev/shm pages are mapped by the workers that wrote them, so workers = workers PSS - /dev/shm.
* older runs: RSS (shared pages counted once per process; ~10% too high at level 6). Workers
  exclude /dev/shm when the run recorded it, and include it otherwise.

A second panel shows GPU memory (max over ranks). With NUMA columns (``cpu_mem_*``), a line marks
the node's host memory in use and its capacity. Writes ``<run>_memory_view.png`` and
``<run>_memory_view.json`` (the series, for interactive pages).

Example:
    python scripts/plot_memory_view.py --run experiments/martin/<run_id> [--stage train]
"""

import argparse
import bisect
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

GIB = 1024**3
COLOURS = {"ranks": "#2a78d6", "workers": "#eb6834", "shm": "#1baf7a"}
LABELS = {
    "ranks": "training ranks",
    "workers": "loader workers (excl. /dev/shm)",
    "shm": "/dev/shm: samples waiting for training",
}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _float(value):
    return float(value) if value not in (None, "") else None


def load_memory(directory, stage="train"):
    """{(hostname, rank): rows sorted by time}, rows restricted to one stage."""
    memory = defaultdict(list)
    for path in sorted(Path(directory).rglob("memory_rank*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("timestamp") or row.get("stage", stage) != stage:
                    continue
                parsed = {
                    key: _float(value)
                    for key, value in row.items()
                    if key.endswith("bytes") and key != "worker_rss_bytes"
                }
                parsed["timestamp"] = float(row["timestamp"])
                parsed["step"] = int(row["step"])
                memory[(row.get("hostname", "unknown"), int(row["rank"]))].append(parsed)
    for rows in memory.values():
        rows.sort(key=lambda r: r["timestamp"])
    return dict(memory)


def node_view(memory, host, interval=1.0, max_gap=2.0):
    """Resample one node's ranks onto a shared grid and split memory into categories (GiB)."""
    ranks = {rank: rows for (h, rank), rows in memory.items() if h == host and rows}
    full = all(r.get("rank_pss_bytes") is not None for rows in ranks.values() for r in rows[:1])
    start = min(rows[0]["timestamp"] for rows in ranks.values())
    end = max(rows[-1]["timestamp"] for rows in ranks.values())
    times = [start + i * interval for i in range(int((end - start) / interval) + 1)]

    def at(rows, t):
        i = bisect.bisect_right([r["timestamp"] for r in rows], t) - 1
        return rows[i] if i >= 0 and t - rows[i]["timestamp"] <= max_gap else None

    stamps = {rank: [r["timestamp"] for r in rows] for rank, rows in ranks.items()}

    def at_fast(rank, t):
        i = bisect.bisect_right(stamps[rank], t) - 1
        rows = ranks[rank]
        return rows[i] if i >= 0 and t - rows[i]["timestamp"] <= max_gap else None

    rank_key = "rank_pss_bytes" if full else "rank_rss_bytes"
    worker_key = "workers_pss_bytes" if full else "workers_rss_bytes"
    view = {
        k: []
        for k in (
            "t",
            "step",
            "ranks",
            "workers",
            "shm",
            "gpu_alloc",
            "gpu_res",
            "host_used",
            "host_total",
        )
    }
    has_shm = any(r.get("shm_used_bytes") is not None for rows in ranks.values() for r in rows[:1])
    first = min(ranks)
    for t in times:
        rows = [at_fast(rank, t) for rank in ranks]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue
        shm = max((r.get("shm_used_bytes") or 0.0) for r in rows) if has_shm else 0.0
        workers = sum(r.get(worker_key) or 0.0 for r in rows)
        view["t"].append(t - start)
        lead = at_fast(first, t)
        view["step"].append(lead["step"] if lead else None)
        view["ranks"].append(sum(r.get(rank_key) or 0.0 for r in rows) / GIB)
        view["workers"].append(max(workers - shm, 0.0) / GIB)
        view["shm"].append(shm / GIB)
        view["gpu_alloc"].append(max(r.get("gpu_allocated_bytes") or 0.0 for r in rows) / GIB)
        view["gpu_res"].append(max(r.get("gpu_reserved_bytes") or 0.0 for r in rows) / GIB)
        used = [r.get("cpu_mem_used_bytes") for r in rows if r.get("cpu_mem_used_bytes")]
        total = [r.get("cpu_mem_total_bytes") for r in rows if r.get("cpu_mem_total_bytes")]
        view["host_used"].append(max(used) / GIB if used else None)
        view["host_total"].append(max(total) / GIB if total else None)
    view["measure"] = "PSS" if full else "RSS"
    view["has_shm"] = has_shm
    return view


def plot(view, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, gax) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    t = view["t"]
    keys = ["ranks", "workers"] + (["shm"] if view["has_shm"] else [])
    labels = dict(LABELS)
    if not view["has_shm"]:
        labels["workers"] = "loader workers (incl. their /dev/shm; not recorded separately)"
    ax.stackplot(
        t,
        *[view[k] for k in keys],
        colors=[COLOURS[k] for k in keys],
        labels=[labels[k] for k in keys],
        edgecolor="white",
        linewidth=0.3,
    )
    if any(v is not None for v in view["host_used"]):
        used = [v if v is not None else float("nan") for v in view["host_used"]]
        ax.plot(t, used, color=INK, linewidth=1.5, label="node host memory in use (NUMA)")
        cap = max(v for v in view["host_total"] if v is not None)
        ax.axhline(cap, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.annotate(
            f"host capacity {cap:.0f} GiB",
            (t[0], cap),
            xytext=(4, 4),
            textcoords="offset points",
            color=INK_MUTED,
            fontsize=9,
        )
    totals = [sum(view[k][i] for k in keys) for i in range(len(t))]
    peak = max(range(len(t)), key=totals.__getitem__)
    ax.plot([t[peak]], [totals[peak]], marker="o", markersize=5, color=INK)
    ax.set_ylabel(f"host memory, GiB ({view['measure']})", color=INK)
    fig.suptitle(
        f"{title} — peak {totals[peak]:.0f} GiB at {t[peak]:.0f} s",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=4, frameon=False, fontsize=9)
    gax.plot(t, view["gpu_alloc"], color=COLOURS["ranks"], linewidth=1, label="allocated")
    gax.plot(
        t, view["gpu_res"], color=COLOURS["ranks"], linewidth=1, linestyle="--", label="reserved"
    )
    gax.set_ylabel("GPU GiB\n(max over ranks)", color=INK)
    gax.set_xlabel("seconds since first sample", color=INK)
    gax.legend(loc="upper right", frameon=False, fontsize=9)
    for a in (ax, gax):
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=INK_MUTED)
        a.set_ylim(bottom=0)
    # training step ticks along the top edge
    ticks, tick_labels, last = [], [], None
    for x, step in zip(t, view["step"], strict=True):
        if step is not None and step != last and step % 10 == 0:
            ticks.append(x)
            tick_labels.append(str(step))
        last = step
    top = ax.secondary_xaxis("top")
    top.set_xticks(ticks, tick_labels)
    top.tick_params(colors=INK_MUTED, labelsize=8)
    top.set_xlabel("training step", color=INK_MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Default: --run/plots")
    parser.add_argument("--stage", default="train")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    output = args.output or args.run / "plots"
    output.mkdir(parents=True, exist_ok=True)
    memory = load_memory(args.run, args.stage)
    hosts = sorted({h for h, _ in memory})
    if not hosts:
        sys.exit(f"no memory_rank*.csv rows for stage {args.stage} in {args.run}")
    name = args.run.name
    for host in hosts:
        view = node_view(memory, host, args.interval)
        suffix = "" if len(hosts) == 1 else f"_{host.split('.')[0]}"
        title = args.title or f"{name}: what is in host memory ({view['measure']})"
        png = output / f"{name}{suffix}_memory_view.png"
        plot(view, title, png)
        (output / f"{name}{suffix}_memory_view.json").write_text(json.dumps(view))
        sys.stdout.write(f"{png}\n")


if __name__ == "__main__":
    main()
