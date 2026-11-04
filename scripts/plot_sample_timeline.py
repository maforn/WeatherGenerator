#!/usr/bin/env python3
"""Loader samples as allocations over time: the memory_viz "Active Memory Timeline" for one rank.

PyTorch's memory_viz stacks every live CUDA allocation as its own band. This does the same for
the data loader on one training rank: every training sample is a band that

* starts when a worker finishes building it and hands it over (it now waits in /dev/shm),
* ends when the training loop receives it as a batch,
* is as tall as the sample's hand-off size.

``--show-build`` also draws the build phase (lighter), sized by the worker's RSS growth; that
underestimates later builds because worker RSS does not fall back between samples.

Bands are stacked in order of build start, like allocations in memory_viz, so gaps close when a
sample is consumed. Sample-to-step matching follows PyTorch's in-order round robin over workers.
Hand-off sizes come from the run's ``payload`` records (``profile_payload_bytes``); otherwise from
``--sizes-from`` (another run's payloads by sample id, source tokens doubled for fp32 with
``--fp32``), otherwise ``--default-gib``.

Writes ``<run>_sample_timeline.json`` (bands, for interactive pages) and ``.png``.

Example:
    python scripts/plot_sample_timeline.py --run experiments/martin/jnsctk9d \
        --sizes-from experiments/martin/va54ajr5 --fp32 --xmax 520 --ymax 40
"""

import argparse
import json
import sys
from pathlib import Path

GIB = 1024**3
SOURCE_GROUP = "source_samples.samples"


def read_jsonl(path):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def payload_gib(payload, fp32):
    """Hand-off bytes of one sample in GiB; with fp32, undo the bf16 cast of source tokens."""
    total = payload["torch_bytes"] + payload.get("numpy_bytes", 0)
    if fp32:
        total += payload.get("groups", {}).get(SOURCE_GROUP, 0)
    return total / GIB


def load_sizes(run_dir, rank, fp32):
    sizes = {}
    profile = Path(run_dir) / "dataloader_profile"
    for path in profile.glob(f"samples_rank{rank:04d}_worker*.jsonl"):
        for rec in read_jsonl(path):
            if rec.get("payload"):
                sizes[rec["sample_id"]] = payload_gib(rec["payload"], fp32)
    return sizes


def sample_bands(run_dir, rank=0, stage="train", sizes=None, default_gib=None, fp32=False):
    profile = Path(run_dir) / "dataloader_profile"
    workers = {}
    for path in sorted(profile.glob(f"samples_rank{rank:04d}_worker*.jsonl")):
        recs = [r for r in read_jsonl(path) if r.get("stage") == stage and r.get("valid", True)]
        if recs:
            recs.sort(key=lambda r: r["timestamp"])
            workers[recs[0]["worker_id"]] = recs
    waits = [
        w
        for w in read_jsonl(profile / f"loader_wait_rank{rank:04d}.jsonl")
        if w.get("stage") == stage
    ]
    waits.sort(key=lambda w: w["step"])
    # PyTorch yields batches in order, round robin over workers, skipping exhausted ones.
    queues = {wid: list(recs) for wid, recs in sorted(workers.items())}
    order = []
    while any(queues.values()):
        for wid in sorted(queues):
            if queues[wid]:
                order.append((wid, queues[wid].pop(0)))
    own = load_sizes(run_dir, rank, fp32=False) if not fp32 else {}
    known = [v for v in (sizes or {}).values()]
    mean_size = sum(known) / len(known) if known else default_gib
    # Sample records are stamped when the build starts; wait records when the batch arrives.
    origin = min(r["timestamp"] for recs in workers.values() for r in recs)
    bands, estimated = [], 0
    for step, (wid, rec) in enumerate(order):
        if step >= len(waits):
            break
        sid = rec["sample_id"]
        guessed = sid not in own and not (sizes and sid in sizes)
        if sid in own:
            size = own[sid]
        elif sizes and sid in sizes:
            size = sizes[sid]
        else:
            size = mean_size
            estimated += 1
        build = max(rec.get("rss_after_bytes", 0) - rec.get("rss_before_bytes", 0), 0) / GIB
        bands.append(
            {
                "id": sid,
                "worker": wid,
                "step": step,
                "t0": round(rec["timestamp"] - origin, 2),
                "t1": round(rec["timestamp"] + rec["total_seconds"] - origin, 2),
                "t2": round(
                    max(waits[step]["timestamp"], rec["timestamp"] + rec["total_seconds"]) - origin,
                    2,
                ),
                "build": round(build, 3),
                "size": round(size, 3),
                "estimated": guessed,
            }
        )
    return {
        "bands": bands,
        "estimated_sizes": estimated,
        "workers": len(workers),
        "steps": len(bands),
    }


def stack(bands, t, show_build=False):
    """Heights of all live bands at time t, in the order of ``bands`` (memory_viz stacking)."""
    out = []
    for b in bands:
        if show_build and b["t0"] <= t < b["t1"]:
            out.append((b, b["build"], True))
        elif b["t1"] <= t < b["t2"]:
            out.append((b, b["size"], False))
    return out


def plot(result, title, path, xmax=None, ymax=None, show_build=False):
    import colorsys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    bands = sorted(result["bands"], key=lambda b: b["t0"] if show_build else b["t1"])
    end = xmax or max(b["t2"] for b in bands)
    grid = [i * 0.5 for i in range(int(end / 0.5) + 2)]
    tops = {b["id"]: [None] * len(grid) for b in bands}
    bottoms = {b["id"]: [None] * len(grid) for b in bands}
    building = {b["id"]: [False] * len(grid) for b in bands}
    totals = []
    for i, t in enumerate(grid):
        level = 0.0
        for b, h, is_building in stack(bands, t, show_build):
            bottoms[b["id"]][i] = level
            level += h
            tops[b["id"]][i] = level
            building[b["id"]][i] = is_building
        totals.append(level)
    fig, ax = plt.subplots(figsize=(13, 5))
    for k, b in enumerate(bands):
        hue = (k * 0.61803398875) % 1.0
        colour = colorsys.hls_to_rgb(hue, 0.62, 0.55)
        for phase in (True, False):  # building (light), then built and waiting (full colour)
            idx = [
                i
                for i in range(len(grid))
                if tops[b["id"]][i] is not None and building[b["id"]][i] == phase
            ]
            if not idx:
                continue
            upper = [(grid[i], tops[b["id"]][i]) for i in idx]
            lower = [(grid[i], bottoms[b["id"]][i]) for i in reversed(idx)]
            ax.add_patch(
                Polygon(
                    upper + lower,
                    closed=True,
                    facecolor=colour,
                    alpha=0.35 if phase else 1.0,
                    edgecolor="white",
                    linewidth=0.3,
                )
            )
    ax.set_xlim(0, end)
    ax.set_ylim(0, ymax or max(totals) * 1.05)
    ax.set_xlabel("seconds since the first sample started")
    ax.set_ylabel("GiB on this rank")
    note = "  (light = being built, solid = waiting)" if show_build else ""
    ax.set_title(title + note, loc="left", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--sizes-from", type=Path, help="Run with payload records, by sample id")
    parser.add_argument("--fp32", action="store_true", help="This run handed off fp32 sources")
    parser.add_argument("--default-gib", type=float, default=None)
    parser.add_argument("--xmax", type=float)
    parser.add_argument("--ymax", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show-build", action="store_true")
    args = parser.parse_args(argv)
    sizes = load_sizes(args.sizes_from, args.rank, args.fp32) if args.sizes_from else None
    result = sample_bands(
        args.run, args.rank, sizes=sizes, default_gib=args.default_gib, fp32=args.fp32
    )
    if not result["bands"]:
        sys.exit(f"no training samples for rank {args.rank} in {args.run}")
    output = args.output or args.run / "plots"
    output.mkdir(parents=True, exist_ok=True)
    name = args.run.name
    (output / f"{name}_sample_timeline.json").write_text(json.dumps(result))
    title = f"{name}, rank {args.rank}: each finished sample, from hand-off until training takes it"
    plot(
        result, title, output / f"{name}_sample_timeline.png", args.xmax, args.ymax, args.show_build
    )
    sys.stdout.write(
        f"{output / (name + '_sample_timeline.png')} ({result['steps']} samples, "
        f"{result['estimated_sizes']} sizes estimated)\n"
    )


if __name__ == "__main__":
    main()
