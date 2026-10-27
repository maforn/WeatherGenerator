#!/usr/bin/env python3
"""Break down the peak GPU memory of a CUDA allocator snapshot (profiling.memory_profiling).

Replays the allocation events, finds the moment with the most memory allocated, and groups
the memory live at that moment by category, by model part (the outermost weathergen class in
the call stack, e.g. ForecastingEngine) and by operation (the innermost one). Memory allocated
before recording started (the model weights) has no call stack and is shown as such.

  python scripts/analyze_memory_snapshot.py <snapshot.pickle> [--top N]
"""

import argparse
import ast
import collections
import functools
import pickle
from pathlib import Path

GIB = 2**30


@functools.cache
def _classes(filename: str) -> list[tuple[int, int, str]]:
    """(first line, last line, class name) of every class in a source file."""
    try:
        tree = ast.parse(Path(filename).read_text())
    except (OSError, SyntaxError):
        return []
    return [(n.lineno, n.end_lineno, n.name) for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def where(frame: dict) -> str:
    """Class.method of a frame, or file:function outside a class."""
    owners = [c for c in _classes(frame["filename"]) if c[0] <= frame["line"] <= c[1]]
    if owners:
        return f"{max(owners)[2]}.{frame['name']}"  # innermost class
    return f"{frame['filename'].rsplit('/', 1)[-1]}:{frame['name']}"


def weathergen_frames(frames: list[dict]) -> list[dict]:
    return [f for f in frames if "/weathergen/" in f["filename"] and "/train/" not in f["filename"]]


def model_part(frames: list[dict]) -> str:
    """The model part an allocation belongs to: the first model frame below Model.forward
    (EncoderModule.forward, ForecastingEngine.forward, Model.predict_decoders, ...)."""
    outer_to_inner = [where(f) for f in reversed(frames) if "/weathergen/model/" in f["filename"]]
    parts = [p for p in outer_to_inner if p != "Model.forward"]
    return parts[0] if parts else "(no model frame)"


def category(frames: list[dict]) -> str:
    """Coarse owner of an allocation, from its Python call stack (innermost frame first)."""
    files = [f["filename"] for f in frames]
    names = [f["name"] for f in frames]
    if "to_device" in names:
        return "input batch"
    if any("optim" in f for f in files):
        return "optimizer"
    if any("fsdp" in f for f in files):
        return "FSDP buffers"
    if any("backward" in n for n in names) or any("autograd" in f for f in files):
        return "backward"
    return "forward"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("snapshot")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    with open(args.snapshot, "rb") as f:
        snapshot = pickle.load(f)
    events = next(t for t in snapshot["device_traces"] if t)
    blocks = [b for s in snapshot["segments"] for b in s["blocks"]]
    allocated_at_end = sum(b["size"] for b in blocks if b["state"] == "active_allocated")

    # first pass: memory allocated before recording started = what is left at the end without a
    # recorded allocation, plus what was freed during recording without a recorded allocation
    live: dict[int, dict] = {}
    recorded = freed_unrecorded = 0
    for e in events:
        if e["action"] == "alloc":
            live[e["addr"]] = e
            recorded += e["size"]
        elif e["action"] == "free_completed":
            if e["addr"] in live:
                recorded -= live.pop(e["addr"])["size"]
            else:
                freed_unrecorded += e["size"]
    start = allocated_at_end - recorded + freed_unrecorded

    # second pass: replay from that start and keep the moment with the most bytes allocated
    live = {}
    before, total = start, start
    peak, peak_before, peak_live, peak_event = 0, start, {}, events[0]
    for e in events:
        if e["action"] == "alloc":
            live[e["addr"]] = e
            total += e["size"]
            if total > peak:
                peak, peak_before, peak_live, peak_event = total, before, dict(live), e
        elif e["action"] == "free_completed":
            if e["addr"] in live:
                total -= live.pop(e["addr"])["size"]
            else:
                total -= e["size"]
                before -= e["size"]
    before = peak_before  # allocated before recording and still alive at the peak

    out = []
    seconds = (events[-1]["time_us"] - events[0]["time_us"]) / 1e6
    out.append(f"{len(events)} events over {seconds:.1f} s")
    out.append(
        f"peak allocated {peak / GIB:.2f} GiB "
        f"({before / GIB:.2f} allocated before recording + {(peak - before) / GIB:.2f} recorded), "
        f"at {(peak_event['time_us'] - events[0]['time_us']) / 1e6:.1f} s"
    )
    stack = " <- ".join(where(f) for f in peak_event["frames"][:8])
    out.append(f"allocation that reached the peak ({peak_event['size'] / GIB:.2f} GiB): {stack}")

    by_category = collections.Counter({"allocated before recording (weights etc.)": before})
    by_part = collections.Counter()
    by_op = collections.Counter()
    sizes_by_part: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for e in peak_live.values():
        frames = weathergen_frames(e["frames"])
        cat = category(e["frames"])
        part = f"{cat:<12} {model_part(e['frames'])}"
        by_category[cat] += e["size"]
        by_part[part] += e["size"]
        by_op[f"{cat:<12} {where(frames[0]) if frames else '(outside weathergen)'}"] += e["size"]
        sizes_by_part[part][e["size"]] += 1

    for title, counter in (
        ("by category", by_category),
        ("by model part", by_part),
        ("by operation (innermost weathergen frame)", by_op),
    ):
        out.append(f"\nlive at the peak, {title}:")
        out += [f"  {size / GIB:7.2f} GiB  {name}" for name, size in counter.most_common(args.top)]

    # which tensors make up the largest parts: most memory-heavy allocation sizes
    out.append("\nlargest model parts, by allocation size (count x MiB):")
    for part, _ in by_part.most_common(5):
        sizes = sorted(sizes_by_part[part].items(), key=lambda x: -x[0] * x[1])[:4]
        out.append(f"  {part.strip()}: " + ", ".join(f"{n} x {s / 2**20:.1f}" for s, n in sizes))
    print("\n".join(out))  # noqa: T201


if __name__ == "__main__":
    main()
