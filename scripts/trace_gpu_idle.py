#!/usr/bin/env python3
"""Where the GPU idles in a PyTorch profiler trace, per profiled step.

For every ProfilerStep#N of a Chrome trace written by torch.profiler (rank 0's
<timestamp>_rank_0.json.gz), takes the union of all GPU activity (kernels, memcpy, memset) and
reports the busy share (all kernels, compute only, NCCL only), the idle time by gap size, the
idle time per training phase (before backward = data + forward + loss, backward, optimizer,
after the optimizer = EMA and logging), the time the main thread waits for the DataLoader, and
the longest gaps with the Python frames that run during them (needs with_stack).

Many short gaps (< 1 ms) mean the GPU runs out of queued kernels between small launches (host
dispatch / launch overhead); a few long gaps mean a blocking host step. with_stack and the
profiler itself add host overhead, so idle shares are upper bounds for an unprofiled run.

  python scripts/trace_gpu_idle.py TRACE.json.gz [--gaps 8] [--json OUT.json]
"""

import argparse
import gzip
import json
from pathlib import Path

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
GAP_BINS = ((20, "<20us"), (100, "20-100us"), (1000, "0.1-1ms"), (5000, "1-5ms"))


def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def clip(events, t0, t1):
    return [
        (max(e["ts"], t0), min(e["ts"] + e["dur"], t1))
        for e in events
        if e["ts"] < t1 and e["ts"] + e["dur"] > t0
    ]


def covered(intervals):
    return sum(end - start for start, end in merge(intervals))


def overlap(gaps, t0, t1):
    return sum(max(0, min(g1, t1) - max(g0, t0)) for g0, g1 in gaps)


def analyse_step(step, gpu, main, num_gaps):
    t0, t1 = step["ts"], step["ts"] + step["dur"]
    in_step = clip(gpu, t0, t1)
    nccl = clip([e for e in gpu if "nccl" in e["name"].lower()], t0, t1)
    compute = clip([e for e in gpu if "nccl" not in e["name"].lower()], t0, t1)
    gaps, prev = [], t0
    for start, end in merge(in_step):
        if start > prev:
            gaps.append((prev, start))
        prev = end
    if t1 > prev:
        gaps.append((prev, t1))

    by_size = {label: 0.0 for _, label in GAP_BINS} | {">5ms": 0.0}
    for g0, g1 in gaps:
        label = next((lbl for limit, lbl in GAP_BINS if g1 - g0 < limit), ">5ms")
        by_size[label] += (g1 - g0) / 1e6

    ops = [e for e in main if t0 <= e["ts"] < t1]
    backward = [
        e for e in ops if e["name"].startswith("torch/_tensor.py(") and "backward" in e["name"]
    ]
    optimizer = [e for e in ops if e["name"].startswith("Optimizer.step")]
    phases = {}
    if backward and optimizer:
        b0, b1 = backward[0]["ts"], backward[0]["ts"] + backward[0]["dur"]
        o0, o1 = optimizer[0]["ts"], optimizer[0]["ts"] + optimizer[0]["dur"]
        bounds = {
            "before backward (data, forward, loss)": (t0, b0),
            "backward": (b0, b1),
            "backward -> optimizer": (b1, o0),
            "optimizer": (o0, o1),
            "after optimizer (EMA, logging)": (o1, t1),
        }
        phases = {
            name: {"s": (b - a) / 1e6, "gpu_idle_s": overlap(gaps, a, b) / 1e6}
            for name, (a, b) in bounds.items()
        }
    loader = sum(e["dur"] for e in ops if e["name"].startswith("enumerate(DataLoader)"))

    longest = []
    for g0, g1 in sorted(gaps, key=lambda g: g[0] - g[1])[:num_gaps]:
        mid = (g0 + g1) / 2
        frames = sorted(
            (
                e
                for e in ops
                if e["ts"] <= mid <= e["ts"] + e["dur"] and not e["name"].startswith("ProfilerStep")
            ),
            key=lambda e: e["dur"],
        )
        longest.append(
            {
                "ms": (g1 - g0) / 1e3,
                "at_ms": (g0 - t0) / 1e3,
                "frames": [f"{e['name'][:70]} ({e['dur'] / 1e3:.0f} ms)" for e in frames[:4]],
            }
        )
    return {
        "step": step["name"],
        "step_s": step["dur"] / 1e6,
        "gpu_ops": sum(1 for e in gpu if t0 <= e["ts"] < t1),
        "busy_any": covered(in_step) / step["dur"],
        "busy_compute": covered(compute) / step["dur"],
        "busy_nccl": covered(nccl) / step["dur"],
        "idle_s": sum(g1 - g0 for g0, g1 in gaps) / 1e6,
        "num_gaps": len(gaps),
        "idle_by_gap_size_s": by_size,
        "phases": phases,
        "dataloader_wait_ms": loader / 1e3,
        "longest_gaps": longest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--gaps", type=int, default=8, help="longest gaps to list per step")
    parser.add_argument("--json", type=Path, help="write the results here")
    args = parser.parse_args()

    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace) as f:
        events = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]
    steps = [
        e
        for e in events
        if e.get("cat") == "user_annotation" and e.get("name", "").startswith("ProfilerStep#")
    ]
    if not steps:
        raise SystemExit("no ProfilerStep# annotations in the trace")
    gpu = [e for e in events if e.get("cat") in GPU_CATS]
    tid, pid = steps[0]["tid"], steps[0]["pid"]
    main_thread = [e for e in events if e.get("tid") == tid and e.get("pid") == pid]

    results = [analyse_step(s, gpu, main_thread, args.gaps) for s in steps]
    for r in results:
        print(  # noqa: T201
            f"{r['step']}: {r['step_s']:.3f} s, {r['gpu_ops']} GPU ops, GPU busy "
            f"{100 * r['busy_any']:.1f}% (compute {100 * r['busy_compute']:.1f}%, NCCL "
            f"{100 * r['busy_nccl']:.1f}%), idle {r['idle_s']:.3f} s in {r['num_gaps']} gaps, "
            f"DataLoader wait {r['dataloader_wait_ms']:.2f} ms"
        )
        print(  # noqa: T201
            "  idle by gap size (s): "
            + ", ".join(f"{k} {v:.3f}" for k, v in r["idle_by_gap_size_s"].items())
        )
        for name, p in r["phases"].items():
            share = 100 * p["gpu_idle_s"] / p["s"] if p["s"] else 0
            print(  # noqa: T201
                f"  {name:38s} {p['s']:6.3f} s, GPU idle {p['gpu_idle_s']:6.3f} s ({share:4.1f}%)"
            )
        for g in r["longest_gaps"]:
            print(f"  gap {g['ms']:6.1f} ms at +{g['at_ms']:7.1f} ms: {' < '.join(g['frames'])}")  # noqa: T201
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
