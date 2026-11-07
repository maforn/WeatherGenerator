#!/usr/bin/env python3
"""Where the GPU time of an Nsight Systems trace goes (make profile writes the trace).

Prints, for the training steps in the trace (NVTX batch_ ranges, profiling.nvtx_annotate):
  - GPU kernel time by kind (NCCL, flash-attention, GEMM, norms, elementwise, ...);
  - how busy the GPU is: compute kernels, NCCL alone, idle;
  - what the training thread does while the GPU is idle, and where it synchronizes
    (innermost NVTX range; config/profiling/annotations.json names the model functions).

  module load Stages/2025 GCCcore/.13.3.0 Nsight-Systems/2024.7.1   # on JUPITER
  nsys export --type sqlite -o trace.sqlite trace.nsys-rep
  python scripts/analyze_nsys.py trace.sqlite
"""

import argparse
import bisect
import re
import sqlite3
from collections import defaultdict

KINDS = [
    ("NCCL all-gather", r"ncclDevKernel_AllGather"),
    ("NCCL reduce-scatter", r"ncclDevKernel_ReduceScatter"),
    ("NCCL other", r"nccl"),
    ("flash-attn forward", r"FlashAttnFwd|flash_fwd"),
    ("flash-attn backward", r"FlashAttnBwd|flash_bwd|convert_dq"),
    ("GEMM", r"gemm|Gemm|sm90_xmma|nvjet|cutlass::Kernel|cublas"),
    ("norm", r"layer_norm|LayerNorm|rms|RMS"),
    ("elementwise", r"elementwise|vectorized|unrolled"),
    ("reduction", r"reduce"),
    ("copy/cat/index", r"copy|Copy|cat|index|gather|scatter|Index"),
    ("optimizer", r"adam|Adam|multi_tensor"),
]


def union(intervals: list[tuple[int, int]]) -> int:
    """Total length covered by the intervals."""
    total, start, end = 0, None, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += 0 if end is None else end - start
            start, end = s, e
        else:
            end = max(end, e)
    return total + (0 if end is None else end - start)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("sqlite")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    db = sqlite3.connect(args.sqlite)
    out = []

    # the training thread and its steps (skip the first, partial one)
    tid = db.execute(
        "SELECT globalTid FROM NVTX_EVENTS WHERE text LIKE 'batch_%' LIMIT 1"
    ).fetchone()
    if tid is None:
        raise SystemExit("no batch_ NVTX ranges: profile with profiling.nvtx_annotate=True")
    ranges = db.execute(
        "SELECT n.start, n.end, coalesce(n.text, s.value) FROM NVTX_EVENTS n "
        "LEFT JOIN StringIds s ON n.textId = s.id WHERE n.globalTid = ? AND n.end IS NOT NULL "
        "ORDER BY n.start",
        tid,
    ).fetchall()
    steps = [(s, e) for s, e, name in ranges if name and name.startswith("batch_")]
    lo, hi = steps[1][0], steps[-1][1]
    num_steps = len(steps) - 2
    per_step = (hi - lo) / 1e9 / num_steps
    out.append(f"{num_steps} steps, {per_step:.2f} s per step (with Nsight overhead)")

    kernels = db.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName = s.id WHERE k.end > ? AND k.start < ?",
        (lo, hi),
    ).fetchall()
    by_kind: dict[str, int] = defaultdict(int)
    for s, e, name in kernels:
        by_kind[next((k for k, pattern in KINDS if re.search(pattern, name)), "other")] += e - s
    total = sum(by_kind.values())
    out.append(
        f"\nGPU kernel time by kind ({total / 1e9 / num_steps:.2f} s per step, all streams):"
    )
    for kind, t in sorted(by_kind.items(), key=lambda x: -x[1]):
        out.append(f"  {100 * t / total:5.1f}%  {kind}")

    nccl = [(s, e) for s, e, name in kernels if "nccl" in name.lower()]
    compute = [(s, e) for s, e, name in kernels if "nccl" not in name.lower()]
    window, busy_compute, busy_any = hi - lo, union(compute), union(compute + nccl)
    out.append(
        f"\nGPU busy: compute kernels {100 * busy_compute / window:.1f}%, "
        f"NCCL alone {100 * (busy_any - busy_compute) / window:.1f}%, "
        f"idle {100 * (1 - busy_any / window):.1f}%"
    )

    starts = [r[0] for r in ranges]

    def innermost(t: int) -> str:
        """Innermost NVTX range of the training thread containing t (ranges nest)."""
        for j in range(bisect.bisect_right(starts, t) - 1, -1, -1):
            s, e, name = ranges[j]
            if s <= t <= e:
                return (name or "").replace("WeatherGen:weathergen.", "").replace("PyTorch:", "")
        return "(no range)"

    idle: dict[str, int] = defaultdict(int)
    cur = lo
    for s, e, _ in sorted(kernels):
        if s > cur + 20_000:  # gaps over 20 us
            idle[innermost((cur + s) // 2)] += s - cur
        cur = max(cur, e)
    out.append("\nGPU idle per step, by what the training thread does meanwhile:")
    for name, t in sorted(idle.items(), key=lambda x: -x[1])[: args.top]:
        out.append(f"  {t / 1e9 / num_steps:5.2f} s  {name[:90]}")

    syncs = db.execute(
        "SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId = s.id WHERE r.globalTid = ? "
        "AND s.value LIKE 'cudaStreamSynchronize%' AND r.start > ? AND r.end < ?",
        (tid[0], lo, hi),
    ).fetchall()
    launches = db.execute(
        "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId = s.id "
        "WHERE s.value LIKE 'c%LaunchKernel%' AND r.start > ? AND r.end < ?",
        (lo, hi),
    ).fetchone()[0]
    wait = sum(e - s for s, e in syncs) / 1e9 / num_steps
    out.append(
        f"\nper step: {launches / num_steps:.0f} kernel launches, {len(syncs) / num_steps:.0f} "
        f"stream synchronizations ({wait:.2f} s waiting)"
    )
    print("\n".join(out))  # noqa: T201


if __name__ == "__main__":
    main()
