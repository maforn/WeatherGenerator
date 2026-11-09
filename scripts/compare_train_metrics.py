#!/usr/bin/env python3
"""Compare training runs from their <run_id>_train_metrics.json (train_logging.metrics: 1).

Step time is the difference between consecutive metrics timestamps (one line per step), so it
works for any branch, with or without the loader profiling. Reports, per run: steps, median
and mean step time after --skip steps, the number of steps longer than --slow seconds, the GPU
peak (max allocated / reserved over the run, global = max over ranks), the loss at the first
and last logged step, and the share of the wall time spent in the excess of the slow steps over
a normal step (the GPU waiting, for data when slow steps recur every num_workers-th step).

  python scripts/compare_train_metrics.py LABEL=PATH [LABEL=PATH ...] [--skip 10] [--json OUT]
"""

import argparse
import json
import statistics
from pathlib import Path


def summarise(path, skip, slow):
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    lines = [r for r in lines if r.get("stage", "train") == "train"]
    stamps = [r["weathergen.timestamp"] / 1000 for r in lines]
    steps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    after = steps[skip:]
    losses = [r["loss_avg_mean"] for r in lines if r.get("loss_avg_mean") is not None]

    def peak(key):
        values = [r.get(key) for r in lines if r.get(key) is not None]
        return max(values) if values else None

    # excess of the slow steps over a normal (median fast) step: time the GPU waited, mostly
    # for data when the slow steps come at a fixed period (every num_workers-th step)
    fast = [s for s in after if s <= slow]
    normal = statistics.median(fast) if fast else 0.0
    waited = sum(s - normal for s in after if s > slow)
    return {
        "steps_logged": len(lines),
        "wall_s": sum(after),
        "waited_share": waited / sum(after) if after else None,
        "median_s": statistics.median(after) if after else None,
        "mean_s": statistics.fmean(after) if after else None,
        "slow_steps": sum(1 for s in steps if s > slow),
        "max_step_s": max(steps) if steps else None,
        "gpu_max_allocated_gib": peak("performance.memory.global.max_allocated_gib"),
        "gpu_max_reserved_gib": peak("performance.memory.global.max_reserved_gib"),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("runs", nargs="+", help="LABEL=path/to/<run>_train_metrics.json")
    parser.add_argument("--skip", type=int, default=10, help="steps left out of median/mean")
    parser.add_argument("--slow", type=float, default=5.0, help="seconds counted as a slow step")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    results = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        results[label] = summarise(path, args.skip, args.slow)

    def fmt(value, spec):
        return format(value, spec) if value is not None else "-"

    print(  # noqa: T201
        f"{'run':22s} {'steps':>5s} {'median s':>8s} {'mean s':>7s} {'slow':>5s} "
        f"{'max s':>6s} {'waited':>6s} {'GPU alloc':>9s} {'GPU res':>8s} {'loss first→last':>16s}"
    )
    for label, r in results.items():
        print(  # noqa: T201
            f"{label:22s} {r['steps_logged']:5d} {fmt(r['median_s'], '8.3f')} "
            f"{fmt(r['mean_s'], '7.3f')} {r['slow_steps']:5d} {fmt(r['max_step_s'], '6.1f')} "
            f"{fmt(100 * r['waited_share'] if r['waited_share'] is not None else None, '5.0f')}% "
            f"{fmt(r['gpu_max_allocated_gib'], '9.2f')} {fmt(r['gpu_max_reserved_gib'], '8.2f')} "
            f"{fmt(r['loss_first'], '7.4f')}→{fmt(r['loss_last'], '.4f')}"
        )
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
