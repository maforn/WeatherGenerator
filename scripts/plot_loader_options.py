#!/usr/bin/env python3
"""Slide plot of my five data-loader options merged into dphpc/dev (PRs #6-#10), each alone at L5.

Every value comes from the runs' raw loader profiles (dataloader_profile/samples_rank*_worker*.jsonl
and loader_wait_rank*.jsonl) in the main checkout's experiments/martin/<run>/. No estimates.

Four options make a loader stage faster. Each is judged on that stage (mean seconds per sample over
the 400 samples) against the runs with the same samples that leave the stage alone: the all-off
baseline and the other single-option runs. The baseline's node read ~15 % slower than the others,
so the reference is the median and range of all those runs, not the baseline alone. This is the
"other runs" column of the PR descriptions. The even-split run is never a reference for a stage:
each worker samples different dates there, so its samples differ.

even_workset_split changes no stage, only which worker gets the leftover batches, so its row is
the time spent waiting for batches: summed over steps 1-99 per GPU, mean of the 4 GPUs. Step 0,
which waits while the workers start, is excluded.

  WeatherGenerator/.venv/bin/python scripts/plot_loader_options.py
"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from plot_midterm_loader import BLUE, GRAY, INK, MUTED, OUT, finish, json_lines, report

RUNS_DIR = Path(__file__).resolve().parents[2] / "WeatherGenerator" / "experiments" / "martin"
OBS_SKIP = ("ERA5", "ERA5_in")  # the two anemoi streams; the other 7 are observation streams

# 2026-11-06, martin/exp/single-flags feda61bb: L5, 1 node x 4 GPUs, 12 workers, seed 1, 100 steps
BASE = "spcoicgc"
MERGED = {  # option: (PR into dphpc/dev, run with only this option on)
    "even_workset_split": (6, "vdhwy1yz"),
    "obs_single_read": (7, "fm6x4ski"),
    "target_requested_fields_only": (8, "dom75441"),
    "skip_all_valid_filter_copy": (9, "ko1zio6z"),
    "anemoi_select_vars": (10, "zi2h7d7a"),
}
# My other four options had no effect alone and were never PR'd. Their runs (same day, same
# samples) are used only as reference runs.
REFERENCE_ONLY = {
    "anemoi_avoid_dtype_copy": "hq52fbnw",
    "tokenize_coords_only": "cv2iljv2",
    "source_direct_build": "t1te97ff",
    "skip_train_inverse_indices": "mit3pxf0",
}
RUNS = {"baseline": BASE, **{o: r for o, (_, r) in MERGED.items()}, **REFERENCE_ONLY}


def obs_read(st):
    return sum(v["read_seconds"] for n, v in st.items() if n not in OBS_SKIP)


def nan_filter(st):
    return sum(v.get("normalize_seconds", 0) for v in st.values())


def build_all(st):
    return sum(v.get("build_seconds", 0) for v in st.values())


def era5_read(st):
    return st["ERA5"]["read_seconds"]


# option: (stage, metric on one sample's streams, options that change the same metric).
# skip_all_valid_filter_copy also skips the obs time-window copy, which sits in the obs read.
STAGES = {
    "obs_single_read": (
        "obs read, 7 streams",
        obs_read,
        {"obs_single_read", "skip_all_valid_filter_copy"},
    ),
    "target_requested_fields_only": (
        "build",
        build_all,
        {"target_requested_fields_only", "source_direct_build", "skip_train_inverse_indices"},
    ),
    "skip_all_valid_filter_copy": ("NaN filter", nan_filter, {"skip_all_valid_filter_copy"}),
    "anemoi_select_vars": (
        "ERA5 read",
        era5_read,
        {"anemoi_select_vars", "anemoi_avoid_dtype_copy"},
    ),
}


def samples(run):
    out = []
    for path in sorted((RUNS_DIR / run / "dataloader_profile").glob("samples_rank*_worker*.jsonl")):
        out += [r for r in json_lines(path) if r.get("stage") == "train" and r.get("valid", True)]
    return out


def waits(run):
    """Waiting for batches per GPU (steps >= 1, mean of the GPUs) and each step's longest wait."""
    per_gpu, by_step = [], {}
    for path in sorted((RUNS_DIR / run / "dataloader_profile").glob("loader_wait_rank*.jsonl")):
        rows = [r for r in json_lines(path) if r.get("stage") == "train" and r["step"] >= 1]
        per_gpu.append(sum(r["wait_seconds"] for r in rows))
        for r in rows:
            by_step[r["step"]] = max(by_step.get(r["step"], 0.0), r["wait_seconds"])
    steps = sorted(by_step)
    return {
        "wait_per_gpu_s": statistics.fmean(per_gpu),
        "longest_wait_s": max(by_step.values()),
        "steps": steps,
        "longest_wait_per_step_s": [by_step[s] for s in steps],
    }


def compare(value, ref):
    mid = statistics.median(ref)
    return {
        "value_s": value,
        "reference_median_s": mid,
        "reference_range_s": [min(ref), max(ref)],
        "reference_runs": len(ref),
        "change_pct": 100 * (value / mid - 1),
        "range_pct": [100 * (min(ref) / mid - 1), 100 * (max(ref) / mid - 1)],
    }


def measure():
    data = {o: samples(r) for o, r in RUNS.items()}
    timing = {o: waits(r) for o, r in RUNS.items()}
    numbers = {}
    for opt, (pr, run) in MERGED.items():
        if opt == "even_workset_split":
            stage = "waiting per GPU"
            value = timing[opt]["wait_per_gpu_s"]
            ref = [timing[o]["wait_per_gpu_s"] for o in RUNS if o != opt]
        else:
            stage, fn, touch = STAGES[opt]
            value = statistics.fmean(fn(s["streams"]) for s in data[opt])
            ref = [
                statistics.fmean(fn(s["streams"]) for s in data[o])
                for o in RUNS
                if o not in touch and o != "even_workset_split"
            ]
        numbers[opt] = {"pr": pr, "run": run, "stage": stage, **compare(value, ref)}
    return numbers, timing


def seconds(x):
    return f"{x:.0f} s" if x >= 10 else f"{x:.1f} s" if x >= 1 else f"{x:.2f} s"


def plot_options(ax, numbers):
    rows = sorted(numbers.items(), key=lambda kv: kv[1]["change_pct"], reverse=True)
    names = ax.get_yaxis_transform()  # x in axes units, y in data units
    for y, (opt, n) in enumerate(rows):
        lo, hi = n["range_pct"]
        change = n["change_pct"]
        ax.barh(y, hi - lo, left=lo, height=0.42, color="#d1d5db", zorder=1)
        ax.plot(change, y, marker="o", ms=11, color=BLUE, zorder=3)
        right = lo - change >= 20  # the percentage goes right of the dot when it fits
        ax.text(
            change + 3 if right else change - 3,
            y,
            f"{change:+.0f} %".replace("-", "\u2212"),
            ha="left" if right else "right",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            -0.02,
            y + 0.13,
            f"{opt}  (#{n['pr']})",
            transform=names,
            ha="right",
            va="bottom",
            fontsize=11,
            color=INK,
        )
        ax.text(
            -0.02,
            y + 0.07,
            f"{n['stage']}: {seconds(n['reference_median_s'])} → {seconds(n['value_s'])}",
            transform=names,
            ha="right",
            va="top",
            fontsize=10,
            color=MUTED,
        )
    ax.axvline(0, color=MUTED, lw=1, zorder=2)
    ax.set_yticks([])
    ax.set_xlim(-95, 25)
    ax.set_xticks(range(-90, 30, 15))
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("change against the other runs' median (%)", loc="left")
    ax.set_title("Each option alone", loc="left")


def plot_even_split(ax, off, on):
    ax.plot(off["steps"], off["longest_wait_per_step_s"], color=GRAY, lw=1.8, zorder=2)
    ax.plot(on["steps"], on["longest_wait_per_step_s"], color=BLUE, lw=2.2, zorder=3)
    ax.text(
        94,
        off["longest_wait_s"],
        f"all options off: the GPUs wait for\nthe last worker's extra batches\n"
        f"{off['wait_per_gpu_s']:.0f} s waiting per GPU",
        ha="right",
        va="top",
        fontsize=10.5,
        color=MUTED,
    )
    ax.text(
        4,
        13,
        f"only even_workset_split\n{on['wait_per_gpu_s']:.0f} s waiting per GPU",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=BLUE,
        fontweight="bold",
    )
    ax.set_xlim(0, 101)
    ax.set_ylim(0, off["longest_wait_s"] * 1.12)
    ax.set_xlabel("step of the 100-step mini-epoch", loc="left")
    ax.set_ylabel("longest wait of the 4 GPUs (s)")
    ax.set_title("Even split: wait per step", loc="left")


def main():
    numbers, timing = measure()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [1.45, 1]})
    plot_options(ax, numbers)
    plot_even_split(ax2, timing["baseline"], timing["even_workset_split"])

    ev = numbers["even_workset_split"]
    lo, hi = ev["reference_range_s"]
    finish(
        fig,
        "10_loader_options_l5",
        "My five merged loader options, each measured alone",
        f"Four make one loader stage faster (dot), against the runs with the same samples that "
        f"leave that stage alone (gray bar: their range). The even split removes the wait for "
        f"the last worker's extra batches at the end of each mini-epoch: {lo:.0f}–{hi:.0f} s per "
        f"GPU in the other runs, {ev['value_s']:.0f} s with it.",
        f"6 Nov, L5, 1 node x 4 GPUs, 12 workers/GPU, seed 1, 100 steps (martin/exp/single-flags "
        f"feda61bb). All off {BASE}; "
        + ", ".join(f"{o} {r}" for o, (_, r) in MERGED.items())
        + "; reference only (my four unmerged options) "
        + ", ".join(REFERENCE_ONLY.values())
        + ".\nStage = mean s per sample over 400 samples; waiting = steps 1-99, mean of 4 GPUs. "
        "anemoi_select_vars was measured with the PR's first version (anemoi select view).",
        left=0.29,
    )
    out = {
        **numbers,
        "even_split_waits": {
            "all_off": {k: v for k, v in timing["baseline"].items() if k != "steps"},
            "even_workset_split": {
                k: v for k, v in timing["even_workset_split"].items() if k != "steps"
            },
        },
    }
    (OUT / "numbers_options.json").write_text(json.dumps(out, indent=1))
    report(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
