#!/usr/bin/env python3
"""Slide plot of Martin's nine data-loader options, each measured alone at L5.

Every value comes from the runs' raw loader profiles (dataloader_profile/samples_rank*_worker*.jsonl
and loader_wait_rank*.jsonl) in the main checkout's experiments/martin/<run>/. No estimates.

Each option is judged on the loader stage it changes (median seconds per sample over the 400
samples), against the runs that leave that stage alone. The baseline's node read ~15 % slower than
the others, so the reference is the median and range of all those runs, not the baseline alone.
even_workset_split changes no per-sample stage, only how samples are spread over the workers, so
its row is the step time. Step time = mean interval between batch arrivals from step 10 on, per
GPU, averaged over the 4 GPUs (from the loader wait log's timestamps).

  WeatherGenerator/.venv/bin/python scripts/plot_loader_options.py
"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from plot_midterm_loader import (
    BLUE,
    GRAY,
    INK,
    MUTED,
    OUT,
    finish,
    json_lines,
    label_bars,
    report,
)

RUNS_DIR = Path(__file__).resolve().parents[2] / "WeatherGenerator" / "experiments" / "martin"
OBS_SKIP = ("ERA5", "ERA5_in")  # the two anemoi streams; the other 7 are observation streams

# 2026-11-06, martin/exp/single-flags feda61bb: L5, 1 node x 4 GPUs, 12 workers, seed 1, 100 steps
BASE = "spcoicgc"
SINGLE = {  # option: run
    "even_workset_split": "vdhwy1yz",
    "anemoi_select_vars": "zi2h7d7a",
    "anemoi_avoid_dtype_copy": "hq52fbnw",
    "obs_single_read": "fm6x4ski",
    "skip_all_valid_filter_copy": "ko1zio6z",
    "tokenize_coords_only": "cv2iljv2",
    "source_direct_build": "t1te97ff",
    "target_requested_fields_only": "dom75441",
    "skip_train_inverse_indices": "mit3pxf0",
}
MERGED = {"even_workset_split", "anemoi_select_vars", "obs_single_read",
          "skip_all_valid_filter_copy", "target_requested_fields_only"}  # PRs #6-#9, dphpc/dev
# 2026-11-01, martin/backup-dump: every option off (jnsctk9d) and all nine on (svxaue1j)
ALL_OFF, ALL_ON = "jnsctk9d", "svxaue1j"


def per_sample(samples, fn):
    return statistics.median(fn(s["streams"]) for s in samples)


BUILD_TOUCH = {"source_direct_build", "target_requested_fields_only", "skip_train_inverse_indices"}
ANEMOI_TOUCH = {"anemoi_select_vars", "anemoi_avoid_dtype_copy"}


def build_all(st):
    return sum(v.get("build_seconds", 0) for v in st.values())


# option: (stage label, metric on one sample's streams, options that change the same metric)
METRICS = {
    "anemoi_select_vars": ("ERA5 read", lambda st: st["ERA5"]["read_seconds"], ANEMOI_TOUCH),
    "anemoi_avoid_dtype_copy": (
        "ERA5 + ERA5_in read",
        lambda st: st["ERA5"]["read_seconds"] + st["ERA5_in"]["read_seconds"],
        ANEMOI_TOUCH,
    ),
    "obs_single_read": (
        "obs read, 7 streams",
        lambda st: sum(v["read_seconds"] for n, v in st.items() if n not in OBS_SKIP),
        {"obs_single_read"},
    ),
    "skip_all_valid_filter_copy": (
        "NaN filter",
        lambda st: sum(v.get("normalize_seconds", 0) for v in st.values()),
        {"skip_all_valid_filter_copy"},
    ),
    "tokenize_coords_only": ("ERA5_in tokenize", lambda st: st["ERA5_in"]["tokenize_seconds"],
                             {"tokenize_coords_only"}),
    "source_direct_build": (
        "build, source",
        lambda st: sum(v.get("build_seconds", 0) for n, v in st.items() if n != "ERA5"),
        BUILD_TOUCH,
    ),
    "target_requested_fields_only": ("build, all", build_all, BUILD_TOUCH),
    "skip_train_inverse_indices": ("build, all", build_all, BUILD_TOUCH),
}


def samples(run):
    out = []
    for path in sorted((RUNS_DIR / run / "dataloader_profile").glob("samples_rank*_worker*.jsonl")):
        out += [r for r in json_lines(path) if r.get("stage") == "train" and r.get("valid", True)]
    return out


def steps(run):
    """Mean step from step 10 on, loader wait per GPU after the first batch, waits over 5 s."""
    step, wait, long_waits = [], [], 0
    for path in sorted((RUNS_DIR / run / "dataloader_profile").glob("loader_wait_rank*.jsonl")):
        rows = [r for r in json_lines(path) if r.get("stage") == "train"]
        ts = [r["timestamp"] for r in rows]
        step.append(statistics.fmean(b - a for a, b in list(zip(ts, ts[1:], strict=False))[9:]))
        later = [r["wait_seconds"] for r in rows if r["step"] >= 1]
        wait.append(sum(later))
        long_waits += sum(w > 5 for w in later)
    return {"step_s": statistics.fmean(step), "wait_per_gpu_s": statistics.fmean(wait),
            "waits_over_5s": long_waits}


def main():
    runs = {"baseline": BASE, **SINGLE}
    data = {opt: samples(run) for opt, run in runs.items()}
    timing = {opt: steps(run) for opt, run in runs.items()}
    rows, numbers = [], {"single": {}, "together": {}}
    for opt, run in SINGLE.items():
        if opt == "even_workset_split":
            label = "step time"
            value = timing[opt]["step_s"]
            ref = [timing[o]["step_s"] for o in runs if o != opt]
        else:
            label, fn, touch = METRICS[opt]
            value = per_sample(data[opt], fn)
            ref = [per_sample(data[o], fn) for o in runs if o not in touch]
        mid = statistics.median(ref)
        rows.append((opt, label, 100 * (value / mid - 1), 100 * (min(ref) / mid - 1),
                     100 * (max(ref) / mid - 1)))
        numbers["single"][opt] = {"run": run, "stage": label, "value_s": value,
                                  "reference_median_s": mid,
                                  "reference_range_s": [min(ref), max(ref)],
                                  "reference_runs": len(ref), "change_pct": rows[-1][2],
                                  "merged": opt in MERGED}
    off, on = steps(ALL_OFF), steps(ALL_ON)
    numbers["together"] = {"all_off": {"run": ALL_OFF, **off}, "all_on": {"run": ALL_ON, **on}}

    # merged options first, each group sorted by effect
    merged = sorted((r for r in rows if r[0] in MERGED), key=lambda r: r[2])
    other = sorted((r for r in rows if r[0] not in MERGED), key=lambda r: r[2])
    layout = [("Merged into dphpc/dev", None), *[(None, r) for r in merged],
              ("Not merged: no change outside the noise", None), *[(None, r) for r in other]]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [2.4, 1]})
    x_lo, x_hi, x_label = -56, 52, 24  # the stage names sit in a column right of x = 24
    y = len(layout)
    ticks, tick_labels = [], []
    for header, row in layout:
        y -= 1
        if header:
            ax.text(x_lo + 1, y, header, ha="left", va="center", fontsize=11.5, fontweight="bold",
                    color=INK)
            continue
        opt, label, change, lo, hi = row
        ax.barh(y, hi - lo, left=lo, height=0.5, color="#d1d5db", zorder=1)
        merged_opt = opt in MERGED
        ax.plot(change, y, marker="o", ms=10, zorder=3, color=BLUE if merged_opt else "white",
                markeredgecolor=BLUE if merged_opt else MUTED, markeredgewidth=2)
        # merged: right of the dot unless the gray bar is too close; not merged: left of the bar
        right = merged_opt and lo - change >= 12
        ax.text(change + 2.2 if right else min(change, lo) - 2.2, y, f"{change:+.0f} %",
                ha="left" if right else "right", va="center", fontsize=11, color=INK)
        ax.text(x_label, y, label, ha="left", va="center", fontsize=10, color=MUTED)
        ticks.append(y)
        tick_labels.append(opt)
    ax.axvline(0, color=MUTED, lw=1, zorder=2)
    ax.text(x_label, len(layout) - 1, "stage", ha="left", va="center", fontsize=10,
            fontweight="bold", color=MUTED)
    ax.set_yticks(ticks, tick_labels, fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks(range(-50, 20, 10))
    ax.set_ylim(-0.6, len(layout) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("change in that stage against the other runs (%)", loc="left")
    ax.set_title("Each option alone", loc="left")

    bars = ax2.bar([0, 1], [off["step_s"], on["step_s"]], width=0.6, color=[GRAY, BLUE],
                   edgecolor="white", linewidth=2)
    ax2.set_ylim(0, off["step_s"] * 1.5)
    label_bars(ax2, bars, "{:.2f} s")
    ax2.set_xticks([0, 1], ["all off", "all nine on"], fontsize=12)
    ax2.set_ylabel("step time (s)")
    ax2.set_title("All nine together", loc="left")
    ax2.text(0.5, off["step_s"] * 1.38,
             f"loader wait per GPU {off['wait_per_gpu_s']:.0f} → {on['wait_per_gpu_s']:.0f} s\n"
             f"waits over 5 s: {off['waits_over_5s']} → {on['waits_over_5s']}",
             ha="center", va="center", fontsize=10.5, color=INK)

    sel = numbers["single"]
    finish(
        fig,
        "10_loader_options_l5",
        "Five of my nine loader options help, and those five are merged",
        f"Each option alone at L5 on the same 400 samples (dot), against the spread of the runs "
        f"that leave that stage alone (gray bar). Obs read "
        f"{sel['obs_single_read']['change_pct']:+.0f} %, build "
        f"{sel['target_requested_fields_only']['change_pct']:+.0f} %, NaN filter "
        f"{sel['skip_all_valid_filter_copy']['change_pct']:+.0f} %, step time with the even split "
        f"{sel['even_workset_split']['change_pct']:+.0f} %. All nine together: "
        f"{off['step_s']:.2f} → {on['step_s']:.2f} s per step "
        f"({100 * (on['step_s'] / off['step_s'] - 1):+.0f} %).",
        f"Left (6 Nov, martin/exp/single-flags feda61bb): baseline {BASE}, "
        + ", ".join(f"{o} {r}" for o, r in SINGLE.items())
        + f"\nRight (1 Nov, martin/backup-dump): all off {ALL_OFF}, all on {ALL_ON}. "
        "L5, 1 node x 4 GPUs, 12 workers/GPU, seed 1, 100 steps. Stage = median s per sample; "
        "step = mean interval between batches from step 10 on, mean of 4 GPUs.",
        left=0.25,
    )
    (OUT / "numbers_options.json").write_text(json.dumps(numbers, indent=1))
    report(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
