#!/usr/bin/env python3
"""Slide plots of the data-loader bottleneck study, computed from the measured runs only.

Every value comes from a run's raw output in experiments/ (loader profiles, training metrics,
read-stress, SPLOAD, GPU-idle traces). No estimates, projections or what-ifs. Each figure is
written to experiments/martin/midterm_plots/ with the values it shows in numbers.json.

  WeatherGenerator/.venv/bin/python scripts/plot_midterm_loader.py
"""

import csv
import json
import statistics
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "experiments"
OUT = ROOT / "martin" / "midterm_plots"
GIB = 2**30

# Data-parallel runs on 1 node, 4 GPUs, same seed and code branch (martin/exp/bottleneck).
LEVEL_RUNS = [
    {"level": 5, "run": "ykm2kcxf", "label": "A", "workers": 12},
    {"level": 6, "run": "pv39uted", "label": "C", "workers": 8},
    {"level": 7, "run": "ywaqq3tw", "label": "D", "workers": 1},
]
W4_RUN = {"level": 5, "run": "kdi7cfy9", "label": "W4", "workers": 4}

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#9aa3ae"
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)
BAR = {"edgecolor": "white", "linewidth": 2}


def report(*parts):
    sys.stdout.write(" ".join(str(p) for p in parts) + "\n")


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def json_lines(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train_rows(run_dir):
    (path,) = run_dir.glob("*_train_metrics.json")
    return [
        r
        for r in json_lines(path)
        if r.get("stage") == "train" and "elapsed_training_time_seconds" in r
    ]


def step_seconds(rows):
    t = [float(r["elapsed_training_time_seconds"]) for r in rows]
    return [b - a for a, b in zip(t, t[1:], strict=False)]


def loader_wait_share(run_dir, rows):
    """Share of step time the training loop waited for the next batch (steps >= 1, all ranks).

    Step 0 is excluded: it waits for the first batch while the workers start.
    """
    mean_step = statistics.mean(step_seconds(rows))
    shares = []
    for path in sorted((run_dir / "dataloader_profile").glob("loader_wait_rank*.jsonl")):
        waits = [r["wait_seconds"] for r in json_lines(path) if r.get("step", 0) >= 1]
        shares.append(statistics.mean(waits) / mean_step)
    return statistics.mean(shares), max(shares)


def loader_samples(run_dir):
    samples = []
    for path in sorted((run_dir / "dataloader_profile").glob("samples_rank*_worker*.jsonl")):
        samples += [
            r for r in json_lines(path) if r.get("stage") == "train" and r.get("valid", True)
        ]
    return samples


def stage_split(samples):
    """Median seconds per sample in one worker: read+normalize, tokenize, build, rest."""
    parts = {"read": [], "tokenize": [], "build": [], "other": [], "total": []}
    for s in samples:
        streams = s["streams"].values()
        read = sum(v.get("read_seconds", 0) + v.get("normalize_seconds", 0) for v in streams)
        tok = sum(v.get("tokenize_seconds", 0) for v in streams)
        build = sum(v.get("build_seconds", 0) for v in streams)
        parts["read"].append(read)
        parts["tokenize"].append(tok)
        parts["build"].append(build)
        parts["other"].append(max(s["total_seconds"] - read - tok - build, 0.0))
        parts["total"].append(s["total_seconds"])
    return {k: statistics.median(v) for k, v in parts.items()}, len(samples)


def memory_peaks(run_dir):
    """Peak RSS per rank (worst of the 4 GPUs) and peak node CPU memory in use."""
    rank_peak, workers_peak, cpu_used, cpu_total = [], [], [], None
    for path in sorted((run_dir / "dataloader_profile").glob("memory_rank*.csv")):
        with open(path, newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r.get("timestamp")]
        rank_peak.append(max(float(r["rank_rss_bytes"]) for r in rows) / GIB)
        workers_peak.append(max(float(r["workers_rss_bytes"]) for r in rows) / GIB)
        cpu_used.append(
            max(float(r["cpu_mem_used_bytes"]) for r in rows if r["cpu_mem_used_bytes"])
        )
        cpu_total = float(rows[0]["cpu_mem_total_bytes"])
    return {
        "training_process_peak_gib": max(rank_peak),
        "workers_peak_gib": max(workers_peak),
        "node_cpu_used_peak_gib": max(cpu_used) / GIB,
        "node_cpu_total_gib": cpu_total / GIB,
    }


def label_bars(ax, bars, fmt, pad=0.01):
    top = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + pad * top,
            fmt.format(h),
            ha="center",
            va="bottom",
            color=INK,
            fontsize=12,
        )


def finish(fig, name, title, subtitle, source, left=0.07, bottom_extra=0.0):
    """Title, wrapped subtitle and source line, and margins that fit them."""
    sub = textwrap.fill(subtitle, 112)
    src = "\n".join(textwrap.fill(part, 128) for part in source.split("\n"))
    n_sub, n_src = sub.count("\n") + 1, src.count("\n") + 1
    fig.text(0.02, 0.975, title, ha="left", va="top", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.02, 0.905, sub, ha="left", va="top", fontsize=12.5, color=MUTED, linespacing=1.4)
    fig.text(
        0.02,
        0.015,
        src,
        ha="left",
        va="bottom",
        fontsize=10,
        color=MUTED,
        family="monospace",
        linespacing=1.4,
    )
    fig.subplots_adjust(
        top=0.80 - 0.045 * n_sub,
        bottom=0.115 + 0.035 * n_src + bottom_extra,
        left=left,
        right=0.98,
        wspace=0.28,
    )
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def level_ticks(runs):
    return [
        f"L{r['level']}\n{r['workers']} worker{'s' if r['workers'] > 1 else ''}/GPU" for r in runs
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = {}

    # ---- per-level data-parallel runs -------------------------------------------------------
    per_level = []
    for spec in LEVEL_RUNS + [W4_RUN]:
        d = ROOT / "martin" / spec["run"]
        rows = train_rows(d)
        steps = step_seconds(rows)
        wait_mean, wait_max = loader_wait_share(d, rows)
        samples = loader_samples(d)
        split, n = stage_split(samples)
        payload = statistics.median(
            s["payload"]["groups"]["source_samples.samples"] / GIB for s in samples
        )
        per_level.append(
            {
                **spec,
                "steps": len(rows),
                "step_median_s": statistics.median(steps),
                "loader_wait_share_mean": wait_mean,
                "loader_wait_share_worst_rank": wait_max,
                "loader_seconds_per_sample": split,
                "loader_samples_measured": n,
                "source_tokens_per_sample_gib": payload,
                "gpu_peak_allocated_gib": max(
                    float(r["performance.memory.train.max_allocated_gib"])
                    for r in rows
                    if "performance.memory.train.max_allocated_gib" in r
                ),
                **memory_peaks(d),
            }
        )
    numbers["per_level_runs"] = per_level
    lv = per_level[:3]
    a_run, w4 = per_level[0], per_level[3]
    runs_note = "Data parallel, 1 node, 4 GPUs. Runs " + ", ".join(
        f"L{r['level']} {r['run']}" for r in lv
    )

    # ---- 01: step time vs waiting for data ------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6))
    x = range(len(lv))
    b = ax1.bar(x, [r["step_median_s"] for r in lv], color=BLUE, width=0.55, **BAR)
    ax1.set_title("GPU step time (median, s)", loc="left")
    ax1.set_ylim(0, max(r["step_median_s"] for r in lv) * 1.15)
    label_bars(ax1, b, "{:.1f} s")
    b = ax2.bar(x, [100 * r["loader_wait_share_mean"] for r in lv], color=ORANGE, width=0.55, **BAR)
    ax2.set_title("Waiting for data (% of step)", loc="left")
    ax2.set_ylim(0, 100)
    label_bars(ax2, b, "{:.1f} %")
    for ax in (ax1, ax2):
        ax.set_xticks(list(x), level_ticks(lv))
    finish(
        fig,
        "01_loader_keeps_up",
        "The loader keeps up at levels 5, 6 and 7",
        f"The step time grows {lv[2]['step_median_s'] / lv[0]['step_median_s']:.0f}x from L5 to "
        "L7, while the time waiting for the next batch stays near zero.",
        runs_note + ". Wait share: mean over steps >= 1 and the 4 ranks.",
        bottom_extra=0.04,
    )

    # ---- 02: loader time per sample, by stage ---------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.6))
    stages = [
        ("read", "Read + normalize", BLUE),
        ("tokenize", "Tokenize", ORANGE),
        ("build", "Build padded tokens", GREEN),
        ("other", "Other", GRAY),
    ]
    left = [0.0] * len(lv)
    y = list(range(len(lv)))[::-1]
    for key, name, color in stages:
        vals = [r["loader_seconds_per_sample"][key] for r in lv]
        ax.barh(y, vals, left=left, color=color, height=0.55, label=name, **BAR)
        for yi, lft, v in zip(y, left, vals, strict=True):
            if v > 4:
                ax.text(
                    lft + v / 2,
                    yi,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                    fontweight="bold",
                )
        left = [lft + v for lft, v in zip(left, vals, strict=True)]
    for yi, r in zip(y, lv, strict=True):
        tot = r["loader_seconds_per_sample"]["total"]
        ax.text(tot + 1, yi, f"{tot:.0f} s", va="center", color=INK, fontsize=12)
    ax.set_yticks(y, [f"L{r['level']}" for r in lv])
    ax.set_xlabel("Seconds per sample, one worker (median)")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(left) * 1.12)
    ax.legend(loc="upper right", ncol=1)
    finish(
        fig,
        "02_loader_stage_time",
        "Building the padded tokens is what grows with the level",
        "Reading takes the same time at every level; building the source tokens goes from "
        f"{lv[0]['loader_seconds_per_sample']['build']:.0f} s to "
        f"{lv[2]['loader_seconds_per_sample']['build']:.0f} s per sample.",
        runs_note
        + ". "
        + ", ".join(f"{r['loader_samples_measured']} samples at L{r['level']}" for r in lv)
        + ".",
    )

    # ---- 03: source tokens per sample ------------------------------------------------------
    pad = load_json(ROOT / "martin" / "padding" / "padding_summary.json")
    pad_share = {}
    for level in (5, 6, 7):
        rows_l = [p for p in pad if p["level"] == level]
        pad_share[level] = 1 - sum(p["points"] for p in rows_l) / sum(
            p["padded_rows"] for p in rows_l
        )
    numbers["padding_share_counted"] = pad_share
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6))
    b = ax1.bar(x, [r["source_tokens_per_sample_gib"] for r in lv], color=BLUE, width=0.55, **BAR)
    ax1.set_title("Source tokens per sample (GiB)", loc="left")
    ax1.set_ylim(0, max(r["source_tokens_per_sample_gib"] for r in lv) * 1.15)
    label_bars(ax1, b, "{:.1f}")
    b = ax2.bar(x, [100 * pad_share[r["level"]] for r in lv], color=ORANGE, width=0.55, **BAR)
    ax2.set_title("Rows that are padding (%)", loc="left")
    ax2.set_ylim(0, 112)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    label_bars(ax2, b, "{:.1f} %")
    ax1.set_xticks(list(x), [f"L{r['level']}" for r in lv])
    ax2.set_xticks(list(x), [f"L{r['level']}" for r in lv])
    finish(
        fig,
        "03_source_tokens_padding",
        "The batch grows because of padding, not because of more data",
        "Each occupied cell is padded to whole tokens of 1024 rows (geostationary streams). "
        "Finer cells hold fewer points, so most rows are zeros.",
        "Left: median size measured in the loader workers of "
        + ", ".join(r["run"] for r in lv)
        + ".\nRight: rows counted on 4 real samples with scripts/padding_stats.py "
        "(checked exact against the real tokenizer at L5), experiments/martin/padding.",
    )

    # ---- 04: host memory ----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [3, 2]})
    w = 0.36
    xs = list(x)
    b1 = ax1.bar(
        [i - w / 2 for i in xs],
        [r["training_process_peak_gib"] for r in lv],
        width=w,
        color=BLUE,
        label="Training process",
        **BAR,
    )
    b2 = ax1.bar(
        [i + w / 2 for i in xs],
        [r["workers_peak_gib"] for r in lv],
        width=w,
        color=ORANGE,
        label="Its loader workers, together",
        **BAR,
    )
    ax1.set_ylim(
        0, max(max(r["training_process_peak_gib"], r["workers_peak_gib"]) for r in lv) * 1.4
    )
    label_bars(ax1, b1, "{:.0f}")
    label_bars(ax1, b2, "{:.0f}")
    ax1.set_title("Peak memory per GPU (GiB)", loc="left")
    ax1.set_xticks(xs, level_ticks(lv))
    ax1.legend(loc="upper left")
    cap = lv[0]["node_cpu_total_gib"]
    b = ax2.bar(xs, [r["node_cpu_used_peak_gib"] for r in lv], color=GREEN, width=0.55, **BAR)
    ax2.axhline(cap, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    ax2.text(-0.4, cap * 1.02, f"node CPU memory {cap:.0f} GiB", color=INK, fontsize=11)
    ax2.set_ylim(0, cap * 1.15)
    label_bars(ax2, b, "{:.0f}")
    ax2.set_title("Node CPU memory, peak (GiB)", loc="left")
    ax2.set_xticks(xs, [f"L{r['level']}" for r in lv])
    used = [r["node_cpu_used_peak_gib"] for r in lv]
    finish(
        fig,
        "04_host_memory",
        "Host memory: the loader workers hold most of it",
        f"The training process grows from {lv[0]['training_process_peak_gib']:.0f} to "
        f"{lv[2]['training_process_peak_gib']:.0f} GiB with the batch. Node CPU memory peaked at "
        f"{min(used):.0f}-{max(used):.0f} of {cap:.0f} GiB in all three runs (12, 8 and 1 "
        "workers per GPU).",
        runs_note + ". Sampled RSS; worst of the 4 GPUs. Node: CPU memory only (GPU memory "
        "excluded).",
    )

    # ---- 05: read throughput -----------------------------------------------------------------
    # The 256-reader point was skipped in the job (no measurement), so it is left out.
    rs = [
        r
        for r in load_json(ROOT / "martin" / "read_stress" / "summary.json")
        if "samples_per_s" in r
    ]
    rows_a = train_rows(ROOT / "martin" / a_run["run"])
    t0, t1 = rows_a[10], rows_a[-1]
    consume = (t1["num_samples"] - t0["num_samples"]) / (
        t1["elapsed_training_time_seconds"] - t0["elapsed_training_time_seconds"]
    )
    numbers["read_stress"] = [
        {k: r[k] for k in ("procs", "samples_per_s", "wall_s_median", "node_iowait_mean")}
        for r in rs
    ]
    numbers["l5_training_samples_per_s"] = consume
    fig, ax = plt.subplots(figsize=(11, 5.6))
    procs = [r["procs"] for r in rs]
    sps = [r["samples_per_s"] for r in rs]
    ax.plot(
        procs,
        sps,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=2,
        label="Read only, measured",
    )
    for p, s in zip(procs, sps, strict=True):
        if s < 1.5:  # first point sits under the training line: label it to the right
            ax.text(p + 4, s, f"{s:.2f}", ha="left", va="center", color=INK, fontsize=12)
        else:
            ax.text(p, s + 0.5, f"{s:.1f}", ha="center", color=INK, fontsize=12)
    ax.axhline(consume, color=ORANGE, linewidth=2)
    ax.text(
        200,
        consume + 0.35,
        f"L5 training uses {consume:.2f} samples/s",
        ha="right",
        color=INK,
        fontsize=12,
    )
    ax.set_xlabel("Parallel reader processes on one node")
    ax.set_ylabel("Samples per second, whole node")
    ax.set_xticks(procs)
    ax.set_ylim(0, max(sps) * 1.2)
    best = max(sps)
    iow = max(r["node_iowait_mean"] for r in rs)
    finish(
        fig,
        "05_read_throughput",
        f"Storage is not the limit: one node reads {best / consume:.0f}x what L5 training uses",
        f"Peak {best:.1f} samples/s at 128 readers; more readers no longer help. I/O wait was at "
        f"most {100 * iow:.1f} % of CPU time at every reader count.",
        "READ, job 2766316, scripts/read_stress.py, experiments/martin/read_stress. Training: "
        f"run {a_run['run']} (L5), samples/s from steps 10-{len(rows_a) - 1}.",
    )

    # ---- 06: 12 vs 4 workers at L5 -------------------------------------------------------------
    pair = [a_run, w4]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6))
    xp = [0, 1]
    ticks = [f"{r['workers']} workers/GPU\n{r['run']}" for r in pair]
    b = ax1.bar(xp, [r["step_median_s"] for r in pair], color=BLUE, width=0.5, **BAR)
    ax1.set_ylim(0, max(r["step_median_s"] for r in pair) * 1.25)
    label_bars(ax1, b, "{:.2f} s")
    ax1.set_title("GPU step time (median, s)", loc="left")
    b = ax2.bar(xp, [r["node_cpu_used_peak_gib"] for r in pair], color=GREEN, width=0.5, **BAR)
    ax2.set_ylim(0, max(r["node_cpu_used_peak_gib"] for r in pair) * 1.25)
    label_bars(ax2, b, "{:.0f}")
    ax2.set_title("Node CPU memory in use, peak (GiB)", loc="left")
    for axx in (ax1, ax2):
        axx.set_xticks(xp, ticks)
    finish(
        fig,
        "06_workers_l5",
        "At L5, 4 workers per GPU train as fast as 12, with half the memory",
        f"Loader wait: {100 * a_run['loader_wait_share_mean']:.1f} % with 12 workers, "
        f"{100 * w4['loader_wait_share_mean']:.1f} % with 4.",
        "Data parallel, 1 node, 4 GPUs, L5, same seed. Runs "
        f"{a_run['run']} (12 workers), {w4['run']} (4 workers).",
        bottom_extra=0.04,
    )

    # ---- 07: spatial parallelism, loader side (SPLOAD) ------------------------------------------
    sp = load_json(ROOT / "martin" / "sp_loader" / "summary.json")
    spl = {}
    for level in (5, 7):
        dp = [r for r in sp if r["level"] == level and r["size"] == 1][0]
        ranks = [r for r in sp if r["level"] == level and r["size"] == 4]
        spl[level] = {
            "dp_seconds": dp["total_median"],
            "sp_rank_seconds": [r["total_median"] for r in ranks],
            "dp_peak_gib": dp["peak_rss_gib"],
            "sp_rank_peak_gib": [r["peak_rss_gib"] for r in ranks],
        }
    numbers["sp_loader"] = spl
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, key_dp, key_sp, title, fmt in (
        (ax1, "dp_seconds", "sp_rank_seconds", "Loader time per sample (s)", "{:.0f}"),
        (ax2, "dp_peak_gib", "sp_rank_peak_gib", "Loader peak memory (GiB)", "{:.0f}"),
    ):
        vals_dp = [spl[lv_][key_dp] for lv_ in (5, 7)]
        vals_sp = [max(spl[lv_][key_sp]) for lv_ in (5, 7)]
        b1 = ax.bar(
            [-w / 2, 1 - w / 2],
            vals_dp,
            width=w,
            color=BLUE,
            label="Data parallel: one worker, whole sample",
            **BAR,
        )
        b2 = ax.bar(
            [w / 2, 1 + w / 2],
            vals_sp,
            width=w,
            color=ORANGE,
            label="Spatial parallel: one of 4 ranks (slowest)",
            **BAR,
        )
        ax.set_ylim(0, max(vals_dp) * 1.45)
        label_bars(ax, b1, fmt)
        label_bars(ax, b2, fmt)
        ax.set_title(title, loc="left")
        ax.set_xticks([0, 1], ["L5", "L7"])
    ax1.legend(loc="upper left", fontsize=11)
    group = {lvl: sum(spl[lvl]["sp_rank_seconds"]) for lvl in (5, 7)}
    numbers["sp_loader_group_seconds"] = group
    finish(
        fig,
        "07_spatial_parallel_loader",
        "Spatial parallel: one rank's loader is cheaper, the group's is not",
        "Only the source-token build is split; every rank still reads the whole sample and builds "
        f"all targets. The 4 ranks together spend {group[5]:.0f} s per sample at L5 "
        f"(data parallel: {spl[5]['dp_seconds']:.0f} s) and {group[7]:.0f} s at L7 "
        f"({spl[7]['dp_seconds']:.0f} s).",
        "SPLOAD, job 2801802, scripts/sp_loader_cost.py, same 3 samples, single process, no GPU. "
        "experiments/martin/sp_loader.",
    )

    # ---- 08: spatial vs data parallel training at L5 (Alejandro's runs) -------------------------
    sd = load_json(ROOT / "others" / "sp_vs_dp_l5.json")
    pairs = [("SP4_qb0vmkdr", "DP4_pi2vsvu1"), ("SP4_cnkr2ooi", "DP4_n3tagpzt")]
    tr = {}
    for k in [run for pair in pairs for run in pair]:
        v = sd[k]
        per_step = 1 if k.startswith("SP") else 4
        tr[k] = {
            "samples_per_s": per_step / v["mean_s"],
            "gpu_peak_gib": v["gpu_max_allocated_gib"],
            "waited_share": v["waited_share"],
        }
    numbers["sp_vs_dp_training_l5"] = tr
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, key, title, fmt in (
        (ax1, "samples_per_s", "Samples trained per second", "{:.2f}"),
        (ax2, "gpu_peak_gib", "GPU peak memory (GiB)", "{:.1f}"),
    ):
        vs = [tr[s][key] for s, _ in pairs]
        vd = [tr[d][key] for _, d in pairs]
        b1 = ax.bar(
            [-w / 2, 1 - w / 2], vd, width=w, color=BLUE, label="Data parallel, 4 GPUs", **BAR
        )
        b2 = ax.bar(
            [w / 2, 1 + w / 2], vs, width=w, color=ORANGE, label="Spatial parallel, 4 GPUs", **BAR
        )
        ax.set_ylim(0, max(vd + vs) * 1.45)
        label_bars(ax, b1, fmt)
        label_bars(ax, b2, fmt)
        ax.set_title(title, loc="left")
        ax.set_xticks([0, 1], ["Pair 1", "Pair 2"])
    ax1.legend(loc="upper left", fontsize=11)
    waited = [100 * v["waited_share"] for v in tr.values()]
    finish(
        fig,
        "08_spatial_vs_data_parallel_l5",
        "L5: spatial parallel trains fewer samples/s, with less GPU memory",
        f"Caveat: all four runs waited on the loader {min(waited):.0f}-{max(waited):.0f} % of the "
        "time (older loader, 4 workers per GPU), so these are loader-bound runs.",
        "Alejandro's runs, 2026-11-04, L5, 1 node. Pair 1: SP qb0vmkdr, DP pi2vsvu1. "
        "Pair 2: SP cnkr2ooi, DP n3tagpzt.\nSamples/s = samples per step / mean step "
        "(4 per step data parallel, 1 spatial parallel). experiments/others/sp_vs_dp_l5.json.",
    )

    # ---- 09: GPU busy share ---------------------------------------------------------------------
    l5 = load_json(ROOT / "jonah" / "i3f3hngb" / "gpu_idle.json")
    l7 = load_json(ROOT / "martin" / "q4ag4rlz" / "gpu_idle.json")
    gb = [("L5", s) for s in l5] + [("L7", s) for s in l7]
    numbers["gpu_busy"] = [
        {"level": lvl, "step": s["step"], "step_s": s["step_s"], "busy_any": s["busy_any"]}
        for lvl, s in gb
    ]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    labels = [
        f"{lvl}, {s['step'].replace('ProfilerStep#', 'step ')}\n{s['step_s']:.1f} s step"
        for lvl, s in gb
    ]
    yy = list(range(len(gb)))[::-1]
    busy = [100 * s["busy_any"] for _, s in gb]
    ax.barh(yy, busy, color=BLUE, height=0.55, label="GPU busy", **BAR)
    ax.barh(
        yy, [100 - v for v in busy], left=busy, color=GRAY, height=0.55, label="GPU idle", **BAR
    )
    for yi, v in zip(yy, busy, strict=True):
        ax.text(
            v / 2,
            yi,
            f"{v:.0f} % busy",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_yticks(yy, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the step (%)")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    s5 = l5[0]
    sub_ms = (
        s5["idle_by_gap_size_s"]["<20us"]
        + s5["idle_by_gap_size_s"]["20-100us"]
        + s5["idle_by_gap_size_s"]["0.1-1ms"]
    )
    finish(
        fig,
        "09_gpu_busy",
        "At L5 the GPU idles ~40 % of each step; at L7 it is busy 99 %",
        f"L5, step 2: {s5['idle_s']:.2f} s idle in {s5['num_gaps']:,} gaps between kernels, "
        f"{sub_ms:.2f} s of it in gaps under 1 ms. Waiting for data: "
        f"{max(s['dataloader_wait_ms'] for s in l5):.2f} ms per step.",
        "PyTorch profiler traces, scripts/trace_gpu_idle.py. L5: Jonah's run i3f3hngb (data "
        "parallel, 1 node). L7: PROF7 q4ag4rlz.",
        left=0.15,
    )

    with open(OUT / "numbers.json", "w") as handle:
        json.dump(numbers, handle, indent=1)
    for r in lv + [w4]:
        report(
            f"L{r['level']} {r['run']}: step {r['step_median_s']:.2f} s, wait "
            f"{100 * r['loader_wait_share_mean']:.2f} % (worst rank "
            f"{100 * r['loader_wait_share_worst_rank']:.2f}), loader "
            f"{ {k: round(v, 1) for k, v in r['loader_seconds_per_sample'].items()} }, "
            f"tokens {r['source_tokens_per_sample_gib']:.2f} GiB, rank "
            f"{r['training_process_peak_gib']:.1f}, workers {r['workers_peak_gib']:.1f}, node CPU "
            f"{r['node_cpu_used_peak_gib']:.0f}/{r['node_cpu_total_gib']:.0f}, GPU "
            f"{r['gpu_peak_allocated_gib']:.1f}"
        )
    report("padding", {k: round(100 * v, 1) for k, v in pad_share.items()})
    report("L5 training samples/s", round(consume, 3))
    report("sp_loader", spl)
    report("sp_vs_dp", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in tr.items()})


if __name__ == "__main__":
    main()
