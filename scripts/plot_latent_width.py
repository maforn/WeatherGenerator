#!/usr/bin/env python3
"""Slide plots of the latent-width study (L5, 100 steps, dev 4f11a5be + config/optimizations.yml).

Every value comes from a run's own files in experiments/martin/latent/<run_id>/, fetched with
scripts/fetch_latent_run.sh: the per-step training metrics, Slurm accounting (sacct), the memory
sampler that ran inside the job (sample.sh, every 10 s) and the parameter count of the saved
checkpoint. No estimates, projections or what-ifs. The run id is on every figure; the values
shown are written to experiments/martin/latent/figures/numbers.json.

  WeatherGenerator/.venv/bin/python scripts/plot_latent_width.py
"""

import csv
import json
import statistics
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "martin" / "latent"
OUT = ROOT / "figures"
GIB = 2**30
SKIP = 10  # steps left out of the step-time statistics (start-up, first batches)
HALF = 50  # second half of the run, over which the mean loss difference is taken
TARGET_GIB = 20  # Yuejiang: keep the GPU peak under 20 GiB (SUPERVISOR_FEEDBACK.md)

# fixed order and colour per configuration (reference categorical slots 1-3)
CONFIGS = [
    ("base2048", "2048 (current)", "#2a78d6"),
    ("global4096", "global 4096", "#eb6834"),
    ("both4096", "local + global 4096", "#1baf7a"),
]
GRAY = "#9aa3ae"
# the DEV check run: same code, configuration and seed as base2048, so the gap between the two
# is run-to-run noise (non-deterministic GPU kernels); drawn only as a reference in the loss plot
REPEAT = ("dev", "2048 repeat", GRAY)
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

SETUP = (
    "L5, 1 node x 4 GPUs (data parallel, FSDP), 100 steps, 1 sample per GPU per step, seed 1, "
    "12 loader workers, dev 4f11a5be with every option in config/optimizations.yml "
    "(incl. fsdp_cpu_offload, ema_on_cpu)"
)


def load_runs():
    """The fetched runs by configuration name (only configurations that have been fetched)."""
    runs = {}
    for meta_file in sorted(ROOT.glob("*/run.json")):
        meta = json.loads(meta_file.read_text())
        name, run = meta["name"], meta["run_id"]
        d = meta_file.parent
        lines = [json.loads(x) for x in (d / f"{run}_train_metrics.json").read_text().splitlines()]
        # one record per training step; the two records after validation are checkpoint saves
        train = [r for r in lines if r.get("stage", "train") == "train" and "loss_avg_mean" in r]
        val = [r for r in lines if r.get("stage") == "val"]
        stamps = [r["weathergen.timestamp"] / 1000 for r in train]
        mb = [r.get("performance.throughput.global.mb", 0.0) for r in train]
        rec = {
            "name": name,
            "run": run,
            "job": meta["train_job"],
            "loss": [r["loss_avg_mean"] for r in train],
            "lr": [r.get("learning_rate") for r in train],
            "step_s": [b - a for a, b in zip(stamps, stamps[1:], strict=False)],
            "t0": stamps[0],
            "mb_per_step": [b - a for a, b in zip(mb, mb[1:], strict=False)],
            "gpu_alloc": [r.get("performance.memory.train_global.max_allocated_gib") for r in train],
            "gpu_resv": [r.get("performance.memory.train_global.max_reserved_gib") for r in train],
            "val_loss": val[-1].get("LossPhysical.loss_avg") if val else None,
            "sacct": read_sacct(d / "sacct.txt"),
            "params": json.loads((d / "params.json").read_text())
            if (d / "params.json").stat().st_size
            else None,
            "mem": read_sampler(next(iter(d.glob("*.mem.csv")), None)),
        }
        runs[name] = rec
    return runs


def read_sacct(path):
    rows = list(csv.DictReader(path.open(), delimiter="|"))
    out = {"state": rows[0]["State"], "elapsed": rows[0]["Elapsed"], "node": rows[0]["NodeList"]}
    for row in rows:
        if row["JobID"].endswith(".0"):  # the srun step with the 4 training ranks
            out["maxrss_gib"] = kib(row["MaxRSS"]) / GIB
            out["averss_gib"] = kib(row["AveRSS"]) / GIB
    return out


def kib(value):
    units = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
    return float(value[:-1]) * units[value[-1]] if value and value[-1] in units else float(value or 0)


def read_sampler(path):
    if path is None or not path.exists():
        return None
    rows = [r for r in csv.DictReader(path.open()) if r.get("mem_used_b")]
    if not rows:
        return None

    def gpus(field):
        vals = [[float(x) for x in r[field].split(";") if x] for r in rows]
        return vals if any(vals) else None

    return {
        "t": [float(r["unix_s"]) for r in rows],
        "used_gib": [float(r["mem_used_b"]) / GIB for r in rows],
        "shared_gib": [float(r["mem_shared_b"]) / GIB for r in rows],
        "total_gib": float(rows[0]["mem_total_b"]) / GIB,
        "rank_rss_gib": [float(r["rank_rss_kb"]) * 1024 / GIB for r in rows],
        "worker_rss_gib": [float(r["worker_rss_kb"]) * 1024 / GIB for r in rows],
        "gpu_used_mib": gpus("gpu_mem_used_mib"),
    }


def finish(fig, name, title, subtitle, source, bottom_extra=0.0):
    """Title, wrapped subtitle and source line, and margins that fit them."""
    sub = textwrap.fill(subtitle, 112)
    src = "\n".join(textwrap.fill(part, 128) for part in source.split("\n"))
    n_sub, n_src = sub.count("\n") + 1, src.count("\n") + 1
    fig.text(0.02, 0.975, title, ha="left", va="top", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.02, 0.905, sub, ha="left", va="top", fontsize=12.5, color=MUTED, linespacing=1.4)
    fig.text(0.02, 0.015, src, ha="left", va="bottom", fontsize=10, color=MUTED,
             family="monospace", linespacing=1.4)
    fig.subplots_adjust(top=0.80 - 0.045 * n_sub, bottom=0.115 + 0.035 * n_src + bottom_extra,
                        left=0.07, right=0.98, wspace=0.28)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def present(runs):
    return [(n, label, color, runs[n]) for n, label, color in CONFIGS if n in runs]


def source_line(runs, extra=""):
    ids = "  ".join(f"{label}: {r['run']} (job {r['job']})" for _, label, _, r in present(runs))
    return f"Runs  {ids}\n{SETUP}" + (f"\n{extra}" if extra else "")


def label_bars(ax, bars, fmt):
    top = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * top, fmt.format(h), ha="center",
                va="bottom", color=INK, fontsize=12)


def bar_axis(ax, rows, values, fmt, ylabel):
    xs = range(len(rows))
    bars = ax.bar(xs, values, width=0.6, color=[c for _, _, c, _ in rows])
    ax.set_xticks(list(xs), [textwrap.fill(label, 11) for _, label, _, _ in rows])
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel(ylabel)
    label_bars(ax, bars, fmt)


def plot_loss(runs, numbers):
    rows = present(runs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [3, 2]})
    for _, label, color, r in rows:
        ax1.plot(range(len(r["loss"])), r["loss"], color=color, lw=2, label=label)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("training loss (MSE)")
    ax1.set_title("Loss per step (same batches in every run)", loc="left")
    ax1.legend(loc="upper right", fontsize=11)
    base = runs.get("base2048")
    rep = runs.get(REPEAT[0])
    mean_diff = {}
    if base:
        ax2.axhline(0, color=MUTED, lw=1)
        others = [(REPEAT[1], GRAY, rep)] if rep else []
        others += [(label, color, r) for name, label, color, r in rows if name != "base2048"]
        lo = hi = 0.0
        for label, color, r in others:
            n = min(len(r["loss"]), len(base["loss"]))
            diff = [r["loss"][i] - base["loss"][i] for i in range(n)]
            ax2.plot(range(SKIP, n), diff[SKIP:], color=color, lw=1.4 if r is rep else 1.8,
                     label=f"{label} - 2048" + (" (noise)" if r is rep else ""))
            lo, hi = min(lo, *diff[SKIP:]), max(hi, *diff[SKIP:])
            mean_diff[r["run"]] = statistics.mean(diff[HALF:])
        ax2.set_ylim(lo - 0.05 * (hi - lo), hi + 0.75 * (hi - lo))
        ax2.set_xlabel("training step")
        ax2.set_ylabel("loss difference to 2048")
        ax2.set_title(f"Paired difference, steps {SKIP}-99", loc="left")
        ax2.legend(loc="upper right", fontsize=10)
    last = {r["run"]: {"first": r["loss"][0], "last": r["loss"][-1],
                       "mean_last10": statistics.mean(r["loss"][-10:]), "val": r["val_loss"]}
            for _, _, _, r in rows}
    if rep:
        last[rep["run"]] = {"first": rep["loss"][0], "last": rep["loss"][-1],
                            "mean_last10": statistics.mean(rep["loss"][-10:]),
                            "val": rep["val_loss"], "note": "2048 repeat (DEV check run)"}
    for run, d in mean_diff.items():
        last[run][f"mean_diff_to_2048_steps_{HALF}_99"] = d
    numbers["loss"] = last
    shown = [(label, r) for _, label, _, r in rows] + ([(REPEAT[1], rep)] if rep else [])
    sub = f"Last step / validation / mean difference to 2048 over steps {HALF}-99:  " + ";  ".join(
        f"{label} {r['loss'][-1]:.3f} / "
        + (f"{r['val_loss']:.3f}" if r["val_loss"] is not None else "-")
        + (f" / {mean_diff[r['run']]:+.3f}" if r["run"] in mean_diff else " / -")
        for label, r in shown
    )
    lr = [x for x in rows[0][3]["lr"] if x is not None]
    peak_step = next(i for i, x in enumerate(lr) if x >= 0.999 * max(lr))
    noise = (f"Noise reference (gray): 2048 repeat {rep['run']} (job {rep['job']}), same code, "
             f"configuration and seed as 2048. " if rep else "")
    finish(fig, "latent_loss", "Training loss at 2048 vs 4096 latent width", sub,
           source_line(runs, f"{noise}Logged learning rate: {lr[0]:.1e} at step 0, peak "
                       f"{max(lr):.1e} from step {peak_step}, {lr[-1]:.1e} at the last step (at 100 "
                       f"steps the scheduler shortens the warm-up to 10 steps and the cooldown to "
                       f"5); validation = 4 samples."))


def plot_step_time(runs, numbers):
    rows = present(runs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [3, 2]})
    for _, label, color, r in rows:
        ax1.plot(range(1, len(r["step_s"]) + 1), r["step_s"], color=color, lw=1.6, label=label)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("step time (s)")
    ax1.set_title("Time per step", loc="left")
    ax1.set_ylim(0, max(max(r["step_s"]) for _, _, _, r in rows) * 1.3)
    ax1.legend(loc="upper left", fontsize=10.5, ncol=3)
    meds = [statistics.median(r["step_s"][SKIP:]) for _, _, _, r in rows]
    bar_axis(ax2, rows, meds, "{:.2f} s", "median step time (s)")
    ax2.set_title(f"Median, steps {SKIP + 1}-100", loc="left")
    numbers["step_time_s"] = {
        r["run"]: {"median": m, "mean": statistics.mean(r["step_s"][SKIP:]),
                   "max": max(r["step_s"][SKIP:]), "wall_train_s": sum(r["step_s"])}
        for (_, _, _, r), m in zip(rows, meds, strict=True)
    }
    base = meds[0] if rows and rows[0][0] == "base2048" else None
    sub = "; ".join(
        f"{label}: {m:.2f} s" + (f" ({m / base:.2f}x)" if base and name != "base2048" else "")
        for (name, label, _, _), m in zip(rows, meds, strict=True)
    )
    finish(fig, "latent_step_time", "Step time at 2048 vs 4096 latent width",
           f"Median step time after step {SKIP}: {sub}.", source_line(runs,
           "Step time = difference of consecutive per-step metric timestamps (rank 0)."))


def plot_gpu(runs, numbers):
    rows = present(runs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [3, 2]})
    for _, label, color, r in rows:
        ax1.plot(range(len(r["gpu_alloc"])), r["gpu_alloc"], color=color, lw=1.6, label=label)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("GPU peak allocated in the step (GiB)")
    ax1.set_title("Per-step peak, max over the 4 GPUs", loc="left")
    ax1.legend(loc="lower right", fontsize=11)
    ax1.set_ylim(bottom=0)
    peaks = [max(v for v in r["gpu_alloc"] if v is not None) for _, _, _, r in rows]
    resv = [max(v for v in r["gpu_resv"] if v is not None) for _, _, _, r in rows]
    # nvidia-smi memory.used of the busiest GPU, from the in-job sampler: what the device
    # actually holds, i.e. PyTorch's reserved pool plus the CUDA context and NCCL buffers
    smi = [max(max(g) for g in r["mem"]["gpu_used_mib"]) / 1024
           if r["mem"] and r["mem"]["gpu_used_mib"] else 0.0 for _, _, _, r in rows]
    width = 0.26
    kinds = [("allocated (PyTorch)", peaks, 1.0), ("reserved (PyTorch)", resv, 0.6),
             ("used (nvidia-smi)", smi, 0.32)]
    for k, (kind, vals, alpha) in enumerate(kinds):
        xs = [i + (k - 1) * width for i in range(len(rows))]
        bars = ax2.bar(xs, vals, width=width * 0.92, color=[c for _, _, c, _ in rows],
                       alpha=alpha, edgecolor="none")
        ax2.bar([0], [0], color=MUTED, alpha=alpha, label=kind)  # neutral legend swatch
        for b, v in zip(bars, vals, strict=True):
            if v:
                ax2.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.1f}", ha="center",
                         va="bottom", fontsize=9.5, color=INK)
    ax2.axhline(TARGET_GIB, color=INK, lw=1.2, ls="--")
    ax2.text(-0.45, TARGET_GIB + 0.4, f"target: peak < {TARGET_GIB} GiB", fontsize=10, color=INK)
    ax2.set_xticks(range(len(rows)), [textwrap.fill(label, 11) for _, label, _, _ in rows])
    ax2.set_ylim(0, TARGET_GIB * 1.65)
    ax2.set_ylabel("GPU memory, run peak (GiB)")
    ax2.set_title("Run peak (of 96 GiB per GPU)", loc="left")
    ax2.legend(loc="upper left", fontsize=9.5, ncol=1)
    numbers["gpu_gib"] = {r["run"]: {"peak_allocated": p, "peak_reserved": q,
                                     "peak_nvidia_smi_used": m or None}
                          for (_, _, _, r), p, q, m in zip(rows, peaks, resv, smi, strict=True)}
    sub = "; ".join(f"{label}: {p:.1f} GiB allocated, {q:.1f} GiB reserved, {m:.1f} GiB used "
                    f"(nvidia-smi)" for (_, label, _, _), p, q, m in
                    zip(rows, peaks, resv, smi, strict=True))
    finish(fig, "latent_gpu_memory", "GPU memory at 2048 vs 4096 latent width", sub + ".",
           source_line(runs, "PyTorch max_memory_allocated / max_memory_reserved per step "
                       "(train_global = max over ranks); nvidia-smi memory.used of the busiest "
                       f"GPU, sampled every 10 s in the job. Target from Yuejiang: keep the "
                       f"peak under {TARGET_GIB} GiB."))


def plot_host(runs, numbers):
    rows = present(runs)
    sampled = [(n, label, c, r) for n, label, c, r in rows if r["mem"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [3, 2]})
    if sampled:
        for _, label, color, r in sampled:
            m = r["mem"]
            t = [(x - r["t0"]) / 60 for x in m["t"]]
            ax1.plot(t, m["used_gib"], color=color, lw=2, label=f"{label}, node used")
            ax1.plot(t, m["rank_rss_gib"], color=color, lw=1.4, ls="--",
                     label=f"{label}, 4 ranks RSS")
        ax1.set_xlabel("minutes since the first training step")
        ax1.set_ylabel("host memory (GiB)")
        ax1.set_title("Node memory used, CPU + GPU HBM (`free`)", loc="left")
        ax1.set_ylim(0, max(max(r["mem"]["used_gib"]) for _, _, _, r in sampled) * 1.55)
        ax1.legend(loc="upper left", fontsize=9, ncol=2)
    else:
        ax1.text(0.5, 0.5, "no in-job memory samples for these runs", ha="center",
                 transform=ax1.transAxes, color=MUTED)
        ax1.set_axis_off()
    maxrss = [r["sacct"].get("maxrss_gib", 0.0) for _, _, _, r in rows]
    bar_axis(ax2, rows, maxrss, "{:.1f}", "MaxRSS per training task (GiB)")
    ax2.set_title("Slurm MaxRSS, one rank", loc="left")
    numbers["host_gib"] = {
        r["run"]: {
            "slurm_maxrss_per_task": r["sacct"].get("maxrss_gib"),
            "slurm_averss_per_task": r["sacct"].get("averss_gib"),
            "node_used_peak": max(r["mem"]["used_gib"]) if r["mem"] else None,
            "node_shared_peak": max(r["mem"]["shared_gib"]) if r["mem"] else None,
            "ranks_rss_peak": max(r["mem"]["rank_rss_gib"]) if r["mem"] else None,
            "workers_rss_peak": max(r["mem"]["worker_rss_gib"]) if r["mem"] else None,
            "node_total": r["mem"]["total_gib"] if r["mem"] else None,
            # `free` on GH200 counts the GPUs' HBM; used minus the nvidia-smi total at the same
            # sample, max over samples
            "node_used_minus_gpu_peak": max(
                u - sum(g) / 1024 for u, g in zip(r["mem"]["used_gib"], r["mem"]["gpu_used_mib"],
                                                   strict=True) if g
            ) if r["mem"] and r["mem"]["gpu_used_mib"] else None,
        }
        for _, _, _, r in rows
    }
    parts = []
    for (_, label, _, r), mr in zip(rows, maxrss, strict=True):
        s = f"{label}: MaxRSS {mr:.1f} GiB per task"
        if r["mem"]:
            s += (f", node used peak {max(r['mem']['used_gib']):.0f} GiB, ranks "
                  f"{max(r['mem']['rank_rss_gib']):.0f} GiB")
        parts.append(s)
    finish(fig, "latent_host_memory", "Host memory at 2048 vs 4096 latent width",
           "; ".join(parts) + ".", source_line(runs,
           "MaxRSS: sacct, srun step .0 (counts the workers' shared pages per process). "
           "Node used: `free` inside the job (excludes page cache; includes the loader's /dev/shm "
           "and, on GH200, the 4 GPUs' HBM: node total 858 GiB, 478 of it CPU)."))


def plot_params(runs, numbers):
    rows = [(n, label, c, r) for n, label, c, r in present(runs) if r["params"]]
    if not rows:
        return
    groups = [("encoder", "encoder (local, adapter, global)"), ("forecast_engine", "forecast engine"),
              (None, "rest (decoders, heads)")]
    shades = ["#1f2937", "#6b7280", "#c4c9d0"]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    xs = list(range(len(rows)))
    bottom = [0.0] * len(rows)
    for (key, label), shade in zip(groups, shades, strict=True):
        vals = []
        for _, _, _, r in rows:
            mods = r["params"]["by_module"]
            known = sum(mods.get(k, 0) for k, _ in groups if k)
            vals.append((mods.get(key, 0) if key else r["params"]["total"] - known) / 1e6)
        bars = ax.bar(xs, vals, bottom=bottom, width=0.55, color=shade, label=label,
                      edgecolor="white", linewidth=2)
        for b, v, b0 in zip(bars, vals, bottom, strict=True):
            if v > 60:
                ax.text(b.get_x() + b.get_width() / 2, b0 + v / 2, f"{v:,.0f}", ha="center",
                        va="center", fontsize=11, color="white" if shade != shades[-1] else INK)
        bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
    for x, tot in zip(xs, bottom, strict=True):
        ax.text(x, tot * 1.01, f"{tot:,.0f} M", ha="center", va="bottom", fontsize=12.5,
                fontweight="bold", color=INK)
    ax.set_xticks(xs, [label for _, label, _, _ in rows])
    ax.set_ylabel("parameters (millions)")
    ax.legend(loc="upper left", fontsize=11, ncol=3)
    ax.set_ylim(0, max(bottom) * 1.22)
    numbers["params"] = {r["run"]: r["params"] for _, _, _, r in rows}
    sub = "; ".join(f"{label}: {r['params']['total'] / 1e6:,.0f} M parameters"
                    for _, label, _, r in rows)
    finish(fig, "latent_params", "Model size at 2048 vs 4096 latent width", sub + ".",
           source_line(runs, "Counted from each run's last saved checkpoint "
                       "(scripts/count_checkpoint_params.py)."))


def check_same_batches(runs, numbers):
    """Same seed: every run should get the same data volume at every step."""
    rows = present(runs)
    if len(rows) < 2:
        return
    ref = rows[0][3]["mb_per_step"]
    same = {r["run"]: all(abs(a - b) < 1e-6 for a, b in zip(ref, r["mb_per_step"], strict=False))
            for _, _, _, r in rows[1:]}
    numbers["same_mb_per_step_as_" + rows[0][3]["run"]] = same


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {k: v for k, v in load_runs().items() if k in {n for n, _, _ in [*CONFIGS, REPEAT]}}
    numbers = {"runs": {k: {"run": v["run"], "job": v["job"], **v["sacct"]} for k, v in runs.items()}}
    check_same_batches(runs, numbers)
    plot_loss(runs, numbers)
    plot_step_time(runs, numbers)
    plot_gpu(runs, numbers)
    plot_host(runs, numbers)
    plot_params(runs, numbers)
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
