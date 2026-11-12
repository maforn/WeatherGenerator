#!/usr/bin/env python3
"""Slide plots of the 1000-step latent-width runs (L5, dev 4f11a5be + config/optimizations.yml).

2048 vs global 4096 vs global 4096 with half the peak learning rate, 1000 steps each, the
configured learning-rate schedule (256-step warm-up, 512-step cooldown) and validation on 64
samples. Every value comes from the runs' own files in experiments/martin/latent/<run_id>/
(fetched with scripts/fetch_latent_run.sh): per-step training metrics, the validation record,
Slurm accounting and the in-job memory sampler (sample.sh, every 10 s, with the CPU/HBM NUMA
columns). Rolling means are of measured per-step values; nothing is estimated. Values shown are
written to experiments/martin/latent/figures/numbers_long.json.

  WeatherGenerator/.venv/bin/python scripts/plot_latent_long.py
"""

import csv
import json
import statistics
import textwrap

import matplotlib.pyplot as plt
from plot_latent_width import GIB, INK, MUTED, OUT, ROOT, TARGET_GIB, label_bars, load_runs

SKIP = 10  # steps left out of the step-time median (start-up, first batches)
ROLL = 20  # rolling mean over this many steps for the loss curves
ROLL_DIFF = 50  # and for the paired difference
LAST = 100  # final window: the last 100 steps (the end of the cooldown)
BLUE, ORANGE = "#2a78d6", "#eb6834"
# colour follows the width (as in the 100-step figures); the line style / hatch marks the half LR
LONG = [
    ("long2048", "2048", BLUE, "-", None),
    ("longg4096", "global 4096", ORANGE, "-", None),
    ("longg4096lr", "global 4096, half LR", ORANGE, "--", "//"),
]
TICK = {"long2048": "2048", "longg4096": "global\n4096", "longg4096lr": "global 4096\nhalf LR"}
SETUP = (
    "L5, 1 node x 4 GPUs (data parallel, FSDP), 1000 steps, 1 sample per GPU per step, seed 1, "
    "12 loader workers, dev 4f11a5be with every option in config/optimizations.yml "
    "(incl. fsdp_cpu_offload, ema_on_cpu). LR: cosine warm-up 256 steps to the peak, linear "
    "cooldown to 0 over the last 512 steps. Validation: 64 samples after step 1000."
)


def finish(fig, name, title, subtitle, source, *, hspace=0.3, wspace=0.28, left=0.08):
    """Title, wrapped subtitle and source line; margins fixed in inches for any figure height."""
    sub = textwrap.fill(subtitle, 112)
    src = "\n".join(textwrap.fill(part, 128) for part in source.split("\n"))
    n_sub, n_src = sub.count("\n") + 1, src.count("\n") + 1
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.15 / h, title, ha="left", va="top", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.02, 1 - 0.55 / h, sub, ha="left", va="top", fontsize=12.5, color=MUTED,
             linespacing=1.4)
    fig.text(0.02, 0.1 / h, src, ha="left", va="bottom", fontsize=10, color=MUTED,
             family="monospace", linespacing=1.4)
    top_in = 0.55 + 0.25 * n_sub + 0.55  # title + subtitle + room for the panel titles
    bottom_in = 0.1 + 0.19 * n_src + 0.75  # source + room for the x labels
    fig.subplots_adjust(top=1 - top_in / h, bottom=bottom_in / h, left=left, right=0.98,
                        hspace=hspace, wspace=wspace)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def rolling(xs, n):
    out, s = [], 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= n:
            s -= xs[i - n]
        out.append(s / min(i + 1, n))
    return out


def numa(run_dir):
    """CPU-node and HBM-node memory from the sampler's NUMA columns (GiB), one entry per sample."""
    path = next(iter(run_dir.glob("*.mem.csv")))
    rows = [r for r in csv.DictReader(path.open()) if r.get("cpu_numa_total_b")]
    cpu_total = int(rows[0]["cpu_numa_total_b"]) / GIB
    cpu_used = [(int(r["cpu_numa_total_b"]) - int(r["cpu_numa_free_b"])) / GIB for r in rows]
    cpu_anon = [(int(r["cpu_numa_total_b"]) - int(r["cpu_numa_free_b"]) - int(r["cpu_numa_file_b"]))
                / GIB for r in rows]
    shm = [int(r["mem_shared_b"]) / GIB for r in rows]
    smi = [max(float(x) for x in r["gpu_mem_used_mib"].split(";") if x) / 1024
           for r in rows if r["gpu_mem_used_mib"]]
    return {"cpu_total": cpu_total, "cpu_used_peak": max(cpu_used),
            "cpu_free_min": cpu_total - max(cpu_used), "cpu_no_file_peak": max(cpu_anon),
            "shm_peak": max(shm), "smi_peak": max(smi)}


def source(rows, extra=""):
    ids = "  ".join(f"{label}: {r['run']} (job {r['job']})" for _, label, _, _, _, r in rows)
    return f"Runs  {ids}\n{SETUP}" + (f"\n{extra}" if extra else "")


def bars(ax, rows, values, fmt, ylabel, inside=False):
    xs = range(len(rows))
    b = ax.bar(xs, values, width=0.6, color=[c for _, _, c, _, _, _ in rows],
               edgecolor="white", linewidth=0)
    for patch, (_, _, _, _, hatch, _) in zip(b, rows, strict=True):
        patch.set_hatch(hatch)
    ax.set_xticks(list(xs), [TICK[n] for n, *_ in rows], fontsize=12)
    ax.set_ylim(0, max(values) * 1.2)
    if ylabel:
        ax.set_ylabel(ylabel)
    if inside:  # value inside the bar top, clear of a reference line just above
        for bar in b:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.97, fmt.format(
                bar.get_height()), ha="center", va="top", color="white", fontsize=12,
                fontweight="bold", bbox={"facecolor": bar.get_facecolor(), "edgecolor": "none",
                                         "pad": 1.5})
    else:
        label_bars(ax, b, fmt)
    return b


def plot_loss(rows, numbers):
    base = rows[0][5]["loss"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), gridspec_kw={"width_ratios": [3, 2]})
    (ax_loss, ax_val), (ax_diff, ax_lr) = axes
    for _, label, color, ls, _, r in rows:
        steps = range(len(r["loss"]))
        ax_loss.plot(steps, r["loss"], color=color, lw=0.6, alpha=0.18)
        ax_loss.plot(steps, rolling(r["loss"], ROLL), color=color, lw=2, ls=ls, label=label)
        ax_lr.plot(steps, [x or 0 for x in r["lr"]], color=color, lw=2, ls=ls, label=label)
    ax_loss.set_yscale("log")
    ax_loss.set_yticks([0.2, 0.3, 0.5, 0.7, 1.0], ["0.2", "0.3", "0.5", "0.7", "1.0"])
    ax_loss.minorticks_off()
    ax_loss.set_title(f"Training loss (rolling mean of {ROLL} steps; raw faint)", loc="left")
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("loss (MSE, log scale)")
    ax_loss.legend(loc="upper right")

    ax_diff.axhline(0, color=MUTED, lw=1)
    for _, label, color, ls, _, r in rows[1:]:
        diff = [a - b for a, b in zip(r["loss"], base, strict=True)]
        ax_diff.plot(range(len(diff)), rolling(diff, ROLL_DIFF), color=color, lw=2, ls=ls,
                     label=f"{label} - 2048")
    ax_diff.set_title(f"Paired difference to 2048 (rolling mean of {ROLL_DIFF} steps)", loc="left")
    ax_diff.set_xlabel("training step")
    ax_diff.set_ylabel("loss difference")
    ax_diff.legend(loc="upper right")

    ax_lr.set_title("Learning rate (logged)", loc="left")
    ax_lr.set_xlabel("training step")
    ax_lr.set_ylabel("learning rate")
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    val = [r["val_loss"] for *_, r in rows]
    bars(ax_val, rows, val, "{:.3f}", "")
    ax_val.set_title("Validation loss, 64 samples", loc="left")

    last = {r["name"]: statistics.fmean(r["loss"][-LAST:]) for *_, r in rows}
    diff_last = {r["name"]: statistics.fmean(a - b for a, b in zip(r["loss"][-LAST:], base[-LAST:],
                                                                    strict=True))
                 for *_, r in rows[1:]}
    windows = {}
    for a, b in [(0, 250), (250, 500), (500, 750), (750, 1000)]:
        windows[f"{a}-{b - 1}"] = {r["name"]: statistics.fmean(
            x - y for x, y in zip(r["loss"][a:b], base[a:b], strict=True)) for *_, r in rows[1:]}
    numbers["loss"] = {"mean_last_100": last, "diff_to_2048_last_100": diff_last,
                       "diff_to_2048_by_window": windows,
                       "val_loss": {r["name"]: r["val_loss"] for *_, r in rows},
                       "lr_peak": {r["name"]: max(x for x in r["lr"] if x) for *_, r in rows}}
    sub = "; ".join(
        f"{label}: last 100 steps {last[r['name']]:.3f}"
        + (f" ({diff_last[r['name']]:+.3f})" if r["name"] in diff_last else "")
        + f", validation {r['val_loss']:.3f}" for _, label, _, _, _, r in rows)
    finish(fig, "latent_long_loss", "1000 steps: the wider latent ends with a higher loss", sub + ".",
           source(rows, "Half LR: lr_max 2.5e-5 instead of 5e-5 (5e-5 instead of 1e-4 after the "
                        "sqrt(4 GPUs) scaling). One seed per configuration."), hspace=0.55)


def plot_cost(rows, numbers):
    fig, (ax_t, ax_g, ax_c) = plt.subplots(1, 3, figsize=(11, 6.2))
    med = [statistics.median(r["step_s"][SKIP:]) for *_, r in rows]
    bars(ax_t, rows, med, "{:.2f}", "")
    ax_t.set_title("Step time, median (s)", loc="left")
    mem = [numa(ROOT / r["run"]) for *_, r in rows]
    smi = [m["smi_peak"] for m in mem]
    bars(ax_g, rows, smi, "{:.1f}", "", inside=True)
    ax_g.axhline(TARGET_GIB, color=INK, lw=1.2, ls="--")
    ax_g.set_ylim(0, TARGET_GIB * 1.22)
    ax_g.text(2.4, TARGET_GIB * 1.03, f"target {TARGET_GIB}", color=INK, fontsize=11, ha="right")
    ax_g.set_title("GPU peak (GiB)", loc="left")
    used = [m["cpu_used_peak"] for m in mem]
    total = mem[0]["cpu_total"]
    bars(ax_c, rows, used, "{:.0f}", "", inside=True)
    ax_c.axhline(total, color=INK, lw=1.2, ls="--")
    ax_c.set_ylim(0, total * 1.17)
    ax_c.text(2.4, total * 1.03, f"node CPU memory {total:.0f}", color=INK, fontsize=11,
              ha="right")
    ax_c.set_title("CPU memory, peak (GiB)", loc="left")
    numbers["cost"] = {r["name"]: {"median_step_s": m_s, **m}
                       for (*_, r), m_s, m in zip(rows, med, mem, strict=True)}
    sub = "; ".join(
        f"{label}: {m_s:.2f} s per step, GPU {m['smi_peak']:.1f} GiB, CPU {m['cpu_used_peak']:.0f} "
        f"of {total:.0f} GiB (min free {m['cpu_free_min']:.0f})"
        for (_, label, *_), m_s, m in zip(rows, med, mem, strict=True))
    finish(fig, "latent_long_cost", "1000 steps: what the wider latent costs", sub + ".",
           source(rows, f"Step time: median of steps {SKIP}-999. GPU: nvidia-smi memory.used, "
                        "busiest GPU, sampled every 10 s in the job. CPU: used memory of the CPU "
                        "NUMA nodes (/sys/devices/system/node, MemTotal - MemFree, incl. page cache "
                        "and the loader's /dev/shm); free counts the GPUs' HBM too, so it is not "
                        "used here."), wspace=0.22)


def main():
    runs = load_runs()
    rows = [(n, label, c, ls, h, runs[n]) for n, label, c, ls, h in LONG if n in runs]
    numbers = {"runs": {r["name"]: {"run": r["run"], "job": r["job"], **r["sacct"]}
                        for *_, r in rows}}
    plot_loss(rows, numbers)
    plot_cost(rows, numbers)
    (OUT / "numbers_long.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
