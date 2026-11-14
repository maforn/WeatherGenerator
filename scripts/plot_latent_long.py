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


def plot_seeds(runs, numbers):
    """Paired 2048 vs global-4096 differences across the runs: noise, two seeds, 1000 steps."""
    pairs = [  # (label, 2048 run name, other run name, train window, colour)
        ("2048 repeat\n(noise)", "base2048", "dev", (50, 100), "#9aa3ae"),
        ("seed 1\n100 steps", "base2048", "global4096", (50, 100), ORANGE),
        ("seed 2\n100 steps", "seed2b2048", "seed2g4096", (50, 100), ORANGE),
        ("seed 1\n1000 steps", "long2048", "longg4096", (900, 1000), ORANGE),
    ]
    rows = [p for p in pairs if p[1] in runs and p[2] in runs]
    train = [statistics.fmean(b - a for a, b in zip(runs[x]["loss"][w[0]:w[1]],
                                                    runs[y]["loss"][w[0]:w[1]], strict=True))
             for _, x, y, w, _ in rows]
    val = [runs[y]["val_loss"] - runs[x]["val_loss"] for _, x, y, _, _ in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, vals, title in [(ax1, train, "Training loss: mean paired difference"),
                            (ax2, val, "Validation loss: difference")]:
        xs = range(len(rows))
        b = ax.bar(xs, vals, width=0.6, color=[c for *_, c in rows])
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(list(xs), [label for label, *_ in rows], fontsize=12)
        lo, hi = min(vals + [0]), max(vals)
        ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
        for bar, v in zip(b, vals, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (0.01 if v >= 0 else -0.01) *
                    (hi - lo) * 3, f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=12, color=INK)
        ax.set_title(title, loc="left")
        ax.set_ylabel("other run - 2048")
    numbers["seeds"] = {label.replace("\n", " "): {"runs": [runs[x]["run"], runs[y]["run"]],
                                                    "train_window": list(w), "train_diff": t,
                                                    "val_diff": v}
                        for (label, x, y, w, _), t, v in zip(rows, train, val, strict=True)}
    short = {"base2048": "repeat", "seed2b2048": "seed 2", "long2048": "1000 steps"}
    ids = [f"{'seed 1' if y == 'global4096' else short[x]} {runs[x]['run']}/{runs[y]['run']}"
           for _, x, y, *_ in rows]
    finish(fig, "latent_seeds", "Global 4096 vs 2048: the same direction in every pair",
           "Gray: two identical 2048 runs, the run-to-run noise. Orange: global 4096 minus 2048 on "
           "the same batches, for two seeds at 100 steps and for the 1000-step run. Training "
           "window: steps 50-99 (100 steps) or the last 100 steps (1000 steps).",
           f"Runs (2048/other)  {'  '.join(ids[:3])}\n{'  '.join(ids[3:])}  L5, 1 node x 4 GPUs, "
           "dev 4f11a5be + config/optimizations.yml.\nValidation: 4 samples (100 steps) or 64 "
           "(1000 steps). Seed 2 = rng_seed 2 (data order and initialisation).", wspace=0.3)


def plot_schedule(runs, numbers):
    """Global 4096 - 2048 gap by schedule: warm-up 256 or 512, 1000 or 2000 steps."""
    # colour = warm-up length, line style = number of steps; colours checked with the dataviz
    # validator (green is below 3:1 contrast, so every line also has a legend entry, a line style
    # and labelled values)
    green, purple = "#1baf7a", "#6250d6"
    pairs = [  # (label, short tag, 2048 run, 4096 run, colour, line style)
        ("1000 steps, warm-up 256", "1000/256", "long2048", "longg4096", green, "-"),
        ("1000 steps, warm-up 512 (both)", "1000/512", "warm2048", "warmg4096", purple, "-"),
        ("2000 steps, warm-up 256", "2000/256", "s2k2048", "s2kg4096", green, "--"),
        ("2000 steps, warm-up 512 for 4096 (2048 at 256)", "2000/512 4096 only", "s2k2048",
         "s2kwg4096", purple, "--"),
    ]
    rows = [p for p in pairs if p[2] in runs and p[3] in runs]
    if len(rows) < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5), gridspec_kw={"width_ratios": [3, 2]})
    (ax_diff, ax_end), (ax_lr, ax_loss) = axes
    out, rolled = {}, {}
    for label, _, a, b, color, ls in rows:
        ra, rb = runs[a], runs[b]
        diff = [y - x for x, y in zip(ra["loss"], rb["loss"], strict=True)]
        rolled[rb["run"]] = rolling(diff, ROLL_DIFF)
        ax_diff.plot(range(len(diff)), rolled[rb["run"]], color=color, lw=2, ls=ls,
                     label=label)
        ax_lr.plot(range(len(rb["lr"])), [x or 0 for x in rb["lr"]], color=color, lw=2, ls=ls,
                   label=label)
        out[label] = {"runs": [ra["run"], rb["run"]],
                      "loss_last_100": [statistics.fmean(ra["loss"][-LAST:]),
                                        statistics.fmean(rb["loss"][-LAST:])],
                      "train_diff_last_100": statistics.fmean(diff[-LAST:]),
                      "val_loss": [ra["val_loss"], rb["val_loss"]],
                      "val_diff": rb["val_loss"] - ra["val_loss"],
                      "diff_by_250_steps": [statistics.fmean(diff[s:s + 250])
                                            for s in range(0, len(diff), 250)],
                      "median_step_s": [statistics.median(ra["step_s"][SKIP:]),
                                        statistics.median(rb["step_s"][SKIP:])]}
    # loss spikes: steps above twice the median of the 50 steps before, in each 4096 run
    spikes = {}
    for *_, b, _, _ in rows:
        loss = runs[b]["loss"]
        spikes[runs[b]["run"]] = [(i, loss[i]) for i in range(50, len(loss))
                                  if loss[i] > 2 * statistics.median(loss[i - 50:i])]
    out["loss_spikes_4096"] = {r: [[i, v] for i, v in s] for r, s in spikes.items()}
    numbers["schedule"] = out
    ax_diff.axhline(0, color=MUTED, lw=1)
    for run, sp in spikes.items():
        if sp:
            i, v = max(sp, key=lambda t: t[1])
            j = max(range(i, min(i + ROLL_DIFF, len(rolled[run]))), key=lambda k: rolled[run][k])
            big = rolled[run][j] > 0.08  # tall bump: note to its right; small one: note below
            ax_diff.annotate(f"loss spike in {run}\n(step {i}: loss {v:.2f})",
                             xy=(j, rolled[run][j]),
                             xytext=(j + 130, rolled[run][j] - 0.03) if big else (j - 180, -0.04),
                             fontsize=9.5, color=INK, va="center",
                             arrowprops={"arrowstyle": "->", "color": MUTED})
    ax_diff.set_ylim(-0.055, 0.25)
    ax_diff.set_title(f"4096 minus 2048, training loss (rolling {ROLL_DIFF} steps)", loc="left")
    ax_diff.set_xlabel("training step")
    ax_diff.set_ylabel("loss difference")
    ax_diff.legend(loc="upper right", fontsize=9.5)
    ax_lr.set_title("Learning rate of the 4096 run (logged; lines as above)", loc="left")
    ax_lr.set_xlabel("training step")
    ax_lr.set_ylabel("learning rate")
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    xs = range(len(rows))
    tr = [out[label]["train_diff_last_100"] for label, *_ in rows]
    va = [out[label]["val_diff"] for label, *_ in rows]
    w = 0.4
    b1 = ax_end.bar([x - w / 2 for x in xs], tr, width=w, color=[r[4] for r in rows])
    b2 = ax_end.bar([x + w / 2 for x in xs], va, width=w, color="white",
                    edgecolor=[r[4] for r in rows], hatch="///", linewidth=1.5)
    for bars_, vals, dx in [(b1, tr, -0.06), (b2, va, 0.06)]:  # nudged apart so labels don't touch
        for bar, v in zip(bars_, vals, strict=True):
            ax_end.text(bar.get_x() + bar.get_width() / 2 + dx, v + 0.0004, f"{v:+.3f}",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
    ticks = {"1000/256": "1000\nwarm-up\n256", "1000/512": "1000\nwarm-up\n512",
             "2000/256": "2000\nwarm-up\n256", "2000/512 4096 only": "2000\n4096 at\n512"}
    ax_end.set_xticks(list(xs), [ticks[r[1]] for r in rows], fontsize=10)
    ax_end.set_ylim(0, max(tr + va) * 1.6)
    ax_end.set_title("Gap at the end", loc="left")
    ax_end.legend([b1[0], b2[0]], ["training, last 100 steps", "validation, 64 samples"],
                  loc="upper center", fontsize=10)
    ax_end.get_legend().legend_handles[0].set_color(MUTED)
    ax_end.get_legend().legend_handles[1].set_edgecolor(MUTED)

    # every run's own loss over its last 100 steps, grouped by run length
    groups = [  # (x, tick, 2048 run, 4096 run)
        (0, "1000 steps\nwarm-up 256", "long2048", "longg4096"),
        (1, "1000 steps\nwarm-up 512", "warm2048", "warmg4096"),
        (2.6, "2000 steps\nwarm-up 256", "s2k2048", "s2kg4096"),
        (3.6, "2000 steps\nwarm-up 512", None, "s2kwg4096"),
    ]
    for width, idx, color, marker in [("2048", 2, BLUE, "o"), ("global 4096", 3, ORANGE, "s")]:
        for k, seg in enumerate((groups[:2], groups[2:])):
            pts = [(g[0], statistics.fmean(runs[g[idx]]["loss"][-LAST:])) for g in seg
                   if g[idx] in runs]
            ax_loss.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2,
                         marker=marker, ms=8, label=width if k == 0 else None)
            for x, y in pts:
                right = x in (1, 3.6)
                ax_loss.text(x + (0.1 if right else -0.1), y, f"{y:.3f}",
                             ha="left" if right else "right", va="center", fontsize=10,
                             color=INK)
    ax_loss.set_xticks([g[0] for g in groups], [g[1].split("\n")[1].replace("warm-up ", "")
                                                for g in groups], fontsize=11)
    ax_loss.set_xlabel("warm-up steps")
    ax_loss.set_xlim(-0.8, 4.4)
    lo, hi = ax_loss.get_ylim()
    ax_loss.set_ylim(lo, hi + 0.012)
    for x, name, run in [(0.5, "1000 steps", "longg4096"), (3.1, "2000 steps", "s2kg4096")]:
        top = statistics.fmean(runs[run]["loss"][-LAST:])
        ax_loss.text(x, top + 0.009, name, ha="center", va="bottom", fontsize=11,
                     fontweight="bold", color=MUTED)
    ax_loss.set_title("Loss, last 100 steps of each run", loc="left")
    ax_loss.legend(loc="upper right", fontsize=10)

    get = lambda lab, k: out[lab][k] if lab in out else float("nan")  # noqa: E731
    best = {n: statistics.fmean(runs[r]["loss"][-LAST:]) for n, r in
            [("b1", "long2048"), ("g1", "warmg4096"), ("b2", "s2k2048"), ("g2", "s2kwg4096")]
            if r in runs}
    sub = (f"End gap, training loss: {get(rows[0][0], 'train_diff_last_100'):+.3f} at 1000 steps "
           f"and {get(pairs[2][0], 'train_diff_last_100'):+.3f} at 2000 steps with the default "
           f"warm-up. A 512-step warm-up lowers global 4096's loss by "
           f"{best.get('g1', 0) - statistics.fmean(runs['longg4096']['loss'][-LAST:]):+.3f} "
           f"(1000 steps) and "
           f"{best.get('g2', 0) - statistics.fmean(runs['s2kg4096']['loss'][-LAST:]):+.3f} "
           f"(2000 steps), but with each width on its best schedule 4096 is still "
           f"{best.get('g1', 0) - best.get('b1', 0):+.3f} and "
           f"{best.get('g2', 0) - best.get('b2', 0):+.3f} behind 2048.")
    ids = "  ".join(f"{tag}: {out[label]['runs'][0]}/{out[label]['runs'][1]}"
                    for label, tag, *_ in rows)
    finish(fig, "latent_schedule", "A longer warm-up helps global 4096, but it stays behind 2048",
           sub,
           f"Runs (steps/warm-up: 2048/4096)  {ids}\n"
           "L5, 1 node x 4 GPUs, dev 4f11a5be + config/optimizations.yml, seed 1, one run per "
           "configuration, peak LR 1e-4.\nAt 1000 steps warm-up 512 also shortens the cooldown "
           "from 512 to 256 steps (exp_latent_warmup512.yml); at 2000 steps the cooldown stays 512 "
           "(exp_latent_warmup512_only.yml).\nLoss spike = a step above twice the median of the 50 "
           "steps before it; validation: 64 samples.", hspace=0.55)


def main():
    runs = load_runs()
    rows = [(n, label, c, ls, h, runs[n]) for n, label, c, ls, h in LONG if n in runs]
    numbers = {"runs": {r["name"]: {"run": r["run"], "job": r["job"], **r["sacct"]}
                        for *_, r in rows}}
    plot_loss(rows, numbers)
    plot_cost(rows, numbers)
    plot_seeds(runs, numbers)
    plot_schedule(runs, numbers)
    (OUT / "numbers_long.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
