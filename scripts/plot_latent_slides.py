#!/usr/bin/env python3
"""Bare slide versions of the latent-width figures: no title or subtitle in the image, only short
panel labels and one line of run information. The message goes on the slide as keywords.

Same measured values as scripts/plot_latent_long.py (latent_long_cost.png, latent_schedule.png),
read from the same run folders. Writes slide_latent_cost.png and slide_latent_loss.png next to
them.

  WeatherGenerator/.venv/bin/python scripts/plot_latent_slides.py
"""

import statistics

import matplotlib.pyplot as plt
from plot_latent_long import BLUE, LAST, ORANGE, ROLL_DIFF, ROOT, SKIP, numa, rolling
from plot_latent_width import INK, MUTED, OUT, TARGET_GIB, label_bars, load_runs

GREEN, PURPLE = "#1baf7a", "#6250d6"
SIZE = (11, 4.7)  # fills a 16:9 slide under a title and one keyword line


def save(fig, name, runs_line):
    fig.text(0.01, 0.015, runs_line, ha="left", va="bottom", fontsize=9, color=MUTED,
             family="monospace")
    fig.subplots_adjust(top=0.88, bottom=0.17 + 0.035 * runs_line.count("\n"), left=0.07,
                        right=0.98, wspace=0.3)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def inside_top(ax, bar, fmt, color, box=False):
    """Value label just inside the bar top, clear of the reference line above it."""
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h * 0.97, fmt.format(h), ha="center", va="top",
            color=color, fontsize=11.5, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1} if box else None)


def gpu_panel(ax, rows, mem):
    """PyTorch's allocated peak (what the team's memory slides compare) next to nvidia-smi's
    memory used (everything the process holds on the GPU, sampled every 10 s)."""
    colors = [c for *_, c in rows]
    alloc = [max(v for v in r["gpu_alloc"] if v is not None) for _, r, _ in rows]
    smi = [m["smi_peak"] for m in mem]
    w = 0.38
    b1 = ax.bar([x - w / 2 for x in (0, 1)], alloc, width=w, color=colors)
    b2 = ax.bar([x + w / 2 for x in (0, 1)], smi, width=w, color="white", edgecolor=colors,
                hatch="///", linewidth=1.5)
    for bar in b1:
        inside_top(ax, bar, "{:.1f}", "white")
    for bar in b2:
        inside_top(ax, bar, "{:.1f}", INK, box=True)
    leg = ax.legend([b1[0], b2[0]], ["allocated (PyTorch)", "nvidia-smi"], loc="upper left",
                    fontsize=9.5, frameon=False)
    leg.legend_handles[0].set_color(MUTED)
    leg.legend_handles[1].set_edgecolor(MUTED)


def cost(runs):
    rows = [("2048", runs["long2048"], BLUE), ("global 4096", runs["longg4096"], ORANGE)]
    mem = [numa(ROOT / r["run"]) for _, r, _ in rows]
    panels = [
        ("Step time (s)", [statistics.median(r["step_s"][SKIP:]) for _, r, _ in rows], "{:.2f}"),
        ("GPU memory, peak (GiB)", None, None),
        ("CPU memory, peak (GiB)", [m["cpu_used_peak"] for m in mem], "{:.0f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=SIZE)
    for k, (ax, (title, values, fmt)) in enumerate(zip(axes, panels, strict=True)):
        if values is None:
            gpu_panel(ax, rows, mem)
        else:
            bars = ax.bar([0, 1], values, width=0.6, color=[c for *_, c in rows])
            ax.set_ylim(0, max(values) * 1.25)
            if k == 0:
                label_bars(ax, bars, fmt)
            else:
                for bar in bars:
                    inside_top(ax, bar, fmt, "white")
        ax.set_xticks([0, 1], [name for name, *_ in rows], fontsize=12)
        ax.set_title(title, loc="left")
    axes[1].axhline(TARGET_GIB, color=INK, lw=1.2, ls="--")
    axes[1].text(1.42, TARGET_GIB * 1.02, f"{TARGET_GIB} GiB target", ha="right", va="bottom",
                 fontsize=10, color=INK)
    axes[1].set_ylim(0, TARGET_GIB * 1.25)
    total = mem[0]["cpu_total"]
    axes[2].axhline(total, color=INK, lw=1.2, ls="--")
    axes[2].text(1.42, total * 1.02, f"node {total:.0f} GiB", ha="right", va="bottom",
                 fontsize=10, color=INK)
    axes[2].set_ylim(0, total * 1.22)
    save(fig, "slide_latent_cost",
         f"Runs: 2048 {rows[0][1]['run']}, global 4096 {rows[1][1]['run']}. "
         "L5, 1 node x 4 GPUs, 1000 steps, seed 1.")


def loss(runs):
    pairs = [  # (legend, 2048 run, 4096 run, colour, line style)
        ("1000 steps, warm-up 256", "long2048", "longg4096", GREEN, "-"),
        ("1000 steps, warm-up 512", "warm2048", "warmg4096", PURPLE, "-"),
        ("2000 steps, warm-up 256", "s2k2048", "s2kg4096", GREEN, "--"),
        ("2000 steps, warm-up 512 (4096 only)", "s2k2048", "s2kwg4096", PURPLE, "--"),
    ]
    fig, (ax, ax_end) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw={"width_ratios": [3, 2]})
    ax.axhline(0, color=MUTED, lw=1)
    train, val = [], []
    for label, a, b, color, ls in pairs:
        ra, rb = runs[a], runs[b]
        diff = [y - x for x, y in zip(ra["loss"], rb["loss"], strict=True)]
        rolled = rolling(diff, ROLL_DIFF)
        ax.plot(range(len(diff)), rolled, color=color, lw=2, ls=ls, label=label)
        train.append(statistics.fmean(diff[-LAST:]))
        val.append(rb["val_loss"] - ra["val_loss"])
        loss_b = rb["loss"]
        spikes = [i for i in range(50, len(loss_b))
                  if loss_b[i] > 2 * statistics.median(loss_b[i - 50:i])]
        if spikes:  # mark the bump in the rolling difference that the spike causes
            j = max(range(spikes[0], min(spikes[0] + ROLL_DIFF, len(rolled))),
                    key=lambda k: rolled[k])
            text_xy = (j + 80, rolled[j] + 0.025) if rolled[j] > 0.08 else (j + 40, -0.035)
            ax.annotate("spike", xy=(j, rolled[j]), xytext=text_xy, fontsize=10, color=INK,
                        va="center", arrowprops={"arrowstyle": "->", "color": MUTED})
    ax.set_ylim(-0.05, 0.27)
    ax.set_title("Loss gap, 4096 minus 2048", loc="left")
    ax.set_xlabel("training step")
    ax.legend(loc="upper center", fontsize=9.5, ncol=2, columnspacing=1.2)

    xs = range(len(pairs))
    w = 0.4
    b1 = ax_end.bar([x - w / 2 for x in xs], train, width=w, color=[p[3] for p in pairs])
    b2 = ax_end.bar([x + w / 2 for x in xs], val, width=w, color="white",
                    edgecolor=[p[3] for p in pairs], hatch="///", linewidth=1.5)
    for bars, vals, dx in [(b1, train, -0.06), (b2, val, 0.06)]:
        for bar, v in zip(bars, vals, strict=True):
            ax_end.text(bar.get_x() + bar.get_width() / 2 + dx, v + 0.0004, f"+{v:.3f}",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
    ax_end.set_xticks(list(xs), ["1000\n256", "1000\n512", "2000\n256", "2000\n512"], fontsize=10)
    ax_end.set_xlabel("steps / warm-up", fontsize=10)
    ax_end.set_ylim(0, max(train + val) * 1.6)
    ax_end.set_title("Gap at the end", loc="left")
    ax_end.legend([b1[0], b2[0]], ["training", "validation"], loc="upper center", fontsize=10,
                  ncol=2)
    ax_end.get_legend().legend_handles[0].set_color(MUTED)
    ax_end.get_legend().legend_handles[1].set_edgecolor(MUTED)
    ids = ", ".join(f"{runs[a]['run']}/{runs[b]['run']}" for _, a, b, *_ in pairs)
    save(fig, "slide_latent_loss",
         f"Runs (2048/4096): {ids}\nL5, 1 node x 4 GPUs, seed 1, peak LR 1e-4. "
         f"Rolling {ROLL_DIFF} steps; end = last {LAST} steps; validation 64 samples.")


def main():
    runs = load_runs()
    cost(runs)
    loss(runs)


if __name__ == "__main__":
    main()
