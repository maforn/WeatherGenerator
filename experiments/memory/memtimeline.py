# ruff: noqa: T201
# (command-line report: printing is the output)
"""Active-memory timelines from CUDA allocator snapshots, stacked by model part.

Like pytorch.org/memory_viz, but several runs side by side on one GiB scale and with
allocations grouped by the WeatherGenerator component that made them.

usage: python -I memtimeline.py out.png title "label=path.pickle" ["label=path.pickle" ...]
Needs numpy and matplotlib. Panels share the y-axis; x is the allocator event index (as in
memory_viz), so data-loading waits do not stretch the plot. Backward-pass allocations carry
no Python frame of ours; they are assigned to the component of the most recent allocation
that had one (in a checkpointed backward that is the block being recomputed), which is a
heuristic, not a measurement.
"""

import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GIB = 2**30
HBM_GIB = 97871 / 1024  # nvidia-smi memory.total of a JUPITER GH200 (job 2541772)
# pipeline order; colours are the reference categorical slots 1-6 in fixed order
CATS = [
    ("static (params, optimizer)", "#2a78d6"),
    ("batch on GPU", "#eb6834"),
    ("embedding (EmbeddingEngine)", "#1baf7a"),
    ("local + global assimilation", "#eda100"),
    ("forecast engine", "#e87ba4"),
    ("decoder", "#008300"),
    ("unattributed", "#b4b4b0"),
]
STATIC, BATCH, EMBED, ENC, FE, DEC, OTHER = range(len(CATS))


def category(frames):
    for f in frames:  # innermost first; skip generic layers until a component frame
        fn = f["filename"]
        if "/src/weathergen/" not in fn:
            continue
        rel, line, name = fn.split("/src/weathergen/")[-1], f["line"], f["name"]
        if rel.startswith("datasets/"):
            return BATCH
        if rel == "model/embeddings.py":
            return EMBED
        if rel == "model/engines.py":  # class line ranges in engines.py on dphpc/dev
            if line < 206:
                return EMBED
            if line < 546:
                return ENC
            if line < 642:
                return FE
            return DEC
        if rel == "model/encoder.py":
            return ENC
        if rel == "model/spatial_parallel.py" or name == "predict_decoders":
            return DEC
        if "fsdp" in rel:
            return STATIC
    for f in frames:
        if "/fsdp/" in f["filename"]:
            return STATIC
    return OTHER


def timeline(path, bins=1600):
    with open(path, "rb") as fh:
        s = pickle.load(fh)
    trace = s["device_traces"][0]
    end_active = sum(
        b["size"]
        for seg in s["segments"]
        for b in seg["blocks"]
        if b["state"] == "active_allocated"
    )
    freed_pre, live_end = 0, {}
    for e in trace:
        if e["action"] == "alloc":
            live_end[e["addr"]] = e["size"]
        elif e["action"] == "free_completed":
            if e["addr"] in live_end:
                del live_end[e["addr"]]
            else:
                freed_pre += e["size"]
    pre = end_active - sum(live_end.values()) + freed_pre

    cur = np.zeros(len(CATS))
    cur[STATIC] = pre
    live = {}
    times, rows = [], []
    last = OTHER
    for e in trace:
        a = e["action"]
        if a == "alloc":
            c = category(e["frames"])
            if c in (EMBED, ENC, FE, DEC):
                last = c
            elif c == OTHER:
                c = last
            live[e["addr"]] = (e["size"], c)
            cur[c] += e["size"]
        elif a == "free_completed":
            if e["addr"] in live:
                size, c = live.pop(e["addr"])
                cur[c] -= size
            else:
                cur[STATIC] -= e["size"]
        else:
            continue
        times.append(len(times) / 1000)
        rows.append(cur.copy())
    times, rows = np.array(times), np.array(rows) / GIB
    # keep, per time bin, the instant with the largest total so peaks survive downsampling
    total = rows.sum(1)
    idx = np.minimum((times / times[-1] * bins).astype(int), bins - 1)
    keep = [np.flatnonzero(idx == b) for b in range(bins)]
    keep = [k[np.argmax(total[k])] for k in keep if len(k)]
    return times[keep], rows[keep], total.max(), times[np.argmax(total)], rows[np.argmax(total)]


def main():
    out, title, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    runs = [(sp.rsplit("=", 1)[0], timeline(sp.rsplit("=", 1)[1])) for sp in specs]
    ymax = max(max(r[2] for _, r in runs) * 1.12, HBM_GIB * 1.06)
    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    ink, muted, grid = "#1f1f1e", "#6b6b68", "#e6e6e3"
    for ax, (label, (t, rows, peak, tpeak, at_peak)) in zip(axes, runs, strict=True):
        ax.set_facecolor("#fcfcfb")
        ax.stackplot(t, rows.T, colors=[c for _, c in CATS], linewidth=0)
        ax.axhline(HBM_GIB, color=muted, lw=1, ls=(0, (4, 3)))
        ax.text(
            t[-1],
            HBM_GIB,
            f"GPU memory {HBM_GIB:.1f} GiB ",
            ha="right",
            va="bottom",
            fontsize=8,
            color=muted,
        )
        ax.annotate(
            f"peak {peak:.1f} GiB",
            (tpeak, peak),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=ink,
            fontweight="bold",
        )
        ax.set_title(label, fontsize=11, color=ink, loc="left")
        ax.set_xlabel("allocator events (thousands)", fontsize=8.5, color=muted)
        ax.set_xlim(0, t[-1])
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
        parts = zip(CATS, at_peak, strict=True)
        print(
            f"{label}: peak {peak:.2f} GiB at event {tpeak * 1000:.0f}: "
            + ", ".join(f"{n.split(' (')[0]} {v:.1f}" for (n, _), v in parts)
        )
    axes[0].set_ylabel("active GPU memory, rank 0 (GiB)", fontsize=9, color=muted)
    axes[0].set_ylim(0, ymax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in CATS]
    fig.legend(
        handles,
        [n for n, _ in CATS],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=ink, fontweight="bold")
    fig.text(
        0.01,
        0.905,
        "Rank 0, 3 training steps. Backward-pass allocations are assigned to the "
        "component that allocated last (heuristic).",
        fontsize=8.5,
        color=muted,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.9))
    fig.savefig(out, dpi=200, facecolor="white")


if __name__ == "__main__":
    main()
