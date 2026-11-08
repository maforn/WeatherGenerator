"""READ (job 2766316): read-stage throughput and per-sample time split vs concurrent readers.

Usage: python read_stress_plot.py <read_stress dir>  (writes plots/read_stress.png and
read_stress_breakdown.json in that directory)
"""

import glob
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(sys.argv[1])
summary = {s["procs"]: s for s in json.loads((root / "summary.json").read_text()) if s.get("samples")}
rows = []
for procs, s in sorted(summary.items()):
    recs = [
        json.loads(line)
        for f in glob.glob(str(root / f"p{procs:03d}" / "*.jsonl"))
        for line in open(f)
        if line.strip()
    ]
    med = {k: st.median(r[k] for r in recs) for k in ("wall_s", "cpu_s", "read_s", "normalize_s")}
    rows.append(
        {
            "procs": procs,
            "samples": len(recs),
            "samples_per_s": s["samples_per_s"],
            "decoded_gb_per_s": s["decoded_gb_per_s"],
            "rchar_gb_per_s": s["rchar_gb_per_s"],
            "rchar_over_decoded": sum(r["rchar"] for r in recs)
            / sum(r["decoded_bytes"] for r in recs),
            "node_cpu_busy_mean": s["node_cpu_busy_mean"],
            "node_iowait_mean": s["node_iowait_mean"],
            **{k + "_median": v for k, v in med.items()},
            "off_cpu_s": med["wall_s"] - med["cpu_s"],
        }
    )
(root / "read_stress_breakdown.json").write_text(json.dumps(rows, indent=1))

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a8a7a2"
P = [r["procs"] for r in rows]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF)
for ax in (ax1, ax2):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

# left: throughput vs readers, with linear scaling from one reader
one = rows[0]["samples_per_s"]
ax1.plot(P, [one * p for p in P], color=GRAY, lw=1.5, ls="--")
ax1.text(70, one * 70 + 0.4, "linear from 1 reader", color=INK2, fontsize=8.5, rotation=0)
ax1.plot(P, [r["samples_per_s"] for r in rows], color=BLUE, lw=2, marker="o", ms=8,
         markeredgecolor=SURF, markeredgewidth=2)
for r in rows:
    ax1.annotate(f"{r['samples_per_s']:.1f}", (r["procs"], r["samples_per_s"]),
                 textcoords="offset points", xytext=(10, 4) if r["procs"] == 1 else (0, -16),
                 ha="left" if r["procs"] == 1 else "center", fontsize=8.5, color=INK)
ax1.axhline(4 / 4.1, color=ORANGE, lw=1.5)
ax1.text(196, 4 / 4.1 + 0.35, "L5 training needs ~1/s\n(4 GPUs, 4.1 s step)", color=INK,
         fontsize=8.5, ha="right", va="bottom")
ax1.set_ylim(0, 17)
ax1.set_xlim(-6, 200)
ax1.set_xticks(P)
ax1.set_xlabel("concurrent reader processes (1 node, 288 cores)", color=INK2, fontsize=9)
ax1.set_title("Samples read per second", loc="left", fontsize=10, color=INK)

# right: per-sample time split (medians)
x = list(range(len(rows)))
read = [r["read_s_median"] for r in rows]
norm = [r["normalize_s_median"] for r in rows]
off = [r["off_cpu_s"] for r in rows]
ax2.bar(x, read, width=0.6, color=BLUE, edgecolor=SURF, linewidth=2, label="read (zarr/Lustre + decode)")
ax2.bar(x, norm, width=0.6, bottom=read, color=ORANGE, edgecolor=SURF, linewidth=2,
        label="normalize (numpy only, no I/O)")
for i, r in enumerate(rows):
    ax2.text(i, r["wall_s_median"] + 0.35, f"{r['wall_s_median']:.1f} s\noff-CPU {off[i]:.1f} s",
             ha="center", fontsize=8, color=INK)
ax2.set_xticks(x)
ax2.set_xticklabels([str(p) for p in P])
ax2.set_ylim(0, 17)
ax2.set_xlabel("concurrent reader processes", color=INK2, fontsize=9)
ax2.set_title("Seconds per sample (median), split by stage", loc="left", fontsize=10, color=INK)
ax2.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)

fig.suptitle("Read stage on 1 node: throughput stops at ~13.6 samples/s (128 readers); the limit "
             "is on the CPU side, not storage", x=0.01, ha="left", fontsize=11.5, color=INK,
             fontweight="bold")
fig.text(0.01, 0.905, "READ, job 2766316: each reader reads distinct L5-config windows "
         "(_get_data_windows, 2 forecast steps, 0.62 GB decoded per sample, ~90 s per level).\n"
         "Off-CPU time per sample stays ~1.2 s at every level, while normalize, which does no "
         "I/O, slows 9x: shared memory bandwidth/cache, not storage. 256 readers skipped "
         "(memory).", fontsize=8.5, color=INK2, va="top")
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig(root / "plots" / "read_stress.png", dpi=150, facecolor=SURF)
print(json.dumps(rows, indent=1))
