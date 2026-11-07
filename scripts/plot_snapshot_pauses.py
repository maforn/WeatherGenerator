"""One training step at level 6 and 7: GPU memory allocated over time, pauses shaded."""
import pickle, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, SERIES, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#d9d8d4"

def load(path):
    snap = pickle.load(open(path, "rb"))
    tr = [e for e in max(snap["device_traces"], key=len) if "time_us" in e]
    has = lambda e, s: any(s in f"{f['filename']}:{f['name']}" for f in e.get("frames", []))
    ends = [i for i, e in enumerate(tr) if has(e, "ema.py") and (i == 0 or not has(tr[i-1], "ema.py"))]
    a, b = ends[1], ends[2]  # the second complete step
    alloc, ts, ys = 0, [], []
    for i, e in enumerate(tr[: b + 1]):
        if e["action"] == "alloc": alloc += e["size"]
        elif e["action"] == "free_completed": alloc -= e["size"]
        if i >= a: ts.append((e["time_us"] - tr[a]["time_us"]) / 1e6); ys.append(alloc / 2**30)
    pauses = [(ts[k-1], ts[k]) for k in range(1, len(ts)) if ts[k] - ts[k-1] >= 0.5]
    return ts, ys, pauses

panels = [(sys.argv[1], "Level 6 (k8eytats)"), (sys.argv[2], "Level 7 (ft98r6re)")]
fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), facecolor=SURFACE)
for ax, (path, name) in zip(axes, panels):
    ts, ys, pauses = load(path)
    step = ts[-1]; idle = sum(b - a for a, b in pauses)
    for a, b in pauses:
        ax.axvspan(a, b, color=SHADE, lw=0, zorder=0)
    ax.plot(ts, ys, color=SERIES, lw=1.4, drawstyle="steps-post", zorder=2)
    ax.set_facecolor(SURFACE); ax.set_xlim(0, step); ax.set_ylim(0, max(ys) * 1.12)
    ax.set_title(f"{name}: {step:.1f} s step, main thread idle {100*idle/step:.0f}% of it",
                 loc="left", color=INK, fontsize=11)
    ax.set_ylabel("GPU allocated (GiB)", color=INK2, fontsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
axes[-1].set_xlabel("seconds into the step", color=INK2, fontsize=9)
fig.text(0.01, 0.005, "Grey: the main thread makes no GPU allocation or free for at least 0.5 s "
         "(4 GH200, 1 node, CUDA allocator snapshot, rank 0, second profiled step).",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(sys.argv[3], dpi=150, facecolor=SURFACE)
print("saved", sys.argv[3])
