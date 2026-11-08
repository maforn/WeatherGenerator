"""Plot PROF7's step-2 GPU timeline (coarse phases) and GPU time per part; write a JSON summary."""

import json
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

seg_pkl, labels_pkl, out_png, out_json = sys.argv[1:5]
segs = pickle.load(open(seg_pkl, "rb"))
lab = pickle.load(open(labels_pkl, "rb"))
T0, T1 = lab["T0"], lab["T1"]

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = {"fe_fwd": "#2a78d6", "fe_rec": "#eb6834", "fe_bwd": "#1baf7a", "enc": "#eda100",
     "rest": "#6f6e69"}
NAME = {"fe_fwd": "Forecast engine forward", "fe_rec": "Forecast engine recompute",
        "fe_bwd": "Forecast engine backward", "enc": "Encoder (fwd, recompute, bwd)",
        "rest": "Rest of backward, optimizer"}

ENC_BWD_FROM = None  # global-attention backward after the last FE recompute belongs to the encoder
rows = []
for s in segs:
    a, b, g = (s[0] - T0) / 1e6, (s[1] - T0) / 1e6, s[2]
    rows.append([a, b, g])
last_fe = max(r[1] for r in rows if r[2] == "FE recompute")
first_enc_rec = min(r[0] for r in rows if r[2] == "encoder global recompute")
cat_rows = []
for a, b, g in rows:
    if g == "FE forward":
        c = "fe_fwd"
    elif g == "FE recompute":
        c = "fe_rec"
    elif g == "global-attention backward":
        c = "fe_bwd" if a < first_enc_rec else "enc"
    elif g.startswith("encoder global") or (g == "other forward" and a < 6):
        c = "enc"
    else:
        c = "rest"
    cat_rows.append((a, b, c))

# per-part wall time; split FE recompute into the whole-step rerun (>5 s) and the per-group reruns
parts = {}
for a, b, c in cat_rows:
    key = c
    if c == "fe_rec":
        key = "fe_rec_step" if b - a > 5 else "fe_rec_group"
    parts[key] = parts.get(key, 0.0) + (b - a)
step = (T1 - T0) / 1e6

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.4), gridspec_kw={"height_ratios": [1.25, 1]},
                               facecolor=SURF)
for ax in (ax1, ax2):
    ax.set_facecolor(SURF)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)

# --- timeline band
for a, b, c in cat_rows:
    ax1.barh(0, b - a, left=a, height=0.5, color=C[c], edgecolor=SURF, linewidth=1.5)
ax1.set_xlim(0, step)
ax1.set_ylim(-1.55, 1.45)
ax1.set_yticks([])
ax1.set_xlabel("seconds from the start of the step (GPU 0)", color=INK2, fontsize=9)


def bracket(ax, a, b, y, text, below=False):
    d = -0.12 if below else 0.12
    ax.plot([a, a, b, b], [y - d, y, y, y - d], color=INK2, lw=1)
    ax.text((a + b) / 2, y + (-0.1 if below else 0.08), text, ha="center",
            va="top" if below else "bottom", fontsize=8.5, color=INK)


fe_fwd = sorted((a, b) for a, b, c in cat_rows if c == "fe_fwd")
fwd_a, fwd_b = fe_fwd[0][0], fe_fwd[-1][1]
bracket(ax1, fwd_a, fwd_b, 0.42, f"forward, steps 1+2\n2 × {(fwd_b - fwd_a) / 2:.1f} s")
outer = sorted((a, b) for a, b, c in cat_rows if c == "fe_rec" and b - a > 5)
for i, (a, b) in enumerate(outer):
    bracket(ax1, a, b, 0.42, f"rerun of step {2 - i}\n{b - a:.1f} s")
first_enc_bwd = min(a for a, b, c in cat_rows if c == "enc" and a > outer[-1][1])
for i, (oa, ob) in enumerate(outer):
    end = outer[i + 1][0] if i + 1 < len(outer) else first_enc_bwd
    bracket(ax1, ob, end, -0.42, f"backward of step {2 - i}, {end - ob:.0f} s:\n"
            "8 × (rerun 2 blocks 1.5 s + backward 3.8 s)", below=True)
ax1.set_ylim(-1.0, 0.9)
handles = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in C]
fig.legend(handles, [NAME[k] for k in C], ncol=5, loc="upper left", bbox_to_anchor=(0.01, 0.885),
           frameon=False, fontsize=8.5, handlelength=1.2, columnspacing=1.4, labelcolor=INK)

# --- per-part bars
order = [("fe_bwd", "Forecast engine backward", "fe_bwd"),
         ("fe_fwd", "Forecast engine forward", "fe_fwd"),
         ("fe_rec_step", "Forecast engine rerun, whole step\n(fe_checkpoint_per_step)", "fe_rec"),
         ("fe_rec_group", "Forecast engine rerun, per 2 blocks", "fe_rec"),
         ("enc", "Encoder (fwd, recompute, bwd)", "enc"),
         ("rest", "Rest of backward, optimizer", "rest")]
ys = list(range(len(order)))[::-1]
for y, (k, label, c) in zip(ys, order):
    v = parts.get(k, 0.0)
    ax2.barh(y, v, height=0.62, color=C[c])
    ax2.text(v + 0.8, y, f"{v:.0f} s ({100 * v / step:.0f}%)", va="center", fontsize=9, color=INK)
ax2.set_yticks(ys)
ax2.set_yticklabels([o[1] for o in order], fontsize=9, color=INK)
ax2.set_xlim(0, max(parts.values()) * 1.18)
ax2.set_xlabel("seconds of the 160 s step", color=INK2, fontsize=9)
ax2.xaxis.grid(True, color=GRID, lw=0.8)
ax2.set_axisbelow(True)
ax2.tick_params(axis="y", length=0)

fig.suptitle("Level 7: one training step is 160 s of GPU work, 99% busy; global attention over "
             "196,608 cells is 86% of it", x=0.01, ha="left", fontsize=12, color=INK,
             fontweight="bold")
fig.text(0.01, 0.925, "PROF7 q4ag4rlz (job 2764672): PyTorch profiler, step 2 on GPU 0, same "
         "sample as SNAP7 ft98r6re; 1 node, 4 GH200, 2 forecast steps, 16 dense-attention forecast "
         "blocks.\nSNAP7's forward pauses (12–15 s) are the two forecast-step forwards; its backward "
         "pauses (34–41 s) are the two backward stretches below the band.", fontsize=8.5,
         color=INK2, va="top")
fig.tight_layout(rect=(0, 0, 1, 0.83))
fig.savefig(out_png, dpi=150, facecolor=SURF)

json.dump({"step_s": round(step, 2), "parts_s": {k: round(v, 2) for k, v in parts.items()},
           "segments": [(round(a, 2), round(b, 2), c) for a, b, c in cat_rows]},
          open(out_json, "w"), indent=1)
print({k: round(v, 1) for k, v in parts.items()}, "sum", round(sum(parts.values()), 1),
      "step", round(step, 1))
