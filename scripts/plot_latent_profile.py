#!/usr/bin/env python3
"""Where the extra step time of the global-4096 latent goes: PyTorch profiler, 2048 vs global 4096.

Reads experiments/martin/latent/profile/<run>.json, written by scripts/latent_prof.py from the
rank-0 traces of the profiling runs f19gti52 (2048) and yjd08xgt (global 4096): L5, 100-step
setup, profiler over steps 10-13 (exp_latent_torchprof.yml). Values are means over the 4
profiled steps. Kernel durations are measured on the GPU; wall and CPU times include the
profiler's own overhead, so only differences between the two runs are compared.

  WeatherGenerator/.venv/bin/python scripts/plot_latent_profile.py
"""

import json
import statistics

import matplotlib.pyplot as plt
from plot_latent_width import INK, MUTED, OUT, ROOT

P = ROOT / "profile"
RUNS = json.loads((P / "runs.json").read_text())
ORDER = [("f19gti52", "#2a78d6"), ("yjd08xgt", "#eb6834")]
CLASSES = ["gemm", "nccl", "flash-attention forward", "flash-attention backward",
           "elementwise/other", "copy host->device", "copy device->host"]
LABEL = {"gemm": "GEMM", "nccl": "NCCL (FSDP)", "flash-attention forward": "FlashAttn fwd",
         "flash-attention backward": "FlashAttn bwd", "elementwise/other": "elementwise",
         "copy host->device": "H2D copy", "copy device->host": "D2H copy"}
PHASE = {"other (loss, optimizer, comm)": "other*", "forward model.forecast_engine": "fwd forecast",
         "backward": "backward", "recompute model.forecast_engine": "recompute forecast",
         "forward model.encoder.ae_global_engine": "fwd global eng.",
         "copies/memset": "copies"}


def mean(run, f):
    return statistics.fmean(f(s) for s in run["steps"])


def hbars(ax, names, labels, data, fmt):
    h = 0.38
    for i, (run, color) in enumerate(ORDER):
        vals = [data[run][n] for n in names]
        ys = [j + (i - 0.5) * h for j in range(len(names))]
        bars = ax.barh(ys, vals, height=h, color=color, label=RUNS[run]["label"])
        for b, v in zip(bars, vals, strict=True):
            ax.text(v + 0.012, b.get_y() + h / 2, fmt.format(v), va="center", fontsize=10.5,
                    color=INK)
    ax.set_yticks(range(len(names)), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


def main():
    runs = {r: json.loads((P / f"{r}.json").read_text()) for r, _ in ORDER}
    a, b = (r for r, _ in ORDER)
    cls = {r: {c: mean(runs[r], lambda s, c=c: s["kernel_s_by_class"].get(c, 0.0)) for c in CLASSES}
           for r in runs}
    phases = sorted({p for r in runs.values() for s in r["steps"] for p in s["kernel_s_by_phase"]})
    ph = {r: {p: mean(runs[r], lambda s, p=p: s["kernel_s_by_phase"].get(p, 0.0)) for p in phases}
          for r in runs}
    top = sorted(phases, key=lambda p: -(ph[b][p] - ph[a][p]))[:6]
    summ = {r: {k: mean(runs[r], lambda s, k=k: s[k]) for k in
                ("wall_s", "gpu_busy_s", "optimizer_step_cpu_s")} for r in runs}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6.4))
    hbars(ax1, CLASSES, [LABEL[c] for c in CLASSES], cls, "{:.2f}")
    ax1.set_title("GPU kernel time per step by kind (s)", loc="left")
    ax1.set_xlim(0, max(max(v.values()) for v in cls.values()) * 1.25)
    ax1.legend(loc="lower right")
    hbars(ax2, top, [PHASE.get(p, p.replace("model.encoder.", "").replace("model.", ""))
                     for p in top], ph, "{:.2f}")
    ax2.set_title("Largest increases by part (s)", loc="left")
    ax2.set_xlim(0, max(max(ph[r][p] for p in top) for r in runs) * 1.3)

    numbers = {"runs": RUNS, "per_step_mean_s": summ, "kernel_s_by_class": cls,
               "kernel_s_by_phase": ph}
    (OUT / "numbers_profile.json").write_text(json.dumps(numbers, indent=1))
    d = {k: summ[b][k] - summ[a][k] for k in summ[a]}
    sub = (f"Per profiled step, global 4096 vs 2048: GPU busy {summ[a]['gpu_busy_s']:.2f} -> "
           f"{summ[b]['gpu_busy_s']:.2f} s ({d['gpu_busy_s']:+.2f} s), wall {d['wall_s']:+.2f} s "
           f"(unprofiled median: +1.42 s). CPU AdamW step (fsdp_cpu_offload) "
           f"{summ[a]['optimizer_step_cpu_s']:.2f} -> {summ[b]['optimizer_step_cpu_s']:.2f} s. "
           "The extra time is GPU work: matrix multiplies in the forecast engine and FSDP "
           "communication.")
    from plot_latent_long import finish
    finish(fig, "latent_profile", "Where the extra step time of the wider latent goes", sub,
           f"Runs  2048: {a} (job {RUNS[a]['job']})  global 4096: {b} (job {RUNS[b]['job']})\n"
           "L5, 1 node x 4 GPUs, dev 4f11a5be + config/optimizations.yml (fsdp_cpu_offload, "
           "ema_on_cpu). PyTorch profiler on rank 0, steps 10-13 (exp_latent_torchprof.yml); "
           "means over the 4 steps. Kernel times measured on the GPU; overlapping kernels on "
           "different streams are counted in each bar. other* = kernels outside module "
           "forwards and autograd backward ops (loss, optimizer, FSDP communication). "
           "Analysis: scripts/latent_prof.py.", wspace=0.5, left=0.13)
    print(json.dumps({"summary": summ, "class": cls}, indent=1))


if __name__ == "__main__":
    main()
