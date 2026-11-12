#!/usr/bin/env python3
"""Where the step time goes: compare PyTorch profiler traces of the latent-width runs.

Adapted from experiments/martin/q4ag4rlz/tools/prof7_{extract,analyze}.py on martin/exp/bottleneck
(one step at L7) for several profiled steps, the CPU optimizer step and host-device copies.
Runs on the JUPITER login node with the system python3 (stdlib only), so the ~200 MB traces
are not copied:

  python3 latent_prof.py extract <trace.json.gz> <out.pkl>      # stream, keep compact tuples
  python3 latent_prof.py analyze <out.pkl> <label> > <label>.json

For each profiled step (CPU-side ProfilerStep#N window): wall time, the CPU optimizer step
(Optimizer.step#AdamW.step; the AdamW update runs on the CPU under fsdp_cpu_offload), GPU busy
time (union of kernel, memcpy and memset intervals), kernel time by class (FlashAttention,
GEMM, NCCL, copies, other) and by phase: forward or recompute per top-level module (GPU-side
nn.Module annotations of the profiling wrapper), backward (kernels launched under an autograd
backward op), or other. Times are in seconds. CPU times include the profiler's own overhead.
"""

import bisect
import collections
import gzip
import json
import pickle
import sys

KEEP_CPU = ("autograd::engine::evaluate_function", "aten::_fused_adamw_")


def extract(src, dst):
    out = {"kernel": [], "gpu_user_annotation": [], "cpu_op": [], "cuda_runtime": [],
           "user_annotation": []}
    buf, inside, skip = [], False, False
    with gzip.open(src, "rt") as f:
        for line in f:
            if not inside:
                if line.startswith("  {") and line.strip() == "{":
                    inside, buf, skip = True, [line], False
                continue
            if len(buf) == 1 and ('"cat": "python_function"' in line
                                  or '"cat": "cpu_instant_event"' in line
                                  or '"cat": "ac2g"' in line or '"cat": "fwdbwd"' in line):
                skip = True
            if not skip:
                buf.append(line)
            if line.startswith("  }"):
                inside = False
                if skip:
                    continue
                try:
                    e = json.loads("".join(buf).rstrip().rstrip(","))
                except json.JSONDecodeError:
                    continue
                cat, a = e.get("cat"), e.get("args", {})
                if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
                    out["kernel"].append((e["name"], e["ts"], e["dur"], a.get("device"),
                                          a.get("stream"), a.get("correlation"), cat))
                elif cat == "gpu_user_annotation":
                    out["gpu_user_annotation"].append((e["name"], e["ts"], e["dur"],
                                                       a.get("device"), a.get("stream")))
                elif cat == "user_annotation":
                    out["user_annotation"].append((e["name"], e["ts"], e["dur"], e.get("tid")))
                elif cat == "cuda_runtime":
                    out["cuda_runtime"].append((e["name"], e["ts"], e["dur"], e.get("tid"),
                                                a.get("correlation")))
                elif cat == "cpu_op" and e["name"].startswith(KEEP_CPU):
                    out["cpu_op"].append((e["name"], e["ts"], e["dur"], e.get("tid")))
    with open(dst, "wb") as f:
        pickle.dump(out, f)
    print({k: len(v) for k, v in out.items()}, file=sys.stderr)


def kernel_class(name, cat):
    n = name.lower()
    if cat == "gpu_memcpy":
        if "htod" in n:
            return "copy host->device"
        if "dtoh" in n:
            return "copy device->host"
        return "copy device->device"
    if cat == "gpu_memset":
        return "memset"
    if "flash" in n:
        return "flash-attention backward" if ("bwd" in n or "backward" in n) else \
            "flash-attention forward"
    if "nccl" in n:
        return "nccl"
    if "gemm" in n or "cutlass" in n or "xmma" in n or "nvjet" in n or "sm90" in n:
        return "gemm"
    return "elementwise/other"


def top_module(mods):
    m = min(mods, key=len)  # outermost annotation
    parts = m.split(".")
    if len(parts) > 2 and parts[1] == "encoder":
        return ".".join(parts[:3])
    return ".".join(parts[:2])


def union(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0.0)


def analyze(path, label):
    d = pickle.load(open(path, "rb"))
    kernels = sorted(d["kernel"], key=lambda k: k[1])
    k_starts = [k[1] for k in kernels]
    ann = sorted((g for g in d["gpu_user_annotation"] if g[0].startswith("nn.Module: ")),
                 key=lambda g: g[1])
    ann_starts = [a[1] for a in ann]
    rt = {r[4]: r for r in d["cuda_runtime"] if r[4] is not None}
    bwd_by_tid = collections.defaultdict(list)
    for o in d["cpu_op"]:
        if o[0].startswith("autograd::engine::evaluate_function"):
            bwd_by_tid[o[3]].append(o)
    for v in bwd_by_tid.values():
        v.sort(key=lambda o: o[1])
    bwd_starts = {t: [o[1] for o in v] for t, v in bwd_by_tid.items()}

    def modules_at(ts, te):
        i = bisect.bisect_right(ann_starts, ts)
        return [ann[j][0][len("nn.Module: "):] for j in range(i - 1, max(-1, i - 4000), -1)
                if ann[j][1] <= ts and ann[j][1] + ann[j][2] >= te]

    def in_backward(corr):
        r = rt.get(corr)
        if r is None:
            return False
        v, ts = bwd_by_tid.get(r[3], []), r[1]
        i = bisect.bisect_right(bwd_starts.get(r[3], []), ts)
        return any(v[j][1] <= ts <= v[j][1] + v[j][2] for j in range(i - 1, max(-1, i - 3000), -1))

    steps = sorted((u for u in d["user_annotation"] if u[0].startswith("ProfilerStep#")),
                   key=lambda u: u[1])
    opt = [u for u in d["user_annotation"] if u[0].startswith("Optimizer.step")]
    zero = [u for u in d["user_annotation"] if u[0].startswith("Optimizer.zero_grad")]
    out = {"label": label, "steps": []}
    for name, t0, dur, _ in steps:
        t1 = t0 + dur
        ks = kernels[bisect.bisect_left(k_starts, t0):bisect.bisect_left(k_starts, t1)]
        by_cls, by_phase = collections.Counter(), collections.Counter()
        for kname, ts, kdur, _dev, _st, corr, cat in ks:
            by_cls[kernel_class(kname, cat)] += kdur
            if cat != "kernel":
                by_phase["copies/memset"] += kdur
                continue
            mods = modules_at(ts, ts + kdur)
            bwd = in_backward(corr)
            if mods:
                by_phase[("recompute " if bwd else "forward ") + top_module(mods)] += kdur
            elif bwd:
                by_phase["backward"] += kdur
            else:
                by_phase["other (loss, optimizer, comm)"] += kdur
        o = [u for u in opt if t0 <= u[1] < t1]
        z = [u for u in zero if t0 <= u[1] < t1]
        o_win = (o[0][1], o[0][1] + o[0][2]) if o else None
        gpu_in_opt = union([(k[1], min(k[1] + k[2], o_win[1])) for k in ks
                            if o_win and o_win[0] <= k[1] < o_win[1]]) if o_win else 0.0
        out["steps"].append({
            "step": name, "wall_s": dur / 1e6,
            "gpu_busy_s": union([(k[1], k[1] + k[2]) for k in ks]) / 1e6,
            "optimizer_step_cpu_s": sum(u[2] for u in o) / 1e6,
            "gpu_busy_during_optimizer_step_s": gpu_in_opt / 1e6,
            "zero_grad_cpu_s": sum(u[2] for u in z) / 1e6,
            "kernel_s_by_class": {k: v / 1e6 for k, v in by_cls.most_common()},
            "kernel_s_by_phase": {k: v / 1e6 for k, v in by_phase.most_common()},
            "n_kernels": len(ks),
        })
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3])
    else:
        analyze(sys.argv[2], sys.argv[3])
