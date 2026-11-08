"""GPU-time breakdown of one profiled step from prof7_extract.py output.

Prints: kernel time by class, by top-level module (forward, from the GPU-side nn.Module
annotations) and by backward op (via correlation -> launch -> enclosing autograd cpu_op),
FlashAttention calls by input shape, GPU idle gaps, and a merged timeline of phases."""

import bisect
import collections
import pickle
import sys

d = pickle.load(open(sys.argv[1], "rb"))
K = sorted(d["kernel"], key=lambda x: x[1])
step = [g for g in d["gpu_user_annotation"] if g[0].startswith("ProfilerStep")][0]
T0, T1 = step[1], step[1] + step[2]
K = [k for k in K if k[1] >= T0 - 1 and k[1] <= T1]


def cls(name):
    n = name.lower()
    if "flash" in n:
        return "flash_bwd" if ("bwd" in n or "backward" in n) else "flash_fwd"
    if "nccl" in n:
        return "nccl"
    if "memcpy" in n or "memset" in n:
        return "memcpy/memset"
    if "gemm" in n or "cutlass" in n or "xmma" in n or "nvjet" in n or "sm90" in n:
        return "gemm"
    return "elementwise/other"


# GPU-side module annotations: innermost and top-level containing each kernel
ann = sorted([g for g in d["gpu_user_annotation"] if g[0].startswith("nn.Module: ")],
             key=lambda g: g[1])
ann_starts = [a[1] for a in ann]


def modules_at(ts, te):
    """All nn.Module annotations containing [ts, te] (scan back a bounded window)."""
    i = bisect.bisect_right(ann_starts, ts)
    res = []
    for j in range(i - 1, max(-1, i - 4000), -1):
        a = ann[j]
        if a[1] <= ts and a[1] + a[2] >= te:
            res.append(a[0][len("nn.Module: "):])
    return res


def top_module(mods):
    if not mods:
        return None
    # shortest name = outermost; keep two levels: model.<x>
    m = min(mods, key=len)
    parts = m.split(".")
    if parts[1] in ("forecast_engine",):
        return "model.forecast_engine"
    if parts[1] == "encoder" and len(parts) > 2:
        return ".".join(parts[:3])
    return ".".join(parts[:3]) if len(parts) > 2 else m


# backward attribution: kernel correlation -> runtime launch (tid, ts) -> innermost cpu_op
rt = {r[4]: r for r in d["cuda_runtime"] if r[4] is not None}
ops_by_tid = collections.defaultdict(list)
for o in d["cpu_op"]:
    ops_by_tid[o[3]].append(o)
for t in ops_by_tid:
    ops_by_tid[t].sort(key=lambda o: o[1])
starts_by_tid = {t: [o[1] for o in v] for t, v in ops_by_tid.items()}


def cpu_ops_at(corr):
    r = rt.get(corr)
    if r is None:
        return []
    tid, ts = r[3], r[1]
    v = ops_by_tid.get(tid, [])
    i = bisect.bisect_right(starts_by_tid.get(tid, []), ts)
    res = []
    for j in range(i - 1, max(-1, i - 3000), -1):
        o = v[j]
        if o[1] <= ts <= o[1] + o[2]:
            res.append(o)
    return res  # innermost first


by_cls = collections.Counter()
by_phase = collections.Counter()
by_phase_cls = collections.Counter()
flash_shapes = collections.defaultdict(lambda: [0, 0.0, collections.Counter()])
labels = []
for k in K:
    name, ts, dur, dev, stream, corr, cat = k
    c = cls(name)
    by_cls[c] += dur
    mods = modules_at(ts, ts + dur)
    ops = cpu_ops_at(corr)
    bwd = [o for o in ops if o[0].startswith("autograd::engine::evaluate_function")]
    if mods:
        tm = top_module(mods)
        phase = ("recompute " if bwd else "fwd ") + tm
    elif bwd:
        phase = "bwd " + bwd[-1][0].split(": ")[-1]  # outermost autograd function
    else:
        phase = "other (optimizer/comm/loss)"
    by_phase[phase] += dur
    by_phase_cls[(phase, c)] += dur
    labels.append((ts, dur, phase, c))
    if c.startswith("flash"):
        fl = [o for o in ops if o[0].startswith("flash_attn_3::")]
        shape = str(fl[0][4][:3]) if fl else "?"
        where = (min(mods, key=len) if mods else "bwd")
        key = (c, shape)
        flash_shapes[key][0] += 1
        flash_shapes[key][1] += dur
        flash_shapes[key][2][top_module(mods) if mods else "bwd"] += 1

step_s = (T1 - T0) / 1e6
tot = sum(by_cls.values())
print(f"step (GPU ProfilerStep#2) {step_s:.1f} s; kernel time {tot/1e6:.1f} s")
# busy union
busy, cur_s, cur_e = 0.0, None, None
gaps = []
for ts, dur, ph, c in sorted(labels):
    te = ts + dur
    if cur_e is None or ts > cur_e:
        if cur_e is not None:
            gaps.append((cur_e, ts))
            busy += cur_e - cur_s
        cur_s, cur_e = ts, te
    else:
        cur_e = max(cur_e, te)
busy += cur_e - cur_s
print(f"GPU busy (union) {busy/1e6:.1f} s = {100*busy/(T1-T0):.1f}% of step; "
      f"idle {(T1-T0-busy)/1e6:.1f} s")
big = sorted([g for g in gaps if g[1] - g[0] > 2e5], key=lambda g: -(g[1] - g[0]))[:10]
print("largest idle gaps (s at, len):",
      [(round((a - T0) / 1e6, 1), round((b - a) / 1e6, 2)) for a, b in big])
print("\n== by kernel class ==")
for c, v in by_cls.most_common():
    print(f"  {c:22s} {v/1e6:7.1f} s  {100*v/tot:5.1f}%")
print("\n== by phase ==")
for p, v in by_phase.most_common(25):
    fl = sum(by_phase_cls[(p, c)] for c in ("flash_fwd", "flash_bwd"))
    print(f"  {p:70s} {v/1e6:7.1f} s  {100*v/tot:5.1f}%  (flash {fl/1e6:.1f} s)")
print("\n== FlashAttention by (class, q/k/v dims) ==")
for (c, shape), (n, t, where) in sorted(flash_shapes.items(), key=lambda x: -x[1][1]):
    print(f"  {c:10s} {shape:60s} n={n:4d} {t/1e6:7.2f} s  avg {t/n/1e3:8.1f} ms  {dict(where)}")

# merged timeline: consecutive kernels with the same phase, segments >= 1 s
print("\n== timeline (segments >= 1.0 s; t from step start) ==")
seg = None
segs = []
for ts, dur, ph, c in sorted(labels):
    if seg and seg[2] == ph and ts - seg[1] < 1e6:
        seg[1] = max(seg[1], ts + dur)
        seg[3] += dur
        if c.startswith("flash"):
            seg[4] += dur
    else:
        if seg:
            segs.append(seg)
        seg = [ts, ts + dur, ph, dur, dur if c.startswith("flash") else 0.0]
segs.append(seg)
for s in segs:
    if s[1] - s[0] >= 1e6:
        print(f"  {(s[0]-T0)/1e6:6.1f}-{(s[1]-T0)/1e6:6.1f} s  {(s[1]-s[0])/1e6:5.1f} s  "
              f"{s[2]:60s} flash {100*s[4]/max(s[3],1):3.0f}%")
pickle.dump({"T0": T0, "T1": T1, "labels": labels, "segs": segs},
            open(sys.argv[1].replace(".pkl", "_labels.pkl"), "wb"))
