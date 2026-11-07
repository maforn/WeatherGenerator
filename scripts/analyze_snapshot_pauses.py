"""Per training step: time, and how much of it the main thread spends in pauses (no allocator
events for >= 0.5 s) before the decoders (forward) and during backward."""
import pickle, sys
snap = pickle.load(open(sys.argv[1], "rb"))
tr = [e for e in max(snap["device_traces"], key=len) if "time_us" in e]
t = lambda e: e["time_us"] / 1e6
def has(e, s): return any(s in f"{f['filename']}:{f['name']}" for f in e.get("frames", []))
ends = [i for i, e in enumerate(tr) if has(e, "ema.py") and (i == 0 or not has(tr[i-1], "ema.py"))]
for a, b in zip(ends, ends[1:]):
    fwd = bwd = other = 0.0; phase = "fwd"
    for i in range(a + 1, b):
        if has(tr[i], "_tensor.py:backward"): phase = "bwd"
        g = t(tr[i]) - t(tr[i-1])
        if g < 0.5: continue
        if has(tr[i-1], "predict_decoders"): fwd += g
        elif phase == "bwd" or not tr[i-1].get("frames"): bwd += g
        else: other += g
    step = t(tr[b]) - t(tr[a])
    print(f"step {step:6.1f} s | pauses: forward before decoders {fwd:5.1f} s, backward {bwd:5.1f} s,"
          f" other {other:4.1f} s | total {100*(fwd+bwd+other)/step:3.0f}% of the step")
