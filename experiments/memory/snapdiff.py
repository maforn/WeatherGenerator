# ruff: noqa: T201
# (command-line report: printing is the output)
"""Compare two CUDA allocator snapshots: live memory at each run's peak, grouped by call site.

usage: python -I snapdiff.py a.pickle b.pickle [--top N] [--depth K]
Site = innermost frame inside src/weathergen (file:line func), else innermost frame.
Memory allocated before recording started is reported as 'pre-trace'.
"""

import collections
import pickle
import sys

GIB = 2**30


def site_of(frames, depth=1):
    wg = [f for f in frames if "/src/weathergen/" in f["filename"]]
    use = wg[:depth] if wg else frames[:1]
    if not use:
        return "<no frames>"

    def short(fn):
        return fn.split("/src/weathergen/")[-1].split("/site-packages/")[-1]

    return " < ".join(f"{short(f['filename'])}:{f['line']} {f['name']}" for f in use)


def peak_breakdown(path, depth):
    with open(path, "rb") as fh:
        s = pickle.load(fh)
    trace = s["device_traces"][0]
    end_active = sum(
        b["size"]
        for seg in s["segments"]
        for b in seg["blocks"]
        if b["state"] == "active_allocated"
    )
    live = {}  # addr -> (size, site)
    freed_pre = 0
    for e in trace:
        a = e["action"]
        if a == "alloc":
            live[e["addr"]] = (e["size"], site_of(e["frames"], depth))
        elif a == "free_completed":
            if e["addr"] in live:
                del live[e["addr"]]
            else:
                freed_pre += e["size"]
    pre0 = end_active - sum(v[0] for v in live.values()) + freed_pre

    # second pass: find the peak and the live set at that moment
    live, cur_pre, tracked, best = {}, pre0, 0, (-1, None, None)
    for i, e in enumerate(trace):
        a = e["action"]
        if a == "alloc":
            live[e["addr"]] = (e["size"], site_of(e["frames"], depth))
            tracked += e["size"]
            if cur_pre + tracked > best[0]:
                best = (cur_pre + tracked, i, None)
                peak_live, peak_pre = dict(live), cur_pre
        elif a == "free_completed":
            if e["addr"] in live:
                tracked -= live.pop(e["addr"])[0]
            else:
                cur_pre -= e["size"]
    by = collections.Counter()
    for size, site in peak_live.values():
        by[site] += size
    by["pre-trace (params, optimizer, ...)"] += peak_pre
    peak_frames = trace[best[1]]["frames"]
    return best[0], by, site_of(peak_frames, 3), len(trace)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
    depth = int(sys.argv[sys.argv.index("--depth") + 1]) if "--depth" in sys.argv else 1
    if "--top" in sys.argv:
        args.remove(str(top))
    if "--depth" in sys.argv:
        args.remove(str(depth))
    (pa, ba, sa, na), (pb, bb, sb, nb) = (peak_breakdown(p, depth) for p in args)
    print(f"A {args[0]}: peak {pa / GIB:.2f} GiB, {na} events, at {sa}")
    print(f"B {args[1]}: peak {pb / GIB:.2f} GiB, {nb} events, at {sb}")
    print(f"delta {(pb - pa) / GIB:+.2f} GiB\n")
    sites = sorted(set(ba) | set(bb), key=lambda k: (-(bb[k] - ba[k]), k))
    print(f"{'A GiB':>7} {'B GiB':>7} {'delta':>7} {'ratio':>6}  site")
    for k in sites[:top]:
        r = f"{bb[k] / ba[k]:.1f}" if ba[k] else "new"
        print(f"{ba[k] / GIB:7.2f} {bb[k] / GIB:7.2f} {(bb[k] - ba[k]) / GIB:+7.2f} {r:>6}  {k}")
    rest_a = sum(ba[k] for k in sites[top:]) / GIB
    rest_b = sum(bb[k] for k in sites[top:]) / GIB
    print(
        f"{rest_a:7.2f} {rest_b:7.2f} {rest_b - rest_a:+7.2f}         "
        f"(other {len(sites) - top} sites)"
    )


if __name__ == "__main__":
    main()
