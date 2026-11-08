"""Stream a PyTorch profiler Chrome trace (.json.gz) and keep only the events needed for a
GPU-time breakdown. Writes a pickle of compact tuples. python_function events are skipped."""

import gzip
import json
import pickle
import sys

src, dst = sys.argv[1], sys.argv[2]
KEEP_CPU = ("flash_attn_3::", "autograd::engine::evaluate_function", "ProfilerStep",
            "CheckpointFunctionBackward", "torch.utils.checkpoint")
out = {"kernel": [], "gpu_user_annotation": [], "cpu_op": [], "cuda_runtime": [],
       "user_annotation": []}
buf, depth, inside, skip = [], 0, False, False
with gzip.open(src, "rt") as f:
    for line in f:
        if not inside:
            if line.startswith("  {") and line.strip() == "{":
                inside, buf, skip = True, [line], False
            continue
        if len(buf) == 1 and ('"cat": "python_function"' in line
                              or '"cat": "cpu_instant_event"' in line):
            skip = True
        if not skip:
            buf.append(line)
        if line.startswith("  }"):
            inside = False
            if skip:
                continue
            text = "".join(buf).rstrip().rstrip(",")
            try:
                e = json.loads(text)
            except json.JSONDecodeError:
                continue
            cat = e.get("cat")
            a = e.get("args", {})
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
                out["cpu_op"].append((e["name"], e["ts"], e["dur"], e.get("tid"),
                                      a.get("Input Dims")))
with open(dst, "wb") as f:
    pickle.dump(out, f)
print({k: len(v) for k, v in out.items()})
