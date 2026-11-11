"""Parameter count of a saved WeatherGenerator checkpoint, total and by top-level module."""
import json
import sys
from collections import defaultdict

import torch

sd = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=True)
if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
    sd = sd["model"]
by_module = defaultdict(int)
dtypes = defaultdict(int)
for k, v in sd.items():
    if not torch.is_tensor(v):
        continue
    parts = k.split(".")
    by_module[parts[0] if parts[0] != "module" else parts[1]] += v.numel()
    dtypes[str(v.dtype)] += v.numel()
print(json.dumps({"total": sum(by_module.values()), "by_module": dict(sorted(by_module.items())),
                  "dtypes": dict(dtypes), "n_tensors": len(sd)}, indent=1))
