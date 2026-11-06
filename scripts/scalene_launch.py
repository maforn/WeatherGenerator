#!/usr/bin/env python3
"""Run scalene with its torch.profiler integration (GPU time per line, --gpu) started late.

Scalene starts torch.profiler at launch; in the training it crashes the process while the model
is built (SIGSEGV after a profiler warning on parameter creation). With
SCALENE_TORCH_PROFILER_DELAY=<seconds> the profiler starts on scalene's first CPU sample after
that delay instead, i.e. once the training loop runs, and only in this process (not in forked
DataLoader workers). Scalene itself is not changed.

  python scripts/scalene_launch.py run --gpu -o profile.json program.py --- args
"""

import os
import sys
import time


def defer_torch_profiler(delay: float) -> None:
    """Start scalene's torch profiler after delay seconds. In a function: scalene runs the
    profiled program in this module's namespace, which would overwrite module-level names."""
    # what torch.profiler imports lazily on first use: imported inside scalene's signal handler,
    # in the middle of the program's own imports, it fails as a circular import
    import scalene.scalene_torch as scalene_torch
    import torch._inductor.utils  # noqa: F401
    import torch.profiler  # noqa: F401

    profiler = scalene_torch.TorchProfiler
    start, step, t0, pid = profiler.start, profiler.step, time.time(), os.getpid()

    def deferred_start(self) -> None:
        self.deferred = True
        sys.stderr.write(f"[scalene_launch] torch profiler starts in {delay:.0f} s\n")

    def deferred_step(self) -> None:
        if not getattr(self, "deferred", False):
            step(self)
        elif os.getpid() == pid and time.time() - t0 >= delay:
            self.deferred = False
            start(self)
            sys.stderr.write(f"[scalene_launch] torch profiler on: {self._enabled}\n")

    profiler.start, profiler.step = deferred_start, deferred_step


def main() -> None:
    from scalene.__main__ import main as scalene_main
    from scalene.scalene_arguments import ScaleneArguments

    # scalene re-runs itself in a new process (without the patch above) unless its environment
    # is set; without --memory it only needs this (pass no --allocation-sampling-window)
    window = str(ScaleneArguments().allocation_sampling_window)
    os.environ.setdefault("SCALENE_ALLOCATION_SAMPLING_WINDOW", window)
    delay = float(os.environ.get("SCALENE_TORCH_PROFILER_DELAY", "0"))
    if delay > 0:
        defer_torch_profiler(delay)
    sys.argv[0] = "scalene"
    scalene_main()


if __name__ == "__main__":
    main()
