#!/usr/bin/env python3
"""Profile the data loading of a training config on the CPU, without training.
k

Builds the training sampler like the trainer does, builds batches in this process
(no DataLoader workers) under cProfile, and prints the seconds per batch for each
stage plus the functions that take the most time. Use it to measure data-loading
changes without a full training run, e.g. through `make profile-data`.

  python scripts/profile_data_loading.py --base-config BASE [--config C ...] [--batches N]
"""

import argparse
import cProfile
import pstats
import time
from pathlib import Path

import torch.multiprocessing as mp

from weathergen.common import config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.train.trainer import Trainer
from weathergen.train.utils import TRAIN, cfg_keys_to_filter, filter_config_by_enabled


def _send(batch, queue, done) -> None:
    queue.put((time.time(), batch))
    # keep this process, and with it the shared-memory handles, alive until received
    done.wait()


def handoff_seconds(batch) -> float:
    """Seconds to hand one batch from a forked worker to this process, the way the
    DataLoader does it (torch.multiprocessing queue, tensors through shared memory)."""
    ctx = mp.get_context("fork")
    queue, done = ctx.Queue(), ctx.Event()
    worker = ctx.Process(target=_send, args=(batch, queue, done))
    worker.start()
    t_sent, _ = queue.get()
    seconds = time.time() - t_sent
    done.set()
    worker.join()
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--config", type=Path, nargs="*", default=[])
    parser.add_argument("--options", nargs="*", default=[], help="config overrides, key=value")
    parser.add_argument(
        "--batches", type=int, default=3, help="batches to profile, after one warm-up"
    )
    parser.add_argument("--top", type=int, default=30, help="functions to list")
    args = parser.parse_args()

    cf = config.load_merge_configs(
        None, None, None, args.base_config, *args.config, config.from_cli_arglist(args.options)
    )
    cf = Trainer.init_ddp(cf)
    cf.streams = config.load_streams(Path(cf.streams_directory))
    mode_cfg = filter_config_by_enabled(cf.training_config, cfg_keys_to_filter)
    batches = iter(MultiStreamDataSampler(cf, mode_cfg, stage=TRAIN))

    # the first batch opens the datasets and fills caches; keep it out of the numbers
    next(batches)

    profile = cProfile.Profile()
    stages: dict[str, float] = {}
    profile.enable()
    for _ in range(args.batches):
        batch = next(batches)
        for stage, seconds in batch.timings.items():
            stages[stage] = stages.get(stage, 0.0) + seconds
    profile.disable()

    print(f"\nseconds per batch, mean of {args.batches} (with cProfile overhead):")  # noqa: T201
    for stage, seconds in sorted(stages.items(), key=lambda x: -x[1]):
        print(f"  {stage:<10} {seconds / args.batches:8.2f}")  # noqa: T201
    handoff = [handoff_seconds(batch) for _ in range(3)]
    print(f"  handoff    {sum(handoff) / 3:8.2f}  (worker -> main, mean of 3, no cProfile)")  # noqa: T201
    print("\nby own time (tottime):")  # noqa: T201
    pstats.Stats(profile).strip_dirs().sort_stats("tottime").print_stats(args.top)
    print("by cumulative time, weathergen functions only:")  # noqa: T201
    pstats.Stats(profile).sort_stats("cumulative").print_stats("weathergen", args.top)


if __name__ == "__main__":
    main()
