#!/usr/bin/env python3
"""Build training batches in this process (no DataLoader workers, no GPU), to profile the
data loading, e.g. with scalene. Prints the seconds per batch and, with --checksum, a hash
of every batch, so that a data-loading change can be checked to give the same batches.

  python scripts/profile_data_loading.py --base-config BASE [--config C ...] [--batches N]
"""

import argparse
import hashlib
import resource
import time
from pathlib import Path

import numpy as np
import torch

from weathergen.common import config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.datasets.packed_tokens import PackedTokens
from weathergen.train.trainer import Trainer
from weathergen.train.utils import TRAIN, cfg_keys_to_filter, filter_config_by_enabled


def checksum(obj, digest, seen) -> int:
    """Hash every tensor and array reachable from obj, in a fixed order; returns their bytes."""
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, PackedTokens):
        # hash the padded tokens, the same as without source_tokens_unpadded
        checksum(obj.unpad(), digest, set())
        return obj.nbytes
    if isinstance(obj, torch.Tensor):
        digest.update(obj.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return obj.numel() * obj.element_size()
    if hasattr(obj, "__array_interface__"):
        digest.update(np.ascontiguousarray(obj).tobytes())
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(checksum(obj[key], digest, seen) for key in sorted(obj, key=str))
    if isinstance(obj, list | tuple):
        return sum(checksum(item, digest, seen) for item in obj)
    if hasattr(obj, "__dict__"):
        return checksum(vars(obj), digest, seen)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--config", type=Path, nargs="*", default=[])
    parser.add_argument("--options", nargs="*", default=[], help="config overrides, key=value")
    parser.add_argument("--batches", type=int, default=3, help="batches after one warm-up")
    parser.add_argument("--checksum", action="store_true", help="print a hash of every batch")
    args = parser.parse_args()

    cf = config.load_merge_configs(
        None, None, None, args.base_config, *args.config, config.from_cli_arglist(args.options)
    )
    cf = Trainer.init_ddp(cf)
    cf.streams = config.load_streams(Path(cf.streams_directory))
    mode_cfg = filter_config_by_enabled(cf.training_config, cfg_keys_to_filter)
    batches = iter(MultiStreamDataSampler(cf, mode_cfg, stage=TRAIN))

    # the first batch opens the datasets and fills caches; keep it out of the timing
    next(batches)
    seconds = []
    for i in range(args.batches):
        t = time.perf_counter()
        batch = next(batches)
        seconds.append(time.perf_counter() - t)
        if args.checksum:
            digest = hashlib.sha256()
            size = checksum(batch, digest, set())
            print(f"batch {i}: {digest.hexdigest()[:16]}, {size / 2**20:.0f} MiB")  # noqa: T201
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # KiB on Linux
    print(
        f"{sum(seconds) / len(seconds):.2f} s per batch ({len(seconds)} batches), peak {peak:.1f} GiB"
    )  # noqa: T201


if __name__ == "__main__":
    main()
