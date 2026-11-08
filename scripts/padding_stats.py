#!/usr/bin/env python3
"""How much of the source tokens is padding, per stream and HEALPix level.

Reads a few real samples (the sampler's source windows of every stream, as training does) and,
for each level, counts the points per HEALPix cell the way hpy_splits does (ang2pix, nested; per
time step for tokenize_spacetime streams). Every occupied cell is padded up to a multiple of the
stream's token_size, so

    padded rows = sum over occupied cells of ceil(points / token_size) * token_size

Reports real points, tokens, occupied cells, padded rows, the padding share and the size of the
source tokens (row width 1 + 5 + 2 + geoinfos + channels) in fp32 and bf16. Counts are for all
cells, before the source masking keeps a subset of them. Level --check is also run through the
real tokenizer (tokenize_space / tokenize_spacetime) to confirm the counts are exact.
Reads only; no tokens are built, so it runs in a few GiB on a login node.

  python scripts/padding_stats.py --base-config BASE [--config C ...] --out DIR \
      --levels 5 6 7 8 --samples 4
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy_healpix.healpy import ang2pix

from weathergen.common import config
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.datasets.tokenizer_masking import readerdata_to_torch
from weathergen.datasets.tokenizer_utils import (
    theta_phi_to_standard_coords,
    tokenize_space,
    tokenize_spacetime,
)
from weathergen.train.trainer import Trainer
from weathergen.train.utils import TRAIN, cfg_keys_to_filter, filter_config_by_enabled

GIB = 2**30
TIME_WIDTH = 5  # encode_times_source
LOCAL_COORDS_WIDTH = 2


def cell_counts(thetas, phis, hl):
    cells = ang2pix(2**hl, thetas, phis, nest=True)
    return np.bincount(np.asarray(cells), minlength=12 * 4**hl)


def padding_counts(rdata, token_size, hl, spacetime, what_if=()):
    """Tokens and padded rows of one window, as hpy_splits pads them (pad_tokens=True), and the
    padded rows the same window would have with each token size in what_if."""
    thetas, phis = theta_phi_to_standard_coords(rdata.coords)
    thetas, phis = np.asarray(thetas), np.asarray(phis)
    groups = [slice(None)]
    if spacetime:
        times = np.asarray(rdata.datetimes)
        groups = [times == t for t in np.unique(times)]
    tokens = padded = occupied = 0
    padded_if = dict.fromkeys(what_if, 0)
    for group in groups:
        lens = cell_counts(thetas[group], phis[group], hl)
        num_tokens = -(-lens // token_size)
        tokens += int(num_tokens.sum())
        padded += int((num_tokens * token_size).sum())
        occupied += int((lens > 0).sum())
        for size in what_if:
            padded_if[size] += int((-(-lens // size) * size).sum())
    return {
        "tokens": tokens,
        "padded_rows": padded,
        "occupied_cells": occupied,
        **{f"padded_rows_ts{size}": rows for size, rows in padded_if.items()},
    }


def tokenizer_rows(rdata, token_size, hl, spacetime):
    """Padded rows from the real tokenizer, to check padding_counts."""
    tok = tokenize_spacetime if spacetime else tokenize_space
    _, idxs_cells_lens = tok(rdata, token_size, hl, True)
    return sum(sum(lens) for lens in idxs_cells_lens)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--config", type=Path, nargs="*", default=[])
    parser.add_argument("--options", nargs="*", default=[], help="config overrides, key=value")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--stride", type=int, default=6, help="time steps between windows")
    parser.add_argument("--check", type=int, default=5, help="level checked against tokenizer")
    parser.add_argument(
        "--what-if",
        type=int,
        nargs="*",
        default=[16, 32, 64, 128, 256],
        help="other token sizes for the streams with token_size > 64 (padded rows only)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cf = config.load_merge_configs(
        None, None, None, args.base_config, *args.config, config.from_cli_arglist(args.options)
    )
    cf = Trainer.init_ddp(cf)
    cf.streams = config.load_streams(Path(cf.streams_directory))
    mode_cfg = filter_config_by_enabled(cf.training_config, cfg_keys_to_filter)
    sampler = MultiStreamDataSampler(cf, mode_cfg, stage=TRAIN)
    perms, forecast_steps = sampler.reset()
    steps_in = max(sc.get("num_steps_input", 1) for sc in mode_cfg.get("model_input").values())
    # the same distinct windows, in the same order, as scripts/read_stress.py
    indices = np.unique(np.asarray(perms))
    indices = indices[indices % args.stride == 0]
    indices = np.random.default_rng(0).permutation(indices)[: args.samples]

    records = []
    for idx in indices.tolist():
        start = time.perf_counter()
        for name, stream in sampler.streams_datasets.items():
            info = stream.info
            token_size = info["token_size"]
            spacetime = info.get("tokenize_spacetime", False)
            inputs, _ = sampler._get_data_windows(
                idx, int(forecast_steps[0]), steps_in, stream.readers, None, name
            )
            for step, rdata in enumerate(inputs):
                spoof = bool(getattr(rdata, "is_spoof", False)) or rdata.is_empty()
                rdata = readerdata_to_torch(rdata)
                points = 0 if spoof else len(rdata.coords)
                width = (
                    1
                    + TIME_WIDTH
                    + LOCAL_COORDS_WIDTH
                    + rdata.geoinfos.shape[1]
                    + rdata.data.shape[1]
                )
                for hl in args.levels:
                    rec = {
                        "idx": idx,
                        "stream": name,
                        "step": step,
                        "level": hl,
                        "token_size": token_size,
                        "spacetime": spacetime,
                        "spoof": spoof,
                        "points": points,
                        "row_width": width,
                        "cells": 12 * 4**hl,
                    }
                    if not spoof:
                        what_if = args.what_if if token_size > 64 else ()
                        rec |= padding_counts(rdata, token_size, hl, spacetime, what_if)
                        if hl == args.check and idx == indices[0]:
                            rows = tokenizer_rows(rdata, token_size, hl, spacetime)
                            assert rows == rec["padded_rows"], (name, hl, rows, rec)
                            rec["checked"] = True
                    records.append(rec)
        print(f"sample {idx}: {time.perf_counter() - start:.1f} s", flush=True)  # noqa: T201

    (args.out / "padding_records.json").write_text(json.dumps(records, indent=1))

    # mean over samples, per stream and level
    acc = defaultdict(lambda: defaultdict(float))
    for r in records:
        if r["spoof"]:
            continue
        a = acc[(r["stream"], r["level"])]
        a["token_size"] = r["token_size"]
        for k in ("points", "tokens", "padded_rows", "occupied_cells"):
            a[k] += r[k] / len(indices)
        a["fp32_gib"] += r["padded_rows"] * r["row_width"] * 4 / GIB / len(indices)
        a["real_fp32_gib"] += r["points"] * r["row_width"] * 4 / GIB / len(indices)
        for size in args.what_if:
            # streams outside the what-if keep their own token size
            rows = r.get(f"padded_rows_ts{size}", r["padded_rows"])
            a[f"bf16_gib_ts{size}"] += rows * r["row_width"] * 2 / GIB / len(indices)
    summary = []
    for (stream, hl), a in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        summary.append(
            {
                "stream": stream,
                "level": hl,
                **{k: round(v, 3) for k, v in a.items()},
                "padding_share": 1 - a["points"] / a["padded_rows"] if a["padded_rows"] else 0,
                "points_per_occupied_cell": a["points"] / a["occupied_cells"]
                if a["occupied_cells"]
                else 0,
                "bf16_gib": a["fp32_gib"] / 2,
            }
        )
    for hl in args.levels:
        rows = [s for s in summary if s["level"] == hl]
        summary.append(
            {
                "stream": "ALL",
                "level": hl,
                **{
                    k: sum(s[k] for s in rows)
                    for k in (
                        "points",
                        "tokens",
                        "padded_rows",
                        "fp32_gib",
                        "bf16_gib",
                        "real_fp32_gib",
                        *(f"bf16_gib_ts{size}" for size in args.what_if),
                    )
                },
            }
        )
        summary[-1]["padding_share"] = 1 - summary[-1]["points"] / summary[-1]["padded_rows"]
        print(  # noqa: T201
            f"level {hl}: source tokens bf16 {summary[-1]['bf16_gib']:.2f} GiB with the "
            f"configured token sizes, {summary[-1]['real_fp32_gib'] / 2:.2f} GiB without "
            "padding; with every token_size > 64 set to "
            + ", ".join(f"{t}: {summary[-1][f'bf16_gib_ts{t}']:.2f}" for t in args.what_if)
        )
    (args.out / "padding_summary.json").write_text(json.dumps(summary, indent=1))

    print(  # noqa: T201
        f"{'stream':28s} lvl  tok  points(M) occ.cells pts/cell  tokens  padded(M) pad%  "
        "fp32 GiB  bf16 GiB"
    )
    for s in summary:
        print(  # noqa: T201
            f"{s['stream'][:28]:28s} {s['level']:3d} {int(s.get('token_size', 0)):4d} "
            f"{s['points'] / 1e6:9.2f} {int(s.get('occupied_cells', 0)):9d} "
            f"{s.get('points_per_occupied_cell', 0):8.1f} {int(s['tokens']):7d} "
            f"{s['padded_rows'] / 1e6:9.1f} {100 * s['padding_share']:5.1f} "
            f"{s['fp32_gib']:9.2f} {s['bf16_gib']:9.2f}"
        )


if __name__ == "__main__":
    main()
