# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""`data_loading.fast_target_build`: targets without the parts each view discards.

Synthetic target windows are tokenized as in the sampler (without padding), then
TokenizerMasking.get_target_coords and get_target_values, and
MultiStreamDataSampler._build_stream_data_output with StreamData, run with the option off and
on. Every output must have the same type, dtype, shape and bytes; in training the inverse point
order, which StreamData drops, may be skipped.
"""

import types

import numpy as np
import pytest
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler
from weathergen.datasets.stream_data import StreamData
from weathergen.datasets.tokenizer_masking import TokenizerMasking
from weathergen.train.utils import TRAIN, VAL

HL = 3
NUM_CELLS = 12 * 4**HL
T0 = np.datetime64("2021-03-01T06:00", "ns")
HOUR = np.timedelta64(1, "h")
TIME_WIN = (T0, T0 + 6 * HOUR)


def _same(a, b):
    """Same type, dtype, shape and bytes, for tensors, numpy arrays, lists and None."""
    assert type(a) is type(b)
    if a is None:
        return
    if isinstance(a, list | tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _same(x, y)
        return
    if isinstance(a, torch.Tensor):
        assert a.requires_grad == b.requires_grad
        a, b = a.detach().contiguous().numpy(), b.detach().contiguous().numpy()
    if isinstance(a, bool | int | float | str):
        assert a == b
        return
    assert a.dtype == b.dtype and a.shape == b.shape
    # compared outside the assert: pytest would otherwise diff the bytes
    equal = np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    assert equal, "bytes differ"


def _era5(rng, num_grid=4000, steps=2, num_ch=6, num_geo=2, subsample=5000):
    """Gridded, the same grid at every step, subsampled as with max_num_targets."""
    z = rng.uniform(-1, 1, num_grid)
    grid = np.stack([np.rad2deg(np.arcsin(z)), rng.uniform(-180, 180, num_grid)], 1)
    coords = np.tile(grid, (steps, 1)).astype(np.float32)
    times = np.repeat(T0 + 3 * HOUR * np.arange(steps), num_grid)
    keep = np.sort(rng.choice(len(coords), subsample, replace=False))
    data = rng.standard_normal((len(coords), num_ch)).astype(np.float32)
    data[rng.uniform(size=data.shape) < 0.03] = np.nan
    geoinfos = rng.standard_normal((len(coords), num_geo)).astype(np.float32)
    return IOReaderData(coords[keep], geoinfos[keep], data[keep], times[keep])


def _obs(rng, num=2000, num_ch=12, num_geo=3):
    """Sparse, a time per point."""
    coords = np.stack([rng.uniform(-80, 80, num), rng.uniform(-180, 180, num)], 1)
    seconds = np.sort(rng.integers(0, 6 * 3600 * 10**9, num)).astype("timedelta64[ns]")
    data = rng.standard_normal((num, num_ch)).astype(np.float32)
    geoinfos = rng.standard_normal((num, num_geo)).astype(np.float32)
    return IOReaderData(coords.astype(np.float32), geoinfos, data, T0 + seconds)


def _without_values(rng):
    """Target points of inference without targets: data has no rows."""
    rdata = _era5(rng)
    rdata.data = rdata.data[:0]
    return rdata


WINDOWS = {
    "era5": (_era5, 64),
    "era5_small_tokens": (_era5, 7),
    "obs": (_obs, 16),
    "without_values": (_without_values, 64),
}


def _cell_masks(populated):
    rng = np.random.default_rng(3)
    return {
        "all": np.ones(NUM_CELLS, dtype=bool),
        "random": rng.uniform(size=NUM_CELLS) < 0.4,
        "none": np.zeros(NUM_CELLS, dtype=bool),
        "only_empty_cells": ~populated,
        "torch_float": torch.from_numpy((rng.uniform(size=NUM_CELLS) < 0.5).astype(np.float32)),
    }


def _tokenizer(fast):
    masker = types.SimpleNamespace(stage=TRAIN, reset_rng=lambda rng: None)
    return TokenizerMasking(HL, masker, fast_target_build=fast)


def _setup(window):
    make, token_size = WINDOWS[window]
    rdata = make(np.random.default_rng(0))
    stream_info = {"name": window, "token_size": token_size, "stream_id": 4}
    (token_data,) = _tokenizer(False).get_tokens_windows(stream_info, [rdata], False)
    populated = np.array([len(lens) > 0 for lens in token_data[1]])
    return rdata, stream_info, token_data, populated


@pytest.mark.parametrize("window", list(WINDOWS))
def test_target_coords_and_values_are_bit_identical(window):
    rdata, stream_info, token_data, populated = _setup(window)
    for cell_mask in _cell_masks(populated).values():
        args = (stream_info, rdata, token_data, TIME_WIN, cell_mask)
        coords_off = _tokenizer(False).get_target_coords(*args)
        values_off = _tokenizer(False).get_target_values(*args)
        _same(coords_off, _tokenizer(True).get_target_coords(*args))
        _same(values_off, _tokenizer(True).get_target_values(*args))

        # in training the inverse order is skipped, everything else stays
        coords_on = _tokenizer(True).get_target_coords(*args, with_inverse_order=False)
        values_on = _tokenizer(True).get_target_values(*args, with_inverse_order=False)
        assert coords_on[-1] is None and values_on[-1] is None
        _same(coords_off[:-1], coords_on[:-1])
        _same(values_off[:-1], values_on[:-1])


def _sampler(fast, stage):
    sampler = MultiStreamDataSampler.__new__(MultiStreamDataSampler)
    sampler.output_offset = 0
    sampler.time_step = sampler.step_timedelta = 6 * HOUR
    sampler.time_window_handler = TimeWindowHandler(T0, T0 + 48 * HOUR, 6 * HOUR, 6 * HOUR)
    sampler.tokenizer = _tokenizer(fast)
    sampler._stage = stage
    sampler.fast_target_build = fast
    return sampler


def _stream_data_fields(stream_data):
    names = [
        "target_coords",
        "target_coords_lens",
        "target_coords_lens_max",
        "target_coords_raw",
        "target_times_raw",
        "target_tokens",
        "idxs_inv",
        "target_is_spoof",
    ]
    return {name: getattr(stream_data, name) for name in names}


@pytest.mark.parametrize("stage", [TRAIN, VAL])
@pytest.mark.parametrize("window", ["era5", "obs", "without_values"])
def test_sampler_target_stream_data_is_bit_identical(window, stage):
    rng = np.random.default_rng(5)
    make, token_size = WINDOWS[window]
    stream_info = {"name": window, "token_size": token_size, "stream_id": 4}
    output_data = [make(rng), make(rng)]  # two forecast steps
    (_, _, _, populated) = _setup(window)
    for cell_mask in _cell_masks(populated).values():
        fields = []
        for fast in (False, True):
            sampler = _sampler(fast, stage)
            tokens = sampler.tokenizer.get_tokens_windows(stream_info, output_data, False)
            stream_data = sampler._build_stream_data_output(
                ["target_coords", "target_values"],
                StreamData(0, 1, 2, NUM_CELLS),
                0,
                stream_info,
                2,
                output_data,
                tokens,
                cell_mask,
            )
            fields.append(_stream_data_fields(stream_data))
        assert fields[0].keys() == fields[1].keys()
        for name in fields[0]:
            _same(fields[0][name], fields[1][name])
        if stage == VAL and cell_mask.sum() > 0:
            assert any(i is not None and len(i) > 0 for i in fields[1]["idxs_inv"])


def test_option_is_off_by_default():
    assert TokenizerMasking(HL, types.SimpleNamespace(stage=TRAIN)).fast_target_build is False
