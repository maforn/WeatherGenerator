# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""`data_loading.fast_source_build`: source tokens written into one tensor.

Synthetic windows shaped like the streams are tokenized as in the sampler
(TokenizerMasking.get_tokens_windows), then TokenizerMasking.get_source and
StreamData.add_source build the source tokens with the option off and on. The tokens and the
tokens per cell must have the same dtype, shape and bytes.
"""

import types

import numpy as np
import pytest
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.stream_data import StreamData
from weathergen.datasets.tokenizer_masking import TokenizerMasking
from weathergen.train.utils import TRAIN

HL = 3
NUM_CELLS = 12 * 4**HL
T0 = np.datetime64("2021-03-01T06:00", "ns")
HOUR = np.timedelta64(1, "h")
TIME_WIN = (T0, T0 + 6 * HOUR)


def _same(a, b):
    """Same dtype, shape and bytes (NaN payloads and signed zeros included)."""
    assert type(a) is type(b) and a.dtype == b.dtype and a.shape == b.shape
    # compared outside the assert: pytest would otherwise diff the bytes
    equal = a.contiguous().numpy().tobytes() == b.contiguous().numpy().tobytes()
    assert equal, "bytes differ"


def _channels(rng, num, num_ch, dtype=np.float32):
    data = (rng.standard_normal((num, num_ch)) * 2).astype(dtype)
    data[rng.uniform(size=data.shape) < 0.05] = np.nan
    data[rng.uniform(size=data.shape) < 0.01] = -0.0
    return data


def _geo(rng, num_pixels=4000, hours=6, num_ch=8, num_geo=6):
    """Dense, the same pixels every hour, one time per hour."""
    lat, lon = rng.uniform(-60, 60, num_pixels), rng.uniform(-135, -15, num_pixels)
    coords, times = [], []
    for h in range(hours):
        keep = rng.uniform(size=num_pixels) < 0.95
        coords.append(np.stack([lat[keep], lon[keep]], 1))
        times.append(np.full(keep.sum(), T0 + h * HOUR))
    coords = np.concatenate(coords).astype(np.float32)
    num = len(coords)
    geoinfos = rng.standard_normal((num, num_geo)).astype(np.float32)
    return IOReaderData(coords, geoinfos, _channels(rng, num, num_ch), np.concatenate(times))


def _iasi(rng, num=1500, num_ch=40, num_geo=4, data_dtype=np.float32):
    """Sparse (padding-heavy tokens), many channels, a time per point."""
    coords = np.stack([rng.uniform(-80, 80, num), rng.uniform(-180, 180, num)], 1)
    seconds = np.sort(rng.integers(0, 6 * 3600 * 10**9, num)).astype("timedelta64[ns]")
    geoinfos = rng.standard_normal((num, num_geo)).astype(np.float32)
    data = _channels(rng, num, num_ch, data_dtype)
    return IOReaderData(coords.astype(np.float32), geoinfos, data, T0 + seconds)


def _era5_in(rng, num_grid=3000, steps=2, num_ch=10, num_geo=3):
    """Gridded, the same grid at every step (tokenize_spacetime)."""
    z = rng.uniform(-1, 1, num_grid)
    grid = np.stack([np.rad2deg(np.arcsin(z)), rng.uniform(-180, 180, num_grid)], 1)
    coords = np.tile(grid, (steps, 1)).astype(np.float32)
    times = np.repeat(T0 + 3 * HOUR * np.arange(steps), num_grid)
    geoinfos = rng.standard_normal((len(coords), num_geo)).astype(np.float64)
    return IOReaderData(coords, geoinfos, _channels(rng, len(coords), num_ch), times)


def _single_point(rng):
    return IOReaderData(
        np.array([[12.5, -40.0]], dtype=np.float32),
        np.array([[0.5, -0.25]], dtype=np.float32),
        np.array([[np.nan, 1.5, -0.0]], dtype=np.float32),
        np.array([T0 + HOUR]),
    )


def _strided(rng):
    """Arrays that are views of one block, as data_loading.fast_read_copies passes them."""
    rdata = _geo(rng, num_pixels=1500)
    block = np.concatenate([rdata.coords, rdata.geoinfos, rdata.data], 1)
    return IOReaderData(block[:, :2], block[:, 2:8], block[:, 8:], rdata.datetimes)


WINDOWS = {
    # name: (window, token_size, tokenize_spacetime)
    "geo": (_geo, 64, False),
    "geo_small_tokens": (_geo, 16, False),
    "iasi_padding_heavy": (_iasi, 32, False),
    "iasi_float64_data": (lambda rng: _iasi(rng, data_dtype=np.float64), 32, False),
    "era5_in_spacetime": (_era5_in, 8, True),
    "single_point": (_single_point, 4, False),
    "strided_views": (_strided, 48, False),
}


def _cell_masks(rdata_cells_populated):
    rng = np.random.default_rng(3)
    populated = rdata_cells_populated
    masks = {
        "all": np.ones(NUM_CELLS, dtype=bool),
        "random": rng.uniform(size=NUM_CELLS) < 0.4,
        "none": np.zeros(NUM_CELLS, dtype=bool),
        "only_empty_cells": ~populated,
        "torch_float": torch.from_numpy((rng.uniform(size=NUM_CELLS) < 0.5).astype(np.float32)),
    }
    first = np.zeros(NUM_CELLS, dtype=bool)
    first[np.argmax(populated)] = True  # first row of the tokens is a real point
    masks["first_populated_cell"] = first
    return masks


def _tokenizer(fast):
    masker = types.SimpleNamespace(stage=TRAIN, reset_rng=lambda rng: None)
    return TokenizerMasking(HL, masker, fast_source_build=fast)


def _source(fast, stream_info, rdata, cell_mask):
    tokenizer = _tokenizer(fast)
    (token_data,) = tokenizer.get_tokens_windows(stream_info, [rdata], True)
    cells, lens = tokenizer.get_source(stream_info, rdata, token_data, TIME_WIN, cell_mask)
    stream_data = StreamData(0, 1, 1, NUM_CELLS)
    stream_data.add_source(TRAIN, 0, None, lens, cells, False)
    return stream_data.source_tokens_cells[0], stream_data.source_tokens_lens[0]


@pytest.mark.parametrize("diagnostic", [False, True])
@pytest.mark.parametrize("window", list(WINDOWS))
def test_source_tokens_are_bit_identical(window, diagnostic):
    make, token_size, spacetime = WINDOWS[window]
    rdata = make(np.random.default_rng(0))
    stream_info = {
        "name": window,
        "token_size": token_size,
        "stream_id": 5,
        "tokenize_spacetime": spacetime,
        "train_source_channels": [] if diagnostic else ["ch"],
    }
    (token_data,) = _tokenizer(False).get_tokens_windows(stream_info, [rdata], True)
    populated = np.array([len(lens) > 0 for lens in token_data[1]])
    for name, cell_mask in _cell_masks(populated).items():
        off = _source(False, stream_info, rdata, cell_mask)
        on = _source(True, stream_info, rdata, cell_mask)
        for a, b in zip(off, on, strict=True):
            _same(a, b)
        assert name != "all" or off[0].numel() > 0


def test_option_is_off_by_default():
    masker = types.SimpleNamespace(stage=TRAIN)
    assert TokenizerMasking(HL, masker).fast_source_build is False


@pytest.mark.parametrize("window", ["geo", "iasi_padding_heavy", "era5_in_spacetime"])
def test_the_direct_build_is_used(window):
    make, token_size, spacetime = WINDOWS[window]
    rdata = make(np.random.default_rng(0))
    stream_info = {"name": window, "token_size": token_size, "stream_id": 5}
    stream_info |= {"tokenize_spacetime": spacetime, "train_source_channels": ["ch"]}
    tokenizer = _tokenizer(True)
    (token_data,) = tokenizer.get_tokens_windows(stream_info, [rdata], True)
    cells, _ = tokenizer.get_source(
        stream_info, rdata, token_data, TIME_WIN, np.ones(NUM_CELLS, dtype=bool)
    )
    assert isinstance(cells, torch.Tensor) and cells.shape[1] == token_size
