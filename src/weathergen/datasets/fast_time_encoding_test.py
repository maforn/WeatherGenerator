# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""`data_loading.fast_time_encoding`: source times encoded once per run of equal times.

encode_times_source_runs must return the bytes of encode_times_source for windows with few
and with many distinct times, and the source tokens built by TokenizerMasking.get_source and
StreamData.add_source must not change (with and without data_loading.fast_source_build).
"""

import types

import numpy as np
import pytest

from weathergen.common.io import IOReaderData
from weathergen.datasets import tokenizer_utils
from weathergen.datasets.stream_data import StreamData
from weathergen.datasets.tokenizer_masking import TokenizerMasking
from weathergen.datasets.tokenizer_utils import encode_times_source, encode_times_source_runs
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


def _hourly(rng, n=20000, start=T0, hours=6):
    """Geostationary-like: sorted, one time per hour."""
    return np.sort(start + rng.integers(0, hours, n) * HOUR)


def _times(name):
    rng = np.random.default_rng(0)
    ns = np.timedelta64(1, "ns")
    match name:
        case "geo_hourly":
            return _hourly(rng)
        case "era5_grid_repeat":
            return np.repeat(T0 + 3 * HOUR * np.arange(2), 5000)
        case "interleaved_shared":  # runs of equal times, not sorted
            return np.repeat(T0 + HOUR * rng.integers(0, 6, 400), rng.integers(1, 60, 400))
        case "per_point_ns":  # a time per point: falls back
            return np.sort(T0 + rng.integers(0, 6 * 3600 * 10**9, 20000) * ns)
        case "near_threshold":  # half the points share their time with the next one
            t = np.sort(T0 + rng.integers(0, 6 * 3600 * 10**9, 6000) * ns)
            return np.repeat(t, rng.integers(1, 4, len(t)))
        case "minutes_and_seconds":
            return np.sort(T0 + rng.integers(0, 6 * 60, 30000) * np.timedelta64(1, "m"))
        case "new_year_leap_day":
            start = np.datetime64("2020-12-31T21:00", "ns")
            return _hourly(rng, start=start)
        case "before_window_start":
            return _hourly(rng, start=T0 - 3 * HOUR)
        case "seconds_unit":
            return _hourly(rng).astype("datetime64[s]")
        case "with_nat":
            t = _hourly(rng)
            t[[5, 6, 7, 4000]] = np.datetime64("NaT")
            return t
        case "two_points":
            return np.array([T0, T0])
        case "one_point":
            return np.array([T0 + HOUR])
    raise ValueError(name)


TIME_CASES = [
    "geo_hourly",
    "era5_grid_repeat",
    "interleaved_shared",
    "per_point_ns",
    "near_threshold",
    "minutes_and_seconds",
    "new_year_leap_day",
    "before_window_start",
    "seconds_unit",
    "with_nat",
    "two_points",
    "one_point",
]


@pytest.mark.parametrize("case", TIME_CASES)
def test_encoded_times_are_bit_identical(case):
    times = _times(case)
    _same(encode_times_source(times, TIME_WIN), encode_times_source_runs(times, TIME_WIN))


def test_runs_are_used_for_shared_times_only(monkeypatch):
    calls = []
    original = tokenizer_utils.encode_times_source

    def counted(times, time_win):
        calls.append(len(times))
        return original(times, time_win)

    monkeypatch.setattr(tokenizer_utils, "encode_times_source", counted)
    encode_times_source_runs(_times("geo_hourly"), TIME_WIN)
    encode_times_source_runs(_times("per_point_ns"), TIME_WIN)
    assert calls == [6, 20000]


def _window(name):
    rng = np.random.default_rng(1)
    if name == "geo":
        times = _hourly(rng, n=15000)
        n = len(times)
        coords = np.stack([rng.uniform(-60, 60, n), rng.uniform(-135, -15, n)], 1)
    elif name == "era5_in":
        grid = np.stack([rng.uniform(-90, 90, 3000), rng.uniform(-180, 180, 3000)], 1)
        coords = np.tile(grid, (2, 1))
        times = np.repeat(T0 + 3 * HOUR * np.arange(2), 3000)
        n = len(times)
    else:  # IASI-like, a time per point
        n = 1500
        coords = np.stack([rng.uniform(-80, 80, n), rng.uniform(-180, 180, n)], 1)
        times = _times("per_point_ns")[:n]
    data = rng.standard_normal((n, 7)).astype(np.float32)
    data[rng.uniform(size=data.shape) < 0.05] = np.nan
    geoinfos = rng.standard_normal((n, 3)).astype(np.float32)
    return IOReaderData(coords.astype(np.float32), geoinfos, data, times)


@pytest.mark.parametrize("source_build", [False, True])
@pytest.mark.parametrize(
    ("window", "token_size", "spacetime"),
    [("geo", 32, False), ("era5_in", 8, True), ("iasi", 16, False)],
)
def test_source_tokens_are_bit_identical(window, token_size, spacetime, source_build):
    rdata = _window(window)
    stream_info = {"name": window, "token_size": token_size, "stream_id": 2}
    stream_info |= {"tokenize_spacetime": spacetime, "train_source_channels": ["ch"]}
    masker = types.SimpleNamespace(stage=TRAIN)
    rng = np.random.default_rng(2)
    for cell_mask in (np.ones(NUM_CELLS, dtype=bool), rng.uniform(size=NUM_CELLS) < 0.5):
        out = []
        for fast in (False, True):
            tokenizer = TokenizerMasking(
                HL, masker, fast_source_build=source_build, fast_time_encoding=fast
            )
            (token_data,) = tokenizer.get_tokens_windows(stream_info, [rdata], True)
            cells, lens = tokenizer.get_source(stream_info, rdata, token_data, TIME_WIN, cell_mask)
            stream_data = StreamData(0, 1, 1, NUM_CELLS)
            stream_data.add_source(TRAIN, 0, None, lens, cells, False)
            out.append((stream_data.source_tokens_cells[0], stream_data.source_tokens_lens[0]))
        for a, b in zip(*out, strict=True):
            _same(a, b)


def test_option_is_off_by_default():
    tokenizer = TokenizerMasking(HL, types.SimpleNamespace(stage=TRAIN))
    assert tokenizer.encode_times_source is encode_times_source
