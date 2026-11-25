# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""`data_loading.fast_read_copies`: fewer copies between the read and the torch tensors.

Synthetic obs zarr stores are read by DataReaderObs and an ERA5-like cube by DataReaderAnemoi,
then go through collect_datasources, tokenization, TokenizerMasking.get_source with
StreamData.add_source, and get_target_coords / get_target_values, as in the sampler. With the
option every output must have the same dtype, shape and bytes as without.
"""

import types

import numpy as np
import pytest
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import (
    DataReaderBase,
    DTRange,
    ReaderData,
    TimeWindowHandler,
)
from weathergen.datasets.data_reader_obs import DataReaderObs
from weathergen.datasets.multi_stream_data_sampler import collect_datasources
from weathergen.datasets.stream_data import StreamData
from weathergen.datasets.tokenizer_masking import TokenizerMasking, readerdata_to_torch
from weathergen.train.utils import TRAIN

HL = 3
NUM_CELLS = 12 * 4**HL
T0 = np.datetime64("2021-03-01T00:00", "ns")
HOUR = np.timedelta64(1, "h")
WINDOW_HOURS = 6
NUM_WINDOWS = 4


def _same(a, b):
    """Same type, dtype, shape and bytes (NaN payloads and signed zeros included)."""
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        a, b = a.numpy(), b.numpy()
    assert a.dtype == b.dtype and a.shape == b.shape
    # compared outside the assert: pytest would otherwise diff the bytes
    equal = np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    assert equal, "bytes differ"


def _same_reader_data(a, b):
    assert type(a) is type(b) and a.is_spoof == b.is_spoof
    for name in ("coords", "geoinfos", "data", "datetimes"):
        _same(getattr(a, name), getattr(b, name))


# synthetic obs streams


def _geo_points(rng, num):
    """Fixed pixels of a geostationary-like disk."""
    lat = rng.uniform(-60, 60, num)
    lon = rng.uniform(-135, -15, num)
    return lat, lon


def _write_obs_zarr(path, rng, kind, start_hour, num_hours, nan_rows, index_shift_minutes):
    """An obs zarr as DataReaderObs reads it: data (rows, columns) with colnames, means and
    vars, dates (rows, 1), and the hourly index of the first row of every hour."""
    import zarr

    if kind == "geo":  # dense, same pixels every hour, one timestamp per hour
        num_geo, num_ch = 6, 8
        lat0, lon0 = _geo_points(rng, 3000)
        rows = []
        for h in range(start_hour, start_hour + num_hours):
            keep = rng.uniform(size=len(lat0)) < 0.95
            t = np.full(keep.sum(), T0 + h * HOUR)
            rows.append((lat0[keep], lon0[keep], t))
    else:  # IASI-like: sparse, many channels, every point at its own time
        num_geo, num_ch = 4, 40
        rows = []
        for h in range(start_hour, start_hour + num_hours):
            n = int(rng.integers(150, 250))
            seconds = np.sort(rng.integers(0, 3600 * 10**9, n)).astype("timedelta64[ns]")
            lat, lon = rng.uniform(-80, 80, n), rng.uniform(-180, 180, n)
            rows.append((lat, lon, T0 + h * HOUR + seconds))
    lat = np.concatenate([r[0] for r in rows])
    lon = np.concatenate([r[1] for r in rows])
    times = np.concatenate([r[2] for r in rows])
    num_rows = len(times)

    colnames = (
        ["lat", "lon"]
        + [f"geo_{i}" for i in range(num_geo)]
        + [f"obsvalue_ch{i}_0" for i in range(num_ch)]
    )
    data = np.empty((num_rows, len(colnames)), dtype=np.float32)
    data[:, 0], data[:, 1] = lat, lon
    data[:, 2:] = rng.standard_normal((num_rows, len(colnames) - 2)) * 3 + 1
    # missing channel values are common; they reach the tokens as NaN and are masked there
    data[:, 2 + num_geo :][rng.uniform(size=(num_rows, num_ch)) < 0.05] = np.nan
    data[rng.uniform(size=num_rows) < 0.02, 2 + num_geo] = -0.0
    if nan_rows:
        bad = rng.choice(num_rows, max(1, num_rows // 50), replace=False)
        data[bad[: len(bad) // 2], 0] = np.nan  # lat
        data[bad[len(bad) // 2 :], 2] = np.nan  # a geoinfo
    means = data.astype(np.float64)
    means = np.nanmean(means, axis=0)
    variances = np.nanvar(data.astype(np.float64), axis=0)
    variances[3] = 0.0  # a constant geoinfo field: normalize_geoinfos divides by 1.0

    # hourly index: first row at or after each hour (shifted: rows of the previous window enter
    # the row range and only the time mask removes them)
    num_index = start_hour + num_hours + 2 * WINDOW_HOURS + 2
    hours = T0 + np.arange(num_index) * HOUR - np.timedelta64(index_shift_minutes, "m")
    hrly_index = np.searchsorted(times, hours).astype(np.int64)

    group = zarr.open_group(str(path), mode="w")
    arr = group.create_array("data", data=data)
    arr.attrs["colnames"] = colnames
    arr.attrs["means"] = means.tolist()
    arr.attrs["vars"] = variances.tolist()
    group.create_array("dates", data=times.reshape(-1, 1))
    group.create_array("idx_202103010000_1", data=hrly_index)
    return colnames


def _tw_handler():
    return TimeWindowHandler(
        T0, T0 + (NUM_WINDOWS + 1) * WINDOW_HOURS * HOUR, WINDOW_HOURS * HOUR, WINDOW_HOURS * HOUR
    )


def _stream_info(kind, geoinfo_channels):
    info = {
        "name": f"obs_{kind}",
        "base_datetime": "2021-03-01T00:00:00",
        "token_size": 64 if kind == "geo" else 32,
        "stream_id": 3,
        "source": ["obsvalue"],
        "target": ["obsvalue_ch1", "obsvalue_ch2"],
    }
    if geoinfo_channels is not None:
        info["geoinfo_channels"] = geoinfo_channels
    return info


def _obs_readers(tmp_path, kind, files, nan_rows, index_shift, geoinfo_channels, fast):
    """files: (start_hour, num_hours) per zarr; the same seed gives the same stores."""
    readers = []
    for i, (start_hour, num_hours) in enumerate(files):
        path = tmp_path / f"{kind}_{i}_{fast}.zarr"
        rng = np.random.default_rng(100 + i)
        _write_obs_zarr(path, rng, kind, start_hour, num_hours, nan_rows, index_shift)
        info = _stream_info(kind, geoinfo_channels)
        reader = DataReaderObs(_tw_handler(), path, info, TRAIN)
        reader.fast_read_copies = fast
        readers.append(reader)
    return readers


# ERA5-like gridded stream: the anemoi reader on a stand-in cube


class _Cube:
    """Stand-in for an anemoi dataset: (time, var, ensemble, grid) and its dates."""

    def __init__(self, num_grid, num_vars, seed=0):
        rng = np.random.default_rng(seed)
        self.dates = T0 + HOUR * np.arange(64)
        self._values = rng.standard_normal((64, num_vars, 1, num_grid)).astype(np.float32)

    def __getitem__(self, key):
        return self._values[key]


def _era5_reader(fast, stats_dtype, num_grid=2000, num_vars=12):
    reader = object.__new__(DataReaderAnemoi)
    reader.stream_info = {
        "name": "ERA5",
        "token_size": 16,
        "stream_id": 1,
        "max_num_targets": 3000,
    }
    reader.ds, reader.len = _Cube(num_grid, num_vars), 64
    reader.source_idx, reader.target_idx = [0, 1, 2, 5, 6, 7, 8], [5, 6, 2]
    reader.geoinfo_idx = [9, 10, 11]
    rng = np.random.default_rng(1)
    z = rng.uniform(-1, 1, num_grid)
    reader._latlon = np.stack(
        [np.rad2deg(np.arcsin(z)), rng.uniform(-180, 180, num_grid)], axis=1
    ).astype(np.float32)
    stats = rng.uniform(0.5, 2.0, (2, num_vars))
    reader.mean = stats[0].astype(np.float32)
    reader.stdev = stats[1].astype(np.float32)
    reader.mean_geoinfo = stats[0][reader.geoinfo_idx].astype(stats_dtype)
    reader.stdev_geoinfo = stats[1][reader.geoinfo_idx].astype(stats_dtype)

    def dataset_idxs(idx):
        start = T0 + idx * WINDOW_HOURS * HOUR
        steps = np.arange(idx * WINDOW_HOURS, idx * WINDOW_HOURS + 2, dtype=np.int64)
        return steps, DTRange(start, start + WINDOW_HOURS * HOUR)

    reader._get_dataset_idxs = dataset_idxs
    reader.fast_read_copies = fast
    return reader


# the sampler's path for one stream and window


def _tokenizer(fast):
    masker = types.SimpleNamespace(stage=TRAIN, reset_rng=lambda rng: None)
    return TokenizerMasking(HL, masker, fast_read_copies=fast)


def _window(idx):
    start = T0 + idx * WINDOW_HOURS * HOUR
    return (start, start + WINDOW_HOURS * HOUR)


def _run(readers, idx, fast, cell_mask):
    """collect_datasources, tokenization, source and target building for window idx."""
    stream_info = dict(readers[0].stream_info)
    stream_info["train_source_channels"] = list(range(len(readers[0].source_idx)))
    rng = np.random.default_rng(7)
    out = {}
    source = collect_datasources(readers, idx, "source", rng, fast_read_copies=fast)
    target = collect_datasources(readers, idx, "target", rng, fast_read_copies=fast)
    out["source_rdata"], out["target_rdata"] = source, target
    # remember the numpy arrays: tokenization replaces them by tensors
    out["source_np"] = IOReaderData(source.coords, source.geoinfos, source.data, source.datetimes)

    tokenizer = _tokenizer(fast)
    if source.is_empty():
        return out
    (source_tokens,) = tokenizer.get_tokens_windows(stream_info, [source], True)
    (target_tokens,) = tokenizer.get_tokens_windows(stream_info, [target], False)
    stream_data = StreamData(idx, 1, 1, NUM_CELLS)
    cells, lens = tokenizer.get_source(stream_info, source, source_tokens, _window(idx), cell_mask)
    stream_data.add_source(TRAIN, 0, source, lens, cells, False)
    out["source_tokens"] = stream_data.source_tokens_cells[0]
    out["source_lens"] = stream_data.source_tokens_lens[0]
    if not target.is_empty():
        out["target_coords"] = tokenizer.get_target_coords(
            stream_info, target, target_tokens, _window(idx), cell_mask
        )
        out["target_values"] = tokenizer.get_target_values(
            stream_info, target, target_tokens, _window(idx), cell_mask
        )
    return out


def _compare(off, on):
    assert off.keys() == on.keys()
    for key in off:
        if key in ("source_rdata", "target_rdata"):
            _same_reader_data(off[key], on[key])
        elif key == "source_np":
            for name in ("coords", "geoinfos", "data", "datetimes"):
                _same(getattr(off[key], name), getattr(on[key], name))
        elif key in ("target_coords", "target_values"):
            assert len(off[key]) == len(on[key])
            for a, b in zip(off[key], on[key], strict=True):
                if a is None or b is None:
                    assert a is None and b is None
                else:
                    _same(a, b)
        else:
            _same(off[key], on[key])


def _cell_masks():
    rng = np.random.default_rng(3)
    return [np.ones(NUM_CELLS, dtype=bool), rng.uniform(size=NUM_CELLS) < 0.4]


OBS_CASES = {
    # kind, files (start hour, hours), NaN rows, index shift (min), geoinfo channels
    "geo": ("geo", [(0, 30)], False, 0, None),
    "geo_nan_rows": ("geo", [(0, 30)], True, 0, None),
    "geo_time_mask_drops_rows": ("geo", [(0, 30)], False, 90, None),
    "geo_two_files": ("geo", [(0, 9), (9, 21)], True, 0, None),
    "iasi": ("iasi", [(0, 30)], False, 0, None),
    "iasi_nan_rows_time_mask": ("iasi", [(0, 30)], True, 20, None),
    "iasi_no_geoinfos": ("iasi", [(0, 30)], False, 0, []),
    "iasi_geoinfo_is_a_channel": ("iasi", [(0, 30)], True, 0, ["geo_1", "obsvalue_ch2_0"]),
    "iasi_three_files": ("iasi", [(0, 7), (7, 5), (12, 18)], False, 10, None),
}


@pytest.mark.parametrize("case", list(OBS_CASES))
def test_obs_stream_outputs_are_bit_identical(tmp_path, case):
    kind, files, nan_rows, shift, geoinfo_channels = OBS_CASES[case]
    readers_off = _obs_readers(tmp_path, kind, files, nan_rows, shift, geoinfo_channels, False)
    readers_on = _obs_readers(tmp_path, kind, files, nan_rows, shift, geoinfo_channels, True)
    for idx in range(NUM_WINDOWS):
        for cell_mask in _cell_masks():
            _compare(
                _run(readers_off, idx, False, cell_mask), _run(readers_on, idx, True, cell_mask)
            )


@pytest.mark.parametrize("stats_dtype", [np.float32, np.float64])
def test_era5_stream_outputs_are_bit_identical(stats_dtype):
    for idx in range(NUM_WINDOWS):
        for cell_mask in _cell_masks():
            off = _run([_era5_reader(False, stats_dtype)], idx, False, cell_mask)
            on = _run([_era5_reader(True, stats_dtype)], idx, True, cell_mask)
            _compare(off, on)


def _record_reads(reader):
    """Keep what reader.get_source returns."""
    read, get_source = [], reader.get_source

    def recorded(idx):
        read.append(get_source(idx))
        return read[-1]

    reader.get_source = recorded
    return read


def test_obs_reader_skips_the_copies(tmp_path):
    """The option does what it is for: one block of columns, no NaN-filter or combine copy, and
    tensors on the same memory."""
    (reader,) = _obs_readers(tmp_path, "geo", [(0, 30)], False, 0, None, True)
    read = _record_reads(reader)
    collected = collect_datasources([reader], 1, "source", None, fast_read_copies=True)
    (rdata,) = read
    assert rdata.coords.base is not None and rdata.coords.base is rdata.data.base
    assert collected.data is rdata.data and collected.geoinfos is rdata.geoinfos
    data = collected.data
    readerdata_to_torch(collected, share_memory=True)
    assert np.shares_memory(collected.data.numpy(), data)


def test_nan_filter_still_copies_arrays_of_other_readers():
    """Readers that do not declare private arrays (e.g. cached ones) keep the copy, because the
    normalization writes into the arrays."""

    class _CachedReader(DataReaderAnemoi):
        pass

    reader = _era5_reader(True, np.float32)
    read = _record_reads(reader)
    collected = collect_datasources([reader], 1, "source", None, fast_read_copies=True)
    assert collected.data is read[0].data  # the declaring class: no copy

    reader = _era5_reader(True, np.float32)
    reader.__class__ = _CachedReader
    read = _record_reads(reader)
    collected = collect_datasources([reader], 1, "source", None, fast_read_copies=True)
    assert not np.shares_memory(collected.data, read[0].data)


def test_combine_keeps_promotion_with_an_empty_first_reader():
    """An empty first item (ReaderData.empty has generic datetime64) makes combine promote, so
    the copy-free path must not be taken."""
    rng = np.random.default_rng(0)
    full = ReaderData(
        rng.standard_normal((5, 2)).astype(np.float32),
        np.zeros((5, 1), dtype=np.float32),
        rng.standard_normal((5, 3)).astype(np.float32),
        np.repeat(T0, 5),
    )
    empty = ReaderData.empty(num_data_fields=3, num_geo_fields=1)
    readers = []
    for rdata in (empty, full):
        reader = types.SimpleNamespace(
            stream_info={"name": "s"},
            get_source=lambda idx, r=rdata: r,
            normalize_source_channels=lambda d: d,
            normalize_geoinfos=lambda g: g,
        )
        readers.append(reader)
    off = collect_datasources(readers, 0, "source", None)
    on = collect_datasources(readers, 0, "source", None, fast_read_copies=True)
    _same_reader_data(off, on)


def test_option_is_off_by_default():
    assert DataReaderBase.fast_read_copies is False
    assert _tokenizer(False).fast_read_copies is False
    assert TokenizerMasking(HL, types.SimpleNamespace(stage=TRAIN)).fast_read_copies is False
