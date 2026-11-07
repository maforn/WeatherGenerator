"""Exact-output checks for the optional reader paths, using tiny in-memory datasets."""

import unittest
from types import SimpleNamespace

import numpy as np

from weathergen.datasets.data_reader_anemoi import DataReaderAnemoi
from weathergen.datasets.data_reader_base import ReaderData
from weathergen.datasets.data_reader_obs import DataReaderObs
from weathergen.readers_extra.data_reader_anemoi_operan import DataReaderAnemoiOperan


class ArrayStore:
    def __init__(self, values):
        self.values = values
        self.oindex = self
        self.reads = []

    def __getitem__(self, index):
        rows, cols = index
        self.reads.append((rows, list(cols)))
        return self.values[rows][:, cols].copy()


class AnemoiDataset:
    def __init__(self, values):
        self.values = values
        self.dates = np.arange("2020-01-01T00", "2020-01-01T04", dtype="datetime64[h]")
        self.shape = values.shape
        self.reads = []

    def __getitem__(self, index):
        self.reads.append(index)
        # Deliberately return a view: the reader must not mutate cached data.
        return self.values[index]


class ReaderOptimizationTests(unittest.TestCase):
    def assert_reader_equal(self, expected, actual):
        for field in ("coords", "geoinfos", "data", "datetimes"):
            left, right = getattr(expected, field), getattr(actual, field)
            self.assertEqual(left.dtype, right.dtype)
            np.testing.assert_array_equal(left, right)

    def make_obs(self, flags, empty_geo=False):
        reader = object.__new__(DataReaderObs)
        for flag, value in flags.items():  # the sampler sets these from cf
            setattr(reader, flag, value)
        values = np.arange(42, dtype=np.float32).reshape(6, 7)
        reader.data = ArrayStore(values)
        dates = np.arange("2020-01-01T00", "2020-01-01T06", dtype="datetime64[h]")
        reader.dt = dates[:, None]
        reader.indices_start, reader.indices_end = [0], [6]
        reader.coords_idx = [5, 1]
        reader.geoinfo_idx = [] if empty_geo else [3, 5]
        reader.time_window_handler = SimpleNamespace(
            window=lambda _: SimpleNamespace(start=dates[1], end=dates[5])
        )
        return reader

    def test_obs_union_preserves_order_overlap_duplicates_and_mask(self):
        for empty_geo in (False, True):
            original = self.make_obs({}, empty_geo)
            optimized = self.make_obs({"obs_single_read": True}, empty_geo)
            expected = original._get(0, [4, 2, 4])
            actual = optimized._get(0, [4, 2, 4])
            self.assert_reader_equal(expected, actual)
            self.assertEqual(len(optimized.data.reads), 1)
            self.assertEqual(len(original.data.reads), 2 if empty_geo else 3)
            actual.data[:] = -100
            self.assertGreaterEqual(optimized.data.values.min(), 0)

    def test_obs_all_valid_and_empty_channel_paths(self):
        reader = self.make_obs({"obs_single_read": True, "skip_all_valid_filter_copy": True})
        reader.time_window_handler.window = lambda _: SimpleNamespace(
            start=reader.dt[0, 0], end=reader.dt[-1, 0] + np.timedelta64(1, "h")
        )
        actual = reader._get(0, [2])
        np.testing.assert_array_equal(actual.data[:, 0], reader.data.values[:, 2])
        self.assertEqual(reader._get(0, []).data.shape, (0, 0))
        self.assertEqual(reader._get(2, [2]).data.shape, (0, 1))

    def test_nan_filter_fast_path_keeps_partial_filtering(self):
        for bad in (False, True):
            data = ReaderData(
                np.ones((4, 2), np.float32),
                np.ones((4, 1), np.float32),
                np.arange(8, dtype=np.float32).reshape(4, 2),
                np.arange(4).astype("datetime64[h]"),
            )
            if bad:
                data.coords[1, 0] = np.nan
                data.geoinfos[2, 0] = np.nan
            expected = data.remove_nan_coords_and_geoinfos()
            actual = data.remove_nan_coords_and_geoinfos(skip_all_valid_filter_copy=True)
            self.assert_reader_equal(expected, actual)
            # views only when every row is valid
            for field in ("coords", "geoinfos", "data", "datetimes"):
                source, output = getattr(data, field), getattr(actual, field)
                self.assertEqual(np.shares_memory(output, source), not bad)
                self.assertFalse(np.shares_memory(getattr(expected, field), source))

    def make_anemoi(self, cls, flags, dtype, geoinfo_idx=(3, 1)):
        reader = object.__new__(cls)
        for flag, value in flags.items():  # the sampler sets these from cf
            setattr(reader, flag, value)
        reader.ds = AnemoiDataset(np.arange(160, dtype=dtype).reshape(4, 5, 2, 4))
        reader.len = 4
        reader.geoinfo_idx = list(geoinfo_idx)
        reader.latitudes = np.arange(4, dtype=np.float32)
        reader.longitudes = np.arange(4, dtype=np.float32) + 30
        reader.stream_info = {"nominal_time_mapping": {str(i): str(i) for i in range(24)}}
        reader._get_dataset_idxs = lambda _: (
            np.array([1, 2]),
            SimpleNamespace(start=reader.ds.dates[1], end=reader.ds.dates[3]),
        )
        return reader

    def test_anemoi_flags_preserve_values_and_dataset_storage(self):
        # unsorted, repeated, and overlapping with the geoinfos (3, 1)
        for channels in ([4, 0, 4], [1, 2]):
            for cls in (DataReaderAnemoi, DataReaderAnemoiOperan):
                for dtype in (np.float32, np.float64):
                    for selection, avoid_copy in ((False, True), (True, False), (True, True)):
                        flags = {
                            "anemoi_select_vars": selection,
                            "anemoi_avoid_dtype_copy": avoid_copy,
                        }
                        baseline = self.make_anemoi(cls, {}, dtype)
                        reader = self.make_anemoi(cls, flags, dtype)
                        saved = reader.ds.values.copy()
                        expected = baseline._get(0, channels)
                        actual = reader._get(0, channels)
                        self.assert_reader_equal(expected, actual)
                        # the read itself is unchanged: one slice of every variable
                        self.assertEqual(reader.ds.reads, baseline.ds.reads)
                        actual.data[:] = -1
                        actual.geoinfos[:] = -2
                        np.testing.assert_array_equal(reader.ds.values, saved)

    def test_anemoi_selection_converts_only_used_variables(self):
        for dtype in (np.float32, np.float64):
            reader = self.make_anemoi(DataReaderAnemoi, {"anemoi_select_vars": True}, dtype)
            raw = reader.ds.values[1:3]
            data, channels, geoinfos = reader._read_variables(1, 3, [4, 4])
            # variables 1, 3 (geoinfos) and 4 (channel) of 5, in that order
            self.assertEqual(data.shape, (2, 3, 4))
            self.assertEqual(data.dtype, np.float32)
            self.assertEqual((channels, geoinfos), ([2, 2], [1, 0]))
            np.testing.assert_array_equal(data, raw[:, [1, 3, 4], 0].astype(np.float32))
            self.assertFalse(np.shares_memory(data, raw))

    def test_anemoi_selection_empty_channels_or_geoinfos(self):
        for cls in (DataReaderAnemoi, DataReaderAnemoiOperan):
            for channels, geoinfo_idx in (([], (3, 1)), ([2], ()), ([], ())):
                baseline = self.make_anemoi(cls, {}, np.float32, geoinfo_idx)
                reader = self.make_anemoi(
                    cls, {"anemoi_select_vars": True}, np.float32, geoinfo_idx
                )
                self.assert_reader_equal(baseline._get(0, channels), reader._get(0, channels))


if __name__ == "__main__":
    unittest.main()
