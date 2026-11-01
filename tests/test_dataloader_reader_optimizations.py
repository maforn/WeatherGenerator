"""Exact-output checks for the optional reader paths, using tiny in-memory datasets."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
    def __init__(self, values, variables=None):
        self.values = values
        self.variables = variables or [f"v{i}" for i in range(values.shape[1])]
        self.dates = np.arange("2020-01-01T00", "2020-01-01T04", dtype="datetime64[h]")
        self.shape = values.shape

    def __getitem__(self, index):
        # Deliberately return a view: the reader must not mutate cached data.
        return self.values[index]


def select_variables(dataset, *, select):
    # Reverse the returned order to test explicit remapping, not assumptions.
    names = list(reversed(select))
    indices = [dataset.variables.index(name) for name in names]
    return AnemoiDataset(dataset.values[:, indices], names)


class ReaderOptimizationTests(unittest.TestCase):
    def assert_reader_equal(self, expected, actual):
        for field in ("coords", "geoinfos", "data", "datetimes"):
            left, right = getattr(expected, field), getattr(actual, field)
            self.assertEqual(left.dtype, right.dtype)
            np.testing.assert_array_equal(left, right)

    def make_obs(self, flags, empty_geo=False):
        reader = object.__new__(DataReaderObs)
        reader.optimizations = flags
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
            if not bad:
                self.assertIs(actual.data, data.data)
                self.assertFalse(np.shares_memory(expected.data, data.data))

    def make_anemoi(self, cls, flags, dtype):
        reader = object.__new__(cls)
        reader.optimizations = flags
        reader._selected_datasets = {}
        reader.ds = AnemoiDataset(np.arange(160, dtype=dtype).reshape(4, 5, 2, 4))
        reader.len = 4
        reader.geoinfo_idx = [3, 1]
        reader.latitudes = np.arange(4, dtype=np.float32)
        reader.longitudes = np.arange(4, dtype=np.float32) + 30
        reader.stream_info = {"nominal_time_mapping": {str(i): str(i) for i in range(24)}}
        reader._get_dataset_idxs = lambda _: (
            np.array([1, 2]), SimpleNamespace(start=reader.ds.dates[1], end=reader.ds.dates[3])
        )
        return reader

    def test_anemoi_flags_preserve_values_and_dataset_storage(self):
        with patch(
            "weathergen.datasets.data_reader_anemoi.anemoi_datasets.open_dataset",
            side_effect=select_variables,
        ) as open_dataset:
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
                        expected = baseline._get(0, [4, 0, 4])
                        actual = reader._get(0, [4, 0, 4])
                        self.assert_reader_equal(expected, actual)
                        actual.data[:] = -1
                        actual.geoinfos[:] = -2
                        np.testing.assert_array_equal(reader.ds.values, saved)
                        calls = open_dataset.call_count
                        self.assert_reader_equal(expected, reader._get(0, [4, 0, 4]))
                        self.assertEqual(calls, open_dataset.call_count)

    def test_anemoi_selection_rejects_unsupported_api(self):
        reader = self.make_anemoi(DataReaderAnemoi, {"anemoi_select_vars": True}, np.float32)
        with patch(
            "weathergen.datasets.data_reader_anemoi.anemoi_datasets.open_dataset",
            side_effect=TypeError("unsupported select"),
        ), self.assertRaisesRegex(RuntimeError, "anemoi_select_vars requires"):
            reader._get(0, [4])

    def test_anemoi_selection_rejects_wrong_variables(self):
        reader = self.make_anemoi(DataReaderAnemoi, {"anemoi_select_vars": True}, np.float32)
        with patch(
            "weathergen.datasets.data_reader_anemoi.anemoi_datasets.open_dataset",
            return_value=reader.ds,
        ), self.assertRaisesRegex(RuntimeError, "anemoi_select_vars requires"):
            reader._get(0, [4])


if __name__ == "__main__":
    unittest.main()
