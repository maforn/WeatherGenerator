"""Synthetic parity checks for opt-in CPU tokenizer optimizations."""

import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from weathergen.common.io import IOReaderData
from weathergen.datasets import stream_data as stream_data_module
from weathergen.datasets import tokenizer_utils
from weathergen.datasets.stream_data import StreamData
from weathergen.datasets.tokenizer_masking import TokenizerMasking
from weathergen.datasets.tokenizer_utils import tokenize_spacetime
from weathergen.train.utils import TRAIN

FLAGS = {
    "target_requested_fields_only": True,
    "skip_train_inverse_indices": True,
    "tokenize_coords_only": True,
    "source_direct_build": True,
}
STREAM = {"stream_id": 2, "token_size": 4}
WINDOW = (np.datetime64("2020-01-01T00"), np.datetime64("2020-01-01T06"))


def sample():
    return IOReaderData(
        coords=torch.tensor([[45.0, 5.0], [46.0, 6.0], [-45.0, 70.0], [0.0, -90.0], [12.0, 40.0]]),
        geoinfos=torch.arange(10, dtype=torch.float32).reshape(5, 2),
        data=torch.tensor([[1.0, 2.0], [3.0, float("nan")], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]),
        datetimes=np.array(["2020-01-01T00"] * 3 + ["2020-01-01T03"] * 2, dtype="datetime64[ns]"),
    )


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    elif isinstance(a, np.ndarray):  # noqa: TID251 (runtime type check, not an annotation)
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, tuple | list):
        assert len(a) == len(b)
        for first, second in zip(a, b, strict=True):
            assert_equal(first, second)
    else:
        assert a == b


@pytest.fixture
def tokenizers():
    baseline = TokenizerMasking(0, None)
    optimized = TokenizerMasking(0, None, optimizations=FLAGS)
    return baseline, optimized


@pytest.mark.parametrize("selection", ["all", "some", "none"])
def test_source_direct_build_exact_after_packaging(tokenizers, selection):
    baseline, optimized = tokenizers
    rdata = sample()
    original = copy.deepcopy(rdata)
    tokens = baseline.get_tokens_windows(STREAM, [rdata], True)[0]
    mask = np.ones(12, dtype=bool)
    if selection == "some":
        mask[::2] = False
    elif selection == "none":
        mask[:] = False
    old, old_lens = baseline.get_source(STREAM, rdata, tokens, WINDOW, mask)
    new, new_lens = optimized.get_source(STREAM, rdata, tokens, WINDOW, mask)
    left, right = StreamData(0, 1, 1, 12), StreamData(0, 1, 1, 12)
    left.add_source(TRAIN, 0, rdata, old_lens, old, False)
    right.add_source(TRAIN, 0, rdata, new_lens, new, False)
    assert_equal(left.source_tokens_cells, right.source_tokens_cells)
    assert_equal(left.source_tokens_lens, right.source_tokens_lens)
    if isinstance(new, torch.Tensor):
        assert right.source_tokens_cells[0].data_ptr() == new.data_ptr()
    for field in ("coords", "geoinfos", "data", "datetimes"):
        assert_equal(getattr(original, field), getattr(rdata, field))


@pytest.mark.parametrize("selection", ["all", "some", "none"])
def test_target_requested_fields_exact(tokenizers, selection):
    baseline, optimized = tokenizers
    rdata = sample()
    tokens = baseline.get_tokens_windows(STREAM, [rdata], False)[0]
    mask = np.ones(12, dtype=bool)
    if selection == "some":
        mask[::2] = False
    elif selection == "none":
        mask[:] = False
    assert_equal(
        baseline.get_target_coords(STREAM, rdata, tokens, WINDOW, mask),
        optimized.get_target_coords(STREAM, rdata, tokens, WINDOW, mask),
    )
    expected = baseline.get_target_values(STREAM, rdata, tokens, WINDOW, mask)
    with patch.object(
        tokenizer_utils,
        "get_target_coords_local",
        side_effect=AssertionError("values must not build coordinate features"),
    ):
        actual = optimized.get_target_values(STREAM, rdata, tokens, WINDOW, mask)
    assert_equal(expected, actual)


def test_inverse_indices_skipped_only_when_requested_and_enabled(tokenizers):
    baseline, optimized = tokenizers
    rdata = sample()
    tokens = baseline.get_tokens_windows(STREAM, [rdata], False)[0]
    mask = np.ones(12, dtype=bool)
    expected = baseline.get_target_values(STREAM, rdata, tokens, WINDOW, mask)
    # Old configs keep sorting even when the caller identifies training.
    assert_equal(
        expected,
        baseline.get_target_values(STREAM, rdata, tokens, WINDOW, mask, need_inverse_indices=False),
    )
    # Evaluation keeps the inverse permutation with the optimization enabled.
    assert_equal(expected, optimized.get_target_values(STREAM, rdata, tokens, WINDOW, mask))
    with patch("torch.sort", side_effect=AssertionError("training must not sort inverse indices")):
        actual = optimized.get_target_values(
            STREAM, rdata, tokens, WINDOW, mask, need_inverse_indices=False
        )
    assert_equal(expected[:3], actual[:3])
    assert actual[3] is None


@pytest.mark.parametrize("padding", [True, False])
def test_coords_only_spacetime_indices_exact(padding):
    rdata = sample()
    expected = tokenize_spacetime(rdata, 4, 0, pad_tokens=padding)
    actual = tokenize_spacetime(rdata, 4, 0, pad_tokens=padding, coords_only=True)
    assert_equal(expected, actual)


def test_empty_windows_preserved(tokenizers):
    baseline, optimized = tokenizers
    rdata = sample()
    for field in ("coords", "geoinfos", "data", "datetimes"):
        setattr(rdata, field, getattr(rdata, field)[:0])
    for padding in (True, False):
        assert baseline.get_tokens_windows(STREAM, [rdata], padding) == [(None, None)]
        assert optimized.get_tokens_windows(STREAM, [rdata], padding) == [(None, None)]


def test_all_flags_explicitly_off_preserve_legacy_representation(tokenizers):
    baseline, _ = tokenizers
    disabled = TokenizerMasking(0, None, optimizations=dict.fromkeys(FLAGS, False))
    rdata = sample()
    tokens = baseline.get_tokens_windows(STREAM, [rdata], True)[0]
    mask = np.ones(12, dtype=bool)
    expected = baseline.get_source(STREAM, rdata, tokens, WINDOW, mask)
    actual = disabled.get_source(STREAM, rdata, tokens, WINDOW, mask)
    assert isinstance(actual[0], tuple)
    assert_equal(expected, actual)


def larger_sample(num_points=3000, seed=0):
    """Random points over the sphere and two times, with NaN in the data, for chunked builds."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(num_points, 4)).astype(np.float32)
    data[rng.random(data.shape) < 0.05] = np.nan
    times = np.array(["2020-01-01T00", "2020-01-01T03"], dtype="datetime64[ns]")
    return IOReaderData(
        coords=torch.tensor(
            np.stack([rng.uniform(-90, 90, num_points), rng.uniform(-180, 180, num_points)], 1),
            dtype=torch.float32,
        ),
        geoinfos=torch.tensor(rng.normal(size=(num_points, 3)), dtype=torch.float32),
        data=torch.from_numpy(data),
        datetimes=times[rng.integers(0, 2, num_points)],
    )


@pytest.mark.parametrize("source_dtype", [None, torch.bfloat16])
@pytest.mark.parametrize("selection", ["all", "some", "none"])
@pytest.mark.parametrize("chunk_rows", [None, 64])
@pytest.mark.parametrize("make_sample", [sample, larger_sample])
def test_source_build_in_place_exact_after_packaging(
    monkeypatch, source_dtype, selection, chunk_rows, make_sample
):
    # data loader workers run one torch thread; with more, the old build's elementwise math
    # splits rows between threads, and rounding then depends on the thread count
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if chunk_rows is not None:
            monkeypatch.setattr(tokenizer_utils, "BUILD_CHUNK_ROWS", chunk_rows)
            monkeypatch.setattr(stream_data_module, "NAN_CHUNK_ELEMENTS", 96)
        baseline = TokenizerMasking(0, None)
        optimized = TokenizerMasking(
            0,
            None,
            optimizations=FLAGS | {"source_build_in_place": True},
            source_dtype=source_dtype,
        )
        rdata = make_sample()
        original = copy.deepcopy(rdata)
        tokens = baseline.get_tokens_windows(STREAM, [rdata], True)[0]
        mask = np.ones(12, dtype=bool)
        if selection == "some":
            mask[::2] = False
        elif selection == "none":
            mask[:] = False
        old, old_lens = baseline.get_source(STREAM, rdata, tokens, WINDOW, mask)
        new, new_lens = optimized.get_source(STREAM, rdata, tokens, WINDOW, mask)
        left = StreamData(0, 1, 1, 12, source_dtype=source_dtype)
        right = StreamData(0, 1, 1, 12, source_dtype=source_dtype, nan_in_chunks=True)
        left.add_source(TRAIN, 0, rdata, old_lens, old, False)
        right.add_source(TRAIN, 0, rdata, new_lens, new, False)
        assert_equal(left.source_tokens_cells, right.source_tokens_cells)
        assert_equal(left.source_tokens_lens, right.source_tokens_lens)
        if selection != "none":
            # built in the source dtype: StreamData keeps the tokenizer's tensor, no copy
            assert right.source_tokens_cells[0].data_ptr() == new.data_ptr()
        for field in ("coords", "geoinfos", "data", "datetimes"):
            assert_equal(getattr(original, field), getattr(rdata, field))
    finally:
        torch.set_num_threads(threads)
