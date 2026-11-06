"""CPU checks for optional DataLoader arguments and the even worker split."""

import types
from unittest.mock import patch

import pytest

from weathergen.datasets.loader_options import loader_options
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler

BASE = {"batch_size": None, "batch_sampler": None, "shuffle": False}


def test_defaults_match_original_arguments():
    assert loader_options({"num_workers": 12}) == {**BASE, "num_workers": 12}


def test_explicit_prefetch_is_forwarded():
    options = loader_options({"num_workers": 4, "prefetch_factor": 1})
    assert options == {**BASE, "num_workers": 4, "prefetch_factor": 1}


def test_zero_workers_drops_prefetch():
    assert loader_options({"num_workers": 0, "prefetch_factor": 4}) == {**BASE, "num_workers": 0}


@pytest.mark.parametrize(
    "settings",
    [
        {"num_workers": -1},
        {"num_workers": True},
        {"num_workers": 2, "prefetch_factor": 0},
    ],
)
def test_invalid_settings_raise(settings):
    with pytest.raises(ValueError):
        loader_options(settings)


class _StubSampler:
    def __init__(self, length, batch_size, even, ddp_rank):
        self.len = length
        self.batch_size = batch_size
        self.optimizations = {"even_workset_split": even}
        self.spatial_parallel = types.SimpleNamespace(ddp_rank=ddp_rank, ddp_world_size=2)

    def __len__(self):
        return self.len


def _worksets(length, batch_size, num_workers, even, ddp_rank=0):
    sampler = _StubSampler(length, batch_size, even, ddp_rank)
    ranges = []
    for worker_id in range(num_workers):
        info = types.SimpleNamespace(id=worker_id, num_workers=num_workers)
        with patch("torch.utils.data.get_worker_info", return_value=info):
            ranges.append(MultiStreamDataSampler.worker_workset(sampler))
    return ranges


def _batches(ranges, batch_size):
    return [len(range(start, end, batch_size)) for start, end in ranges]


def test_original_split_leaves_remainder_on_last_worker():
    ranges = _worksets(length=100, batch_size=1, num_workers=12, even=False)
    assert _batches(ranges, 1) == [8] * 11 + [12]


@pytest.mark.parametrize("ddp_rank", [0, 1])
def test_even_split_is_contiguous_and_balanced(ddp_rank):
    ranges = _worksets(length=100, batch_size=1, num_workers=12, even=True, ddp_rank=ddp_rank)
    assert _batches(ranges, 1) == [9] * 4 + [8] * 8
    assert ranges[0][0] == ddp_rank * 100 and ranges[-1][1] == (ddp_rank + 1) * 100
    assert all(prev[1] == nxt[0] for prev, nxt in zip(ranges, ranges[1:], strict=False))


def test_even_split_only_emits_complete_batches():
    ranges = _worksets(length=50, batch_size=4, num_workers=5, even=True)
    # 12 complete batches over 5 workers; the 2 leftover samples are not emitted.
    assert _batches(ranges, 4) == [3, 3, 2, 2, 2]
    assert all((end - start) % 4 == 0 for start, end in ranges)
