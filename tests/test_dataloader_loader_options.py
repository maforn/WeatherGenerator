"""CPU checks for optional DataLoader arguments."""

import pytest

from weathergen.datasets.loader_options import loader_options

BASE = {"batch_size": None, "batch_sampler": None, "shuffle": False}


def test_defaults_match_original_arguments():
    assert loader_options({"num_workers": 12}) == {**BASE, "num_workers": 12}


def test_explicit_prefetch_and_persistent_are_forwarded():
    options = loader_options({"num_workers": 4, "prefetch_factor": 1, "persistent_workers": True})
    assert options == {**BASE, "num_workers": 4, "prefetch_factor": 1, "persistent_workers": True}


def test_zero_workers_drops_prefetch():
    assert loader_options({"num_workers": 0, "prefetch_factor": 4}) == {**BASE, "num_workers": 0}


@pytest.mark.parametrize(
    "settings",
    [
        {"num_workers": -1},
        {"num_workers": True},
        {"num_workers": 0, "persistent_workers": True},
        {"num_workers": 2, "persistent_workers": "yes"},
        {"num_workers": 2, "prefetch_factor": 0},
    ],
)
def test_invalid_settings_raise(settings):
    with pytest.raises(ValueError):
        loader_options(settings)
