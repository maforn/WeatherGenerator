"""`data_loading.release_yielded_batch`: the sampler drops the batch it yielded before it builds
the next one, so a DataLoader worker does not keep the previous batch's shared memory mapped."""

import gc
import weakref

import numpy as np
import pytest

from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler


class FakeBatch:
    """Stands in for ModelBatch: valid, and weak-referenceable."""

    def sources_empty(self):
        return False

    def is_nan(self):
        return False

    def targets_empty(self):
        return False


def make_sampler(release: bool, alive_during_build: list):
    """A sampler with only what `__iter__` needs; `_get_batch` records whether the previously
    yielded batch is still alive when the next one is built."""
    sampler = object.__new__(MultiStreamDataSampler)
    sampler.len = 3
    sampler.batch_size = 1
    sampler.mode_cfg = {"training_mode": ["masking"], "with_targets": True}
    sampler.source_tokens_bf16 = False
    sampler.release_yielded_batch = release
    sampler.worker_workset = lambda: (0, 3)
    sampler.reset = lambda: (np.arange(3), np.zeros(3, dtype=int))
    last = {"ref": None}

    def get_batch(idx, num_forecast_steps):
        gc.collect()
        if last["ref"] is not None:
            alive_during_build.append(last["ref"]() is not None)
        batch = FakeBatch()
        last["ref"] = weakref.ref(batch)
        return batch

    sampler._get_batch = get_batch
    return sampler


@pytest.mark.parametrize("release", [False, True])
def test_previous_batch_alive_while_next_is_built(release):
    alive = []
    sampler = make_sampler(release, alive)
    # consume like the DataLoader worker loop: take the batch, hand it off, drop it
    count = 0
    for batch in sampler:
        count += 1
        del batch
    assert count == 3
    assert len(alive) == 2
    # off: the generator frame still holds the previous batch; on: nothing does
    assert alive == [not release] * 2


def test_same_batches_with_and_without_release():
    seen = {}
    for release in (False, True):
        sampler = make_sampler(release, [])
        ids = []
        orig = sampler._get_batch

        def get_batch(idx, n, orig=orig, ids=ids):
            ids.append((int(idx), int(n)))
            return orig(idx, n)

        sampler._get_batch = get_batch
        list(sampler)
        seen[release] = ids
    assert seen[False] == seen[True]
