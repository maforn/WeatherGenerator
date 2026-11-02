# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for converting the source tokens in the data loader workers
(``source_tokens_in_mixed_precision``).

The model input must be the same as when the tokens are converted on the GPU: the embedding
casts them to the mixed-precision dtype first. Runs on CPU with small synthetic tokens.
"""

import pytest
import torch
from omegaconf import OmegaConf

from weathergen.datasets.multi_stream_data_sampler import get_source_tokens_dtype
from weathergen.datasets.stream_data import StreamData
from weathergen.train.utils import TRAIN

NUM_CELLS = 12  # HEALPix level 0
TOKEN_SIZE = 8
NUM_CHANNELS = 7


def _tokens(seed: int = 0, num_tokens: int = 20) -> list[torch.Tensor]:
    """One fp32 tensor (token size, channels) per token, as the tokenizer returns them.

    Values span several orders of magnitude, so that rounding to bf16 is not trivial, and
    some are NaN, which add_source replaces by the mask value.
    """
    g = torch.Generator().manual_seed(seed)
    tokens = []
    for _ in range(num_tokens):
        t = torch.randn(TOKEN_SIZE, NUM_CHANNELS, generator=g)
        t = t * 10.0 ** torch.randint(-3, 4, (TOKEN_SIZE, NUM_CHANNELS), generator=g)
        t[torch.rand(t.shape, generator=g) < 0.1] = float("nan")
        tokens.append(t)
    return tokens


def _stream_data(tokens: list[torch.Tensor], source_dtype: torch.dtype | None) -> StreamData:
    sd = StreamData(0, 1, 1, NUM_CELLS, source_dtype=source_dtype)
    lens = torch.zeros(NUM_CELLS, dtype=torch.int32)
    lens[0] = len(tokens)
    sd.add_source(TRAIN, 0, None, lens, [t.clone() for t in tokens], False)
    return sd


def _masked_fp32(tokens: list[torch.Tensor]) -> torch.Tensor:
    ref = torch.stack(tokens)
    return torch.where(torch.isnan(ref), torch.zeros_like(ref), ref)


# ---------------------------------------------------------------------------
# StreamData.add_source
# ---------------------------------------------------------------------------


def test_default_keeps_tokenizer_dtype():
    tokens = _tokens()
    sd = _stream_data(tokens, None)
    out = sd.source_tokens_cells[0]
    assert out.dtype == torch.float32
    assert torch.equal(out, _masked_fp32(tokens))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_converted_after_nan_masking(dtype):
    tokens = _tokens()
    out = _stream_data(tokens, dtype).source_tokens_cells[0]
    assert out.dtype == dtype
    assert not torch.isnan(out).any()
    assert torch.equal(out, _masked_fp32(tokens).to(dtype))


def test_half_the_bytes():
    tokens = _tokens()
    fp32 = _stream_data(tokens, None).source_tokens_cells[0]
    bf16 = _stream_data(tokens, torch.bfloat16).source_tokens_cells[0]
    assert bf16.nbytes * 2 == fp32.nbytes


def test_empty_source_step():
    # tokenize_apply_mask_source returns [torch.tensor([])] for a stream without data
    sd = StreamData(0, 1, 1, NUM_CELLS, source_dtype=torch.bfloat16)
    lens = torch.zeros(NUM_CELLS, dtype=torch.int32)
    sd.add_source(TRAIN, 0, None, lens, [torch.tensor([])], False)
    assert sd.source_tokens_cells[0].dtype == torch.bfloat16
    assert sd.source_tokens_cells[0].numel() == 0
    assert sd.source_empty()


# ---------------------------------------------------------------------------
# Same model input as converting on the device (the previous behaviour)
# ---------------------------------------------------------------------------


def _embedding_input(samples: list[StreamData]) -> torch.Tensor:
    """What EmbeddingEngine.forward builds from one stream: cat over samples, then cast."""
    return torch.cat([s.source_tokens_cells[0] for s in samples]).to(torch.bfloat16)


def test_same_model_input_as_converting_on_device():
    tokens = [_tokens(seed) for seed in range(3)]
    device = "cpu"
    # previous behaviour: fp32 from the worker, converted in to_device
    old = [_stream_data(t, None).to_device(device, torch.bfloat16) for t in tokens]
    # new behaviour: bf16 from the worker; the trainer still passes the dtype
    new = [_stream_data(t, torch.bfloat16).to_device(device, torch.bfloat16) for t in tokens]
    for o, n in zip(old, new, strict=True):
        assert n.source_tokens_cells[0].dtype == torch.bfloat16
        assert torch.equal(o.source_tokens_cells[0], n.source_tokens_cells[0])
    assert torch.equal(_embedding_input(old), _embedding_input(new))


def test_same_model_input_without_cast_in_to_device():
    # validation calls to_device without a dtype; the embedding input must not change either
    tokens = _tokens()
    old = _stream_data(tokens, None).to_device("cpu")
    new = _stream_data(tokens, torch.bfloat16).to_device("cpu")
    assert torch.equal(_embedding_input([old]), _embedding_input([new]))


# ---------------------------------------------------------------------------
# Hand-off from a data loader worker to the main process
# ---------------------------------------------------------------------------


class _StreamDataset(torch.utils.data.IterableDataset):
    """Builds StreamData inside the worker, like MultiStreamDataSampler."""

    def __init__(self, source_dtype):
        super().__init__()
        self.source_dtype = source_dtype

    def __iter__(self):
        for seed in range(3):
            yield _stream_data(_tokens(seed), self.source_dtype)


@pytest.mark.parametrize("source_dtype", [None, torch.bfloat16])
def test_worker_handoff(source_dtype):
    # batch_size=None and a worker process, as in the trainer: tensors cross through shared
    # memory and must arrive with the dtype and values they had in the worker
    loader = torch.utils.data.DataLoader(
        _StreamDataset(source_dtype), batch_size=None, num_workers=1
    )
    received = list(loader)
    assert len(received) == 3
    for seed, sd in enumerate(received):
        expected = _masked_fp32(_tokens(seed))
        if source_dtype is not None:
            expected = expected.to(source_dtype)
        assert sd.source_tokens_cells[0].dtype == expected.dtype
        assert torch.equal(sd.source_tokens_cells[0], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinning needs CUDA")
def test_pin_memory_keeps_dtype():
    sd = _stream_data(_tokens(), torch.bfloat16).pin_memory()
    assert sd.source_tokens_cells[0].is_pinned()
    assert sd.source_tokens_cells[0].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mixed_precision_dtype, switch, expected",
    [
        ("bf16", None, None),  # switch not in the config
        ("bf16", False, None),
        ("bf16", True, torch.bfloat16),
        ("fp16", True, torch.float16),
    ],
)
def test_get_source_tokens_dtype(mixed_precision_dtype, switch, expected):
    cfg = {"mixed_precision_dtype": mixed_precision_dtype}
    if switch is not None:
        cfg["source_tokens_in_mixed_precision"] = switch
    assert get_source_tokens_dtype(OmegaConf.create(cfg)) == expected
