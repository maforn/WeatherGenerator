# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""`data_loading.source_tokens_bf16`: source tokens cast to bf16 in the data loader workers.

The batches are built through StreamData.add_source (NaN masking), ModelBatch and get_tokens_lens
as in the sampler. With the option the tensor EmbeddingEngine.forward passes to the stream
embedding, and its output, must have the same bits as without; the option off changes nothing.
"""

import copy
import importlib.util
import sys
import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.common.config import from_cli_arglist
from weathergen.datasets.batch import ModelBatch, SampleMetaData
from weathergen.datasets.multi_stream_data_sampler import (
    MultiStreamDataSampler,
    cast_source_tokens,
    source_tokens_bf16_enabled,
)
from weathergen.datasets.stream_data import StreamData
from weathergen.datasets.utils import get_tokens_lens
from weathergen.train.utils import TRAIN

BF16 = torch.bfloat16
STREAMS = {"era5": 5, "obs": 3}  # name: channels
CELLS = 12
TOKEN = 4
D = 16


@pytest.fixture
def embedding_engine_cls():
    """EmbeddingEngine. engines.py imports flash_attn, which this path does not call: where it is
    not installed (CPU-only machines) a stub stands in during the test, and the modules imported
    with it are unloaded afterwards, so that other tests still see no flash_attn."""
    stubbed = importlib.util.find_spec("flash_attn") is None
    loaded = set(sys.modules)
    if stubbed:
        stub = types.ModuleType("flash_attn")
        stub.flash_attn_func = stub.flash_attn_varlen_func = None
        sys.modules["flash_attn"] = stub
    try:
        from weathergen.model.engines import EmbeddingEngine

        yield EmbeddingEngine
    finally:
        new = set(sys.modules) - loaded if stubbed else set()
        for name in [n for n in new if n == "flash_attn" or n.startswith("weathergen.")]:
            del sys.modules[name]
            parent, _, child = name.rpartition(".")
            if parent in sys.modules:
                vars(sys.modules[parent]).pop(child, None)


def _same_bits(a, b):
    ints = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return a.dtype == b.dtype and torch.equal(a.view(ints), b.view(ints))


def _tokens(gen, num_tokens, num_channels):
    """fp32 tokens over a wide range, with bf16 rounding ties, signed zeros, subnormals and
    NaNs."""
    x = torch.randn((num_tokens, TOKEN, num_channels), generator=gen)
    x *= torch.logspace(-3, 3, num_channels)
    special = torch.tensor([1.0 + 2**-8, 1.0 + 3 * 2**-8, -0.0, 1e-40, -3e38, float("nan")])
    idx = torch.randperm(x.numel(), generator=gen)[: 2 * len(special)]
    x.view(-1)[idx] = special.repeat(2)
    return x


def _stream_data(gen, num_channels, input_steps):
    stream_data = StreamData(idx=0, input_steps=input_steps, output_steps=1, healpix_cells=CELLS)
    for step in range(input_steps):
        lens = torch.randint(0, 4, (CELLS,), dtype=torch.int32, generator=gen)
        lens[0] += 1
        tokens = _tokens(gen, int(lens.sum()), num_channels)
        stream_data.add_source(TRAIN, step, None, lens, list(tokens), False)
    return stream_data


def _batch(seed, input_steps=1):
    """Two student views and one teacher view with network input, all streams present."""
    gen = torch.Generator().manual_seed(seed)
    batch = ModelBatch(list(STREAMS), 2, 1, 0, 1)
    batch.set_source_target_matching(np.array([0, 0]))
    for name, channels in STREAMS.items():
        for sidx in range(2):
            sdata = _stream_data(gen, channels, input_steps)
            batch.add_source_stream(sidx, name, sdata, SampleMetaData(params={}))
        sdata = _stream_data(gen, channels, input_steps)
        batch.add_target_stream(0, name, sdata, SampleMetaData(params={}))
    for samples in (batch.source_samples, batch.target_samples):
        samples.set_tokens_lens(get_tokens_lens(list(STREAMS), samples, input_steps))
    return batch


def _all_source_tokens(samples):
    return [
        t for s in samples.samples for sd in s.streams_data.values() for t in sd.source_tokens_cells
    ]


def _cf(**data_loading):
    return OmegaConf.create(
        {
            "with_mixed_precision": True,
            "mixed_precision_dtype": "bf16",
            "ae_local_dim_embed": D,
            "streams": {n: {"token_size": TOKEN, "embed": {"net": "linear"}} for n in STREAMS},
            "data_loading": data_loading,
        }
    )


def _embed(engine, batch, pe_embed):
    """Run EmbeddingEngine.forward under bf16 autocast, as in training; return the input of
    each stream embedding and the output."""
    seen = {}
    hooks = [
        embed.register_forward_pre_hook(lambda _, args, n=name: seen.__setitem__(n, args[0]))
        for name, embed in engine.embeds.items()
    ]
    with torch.autocast("cpu", dtype=BF16):
        out = engine(batch.get_source_samples(), pe_embed)
    for hook in hooks:
        hook.remove()
    return seen, out


@pytest.mark.parametrize("input_steps", [1, 2])
def test_embedding_input_and_output_are_bit_identical(embedding_engine_cls, input_steps):
    batch_off = _batch(seed=input_steps, input_steps=input_steps)
    batch_on = copy.deepcopy(batch_off)
    cast_source_tokens(batch_on, BF16)
    assert {t.dtype for t in _all_source_tokens(batch_off.source_samples)} == {torch.float32}
    assert {t.dtype for t in _all_source_tokens(batch_on.source_samples)} == {BF16}

    torch.manual_seed(0)
    engine = embedding_engine_cls(_cf(), list(STREAMS.values()))
    pe_embed = torch.randn(64, D)
    seen_off, out_off = _embed(engine, batch_off, pe_embed)
    seen_on, out_on = _embed(engine, batch_on, pe_embed)

    assert seen_off.keys() == seen_on.keys() == STREAMS.keys()
    for name in STREAMS:
        assert seen_on[name].dtype == BF16
        assert _same_bits(seen_off[name], seen_on[name])
    assert _same_bits(out_off, out_on)


def test_cast_converts_the_student_views_only():
    batch = _batch(seed=3)
    before = copy.deepcopy(batch)
    cast_source_tokens(batch, BF16)

    student = _all_source_tokens(batch.source_samples), _all_source_tokens(before.source_samples)
    for now, fp32 in zip(*student, strict=True):
        assert now.dtype == BF16 and 2 * now.nbytes == fp32.nbytes
        assert not now.isnan().any(), "NaNs are masked to 0 before the cast"
        assert _same_bits(now, fp32.to(BF16))
    # teacher views: a FrozenTeacher's embedding has the dtype of its own run config
    teacher = _all_source_tokens(batch.target_samples), _all_source_tokens(before.target_samples)
    for now, fp32 in zip(*teacher, strict=True):
        assert _same_bits(now, fp32)


def test_cast_keeps_missing_steps_and_streams():
    batch = ModelBatch(["a", "b"], 1, 1, 0, 1)
    stream_data = StreamData(idx=0, input_steps=2, output_steps=1, healpix_cells=CELLS)
    lens = torch.ones(CELLS, dtype=torch.int32)
    stream_data.add_source(TRAIN, 0, None, lens, list(torch.randn(CELLS, TOKEN, 2)), False)
    batch.add_source_stream(0, "a", stream_data, SampleMetaData(params={}))  # no "b"

    cast_source_tokens(batch, BF16)

    assert batch.source_samples.samples[0].streams_data["b"] is None
    assert stream_data.source_tokens_cells[0].dtype == BF16
    assert stream_data.source_tokens_cells[1] is None


@pytest.mark.parametrize(
    ("data_loading", "with_mixed_precision", "mixed_precision_dtype", "enabled"),
    [
        ({}, True, "bf16", False),  # configs without the option
        ({"source_tokens_bf16": False}, True, "bf16", False),
        ({"source_tokens_bf16": True}, True, "bf16", True),
        ({"source_tokens_bf16": True}, True, "fp16", False),
        ({"source_tokens_bf16": True}, True, "fp32", False),
        ({"source_tokens_bf16": True}, False, "bf16", False),
    ],
)
def test_option_is_off_by_default_and_needs_bf16_mixed_precision(
    data_loading, with_mixed_precision, mixed_precision_dtype, enabled
):
    cf = _cf(**data_loading)
    cf.with_mixed_precision = with_mixed_precision
    cf.mixed_precision_dtype = mixed_precision_dtype
    assert source_tokens_bf16_enabled(cf) is enabled


def test_option_from_the_command_line():
    cli = from_cli_arglist(["data_loading.source_tokens_bf16=True"])
    assert source_tokens_bf16_enabled(OmegaConf.merge(_cf(), cli))


@pytest.mark.parametrize("enabled", [False, True])
def test_sampler_casts_right_before_the_yield_only_when_enabled(enabled):
    batch = _batch(seed=4)
    fp32 = copy.deepcopy(batch)
    sampler = MultiStreamDataSampler.__new__(MultiStreamDataSampler)
    sampler.source_tokens_bf16 = enabled
    sampler.mode_cfg = {"training_mode": ["masking"], "with_targets": False}
    sampler.len, sampler.batch_size = 1, 1
    sampler.worker_workset = lambda: (0, 1)
    sampler.reset = lambda: (np.array([0]), np.array([0]))
    sampler._get_batch = lambda idx, num_forecast_steps: batch

    (yielded,) = list(sampler)

    assert yielded is batch
    tokens = _all_source_tokens(yielded.source_samples), _all_source_tokens(fp32.source_samples)
    for now, before in zip(*tokens, strict=True):
        assert _same_bits(now, before.to(BF16) if enabled else before)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
        ),
    ],
)
def test_cast_rounds_to_nearest_even(device):
    """The worker casts on the CPU, the embedding casts on the GPU: both must round to nearest
    even. Random fp32 bit patterns cover all exponents, subnormals and infinities."""
    gen = torch.Generator().manual_seed(0)
    bits = torch.randint(-(2**31), 2**31, (1 << 20,), dtype=torch.int64, generator=gen)
    x = bits.to(torch.int32).view(torch.float32)
    x = torch.where(x.isnan(), torch.zeros_like(x), x)  # NaNs are 0 before the cast

    got = x.to(device).to(BF16).cpu().view(torch.int16).to(torch.int64) & 0xFFFF
    u = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    assert torch.equal(got, (u + 0x7FFF + ((u >> 16) & 1)) >> 16)
