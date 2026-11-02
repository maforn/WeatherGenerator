"""optimizations.source_tokens_bf16: loader-side cast must give the model the same bf16 input."""

from types import SimpleNamespace

import torch

from weathergen.datasets.multi_stream_data_sampler import cast_source_tokens


def make_batch(tensors):
    stream = SimpleNamespace(source_tokens_cells=list(tensors))
    sample = SimpleNamespace(streams_data={"ERA5": stream, "empty": None})
    return SimpleNamespace(source_samples=SimpleNamespace(samples=[sample])), stream


def test_cast_matches_model_side_cast_bit_for_bit():
    torch.manual_seed(0)
    # wide dynamic range, including values that round differently in bf16
    fp32 = torch.randn(64, 12, 7) * torch.logspace(-6, 6, 7)
    batch, stream = make_batch([fp32.clone(), None])
    cast_source_tokens(batch, torch.bfloat16)
    loader_side = stream.source_tokens_cells[0]
    assert loader_side.dtype == torch.bfloat16
    assert stream.source_tokens_cells[1] is None
    # EmbeddingEngine.forward: torch.cat(sdata).to(tokens_all.dtype) with dtype bf16
    model_side_old = torch.cat([fp32]).to(torch.bfloat16)
    model_side_new = torch.cat([loader_side]).to(torch.bfloat16)
    assert torch.equal(model_side_old.view(torch.int16), model_side_new.view(torch.int16))
    assert loader_side.element_size() * 2 == fp32.element_size()


def test_non_float_tensors_untouched():
    ints = torch.arange(10)
    batch, stream = make_batch([ints])
    cast_source_tokens(batch, torch.bfloat16)
    assert stream.source_tokens_cells[0] is ints
