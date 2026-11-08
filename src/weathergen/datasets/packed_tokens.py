# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PackedTokens:
    """Source tokens without their padding (`source_tokens_unpadded`); `unpad()` gives the
    (tokens, token_size, channels) tensor of tokenize_apply_mask_source.

    rows: the real points of all tokens, (points, channels)
    lens: the number of real points of each token, (tokens,); stays on the CPU
    pad: the row each token is padded with, (tokens, channels)
    """

    rows: Tensor
    lens: Tensor
    pad: Tensor
    token_size: int

    @staticmethod
    def cat(tokens: list["PackedTokens"], device=None) -> "PackedTokens":
        rows, pad = (
            torch.cat([getattr(t, f).to(device=device, non_blocking=True) for t in tokens])
            for f in ("rows", "pad")
        )
        return PackedTokens(rows, torch.cat([t.lens for t in tokens]), pad, tokens[0].token_size)

    def to(self, device=None, dtype=None, non_blocking=False):
        def move(t):
            return t.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return PackedTokens(move(self.rows), self.lens, move(self.pad), self.token_size)

    def pin_memory(self):
        pinned = (self.rows.pin_memory(), self.lens.pin_memory(), self.pad.pin_memory())
        return PackedTokens(*pinned, self.token_size)

    def numel(self):
        return self.rows.numel()

    def __len__(self):
        return len(self.lens)

    @property
    def nbytes(self):
        return self.rows.nbytes + self.lens.nbytes + self.pad.nbytes

    def tensor_split(self, sections: int) -> list["PackedTokens"]:
        """The tokens in the same pieces as Tensor.tensor_split(sections) on the padded tokens."""
        num, rest = divmod(len(self), sections)
        ends = torch.tensor([num + (i < rest) for i in range(sections)]).cumsum(0).tolist()
        row_ends = [0, *self.lens.cumsum(0).tolist()]
        return [
            PackedTokens(
                self.rows[row_ends[a] : row_ends[b]], self.lens[a:b], self.pad[a:b], self.token_size
            )
            for a, b in zip([0, *ends[:-1]], ends, strict=True)
        ]

    def unpad(self) -> Tensor:
        device = self.rows.device
        out = self.pad.unsqueeze(1).repeat(1, self.token_size, 1)
        # position of each real point: its token's first slot plus its place within the token
        lens = self.lens.to(device, non_blocking=True)
        num = self.rows.shape[0]
        token = torch.repeat_interleave(
            torch.arange(len(lens), device=device), lens, output_size=num
        )
        first = torch.cumsum(lens, 0) - lens
        pos = token * self.token_size + torch.arange(num, device=device) - first[token]
        out.flatten(0, 1).index_copy_(0, pos, self.rows)
        return out
