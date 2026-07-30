# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Pluggable input source for the sim-vs-device harness.

The default is real WikiText-2 prefill windows: dense natural text that
populates the KV cache with real values across the prefill slices (unlike the
fixed-prompt, zero-KV ``sample_input`` path), which is what stresses the KV
widening / attention interior we most suspect. Swap in another source by
implementing :class:`DatasetSource`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from transformers import PreTrainedTokenizerBase

from qai_hub_models.datasets.wikitext.wikitext import WikiText
from qai_hub_models.utils.base_dataset import DatasetSplit


class DatasetSource(ABC):
    """Yields (input_ids, attention_mask) prefill windows for the harness."""

    @abstractmethod
    def windows(self, num_windows: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return up to ``num_windows`` (input_ids, attention_mask) pairs.

        Each tensor is shaped ``(1, context_length)`` -- a single prefill
        window that the Generator slices internally.
        """


class WikiTextSource(DatasetSource):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        context_length: int,
        sequence_length: int,
        split: DatasetSplit = DatasetSplit.TEST,
    ) -> None:
        # block_size mirrors the calibration/eval convention (== seq len bucket).
        self._ds = WikiText(
            tokenizer=tokenizer,
            block_size=sequence_length,
            context_length=context_length,
            split=split,
        )

    def windows(self, num_windows: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        n = min(num_windows, len(self._ds))
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(n):
            item = self._ds[i]
            out.append((item["input_ids"], item["attention_mask"]))
        return out


def build_source(
    name: str,
    tokenizer: PreTrainedTokenizerBase,
    context_length: int,
    sequence_length: int,
) -> DatasetSource:
    """Factory so the CLI can select a source by name."""
    if name == "wikitext":
        return WikiTextSource(tokenizer, context_length, sequence_length)
    raise ValueError(
        f"Unknown dataset source '{name}'. Implement a DatasetSource subclass "
        "and register it in build_source()."
    )
