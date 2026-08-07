# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
r"""
Chat-formatted WikiText for instruction-tuned (``-it``) LLMs.

Rebuilds the WikiText corpus as user/assistant conversations rendered through
``tokenizer.apply_chat_template``, so calibration ranges cover the control
tokens an ``-it`` model sees at inference.
"""

from __future__ import annotations

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from qai_hub_models.datasets.wikitext.wikitext import WikiText
from qai_hub_models.utils.base_dataset import DatasetMetadata, DatasetSplit


def _parse_header(line: str) -> str | None:
    """Return the title of a wikitext ``= Section =`` header, else ``None``."""
    s = line.strip()
    if len(s) < 3 or not s.startswith("=") or not s.endswith("="):
        return None

    def run_len(text: str) -> int:
        """Count the leading ``=`` markers, allowing single spaces between."""
        n = i = 0
        while i < len(text):
            if text[i] == "=":
                n += 1
                i += 1
            elif text[i] == " " and i + 1 < len(text) and text[i + 1] == "=":
                i += 1
            else:
                break
        return n

    lead = run_len(s)
    if lead != run_len(s[::-1]):
        return None
    body = s
    for _ in range(lead):
        body = body.lstrip().lstrip("=")
        body = body.rstrip().rstrip("=")
    return body.strip() or None


# Instruction phrasings cycled over section headers for variety.
_INSTRUCTIONS = (
    "Tell me about {topic}.",
    "What can you tell me about {topic}?",
    "Write a short summary of {topic}.",
    "Explain {topic}.",
    "Give me some background on {topic}.",
    "What is {topic}?",
)

# Cap on the prose used for one model turn.
_MAX_ANSWER_CHARS = 1200

# Minimum prose length for a usable section.
_MIN_ANSWER_CHARS = 80


def build_chat_corpus(
    tokenizer: PreTrainedTokenizerBase,
    raw_dataset: Dataset,
) -> str:
    """Render WikiText sections as chat-templated user/model conversations.

    Parameters
    ----------
    tokenizer
        Tokenizer whose ``chat_template`` defines the turn markup.
    raw_dataset
        A wikitext-2-raw split (one line per ``text`` entry).

    Returns
    -------
    str
        Concatenated conversations, each already prefixed with the template's
        ``bos`` token.
    """
    conversations: list[str] = []
    topic: str | None = None
    prose: list[str] = []

    def flush() -> None:
        if topic is None:
            return
        answer = " ".join(prose).strip()
        if len(answer) < _MIN_ANSWER_CHARS:
            return
        instruction = _INSTRUCTIONS[len(conversations) % len(_INSTRUCTIONS)]
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": instruction.format(topic=topic)},
                {"role": "assistant", "content": answer[:_MAX_ANSWER_CHARS]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        assert isinstance(rendered, str)
        conversations.append(rendered)

    for line in raw_dataset["text"]:
        header = _parse_header(line)
        if header is not None:
            flush()
            topic = header
            prose = []
        elif line.strip():
            prose.append(line.strip())
    flush()

    if not conversations:
        raise ValueError(
            "No chat conversations could be built from the WikiText split; the "
            "section-header format may have changed."
        )
    return "".join(conversations)


class WikiTextChat(WikiText):
    """WikiText rendered through the model's chat template.

    Drop-in replacement for ``WikiText``: identical chunking, ``__len__`` and
    ``__getitem__``, differing only in the token stream being chat-formatted.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int = 128,
        context_length: int = 4096,
        split: DatasetSplit = DatasetSplit.TEST,
        num_samples: int = 0,
    ) -> None:
        if tokenizer.chat_template is None:
            raise ValueError(
                f"{type(self).__name__} requires a tokenizer with a "
                f"chat_template; use WikiText for base (non-instruct) models."
            )
        self.block_size = block_size
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.num_samples = num_samples

        if split == DatasetSplit.TEST:
            self.split_str = "test"
        elif split == DatasetSplit.TRAIN:
            self.split_str = "train"
        else:
            raise ValueError(
                "WikiTextChat currently only supports `test` and `train` split"
            )

        corpus = build_chat_corpus(tokenizer, self.load_raw_dataset())
        # The chat template already emits bos per conversation, so the corpus is
        # self-delimiting; don't add another.
        self.tokens = self.tokenizer(
            corpus,
            return_tensors="pt",
            add_special_tokens=False,
        )

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://huggingface.co/datasets/Salesforce/wikitext",
            split_description="test split, rendered as chat-templated turns",
        )

    @classmethod
    def dataset_name(cls) -> str:
        # Must contain "wikitext" -- get_evaluator() selects PerplexityEvaluator
        # by substring match on the task name.
        return "wikitext_chat"


def chat_control_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Return turn-delimiter token ids the chat template adds (e.g. [105, 106]
    for gemma-4). Used by calibration pre-flight checks.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(rendered, str)
    templated = set(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    plain = set(tokenizer("hi", add_special_tokens=True)["input_ids"])
    excluded = {tokenizer.bos_token_id, tokenizer.eos_token_id}
    special = set(tokenizer.all_special_ids) - excluded
    return sorted((templated - plain) & special)


def count_token_ids(ids: torch.Tensor, wanted: list[int]) -> dict[int, int]:
    """Count occurrences of ``wanted`` ids in ``ids`` (flattened)."""
    flat = ids.flatten()
    return {t: int((flat == t).sum()) for t in wanted}


__all__ = [
    "WikiTextChat",
    "build_chat_corpus",
    "chat_control_token_ids",
    "count_token_ids",
]
