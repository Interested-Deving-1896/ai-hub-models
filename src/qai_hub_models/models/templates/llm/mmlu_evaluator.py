# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import textwrap

import torch
from torch.nn import functional as F
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from qai_hub_models.models.templates.llm.llm_evaluator import LLMEvaluator
from qai_hub_models.utils.metrics import (
    MMLU,
    MetricMetadata,
)


class MMLUEvaluator(LLMEvaluator):
    """Evaluator for computing MMLU of a Large Language Model.
    This may not be as generic as hoped and may need work. Works with Llama 3.2 3B.
    """

    def __init__(
        self,
        context_length: int,
        device: torch.device,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.context_length = context_length
        self.device = device
        self.choices = self._get_choices(tokenizer)
        self.tokenizer = tokenizer
        self.reset()

    @staticmethod
    def _get_choices(
        tokenizer: PreTrainedTokenizerBase,
    ) -> torch.Tensor:
        def tokenize_letter(letter: str) -> torch.Tensor:
            return tokenizer(letter, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ][0, -1:]

        return torch.cat(
            [
                tokenize_letter("Answer: A"),
                tokenize_letter("Answer: B"),
                tokenize_letter("Answer: C"),
                tokenize_letter("Answer: D"),
            ],
            dim=-1,
        )

    def add_batch(
        self,
        output: CausalLMOutputWithPast,
        gt: torch.Tensor,
    ) -> None:
        self.batch_index += 1
        assert output.logits is not None
        logits = output.logits[0]

        choices = self.choices.to(logits.device)
        gt = gt.to(logits.device)

        top_token_id = logits[-1].argmax()
        answers = logits[:, choices]
        index = answers[-1].argmax()
        prediction = choices[index]
        self.top_is_valid += int(top_token_id in choices)

        correct = prediction == gt
        self.correct_answers += int(correct)

        logsoft_q = F.log_softmax(logits, dim=-1)
        self.neg_log_likelihood += float(-logsoft_q[-1, gt])

    def reset(self) -> None:
        self.correct_answers = 0
        self.neg_log_likelihood = 0.0
        self.top_is_valid = 0
        self.batch_index = 0

    def get_accuracy_score(self) -> float:
        return self.correct_answers / self.batch_index

    def get_avg_neg_log_likelihood(self) -> float:
        return self.neg_log_likelihood / self.batch_index

    def get_avg_valid_answers(self) -> float:
        return self.top_is_valid / self.batch_index

    def formatted_accuracy(self) -> str:
        return textwrap.dedent(
            f"""
                MMLU: {self.get_accuracy_score():.2%} (higher is better)
                Top prediction is valid answer: {self.get_avg_valid_answers():.1%}
                Avg NLL: {self.get_avg_neg_log_likelihood():.3f} (lower is better)
            """
        ).lstrip()

    def get_metric_metadata(self) -> MetricMetadata:
        return MMLU
