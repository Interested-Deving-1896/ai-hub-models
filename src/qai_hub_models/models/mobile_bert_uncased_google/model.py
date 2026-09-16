# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from qai_hub.client import Device
from torch import nn
from transformers import AutoModelForMaskedLM, MobileBertTokenizer
from transformers.models.mobilebert.modeling_mobilebert import NoNorm

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.templates.bert_hf.maskedlm_evaluator import MaskedLMEvaluator
from qai_hub_models.models.templates.bert_hf.model import BaseBertModel
from qai_hub_models.models.templates.bert_hf.model_patches import (
    patch_get_extended_attention_mask,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import OutputSpec, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
WEIGHTS_NAME = "google/mobilebert-uncased"

# MobileBERT uses NoNorm (a plain affine) rather than LayerNorm, so nothing bounds the
# residual: token 0 reaches ~1e8 and sets the per-tensor activation range for everyone.
NONORM_CLAMP = 1000.0


class _ClampedNoNorm(nn.Module):
    """NoNorm plus an output clamp, so the bound lands in the exported graph."""

    def __init__(self, inner: NoNorm, bound: float) -> None:
        super().__init__()
        self.inner = inner
        self.bound = bound

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x).clamp(-self.bound, self.bound)


def _clamp_nonorms(module: nn.Module, bound: float) -> int:
    """Replace every NoNorm under ``module`` with a clamped one; returns how many."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, NoNorm):
            setattr(module, name, _ClampedNoNorm(child, bound))
            replaced += 1
        else:
            replaced += _clamp_nonorms(child, bound)
    return replaced


class _LogitsMaskedLMEvaluator(MaskedLMEvaluator):
    """MaskedLMEvaluator for a model that outputs logits rather than a token id."""

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor) -> None:
        super().add_batch(output.argmax(dim=-1), gt)


class MobileBertUncasedGoogle(BaseBertModel):
    """Exportable HuggingFace Distillbert Model"""

    @staticmethod
    def default_weights() -> str:
        return WEIGHTS_NAME

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_masks: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass returning vocabulary logits at the mask position.

        Parameters
        ----------
        input_tokens
            Input tokens of shape [batch_size, seq_len].
        attention_masks
            Attention masks of shape [batch_size, seq_len].
        mask_indices
            Tensor of shape [batch_size] with index of [MASK] per sample.

        Returns
        -------
        masked_logits : torch.Tensor
            Vocabulary logits at the mask position, shape [batch_size, vocab_size].
        """
        # Logits, not an argmax token id: PSNR is undefined over vocabulary indices, so
        # mixed precision would have no layer-ranking signal. Argmax is in the evaluator.
        logits = self.model(input_tokens, attention_mask=attention_masks).logits
        batch_indices = torch.arange(input_tokens.shape[0])
        return logits[batch_indices, mask_indices.to(torch.int64)]

    def get_output_spec(self) -> OutputSpec:
        return {
            "logits": TensorSpec(),
        }

    def get_evaluator(self) -> BaseEvaluator:
        return _LogitsMaskedLMEvaluator()

    def get_hub_litemp_percentage(self, precision: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 1

    @classmethod
    def from_pretrained(cls, weights: str = WEIGHTS_NAME) -> MobileBertUncasedGoogle:
        """Load HuggingFace Bert Model for Embeddings."""
        model = AutoModelForMaskedLM.from_pretrained(weights)
        tokenizer = MobileBertTokenizer.from_pretrained(weights)
        model.mobilebert.get_extended_attention_mask = patch_get_extended_attention_mask
        assert _clamp_nonorms(model, NONORM_CLAMP) > 0, (
            "No NoNorm modules were clamped."
        )
        return cls(model, tokenizer)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        # -O2 works around severe 2.50 numerical regressions (tetracode #21273).
        if target_runtime == TargetRuntime.QNN_DLC:
            other_compile_options += " -O2"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )
