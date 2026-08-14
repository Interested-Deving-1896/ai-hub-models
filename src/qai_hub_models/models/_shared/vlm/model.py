# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CALIBRATION_SEQ_LEN,
    DEFAULT_CONTEXT_LENGTH,
    LLMDynamic_AIMETOnnx,
)

if TYPE_CHECKING:
    from qai_hub.public_rest_api import DatasetEntries

    from qai_hub_models.models._shared.lm_schema import DatasetSpec


logger = logging.getLogger(__name__)

# Fallback default calibration/export image size (width, height), matching the
# dataset/eval convention (PIL Image.resize takes (width, height)). Individual
# models author their own default image dimensions and pass them down; this is
# the generic default for callers that don't.
DEFAULT_IMAGE_SIZE: tuple[int, int] = (512, 512)


class VLMDynamic_AIMETOnnx(LLMDynamic_AIMETOnnx):
    """Dynamic-shape AIMET-ONNX base for vision-language models.

    Owns the model-agnostic calibration data pipeline. Subclasses provide
    ``get_input_spec`` (which must accept ``image_size`` and derive its
    visual-token count from it) so the prefill data matches the exported graph.
    """

    def _load_calibration_vision_model(self) -> torch.nn.Module | None:
        """Load the HF vision model for multimodal calibration samples."""
        try:
            from transformers import AutoModel

            hf_repo = getattr(self, "_hf_repo_name", None)
            if hf_repo is None and self.checkpoint is not None:
                hf_repo = self.checkpoint
            if hf_repo is None:
                hf_repo = self.llm_config._name_or_path

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            hf_model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
            visual = hf_model.visual.eval().to(device)
            del hf_model
            return visual
        except Exception:
            logger.warning(
                "Failed to load vision model for calibration; "
                "multimodal samples will use text-only prefill.",
                exc_info=True,
            )
            return None

    def prefill_dataset(
        self,
        dataset_spec: DatasetSpec,
        num_samples: int = 0,
        sequence_length: int = DEFAULT_CALIBRATION_SEQ_LEN,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        image_size: tuple[int, int] | None = DEFAULT_IMAGE_SIZE,
    ) -> DatasetEntries | None:
        """Resolve a dataset spec and prefill it, engaging vision iff the spec is
        image-bearing (VLM processor + vision model + multimodal kwargs); text-only
        specs (e.g. WikiText weight-opt) take the plain text path.
        """
        from torch.utils.data import DataLoader

        from qai_hub_models.datasets import instantiate_dataset
        from qai_hub_models.models._shared.llm.generator_factory import make_generator
        from qai_hub_models.models._shared.llm.quantize import (
            resolve_dataset_cls,
            spec_is_multimodal,
        )
        from qai_hub_models.utils.base_dataset import DatasetSplit
        from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

        if num_samples == 0:
            num_samples = math.ceil(80000 / context_length)

        dataset_cls = resolve_dataset_cls(dataset_spec, self.extra_interleaved_datasets)
        multimodal = spec_is_multimodal(dataset_spec)

        # Image datasets need the VLM processor (not just the tokenizer).
        dataset_kwargs: dict = dict(
            tokenizer=self.tokenizer,
            block_size=sequence_length,
            context_length=context_length,
            num_samples=num_samples,
        )
        if multimodal:
            from transformers import AutoProcessor

            hf_repo = getattr(self, "_hf_repo_name", None)
            if hf_repo is None and self.checkpoint is not None:
                hf_repo = self.checkpoint
            if hf_repo is None:
                hf_repo = self.llm_config._name_or_path
            dataset_kwargs["processor"] = AutoProcessor.from_pretrained(
                hf_repo, trust_remote_code=True
            )
            dataset_kwargs["image_size"] = image_size

        dataset = instantiate_dataset(
            dataset_cls, DatasetSplit.TRAIN, input_spec=None, **dataset_kwargs
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

        input_spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=self.llm_io_type,
            image_size=image_size,  # type: ignore[call-arg]
        )
        assert input_spec is not None

        # Vision model only for image data; text-only prefill skips it.
        vision_model = self._load_calibration_vision_model() if multimodal else None
        generator = make_generator(
            self,
            sequence_length=sequence_length,
            context_length=context_length,
            vision_model=vision_model,
            model_cls=self.FPModel,
        )

        def sample_to_kwargs(
            sample: tuple[torch.Tensor, ...], device: torch.device
        ) -> dict[str, torch.Tensor | None]:
            input_ids, attention_mask, *rest = sample
            kwargs: dict[str, torch.Tensor | None] = dict(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            if multimodal:
                pixel_values = rest[1] if len(rest) > 1 else None
                kwargs["pixel_values"] = (
                    pixel_values.to(device) if pixel_values is not None else None
                )
                image_grid_thw = rest[2] if len(rest) > 2 else None
                kwargs["image_grid_thw"] = (
                    image_grid_thw.to(device) if image_grid_thw is not None else None
                )
            return kwargs

        desc = (
            "Pre-filling calibration data (Interleaved WikiText/AOKVQA)"
            if multimodal
            else "Pre-filling weight optimization data (WikiText)"
        )
        inputs = self._prefill_dataset(
            generator,
            dataloader,
            num_inputs=len(input_spec),
            sample_to_kwargs=sample_to_kwargs,
            desc=desc,
        )
        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))
