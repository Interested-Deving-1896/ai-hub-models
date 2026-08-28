# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from typing import cast

import numpy as np
import torch
from qai_hub.client import DatasetEntries
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from qai_hub_models.datasets import DatasetSplit, instantiate_dataset
from qai_hub_models.models.owlv2.model import OwlV2
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.templates.owl.app import OwlApp
from qai_hub_models.utils.base_app import (
    CollectionAppEvaluateProtocol,
    CollectionAppQuantizeProtocol,
    CollectionModelEvalGenerator,
)
from qai_hub_models.utils.evaluate.helpers import sample_dataset
from qai_hub_models.utils.inference import AsyncOnDeviceModel, AsyncOnDeviceResult
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries


class OwlV2CollectionApp(
    OwlApp, CollectionAppEvaluateProtocol, CollectionAppQuantizeProtocol
):
    """
    End-to-end inference app for the OwlV2 collection model.

    Uses two separate components:
      * vision_encoder: pixel_values → image_embeds
      * text_detector:  input_ids + attention_mask + image_embeds → boxes, scores, labels
    """

    def __init__(
        self,
        vision_encoder: Callable[..., torch.Tensor],
        text_detector: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        hf_model_id: str = "google/owlv2-base-patch16-ensemble",
        model_image_height: int | None = None,
        model_image_width: int | None = None,
        nms_score_threshold: float = 0.1,
        nms_iou_threshold: float = 0.5,
    ) -> None:
        self.vision_encoder = vision_encoder
        self.text_detector = text_detector
        self.processor = AutoProcessor.from_pretrained(
            hf_model_id,
            use_fast=True,
            size={"height": model_image_height, "width": model_image_width},
        )
        self.model_image_height = model_image_height
        self.model_image_width = model_image_width
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def inference(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ── Vision encoder ───────────────────────────────────────────────────
        image_embeds = self.vision_encoder(pixel_values)

        # ── Text detector ────────────────────────────────────────────────────
        pred_boxes, pred_scores, pred_class_idx = self.text_detector(
            input_ids, attention_mask, image_embeds
        )
        return pred_boxes, pred_scores, pred_class_idx

    def _map_boxes_to_original(
        self,
        boxes: torch.Tensor,
        proc_h: int,
        proc_w: int,
        orig_h: int,
        orig_w: int,
    ) -> torch.Tensor:
        """
        Invert the OWLv2 processor transform: aspect-preserving resize with
        bottom-right padding to a square, so content starts at (0, 0).

        The processor pads the original image to a square (max(H, W), max(W, H))
        with zeros at the right and bottom, then resizes to (proc_H, proc_W).
        The scale is uniform: scale = proc_H / max(orig_H, orig_W).
        """
        if len(boxes) == 0:
            return boxes
        scale = proc_h / max(orig_h, orig_w)
        boxes = boxes.clone().float()
        boxes /= scale
        boxes[:, 0] = boxes[:, 0].clamp(0, orig_w)
        boxes[:, 1] = boxes[:, 1].clamp(0, orig_h)
        boxes[:, 2] = boxes[:, 2].clamp(0, orig_w)
        boxes[:, 3] = boxes[:, 3].clamp(0, orig_h)
        return boxes

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: OwlV2,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        """
        Produces calibration data for a quantize job for the given component.

        For the ``vision`` component, the calibration inputs are just
        ``pixel_values`` drawn from the COCO-OWL dataset.

        For the ``text`` component, the calibration inputs are
        ``input_ids``, ``attention_mask``, and ``image_embeds``.
        ``image_embeds`` are obtained by running the vision encoder on the
        corresponding ``pixel_values`` from the dataset, mirroring the
        real inference pipeline.

        Parameters
        ----------
        collection_model
            The parent OwlV2 collection model.
        component_name
            Either ``"vision"`` or ``"text"``.
        input_specs
            Per-component input specs. If None, uses each component's defaults.
        num_samples
            Number of calibration samples. Defaults to the dataset's
            ``default_num_calibration_samples()``.

        Returns
        -------
        DatasetEntries
            Dataset compatible with the format expected by AI Hub Workbench.
        """
        model = collection_model.components[component_name]
        input_spec = (
            input_specs[component_name] if input_specs else model.get_input_spec()
        )
        vision_fpm = collection_model.components["vision"]
        batch_size = get_batch_size(input_spec) or 1

        calibration_dataset_cls = collection_model.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None, (
            "OwlV2 collection model must provide a calibration dataset."
        )
        dataset = instantiate_dataset(calibration_dataset_cls, DatasetSplit.TRAIN)
        num_samples = num_samples or dataset.default_num_calibration_samples()
        # Round down to a multiple of batch_size
        num_samples = (num_samples // batch_size) * batch_size
        print(f"Loading {num_samples} calibration samples.")
        torch_dataset = sample_dataset(dataset, num_samples)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size)

        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(input_spec))
        ]

        for (pixel_values, input_ids, attention_mask), _ in dataloader:
            if component_name == "vision":
                # Vision encoder only needs pixel_values
                inputs[0].append(pixel_values)
            elif component_name == "text":
                # Text detector needs input_ids, attention_mask, and image_embeds.
                # image_embeds are produced by running the vision encoder.
                with torch.no_grad():
                    image_embeds = vision_fpm(pixel_values)
                inputs[0].append(input_ids)
                inputs[1].append(attention_mask)
                inputs[2].append(image_embeds)
            else:
                raise NotImplementedError(
                    f"Calibration data collection not implemented for component: {component_name!r}"
                )

        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult] | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        image, input_id, attention_mask = model_input
        vision_output = self.vision_encoder(cast(torch.Tensor, image))

        if isinstance(vision_output, AsyncOnDeviceResult):
            yield vision_output
            image_embeds = cast(torch.Tensor, vision_output.wait())
            text_output = self.text_detector(
                input_id,
                attention_mask,
                image_embeds.split(model_batch_size, dim=0),
            )
        else:
            yield (vision_output,)
            text_output = self.text_detector(
                cast(torch.Tensor, input_id),
                cast(torch.Tensor, attention_mask),
                vision_output,
            )

        yield text_output
        return text_output

    @classmethod
    def from_components(
        cls,
        models: Sequence[ExecutableModelProtocol] | Sequence[AsyncOnDeviceModel],
    ) -> OwlV2CollectionApp:
        return cls(
            vision_encoder=models[0],  # type: ignore[arg-type]
            text_detector=models[1],  # type: ignore[arg-type]
        )
