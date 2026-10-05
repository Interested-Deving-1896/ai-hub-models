# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper
from torch import nn


@dataclass(frozen=True)
class HFNetTorchConfig:
    """Torch-port configuration aligned with the upstream HFNet export config."""

    depth_multiplier: float = 0.75
    n_clusters: int = 32
    descriptor_dim: int = 256
    dimensionality_reduction: int = 4096
    detector_grid: int = 8
    detector_threshold: float = 0.005
    nms_radius: int = 4
    num_keypoints: int = 1000


class _FusedConvRelu6(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=True,
        )
        self.activate = activate

    def _same_upper_pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x

        in_h, in_w = x.shape[-2], x.shape[-1]
        out_h = (in_h + self.stride - 1) // self.stride
        out_w = (in_w + self.stride - 1) // self.stride

        pad_h = max((out_h - 1) * self.stride + self.kernel_size - in_h, 0)
        pad_w = max((out_w - 1) * self.stride + self.kernel_size - in_w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._same_upper_pad(x)
        x = self.conv(x)
        if self.activate:
            x = F.relu6(x)
        return x


class _FusedExpandedConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_channels: int | None,
        use_residual: bool,
    ) -> None:
        super().__init__()
        self.expand: _FusedConvRelu6 | None = None
        cur_channels = in_channels
        if expand_channels is not None:
            self.expand = _FusedConvRelu6(
                in_channels,
                expand_channels,
                kernel_size=1,
                stride=1,
                activate=True,
            )
            cur_channels = expand_channels

        self.depthwise = _FusedConvRelu6(
            cur_channels,
            cur_channels,
            kernel_size=3,
            stride=stride,
            groups=cur_channels,
            activate=True,
        )
        self.project = _FusedConvRelu6(
            cur_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            activate=False,
        )
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.expand is not None:
            x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return x


class _HFNetBackbone(nn.Module):
    """HFNet MobileNetV2 variant matching upstream MOBILENET_DEF."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = _FusedConvRelu6(1, 24, kernel_size=3, stride=2, activate=True)
        self.blocks = nn.ModuleList(
            [
                _FusedExpandedConv(
                    24, 16, stride=1, expand_channels=None, use_residual=False
                ),
                _FusedExpandedConv(
                    16, 24, stride=2, expand_channels=96, use_residual=False
                ),
                _FusedExpandedConv(
                    24, 24, stride=1, expand_channels=144, use_residual=True
                ),
                _FusedExpandedConv(
                    24, 24, stride=2, expand_channels=144, use_residual=False
                ),
                _FusedExpandedConv(
                    24, 48, stride=1, expand_channels=144, use_residual=False
                ),
                _FusedExpandedConv(
                    48, 96, stride=1, expand_channels=288, use_residual=False
                ),
                _FusedExpandedConv(
                    96, 48, stride=2, expand_channels=576, use_residual=False
                ),
                _FusedExpandedConv(
                    48, 48, stride=1, expand_channels=288, use_residual=True
                ),
                _FusedExpandedConv(
                    48, 48, stride=1, expand_channels=288, use_residual=True
                ),
                _FusedExpandedConv(
                    48, 48, stride=1, expand_channels=288, use_residual=True
                ),
                _FusedExpandedConv(
                    48, 72, stride=1, expand_channels=288, use_residual=False
                ),
                _FusedExpandedConv(
                    72, 72, stride=1, expand_channels=432, use_residual=True
                ),
                _FusedExpandedConv(
                    72, 72, stride=1, expand_channels=432, use_residual=True
                ),
                _FusedExpandedConv(
                    72, 120, stride=2, expand_channels=432, use_residual=False
                ),
                _FusedExpandedConv(
                    120, 120, stride=1, expand_channels=720, use_residual=True
                ),
                _FusedExpandedConv(
                    120, 120, stride=1, expand_channels=720, use_residual=True
                ),
                _FusedExpandedConv(
                    120, 240, stride=1, expand_channels=720, use_residual=False
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        local_feat: torch.Tensor | None = None
        global_feat: torch.Tensor | None = None
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx == 5:
                local_feat = x
            if idx == 16:
                global_feat = x
        if local_feat is None or global_feat is None:
            raise RuntimeError(
                "Failed to produce HFNet local/global backbone endpoints"
            )
        return local_feat, global_feat


class HFNetTorch(nn.Module):
    """Torch-native HFNet model with weights loaded from source ONNX."""

    def __init__(self, config: HFNetTorchConfig | None = None) -> None:
        super().__init__()
        self.config = config or HFNetTorchConfig()

        if self.config.depth_multiplier != 0.75:
            raise ValueError(
                "Current HFNet torch scaffold expects depth_multiplier=0.75 to match ONNX"
            )

        self.backbone = _HFNetBackbone()
        global_channels = 240

        self.descriptor_head = nn.Sequential(
            nn.Conv2d(96, self.config.descriptor_dim, kernel_size=3, padding=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                self.config.descriptor_dim,
                self.config.descriptor_dim,
                kernel_size=1,
            ),
        )
        self.detector_head = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                128,
                1 + self.config.detector_grid**2,
                kernel_size=1,
                bias=True,
            ),
        )

        self.vlad_memberships = nn.Conv2d(global_channels, self.config.n_clusters, 1)
        self.vlad_memberships_bn = nn.BatchNorm2d(self.config.n_clusters, eps=1e-3)
        self.vlad_clusters = nn.Parameter(
            torch.randn(self.config.n_clusters, global_channels) * 0.02
        )
        self.global_projection = nn.Linear(
            self.config.n_clusters * global_channels,
            self.config.dimensionality_reduction,
        )

        self._nms_iterations = 3

    @staticmethod
    def _simple_nms(
        scores: torch.Tensor, radius: int, iterations: int = 3
    ) -> torch.Tensor:
        if radius <= 0:
            return scores

        pad = radius
        kernel = 2 * radius + 1

        def _max_pool(x: torch.Tensor) -> torch.Tensor:
            return F.max_pool2d(x.unsqueeze(1), kernel, stride=1, padding=pad).squeeze(
                1
            )

        zeros = torch.zeros_like(scores)
        max_mask = scores == _max_pool(scores)
        for _ in range(iterations - 1):
            supp_mask = _max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == _max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)

    def _extract_keypoints(
        self,
        scores_dense: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, height, width = scores_dense.shape
        flat_scores = scores_dense.reshape(batch, -1)
        k = min(self.config.num_keypoints, height * width)

        if k == 0:
            return (
                torch.zeros(
                    (batch, 0, 2), dtype=torch.int32, device=scores_dense.device
                ),
                torch.zeros(
                    (batch, 0), dtype=scores_dense.dtype, device=scores_dense.device
                ),
            )

        top_scores, flat_idx = torch.topk(flat_scores, k, dim=1)
        below_threshold = top_scores < self.config.detector_threshold
        top_scores = torch.where(
            below_threshold, torch.zeros_like(top_scores), top_scores
        )

        y = torch.div(flat_idx, width, rounding_mode="floor")
        x = flat_idx - y * width
        keypoints_xy = torch.stack((x, y), dim=-1).to(torch.int32)
        return keypoints_xy, top_scores

    def _vlad_descriptor(self, feat: torch.Tensor) -> torch.Tensor:
        memberships = torch.softmax(
            self.vlad_memberships_bn(self.vlad_memberships(feat)),
            dim=1,
        )
        x = feat.unsqueeze(1)
        clusters = self.vlad_clusters.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        residuals = (clusters - x) * memberships.unsqueeze(2)
        vlad = residuals.sum(dim=(-1, -2))
        vlad = F.normalize(vlad, dim=1)
        vlad = vlad.reshape(vlad.shape[0], -1)
        vlad = F.normalize(vlad, dim=1)
        global_desc = self.global_projection(vlad)
        return F.normalize(global_desc, dim=1)

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Expected model input follows existing contract: NHWC grayscale 0..255.
        x = image.permute(0, 3, 1, 2)
        x = (x - 128.0) / 128.0

        local_feat, global_feat = self.backbone(x)
        logits_nchw = self.detector_head(local_feat)
        logits = logits_nchw.permute(0, 2, 3, 1)
        prob_full = torch.softmax(logits, dim=-1)
        prob = prob_full[..., :-1]

        prob_nchw = prob.permute(0, 3, 1, 2)
        scores_dense = F.pixel_shuffle(prob_nchw, self.config.detector_grid).squeeze(1)
        scores_dense = self._simple_nms(
            scores_dense,
            self.config.nms_radius,
            iterations=self._nms_iterations,
        )

        keypoints, scores = self._extract_keypoints(scores_dense)
        global_descriptor = self._vlad_descriptor(global_feat)

        return (
            global_descriptor,
            keypoints,
            scores,
            scores_dense,
            logits,
            prob_full,
        )

    @classmethod
    def from_onnx_weights(
        cls,
        onnx_path: str | Path,
        config: HFNetTorchConfig | None = None,
    ) -> HFNetTorch:
        model = cls(config=config)
        model.load_onnx_weights(onnx_path)
        return model

    @staticmethod
    def _fuse_conv_bn(
        weight: np.ndarray,
        bias: np.ndarray | None,
        gamma: np.ndarray,
        beta: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        eps: float = 1e-3,
    ) -> tuple[np.ndarray, np.ndarray]:
        out_c = weight.shape[0]
        if bias is None:
            bias = np.zeros((out_c,), dtype=np.float32)
        scale = gamma / np.sqrt(var + eps)
        fused_w = weight * scale.reshape(-1, 1, 1, 1)
        fused_b = beta + (bias - mean) * scale
        return fused_w.astype(np.float32), fused_b.astype(np.float32)

    @staticmethod
    def _onnx_initializer_map(onnx_model: onnx.ModelProto) -> dict[str, np.ndarray]:
        return {
            init.name: numpy_helper.to_array(init).astype(np.float32)
            for init in onnx_model.graph.initializer
        }

    @staticmethod
    def _get_init(init: dict[str, np.ndarray], name: str) -> np.ndarray:
        if name in init:
            return init[name]
        candidates = [name]
        if name.startswith("pred/"):
            candidates.append(name[len("pred/") :])
            candidates.append(name.replace("pred/", "", 1).replace("_fused_bn", ""))
        for cand in candidates:
            if cand in init:
                return init[cand]
        raise KeyError(name)

    @staticmethod
    def _find_single_by_prefix(init: dict[str, np.ndarray], prefix: str) -> np.ndarray:
        matches = [v for k, v in init.items() if k.startswith(prefix)]
        if len(matches) != 1:
            raise KeyError(
                f"Expected 1 initializer for prefix {prefix}, got {len(matches)}"
            )
        return matches[0]

    def _load_backbone_from_onnx(self, init: dict[str, np.ndarray]) -> None:
        self.backbone.stem.conv.weight.copy_(
            torch.from_numpy(
                self._get_init(init, "pred/MobilenetV2/Conv/Conv2D_weights_fused_bn")
            )
        )
        stem_bias = self.backbone.stem.conv.bias
        if stem_bias is None:
            raise RuntimeError("HFNet stem conv missing bias parameter")
        stem_bias.copy_(
            torch.from_numpy(
                self._get_init(init, "pred/MobilenetV2/Conv/Conv2D_bias_fused_bn")
            )
        )

        for idx, block_module in enumerate(self.backbone.blocks):
            block = cast(_FusedExpandedConv, block_module)
            tf_block = "expanded_conv" if idx == 0 else f"expanded_conv_{idx}"

            expand_layer = block.expand
            if expand_layer is not None:
                expand_name = (
                    f"pred/MobilenetV2/{tf_block}/expand/Conv2D_weights_fused_bn"
                )
                try:
                    expand_w = self._get_init(init, expand_name)
                    expand_b = self._get_init(
                        init,
                        f"pred/MobilenetV2/{tf_block}/expand/Conv2D_bias_fused_bn",
                    )
                except KeyError:
                    raw_w = self._find_single_by_prefix(
                        init,
                        f"MobilenetV2/{tf_block}/expand/weights/",
                    )
                    gamma = self._find_single_by_prefix(
                        init,
                        f"MobilenetV2/{tf_block}/expand/BatchNorm/gamma/",
                    )
                    beta = self._find_single_by_prefix(
                        init,
                        f"MobilenetV2/{tf_block}/expand/BatchNorm/beta/",
                    )
                    mean = self._find_single_by_prefix(
                        init,
                        f"MobilenetV2/{tf_block}/expand/BatchNorm/moving_mean/",
                    )
                    var = self._find_single_by_prefix(
                        init,
                        f"MobilenetV2/{tf_block}/expand/BatchNorm/moving_variance/",
                    )
                    expand_w, expand_b = self._fuse_conv_bn(
                        raw_w,
                        None,
                        gamma,
                        beta,
                        mean,
                        var,
                        eps=1e-3,
                    )

                expand_layer.conv.weight.copy_(torch.from_numpy(expand_w))
                expand_bias = expand_layer.conv.bias
                if expand_bias is None:
                    raise RuntimeError(
                        f"HFNet block {tf_block} expand conv missing bias"
                    )
                expand_bias.copy_(torch.from_numpy(expand_b))

            depth_w = self._get_init(
                init,
                f"pred/MobilenetV2/{tf_block}/depthwise/depthwise_weights_fused_bn",
            )
            depth_b = self._get_init(
                init,
                f"pred/MobilenetV2/{tf_block}/depthwise/depthwise_bias_fused_bn",
            )
            block.depthwise.conv.weight.copy_(torch.from_numpy(depth_w))
            depth_bias = block.depthwise.conv.bias
            if depth_bias is None:
                raise RuntimeError(
                    f"HFNet block {tf_block} depthwise conv missing bias"
                )
            depth_bias.copy_(torch.from_numpy(depth_b))

            proj_w = self._get_init(
                init,
                f"pred/MobilenetV2/{tf_block}/project/Conv2D_weights_fused_bn",
            )
            proj_b = self._get_init(
                init,
                f"pred/MobilenetV2/{tf_block}/project/Conv2D_bias_fused_bn",
            )
            block.project.conv.weight.copy_(torch.from_numpy(proj_w))
            proj_bias = block.project.conv.bias
            if proj_bias is None:
                raise RuntimeError(f"HFNet block {tf_block} project conv missing bias")
            proj_bias.copy_(torch.from_numpy(proj_b))

    def load_onnx_weights(self, onnx_path: str | Path) -> None:
        onnx_model = onnx.load(str(onnx_path))
        init = self._onnx_initializer_map(onnx_model)

        with torch.no_grad():
            self._load_backbone_from_onnx(init)

            det_conv = cast(nn.Conv2d, self.detector_head[0])
            det_bn = cast(nn.BatchNorm2d, self.detector_head[1])
            det_out = cast(nn.Conv2d, self.detector_head[3])

            det_conv.weight.copy_(
                torch.from_numpy(init["local_head/detector/Conv/weights/read__273"])
            )
            det_bn.weight.copy_(
                torch.from_numpy(
                    init["local_head/detector/Conv/BatchNorm/gamma/read__274"]
                )
            )
            det_bn.bias.copy_(
                torch.from_numpy(
                    init["local_head/detector/Conv/BatchNorm/beta/read__275"]
                )
            )
            if det_bn.running_mean is None or det_bn.running_var is None:
                raise RuntimeError("HFNet detector BN missing running stats")
            det_bn.running_mean.copy_(
                torch.from_numpy(
                    init["local_head/detector/Conv/BatchNorm/moving_mean/read__276"]
                )
            )
            det_bn.running_var.copy_(
                torch.from_numpy(
                    init["local_head/detector/Conv/BatchNorm/moving_variance/read__277"]
                )
            )
            det_out.weight.copy_(
                torch.from_numpy(init["local_head/detector/Conv_1/weights/read__278"])
            )
            det_out_bias = det_out.bias
            if det_out_bias is None:
                raise RuntimeError("HFNet detector output conv missing bias")
            det_out_bias.copy_(
                torch.from_numpy(init["local_head/detector/Conv_1/biases/read__279"])
            )

            # Descriptor branch weights are absent in this reduced export graph.
            # Keep random initialization; this branch is not used for current
            # output contract (no local_descriptors output).

            self.vlad_memberships.weight.copy_(
                torch.from_numpy(
                    self._get_init(
                        init,
                        "global_head/vlad/memberships/weights/read__280",
                    )
                )
            )

            self.vlad_memberships_bn.weight.copy_(
                torch.from_numpy(
                    self._get_init(
                        init,
                        "pred/global_head/vlad/memberships/BatchNorm/Const:0",
                    )
                )
            )
            self.vlad_memberships_bn.bias.copy_(
                torch.from_numpy(
                    self._get_init(
                        init,
                        "global_head/vlad/memberships/BatchNorm/beta/read__281",
                    )
                )
            )
            if (
                self.vlad_memberships_bn.running_mean is None
                or self.vlad_memberships_bn.running_var is None
            ):
                raise RuntimeError("HFNet VLAD BN missing running stats")
            self.vlad_memberships_bn.running_mean.copy_(
                torch.from_numpy(
                    self._get_init(
                        init,
                        "global_head/vlad/memberships/BatchNorm/moving_mean/read__282",
                    )
                )
            )
            self.vlad_memberships_bn.running_var.copy_(
                torch.from_numpy(
                    self._get_init(
                        init,
                        "global_head/vlad/memberships/BatchNorm/moving_variance/read__283",
                    )
                )
            )

            self.vlad_clusters.copy_(
                torch.from_numpy(
                    self._get_init(init, "const_fold_opt__1841")[0, :, :, 0, 0]
                )
            )

            self.global_projection.weight.copy_(
                torch.from_numpy(
                    self._get_init(
                        init,
                        "global_head/dimensionality_reduction/weights/read__285",
                    ).T
                )
            )
            self.global_projection.bias.copy_(
                torch.from_numpy(
                    self._get_init(
                        init,
                        "global_head/dimensionality_reduction/biases/read__286",
                    )
                )
            )

            self.config = HFNetTorchConfig(
                depth_multiplier=self.config.depth_multiplier,
                n_clusters=self.config.n_clusters,
                descriptor_dim=self.config.descriptor_dim,
                dimensionality_reduction=self.config.dimensionality_reduction,
                detector_grid=self.config.detector_grid,
                detector_threshold=float(
                    self._get_init(init, "pred/keypoint_extraction/GreaterEqual/y:0")
                ),
                nms_radius=self.config.nms_radius,
                num_keypoints=int(self._get_init(init, "const_fold_opt__1835")),
            )

        self.eval()
