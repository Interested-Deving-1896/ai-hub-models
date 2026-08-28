# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import json

import onnx
import piper_train
import torch
from piper_train.vits.commons import init_weights
from piper_train.vits.models import SynthesizerTrn
from torch import Tensor
from torch.nn import Conv1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm


def build_model_from_onnx(onnx_path: str, config_path: str) -> SynthesizerTrn:
    with open(config_path) as f:
        config = json.load(f)

    num_symbols = config["num_symbols"]  # 256
    num_speakers = config["num_speakers"]  # 1
    sample_rate = config["audio"]["sample_rate"]  # 22050
    print(
        f"Config: num_symbols={num_symbols}, num_speakers={num_speakers}, sample_rate={sample_rate}"
    )

    # default params of SynthesizerTrn
    model_g = SynthesizerTrn(
        n_vocab=num_symbols,
        spec_channels=513,
        segment_size=32,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
        upsample_rates=(8, 8, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16, 8),
        n_speakers=num_speakers,
        gin_channels=0,
        use_sdp=True,
    )

    # Remove weight norm from dec AND flow (ONNX has merged weights for both)
    model_g.eval()
    with torch.no_grad():
        model_g.dec.remove_weight_norm()  # type: ignore[union-attr, operator, unused-ignore]
        for flow_layer in model_g.flow.flows:
            if hasattr(flow_layer, "enc"):
                flow_layer.enc.remove_weight_norm()  # type: ignore[union-attr, operator, unused-ignore]

    # Load ONNX initializers
    print(f"Loading weights from ONNX: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    onnx_weights = {
        init.name: torch.from_numpy(onnx.numpy_helper.to_array(init).copy())
        for init in onnx_model.graph.initializer
    }
    print(f"Found {len(onnx_weights)} weight tensors in ONNX")

    state_dict = model_g.state_dict()
    name_map = {}  # pt_key → onnx_key
    special = {}  # pt_key → tensor (for computed values)

    # ── 1. Direct name matches ────────────────────────────────────────────────
    for pt_key in state_dict:
        if pt_key in onnx_weights:
            name_map[pt_key] = pt_key  # noqa: PERF403

    # ── 2. dp → sdp prefix fallback ──────────────────────────────────────────
    for pt_key in state_dict:
        if pt_key not in name_map and pt_key.startswith("dp."):
            alt = "sdp." + pt_key[3:]
            if alt in onnx_weights:
                name_map[pt_key] = alt

    # ── 3. Embedding: stored as 'sid' (shape [num_symbols, 192]) ─────────────
    emb_key = "enc_p.emb.weight"
    if emb_key not in name_map:
        target = [num_symbols, 192]
        hits = [
            (k, v)
            for k, v in onnx_weights.items()
            if list(v.shape) == target and k not in name_map.values()
        ]
        if hits:
            found, _ = hits[0]
            name_map[emb_key] = found
            print(f"Embedding → ONNX key '{found}'")
        else:
            print("WARNING: embedding not found — audio will be noise!")

    # ── 4. Flow WN weights: auto-named onnx::Conv_* ──────────────────────────
    # Execution order in reverse-mode inference: flows[6], flows[4], flows[2], flows[0]
    # Within each WN: in_layers[i] then res_skip_layers[i], for i in 0..3
    flow_onnx_keys = [
        "onnx::Conv_8168",
        "onnx::Conv_8171",
        "onnx::Conv_8174",
        "onnx::Conv_8177",
        "onnx::Conv_8180",
        "onnx::Conv_8183",
        "onnx::Conv_8186",
        "onnx::Conv_8189",
        "onnx::Conv_8192",
        "onnx::Conv_8195",
        "onnx::Conv_8198",
        "onnx::Conv_8201",
        "onnx::Conv_8204",
        "onnx::Conv_8207",
        "onnx::Conv_8210",
        "onnx::Conv_8213",
        "onnx::Conv_8216",
        "onnx::Conv_8219",
        "onnx::Conv_8222",
        "onnx::Conv_8225",
        "onnx::Conv_8228",
        "onnx::Conv_8231",
        "onnx::Conv_8234",
        "onnx::Conv_8237",
        "onnx::Conv_8240",
        "onnx::Conv_8243",
        "onnx::Conv_8246",
        "onnx::Conv_8249",
        "onnx::Conv_8252",
        "onnx::Conv_8255",
        "onnx::Conv_8258",
        "onnx::Conv_8261",
    ]
    flow_pt_keys = []
    for fi in [6, 4, 2, 0]:
        for li in range(4):
            flow_pt_keys.append(f"flow.flows.{fi}.enc.in_layers.{li}.weight")
            flow_pt_keys.append(f"flow.flows.{fi}.enc.res_skip_layers.{li}.weight")

    for pt_key, onnx_key in zip(flow_pt_keys, flow_onnx_keys, strict=False):
        if onnx_key in onnx_weights:
            name_map[pt_key] = onnx_key  # noqa: PERF403

    # ── 5. dp.flows.0.logs: stored as exp(-logs) → recover with -log() ───────
    if "onnx::Exp_8159" in onnx_weights:
        special["dp.flows.0.logs"] = -torch.log(onnx_weights["onnx::Exp_8159"])
        print("dp.flows.0.logs recovered from onnx::Exp_8159")

    # ── Load state dict ───────────────────────────────────────────────────────
    new_state_dict = {}
    missing = []
    for pt_key in state_dict:
        if pt_key in special:
            new_state_dict[pt_key] = special[pt_key]
        elif pt_key in name_map:
            new_state_dict[pt_key] = onnx_weights[name_map[pt_key]]
        else:
            missing.append(pt_key)
            new_state_dict[pt_key] = state_dict[pt_key]

    print(f"Loaded {len(state_dict) - len(missing)} / {len(state_dict)} parameters")

    # These are expected to be missing (training-only or unused in inference):
    expected_missing = {"enc_q.", "dp.post_", "dp.flows.1."}
    real_missing = [
        k for k in missing if not any(k.startswith(p) for p in expected_missing)
    ]
    if real_missing:
        print(f"⚠ Inference-critical missing: {real_missing}")
    else:
        print("✓ All inference-critical weights loaded!")

    model_g.load_state_dict(new_state_dict, strict=True)
    model_g.eval()
    return model_g


class Generator_Mod(torch.nn.Module):
    def __init__(self, original_Generator: piper_train.vits.models.Generator) -> None:
        super().__init__()
        self.LRELU_SLOPE = original_Generator.LRELU_SLOPE
        self.num_kernels = original_Generator.num_kernels
        self.num_upsamples = original_Generator.num_upsamples
        self.conv_pre = original_Generator.conv_pre
        self.ups = original_Generator.ups
        self.resblocks = original_Generator.resblocks
        # For w8a16 precision, QAIRT needs Conv1d to has bias
        self.conv_post = Conv1d(32, 1, 7, 1, padding=3, bias=True)
        self.conv_post.bias.data.zero_()  # type: ignore[union-attr, unused-ignore]
        self.ups.apply(init_weights)

        if hasattr(original_Generator, "cond"):
            self.cond = original_Generator.cond

    def forward(self, x: Tensor, g: Tensor | None = None) -> Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            x = up(x)
            xs = torch.zeros(1)
            for j, resblock in enumerate(self.resblocks):
                index = j - (i * self.num_kernels)
                if index == 0:
                    xs = resblock(x)
                elif (index > 0) and (index < self.num_kernels):
                    xs += resblock(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self) -> None:
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()  # type: ignore[operator, unused-ignore]
