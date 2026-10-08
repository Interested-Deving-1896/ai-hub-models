# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn.functional as F

from qai_hub_models.datasets.kinetics400.video_utils import (
    read_video_per_second,
    sample_clips,
)
from qai_hub_models.models.vjepa2.app import VJEPA2App
from qai_hub_models.models.vjepa2.dataset import _preprocess_vjepa2
from qai_hub_models.models.vjepa2.model import FRAMES, IMG_SIZE, MODEL_ID, VJEPA2
from qai_hub_models.utils.args import (
    get_model_cli_parser,
    get_on_device_demo_parser,
    model_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_path,
    qaihm_temp_dir,
)

INPUT_VIDEO_PATH = CachedWebModelAsset.from_asset_store(
    "video_classifier", 1, "surfing_cutback.mp4"
)

NUM_GALLERY = 5
NUM_CLIPS = NUM_GALLERY + 1  # 5 gallery + 1 query


def _pool_embedding(emb: torch.Tensor) -> torch.Tensor:
    vec = emb.mean(dim=1).squeeze(0)
    return F.normalize(vec, p=2, dim=-1)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(VJEPA2)
    parser = get_on_device_demo_parser(parser)
    parser.add_argument(
        "--video",
        type=str,
        default=INPUT_VIDEO_PATH,
        help="Path or URL to input video.",
    )
    parser.add_argument(
        "--num-gallery",
        type=int,
        default=NUM_GALLERY,
        help="Number of gallery clips to sample from the video.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = model_from_cli_args(VJEPA2, args)
    app = VJEPA2App(model)
    num_gallery = args.num_gallery
    num_clips = num_gallery + 1

    with qaihm_temp_dir() as tmpdir:
        video_path = load_path(args.video, tmpdir)
        raw = read_video_per_second(str(video_path))  # [T, H, W, C]
    raw_clips = sample_clips(raw, FRAMES, num_clips)  # num_clips x [T, H, W, C]
    clips = [_preprocess_vjepa2(c, FRAMES, IMG_SIZE) for c in raw_clips]
    if not is_test:
        print(
            f"Sampled {num_clips} clips from video, each shape: {tuple(clips[0].shape)}"
        )

    gallery_vecs: list[torch.Tensor] = []
    for i in range(num_gallery):
        emb = app.extract_features(clips[i])  # [1, N, D]
        gallery_vecs.append(_pool_embedding(emb))
        if not is_test:
            print(f"  gallery[{i}] embedding shape: {tuple(emb.shape)}")

    gallery = torch.stack(gallery_vecs)  # [num_gallery, D]

    query_emb = app.extract_features(clips[num_gallery])
    query_vec = _pool_embedding(query_emb)

    if not is_test:
        print(f"\nQuery embedding shape: {tuple(query_emb.shape)}")
        print("\nCosine similarity (query vs gallery):")
        for i, g in enumerate(gallery_vecs):
            sim = float((query_vec * g).sum())
            print(f"  query vs gallery[{i}]: {sim:.6f}")

        sims = (gallery * query_vec.unsqueeze(0)).sum(dim=-1)  # [num_gallery]
        best_idx = int(sims.argmax())
        print(f"\nMost similar: gallery[{best_idx}], similarity = {sims[best_idx]:.6f}")


if __name__ == "__main__":
    main()
