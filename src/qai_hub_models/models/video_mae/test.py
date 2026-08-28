# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.templates.video_classifier.model import INPUT_VIDEO_PATH
from qai_hub_models.models.video_mae.app import VideoMAEApp
from qai_hub_models.models.video_mae.demo import main as demo_main
from qai_hub_models.models.video_mae.model import VideoMAE


def test_task() -> None:
    kinetics_app = VideoMAEApp(model=VideoMAE.from_pretrained())
    dst_path = INPUT_VIDEO_PATH.fetch()
    prediction = kinetics_app.predict(path=dst_path)
    assert "surfing water" in prediction


def test_demo() -> None:
    demo_main(is_test=True)
