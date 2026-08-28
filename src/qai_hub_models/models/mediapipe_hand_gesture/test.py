# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.mediapipe_hand_gesture.app import MediaPipeHandGestureApp
from qai_hub_models.models.mediapipe_hand_gesture.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.mediapipe_hand_gesture.demo import main as demo_main
from qai_hub_models.models.mediapipe_hand_gesture.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MediaPipeHandGesture,
)
from qai_hub_models.models.templates.mediapipe.test_utils import (
    assert_landmarks_close,
    landmarks_from_raw_output,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)

# Golden structured output (bounding box + landmark coordinates) for the demo
# image. Comparing these directly is robust to the cross-environment pixel
# drift that made the old rendered-image comparison flaky.
LANDMARKS_GOLDEN_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "hand_gesture_landmarks_golden.npz"
)

# Because we have not made a modification to the pytorch source network,
# no numerical tests are included for the model; only for the app.


def test_hand_app() -> None:
    image = load_image(INPUT_IMAGE_ADDRESS)
    expected = load_numpy(LANDMARKS_GOLDEN_ADDRESS)
    app = MediaPipeHandGestureApp.from_pretrained(
        MediaPipeHandGesture.from_pretrained(include_detector_postprocessing=True)
    )
    actual = landmarks_from_raw_output(
        app.predict_landmarks_from_image(image, raw_output=True)
    )
    assert_landmarks_close(actual, expected)


def test_hand_app_no_detect_postprocess() -> None:
    image = load_image(INPUT_IMAGE_ADDRESS)
    expected = load_numpy(LANDMARKS_GOLDEN_ADDRESS)
    app = MediaPipeHandGestureApp.from_pretrained(
        MediaPipeHandGesture.from_pretrained(include_detector_postprocessing=False)
    )
    actual = landmarks_from_raw_output(
        app.predict_landmarks_from_image(image, raw_output=True)
    )
    assert_landmarks_close(actual, expected)


def test_demo() -> None:
    demo_main(is_test=True)
