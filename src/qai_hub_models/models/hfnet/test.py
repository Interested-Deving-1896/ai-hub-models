# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

from qai_hub_models.models.hfnet.app import HFNetApp
from qai_hub_models.models.hfnet.demo import main as demo_main
from qai_hub_models.models.hfnet.model import (
    DEFAULT_TORCH_WEIGHTS_FILENAME,
    SOURCE_MODEL,
    HFNet,
    HFNetSourceModelNotFoundError,
)


def test_app_preprocess_shape() -> None:
    image = Image.new("RGB", (1280, 720), color=(127, 127, 127))
    x = HFNetApp.preprocess(image)
    assert tuple(x.shape) == (1, 480, 640, 1)


def test_model_requires_source_model_for_fp() -> None:
    model = HFNet(None)
    image = Image.new("RGB", (640, 480), color=(0, 0, 0))
    x = HFNetApp.preprocess(image)
    with pytest.raises(RuntimeError, match="source_model_path"):
        model(x)


def test_serialize_requires_loaded_torch_model() -> None:
    model = HFNet(None)
    with pytest.raises(RuntimeError, match="local FP runtime"):
        model.serialize("/tmp")


def test_serialize_exports_pt2_from_torch() -> None:
    model = HFNet.from_pretrained()
    with TemporaryDirectory() as tmpdir:
        output = model.serialize(tmpdir)
        assert output == Path(tmpdir) / "hfnet.pt2"
        assert output.exists()


def test_from_pretrained_requires_source_model_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_fetch_failure() -> str:
        raise RuntimeError("missing asset")

    monkeypatch.setattr(
        "qai_hub_models.models.hfnet.model.TORCH_WEIGHTS.fetch",
        _raise_fetch_failure,
    )
    monkeypatch.setattr(
        "qai_hub_models.models.hfnet.model.SOURCE_MODEL.fetch",
        _raise_fetch_failure,
    )

    with pytest.raises(
        HFNetSourceModelNotFoundError, match=DEFAULT_TORCH_WEIGHTS_FILENAME
    ):
        HFNet.from_pretrained(require_source_model=True)


def test_torch_outputs_contract() -> None:
    model = HFNet.from_pretrained()._torch_model
    assert model is not None
    image = HFNetApp.preprocess(Image.new("RGB", (640, 480), color=(127, 127, 127)))
    outputs = model(image)

    assert len(outputs) == 6
    global_descriptor, keypoints, scores, scores_dense, logits, prob_full = outputs

    assert tuple(global_descriptor.shape) == (1, 4096)
    assert keypoints.ndim == 3 and keypoints.shape[0] == 1 and keypoints.shape[-1] == 2
    assert scores.ndim == 2 and scores.shape[0] == 1
    assert tuple(scores_dense.shape) == (1, 480, 640)
    assert tuple(logits.shape) == (1, 60, 80, 65)
    assert tuple(prob_full.shape) == (1, 60, 80, 65)


def test_from_pretrained_matches_onnx_reference() -> None:
    hfnet = HFNet.from_pretrained()
    assert hfnet._torch_model is not None

    app = HFNetApp(hfnet)
    image = Image.new("RGB", (640, 480), color=(127, 127, 127))
    input_tensor = app.preprocess(image)
    image_np = input_tensor.numpy()

    try:
        source_model_path = str(SOURCE_MODEL.fetch())
    except Exception:
        pytest.skip("HFNet ONNX source asset unavailable for parity test")

    session = ort.InferenceSession(
        str(source_model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    expected = session.run(None, {input_name: image_np})

    actual = hfnet(input_tensor)
    actual_np = [out.detach().cpu().numpy() for out in actual]

    assert len(expected) == len(actual_np) == 6
    for expected_out, actual_out in zip(expected, actual_np, strict=True):
        np.testing.assert_allclose(actual_out, expected_out, rtol=1e-4, atol=1e-4)


def test_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    # Verify demo path does not crash in test mode.
    monkeypatch.setattr(
        "qai_hub_models.models.hfnet.demo.demo_model_from_cli_args",
        lambda *_args, **_kwargs: HFNet.from_pretrained(),
    )
    monkeypatch.setattr(
        "qai_hub_models.models.hfnet.demo.load_image",
        lambda *_args, **_kwargs: Image.new("RGB", (640, 480), color=(127, 127, 127)),
    )
    demo_main(is_test=True)
