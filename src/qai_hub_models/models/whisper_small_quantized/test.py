# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from aimet_onnx import quantsim as aimet_quantsim

from qai_hub_models.models.whisper_small_quantized.demo import main as demo_main
from qai_hub_models.models.whisper_small_quantized.model import (
    WhisperSmallEncoderQuantizable,
)
from qai_hub_models.utils.quantization_aimet_onnx import (
    untie_aimet_quantizers_for_op_types,
)


def test_demo() -> None:
    demo_main(is_test=True)


def test_untie_aimet_quantizers_for_op_types() -> None:
    # Tripwire for the tetracode#20671 workaround: if a future aimet drops "Concat"
    # from the default tie list or renames the global, these assertions flag it.
    original = list(aimet_quantsim.op_types_to_tie_qtzrs)
    assert "Concat" in original
    with untie_aimet_quantizers_for_op_types(["Concat"]):
        assert "Concat" not in aimet_quantsim.op_types_to_tie_qtzrs
        assert "AveragePool" in aimet_quantsim.op_types_to_tie_qtzrs
        assert aimet_quantsim._tie_qtzrs
    assert aimet_quantsim.op_types_to_tie_qtzrs == original


def test_encoder_kv_proj_activations_stay_16_bit() -> None:
    encoder = WhisperSmallEncoderQuantizable.from_pretrained()
    assert encoder.quant_sim is not None
    quantizers = encoder.quant_sim.qc_quantize_op_dict
    kv_proj = [
        name
        for name in quantizers
        if name.startswith("/encoder_")
        and ("k_proj" in name or "v_proj" in name)
        and name.endswith("_Conv/Conv_output_0")
    ]
    assert len(kv_proj) == 288
    assert all(quantizers[name].bitwidth == 16 for name in kv_proj)

    cross_cache = [name for name in quantizers if name.startswith("k_cache_cross")]
    assert len(cross_cache) == 12
    assert all(quantizers[name].bitwidth == 8 for name in cross_cache)
