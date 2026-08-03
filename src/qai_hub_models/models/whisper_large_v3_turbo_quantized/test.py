# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.whisper_large_v3_turbo_quantized.demo import (
    main as demo_main,
)


def test_demo() -> None:
    demo_main(is_test=True)
