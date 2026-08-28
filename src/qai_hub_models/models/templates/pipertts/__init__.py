# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import sys

from qai_hub_models.models.templates.pipertts.monotonic_align import (
    patch_monotonic_align,
)

with patch_monotonic_align() as monotonic_align:
    import piper_train.vits

    # piper_train.vits.monotonic_align expects a compiled Cython submodule
    # (.monotonic_align.core) that is not shipped with the pip package.
    # The standalone monotonic_align package provides the same functionality,
    # so redirect the piper_train import to use it instead.
    sys.modules["piper_train.vits.monotonic_align"] = monotonic_align
    piper_train.vits.monotonic_align = monotonic_align  # type: ignore[attr-defined, unused-ignore]
