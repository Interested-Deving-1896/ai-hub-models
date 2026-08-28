# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import importlib
import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock


@contextmanager
def patch_monotonic_align() -> Iterator[Any]:
    """
    Monotonic align is a depenency for PiperTTS that is unused during export, but is imported by dependent packages.
    This will patch it to a mock if it is not installed, so PiperTTS can import the modules that import it.
    """
    mono_align = importlib.util.find_spec("monotonic_align")
    should_patch_mono_align = mono_align is None

    mono_align = mono_align or MagicMock()
    if should_patch_mono_align:
        sys.modules["monotonic_align"] = mono_align  # type: ignore[assignment]

    try:
        yield mono_align
    finally:
        if should_patch_mono_align:
            del sys.modules["monotonic_align"]
