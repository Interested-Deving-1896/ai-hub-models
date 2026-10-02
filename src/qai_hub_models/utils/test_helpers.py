# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Generic test helpers usable from any model unit test.

Does not depend on scorecard.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import pytest

from qai_hub_models.protocols import FromPretrainedProtocol

__all__ = [
    "assert_most_close",
    "assert_most_same",
    "make_cached_from_pretrained_fixture",
]


def assert_most_same(arr1: np.ndarray, arr2: np.ndarray, diff_tol: float) -> None:
    """
    Checks whether most values in the two numpy arrays are the same.

    Particularly for image models, slight differences in the PIL/cv2 envs
    may cause image <-> tensor conversion to be slightly different.

    Instead of using np.assert_allclose, this may be a better way to test image outputs.

    Parameters
    ----------
    arr1
        First input image array.
    arr2
        Second input image array.
    diff_tol
        Float in range [0,1] representing percentage of values
        that can be different while still having the assertion pass.

    Raises
    ------
    AssertionError
        If input arrays are different size, or too many values are different.
    """
    different_values = arr1 != arr2
    assert np.mean(different_values) <= diff_tol, (
        f"More than {diff_tol * 100}% of values were different."
    )


def assert_most_close(
    arr1: np.ndarray,
    arr2: np.ndarray,
    diff_tol: float,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """
    Checks whether most values in the two numpy arrays are close.

    Particularly for image models, slight differences in the PIL/cv2 envs
    may cause image <-> tensor conversion to be slightly different.

    Instead of using np.assert_allclose, this may be a better way to test image outputs.

    Parameters
    ----------
    arr1
        First input image array.
    arr2
        Second input image array.
    diff_tol
        Float in range [0,1] representing percentage of values
        that can be different while still having the assertion pass.
    rtol
        Two values a, b are considered close if the following expresion is true
        `absolute(a - b) <= (atol + rtol * absolute(b))`
        Documentation copied from `np.isclose`.
        Default is 0.0.
    atol
        See rtol documentation.
        Default is 0.0.

    Raises
    ------
    AssertionError
        If input arrays are different size, or too many values are not close.
    """
    not_close_values = ~np.isclose(arr1, arr2, atol=atol, rtol=rtol)
    value_percentage = np.mean(not_close_values)
    assert value_percentage <= diff_tol, (
        f"{value_percentage * 100}% of values were not close (expected: < {diff_tol * 100}% of values)."
    )


def make_cached_from_pretrained_fixture(
    model_cls: type[FromPretrainedProtocol],
) -> Callable[[], Generator[pytest.MonkeyPatch, None, None]]:
    """Build the module-scoped autouse fixture that memoizes
    ``model_cls.from_pretrained``.

    Instantiating the model is expensive, so we load it once per test module and
    monkeypatch ``from_pretrained`` to return the cached instance on subsequent
    calls (keyed by args/kwargs).

    A model that has been torn down by ``release()`` (its ``quant_sim`` freed to
    reclaim CUDA memory between parametrized cases, which sets ``_released``) is
    treated as a cache miss and rebuilt. Models that were not released keep the
    ``getattr`` default and stay cacheable.

    Assign the return value to a module-level name in a conftest, e.g.:

        cached_from_pretrained = make_cached_from_pretrained_fixture(Model)
    """

    @pytest.fixture(scope="module", autouse=True)
    def cached_from_pretrained() -> Generator[pytest.MonkeyPatch, None, None]:
        with pytest.MonkeyPatch.context() as mp:
            pretrained_cache: dict[str, Any] = {}
            from_pretrained = model_cls.from_pretrained
            sig = inspect.signature(from_pretrained)

            def _cached_from_pretrained(*args: Any, **kwargs: Any) -> Any:
                # Key the cache on the exact from_pretrained arguments so that
                # different checkpoints/precisions get distinct cached models.
                cache_key = str(args) + str(kwargs)
                model = pretrained_cache.get(cache_key)
                # A cached AIMET model may have been torn down by release()
                # between parametrized test cases: release() frees quant_sim
                # (and its ORT/CUDA session) to avoid OOM and sets _released.
                # Such an instance is unusable (forward() asserts on quant_sim),
                # so treat it as a miss and rebuild a fresh one. We check the
                # explicit _released flag rather than inferring from _quant_sim:
                # some AIMET models construct lazily with _quant_sim == None and
                # populate it on first access, so a None _quant_sim does not
                # imply the model was released. Non-released and non-AIMET models
                # (no _released attribute) keep the getattr default and stay
                # cached.
                if model is not None and getattr(model, "_released", False):
                    # Drop the released shell so it can be garbage collected
                    # (freeing any remaining resources) before the rebuild.
                    del pretrained_cache[cache_key]
                    model = None
                # Cache hit with a usable model: return it without reloading.
                if model is not None:
                    return model
                # Cache miss (or rebuilt-after-release): load, cache, return.
                non_none_model = from_pretrained(*args, **kwargs)
                pretrained_cache[cache_key] = non_none_model
                return non_none_model

            _cached_from_pretrained.__signature__ = sig  # type: ignore[attr-defined]

            mp.setattr(model_cls, "from_pretrained", _cached_from_pretrained)
            yield mp

    return cached_from_pretrained
