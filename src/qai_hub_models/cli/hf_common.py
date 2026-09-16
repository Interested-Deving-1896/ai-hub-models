# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Helpers and constants shared by the HuggingFace upload and download paths.

``timeout_retry`` lives here rather than in ``scripts/`` because
``upload-to-hf`` is a shipped command: release builds exclude ``scripts/``, so
importing it from there raised ModuleNotFoundError for exactly the external
contributors the command is for.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

CallableRetT = TypeVar("CallableRetT")

COMMUNITY_ORG_NAME = "qualcomm-ai-hub-community"

# Every published card carries this tag, and the tag search below is the index,
# so a personal-namespace repo needs no org membership to be discoverable.
COMMUNITY_TAG = "qai-hub-models"
COMMUNITY_TAG_SEARCH_URL = f"https://huggingface.co/models?other={COMMUNITY_TAG}"


def is_hf_api_timeout_error(e: Exception) -> bool:
    return isinstance(e, TimeoutError)


def timeout_retry(
    do: Callable[[], CallableRetT],
    max_retries: int,
    is_timeout_error: Callable[[Exception], bool] = is_hf_api_timeout_error,
) -> CallableRetT:
    """
    Execute the given callable and return the result.

    If the callable returns a TimeoutError, it will be retried with increasingly large sleep times inbetween calls,
    up to a max number of retries.

    This is used as a workaround for Hugging Face 429 (too many requests!) errors when uploading many models in 1 session.

    Parameters
    ----------
    do
        Callable to do and retry as necessary. Typically you just use (lambda: exp) for this parameter.
    max_retries
        Maximum number of times to retry if we hit timeouts. do() would be executed a maximum of max_retries + 1 times, if it times out on each attempt.
    is_timeout_error
        Returns true if an error (thrown by "do()") is a timeout that should be retried.

    Returns
    -------
    result : CallableRetT
        Result from successful execution of do().

    Raises
    ------
    Exception
        If the allowed number of retries has been exhaused and do() has not succeeded.
    """
    for attempt_idx in range(max_retries + 1):
        try:
            return do()
        except Exception as e:
            if attempt_idx >= max_retries or not is_timeout_error(e):
                raise  # No more retries available, so raise the timeout

        if attempt_idx <= 1:
            time.sleep(10**attempt_idx)  # 1, 10
        else:
            time.sleep(30 * (2**attempt_idx - 2))  # 30, 60, 120, ...
    raise AssertionError()  # line is not reachable
