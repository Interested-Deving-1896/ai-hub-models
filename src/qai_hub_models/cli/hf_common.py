# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Constants shared by the HuggingFace upload and download paths."""

from __future__ import annotations

COMMUNITY_ORG_NAME = "qualcomm-ai-hub-community"

# Every published card carries this tag, and the tag search below is the index,
# so a personal-namespace repo needs no org membership to be discoverable.
COMMUNITY_TAG = "qai-hub-models"
COMMUNITY_TAG_SEARCH_URL = f"https://huggingface.co/models?other={COMMUNITY_TAG}"
