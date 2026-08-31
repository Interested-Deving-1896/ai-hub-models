# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Constants shared by the HuggingFace upload and download paths."""

from __future__ import annotations

COMMUNITY_ORG_NAME = "qualcomm-ai-hub-community"

# Every published card carries this tag, and the tag search below is the index --
# it is live, sortable, and faceted, so no org page has to be curated to list
# contributions. Publishing into a personal namespace therefore needs no
# membership, no review, and no curation step to become discoverable.
COMMUNITY_TAG = "qai-hub-models"
COMMUNITY_TAG_SEARCH_URL = f"https://huggingface.co/models?other={COMMUNITY_TAG}"

# HuggingFace sorts any tag search server-side; `downloads` is the useful default
# for browsing since the trending score decays and most recipes never trend.
COMMUNITY_TAG_POPULAR_URL = f"{COMMUNITY_TAG_SEARCH_URL}&sort=downloads"
