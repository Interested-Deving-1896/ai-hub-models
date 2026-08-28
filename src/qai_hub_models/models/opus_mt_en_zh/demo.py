# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.opus_mt_en_zh.model import OpusMTEnZh
from qai_hub_models.models.templates.opus_mt.demo import opus_mt_demo


def main(is_test: bool = False) -> None:
    opus_mt_demo(OpusMTEnZh, "English", "Chinese", is_test)


if __name__ == "__main__":
    main()
