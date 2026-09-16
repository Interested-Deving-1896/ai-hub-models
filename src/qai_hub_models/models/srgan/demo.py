# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.srgan.model import MODEL_ID, SRGAN
from qai_hub_models.models.templates.super_resolution.demo import super_resolution_demo


def main(is_test: bool = False) -> None:
    super_resolution_demo(SRGAN, MODEL_ID, is_test=is_test)


if __name__ == "__main__":
    main()
