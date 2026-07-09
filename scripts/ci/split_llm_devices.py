#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Split the LLM perf workflow's `device` input into per-pool matrix entries.

Emits a single GITHUB_OUTPUT line `matrix_include=<json-array>` for the
matrix `include:` field. The dedicated QDC pool covers devices that use
QDC_PRIVATE_API_KEY; everything else uses the shared pool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEDICATED_DEVICES = {"cs_8_elite_qrd", "cs_x_elite"}


def split(device_input: str) -> tuple[str, str]:
    device_input = (device_input or "all").strip()
    if device_input.lower() == "all":
        return "all", ",".join(sorted(DEDICATED_DEVICES))

    shared: list[str] = []
    dedicated: list[str] = []
    for raw in device_input.split(","):
        d = raw.strip()
        if not d:
            continue
        (dedicated if d in DEDICATED_DEVICES else shared).append(d)
    return ",".join(shared), ",".join(dedicated)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=os.environ.get("DEVICE_INPUT", "all"))
    args = parser.parse_args(argv)

    shared, dedicated = split(args.device)
    entries = []
    if shared:
        entries.append({"pool": "shared", "devices": shared})
    if dedicated:
        entries.append({"pool": "dedicated", "devices": dedicated})

    matrix_include = json.dumps(entries)
    print(f"shared={shared}   dedicated={dedicated}", file=sys.stderr)
    print(f"matrix include: {matrix_include}", file=sys.stderr)

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as f:
            f.write(f"matrix_include={matrix_include}\n")
    else:
        print(matrix_include)
    return 0


if __name__ == "__main__":
    sys.exit(main())
