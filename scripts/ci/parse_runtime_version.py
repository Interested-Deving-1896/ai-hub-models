# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Parse the compound ``runtime_versions`` input passed through scorecard.yml.

Input is one or more comma-separated segments. Each segment is either a bare
value (interpreted as ``qairt_version``) or ``key=value`` where key is one of
``qairt_version`` / ``geniex_version``. See tests for the full accepted /
rejected set. Writes the parsed values to ``$GITHUB_OUTPUT`` (as step outputs
``qairt_version`` / ``geniex_version``) and, when ``--emit-env`` is set, also
to ``$GITHUB_ENV`` as ``QAIHM_TEST_QAIRT_VERSION`` / ``QAIHM_TEST_GENIEX_VERSION``
so downstream shell steps in the same job pick them up as env vars.
"""

from __future__ import annotations

import argparse
import os
import sys

_KNOWN_KEYS = frozenset({"qairt_version", "geniex_version"})


def parse(raw: str) -> tuple[str, str]:
    values: dict[str, str] = {}
    bare_seen = False
    for seg in raw.split(","):
        seg = seg.strip()
        if not seg:
            continue
        if "=" in seg:
            key, _, val = seg.partition("=")
            key = key.strip()
            val = val.strip()
            if key not in _KNOWN_KEYS:
                raise ValueError(f"Unknown override key: {key!r}")
        else:
            if bare_seen:
                raise ValueError(
                    f"Only one bare (positional) value is allowed; got extra: {seg!r}"
                )
            bare_seen = True
            key, val = "qairt_version", seg
        if key in values:
            raise ValueError(f"Duplicate override key: {key!r}")
        values[key] = val

    qairt = values.get("qairt_version", "") or "qaihm_default"
    geniex = values.get("geniex_version", "")
    if geniex.startswith("v"):
        raise ValueError(
            f"geniex_version must not start with 'v' (use bare SemVer, e.g. '0.3.7'); got {geniex!r}"
        )
    return qairt, geniex


def _write_kv_file(path: str, pairs: list[tuple[str, str]]) -> None:
    # GITHUB_OUTPUT/GITHUB_ENV are line-oriented; newlines in a value would
    # inject an extra key=value line and let a crafted input override the
    # intended one.
    for key, val in pairs:
        if "\n" in val or "\r" in val:
            raise ValueError(f"{key} must not contain newlines: {val!r}")
    with open(path, "a") as f:
        for key, val in pairs:
            f.write(f"{key}={val}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=os.environ.get("RAW", ""))
    ap.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    ap.add_argument("--github-env", default=os.environ.get("GITHUB_ENV"))
    ap.add_argument(
        "--emit-env",
        action="store_true",
        help="Also write QAIHM_TEST_QAIRT_VERSION / QAIHM_TEST_GENIEX_VERSION to $GITHUB_ENV.",
    )
    args = ap.parse_args()

    try:
        qairt, geniex = parse(args.raw)
        print(f"Parsed: qairt_version={qairt!r} geniex_version={geniex!r}")
        if args.github_output:
            _write_kv_file(
                args.github_output,
                [("qairt_version", qairt), ("geniex_version", geniex)],
            )
        if args.emit_env and args.github_env:
            _write_kv_file(
                args.github_env,
                [
                    ("QAIHM_TEST_QAIRT_VERSION", qairt),
                    ("QAIHM_TEST_GENIEX_VERSION", geniex),
                ],
            )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
