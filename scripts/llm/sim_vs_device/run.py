# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""CLI entry point for the sim-vs-device divergence harness for split LLMs.

Builds the QuantSim split model + golden capture once (harness.py), runs Phase A
(per-Part localization), and -- unless disabled -- auto-runs Phase B (projection
taps) on the Part that Phase A's verdict indicates. See README.md for the
methodology.

Usage (1.7B is the default target):

    cd scripts/llm/sim_vs_device
    python run.py \
        --model-id qwen3_1_7b \
        --checkpoint DEFAULT_W4A16 \
        --device "Snapdragon 8 Elite QRD" \
        --num-windows 3

Compile reuse is deferred to AI Hub's own model cache via --model-cache-mode
(default 'disable' -> always compile fresh; 'enable' -> let the service reuse a
previously compiled model). There is no local compile cache. Part 1 (embedding)
is skipped -- an int Gather has no float inputs to diverge, and its output is
injected as golden into Part 2 anyway.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import qai_hub as hub
import tap_per_layer
import tap_per_part
from harness import build_context

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.utils.model_cache import CacheMode

_DEFAULT_REPORT_JSON = os.path.join(tempfile.gettempdir(), "sim_vs_device_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="qwen3_1_7b")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Calibrated AIMET checkpoint dir or DEFAULT_W4A16",
    )
    parser.add_argument("--device", required=True, help="AI Hub device name")
    parser.add_argument("--precision", default="w4a16")
    parser.add_argument("--num-windows", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument(
        "--model-cache-mode",
        choices=["disable", "enable"],
        default="disable",
        help="AI Hub model cache policy for compiled models. 'disable' (default) "
        "always uploads + compiles fresh -- safe when iterating on tapped bundles, "
        "since AI Hub's cache is keyed by name, not content. 'enable' lets the "
        "service reuse a previously compiled model with the same name.",
    )
    parser.add_argument(
        "--report-json",
        default=_DEFAULT_REPORT_JSON,
        help="Where to write the machine-readable metrics dump (set '' to skip). "
        "Defaults to a file in the system temp dir.",
    )
    parser.add_argument(
        "--phase-b",
        choices=["auto", "off", "force", "all"],
        default="auto",
        help="auto: run Phase B iff Phase A flags a Part; off: Phase A only; "
        "force: always run Phase B on the selected Part (needs --phase-b-part "
        "if the verdict picks none); all: run Phase B on EVERY transformer Part "
        "(fans out; cheap for small models).",
    )
    parser.add_argument(
        "--phase-b-part",
        default=None,
        help="Override the Part Phase B instruments (e.g. 'part3_of_4'). "
        "Defaults to the Part selected from Phase A's verdict.",
    )
    parser.add_argument(
        "--skip-phase-a",
        action="store_true",
        help="Skip Phase A localization and go straight to Phase B. Requires "
        "--phase-b all or --phase-b-part (there's no verdict to auto-select from).",
    )
    args = parser.parse_args()

    precision = Precision.parse(args.precision)
    device = hub.Device(args.device)
    target_runtime = TargetRuntime.GENIE
    model_cache_mode = (
        CacheMode.ENABLE if args.model_cache_mode == "enable" else CacheMode.DISABLE
    )

    if args.skip_phase_a and args.phase_b in ("off", "auto"):
        raise SystemExit(
            "--skip-phase-a requires --phase-b all or --phase-b-part "
            "(no Phase A verdict to auto-select a Part from)."
        )

    ctx = build_context(
        args.model_id,
        args.checkpoint,
        precision,
        args.sequence_length,
        args.context_length,
        args.dataset,
        args.num_windows,
    )
    meta = ctx.meta()
    a_json = Path(args.report_json) if args.report_json else None

    # --- Phase A (unless skipped) ---
    a_result = None
    if not args.skip_phase_a:
        a_result = tap_per_part.run(
            ctx, device, target_runtime, model_cache_mode=model_cache_mode
        )
        tap_per_part.print_report(a_result, meta, a_json)

    # --- Phase B dispatch ---
    if args.phase_b == "off":
        return

    if args.phase_b == "all":
        b_results = tap_per_layer.run_all(
            ctx,
            device,
            target_runtime,
            model_cache_mode=model_cache_mode,
        )
        for b_result in b_results:
            b_json = (
                a_json.with_name(
                    f"{a_json.stem}_tap_per_layer_{b_result.part_name}.json"
                )
                if a_json
                else None
            )
            tap_per_layer.print_report(b_result, meta, b_json)
        return

    verdict = a_result.verdict if a_result is not None else None
    target_part = tap_per_layer.select_part(verdict, ctx, override=args.phase_b_part)
    if target_part is None:
        if args.phase_b == "force":
            raise SystemExit(
                "Phase B forced but no Part selected; pass --phase-b-part."
            )
        print("Phase A verdict does not implicate a Part; skipping Phase B.")
        return

    b_result = tap_per_layer.run(
        ctx,
        device,
        target_part,
        target_runtime,
        model_cache_mode=model_cache_mode,
    )
    b_json = a_json.with_name(a_json.stem + "_tap_per_layer.json") if a_json else None
    tap_per_layer.print_report(b_result, meta, b_json)


if __name__ == "__main__":
    main()
