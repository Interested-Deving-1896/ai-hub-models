# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tap per-Part outputs -- coarse sim-vs-device localization (no graph surgery).

(The driver labels this "Phase A".) For each transformer Part, inject the golden
(QuantSim) inputs recorded at its boundary, run one on-device inference, and
compare the Part's boundary outputs -- the hidden-state (residual stream) plus
the free per-layer KV taps that are already graph outputs. No ONNX surgery is
needed: we tap only what the split already exposes at Part boundaries. Produces
the localized / shared_op / clean verdict that decides whether (and which Part)
the finer per-layer tapping (tap_per_layer) runs on.

Compile jobs for all Parts are submitted before any is awaited, and likewise
for inference, so both fan out concurrently on AI Hub.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import qai_hub as hub
from device import (
    CompiledPart,
    collect_compile_part,
    collect_part_inference,
    submit_compile_part,
    submit_part_inference,
)
from golden import PartCapture
from harness import HarnessContext
from metrics import (
    PartMetrics,
    TensorMetric,
    Verdict,
    kv_layer_index,
    summarize,
)

from qai_hub_models import TargetRuntime
from qai_hub_models.utils.model_cache import CacheMode


@dataclass
class PhaseAResult:
    parts_metrics: list[PartMetrics]
    verdict: Verdict


def compare_part(
    cap: PartCapture, device_out: dict[str, list[np.ndarray]], num_layers: int
) -> PartMetrics:
    """Average per-tensor metrics over samples for one Part."""
    pm = PartMetrics(part_name=cap.part_name, num_layers=num_layers)

    def mean_metric(name: str, key: str) -> TensorMetric | None:
        if key not in device_out:
            return None
        n = min(len(cap.outputs), len(device_out[key]))
        if n == 0:
            return None
        sq, cos, mx = [], [], []
        for i in range(n):
            ref = cap.outputs[i].get(key)
            if ref is None:
                continue
            m = TensorMetric.compare(name, ref, np.asarray(device_out[key][i]))
            sq.append(m.sqnr_db)
            cos.append(m.cosine)
            mx.append(m.max_abs_diff)
        if not sq:
            return None
        return TensorMetric(
            name, float(np.mean(sq)), float(np.mean(cos)), float(np.max(mx))
        )

    pm.hidden = mean_metric(cap.hidden_output_name, cap.hidden_output_name)
    for kv_name in cap.kv_output_names:
        m = mean_metric(kv_name, kv_name)
        if m is not None:
            pm.kv.append(m)
    return pm


def run(
    ctx: HarnessContext,
    device: hub.Device,
    target_runtime: TargetRuntime = TargetRuntime.GENIE,
    skip_embedding: bool = True,
    model_cache_mode: CacheMode = CacheMode.DISABLE,
) -> PhaseAResult:
    """Execute Phase A and return per-Part metrics + verdict."""
    presplit = ctx.presplit

    # Parts we actually evaluate. Skip:
    #  - the embedding / zero-layer Part (int Gather, injected as golden anyway);
    #  - the final LM-head Part: its logits graph-prepare OOMs on the HTP at large
    #    seq_len. Phase B covers that Part's decoder layers (with the LM head
    #    pruned away), so Phase A localization doesn't need it.
    todo = []
    for idx, (part, cap) in enumerate(zip(ctx.parts, ctx.captures, strict=False)):
        if skip_embedding and ctx.layers_per_part.get(idx, 0) == 0:
            print(f"  Skipping {cap.part_name} (no decoder layers).")
            continue
        if part.part_id == part.num_splits:
            print(f"  Skipping {cap.part_name} (LM-head part; covered by Phase B).")
            continue
        todo.append((idx, part, cap))

    # --- submit ALL compile jobs, then collect (parallel on AI Hub) ---
    print("Submitting compile jobs for all Parts (parallel) ...")
    cjobs: dict[int, tuple] = {}  # idx -> (cjob, part_name)
    for idx, part, cap in todo:
        cjob = submit_compile_part(
            presplit,
            part,
            idx,
            cap.part_name,
            device,
            ctx.precision,
            ctx.sequence_length,
            ctx.context_length,
            target_runtime,
            model_cache_mode=model_cache_mode,
        )
        cjobs[idx] = (cjob, cap.part_name)
    compiled_by_idx: dict[int, CompiledPart] = {
        idx: collect_compile_part(cjob, idx, part_name)
        for idx, (cjob, part_name) in cjobs.items()
    }

    # --- submit ALL inference jobs, then collect (parallel) ---
    print("Submitting inference jobs for all Parts (parallel) ...")
    jobs_by_idx = {
        idx: submit_part_inference(compiled_by_idx[idx], cap, device)
        for idx, part, cap in todo
    }

    parts_metrics: list[PartMetrics] = []
    for idx, _part, cap in todo:
        device_out = collect_part_inference(jobs_by_idx[idx])
        parts_metrics.append(
            compare_part(cap, device_out, ctx.layers_per_part.get(idx, 0))
        )

    return PhaseAResult(parts_metrics, summarize(parts_metrics))


# ------------------------------- reporting -------------------------------


def _print_kv_detail(pm: PartMetrics) -> None:
    """Full per-layer KV table -- printed only for flagged Parts."""
    rows: dict[int, dict[str, float]] = {}
    for m in pm.kv:
        li = kv_layer_index(m.name)
        rows.setdefault(li, {})
        if "past_key" in m.name:
            rows[li]["k"] = m.sqnr_db
        elif "past_value" in m.name:
            rows[li]["v"] = m.sqnr_db
    if not rows:
        return
    print(f"    per-layer KV SQNR (dB) for {pm.part_name}:")
    print(f"      {'layer':>6}{'past_key':>12}{'past_value':>13}")
    for li in sorted(rows):
        k = rows[li].get("k", float("nan"))
        v = rows[li].get("v", float("nan"))
        print(f"      {li:>6}{k:>12.1f}{v:>13.1f}")


def report_dict(result: PhaseAResult, meta: dict) -> dict:
    """Machine-readable JSON dump of the Phase A run."""
    v = result.verdict
    return {
        "phase": "A",
        "meta": meta,
        "verdict": {
            "kind": v.kind,
            "detail": v.detail,
            "suspect_parts": v.suspect_parts,
            "kv_path_implicated": v.kv_path_implicated,
        },
        "parts": [
            {
                "part_name": pm.part_name,
                "num_layers": pm.num_layers,
                "hidden_sqnr_db": pm.hidden_sqnr,
                "per_layer_hidden_sqnr_db": pm.per_layer_hidden_sqnr,
                "hidden_cosine": pm.hidden.cosine if pm.hidden else None,
                "hidden_max_abs_diff": pm.hidden.max_abs_diff if pm.hidden else None,
                "kv_min_sqnr_db": pm.min_kv_sqnr,
                "kv_mean_sqnr_db": pm.mean_kv_sqnr,
                "kv_per_tensor": [
                    {
                        "name": m.name,
                        "layer": kv_layer_index(m.name),
                        "sqnr_db": m.sqnr_db,
                        "cosine": m.cosine,
                        "max_abs_diff": m.max_abs_diff,
                    }
                    for m in pm.kv
                ],
            }
            for pm in result.parts_metrics
        ],
    }


def print_report(result: PhaseAResult, meta: dict, json_path: Path | None) -> None:
    verdict = result.verdict
    print("\n" + "=" * 78)
    print("PHASE A -- per-Part sim-vs-device (injected golden inputs)")
    print(
        f"  model={meta['model_id']}  precision={meta['precision']}  "
        f"dataset={meta['dataset']}  windows={meta['num_windows']}  "
        f"ar{meta['sequence_length']}_cl{meta['context_length']}"
    )
    print("=" * 78)
    print(
        f"{'Part':<14}{'layers':>7}{'hidden dB':>12}{'/layer dB':>11}"
        f"{'cos':>8}{'KV min dB':>11}{'KV mean dB':>12}"
    )

    flagged = set(verdict.suspect_parts)
    for pm in result.parts_metrics:
        cos = pm.hidden.cosine if pm.hidden else float("nan")
        mark = "  <--" if pm.part_name in flagged else ""
        print(
            f"{pm.part_name:<14}{pm.num_layers:>7}{pm.hidden_sqnr:>12.1f}"
            f"{pm.per_layer_hidden_sqnr:>11.2f}{cos:>8.4f}"
            f"{pm.min_kv_sqnr:>11.1f}{pm.mean_kv_sqnr:>12.1f}{mark}"
        )

    expand = [
        pm
        for pm in result.parts_metrics
        if pm.part_name in flagged or verdict.kv_path_implicated
    ]
    if expand:
        print("-" * 78)
        print("Auto-expanded per-layer KV (flagged / KV-implicated Parts):")
        for pm in expand:
            _print_kv_detail(pm)

    print("-" * 78)
    print(f"VERDICT [{verdict.kind}]: {verdict.detail}")
    if verdict.suspect_parts:
        print(f"  Suspect Part(s): {verdict.suspect_parts}")
    print(f"  KV path implicated: {verdict.kv_path_implicated}")
    print("=" * 78 + "\n")

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report_dict(result, meta), indent=2))
        print(f"Phase A metrics written to {json_path}\n")
