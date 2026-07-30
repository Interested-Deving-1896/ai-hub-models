# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Local-only inspector: validate Phase B tap discovery before spending a compile.

Runs the SAME discovery Phase B uses (``tap_per_layer.discover_taps``): per-layer
residual boundaries from the production split utility, o_proj/down via the stable
weight anchor, and gate_up (the SwiGLU product) via graph back-trace from
down_proj -- gate/up weights are renamed by the optimizer, so topology from the
node we can find is the anchor, not the name. Prints each discovered tap.

``--dump-layer N`` is a raw diagnostic that dumps every initializer + MatMul/Conv
node and back-traces the MLP subgraph for one layer -- use it to see the real
structure when discovery misses something.

This module runs NO device jobs.

    cd scripts/llm/sim_vs_device
    python inspect_taps.py \
        --model-id qwen3_1_7b --checkpoint DEFAULT_W4A16 --part part3_of_4
"""

from __future__ import annotations

import argparse
import json

import onnx
from harness import (
    build_split_model,
    resolve_model_module,
)
from tap_per_layer import discover_taps

from qai_hub_models import Precision
from qai_hub_models.utils.onnx.helpers import ONNXBundle


def _encoding_names(bundle: ONNXBundle) -> tuple[set[str], set[str]]:
    """(activation_encoding_names, param_encoding_names)."""
    if bundle.aimet_encodings_path is None:
        return set(), set()
    data = json.loads(bundle.aimet_encodings_path.read_text())

    def names(section: str) -> set[str]:
        s = data.get(section, [])
        if isinstance(s, dict):
            return set(s.keys())
        return {e["name"] for e in s if isinstance(e, dict) and "name" in e}

    return names("activation_encodings"), names("param_encodings")


def _dump_layer(bundle: ONNXBundle, layer: int) -> None:
    """Diagnostic: dump every initializer + MatMul/Conv node touching a layer.

    Used to discover the real gate/up/down naming when the weight markers miss
    (e.g. plain nn.Linear MatMul weights renamed/fused by the optimize passes).
    """
    model = onnx.load(str(bundle.onnx_graph_path), load_external_data=False)
    act_enc, param_enc = _encoding_names(bundle)
    tag = f"layers.{layer}."

    print(f"\n=== initializers touching layer {layer} ===")
    for init in model.graph.initializer:
        if tag in init.name.replace("/", "."):
            enc = "param-enc" if init.name in param_enc else ""
            print(f"  {init.name}   dims={list(init.dims)}  {enc}")

    print(f"\n=== MatMul / Conv / Gemm nodes touching layer {layer} ===")
    for node in model.graph.node:
        if node.op_type not in ("MatMul", "Conv", "Gemm"):
            continue
        touches = tag in (node.name or "").replace("/", ".") or any(
            tag in i.replace("/", ".") for i in node.input
        )
        if not touches:
            continue
        out = node.output[0] if node.output else "?"
        print(f"  [{node.op_type}] name={node.name}")
        for i in node.input:
            print(f"      in:  {i}")
        print(f"      out: {out}   {'act-enc' if out in act_enc else 'NO-ENC'}")

    # Back-trace from down_proj's input to find the gate/up subgraph (their
    # weights are renamed by the optimizer, so we reach them by topology, not
    # by name -- anchor on down_proj, which we CAN find, and walk backwards).
    producers = {o: n for n in model.graph.node for o in n.output}
    down = next(
        (
            n
            for n in model.graph.node
            if n.op_type == "Conv"
            and any(f"layers.{layer}.mlp.down_proj" in i for i in n.input)
        ),
        None,
    )
    if down is None:
        return
    print(f"\n=== back-trace from down_proj input ({down.input[0]}) ===")
    seen: set[str] = set()
    frontier = [(down.input[0], 0)]
    while frontier:
        tensor, depth = frontier.pop(0)
        if tensor in seen or depth > 6:
            continue
        seen.add(tensor)
        prod = producers.get(tensor)
        if prod is None:
            continue
        enc = "act-enc" if tensor in act_enc else "NO-ENC"
        indent = "  " * (depth + 1)
        wname = next((i for i in prod.input if i in param_enc), "")
        print(
            f"{indent}{tensor} <- [{prod.op_type}] {prod.name}  ({enc})"
            + (f"  w={wname}" if wname else "")
        )
        for i in prod.input:
            if i not in {x.name for x in model.graph.initializer}:
                frontier.append((i, depth + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="qwen3_1_7b")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--precision", default="w4a16")
    parser.add_argument("--part", required=True, help="e.g. part3_of_4")
    parser.add_argument(
        "--dump-layer",
        type=int,
        default=None,
        help="Diagnostic: dump all initializers + MatMul/Conv nodes for this "
        "global layer index and exit (use to find gate/up naming).",
    )
    parser.add_argument(
        "--split-layers",
        action="store_true",
        help="Validate the per-layer sub-graph split (Phase B isolated-input "
        "strategy) locally: cut the part into one sub-graph per layer and print "
        "each sub-graph's I/O + taps. No device jobs.",
    )
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--context-length", type=int, default=4096)
    args = parser.parse_args()

    precision = Precision.parse(args.precision)
    module = resolve_model_module(args.model_id)
    split_model = build_split_model(
        module, args.checkpoint, precision, args.sequence_length, args.context_length
    )
    split_model._ensure_parts()
    cap_names = [
        f"part{i + 1}_of_{len(split_model._parts)}"
        for i in range(len(split_model._parts))
    ]
    idx = cap_names.index(args.part)
    part = split_model._parts[idx]
    bundle = part._get_onnx_bundle()

    if args.dump_layer is not None:
        _dump_layer(bundle, args.dump_layer)
        return

    if args.split_layers:
        from layer_split import (
            add_projection_taps,
            split_part_into_layers,
        )
        from tap_per_layer import _first_transformer_layer

        # Derive the global first-layer index from cumulative per-part layer
        # counts (same logic tap_per_layer uses), not a uniform-split assumption.
        base = _first_transformer_layer(part)
        subs = split_part_into_layers(part, bundle, base)
        print(f"\nPart {args.part}: split into {len(subs)} per-layer sub-graphs\n")
        for sub in subs:
            add_projection_taps(sub)
            print(f"layer {sub.layer}:")
            print(f"  inputs : {sub.input_names}")
            print(f"  taps   : {[(t.projection, t.output_name) for t in sub.taps]}")
        print(
            "\nIf every layer shows a prev-residual input + mask/pos + KV, and "
            "residual/o_proj/gate_up/down taps, the split is sound.\n"
        )
        return

    # Use the exact discovery Phase B will use.
    taps, dropped = discover_taps(part, bundle)
    act_enc, param_enc = _encoding_names(bundle)
    print(
        f"\nPart {args.part}: {len(taps)} taps discovered, {len(dropped)} dropped "
        f"({len(act_enc)} activation encs, {len(param_enc)} param encs)\n"
    )
    print(f"{'layer':>5} {'proj':>10}  tensor -> sanitized_output_name")
    by_layer: dict[int, dict[str, str]] = {}
    for t in taps:
        by_layer.setdefault(t.layer, {})[t.projection] = t.tensor_name
        print(f"{t.layer:>5} {t.projection:>10}  {t.tensor_name} -> {t.output_name}")

    # Coverage grid: which columns resolved per layer.
    cols = ["residual", "o_proj", "gate_up", "down"]
    print(f"\ncoverage:  {'layer':>5} " + "".join(f"{c:>10}" for c in cols))
    for layer in sorted(by_layer):
        marks = "".join(f"{'Y' if c in by_layer[layer] else '-':>10}" for c in cols)
        print(f"           {layer:>5} {marks}")
    if dropped:
        print(
            f"\n{len(dropped)} dropped (no activation encoding): "
            f"{dropped[:5]}{' ...' if len(dropped) > 5 else ''}"
        )
    print()


if __name__ == "__main__":
    main()
