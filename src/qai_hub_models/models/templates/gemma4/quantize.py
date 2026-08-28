# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
r"""
Shared two-stage quantization workflow for Gemma4 models (E2B, E4B, ...).

Starts from an original (non-QAT) checkpoint like every other LLM in the repo:

  Stage A (export-only) -- Export the dynamic-shape ONNX (+ model.data).

  Stage B (calibrate)   -- Build an AIMET-ONNX QuantSim from the stage-A ONNX
    and run compute_encodings() on real data, self-deriving both the weight
    (min-max) and activation (int16) encodings. Optionally runs Sequential MSE
    (``--use-seq-mse``) first, replacing the min-max weight encodings with
    per-output-channel scales chosen to minimize each layer's output MSE.

Stage B requires aimet-onnx + onnxruntime-gpu + a CUDA GPU (e.g. the qaihm-dev
venv). Stage A is CPU-only. Both stages can run in the same venv since the ONNX
is shared (op names stay self-consistent).

Per-model entry points (e.g. gemma_4_e4b_it/quantize.py) call ``main()`` with
the model-specific PreSplit and QuantizablePreSplit classes plus the constants
needed for the CLI help text.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import onnx
import torch

from qai_hub_models import Precision
from qai_hub_models.models.templates.llm.quantize import save_command_args
from qai_hub_models.utils.args import get_quantize_action_with_default
from qai_hub_models.utils.dataset_util import dataset_entries_to_dataloader

logger = logging.getLogger(__name__)

# Files copied alongside the ONNX so the checkpoint is self-contained.
_AUX_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def _resolve_checkpoint(checkpoint: str | None, hf_repo_name: str) -> Path:
    ckpt = checkpoint or os.environ.get("GEMMA4_LOCAL_CHECKPOINT") or hf_repo_name
    path = Path(ckpt)
    if path.exists():
        return path
    # Not a local dir: treat ``ckpt`` as an HF repo id and download the full
    # checkpoint snapshot (safetensors + config + tokenizer).
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(ckpt)))


# Op types Seq-MSE sweeps (aimet_onnx SUPPORTED_MODULES). Non-matmul params
# (e.g. RMSNorm scales) are untouched and stay per-tensor.
_SEQ_MSE_OP_TYPES = ("Conv", "Gemm", "MatMul")


@contextmanager
def _patch_extractor_for_dual_role_outputs() -> Iterator[None]:
    """Let Seq-MSE's subgraph extractor see KV-cache tensors as subgraph inputs.

    Gemma4 exposes each new KV slice as both a graph output and an internal
    Concat input. ONNX shape inference skips ``value_info`` for graph outputs, so
    the extractor's ``vimap`` misses them and raises ``KeyError`` when Seq-MSE
    picks one as a subgraph boundary. Seed vimap from ``graph.output`` (full
    type+shape) for the Seq-MSE run only. Remove once aimet-onnx does this itself.
    """
    from aimet_onnx.utils import LazyExtractor

    original_init = LazyExtractor.__init__

    def patched_init(self: Any, model: onnx.ModelProto) -> None:
        original_init(self, model)
        added = 0
        for out in self.graph.output:
            if out.name not in self.vimap and out.HasField("type"):
                self.vimap[out.name] = out
                added += 1
        if added:
            print(
                f"Seq-MSE: seeded {added} dual-role graph outputs (KV cache) "
                f"into the subgraph extractor's value-info map."
            )

    LazyExtractor.__init__ = patched_init
    try:
        yield
    finally:
        LazyExtractor.__init__ = original_init


# Weights with more out-channels than this can't be reliably swept: too few
# tokens per channel at realistic calib sizes. Only lm_head (262144) crosses it;
# next largest weight is 12288.
_SEQ_MSE_MAX_OUT_CHANNELS = 32768


def _seq_mse_excluded_nodes(onnx_path: str | os.PathLike) -> list[str]:
    """Names of Conv/Gemm/MatMul nodes to keep out of the Seq-MSE sweep."""
    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    inits = {i.name: i for i in graph.initializer}
    excluded = []
    for node in graph.node:
        if node.op_type not in _SEQ_MSE_OP_TYPES:
            continue
        for inp in node.input:
            init = inits.get(inp)
            if (
                init is not None
                and init.dims
                and int(init.dims[0]) > _SEQ_MSE_MAX_OUT_CHANNELS
            ):
                excluded.append(node.name)
                break
    return excluded


def _assert_weights_per_channel(quant_sim: Any, onnx_path: str | os.PathLike) -> None:
    """Fail fast if Seq-MSE's target weights are not per-channel (or per-block).

    Seq-MSE inherits each quantizer's granularity, so a per-tensor weight yields
    one scale after an hours-long sweep. Check only the Conv/Gemm/MatMul weights
    it sweeps (norm scales are per-tensor by design). Targets come from the
    pre-QuantSim ONNX -- QuantSim rewires weights through QcQuantizeOp so they're
    no longer direct node inputs there.
    """
    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    initializers = {init.name for init in graph.initializer}
    targets: set[str] = set()
    for node in graph.node:
        if node.op_type not in _SEQ_MSE_OP_TYPES:
            continue
        targets.update(i for i in node.input if i in initializers)

    per_tensor: list[str] = []
    checked = 0
    for name in sorted(targets):
        qop = quant_sim.qc_quantize_op_dict.get(name)
        if qop is None or not qop.enabled:
            continue
        checked += 1
        # usePerChannelMode is what seq_mse's channelAxis read depends on;
        # blockSize > 0 means per-block, which is finer still and also fine.
        if (
            not qop.quant_info.usePerChannelMode
            and getattr(qop.quant_info, "blockSize", 0) <= 0
        ):
            per_tensor.append(name)
    if checked == 0:
        raise ValueError(
            "Sequential MSE pre-flight found no enabled weight quantizers on "
            f"{_SEQ_MSE_OP_TYPES} ops -- refusing to run a no-op optimization."
        )
    if per_tensor:
        raise ValueError(
            f"Sequential MSE requires per-channel weight quantizers, but "
            f"{len(per_tensor)} of {checked} optimized weights are per-tensor "
            f"(e.g. {per_tensor[:5]}). Seq-MSE optimizes at each quantizer's "
            f"existing granularity, so those would collapse to a single scale."
        )
    print(f"Seq-MSE pre-flight: all {checked} optimized weights are per-channel.")


def export_onnx(
    presplit_cls: Any,
    checkpoint: str | None,
    output_dir: str | os.PathLike,
    sequence_length: int | None = None,
    context_length: int | None = None,
) -> Path:
    """Stage A: export the dynamic-shape ONNX to output_dir.

    Produces model_dynamic.onnx + model.data + tokenizer/config. This directory
    is the --onnx-dir input to stage B.
    """
    src_ckpt = _resolve_checkpoint(checkpoint, presplit_cls.hf_repo_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq = sequence_length or presplit_cls.default_sequence_lengths[0]
    ctx = context_length or presplit_cls.default_context_lengths[0]

    print(f"[1/3] Loading FP model + weights from {src_ckpt} ...")
    m = presplit_cls.from_pretrained(
        checkpoint=str(src_ckpt), host_device=torch.device("cpu")
    )
    m.sequence_length = seq
    m.context_length = ctx
    m.embedding = m.EmbeddingClass(max_length=ctx, config=m.llm_config)

    print(f"[2/3] Exporting dynamic ONNX (seq={seq}, ctx={ctx}) ...")
    bundle = m.get_full_onnx_bundle(output_dir / "_export_tmp")
    src_onnx = Path(bundle.onnx_graph_path)
    dst_onnx = output_dir / "model_dynamic.onnx"
    shutil.copy(src_onnx, dst_onnx)
    for cand in ("model.data", "model.onnx.data"):
        p = src_onnx.parent / cand
        if p.exists():
            shutil.copy(p, output_dir / "model.data")
            break

    print("[3/3] Copying tokenizer/config ...")
    for fn in _AUX_FILES:
        s = src_ckpt / fn
        if s.exists():
            shutil.copy(s, output_dir / fn)

    shutil.rmtree(output_dir / "_export_tmp", ignore_errors=True)
    print(f"\nStage A complete. ONNX export written to {output_dir}")
    print(f"  contents: {sorted(p.name for p in output_dir.iterdir())}")
    return output_dir


def quantize(
    presplit_cls: Any,
    quant_presplit_cls: Any,
    checkpoint: str | None,
    output_dir: str | os.PathLike,
    onnx_dir: str | os.PathLike | None = None,
    precision: Precision = Precision.w4a16,
    num_samples: int = 0,
    sequence_length: int | None = None,
    context_length: int | None = None,
    use_seq_mse: bool = False,
    seq_mse_num_samples: int | None = None,
    raw_text_calibration: bool = False,
) -> Path:
    """Stage B: self-calibrate weight + activation encodings on real data.

    If onnx_dir is None, stage A is run first into output_dir/_onnx_export.

    ``use_seq_mse`` applies Sequential MSE weight optimization before
    calibration.

    ``raw_text_calibration`` calibrates on raw WikiText instead of the default
    chat-templated corpus, reproducing the pre-chat-template behaviour for
    controlled comparisons.
    """
    # aimet-onnx + onnxruntime are stage-B-only (GPU venv) deps. Import lazily
    # so stage A (export_onnx) stays importable in a CPU venv without aimet.
    import onnxruntime

    has_cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    if use_seq_mse and not has_cuda:
        raise ValueError(
            "Sequential MSE requires a CUDA GPU. Re-run stage B on a GPU "
            "machine (qaihm-dev venv) or drop --use-seq-mse."
        )
    if not has_cuda:
        logger.warning(
            "no CUDAExecutionProvider in onnxruntime. AIMET-ONNX "
            "compute_encodings will be very slow / may fail on CPU. Use the "
            "qaihm-dev (GPU) venv for stage B."
        )

    src_ckpt = _resolve_checkpoint(checkpoint, presplit_cls.hf_repo_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq = sequence_length or presplit_cls.default_sequence_lengths[0]
    ctx = context_length or presplit_cls.default_context_lengths[0]

    # Stage A: produce/locate the ONNX export directory.
    if onnx_dir is None:
        onnx_dir = output_dir / "_onnx_export"
        export_onnx(
            presplit_cls=presplit_cls,
            checkpoint=str(src_ckpt),
            output_dir=onnx_dir,
            sequence_length=seq,
            context_length=ctx,
        )
    onnx_dir = Path(onnx_dir)
    onnx_path = onnx_dir / "model_dynamic.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"Expected model_dynamic.onnx in {onnx_dir}. Run stage A "
            f"(--export-only) first."
        )

    # The FP model holds torch weights only for host-side embedding/PLE
    # computation -> keep it on CPU (cheap). The QuantSim (ONNX) session runs
    # the heavy calibration forward, so place it on CUDA when available.
    fp_device = torch.device("cpu")
    quant_device = torch.device("cuda") if has_cuda else torch.device("cpu")

    print("Stage B: AIMET-ONNX weight + activation calibration")
    print(f"  Checkpoint:      {src_ckpt}")
    print(f"  ONNX source:     {onnx_dir}")
    print(f"  Precision:       {precision}")
    print(f"  Output:          {output_dir}")
    print(f"  seq/ctx:         {seq}/{ctx}")
    print(f"  Seq-MSE:         {use_seq_mse}")
    print(f"  Calib data:      {'raw WikiText' if raw_text_calibration else 'chat'}")
    print(f"  QuantSim device: {quant_device}")
    print(f"  ORT providers:   {onnxruntime.get_available_providers()}")
    print()

    print("Creating FP model (weights for host-side embed/PLE) ...")
    fp_model = presplit_cls.from_pretrained(
        checkpoint=str(src_ckpt), host_device=fp_device
    )
    fp_model.sequence_length = seq
    fp_model.context_length = ctx

    print("Creating QuantizablePreSplit (builds QuantSim from ONNX) ...")
    model_quant = quant_presplit_cls.from_pretrained(
        checkpoint=str(onnx_dir),
        fp_model=fp_model,
        precision=precision,
        host_device=quant_device,
    )

    quant_sim = model_quant.quant_sim
    assert quant_sim is not None

    print(f"Building calibration data ({num_samples or 'auto'} samples) ...")
    calib_data = model_quant.get_calibration_data(
        fp_model=fp_model,
        num_samples=num_samples,
        sequence_length=seq,
        context_length=ctx,
        use_chat_template=not raw_text_calibration,
    )
    assert calib_data is not None
    dataloader = dataset_entries_to_dataloader(calib_data)

    if use_seq_mse:
        # _dataloader_to_numpy clamps to min(len(data), num_batches), so the
        # batch count must be in WINDOWS, not samples, or Seq-MSE sweeps a
        # fraction of the activations.
        windows_per_sample = max(1, ctx // seq)
        requested_samples = seq_mse_num_samples or num_samples or 20
        num_seq_mse_batches = min(
            len(dataloader), requested_samples * windows_per_sample
        )
        if num_seq_mse_batches == 0:
            raise ValueError(
                "Sequential MSE requested but the calibration loader is empty."
            )
        # A loader far smaller than windows_per_sample means something upstream
        # truncated the data (e.g. a get_calibration_data keeping only each
        # sample's first window), so the sweep would fit unrepresentative ranges.
        if num_seq_mse_batches < windows_per_sample:
            raise ValueError(
                f"Sequential MSE would run on only {num_seq_mse_batches} "
                f"window(s), fewer than the {windows_per_sample} windows a "
                f"single {ctx}-token sample should yield at sequence_length="
                f"{seq}. The calibration loader is truncating data, so the "
                f"swept weight ranges would not be representative."
            )
        # Seq-MSE optimizes each weight quantizer at whatever granularity it
        # already has, so w4a16 weights must already be per-channel or the
        # hours-long sweep yields one scale per tensor.
        _assert_weights_per_channel(quant_sim, onnx_path)
        print()
        # Timing reference (E2B, H100, 522 weights x 20 candidates): 128 windows
        # took ~28min against ~5min for plain min-max. Cost scales with
        # calibration volume rather than weight count.
        print(
            f"Applying Sequential MSE weight optimization "
            f"({num_seq_mse_batches} windows = "
            f"{num_seq_mse_batches * seq} tokens). Expect tens of minutes."
        )
        print()
        # lm_head is excluded from the sweep: with one per-channel scale per
        # vocab entry, most channels see a handful of tokens at realistic
        # calibration sizes and Seq-MSE fits their range to noise -- measurably
        # worse WikiText PPL than plain min-max.
        excluded = _seq_mse_excluded_nodes(onnx_path)
        if excluded:
            print(f"Seq-MSE: excluding {len(excluded)} node(s): {excluded}")
        with _patch_extractor_for_dual_role_outputs():
            import aimet_onnx

            aimet_onnx.apply_seq_mse(
                model_quant.quant_sim,
                model_quant._dataloader_to_numpy(dataloader, num_seq_mse_batches),
                nodes_to_exclude=excluded or None,
            )
        gc.collect()
        torch.cuda.empty_cache()

    print("Running activation calibration (compute_encodings) ...")
    model_quant.quantize(data=dataloader, num_samples=num_samples or None)

    print(f"Saving calibrated checkpoint to {output_dir} ...")
    model_quant.save_calibrated_checkpoint(output_dir, fp_model)

    # Copy aux files for a self-contained checkpoint.
    for fn in _AUX_FILES:
        s = src_ckpt / fn
        if s.exists():
            shutil.copy(s, output_dir / fn)

    quant_presplit_cls.release()
    presplit_cls.release()

    print(f"\nStage B complete. Calibrated checkpoint at {output_dir}")
    print(f"  contents: {sorted(p.name for p in output_dir.iterdir())}")
    return output_dir


def main(
    presplit_cls: Any,
    quant_presplit_cls: Any,
    model_id: str,
    supported_precisions: list[Precision],
) -> None:
    """CLI entry point. Called by each model's quantize.py with its classes."""
    model_name = presplit_cls.__name__.replace("_PreSplit", "")
    parser = argparse.ArgumentParser(
        description=f"Quantize {model_name} (weight + activation calibration)"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Original (non-QAT) checkpoint dir (safetensors + config.json). "
        "Defaults to GEMMA4_LOCAL_CHECKPOINT or the HF repo id.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for the calibrated checkpoint (stage B), or the "
        "ONNX export (stage A with --export-only).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        default=False,
        help="Stage A only: export the dynamic ONNX to --output-dir and exit "
        "(no activation calibration).",
    )
    parser.add_argument(
        "--onnx-dir",
        default=None,
        help="Directory with a pre-exported stage-A bundle (model_dynamic.onnx "
        "+ model.data). Skips stage A.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of WikiText samples for activation calibration (0 = auto).",
    )
    parser.add_argument(
        "--use-seq-mse",
        action="store_true",
        default=False,
        help="Apply Sequential MSE weight optimization before activation "
        "calibration. Requires a CUDA GPU.",
    )
    parser.add_argument(
        "--seq-mse-num-samples",
        type=int,
        default=None,
        help="Number of samples for Sequential MSE. Defaults to --num-samples.",
    )
    parser.add_argument(
        "--raw-text-calibration",
        action="store_true",
        default=False,
        help="Calibrate on raw WikiText instead of the default chat-templated "
        "corpus. Gemma4 ships only instruction-tuned checkpoints, which never "
        "see raw text on device, so this is for controlled comparison only.",
    )
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument(
        "--precision",
        default=Precision.parse(str(supported_precisions[0])),
        action=get_quantize_action_with_default(supported_precisions[0]),
        choices=[str(p) for p in supported_precisions],
        help="Quantization precision.",
    )

    cli_args = sys.argv[1:]
    args = parser.parse_args(cli_args)

    if args.export_only and args.use_seq_mse:
        raise ValueError(
            "--use-seq-mse is a stage-B (calibration) option; it has no effect "
            "with --export-only."
        )

    if args.export_only:
        export_onnx(
            presplit_cls=presplit_cls,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            context_length=args.context_length,
        )
        return

    quantize(
        presplit_cls=presplit_cls,
        quant_presplit_cls=quant_presplit_cls,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        onnx_dir=args.onnx_dir,
        precision=args.precision,
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        context_length=args.context_length,
        use_seq_mse=args.use_seq_mse,
        seq_mse_num_samples=args.seq_mse_num_samples,
        raw_text_calibration=args.raw_text_calibration,
    )

    save_command_args(Path(args.output_dir) / "args.json", args, cli_args)

    print()
    print("Evaluate:")
    print(
        f"    python -m qai_hub_models.models.{model_id}.evaluate "
        f"--checkpoint {args.output_dir} --task wikitext"
    )
    print("Export:")
    print(
        f"    python -m qai_hub_models.models.{model_id}.export "
        f"--checkpoint {args.output_dir} --device 'Snapdragon 8 Elite QRD' "
        f"--skip-profiling --skip-inferencing --output-dir output"
    )
