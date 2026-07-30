# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""On-device (AI Hub) side of the harness: compile + inference submission.

We do not keep a local compiled-model cache. Reuse is deferred entirely to AI
Hub's own model cache (``get_or_create_cached_model`` / ``CacheMode``),
controlled by the ``--model-cache-mode`` CLI flag. Compiles are submitted, then
collected, so many can run concurrently on the service.

Inference is injected-only and batched: for each Part we submit ONE job whose
``inputs`` dict batches every captured sample under each input name
(``{name: [arr_sample0, arr_sample1, ...]}``). That collapses the ``x samples``
factor to a single job per Part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qai_hub as hub
from golden import PartCapture

from qai_hub_models import Precision
from qai_hub_models.utils.input_spec import TensorSpec, to_hub_input_specs
from qai_hub_models.utils.model_cache import CacheMode, get_or_create_cached_model


@dataclass
class CompiledPart:
    part_index: int
    part_name: str
    linked_model_id: str
    input_order: list[str]  # graph input order from compile target_shapes


def part_input_spec(presplit: object, part: object, seq_len: int, ctx_len: int) -> dict:
    """Per-Part input spec: full-spec shapes where names match, else the
    intermediate hidden state (1, seq, hidden) float32.
    """
    onnx_input_names = part._get_onnx_input_names()
    full_spec = presplit.get_input_spec(
        llm_config=presplit.llm_config.to_dict(),
        sequence_length=seq_len,
        context_length=ctx_len,
        llm_io_type=presplit.llm_io_type,
    )
    spec = {}
    for name in onnx_input_names:
        if name in full_spec:
            spec[name] = full_spec[name]
        else:
            spec[name] = TensorSpec(
                shape=(1, seq_len, presplit.llm_config.hidden_size), dtype="float32"
            )
    return spec


def submit_compile_part(
    presplit: object,
    part: object,
    part_idx: int,
    part_name: str,
    device: hub.Device,
    precision: Precision,
    seq_len: int,
    ctx_len: int,
    target_runtime: object,
    bundle_path: str | None = None,
    cache_name_suffix: str = "",
    model_cache_mode: CacheMode = CacheMode.DISABLE,
) -> object:
    """Upload + submit a Part's compile job WITHOUT waiting.

    Returns the submitted ``hub.client.CompileJob``; collect it later with
    ``collect_compile_part``. Submitting all Parts before waiting lets the
    compiles run concurrently on AI Hub.

    ``bundle_path`` overrides the Part's own ONNX bundle (Phase B passes its
    surgically-tapped bundle). ``cache_name_suffix`` disambiguates the upload
    cache so a tapped bundle never collides with the plain Part.

    ``model_cache_mode`` controls AI Hub's model cache (the only cache this tool
    uses -- there is no local compile cache). DISABLE (default) always uploads
    and compiles fresh, which is the safe default when iterating on tapped
    bundles (AI Hub's cache is keyed by name, not content, so a re-used name
    could otherwise serve a stale graph). Pass ENABLE via --model-cache-mode to
    let the service reuse a previously compiled model.
    """
    num_splits = part.num_splits
    if bundle_path is None:
        bundle_path = part._get_onnx_bundle().bundle_path.as_posix()
    split_input_spec = part_input_spec(presplit, part, seq_len, ctx_len)

    uploaded = get_or_create_cached_model(
        model_name=presplit.model_id,
        model_asset_version=presplit.model_asset_version,
        cache_name=(
            f"debug_ar{seq_len}_cl{ctx_len}_part_{part_idx + 1}_of_{num_splits}"
            f"{cache_name_suffix}"
        ),
        cache_mode=model_cache_mode,
        model_path=bundle_path,
        additional_keys={"precision": str(precision)},
    )

    compile_options = presplit.get_hub_compile_options(
        target_runtime,
        precision,
        "",
        context_graph_name=presplit.get_qnn_context_graph_name(part_idx, num_splits),
    )
    return hub.submit_compile_job(
        model=uploaded,
        input_specs=to_hub_input_specs(split_input_spec),
        device=device,
        name=f"debug_{part_name}{cache_name_suffix}",
        options=compile_options,
    )


def submit_compile_bundle(
    bundle_path: str,
    input_spec: dict,
    device: hub.Device,
    presplit: object,
    precision: Precision,
    part_idx: int,
    num_splits: int,
    target_runtime: object,
    name: str,
    model_cache_mode: CacheMode = CacheMode.DISABLE,
) -> object:
    """Compile an arbitrary tapped bundle (e.g. a per-layer sub-graph).

    Unlike submit_compile_part this takes an explicit input_spec and bundle path
    -- the sub-graph's I/O differs from the part's. ``model_cache_mode`` controls
    AI Hub's model cache (see submit_compile_part); defaults to DISABLE.
    """
    uploaded = get_or_create_cached_model(
        model_name=presplit.model_id,
        model_asset_version=presplit.model_asset_version,
        cache_name=name,
        cache_mode=model_cache_mode,
        model_path=bundle_path,
        additional_keys={"precision": str(precision)},
    )
    compile_options = presplit.get_hub_compile_options(
        target_runtime,
        precision,
        "",
        context_graph_name=presplit.get_qnn_context_graph_name(part_idx, num_splits),
    )
    return hub.submit_compile_job(
        model=uploaded,
        input_specs=to_hub_input_specs(input_spec),
        device=device,
        name=name,
        options=compile_options,
    )


def collect_compile_bundle(cjob: object, label: str) -> CompiledPart:
    """Wait on an arbitrary bundle's compile; return a CompiledPart handle."""
    if not cjob.wait().success:
        raise RuntimeError(f"Compile job {cjob.job_id} for {label} failed.")
    input_order = list(cjob.get_target_shapes().keys())
    return CompiledPart(-1, label, cjob.get_target_model().model_id, input_order)


def submit_feed_inference(
    compiled: CompiledPart,
    feed: dict[str, list[np.ndarray]],
    device: hub.Device,
    label: str,
    options: str = "",
) -> list[InferenceChunkJob]:
    """Submit inference for an explicit {name: [samples]} feed, chunked by size."""
    options = f"{_SINGLE_PASS_OPTION} {options}".strip()
    n = len(next(iter(feed.values()))) if feed else 0
    # Reuse the byte-budget chunker over sample indices.
    sizes = [sum(feed[k][i].nbytes for k in feed) for i in range(n)]
    chunks: list[list[int]] = []
    cur: list[int] = []
    cur_b = 0
    for i, sz in enumerate(sizes):
        if cur and cur_b + sz > _MAX_DATASET_BYTES:
            chunks.append(cur)
            cur, cur_b = [], 0
        cur.append(i)
        cur_b += sz
    if cur:
        chunks.append(cur)

    model = hub.get_model(compiled.linked_model_id)
    submitted: list[InferenceChunkJob] = []
    for ci, idxs in enumerate(chunks):
        inp = {k: [feed[k][i] for i in idxs] for k in feed}
        job = hub.submit_inference_job(
            model=model,
            inputs=inp,
            device=device,
            name=f"{label}_{ci + 1}_of_{len(chunks)}",
            options=options,
        )
        submitted.append(InferenceChunkJob(label, ci, len(chunks), job))
    return submitted


def collect_compile_part(cjob: object, part_idx: int, part_name: str) -> CompiledPart:
    """Wait on a Part's compile job and return its target model handle.

    No link job: link is only needed to weight-share multiple instantiations
    (e.g. AR128 + AR2048) of the same part; we compile a single sequence
    length, so the compile target model is used directly for inference.
    """
    if not cjob.wait().success:
        raise RuntimeError(f"Compile job {cjob.job_id} for {part_name} failed.")
    input_order = list(cjob.get_target_shapes().keys())
    target = cjob.get_target_model()
    return CompiledPart(part_idx, part_name, target.model_id, input_order)


# AI Hub rejects an inference dataset whose serialized flatbuffer exceeds 2 GB.
# Budget below it to leave headroom for framing/metadata overhead.
_MAX_DATASET_BYTES = int(1.6 * 1024**3)

# Inference jobs profile-run the data ~100x by default (--max_profiler_iterations
# default is 100). For a correctness comparison we only need the outputs of a
# single pass, so cap it at 1 to avoid paying ~100x the device time.
_SINGLE_PASS_OPTION = "--max_profiler_iterations 1"


def _sample_nbytes(cap: PartCapture, input_order: list[str], i: int) -> int:
    rec = cap.inputs[i]
    return sum(rec[name].nbytes for name in input_order if name in rec)


def _chunk_indices(cap: PartCapture, input_order: list[str]) -> list[list[int]]:
    """Group sample indices into chunks that each fit the flatbuffer budget.

    KV-cache inputs dominate size (e.g. 10 layers x 2 x ~8 MB per sample), so a
    single batched job can exceed 2 GB. We pack greedily into the fewest chunks
    that each stay under the budget; a single oversized sample still gets its
    own chunk (and the hub error, if any, will be explicit).
    """
    chunks: list[list[int]] = []
    cur: list[int] = []
    cur_bytes = 0
    for i in range(len(cap.inputs)):
        sz = _sample_nbytes(cap, input_order, i)
        if cur and cur_bytes + sz > _MAX_DATASET_BYTES:
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(i)
        cur_bytes += sz
    if cur:
        chunks.append(cur)
    return chunks


def _batched_inputs_for(
    cap: PartCapture, input_order: list[str], indices: list[int]
) -> dict[str, list[np.ndarray]]:
    batched: dict[str, list[np.ndarray]] = {name: [] for name in input_order}
    for i in indices:
        rec = cap.inputs[i]
        for name in input_order:
            if name in rec:
                batched[name].append(rec[name])
    return {k: v for k, v in batched.items() if v}


@dataclass
class InferenceChunkJob:
    """A single submitted inference job (one dataset chunk for one Part)."""

    part_name: str
    chunk_index: int
    num_chunks: int
    job: hub.client.InferenceJob


def submit_part_inference(
    compiled: CompiledPart,
    cap: PartCapture,
    device: hub.Device,
    options: str = "",
) -> list[InferenceChunkJob]:
    """Submit (do NOT wait) all golden-injected inference chunks for a Part.

    Returns the submitted job handles so the caller can fan out across Parts and
    wait on them all together. Chunking keeps each dataset under the flatbuffer
    budget; a Part may therefore map to several jobs. Defaults to a single
    inference pass (see ``_SINGLE_PASS_OPTION``).
    """
    options = f"{_SINGLE_PASS_OPTION} {options}".strip()
    chunks = _chunk_indices(cap, compiled.input_order)
    model = hub.get_model(compiled.linked_model_id)
    if len(chunks) > 1:
        print(
            f"  {cap.part_name}: dataset too large for one job; splitting into "
            f"{len(chunks)} chunks under {_MAX_DATASET_BYTES / 1024**3:.1f} GB."
        )
    submitted: list[InferenceChunkJob] = []
    for ci, indices in enumerate(chunks):
        inputs = _batched_inputs_for(cap, compiled.input_order, indices)
        job = hub.submit_inference_job(
            model=model,
            inputs=inputs,
            device=device,
            name=f"debug_{cap.part_name}_injected_{ci + 1}_of_{len(chunks)}",
            options=options,
        )
        submitted.append(InferenceChunkJob(cap.part_name, ci, len(chunks), job))
    return submitted


def collect_part_inference(
    jobs: list[InferenceChunkJob],
) -> dict[str, list[np.ndarray]]:
    """Wait on a Part's chunk jobs and merge outputs in submission order.

    Raises with the hub message on any chunk failure.
    """
    merged: dict[str, list[np.ndarray]] = {}
    for cj in sorted(jobs, key=lambda j: j.chunk_index):
        status = cj.job.wait()
        if not status.success:
            raise RuntimeError(
                f"Inference job {cj.job.job_id} for {cj.part_name} "
                f"(chunk {cj.chunk_index + 1}/{cj.num_chunks}) failed: "
                f"{status.message}"
            )
        out = cj.job.download_output_data()
        if out is None:
            raise RuntimeError(
                f"Inference job {cj.job.job_id} for {cj.part_name} returned no "
                "output data (job may have failed silently)."
            )
        for name, arrs in dict(out).items():
            merged.setdefault(name, []).extend(arrs)
    return merged
