# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import fcntl
import gc
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import ruamel.yaml
import torch
from packaging.version import Version

from qai_hub_models.scorecard.device import (
    ScorecardDevice,
    cs_8_elite_qrd,
    cs_ventuno_q,
    cs_x_elite,
)

# Minimum torch version required for dynamic-shape ONNX export (dynamo export).
# Note that earlier versions did support dynamic shapes in general, but did
# not work well for LLMs until 2.10. torch >= 2.11 changes the exported graph
# in ways that break our split LLM pipeline (e.g. llama_v3_2_1b_instruct,
# qwen2_5_vl_7b_instruct), so we pin to 2.10.x.
TORCH_DYNAMIC_SHAPE_MIN_VERSION = "2.10"
TORCH_DYNAMIC_SHAPE_BELOW_VERSION = "2.11"
TORCH_SUPPORTS_DYNAMIC_SHAPE = (
    Version(TORCH_DYNAMIC_SHAPE_MIN_VERSION)
    <= Version(torch.__version__)
    < Version(TORCH_DYNAMIC_SHAPE_BELOW_VERSION)
)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


DEDICATED_POOL_DEVICES: frozenset[ScorecardDevice] = frozenset(
    {cs_8_elite_qrd, cs_x_elite, cs_ventuno_q}
)

SHARED_POOL_JOB_LIMIT = 3
DEDICATED_POOL_JOB_LIMIT = 4


def get_qdc_api_token(device: ScorecardDevice) -> str:
    """QDC_PRIVATE_API_KEY for the dedicated pool, QDC_API_TOKEN otherwise."""
    if device in DEDICATED_POOL_DEVICES:
        token = os.environ.get("QDC_PRIVATE_API_KEY")
        if not token:
            raise ValueError(
                f"QDC_PRIVATE_API_KEY is not set; required for {device.name} "
                "(dedicated QDC pool)."
            )
        return token
    token = os.environ.get("QDC_API_TOKEN")
    if not token:
        raise ValueError("QDC_API_TOKEN is not set.")
    return token


def get_qdc_job_limit(device: ScorecardDevice) -> int:
    """Max concurrent QDC jobs allowed under the device's pool."""
    if device in DEDICATED_POOL_DEVICES:
        return DEDICATED_POOL_JOB_LIMIT
    return SHARED_POOL_JOB_LIMIT


class LLMIOType(Enum):
    # Genie-compatible input (with input token ids)
    # Inputs:
    # - input_ids (integer token ids)
    # - attention_mask
    # - position_ids_cos (half size)
    # - position_ids_sin (half size)
    genie_input_ids = "genie_input_ids"

    # Genie-compatible input (with input token embeddings)
    # Inputs:
    # - input_embeds (post-Gather token embeddings)
    # - attention_mask
    # - position_ids_cos (half size)
    # - position_ids_sin (half size)
    genie_input_embeds = "genie_inputs_embeds"

    # Hugging Face original input
    # Inputs:
    # - input_ids (integer token ids)
    # - attention_mask
    # - position_ids (integer position ids)
    huggingface_input_ids = "huggingface_input_ids"


# --- Persistent QDC job records for LLM perf collection ----------------------
# Flat YAML mapping of `<model>_<precision>_<runtime>_<device>` -> record.
# Submit writes one entry per QDC job; collect polls each and on retryable
# failure re-fetches the bundle from release-assets.yaml and resubmits. Kept
# here (not in qdc/qdc_jobs.py) so it stays importable without the QDC SDK.

_JobRecordRetT = TypeVar("_JobRecordRetT")

# Resubmits allowed after the first job, so total tries is this + 1.
DEFAULT_RETRIES = 1


class JobOutcome(str, Enum):
    SUCCESS = "success"
    RETRYABLE_ERROR = "retryable_error"
    RETRYABLE_UNSUCCESSFUL = "retryable_unsuccessful"
    RETRYABLE_EMPTY_LOGS = "retryable_empty_logs"


@dataclass
class JobRecord:
    job_id: str
    attempts_left: int = DEFAULT_RETRIES


def make_key(model_id: str, precision: str, runtime: str, device_name: str) -> str:
    return f"{model_id}_{precision}_{runtime}_{device_name}"


def load_jobs(jobs_file: str | Path) -> dict[str, JobRecord]:
    p = Path(jobs_file)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    with p.open(encoding="utf-8") as f:
        raw = ruamel.yaml.YAML().load(f) or {}
    return {
        k: JobRecord(
            job_id=str(v["job_id"]),
            attempts_left=int(v.get("attempts_left", DEFAULT_RETRIES)),
        )
        for k, v in raw.items()
        if isinstance(v, dict) and "job_id" in v
    }


def poll_and_retry(
    initial_job_id: str,
    attempts_left: int,
    collect_fn: Callable[[str], tuple[_JobRecordRetT, JobOutcome, str | None]],
    resubmit_fn: Callable[[], str],
    on_new_job_id: Callable[[str, int], None] | None = None,
) -> _JobRecordRetT:
    """Poll a QDC job; on retryable failure, resubmit until ``attempts_left`` runs out."""
    job_id = initial_job_id
    while True:
        result, outcome, reason = collect_fn(job_id)
        if outcome is JobOutcome.SUCCESS:
            return result
        if attempts_left <= 0:
            raise RuntimeError(
                f"{reason} after exhausting retry budget. "
                f"Check QDC job logs for details."
            )
        job_id = resubmit_fn()
        attempts_left -= 1
        if on_new_job_id is not None:
            on_new_job_id(job_id, attempts_left)
        print(f"Retrying with new job {job_id} (attempts_left={attempts_left})")


def save_job(
    jobs_file: str | Path,
    key: str,
    job_id: str,
    attempts_left: int = DEFAULT_RETRIES,
) -> None:
    """Upsert one row into ``jobs_file``. fcntl.LOCK_EX-guarded for parallel submitters."""
    p = Path(jobs_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read()
            mapping: dict[str, Any] = (
                dict(ruamel.yaml.YAML().load(raw) or {}) if raw.strip() else {}
            )
            mapping[key] = {"job_id": job_id, "attempts_left": attempts_left}
            f.seek(0)
            f.truncate(0)
            ruamel.yaml.YAML().dump(mapping, f)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
