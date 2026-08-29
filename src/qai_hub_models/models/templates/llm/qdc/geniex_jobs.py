# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

from qualcomm_device_cloud_sdk.models import ArtifactType
from transformers import AutoTokenizer

from qai_hub_models.models.templates.llm.common import (
    JobOutcome,
    get_qdc_job_limit,
)
from qai_hub_models.models.templates.llm.grader.grace import GRACE_TASK_NAME
from qai_hub_models.models.templates.llm.model import LLMBase
from qai_hub_models.models.templates.llm.qdc.qdc_jobs import (
    QDCDevice,
    QDCJobs,
    _safe_extract_zip,
    create_zip,
)
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.device import ScorecardDevice

GENIEX_BENCH_JOB_TIMEOUT = 21600  # 6 hours

DEFAULT_LLM_SYSTEM_PROMPT = LLMBase.default_system_prompt

# Versioned URLs follow the geniex release workflow's flat S3 layout
# (<stem>-<vX.Y.Z>.<ext>); the unversioned mirror is refreshed on every
# stable tag and is used when no version is pinned.
_S3_BASE = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex"


def _bench_url(platform_stem: str, ext: str, version: str | None) -> str:
    # Callers pass bare SemVer ("0.3.7"); S3 filenames are v-prefixed.
    suffix = f"-v{version}" if version else ""
    return f"{_S3_BASE}/geniex-bench-{platform_stem}{suffix}.{ext}"


DEFAULT_CONTEXT_LENGTHS = [512, 1024, 4096]

_MIN_RESULT_SCHEMA_VERSION = 3

_N_GEN = 128

# Each accuracy prompt runs as its own geniex-bench process: the timeout kills a
# crashed/hung DSP run so the loop advances, the sleep lets the DSP release
# before the next process attaches.
_EVAL_N_GEN = 4096
_EVAL_TIMEOUT_S = 600
_EVAL_SLEEP_S = 10

# geniex-bench result schemas this parser reads. 4 kept the fields 3 exposed and
# only added some, so both parse identically. An unlisted version is reported by
# compute_metrics rather than dropped, because a silent skip reads as "the device
# produced nothing" and now costs the committed perf rows.
_SUPPORTED_BENCH_SCHEMAS = frozenset({"3", "4"})


@dataclass
class GenieXBenchMetrics:
    cell_id: str
    plugin: str
    device_alias: str
    context_length: int
    ttft_ms: float
    prefill_tps: float
    decode_tps: float
    prompt_tokens: int
    gen_tokens: int


class GenieXBenchArtifactHandler(ABC):
    @abstractmethod
    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        dest_dir: os.PathLike | str,
        chipset: str,
        matrix_rows: list[str],
        context_lengths: list[int],
        plugin: str,
        qairt_bundles: dict[str, str] | None,
        geniex_version: str | None,
        eval_prompts: list[str] | None,
        run_perf: bool,
    ) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def entry_script(self) -> str | None:
        raise NotImplementedError

    @staticmethod
    def _stage_eval_prompts(
        dest_dir: os.PathLike | str, eval_prompts: list[str]
    ) -> None:
        """Write chat-templated prompts to prompts/prompt_NNN.txt in the artifact.

        The device script feeds each file to its own `geniex-bench --accuracy
        --prompt-file` process; presence of the directory is what turns eval on.
        """
        prompts_dir = os.path.join(dest_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        for idx, prompt in enumerate(eval_prompts):
            with open(
                os.path.join(prompts_dir, f"prompt_{idx:03d}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(prompt)

    @staticmethod
    def _apply_common_replacements(
        text: str,
        context_lengths: list[int],
        run_perf: bool,
        eval_prompts: list[str] | None,
    ) -> str:
        """Substitute the placeholders shared by all device scripts.

        CTX_LIST is comma-separated and the RUN_* flags are 1/0 regardless of
        platform; each script parses these into its own types.
        """
        return (
            text.replace("{EVAL_CTX}", str(max(context_lengths)))
            .replace("{EVAL_N_GEN}", str(_EVAL_N_GEN))
            .replace("{EVAL_TIMEOUT_S}", str(_EVAL_TIMEOUT_S))
            .replace("{EVAL_SLEEP_S}", str(_EVAL_SLEEP_S))
            .replace("{CTX_LIST}", ",".join(str(c) for c in context_lengths))
            .replace("{RUN_PERF}", "1" if run_perf else "0")
            .replace("{RUN_EVAL}", "1" if eval_prompts else "0")
        )

    @staticmethod
    def _rewrite_matrix_for_qairt_bundles(
        matrix_rows: list[str],
        qairt_bundles: dict[str, str],
        device_root: str,
    ) -> list[str]:
        # Replace matrix col-4 with the on-device bundle path so qairt plugin
        # skips the model-manager fetch.
        sep = "\\" if "\\" in device_root else "/"
        out: list[str] = []
        for row in matrix_rows:
            parts = row.split("|")
            name = parts[0]
            if name in qairt_bundles:
                parts[3] = f"{device_root}{sep}qairt_bundles{sep}{name}"
            out.append("|".join(parts))
        return out

    @staticmethod
    def _stage_qairt_bundles(
        dest_dir: os.PathLike | str, qairt_bundles: dict[str, str]
    ) -> None:
        base = os.path.join(dest_dir, "qairt_bundles")
        for model_id, bundle_dir in qairt_bundles.items():
            if not os.path.isdir(bundle_dir):
                raise FileNotFoundError(
                    f"QAIRT bundle for {model_id!r} is not a directory: {bundle_dir!r}"
                )
            shutil.copytree(
                bundle_dir, os.path.join(base, model_id), dirs_exist_ok=True
            )


class GenieXBenchAndroidArtifactHandler(GenieXBenchArtifactHandler):
    DEVICE_ROOT = "/data/local/tmp/pkg-geniex"

    @property
    def entry_script(self) -> str | None:
        return None

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        dest_dir: os.PathLike | str,
        chipset: str,
        matrix_rows: list[str],
        context_lengths: list[int],
        plugin: str,
        qairt_bundles: dict[str, str] | None,
        geniex_version: str | None,
        eval_prompts: list[str] | None,
        run_perf: bool,
    ) -> str:
        ds_dir = os.path.join(curr_dirname, "device_scripts")
        pytest_dir = os.path.join(ds_dir, "geniex_pytest")

        if plugin == "qairt" and qairt_bundles:
            matrix_rows = self._rewrite_matrix_for_qairt_bundles(
                matrix_rows, qairt_bundles, device_root=self.DEVICE_ROOT
            )

        bench_url = _bench_url("android-arm64", "tar.gz", geniex_version)
        test_folder = os.path.join(dest_dir, "tests")
        os.makedirs(test_folder, exist_ok=True)
        for fn in os.listdir(pytest_dir):
            if fn.endswith(".pyc"):
                continue
            src = os.path.join(pytest_dir, fn)
            if not os.path.isfile(src):
                continue
            with open(src, encoding="utf-8") as f:
                content = f.read()
            if fn.endswith(".py"):
                content = self._apply_common_replacements(
                    content.replace("{ANDROID_BENCH_URL}", bench_url)
                    .replace("{PLUGIN}", plugin)
                    .replace("{N_GEN}", str(_N_GEN)),
                    context_lengths,
                    run_perf,
                    eval_prompts,
                )
            out_path = (
                os.path.join(dest_dir, fn)
                if fn == "requirements.txt"
                else os.path.join(test_folder, fn)
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)

        with open(
            os.path.join(dest_dir, "matrix_rows.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(matrix_rows) + "\n")
        with open(os.path.join(dest_dir, "chipset.txt"), "w", encoding="utf-8") as f:
            f.write(chipset + "\n")

        if plugin == "qairt" and qairt_bundles:
            self._stage_qairt_bundles(dest_dir, qairt_bundles)

        if eval_prompts:
            self._stage_eval_prompts(dest_dir, eval_prompts)

        zip_path = os.path.join(os.path.dirname(dest_dir), "geniex_bench_test.zip")
        create_zip(zip_path, dest_dir)
        return zip_path


class GenieXBenchLinuxArtifactHandler(GenieXBenchArtifactHandler):
    DEVICE_ROOT = "/data/local/tmp/TestContent"

    @property
    def entry_script(self) -> str:
        return f"/bin/bash {self.DEVICE_ROOT}/run_geniex_bench_linux.sh"

    def _bench_size_flags(
        self, plugin: str, qairt_bundles: dict[str, str] | None
    ) -> str:
        if plugin == "qairt":
            assert qairt_bundles and len(qairt_bundles) == 1
            (name,) = qairt_bundles
            prompt_path = f"{self.DEVICE_ROOT}/qairt_bundles/{name}/sample_prompt.txt"
            return f'-c "$ctx" -n {_N_GEN} --prompt-file "{prompt_path}"'
        return f'-c "$((ctx + {_N_GEN}))" -p "$ctx" -n {_N_GEN}'

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        dest_dir: os.PathLike | str,
        chipset: str,
        matrix_rows: list[str],
        context_lengths: list[int],
        plugin: str,
        qairt_bundles: dict[str, str] | None,
        geniex_version: str | None,
        eval_prompts: list[str] | None,
        run_perf: bool,
    ) -> str:
        ds_dir = os.path.join(curr_dirname, "device_scripts")
        sh_src = os.path.join(ds_dir, "run_geniex_bench_linux.sh")

        if plugin == "qairt" and qairt_bundles:
            matrix_rows = self._rewrite_matrix_for_qairt_bundles(
                matrix_rows, qairt_bundles, device_root=self.DEVICE_ROOT
            )

        with open(sh_src, encoding="utf-8") as f:
            script = f.read()
        script = self._apply_common_replacements(
            script.replace(
                "{LINUX_BENCH_URL}",
                _bench_url("linux-arm64", "tar.gz", geniex_version),
            )
            .replace("{CHIPSET}", chipset)
            .replace("{MODELS}", "\n".join(matrix_rows))
            .replace(
                "{BENCH_SIZE_FLAGS}",
                self._bench_size_flags(plugin, qairt_bundles),
            ),
            context_lengths,
            run_perf,
            eval_prompts,
        )
        sh_dest = os.path.join(dest_dir, "run_geniex_bench_linux.sh")
        with open(sh_dest, "w", encoding="utf-8") as f:
            f.write(script)
        os.chmod(sh_dest, 0o755)

        if plugin == "qairt" and qairt_bundles:
            self._stage_qairt_bundles(dest_dir, qairt_bundles)

        if eval_prompts:
            self._stage_eval_prompts(dest_dir, eval_prompts)

        zip_path = os.path.join(os.path.dirname(dest_dir), "geniex_bench_test.zip")
        create_zip(zip_path, dest_dir)
        return zip_path


class GenieXBenchWindowsArtifactHandler(GenieXBenchArtifactHandler):
    DEVICE_ROOT = "C:\\Temp\\TestContent"

    @property
    def entry_script(self) -> str:
        return f"{self.DEVICE_ROOT}\\run_geniex_bench_windows.ps1"

    def _bench_size_flags_args(
        self, plugin: str, qairt_bundles: dict[str, str] | None
    ) -> str:
        if plugin == "qairt":
            assert qairt_bundles and len(qairt_bundles) == 1
            (name,) = qairt_bundles
            path = f"{self.DEVICE_ROOT}\\qairt_bundles\\{name}\\sample_prompt.txt"
            return f'"-c", "$ctx", "-n", "{_N_GEN}", "--prompt-file", "{path}",'
        return f'"-c", "$($ctx + {_N_GEN})", "-p", "$ctx", "-n", "{_N_GEN}",'

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        dest_dir: os.PathLike | str,
        chipset: str,
        matrix_rows: list[str],
        context_lengths: list[int],
        plugin: str,
        qairt_bundles: dict[str, str] | None,
        geniex_version: str | None,
        eval_prompts: list[str] | None,
        run_perf: bool,
    ) -> str:
        ds_dir = os.path.join(curr_dirname, "device_scripts")
        ps1_src = os.path.join(ds_dir, "run_geniex_bench_windows.ps1")

        if plugin == "qairt" and qairt_bundles:
            matrix_rows = self._rewrite_matrix_for_qairt_bundles(
                matrix_rows, qairt_bundles, device_root=self.DEVICE_ROOT
            )

        with open(ps1_src, encoding="utf-8") as f:
            script = f.read()
        script = self._apply_common_replacements(
            script.replace(
                "{WINDOWS_BENCH_URL}",
                _bench_url("windows-arm64", "zip", geniex_version),
            )
            .replace("{CHIPSET}", chipset)
            .replace("{MODELS}", "\n".join(matrix_rows))
            .replace(
                "{BENCH_SIZE_FLAGS_ARGS}",
                self._bench_size_flags_args(plugin, qairt_bundles),
            ),
            context_lengths,
            run_perf,
            eval_prompts,
        )
        with open(
            os.path.join(dest_dir, "run_geniex_bench_windows.ps1"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(script)

        if plugin == "qairt" and qairt_bundles:
            self._stage_qairt_bundles(dest_dir, qairt_bundles)

        if eval_prompts:
            self._stage_eval_prompts(dest_dir, eval_prompts)

        zip_path = os.path.join(os.path.dirname(dest_dir), "geniex_bench_test.zip")
        create_zip(zip_path, dest_dir)
        return zip_path


class GenieXBenchQDCJobs(QDCJobs):
    def _get_artifact_handler(
        self, qdc_device: QDCDevice
    ) -> GenieXBenchArtifactHandler:
        if qdc_device.windows_platform:
            return GenieXBenchWindowsArtifactHandler()
        if qdc_device.iot_platform:
            return GenieXBenchLinuxArtifactHandler()
        if qdc_device.mobile_platform:
            return GenieXBenchAndroidArtifactHandler()
        raise NotImplementedError(
            "geniex-bench currently supports Windows (Snapdragon X / X2 "
            "Elite), IoT Linux (Dragonwing IQ-9075 EVK), and Android "
            "(Snapdragon 8 Elite QRD / Gen 5 QRD). "
            f"Device {qdc_device.device.name!r} is none of these."
        )

    def add_job_artifacts(
        self,
        qdc_device: QDCDevice,
        chipset: str,
        matrix_rows: list[str],
        plugin: str,
        context_lengths: list[int] = DEFAULT_CONTEXT_LENGTHS,
        qairt_bundles: dict[str, str] | None = None,
        geniex_version: str | None = None,
        eval_prompts: list[str] | None = None,
        run_perf: bool = True,
    ) -> tuple[list[str], str | None]:
        curr_dirname = os.path.dirname(os.path.abspath(__file__))
        handler = self._get_artifact_handler(qdc_device)
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = handler.create_artifact(
                curr_dirname,
                tmpdir,
                chipset,
                matrix_rows,
                context_lengths,
                plugin,
                qairt_bundles,
                geniex_version,
                eval_prompts,
                run_perf,
            )
            artifact = self.upload_file(zip_path, ArtifactType.TESTSCRIPT)
        return [artifact], handler.entry_script

    def compute_metrics(
        self,
        job_log_files: list,
        save_results_dir: str | None = None,
        save_logs_dir: str | None = None,
    ) -> list[GenieXBenchMetrics]:
        """Parse metrics from QDC logs; if ``save_logs_dir`` is set, keep the raw zips there too."""
        metrics: list[GenieXBenchMetrics] = []
        if save_logs_dir:
            os.makedirs(save_logs_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            for job_log in job_log_files:
                target = os.path.join(tmpdir, "logs", f"{job_log.filename}.zip")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                if not self.try_download_job_log_files(job_log.filename, target):
                    continue
                if save_logs_dir:
                    safe_name = os.path.basename(job_log.filename)
                    shutil.copy(target, os.path.join(save_logs_dir, f"{safe_name}.zip"))
                try:
                    _safe_extract_zip(target, tmpdir)
                except zipfile.BadZipFile:
                    continue

            unreadable_schemas: dict[str, str] = {}
            for root, _, files in os.walk(tmpdir):
                for fn in sorted(files):
                    if not fn.endswith(".json"):
                        continue
                    path = os.path.join(root, fn)
                    parsed = self._parse_cell_metrics(path)
                    if parsed is None:
                        if version := self._unreadable_cell_schema(path):
                            unreadable_schemas[fn] = version
                        continue
                    metrics.append(parsed)
                    if save_results_dir:
                        rel = os.path.relpath(path, tmpdir)
                        dest = os.path.join(save_results_dir, rel)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        shutil.copy(path, dest)

        if metrics:
            print(f"Parsed {len(metrics)} geniex-bench cells:")
            for m in metrics:
                print(
                    f"  [{m.cell_id} ctx={m.context_length}] "
                    f"decode={m.decode_tps:.2f} tok/s, prefill={m.prefill_tps:.2f} tok/s, "
                    f"TTFT={m.ttft_ms:.1f} ms"
                )
        elif unreadable_schemas:
            seen = sorted(set(unreadable_schemas.values()))
            raise RuntimeError(
                f"geniex-bench produced results this parser cannot read: schema "
                f"version(s) {', '.join(seen)}, supported "
                f"{', '.join(sorted(_SUPPORTED_BENCH_SCHEMAS))}. Add the version to "
                "_SUPPORTED_BENCH_SCHEMAS once its fields are confirmed compatible. "
                "Failing rather than reporting no metrics, so a schema bump can't be "
                "replayed as a measurement that never happened.\n"
                + "\n".join(
                    f"  {fn}: schema_version={v}"
                    for fn, v in sorted(unreadable_schemas.items())
                )
            )
        else:
            print(
                "Warning: no geniex-bench results (schema_version >= "
                f"{_MIN_RESULT_SCHEMA_VERSION}) found in logs."
            )

        return metrics

    @staticmethod
    def _unreadable_cell_schema(path: str) -> str | None:
        """Schema version of a bench cell this parser rejected, else None.

        The log bundle also carries the model's own config/tokenizer JSON, so a
        file only counts as a bench cell if it has cell_id and agg.
        """
        try:
            with open(path, encoding="utf-8") as f:
                cell = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(cell, dict) or "cell_id" not in cell or "agg" not in cell:
            return None
        version = str(cell.get("schema_version"))
        return None if version in _SUPPORTED_BENCH_SCHEMAS else version

    @staticmethod
    def _parse_cell_metrics(path: str) -> GenieXBenchMetrics | None:
        try:
            with open(path, encoding="utf-8") as f:
                cell = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(cell, dict):
            return None
        # geniex-bench bumps schema_version for additive changes; the fields we read
        # below are stable, and missing ones are rejected by the None checks.
        try:
            if int(cell.get("schema_version", 0)) < _MIN_RESULT_SCHEMA_VERSION:
                return None
        except (TypeError, ValueError):
            return None
        agg = cell.get("agg") or {}
        params = cell.get("params") or {}

        def med(key: str) -> float | None:
            entry = agg.get(key) or {}
            return entry.get("median")

        ttft = med("ttft_ms")
        prefill = med("prefill_tps")
        decode = med("decode_tps")
        if ttft is None or prefill is None or decode is None:
            return None

        cell_id = cell.get("cell_id") or ""
        _, sep, suffix = cell_id.rpartition("-c")
        ctx = int(suffix) if sep and suffix.isdigit() else int(params.get("n_ctx") or 0)
        if ctx == 0:
            return None

        return GenieXBenchMetrics(
            cell_id=cell_id,
            plugin=cell.get("plugin") or "",
            device_alias=cell.get("device") or "",
            context_length=ctx,
            ttft_ms=float(ttft),
            prefill_tps=float(prefill),
            decode_tps=float(decode),
            prompt_tokens=int((agg.get("prompt_tokens") or {}).get("median") or 0),
            gen_tokens=int((agg.get("gen_tokens") or {}).get("median") or 0),
        )

    def compute_eval_results(
        self,
        job_log_files: list,
        prompts: list[str],
        save_logs_dir: str | None = None,
    ) -> list[dict]:
        """Parse ``geniex_eval_outputs.txt`` into [{idx, prompt, output}].

        The device scripts run ``geniex-bench --accuracy`` once per prompt and
        append each invocation's stdout to a single ``geniex_eval_outputs.txt``,
        with ``===EVAL_IDX_NNN===`` markers separating prompts. If
        ``save_logs_dir`` is set the raw eval log zip lands there too.
        """
        outputs: dict[int, str] = {}
        if save_logs_dir:
            os.makedirs(save_logs_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            for job_log in job_log_files:
                if "geniex_eval_outputs" not in job_log.filename:
                    continue
                target = os.path.join(tmpdir, "logs", f"{job_log.filename}.zip")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                if not self.try_download_job_log_files(job_log.filename, target):
                    continue
                if save_logs_dir:
                    safe_name = os.path.basename(job_log.filename)
                    dest = os.path.join(save_logs_dir, f"{safe_name}.zip")
                    # Skip if compute_metrics already copied this zip in the
                    # same call -- avoids the redundant I/O when both perf
                    # and eval run.
                    if not os.path.exists(dest):
                        shutil.copy(target, dest)
                try:
                    _safe_extract_zip(target, tmpdir)
                except zipfile.BadZipFile:
                    continue

            for root, _, files in os.walk(tmpdir):
                for fn in files:
                    if "geniex_eval_outputs" not in fn or fn.endswith(".zip"):
                        continue
                    with open(os.path.join(root, fn), "rb") as f:
                        outputs.update(
                            _parse_eval_outputs(_decode_device_log(f.read()))
                        )

        return [
            {
                "idx": idx,
                "prompt": prompts[idx] if idx < len(prompts) else "",
                "output": _extract_model_output(outputs.get(idx, "")),
            }
            for idx in sorted(outputs.keys())
        ]


def _decode_device_log(data: bytes) -> str:
    """Decode a device eval log, tolerating CP1252 from older Windows runs.

    PS 5.1's Add-Content defaulted to the ANSI codepage, so logs collected
    before that was pinned to utf8 hold lone high bytes that strict UTF-8
    rejects. Decoding those as CP1252 recovers the original text instead of
    replacing every accented character with U+FFFD.
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _parse_eval_outputs(content: str) -> dict[int, str]:
    """Split ``geniex_eval_outputs.txt`` into a ``{idx: raw_stdout}`` map.

    Format: ``===EVAL_IDX_NNN===`` followed by that prompt's raw geniex-bench
    stdout, up to the next marker.
    """
    outputs: dict[int, str] = {}
    parts = re.split(r"===EVAL_IDX_(\d+)===\n?", content)
    for i in range(1, len(parts) - 1, 2):
        outputs[int(parts[i])] = parts[i + 1].strip()
    return outputs


def _extract_model_output(raw_output: str) -> str:
    """Extract the generated text from one ``geniex-bench --accuracy`` stdout.

    ``--accuracy`` prints each generated line prefixed with ``[gen ] `` and a
    final ``[ok  ] ...`` perf-summary line. We keep only the ``[gen ]`` lines
    (stripping the prefix) and join them; everything else (the ``[ok  ]``
    summary, stray log lines) is dropped.
    """
    gen_prefix = "[gen ] "
    lines = [
        line[len(gen_prefix) :]
        for line in raw_output.splitlines()
        if line.startswith(gen_prefix)
    ]
    return "\n".join(lines).strip()


def save_eval_results_json(results: list[dict], output_path: str) -> None:
    """Save evaluation results to a JSON file, sorted by idx."""
    if not results:
        print("No results to save.")
        return

    results.sort(key=lambda r: r.get("idx", 0))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


def save_eval_metadata_json(
    model_id: str,
    chipset: str,
    precision: str,
    output_path: str,
    path: ScorecardProfilePath,
    dataset_name: str = GRACE_TASK_NAME,
) -> None:
    """Save a sidecar identifying which (model, chipset, precision, path, dataset) an eval JSON belongs to.

    The grader output (``*_eval_grade.json``) carries no model/chipset/precision,
    and the eval filename cannot be parsed unambiguously (model IDs and chipset
    slugs both contain delimiters). collect_llm_accuracy_csv reads this sidecar
    to recover the identity, and skips any grade file that lacks it. ``path`` is
    the scorecard runtime the accuracy row is written under.
    """
    metadata = {
        "model_id": model_id,
        "chipset": chipset,
        "precision": precision,
        "path": path.value,
        "dataset_name": dataset_name,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Eval metadata saved to: {output_path}")


def _apply_chat_template(prompts: list[str], tokenizer_source: str) -> list[str]:
    """Wrap each prompt in the model's chat template so --prompt-file gets a real turn.

    geniex-bench feeds --prompt-file verbatim (no template), so the turn is
    applied host-side. Thinking is disabled to match the genie side.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    templated: list[str] = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": DEFAULT_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        # Templates fail many ways on a system role (Gemma's raise_exception(),
        # ValueError on unsupported roles); fall back to a user-only turn.
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        templated.append(formatted)
    return templated


def _resolve_tokenizer_source(
    plugin: str, model_rows: list[tuple[str, str]], model_id: str | None
) -> str:
    """Where to load the chat template from.

    For qairt, uses the local genie/geniex bundle: its tokenizer matches
    what ships on-device, avoids gated-HF auth, and (with a current
    transformers) parses the newer list-of-dicts chat_template format
    correctly. For llama_cpp (gguf refs have no local tokenizer directory),
    uses ``HF_REPO_NAME`` from the model module.
    """
    if plugin == "qairt":
        return model_rows[0][1]
    hf_repo: str | None = None
    if model_id:
        module = importlib.import_module(f"qai_hub_models.models.{model_id}")
        hf_repo = getattr(module, "HF_REPO_NAME", None)
    if hf_repo:
        return hf_repo
    raise ValueError(
        f"{model_id or '<unknown>'} has no HF_REPO_NAME; cannot resolve a "
        "chat template for llama_cpp eval prompts."
    )


def _hf_repo(model_url: str) -> str:
    if "huggingface.co/" not in model_url:
        raise ValueError(f"Only HuggingFace URLs are supported: {model_url}")
    parts = model_url.split("huggingface.co/")[1].split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse HF repo from: {model_url}")
    if not model_url.endswith(".gguf"):
        raise ValueError(f"Expected .gguf URL, got: {model_url}")
    return f"{parts[0]}/{parts[1]}"


def _build_matrix_rows(
    model_rows: list[tuple[str, str]],
    plugin: str,
    device_alias: str,
    llamacpp_quant: str | None,
) -> tuple[list[str], dict[str, str]]:
    matrix_rows: list[str] = []
    qairt_bundles: dict[str, str] = {}
    for name, ref in model_rows:
        if plugin == "qairt":
            qairt_bundles[name] = ref
            model_id = name
        else:
            model_id = f"{_hf_repo(ref)}:{llamacpp_quant}"
        matrix_rows.append(f"{name}|{plugin}|{device_alias}|{model_id}||")
    return matrix_rows, qairt_bundles


def submit_geniex_bench_only(
    api_token: str,
    hub_device_name: str,
    chipset: str,
    model_rows: list[tuple[str, str]],
    context_lengths: list[int] = DEFAULT_CONTEXT_LENGTHS,
    plugin: str = "llama_cpp",
    device_alias: str = "hybrid",
    job_name: str = "geniex-bench",
    geniex_version: str | None = None,
    llamacpp_quant: str | None = None,
    eval_prompts: list[str] | None = None,
    model_id: str | None = None,
    run_perf: bool = True,
) -> tuple[str, list[str], dict[str, str]]:
    """Upload artifacts and submit a geniex-bench job, returning the id.

    Companion to ``collect_geniex_bench_result``. Also returns the
    computed ``matrix_rows`` and ``qairt_bundles`` so the caller can
    persist them for a later resubmit.

    eval_prompts set => accuracy collection: chat-templated host-side and
    staged into the bundle for one ``geniex-bench --accuracy`` pass. model_id
    resolves the llama_cpp chat template. run_perf=False submits an eval-only
    job (no TPS/TTFT sweep).
    """
    if plugin == "llama_cpp" and not llamacpp_quant:
        raise ValueError("llamacpp_quant is required when plugin='llama_cpp'.")

    qdc_device = QDCDevice(hub_device_name)
    matrix_rows, qairt_bundles = _build_matrix_rows(
        model_rows, plugin, device_alias, llamacpp_quant
    )

    # Eval prep is additive to perf: on failure, drop eval to save the perf
    # data, but let an eval-only run (run_perf=False) raise.
    templated_prompts: list[str] | None = None
    if eval_prompts:
        try:
            templated_prompts = _apply_chat_template(
                eval_prompts, _resolve_tokenizer_source(plugin, model_rows, model_id)
            )
        except Exception as e:
            if not run_perf:
                raise
            print(f"WARNING: eval prep failed ({e}); running perf only.")

    geniex_job = GenieXBenchQDCJobs(
        api_key=api_token,
        app_name_header="GenieXBenchQDCJobApp",
        job_limit=get_qdc_job_limit(ScorecardDevice.get(hub_device_name)),
    )

    job_artifacts, entry_script = geniex_job.add_job_artifacts(
        qdc_device,
        chipset,
        matrix_rows,
        plugin,
        context_lengths=context_lengths,
        qairt_bundles=qairt_bundles or None,
        geniex_version=geniex_version,
        eval_prompts=templated_prompts,
        run_perf=run_perf,
    )

    job_id = geniex_job.submit_automated_job(
        qdc_device,
        job_artifacts,
        entry_script,
        job_name=job_name,
        timeout=GENIEX_BENCH_JOB_TIMEOUT,
    )
    if job_id is None:
        raise RuntimeError("Job submission failed.")
    print(f"Submitted QDC job with ID: {job_id}")
    return job_id, matrix_rows, qairt_bundles


def collect_geniex_bench_result(
    api_token: str,
    hub_device_name: str,
    job_id: str,
    save_results_dir: str | None = None,
    eval_prompts: list[str] | None = None,
    run_perf: bool = True,
    save_logs_dir: str | None = None,
    log_label: str | None = None,
) -> tuple[list[GenieXBenchMetrics], list[dict], JobOutcome, str | None]:
    """Poll a submitted geniex-bench job and download+parse logs on success.

    Returns ``(metrics, eval_results, outcome, reason)``. ``metrics`` and
    ``eval_results`` are empty unless ``outcome`` is SUCCESS; ``reason``
    carries the failure description on a non-SUCCESS outcome. eval_prompts
    is only consulted on success to attach the human-readable prompt text
    to each parsed output (run_perf=False yields eval-only results).
    ``log_label`` names the per-job log archive written under ``save_logs_dir``.
    """
    geniex_job = GenieXBenchQDCJobs(
        api_key=api_token,
        app_name_header="GenieXBenchQDCJobApp",
    )

    job_status = geniex_job.status(job_id)
    job_result = geniex_job.result(job_id)
    print(f"QDC job {job_id} completed with status: {job_status}, result: {job_result}")

    if job_result is not None and job_result != "Successful":
        reason = (
            f"QDC job {job_id} on device '{hub_device_name}' finished with "
            f"status='{job_status}', result='{job_result}'"
        )
        outcome = (
            JobOutcome.RETRYABLE_ERROR
            if job_result == "Error"
            else JobOutcome.RETRYABLE_UNSUCCESSFUL
        )
        print(f"[result={job_result}] {reason}")
        geniex_job.save_job_logs(job_id, save_logs_dir, label=log_label)
        return [], [], outcome, reason

    geniex_job.log_upload_status(job_id)
    # The file listing lags log-upload-status on the QDC backend, so wait
    # for it to populate -- otherwise a successful job yields no metrics.
    job_log_files = geniex_job.get_job_log_files(job_id, wait_for_logs=True)

    if not job_log_files:
        reason = (
            f"QDC job {job_id} on device '{hub_device_name}' reported result="
            f"'{job_result}' but produced no retrievable log files"
        )
        print(f"[empty logs] {reason}")
        return [], [], JobOutcome.RETRYABLE_EMPTY_LOGS, reason

    # Archive the logs before parsing, so they survive whatever verdict follows:
    # a 'Successful' job can still be failed below (e.g. no eval output), and its
    # logs are what explains why.
    geniex_job.save_job_logs(job_id, save_logs_dir, job_log_files, label=log_label)

    metrics = (
        geniex_job.compute_metrics(job_log_files, save_results_dir=save_results_dir)
        if run_perf
        else []
    )
    # eval_prompts holds the raw questions so compute_eval_results labels each
    # output with the human-readable prompt (not the templated form).
    eval_results = (
        geniex_job.compute_eval_results(job_log_files, eval_prompts)
        if eval_prompts
        else []
    )
    # A job can report result='Successful' and return log files while carrying
    # no eval output at all: if the device drops off adb mid-run, QDC_logs is
    # never harvested and only the infrastructure logs come back. Treating that
    # as SUCCESS makes it non-retryable, so the run ends as
    # eval_incomplete (0/N) with an attempt still unspent. Classify it as
    # retryable-empty-logs instead, matching the no-log-files case above.
    if eval_prompts and not eval_results:
        reason = (
            f"QDC job {job_id} on device '{hub_device_name}' reported result="
            f"'{job_result}' but returned no eval output for "
            f"{len(eval_prompts)} prompt(s); the device logs were likely never "
            f"harvested"
        )
        print(f"[no eval output] {reason}")
        return metrics, [], JobOutcome.RETRYABLE_EMPTY_LOGS, reason
    return metrics, eval_results, JobOutcome.SUCCESS, None
