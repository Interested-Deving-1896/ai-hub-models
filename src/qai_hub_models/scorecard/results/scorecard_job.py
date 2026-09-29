# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import datetime
from functools import cached_property
from typing import Any, Generic, TypeVar, cast

import qai_hub as hub
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.scorecard.execution_helpers import wait_for_prerequisite_job
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf

JobTypeVar = TypeVar(
    "JobTypeVar",
    hub.CompileJob,
    hub.LinkJob,
    hub.InferenceJob,
    hub.QuantizeJob,
    hub.ProfileJob,
)
ScorecardJobTypeVar = TypeVar("ScorecardJobTypeVar", bound="ScorecardJob")


"""
The purpose of this class is to fetch & cache all information related to a workbench job.

This allows us to fetch all job information at once, to avoid expensive sequential
workbench API calls when doing analysis on the results.
"""


class ScorecardJob(Generic[JobTypeVar]):
    job_type_class: type[JobTypeVar]
    job_type: hub.JobType

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def wait(self, max_wait_seconds: int | None = None) -> hub.JobStatus:
        s = wait_for_prerequisite_job(self.job, max_wait_seconds)
        self.cache_results()
        return s

    def cache_results(self) -> None:
        # Verify job is the correct type
        if not isinstance(self.job, self.job_type_class):
            raise TypeError(
                f"Job {self.job.job_id}({self.job.name}) is {type(self.job)}. Expected {self.job_type_class.__name__}"
            )

    @cached_property
    def job(self) -> JobTypeVar:
        """
        Get the AI Hub Workbench Job.
        Waits for completion if necessary.
        """
        if not self.job_id:
            raise ValueError("No Job ID")
        return cast(JobTypeVar, hub.get_job(self.job_id))

    @cached_property
    def running(self) -> bool:
        return self._job_status.running

    @cached_property
    def failed(self) -> bool:
        return self._job_status.failure

    @cached_property
    def success(self) -> bool:
        return self._job_status.success

    @cached_property
    def status_message(self) -> str | None:
        return self._job_status.message

    @cached_property
    def _job_status(self) -> hub.JobStatus:
        """Get the job status of the profile job."""
        if self.job_id:
            return self.job.get_status()
        raise ValueError("Can't get status without a job ID.")

    @cached_property
    def job_status(self) -> str:
        """Get the job status of the profile job."""
        if self._job_status.success:
            return "Passed"
        if self._job_status.failure:
            return "Failed"
        return "Skipped"

    @cached_property
    def hub_device(self) -> hub.Device:
        if isinstance(
            self.job, (hub.CompileJob, hub.LinkJob, hub.ProfileJob, hub.InferenceJob)
        ):
            return self.job.device
        raise NotImplementedError("Device is not applicable for this job type")

    @cached_property
    def tool_versions(self) -> ToolVersions:
        return ToolVersions.from_job(self.job, parse_version_tags=True)

    @cached_property
    def date(self) -> datetime.datetime | None:
        if self.job is None:
            return None
        return self.job.date


class QuantizeScorecardJob(ScorecardJob[hub.QuantizeJob]):
    job_type_class = hub.QuantizeJob
    job_type = hub.JobType.QUANTIZE


class CompileScorecardJob(ScorecardJob[hub.CompileJob]):
    job_type_class = hub.CompileJob
    job_type = hub.JobType.COMPILE


class LinkScorecardJob(ScorecardJob[hub.LinkJob]):
    job_type_class = hub.LinkJob
    job_type = hub.JobType.LINK


class ProfileScorecardJob(ScorecardJob[hub.ProfileJob]):
    job_type_class = hub.ProfileJob
    job_type = hub.JobType.PROFILE

    def cache_results(self) -> None:
        super().cache_results()
        if self._job_status.success:
            assert self.profile_results  # Download results immediately

    @cached_property
    def profile_results(self) -> dict[str, Any]:
        """Profile results from profile job."""
        if self.success:
            profile = self.job.download_profile()
            assert isinstance(profile, dict)
            return profile
        raise ValueError("Can't get profile results if job did not succeed.")

    @cached_property
    def inference_time_milliseconds(self) -> float:
        """Get the inference time from the profile job."""
        return float(
            self.profile_results["execution_summary"]["estimated_inference_time"] / 1000
        )

    @cached_property
    def first_load_time_milliseconds(self) -> float:
        """Get the first load time from the profile job."""
        return float(
            self.profile_results["execution_summary"]["first_load_time"] / 1000
        )

    @cached_property
    def warm_load_time_milliseconds(self) -> float:
        """Get the warm load time from the profile job."""
        return float(self.profile_results["execution_summary"]["warm_load_time"] / 1000)

    @cached_property
    def layer_counts(self) -> QAIHMModelPerf.PerformanceDetails.LayerCounts:
        """Count layers per compute unit."""

        def _count_unit(unit: str) -> int:
            return sum(
                1
                for detail in self.profile_results["execution_detail"]
                if detail["compute_unit"] == unit
            )

        cpu = _count_unit("CPU")
        gpu = _count_unit("GPU")
        npu = _count_unit("NPU")

        return QAIHMModelPerf.PerformanceDetails.LayerCounts.from_layers(npu, gpu, cpu)

    @cached_property
    def estimated_peak_memory_range_mb(
        self,
    ) -> QAIHMModelPerf.PerformanceDetails.PeakMemoryRangeMB:
        """Get the estimated peak memory range."""
        low, high = self.profile_results["execution_summary"][
            "inference_memory_peak_range"
        ]
        return QAIHMModelPerf.PerformanceDetails.PeakMemoryRangeMB.from_bytes(low, high)

    @cached_property
    def performance_metrics(self) -> QAIHMModelPerf.PerformanceDetails:
        return QAIHMModelPerf.PerformanceDetails(
            job_id=self.job_id,
            job_status=self.job_status,
            inference_time_milliseconds=(
                self.inference_time_milliseconds if self.success else None
            ),
            estimated_peak_memory_range_mb=(
                self.estimated_peak_memory_range_mb if self.success else None
            ),
            primary_compute_unit=(
                self.layer_counts.primary_compute_unit if self.success else None
            ),
            layer_counts=self.layer_counts if self.success else None,
            tool_versions=self.tool_versions,
        )


class InferenceScorecardJob(ScorecardJob[hub.InferenceJob]):
    job_type_class = hub.InferenceJob
    job_type = hub.JobType.INFERENCE

    @property
    def input_dataset(self) -> DatasetEntries:
        """Input dataset."""
        if self.job.inputs is None:
            raise ValueError("Job has no input dataset.")
        return cast(DatasetEntries, self.job.inputs.download())

    @property
    def output_dataset(self) -> DatasetEntries:
        """Output dataset."""
        if not self.success:
            raise ValueError("Can't get output dataset if job did not succeed.")
        return cast(DatasetEntries, self.job.download_output_data())
