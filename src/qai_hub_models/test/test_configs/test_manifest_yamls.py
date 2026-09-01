# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs._info_yaml_enums import (
    MODEL_DOMAIN_USE_CASES,
    MODEL_STATUS,
)
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION
from qai_hub_models.configs.manifest_yaml import (
    MODEL_DOMAIN,
    MODEL_USE_CASE,
    LMQuantizationDetails,
    QAIHMModelManifest,
    TechnicalDetails,
)
from qai_hub_models.models.templates.lm_schema import Recipe
from qai_hub_models.scorecard.results.yaml import ComponentNamesYaml
from qai_hub_models.scorecard.scorecard_config_yaml import QAIHMModelScorecardConfig
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, QAIHM_WEB_ASSET
from qai_hub_models.utils.device import CANARY_DEVICES
from qai_hub_models.utils.metrics import VALID_METRIC_PAIRS
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_MODELS_ROOT
from qai_hub_models.utils.url_check import validate_urls_exist

HF_PIPELINE_TAGS = {
    "keypoint-detection",
    "text-classification",
    "token-classification",
    "table-question-answering",
    "question-answering",
    "zero-shot-classification",
    "translation",
    "summarization",
    "conversational",
    "feature-extraction",
    "text-generation",
    "text2text-generation",
    "fill-mask",
    "sentence-similarity",
    "text-to-speech",
    "text-to-audio",
    "automatic-speech-recognition",
    "audio-to-audio",
    "audio-classification",
    "voice-activity-detection",
    "depth-estimation",
    "gaze-estimation",
    "image-classification",
    "object-detection",
    "image-segmentation",
    "text-to-image",
    "image-to-text",
    "image-to-image",
    "image-to-video",
    "unconditional-image-generation",
    "video-classification",
    "reinforcement-learning",
    "robotics",
    "tabular-classification",
    "tabular-regression",
    "tabular-to-text",
    "table-to-text",
    "multiple-choice",
    "text-retrieval",
    "time-series-forecasting",
    "text-to-video",
    "visual-question-answering",
    "document-question-answering",
    "zero-shot-image-classification",
    "graph-ml",
    "mask-generation",
    "zero-shot-object-detection",
    "text-to-3d",
    "image-to-3d",
    "video-object-tracking",
    "other",
}


def test_all_domains_accounted_for() -> None:
    # Verify all use cases and domains are accounted for in the mapping
    assert len(MODEL_DOMAIN_USE_CASES) == len(MODEL_DOMAIN)
    use_cases = {
        ucase for ucases in MODEL_DOMAIN_USE_CASES.values() for ucase in ucases
    }
    assert len(use_cases) == len(MODEL_USE_CASE)


def test_model_usecase_to_hf_pipeline_tag() -> None:
    for use_case in MODEL_USE_CASE:
        assert use_case.map_to_hf_pipeline_tag() in HF_PIPELINE_TAGS


def _check_website_facing(manifest: QAIHMModelManifest) -> None:
    """Enforce website / scorecard / publication rules for an in-tree recipe.

    These checks live here, not in ``QAIHMModelManifest.check_fields``, so
    that ``check_fields`` remains a universal-correctness validator that
    external recipes can pass without pulling in Qualcomm-catalog rules
    (banner presence, canary devices, headline copy style, etc.).
    """
    assert manifest.id is not None
    assert manifest.name is not None
    assert manifest.headline is not None

    if " " in manifest.name:
        raise ValueError("Model Name must not have a space.")
    if "_" in manifest.name:
        raise ValueError("Model Name should use dashes (-) instead of underscores.")

    if not manifest.headline.endswith("."):
        raise ValueError("Model headlines must end with a period.")

    for r_model in manifest.related_models:
        if r_model == manifest.id:
            raise ValueError(f"Model {r_model} cannot be related to itself.")

    if (
        manifest.research_paper is not None
        and manifest.research_paper.startswith("https://arxiv.org/")
        and "/abs/" not in manifest.research_paper
    ):
        raise ValueError(
            "Arxiv links should be `abs` links, not link directly to pdfs."
        )

    if manifest.status == MODEL_STATUS.PUBLISHED:
        can_be_published, reason = manifest.can_promote_to_published()
        if not can_be_published:
            raise ValueError(f"Model cannot be published: {reason}")

    if manifest.status == MODEL_STATUS.UNPUBLISHED and not manifest.status_reason:
        raise ValueError(
            "Unpublished models must set `status_reason` in manifest.yaml with "
            "a link to the related issue."
        )
    if manifest.status == MODEL_STATUS.PUBLISHED and manifest.status_reason:
        raise ValueError(
            "`status_reason` in manifest.yaml should not be set for published models."
        )

    if manifest.numerics_benchmark is not None:
        pair = (
            manifest.numerics_benchmark.metric_name,
            manifest.numerics_benchmark.unit,
        )
        if pair not in VALID_METRIC_PAIRS:
            valid_pairs_str = ", ".join(
                f"({n!r}, {u!r})" for n, u in sorted(VALID_METRIC_PAIRS)
            )
            raise ValueError(
                f"numerics_benchmark metric_name={pair[0]!r} with "
                f"unit={pair[1]!r} does not match any known metric. "
                f"Valid pairs:\n  {valid_pairs_str}"
            )

    if manifest.status == MODEL_STATUS.PUBLISHED:
        if not os.path.exists(manifest.get_package_path() / "manifest.yaml"):
            raise ValueError("All published models must have a manifest.yaml")
        if not os.path.exists(manifest.get_package_path() / "perf.yaml"):
            raise ValueError("All published models must have a perf.yaml")
        if not manifest.supports_at_least_1_runtime:
            raise ValueError("Published models must support at least one export path")
        if not manifest.has_static_banner:
            raise ValueError("Published models must have a static asset.")

    expected_qaihm_repo = Path("src") / "qai_hub_models" / "models" / manifest.id
    if expected_qaihm_repo != ASSET_CONFIG.get_qaihm_repo(manifest.id):
        raise ValueError("QAIHM repo not pointing to expected relative path")

    if manifest.model_type_llm and manifest.llm_details is not None:
        if manifest.llm_details.call_to_action in {
            LLM_CALL_TO_ACTION.DOWNLOAD,
            LLM_CALL_TO_ACTION.DOWNLOAD_AND_VIEW_README,
        }:
            if manifest.restrict_model_sharing:
                raise ValueError(
                    "LLM call to action cannot be 'download' when restrict "
                    "model sharing is enabled."
                )
        elif not manifest.restrict_model_sharing and os.path.exists(
            QAIHM_MODELS_ROOT / manifest.id / "release-assets.yaml"
        ):
            raise ValueError(
                "LLM has downloadable assets but the call to action is not 'download'."
            )

    if manifest.default_device not in CANARY_DEVICES:
        raise ValueError(
            f"Default device must be any of these canary devices: {CANARY_DEVICES}"
        )


def _collect_website_urls(manifest: QAIHMModelManifest) -> list[tuple[str, str]]:
    """Build the URL HEAD-check list for an in-tree recipe."""
    assert manifest.id is not None
    urls: list[tuple[str, str]] = []
    if manifest.has_static_banner:
        urls.append(
            (
                ASSET_CONFIG.get_web_asset_url(manifest.id, QAIHM_WEB_ASSET.STATIC_IMG),
                "Static banner does not exist",
            )
        )
    if manifest.has_animated_banner:
        urls.append(
            (
                ASSET_CONFIG.get_web_asset_url(
                    manifest.id, QAIHM_WEB_ASSET.ANIMATED_MOV
                ),
                "Animated banner does not exist",
            )
        )
    if manifest.license:
        urls.append((manifest.license, "License does not exist"))
    if manifest.research_paper:
        urls.append((manifest.research_paper, "Research paper does not exist"))
    if manifest.source_repo:
        urls.append((manifest.source_repo, "Source repo does not exist"))

    if (
        manifest.model_type_llm
        and manifest.llm_details
        and manifest.llm_details.devices
    ):
        for device_runtime_config_mapping in manifest.llm_details.devices.values():
            for runtime_detail in device_runtime_config_mapping.values():
                if runtime_detail.model_download_url.startswith(
                    ("http://", "https://")
                ):
                    url = runtime_detail.model_download_url
                else:
                    version = runtime_detail.model_download_url.split("/")[0][1:]
                    relative_path = "/".join(
                        runtime_detail.model_download_url.split("/")[1:]
                    )
                    url = ASSET_CONFIG.get_model_asset_url(
                        manifest.id, version, relative_path
                    )
                urls.append(
                    (
                        url,
                        f"Download URL does not exist ({runtime_detail.model_download_url})",
                    )
                )
    return urls


def _validate_model(model_id: str) -> None:
    manifest = QAIHMModelManifest.from_model(model_id)
    QAIHMModelManifest.model_validate(manifest)
    assert manifest.id == model_id, (
        f"{model_id} config ID does not match the model's folder name"
    )
    assert manifest.status is not MODEL_STATUS.UNSET, (
        f"{model_id}: in-tree models must set `status` in manifest.yaml to one "
        f"of {[s.value for s in MODEL_STATUS if s is not MODEL_STATUS.UNSET]}. "
        "The `unset` default only exists for external / standalone recipes "
        "authored outside the qai_hub_models package."
    )
    _check_website_facing(manifest)
    validate_urls_exist(_collect_website_urls(manifest))
    manifest.check_geniex_runtime_technical_details()


def test_manifest_yaml() -> None:
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_validate_model, model_id): model_id for model_id in MODEL_IDS
        }
        for future in as_completed(futures):
            model_id = futures[future]
            try:
                future.result()
            except Exception as err:
                errors.append(f"{model_id}: {err!s}")
    assert not errors, f"{len(errors)} model(s) failed validation:\n" + "\n".join(
        errors
    )


def test_export_paths_include_aot_on_jit() -> None:
    """
    JIT models (`requires_aot_prepare=False`) can still be exported to AOT
    runtimes via the JIT-compile + link path. `get_supported_paths_for_export`
    must include those AOT runtimes even though
    `get_supported_paths_for_testing` filters them out.
    """
    manifest = QAIHMModelManifest.from_model("vit")
    assert not manifest.requires_aot_prepare

    testing_paths = manifest.get_supported_paths_for_testing()
    export_paths = manifest.get_supported_paths_for_export()

    for precision, runtimes in testing_paths.items():
        assert precision in export_paths
        assert set(runtimes).issubset(set(export_paths[precision]))

    assert any(TargetRuntime.QNN_CONTEXT_BINARY in rts for rts in export_paths.values())
    assert not any(
        TargetRuntime.QNN_CONTEXT_BINARY in rts for rts in testing_paths.values()
    )


# ---------------------------------------------------------------------------
# lm_quantization_details validation (QAIHMModelManifest.check_fields):
#   1. the section is LLM/VLM-only (requires model_type_llm)
#   2. every recipe key must be in supported_precisions
# These are template-shaped manifests (id=None) so the website-facing
# checks are skipped and only the build/export invariants run.
# ---------------------------------------------------------------------------
def _lm_details() -> LMQuantizationDetails:
    """A minimal valid recipe (default W4A16 precision + a single Calibration)."""
    return LMQuantizationDetails(
        recipe=Recipe.model_validate([{"name": "Calibration"}])
    )


class TestLMQuantizationDetailsValidation:
    def test_llm_with_supported_precision_is_valid(self) -> None:
        manifest = QAIHMModelManifest(
            model_type_llm=True,
            supported_precisions=[Precision.w4a16],
            lm_quantization_details={Precision.w4a16: _lm_details()},
        )
        assert manifest.lm_quantization_details[Precision.w4a16].recipe.backbone

    def test_non_llm_with_recipe_rejected(self) -> None:
        with pytest.raises(ValueError, match="can only be set on LLM/VLM models"):
            QAIHMModelManifest(
                model_type_llm=False,
                supported_precisions=[Precision.w4a16],
                lm_quantization_details={Precision.w4a16: _lm_details()},
            )

    def test_recipe_for_unsupported_precision_rejected(self) -> None:
        # A recipe keyed by a precision absent from supported_precisions is
        # unreachable, so it must fail loud (and name the offending precision).
        with pytest.raises(ValueError, match="are not in"):
            QAIHMModelManifest(
                model_type_llm=True,
                supported_precisions=[Precision.w4],
                lm_quantization_details={Precision.w4a16: _lm_details()},
            )

    def test_empty_recipe_section_is_valid_for_non_llm(self) -> None:
        # An empty section must not trip the LLM-only guard (the guard is on a
        # non-empty section only), so ordinary non-LLM manifests still validate.
        manifest = QAIHMModelManifest(
            model_type_llm=False,
            supported_precisions=[Precision.float],
        )
        assert manifest.lm_quantization_details == {}


class TestStandaloneComponentsValidation:
    """
    standalone_components lives in scorecard-config.yaml, so these cross-file checks run
    through QAIHMModelManifest.validate_standalone_components rather than a validator.
    """

    @staticmethod
    def _manifest(
        standalone_components: dict[str, str],
        model_type_llm: bool = True,
        name: str | None = None,
    ) -> QAIHMModelManifest:
        manifest = QAIHMModelManifest(model_type_llm=model_type_llm, name=name)
        manifest.scorecard_config = QAIHMModelScorecardConfig(
            standalone_components=standalone_components
        )
        return manifest

    def test_llm_with_standalone_component_is_valid(self) -> None:
        manifest = self._manifest({"vision_encoder": "Vision-Encoder"})
        manifest.validate_standalone_components()
        assert manifest.perf_component_key("vision_encoder") == "Vision-Encoder"

    def test_non_llm_rejected(self) -> None:
        manifest = self._manifest({"encoder": "Encoder"}, model_type_llm=False)
        with pytest.raises(ValueError, match="can only be set on LLM/VLM models"):
            manifest.validate_standalone_components()

    def test_label_colliding_with_model_name_rejected(self) -> None:
        # `name` keys the consolidated backbone entry, so reusing it would
        # overwrite the end-to-end numbers with a single component's.
        manifest = self._manifest({"vision_encoder": "My-VLM"}, name="My-VLM")
        with pytest.raises(ValueError, match="collides with the model name"):
            manifest.validate_standalone_components()


def test_standalone_components_are_real_components() -> None:
    """
    Every standalone_components key must name a component the model actually declares
    in Python, otherwise the scorecard would ask to profile a component that does not
    exist and the tab would silently never appear.
    """
    known_components = ComponentNamesYaml.from_intermediates()
    errors: list[str] = []
    for model_id in MODEL_IDS:
        manifest = QAIHMModelManifest.from_model(model_id)
        standalone_components = manifest.scorecard_config.standalone_components
        if not standalone_components:
            continue
        components = known_components.get(model_id)
        if components is None:
            errors.append(
                f"{model_id}: declares standalone_components but has no recorded "
                f"component names; it must be a collection model."
            )
            continue
        unknown = set(standalone_components) - set(components)
        if unknown:
            errors.append(
                f"{model_id}: standalone_components {sorted(unknown)} are not "
                f"components of this model (has {sorted(components)})."
            )
    assert not errors, "\n".join(errors)


# ---------------------------------------------------------------------------
# check_geniex_runtime_technical_details, model_infer_id half: model_infer_id
# points the website at a HuggingFace repo to download, so it is required
# exactly when we may redistribute weights (restrict_model_sharing=False).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "runtime", [TargetRuntime.GENIEX_QAIRT, TargetRuntime.GENIEX_LLAMACPP]
)
@pytest.mark.parametrize(
    ("restricted", "infer_id", "expected_error"),
    [
        (False, "ai-hub-models/Test-Model", None),
        (False, None, "must define model_infer_id"),
        (True, "ai-hub-models/Test-Model", "must not define model_infer_id"),
        (True, None, None),
    ],
)
def test_geniex_model_infer_id(
    runtime: TargetRuntime,
    restricted: bool,
    infer_id: str | None,
    expected_error: str | None,
) -> None:
    details: TechnicalDetails = {"Supported Input Modalities": "text"}
    if infer_id is not None:
        details["model_infer_id"] = infer_id
    manifest = QAIHMModelManifest(
        id="test_model",
        restrict_model_sharing=restricted,
        runtime_technical_details={runtime: details},
    )

    if expected_error is None:
        manifest.check_geniex_runtime_technical_details()
    else:
        with pytest.raises(ValueError, match=expected_error):
            manifest.check_geniex_runtime_technical_details()


def test_geniex_model_infer_id_skipped_without_geniex_details() -> None:
    # Non-GenieX models have no GenieX section; don't force one into existence.
    QAIHMModelManifest(id="test_model").check_geniex_runtime_technical_details()


def test_geniex_runtime_technical_details_reports_all_errors() -> None:
    # A manifest that trips both halves must surface both, not just the first.
    manifest = QAIHMModelManifest(
        id="test_model",
        restrict_model_sharing=False,
        orchestrator_runtimes=[TargetRuntime.GENIEX_LLAMACPP],
        runtime_technical_details={TargetRuntime.GENIEX_QAIRT: {}},
    )
    with pytest.raises(ValueError, match="geniex_llamacpp") as excinfo:
        manifest.check_geniex_runtime_technical_details()
    assert "must define model_infer_id" in str(excinfo.value)
