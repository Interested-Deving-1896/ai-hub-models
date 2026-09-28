# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest

from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.envvars import SpecialModelSetting
from qai_hub_models.scorecard.results.yaml import ComponentNamesYaml
from qai_hub_models.scorecard.scorecard_config_yaml import (
    LLMWeekendGroup,
    QAIHMModelScorecardConfig,
    get_downloadable_llm_model_ids,
    get_llm_model_ids,
    get_week_model_ids,
    validate_llm_weekend_coverage,
)
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS

# Regional Llama variants: test_split: llm, weekend_group: no_week. Never picked
# up by the week1/week2 rotation; only enabled via the llm_no_week token.
REGIONAL_MODELS = {
    "llama_v3_1_sea_lion_3_5_8b_r",
    "llama_v3_elyza_jp_8b",
    "llama_v3_taide_8b_chat",
}


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_pip_commands_require_global_requirements_incompatible(model_id: str) -> None:
    """If manifest.yaml sets any pre/post pip install commands, the model's
    scorecard-config.yaml must have global_requirements_incompatible: true.
    """
    sc = QAIHMModelScorecardConfig.from_model(model_id)
    manifest = sc.manifest
    if not manifest.pre_pip_install_commands and not manifest.post_pip_install_commands:
        return
    assert sc.global_requirements_incompatible, (
        f"{model_id}: pre_pip_install_commands/post_pip_install_commands is set in "
        "manifest.yaml, but global_requirements_incompatible is not true in "
        "scorecard-config.yaml."
    )


def test_duplicate_standalone_component_labels_rejected() -> None:
    # Two components sharing a label would silently merge in perf.yaml.
    with pytest.raises(ValueError, match="labels must be unique"):
        QAIHMModelScorecardConfig(
            standalone_components={
                "vision_encoder": "Encoder",
                "audio_encoder": "Encoder",
            }
        )


class TestStandaloneComponentsValidation:
    """
    standalone_components lives in scorecard-config.yaml but is checked against
    manifest.yaml, so these cross-file checks run through
    QAIHMModelScorecardConfig.validate_standalone_components rather than a validator.
    """

    @staticmethod
    def _config(standalone_components: dict[str, str]) -> QAIHMModelScorecardConfig:
        return QAIHMModelScorecardConfig(standalone_components=standalone_components)

    @staticmethod
    def _manifest(
        model_type_llm: bool = True, name: str | None = None
    ) -> QAIHMModelManifest:
        return QAIHMModelManifest(model_type_llm=model_type_llm, name=name)

    def test_llm_with_standalone_component_is_valid(self) -> None:
        config = self._config({"vision_encoder": "Vision-Encoder"})
        manifest = self._manifest()
        config.validate_standalone_components(manifest)
        assert config.perf_component_key("vision_encoder", manifest) == "Vision-Encoder"

    def test_directly_constructed_config_has_no_manifest(self) -> None:
        # `manifest` only resolves on from_model configs; a directly-constructed one
        # must be given a manifest rather than silently guessing a model id.
        config = self._config({})
        with pytest.raises(ValueError, match="only available on configs loaded via"):
            _ = config.manifest

    def test_non_llm_rejected(self) -> None:
        config = self._config({"encoder": "Encoder"})
        with pytest.raises(ValueError, match="can only be set on LLM/VLM models"):
            config.validate_standalone_components(self._manifest(model_type_llm=False))

    def test_label_colliding_with_model_name_rejected(self) -> None:
        # `name` keys the consolidated backbone entry, so reusing it would
        # overwrite the end-to-end numbers with a single component's.
        config = self._config({"vision_encoder": "My-VLM"})
        with pytest.raises(ValueError, match="collides with the"):
            config.validate_standalone_components(self._manifest(name="My-VLM"))


def test_from_model_and_manifest_are_cached() -> None:
    """
    add_to_perf asks for the same model's config once per export test, so a repeat
    from_model must not re-parse scorecard-config.yaml or manifest.yaml.
    """
    model_id = sorted(MODEL_IDS)[0]
    first = QAIHMModelScorecardConfig.from_model(model_id)
    assert QAIHMModelScorecardConfig.from_model(model_id) is first
    assert first.manifest is first.manifest


def test_standalone_components_are_real_components() -> None:
    """
    Every standalone_components key must name a component the model actually declares
    in Python, otherwise the scorecard would ask to profile a component that does not
    exist and the tab would silently never appear.
    """
    known_components = ComponentNamesYaml.from_intermediates()
    errors: list[str] = []
    for model_id in MODEL_IDS:
        standalone_components = QAIHMModelScorecardConfig.from_model(
            model_id
        ).standalone_components
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


def test_all_llms_have_weekend_group() -> None:
    """Every scorecard-eligible test_split: llm model must set weekend_group."""
    validate_llm_weekend_coverage()


def test_weeks_are_disjoint_and_cover_all_llms() -> None:
    w1 = get_week_model_ids(LLMWeekendGroup.WEEK1)
    w2 = get_week_model_ids(LLMWeekendGroup.WEEK2)
    no_week = get_week_model_ids(LLMWeekendGroup.NO_WEEK)
    assert w1 and w2 and no_week
    assert w1.isdisjoint(w2)
    assert w1.isdisjoint(no_week)
    assert w2.isdisjoint(no_week)
    assert w1 | w2 | no_week == get_llm_model_ids()


def test_downloadable_is_nonempty_subset() -> None:
    downloadable = get_downloadable_llm_model_ids()
    assert downloadable
    assert downloadable <= get_llm_model_ids()


def test_regional_models_are_no_week() -> None:
    """Regional variants run only on-demand via llm_no_week, never on a schedule."""
    llm_ids = get_llm_model_ids()
    no_week = get_week_model_ids(LLMWeekendGroup.NO_WEEK)
    w1 = get_week_model_ids(LLMWeekendGroup.WEEK1)
    w2 = get_week_model_ids(LLMWeekendGroup.WEEK2)
    for model_id in REGIONAL_MODELS:
        assert model_id in llm_ids
        assert model_id in no_week
        assert model_id not in w1
        assert model_id not in w2


def test_tokens_resolve_to_expected_llm_sets() -> None:
    for token, expected in [
        (SpecialModelSetting.LLM_WEEK1, get_week_model_ids(LLMWeekendGroup.WEEK1)),
        (SpecialModelSetting.LLM_WEEK2, get_week_model_ids(LLMWeekendGroup.WEEK2)),
        (
            SpecialModelSetting.LLM_NO_WEEK,
            get_week_model_ids(LLMWeekendGroup.NO_WEEK),
        ),
        (SpecialModelSetting.LLM_DOWNLOADABLE, get_downloadable_llm_model_ids()),
    ]:
        torch_ids, static_ids = validate_and_split_enabled_models({token})
        assert torch_ids == expected
        assert not static_ids


def test_combined_weekend_tokens_round_trip() -> None:
    """The workflow's 'pytorch_no_llm,static,llm_week1,llm_downloadable' string is valid."""
    tokens: set = {
        SpecialModelSetting.PYTORCH_NO_LLM,
        SpecialModelSetting.STATIC,
        SpecialModelSetting.LLM_WEEK1,
        SpecialModelSetting.LLM_DOWNLOADABLE,
    }
    torch_ids, static_ids = validate_and_split_enabled_models(tokens)
    assert get_week_model_ids(LLMWeekendGroup.WEEK1) <= torch_ids
    assert get_downloadable_llm_model_ids() <= torch_ids
    assert static_ids
