# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for the dataset split validation utility in ``utils.validation``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit
from qai_hub_models.utils.private_asset_loaders import UnfetchableDatasetError
from qai_hub_models.utils.validation import (
    DatasetCheck,
    DatasetCheckOutcome,
    DatasetRoles,
    _values_match,
    collect_dataset_roles,
    perform_dataset_split_validation,
    validate_dataset_splits,
)


class _FakeSplitDataset(BaseDataset):
    """Well-behaved dataset: each split serves its own data."""

    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        BaseDataset.__init__(self, Path("."), split=split)
        self.items = [torch.tensor([float(split.value)])]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Any:
        return self.items[idx], self.split.value

    def _download_data(self) -> None:
        raise AssertionError("test dataset should never download")

    @staticmethod
    def default_samples_per_job() -> int:
        return 1


class _FakeLeakyDataset(_FakeSplitDataset):
    """Broken dataset: ignores the split and always serves val data."""

    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        super().__init__(split=split)
        self.items = [torch.tensor([float(DatasetSplit.VAL.value)])]

    def __getitem__(self, idx: int) -> Any:
        return self.items[idx], DatasetSplit.VAL.value


class _FakeUnstableDataset(_FakeSplitDataset):
    """Broken dataset: item 0 changes on every from-scratch instantiation."""

    _loads = 0

    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        super().__init__(split=split)
        type(self)._loads += 1
        self.items = [torch.tensor([float(self._loads)])]


class _FakeValOnlyDataset(_FakeSplitDataset):
    """Dataset that correctly refuses TRAIN rather than aliasing VAL."""

    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        if split is DatasetSplit.TRAIN:
            raise ValueError("this dataset is eval-only")
        super().__init__(split=split)


class _FakeTrainOnlyDataset(_FakeSplitDataset):
    """Dataset that correctly refuses VAL rather than aliasing TRAIN."""

    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        if split is DatasetSplit.VAL:
            raise ValueError("this dataset is calibration-only")
        super().__init__(split=split)


def _model_with_datasets(
    eval_classes: list[type[BaseDataset]] | None = None,
    calib_class: type[BaseDataset] | None = None,
) -> Any:
    return SimpleNamespace(
        get_eval_dataset_classes=lambda: eval_classes or [],
        get_calibration_dataset_cls=lambda: calib_class,
    )


class TestValuesMatch:
    def test_identical_tensors_match(self) -> None:
        assert _values_match(torch.zeros(2), torch.zeros(2)) is True

    def test_different_tensors_do_not_match(self) -> None:
        assert _values_match(torch.zeros(2), torch.ones(2)) is False

    def test_different_shapes_do_not_match(self) -> None:
        assert _values_match(torch.zeros(2), torch.zeros(3)) is False

    def test_tuples_recurse(self) -> None:
        assert _values_match((torch.zeros(2), 1), (torch.zeros(2), 1)) is True
        assert _values_match((torch.zeros(2), 1), (torch.zeros(2), 2)) is False

    def test_mappings_recurse(self) -> None:
        assert _values_match({"a": torch.zeros(1)}, {"a": torch.zeros(1)}) is True
        assert _values_match({"a": torch.zeros(1)}, {"b": torch.zeros(1)}) is False

    def test_numpy_arrays(self) -> None:
        assert _values_match(np.zeros(2), np.zeros(2)) is True
        assert _values_match(np.zeros(2), np.ones(2)) is False

    def test_opaque_objects_are_uncomparable(self) -> None:
        assert _values_match(object(), object()) is None

    def test_mismatch_beats_uncomparable(self) -> None:
        assert _values_match((object(), 1), (object(), 2)) is False


class TestDatasetRoles:
    def test_eval_requires_val(self) -> None:
        roles = DatasetRoles(is_eval=True, is_calibration=False)
        assert roles.required_splits == (DatasetSplit.VAL,)
        assert roles.label == "eval"

    def test_calibration_requires_train(self) -> None:
        roles = DatasetRoles(is_eval=False, is_calibration=True)
        assert roles.required_splits == (DatasetSplit.TRAIN,)
        assert roles.label == "calibration"

    def test_both_roles_require_both_splits(self) -> None:
        roles = DatasetRoles(is_eval=True, is_calibration=True)
        assert roles.required_splits == (DatasetSplit.VAL, DatasetSplit.TRAIN)
        assert roles.label == "eval+calibration"


class TestCollectDatasetRoles:
    def test_separate_classes_get_separate_roles(self) -> None:
        model = _model_with_datasets([_FakeSplitDataset], _FakeLeakyDataset)
        assert collect_dataset_roles(model) == {
            _FakeSplitDataset: DatasetRoles(True, False),
            _FakeLeakyDataset: DatasetRoles(False, True),
        }

    def test_shared_class_gets_both_roles(self) -> None:
        model = _model_with_datasets([_FakeSplitDataset], _FakeSplitDataset)
        assert collect_dataset_roles(model) == {
            _FakeSplitDataset: DatasetRoles(True, True)
        }

    def test_walks_collection_components(self) -> None:
        model = SimpleNamespace(
            components={"a": _model_with_datasets(calib_class=_FakeSplitDataset)},
            get_eval_dataset_classes=list,
            get_calibration_dataset_cls=lambda: None,
        )
        assert collect_dataset_roles(model) == {
            _FakeSplitDataset: DatasetRoles(False, True)
        }

    def test_ignores_non_dataset_values(self) -> None:
        model = _model_with_datasets([cast(Any, str)], None)
        assert collect_dataset_roles(model) == {}


class TestValidateDatasetSplits:
    def _outcomes(self, model: Any) -> dict[DatasetCheck, DatasetCheckOutcome]:
        return {r.check: r.outcome for r in validate_dataset_splits(model)}

    def test_healthy_eval_dataset_passes_all(self) -> None:
        outcomes = self._outcomes(_model_with_datasets([_FakeSplitDataset]))
        assert outcomes == {
            DatasetCheck.REQUIRED_SPLIT: DatasetCheckOutcome.PASS,
            DatasetCheck.DISTINCT_SPLITS: DatasetCheckOutcome.PASS,
            DatasetCheck.VAL_REPRODUCIBLE: DatasetCheckOutcome.PASS,
        }

    def test_calibration_only_skips_repro_check(self) -> None:
        # Reproducibility only matters for the eval split.
        outcomes = self._outcomes(_model_with_datasets(calib_class=_FakeSplitDataset))
        assert DatasetCheck.VAL_REPRODUCIBLE not in outcomes
        assert outcomes[DatasetCheck.DISTINCT_SPLITS] is DatasetCheckOutcome.PASS

    def test_aliased_splits_fail(self) -> None:
        outcomes = self._outcomes(_model_with_datasets([_FakeLeakyDataset]))
        assert outcomes[DatasetCheck.REQUIRED_SPLIT] is DatasetCheckOutcome.PASS
        assert outcomes[DatasetCheck.DISTINCT_SPLITS] is DatasetCheckOutcome.FAIL
        assert outcomes[DatasetCheck.VAL_REPRODUCIBLE] is DatasetCheckOutcome.PASS

    def test_aliased_splits_fail_for_calibration_role_too(self) -> None:
        outcomes = self._outcomes(_model_with_datasets(calib_class=_FakeLeakyDataset))
        assert outcomes[DatasetCheck.DISTINCT_SPLITS] is DatasetCheckOutcome.FAIL

    def test_unstable_val_fails_repro_only(self) -> None:
        outcomes = self._outcomes(_model_with_datasets([_FakeUnstableDataset]))
        assert outcomes[DatasetCheck.DISTINCT_SPLITS] is DatasetCheckOutcome.PASS
        assert outcomes[DatasetCheck.VAL_REPRODUCIBLE] is DatasetCheckOutcome.FAIL

    def test_val_only_dataset_as_eval_passes(self) -> None:
        # Raising on TRAIN is the correct way to be eval-only.
        outcomes = self._outcomes(_model_with_datasets([_FakeValOnlyDataset]))
        assert outcomes == {
            DatasetCheck.REQUIRED_SPLIT: DatasetCheckOutcome.PASS,
            DatasetCheck.DISTINCT_SPLITS: DatasetCheckOutcome.PASS,
            DatasetCheck.VAL_REPRODUCIBLE: DatasetCheckOutcome.PASS,
        }

    def test_val_only_dataset_as_calibration_fails_required(self) -> None:
        # Calibration requests TRAIN, which this dataset refuses to serve.
        results = validate_dataset_splits(
            _model_with_datasets(calib_class=_FakeValOnlyDataset)
        )
        required = results[0]
        assert required.check is DatasetCheck.REQUIRED_SPLIT
        assert required.outcome is DatasetCheckOutcome.FAIL
        assert "role=calibration requires train" in required.detail

    def test_train_only_dataset_as_calibration_passes(self) -> None:
        outcomes = self._outcomes(
            _model_with_datasets(calib_class=_FakeTrainOnlyDataset)
        )
        assert outcomes[DatasetCheck.REQUIRED_SPLIT] is DatasetCheckOutcome.PASS
        assert outcomes[DatasetCheck.DISTINCT_SPLITS] is DatasetCheckOutcome.PASS

    def test_train_only_dataset_as_eval_fails_required(self) -> None:
        results = validate_dataset_splits(_model_with_datasets([_FakeTrainOnlyDataset]))
        assert results[0].outcome is DatasetCheckOutcome.FAIL
        assert "role=eval requires val" in results[0].detail

    def test_dual_role_dataset_requires_both_splits(self) -> None:
        results = validate_dataset_splits(
            _model_with_datasets([_FakeValOnlyDataset], _FakeValOnlyDataset)
        )
        assert results[0].outcome is DatasetCheckOutcome.FAIL
        assert "role=eval+calibration requires val, train" in results[0].detail

    def test_unfetchable_dataset_is_unavailable(self) -> None:
        def _raise(*_a: Any, **_kw: Any) -> None:
            raise UnfetchableDatasetError("secret_data", None)

        with patch("qai_hub_models.utils.validation.instantiate_dataset", _raise):
            outcomes = self._outcomes(_model_with_datasets([_FakeSplitDataset]))
        assert set(outcomes.values()) == {DatasetCheckOutcome.UNAVAILABLE}

    def test_no_datasets_returns_empty(self) -> None:
        assert validate_dataset_splits(_model_with_datasets()) == []


class TestPerformDatasetSplitValidation:
    def _model_cls(self, model: Any) -> Any:
        return SimpleNamespace(from_pretrained=lambda: model)

    def test_raises_on_failure(self) -> None:
        model_cls = self._model_cls(_model_with_datasets([_FakeLeakyDataset]))
        with (
            patch("qai_hub_models.utils.validation.issubclass", return_value=True),
            pytest.raises(AssertionError, match="identical first item"),
        ):
            perform_dataset_split_validation(model_cls)

    def test_passes_when_clean(self) -> None:
        model_cls = self._model_cls(_model_with_datasets([_FakeSplitDataset]))
        with patch("qai_hub_models.utils.validation.issubclass", return_value=True):
            perform_dataset_split_validation(model_cls)

    def test_passes_when_no_datasets(self) -> None:
        model_cls = self._model_cls(_model_with_datasets())
        with patch("qai_hub_models.utils.validation.issubclass", return_value=True):
            perform_dataset_split_validation(model_cls)

    def test_unavailable_does_not_raise(self) -> None:
        def _raise(*_a: Any, **_kw: Any) -> None:
            raise UnfetchableDatasetError("secret_data", None)

        model_cls = self._model_cls(_model_with_datasets([_FakeSplitDataset]))
        with (
            patch("qai_hub_models.utils.validation.instantiate_dataset", _raise),
            patch("qai_hub_models.utils.validation.issubclass", return_value=True),
        ):
            perform_dataset_split_validation(model_cls)
