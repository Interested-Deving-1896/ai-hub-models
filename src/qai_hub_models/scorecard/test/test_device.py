# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.scorecard.device import (
    ScorecardDevice,
    compute_peer_chipsets,
    cs_universal,
)


def test_exactly_one_default() -> None:
    num_defaults = 0
    for device in ScorecardDevice._registry.values():
        if device.is_default:
            num_defaults += 1
    assert num_defaults == 1, (
        f"Must be exactly one default device, found {num_defaults}"
    )


def test_get_universal_reference_returns_default() -> None:
    """
    Test that querying by cs_universal's reference device name returns
    the default device, not the universal device itself.
    """
    result = ScorecardDevice.get(cs_universal.reference_device_name)
    assert result.is_default, (
        f"Expected default device, got {result.name} (is_default={result.is_default})"
    )
    assert result.name != "universal", (
        f"Expected default device, not universal. Got {result.name}"
    )


def test_compute_peer_chipsets_pairs_x_elite_and_x_plus() -> None:
    """X Plus 8-Core is the same NPU as X Elite; we only ever measure X Elite."""
    pair = {"qualcomm-snapdragon-x-elite", "qualcomm-snapdragon-x-plus-8-core"}
    assert compute_peer_chipsets("qualcomm-snapdragon-x-elite") == pair
    assert compute_peer_chipsets("qualcomm-snapdragon-x-plus-8-core") == pair


def test_compute_peer_chipsets_leaves_others_alone() -> None:
    """X2 Elite is a different NPU (htp 81), and mobile chipsets never pair here."""
    for chipset in ("qualcomm-snapdragon-x2-elite", "qualcomm-snapdragon-8-elite"):
        assert compute_peer_chipsets(chipset) == {chipset}


def test_extended_supported_chipsets_unchanged_for_mobile() -> None:
    """The compute pairing must not leak into the mobile proxy list."""
    assert ScorecardDevice.get("cs_8_elite").extended_supported_chipsets == {
        "qualcomm-snapdragon-8-elite-for-galaxy",
        "qualcomm-snapdragon-8gen3",
        "qualcomm-snapdragon-8gen2",
        "qualcomm-snapdragon-8gen1",
        "qualcomm-snapdragon-888",
    }


def test_extended_supported_chipsets_pairs_compute() -> None:
    pair = {"qualcomm-snapdragon-x-elite", "qualcomm-snapdragon-x-plus-8-core"}
    assert ScorecardDevice.get("cs_x_elite").extended_supported_chipsets == pair
    assert ScorecardDevice.get("cs_x_plus_8_core").extended_supported_chipsets == pair
    assert ScorecardDevice.get("cs_x2_elite").extended_supported_chipsets == {
        "qualcomm-snapdragon-x2-elite"
    }
