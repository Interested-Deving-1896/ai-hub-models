# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for SingleSlotCacheMixin's per-class cache slots.

These mirror the real LLM hierarchy, where a split-forward wrapper derives
from the PreSplit model and both use the same checkpoint string as their
cache key. Slots must be keyed by the exact class, or the derived class's
lookup resolves to the base's instance through the MRO.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from qai_hub_models.models.templates.llm.model import SingleSlotCacheMixin

CHECKPOINT = "org/model-name"


class Base(SingleSlotCacheMixin):
    """Stands in for ``<Model>_PreSplit``."""

    def __init__(self, checkpoint: str = CHECKPOINT) -> None:
        self.model: Any = object()
        self.checkpoint = checkpoint

    @classmethod
    def from_pretrained(cls, checkpoint: str = CHECKPOINT) -> Any:
        cached = cls.cache_lookup(checkpoint)
        if cached is not None:
            return cached
        instance = cls(checkpoint)
        cls.cache_store(instance, checkpoint)
        return instance

    def free_memory(self) -> None:
        self.model = None


class Derived(Base):
    """Stands in for ``FPSplitModelWrapper(SplitForwardMixin, <Model>_PreSplit)``."""


@pytest.fixture(autouse=True)
def clear_slots() -> Generator[None, None, None]:
    SingleSlotCacheMixin._cache_slots.clear()
    yield
    SingleSlotCacheMixin._cache_slots.clear()


def test_derived_lookup_does_not_return_base_instance() -> None:
    """A cached base instance must not satisfy a derived class's lookup."""
    base = Base.from_pretrained()
    derived = Derived.from_pretrained()

    assert type(derived) is Derived
    assert derived is not base
    # The base's own cache is untouched by the derived construction.
    assert Base.from_pretrained() is base


def test_base_lookup_does_not_return_derived_instance() -> None:
    """The reverse order must be safe too."""
    derived = Derived.from_pretrained()
    base = Base.from_pretrained()

    assert type(base) is Base
    assert base is not derived
    assert Derived.from_pretrained() is derived


def test_releasing_derived_does_not_free_base() -> None:
    """Releasing a class that cached nothing must not gut a base's instance."""
    base = Base.from_pretrained()

    Derived.release()

    assert base.model is not None
    assert Base.from_pretrained() is base
    assert Base.from_pretrained().model is not None


def test_releasing_base_does_not_free_derived() -> None:
    derived = Derived.from_pretrained()

    Base.release()

    assert derived.model is not None
    assert Derived.from_pretrained() is derived


def test_release_frees_and_evicts_own_instance() -> None:
    base = Base.from_pretrained()

    Base.release()

    assert base.model is None
    fresh = Base.from_pretrained()
    assert fresh is not base
    assert fresh.model is not None


def test_instance_release_evicts_its_own_class_slot() -> None:
    """``instance.release()`` binds to ``type(instance)``, not a base."""
    base = Base.from_pretrained()
    derived = Derived.from_pretrained()

    derived.release()

    assert derived.model is None
    assert base.model is not None
    assert Base.from_pretrained() is base


def test_cache_miss_on_different_key_evicts_old_instance() -> None:
    first = Base.from_pretrained("org/first")
    second = Base.from_pretrained("org/second")

    assert second is not first
    assert first.model is None
    assert Base.from_pretrained("org/second") is second


def test_cache_store_same_instance_is_idempotent() -> None:
    base = Base.from_pretrained()

    Base.cache_store(base, CHECKPOINT)

    assert base.model is not None
    assert Base.from_pretrained() is base
