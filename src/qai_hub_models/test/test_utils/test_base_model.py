# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import Any

from typing_extensions import Self

from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec


class SimpleBaseModel(BaseModel):
    def get_input_spec(*args: Any, **kwargs: Any) -> InputSpec:
        return {}

    def get_output_spec(*args: Any, **kwargs: Any) -> OutputSpec:
        return {}

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls()


def test_collection_model_demo() -> None:
    """Demo on how to use WorkbenchModelCollection"""

    class Component1(SimpleBaseModel):
        pass

    class Component2(SimpleBaseModel):
        pass

    class DummyCollection(WorkbenchModelCollection):
        def __init__(self, component_1: Component1, component_2: Component2) -> None:
            super().__init__({"component_1": component_1, "component_2": component_2})

        @classmethod
        def from_pretrained(cls) -> "DummyCollection":
            return cls(Component1.from_pretrained(), Component2.from_pretrained())

    model = DummyCollection.from_pretrained()

    assert model.component_names == ["component_1", "component_2"]
    assert list(model.components.keys()) == ["component_1", "component_2"]
    assert isinstance(model.components["component_1"], Component1)
    assert isinstance(model.components["component_2"], Component2)

    comp1_instance = Component1()
    comp2_instance = Component2()
    model2 = DummyCollection(comp1_instance, comp2_instance)
    assert model2.components["component_1"] is comp1_instance
    assert model2.components["component_2"] is comp2_instance
