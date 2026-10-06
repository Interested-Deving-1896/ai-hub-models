# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import shutil
from enum import Enum
from pathlib import Path

import numpy as np
import onnx
import torch
from numpy.linalg import norm

from qai_hub_models.models.grootn15.constants import MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.evaluate.helpers import EvalMode
from qai_hub_models.utils.input_spec import InputSpec

ComponentArrays = dict[str, np.ndarray]


class DataSplit(str, Enum):
    CALIB = "calib"
    EVAL = "eval"


class ComponentIOType(str, Enum):
    INPUTS = "inputs"
    OUTPUTS = "outputs"


class ComponentIOSink:
    """Manages component I/O dump and load for a single collection run."""

    def __init__(
        self,
        eval_mode: EvalMode,
        split: DataSplit,
    ) -> None:
        """
        Args:
            eval_mode:  FP or on-device eval mode that produced the I/O.
            split:      CALIB or EVAL data split.
        """
        self.eval_mode = eval_mode
        self.split = split
        self._file_counters: dict[str, int] = {}
        self._cleared = False  # cleared once on first dump

    def __repr__(self) -> str:
        return (
            f"ComponentIOSink("
            f"eval_mode={self.eval_mode.value!r}, "
            f"split={self.split.value!r})"
        )

    # Directory helpers
    def _base_dir(self) -> Path:
        base_dir = ASSET_CONFIG.get_local_store_model_path(
            MODEL_ID, MODEL_ASSET_VERSION, "component_io"
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / self.eval_mode.value / self.split.value

    def _io_dir(
        self,
        component_name: str,
        io_type: ComponentIOType,
    ) -> Path:
        return self._base_dir() / component_name / io_type.value

    # File counter - tracks how many samples dumped per component
    def _next_idx(self, component_name: str) -> int:
        return self._file_counters.get(component_name, 0)

    def _increment(self, component_name: str, count: int) -> None:
        self._file_counters[component_name] = (
            self._file_counters.get(component_name, 0) + count
        )

    # Num samples
    def sample_count(self, component_name: str) -> int:
        base = self._io_dir(component_name, ComponentIOType.INPUTS)
        if not base.is_dir():
            return 0
        # Peek at the first tensor name from the spec
        npy_files = sorted(base.glob("*.npy"))
        if not npy_files:
            return 0
        first_name = npy_files[0].stem.rsplit("_", 1)[0]
        return len(list(base.glob(f"{first_name}_*.npy")))

    # Dump
    def dump(
        self,
        component_name: str,
        inputs: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...] | torch.Tensor,
        input_names: list[str],
        output_names: list[str],
    ) -> None:
        if not self._cleared:
            base = self._base_dir()
            if base.exists():
                shutil.rmtree(base)
            self._cleared = True

        out_list = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
        file_start = self._next_idx(component_name)
        batch_size = inputs[0].shape[0] if inputs else 0

        for io_type, tensors, names in [
            (ComponentIOType.INPUTS, inputs, input_names),
            (ComponentIOType.OUTPUTS, out_list, output_names),
        ]:
            out_dir = self._io_dir(component_name, io_type)
            out_dir.mkdir(parents=True, exist_ok=True)
            for tensor, name in zip(tensors, names, strict=False):
                arr = tensor.detach().cpu().numpy()
                # strip batch dim - one file per sample
                for b in range(arr.shape[0]):
                    idx = file_start + b
                    path = out_dir / f"{name}_{idx:04d}.npy"
                    np.save(str(path), arr[b])

        self._increment(component_name, batch_size)

    # Load
    def load(
        self,
        component_name: str,
        spec: InputSpec,
        io_type: ComponentIOType = ComponentIOType.INPUTS,
        num_samples: int = 1,
    ) -> ComponentArrays:
        base = self._io_dir(component_name, io_type)

        if not base.is_dir():
            raise FileNotFoundError(
                f"Directory not found: {base}\n"
                f"Ensure ComponentIOSink.dump() was called first."
            )

        result: ComponentArrays = {}
        for name in spec:
            samples = []
            for idx in range(num_samples):
                path = base / f"{name}_{idx:04d}.npy"
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing sample {idx} for '{name}' "
                        f"[split={self.split.value}, io={io_type.value}].\n"
                        f"Expected: {path}\n"
                        f"Re-run collection with num_samples >= {num_samples}."
                    )
                arr = np.load(path)

                # Shape/dtype validation with InputSpec
                if spec[name] is not None:
                    expected_shape, expected_dtype = spec[name]
                    arr = arr.astype(expected_dtype)
                    # Restore batch dim if it was stripped on dump
                    if arr.shape == expected_shape[1:]:
                        arr = np.expand_dims(arr, axis=0)
                    if arr.shape != expected_shape:
                        raise ValueError(
                            f"Shape mismatch for '{name}' [{idx}] in '{component_name}': "
                            f"expected {expected_shape}, got {arr.shape}"
                        )
                else:
                    # No spec defined - just restore batch dim
                    arr = np.expand_dims(arr, axis=0)

                samples.append(arr)

            result[name] = np.concatenate(samples, axis=0)

        return result

    def clear(self) -> None:
        base = self._base_dir()
        if base.exists():
            shutil.rmtree(base)


# Generic utils
def onnx_remove_large_initializers(
    onnx_model: onnx.ModelProto, size_thresh: int = 1024 * 1024
) -> tuple[onnx.ModelProto, list[onnx.TensorProto]]:
    replaced_inits = []
    graph = onnx_model.graph
    for init in graph.initializer:
        shape = list(init.dims)
        num_elems = 1
        for s in shape:
            num_elems *= s
        if num_elems <= size_thresh:
            continue
        replaced_inits.append(init)

    for init in replaced_inits:
        del_names = [init.name]
        to_del = [i for i, x in enumerate(graph.initializer) if x.name in del_names]
        for i in reversed(to_del):
            del graph.initializer[i]
        graph.input.extend(
            [
                onnx.helper.make_tensor_value_info(
                    init.name, init.data_type, list(init.dims)
                )
            ]
        )

    return onnx_model, replaced_inits


def onnx_restore_large_initializers(
    onnx_model: onnx.ModelProto, replaced_inits: list[onnx.TensorProto]
) -> onnx.ModelProto:
    graph = onnx_model.graph
    replaced_names = {init.name for init in replaced_inits}
    graph.initializer.extend(replaced_inits)
    to_del = [i for i, x in enumerate(graph.input) if x.name in replaced_names]
    for i in reversed(to_del):
        del graph.input[i]

    return onnx_model


def get_sqnr(
    fp_out: np.ndarray, qt_out: np.ndarray, eps: float = 1e-10
) -> tuple[float, float, float]:
    fp_out, qt_out = fp_out.astype(np.float32), qt_out.astype(np.float32)

    quant_error = fp_out - qt_out
    exp_noise = (quant_error**2).mean() + eps
    exp_signal = (fp_out**2).mean()
    sqnr = exp_signal / exp_noise
    sqnr_db = 10 * np.log10(sqnr)
    return sqnr_db, exp_signal, exp_noise


def get_cossim(fp_out: np.ndarray, qt_out: np.ndarray, eps: float = 1e-10) -> float:
    fp_out, qt_out = fp_out.astype(np.float32), qt_out.astype(np.float32)

    return np.dot(fp_out, qt_out) / (norm(fp_out) * norm(qt_out) + eps)
