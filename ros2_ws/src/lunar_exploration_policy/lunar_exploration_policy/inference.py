"""严格四文件校验后的 ONNX Runtime CPU 推理。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import onnxruntime as ort

from lunar_model_contract import ActionContractV2, ObservationContractV3
from lunar_model_contract.action import ActionContractError
from lunar_model_contract.observation import (
    ObservationContractError,
    validate_observation_inputs,
)
from lunar_model_contract.package import (
    InterfaceModelPackageError,
    validate_interface_model_package,
)


class PolicyRuntimeError(RuntimeError):
    """模型包、ONNX session 或一次推理结果不符合冻结合同。"""


def _shape_matches(actual: Sequence[object], expected: Sequence[int | None]) -> bool:
    if len(actual) != len(expected):
        return False
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        if index == 0 and want is None:
            if got is None or isinstance(got, str):
                continue
            return False
        if got != want:
            return False
    return True


def validate_session_signature(inputs: Sequence[Any], outputs: Sequence[Any]) -> None:
    """验证 ONNX 名称、顺序、动态 batch、其余 shape 和 dtype。"""
    input_names = tuple(item.name for item in inputs)
    if input_names != ObservationContractV3.input_names:
        raise PolicyRuntimeError("ONNX input names/order differ from Observation V3")
    output_names = tuple(item.name for item in outputs)
    if output_names != ActionContractV2.output_names:
        raise PolicyRuntimeError("ONNX output names/order differ from Action V2")
    for item in inputs:
        expected_shape = ObservationContractV3.shapes[item.name]
        if not _shape_matches(item.shape, expected_shape):
            raise PolicyRuntimeError(f"ONNX input {item.name} shape is invalid")
        expected_type = "tensor(bool)" if item.name == "candidate_mask" else "tensor(float)"
        if item.type != expected_type:
            raise PolicyRuntimeError(f"ONNX input {item.name} dtype is invalid")
    for item in outputs:
        expected_shape = ActionContractV2.shapes[item.name]
        if not _shape_matches(item.shape, expected_shape):
            raise PolicyRuntimeError(f"ONNX output {item.name} shape is invalid")
        if item.type != "tensor(float)":
            raise PolicyRuntimeError(f"ONNX output {item.name} dtype is invalid")


class OnnxPolicyRuntime:
    """启动时验证一次包身份，每次推理继续验证动态张量。"""

    def __init__(self, model_dir: str | Path) -> None:
        try:
            self.manifest = validate_interface_model_package(model_dir)
        except InterfaceModelPackageError as error:
            raise PolicyRuntimeError(str(error)) from error
        try:
            self._session = ort.InferenceSession(
                str(Path(model_dir) / "policy.onnx"),
                providers=["CPUExecutionProvider"],
            )
        except Exception as error:
            raise PolicyRuntimeError(f"cannot create ONNX session: {error}") from error
        validate_session_signature(
            self._session.get_inputs(), self._session.get_outputs()
        )

    def infer(
        self, inputs: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        try:
            validate_observation_inputs(inputs)
        except ObservationContractError as error:
            raise PolicyRuntimeError(str(error)) from error
        try:
            values = self._session.run(
                list(ActionContractV2.output_names),
                {name: inputs[name] for name in ObservationContractV3.input_names},
            )
        except Exception as error:
            raise PolicyRuntimeError(f"ONNX inference failed: {error}") from error
        outputs = {
            name: value
            for name, value in zip(ActionContractV2.output_names, values, strict=True)
        }
        try:
            ActionContractV2.validate_outputs(outputs)
        except ActionContractError as error:
            raise PolicyRuntimeError(str(error)) from error
        return outputs


__all__ = [
    "OnnxPolicyRuntime",
    "PolicyRuntimeError",
    "validate_session_signature",
]
