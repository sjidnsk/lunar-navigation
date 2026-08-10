"""Action V2 的确定性联调解释；不在运行时采样。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Protocol

import numpy as np

from lunar_model_contract import ActionContractV2
from lunar_model_contract.action import ActionContractError
from lunar_model_contract.observation import PLATFORM_CONTEXTS


class PolicyDecisionError(ValueError):
    """模型输出或候选 mask 不能形成无歧义动作。"""


@dataclass(frozen=True, slots=True)
class SelectedAction:
    frontier_index: int
    theta_rad: float | None
    value: float


class InferenceRuntime(Protocol):
    def infer(
        self, inputs: Mapping[str, np.ndarray]
    ) -> Mapping[str, np.ndarray]: ...


def select_action(
    outputs: Mapping[str, np.ndarray],
    candidate_mask: np.ndarray,
    platform_type: str,
) -> SelectedAction:
    """在有效候选中取最大 logit，并应用 fed9 theta mask。"""
    if platform_type not in PLATFORM_CONTEXTS:
        raise PolicyDecisionError("platform type is invalid")
    if (
        not isinstance(candidate_mask, np.ndarray)
        or candidate_mask.dtype != np.bool_
        or candidate_mask.shape != (1, 64)
    ):
        raise PolicyDecisionError("candidate_mask must be bool [1,64]")
    if not bool(candidate_mask.any()):
        raise PolicyDecisionError("candidate_mask has no valid candidate")
    try:
        ActionContractV2.validate_outputs(outputs)
    except ActionContractError as error:
        raise PolicyDecisionError(str(error)) from error
    if outputs["frontier_logits"].shape[0] != 1:
        raise PolicyDecisionError("deterministic runtime requires batch size one")
    theta_mu = outputs["theta_mu"]
    if bool(((theta_mu < -math.pi) | (theta_mu >= math.pi)).any()):
        raise PolicyDecisionError("theta_mu must be normalized to [-pi,pi)")
    masked_logits = np.where(
        candidate_mask,
        outputs["frontier_logits"],
        np.asarray(-np.inf, dtype=np.float32),
    )
    index = int(np.argmax(masked_logits[0]))
    theta = None if platform_type == "HOPPER" else float(theta_mu[0, index])
    return SelectedAction(
        frontier_index=index,
        theta_rad=theta,
        value=float(outputs["value"][0]),
    )


class DeterministicPolicy:
    """先处理无候选旁路，再调用 ONNX 并解释输出。"""

    def __init__(self, runtime: InferenceRuntime) -> None:
        if not hasattr(runtime, "infer"):
            raise TypeError("runtime must provide infer()")
        self._runtime = runtime

    def decide(
        self,
        inputs: Mapping[str, np.ndarray],
        platform_type: str,
    ) -> SelectedAction | None:
        mask = inputs.get("candidate_mask")
        if (
            isinstance(mask, np.ndarray)
            and mask.dtype == np.bool_
            and mask.shape == (1, 64)
            and not bool(mask.any())
        ):
            return None
        outputs = self._runtime.infer(inputs)
        return select_action(outputs, mask, platform_type)


__all__ = [
    "DeterministicPolicy",
    "PolicyDecisionError",
    "SelectedAction",
    "select_action",
]
