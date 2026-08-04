"""The frozen four-output action contract for the polar PPO policy."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

import numpy as np


class ActionContractError(ValueError):
    """A policy output violates the frozen action contract."""


class ActionContractV2:
    """Names, geometry, and scalar limits shared by policy consumers."""

    version: Final = "lunar-action-contract/v2"
    output_names: Final = ("frontier_logits", "theta_mu", "theta_kappa", "value")
    candidate_count: Final = 64
    yaw_tolerance_rad: Final = math.pi / 24.0
    theta_kappa_min: Final = 0.05
    theta_kappa_max: Final = 64.0
    shapes: Final = {
        "frontier_logits": (None, 64),
        "theta_mu": (None, 64),
        "theta_kappa": (None, 64),
        "value": (None,),
    }

    @classmethod
    def validate_outputs(cls, mapping: Mapping[str, np.ndarray]) -> None:
        """Validate one finite batch of the four frozen policy outputs."""
        if set(mapping) != set(cls.output_names) or len(mapping) != len(cls.output_names):
            raise ActionContractError("action output names must exactly match contract")

        first_array = mapping[cls.output_names[0]]
        if not isinstance(first_array, np.ndarray) or first_array.ndim == 0:
            raise ActionContractError("frontier_logits must have shape [B,64]")
        batch_size = first_array.shape[0]
        if batch_size < 1:
            raise ActionContractError("action batch size must be positive")

        for name in cls.output_names:
            array = mapping[name]
            expected_shape = cls.shapes[name]
            if not isinstance(array, np.ndarray) or array.ndim != len(expected_shape) or array.shape[0] != batch_size or array.shape[1:] != expected_shape[1:]:
                shape_text = "[" + ",".join(
                    "B" if size is None else str(size) for size in expected_shape
                ) + "]"
                raise ActionContractError(f"{name} must have shape {shape_text}")
            if array.dtype != np.float32:
                raise ActionContractError(f"{name} must be float32")
            if not np.isfinite(array).all():
                raise ActionContractError(f"{name} must contain only finite values")

        theta_kappa = mapping["theta_kappa"]
        if (theta_kappa < cls.theta_kappa_min).any() or (theta_kappa > cls.theta_kappa_max).any():
            raise ActionContractError(
                f"theta_kappa must be within [{cls.theta_kappa_min}, {cls.theta_kappa_max}]"
            )
