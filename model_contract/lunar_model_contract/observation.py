"""The frozen seven-input observation contract for the polar PPO policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np


PLATFORM_CONTEXT_WIDTH: Final = 3
PLATFORM_CONTEXTS: Final = {
    "WHEELED": (1.0, 0.0, 0.0),
    "LEGGED": (0.0, 1.0, 0.0),
    "HOPPER": (0.0, 0.0, 1.0),
}


class ObservationContractError(ValueError):
    """A policy input violates the frozen observation contract."""


class ObservationContractV2:
    """Names, geometry, and field order shared by training and inference."""

    version: Final = "lunar-observation-contract/v2"
    input_names: Final = (
        "prior_channels",
        "coverage_summary",
        "local_crop",
        "frontier_features",
        "pose_features",
        "candidate_mask",
        "platform_context",
    )
    shapes: Final = {
        "prior_channels": (None, 4, 256, 256),
        "coverage_summary": (None, 3, 256, 256),
        "local_crop": (None, 4, 32, 32),
        "frontier_features": (None, 64, 12),
        "pose_features": (None, 6),
        "candidate_mask": (None, 64),
        "platform_context": (None, 3),
    }
    prior_channels: Final = (
        "observed_relative_elevation",
        "mission_priority",
        "observed_physical_obstacle_ratio",
        "active_platform_traversable_ratio",
    )
    coverage_summary_channels: Final = (
        "observed_relative_elevation",
        "mission_priority",
        "active_platform_traversable_ratio",
    )
    local_crop_channels: Final = prior_channels
    frontier_fields: Final = (
        "relative_x",
        "relative_y",
        "distance",
        "bearing_sin",
        "bearing_cos",
        "relative_elevation",
        "slope",
        "normal_sin",
        "normal_cos",
        "normal_confidence",
        "frontier_confidence",
        "coverage_gain",
    )
    pose_fields: Final = (
        "position_x",
        "position_y",
        "yaw_sin",
        "yaw_cos",
        "linear_velocity",
        "angular_velocity",
    )


def _require_float32_finite(name: str, array: np.ndarray) -> None:
    if array.dtype != np.float32:
        raise ObservationContractError(f"{name} must be float32")
    if not np.isfinite(array).all():
        raise ObservationContractError(f"{name} must contain only finite values")


def _shape_text(shape: tuple[int | None, ...]) -> str:
    return "[" + ",".join("B" if size is None else str(size) for size in shape) + "]"


def _require_shape(name: str, array: np.ndarray, batch_size: int) -> None:
    expected_shape = ObservationContractV2.shapes[name]
    if array.ndim != len(expected_shape) or array.shape[1:] != expected_shape[1:]:
        raise ObservationContractError(f"{name} must have shape {_shape_text(expected_shape)}")
    if array.shape[0] != batch_size:
        raise ObservationContractError(f"{name} batch size must match {batch_size}")


def validate_platform_context(array: np.ndarray) -> None:
    """Require a batch of finite, FP32, three-way one-hot platform vectors."""
    if not isinstance(array, np.ndarray) or array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != PLATFORM_CONTEXT_WIDTH:
        raise ObservationContractError("platform_context must be float32 [B,3]")
    if not np.isfinite(array).all() or not np.isin(array, (0.0, 1.0)).all():
        raise ObservationContractError("platform_context must be finite one-hot")
    if not np.equal(array.sum(axis=1), 1.0).all():
        raise ObservationContractError("platform_context must be one-hot")


def validate_observation_inputs(mapping: Mapping[str, np.ndarray]) -> None:
    """Validate one batch of the complete V2 observation mapping.

    A batch with an all-false candidate mask is valid: the builder owns the
    explicit bypass when no frontier candidate can be selected.
    """
    if set(mapping) != set(ObservationContractV2.input_names) or len(mapping) != len(
        ObservationContractV2.input_names
    ):
        raise ObservationContractError("observation input names must exactly match contract")

    first_name = ObservationContractV2.input_names[0]
    first_array = mapping[first_name]
    if not isinstance(first_array, np.ndarray) or first_array.ndim == 0:
        raise ObservationContractError(f"{first_name} must have shape [B,4,256,256]")
    batch_size = first_array.shape[0]
    if batch_size < 1:
        raise ObservationContractError("observation batch size must be positive")

    for name in ObservationContractV2.input_names:
        array = mapping[name]
        if not isinstance(array, np.ndarray):
            raise ObservationContractError(f"{name} must be a numpy array")
        _require_shape(name, array, batch_size)
        if name == "candidate_mask":
            if array.dtype != np.bool_:
                raise ObservationContractError("candidate_mask must be bool [B,64]")
            continue
        _require_float32_finite(name, array)

    validate_platform_context(mapping["platform_context"])
