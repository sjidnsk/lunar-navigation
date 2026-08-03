"""The single shared seven-input policy observation contract."""

from __future__ import annotations

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


class ObservationContractV1:
    """Names and validation shared by training, export, and runtime."""

    input_names: Final = (
        "prior_channels",
        "coverage_summary",
        "local_crop",
        "frontier_features",
        "pose_features",
        "candidate_mask",
        "platform_context",
    )


def validate_platform_context(array: np.ndarray) -> None:
    """Require a batch of finite, FP32, three-way one-hot platform vectors."""
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != PLATFORM_CONTEXT_WIDTH:
        raise ObservationContractError("platform_context must be float32 [B,3]")
    if not np.isfinite(array).all() or not np.isin(array, (0.0, 1.0)).all():
        raise ObservationContractError("platform_context must be finite one-hot")
    if not np.equal(array.sum(axis=1), 1.0).all():
        raise ObservationContractError("platform_context must be one-hot")
