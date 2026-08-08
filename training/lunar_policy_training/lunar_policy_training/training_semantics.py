"""Immutable observation and action semantics for formal policy training."""

from __future__ import annotations

import hashlib
import math


FORMAL_SENSOR_RANGE_M = 30.0
FORMAL_SENSOR_FOV_RAD = 2.0 * math.pi
FORMAL_SUCCESS_COVERAGE_RATIO = 0.95
TRAINING_SEMANTICS_VERSION = (
    "lunar-training-semantics/sensor-30m-360-theta-mask-roi95-unbounded-start-qualified/v4"
)


def training_semantics_sha256() -> str:
    """Return the stable identity of policy-visible training semantics."""
    return hashlib.sha256(
        TRAINING_SEMANTICS_VERSION.encode("utf-8")
    ).hexdigest()


__all__ = [
    "FORMAL_SENSOR_FOV_RAD",
    "FORMAL_SENSOR_RANGE_M",
    "FORMAL_SUCCESS_COVERAGE_RATIO",
    "TRAINING_SEMANTICS_VERSION",
    "training_semantics_sha256",
]
