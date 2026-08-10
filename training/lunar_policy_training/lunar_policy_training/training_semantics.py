"""Immutable observation and action semantics for formal policy training."""

from __future__ import annotations

import hashlib
import math


FORMAL_SENSOR_RANGE_M = 30.0
FORMAL_SENSOR_FOV_RAD = 2.0 * math.pi
FORMAL_SUCCESS_COVERAGE_RATIO = 0.95
FORMAL_TRAINING_SEMANTICS_VERSION = (
    "lunar-training-semantics/"
    "sensor-30m-360-platform-coverable-detail95-unbounded-per-platform-subset/v6"
)
# Keep the established public name as an alias so every identity consumer moves
# atomically without duplicating the semantic string.
TRAINING_SEMANTICS_VERSION = FORMAL_TRAINING_SEMANTICS_VERSION


def training_semantics_sha256() -> str:
    """Return the stable identity of policy-visible training semantics."""
    return hashlib.sha256(
        TRAINING_SEMANTICS_VERSION.encode("utf-8")
    ).hexdigest()


__all__ = [
    "FORMAL_SENSOR_FOV_RAD",
    "FORMAL_SENSOR_RANGE_M",
    "FORMAL_SUCCESS_COVERAGE_RATIO",
    "FORMAL_TRAINING_SEMANTICS_VERSION",
    "TRAINING_SEMANTICS_VERSION",
    "training_semantics_sha256",
]
