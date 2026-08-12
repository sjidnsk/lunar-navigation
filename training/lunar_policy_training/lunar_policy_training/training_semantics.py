"""Immutable observation and action semantics for formal policy training."""

from __future__ import annotations

import hashlib
import math


FORMAL_SENSOR_RANGE_M = 30.0
FORMAL_SENSOR_FOV_RAD = 2.0 * math.pi
FORMAL_MINIMUM_MISSION_COVERABLE_RATIO = 0.95
FORMAL_SUCCESS_COVERAGE_RATIO = 0.95
FORMAL_TRAINING_SEMANTICS_VERSION = (
    "lunar-training-semantics/"
    "sensor-30m-360-platform-physical-coverable-detail95-observed-physical-"
    "candidates-fixed2m-search-domain-snapshot-planner-failure-option-path-"
    "observation-auditable-failure-global-ground-target/v12"
)
# Keep the established public name as an alias so every identity consumer moves
# atomically without duplicating the semantic string.
TRAINING_SEMANTICS_VERSION = FORMAL_TRAINING_SEMANTICS_VERSION


def training_semantics_sha256() -> str:
    """Return the stable identity of policy-visible training semantics."""
    return hashlib.sha256(
        TRAINING_SEMANTICS_VERSION.encode("utf-8")
    ).hexdigest()


def formal_success_first_crossing(previous: float, current: float) -> bool:
    """Return the only legal first-crossing predicate for formal success."""
    values = (previous, current)
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in values
    ):
        raise ValueError("formal coverage ratios must be finite in [0,1]")
    return (
        float(previous)
        < FORMAL_SUCCESS_COVERAGE_RATIO
        <= float(current)
    )


__all__ = [
    "FORMAL_SENSOR_FOV_RAD",
    "FORMAL_SENSOR_RANGE_M",
    "FORMAL_MINIMUM_MISSION_COVERABLE_RATIO",
    "FORMAL_SUCCESS_COVERAGE_RATIO",
    "FORMAL_TRAINING_SEMANTICS_VERSION",
    "TRAINING_SEMANTICS_VERSION",
    "formal_success_first_crossing",
    "training_semantics_sha256",
]
