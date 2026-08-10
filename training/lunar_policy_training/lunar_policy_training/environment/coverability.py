"""Exact platform/start-specific exploration coverability contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import math

import numpy as np

from ..training_semantics import FORMAL_SUCCESS_COVERAGE_RATIO


_PLATFORM_TYPES = frozenset(("WHEELED", "LEGGED", "HOPPER"))


class CoverabilityError(ValueError):
    """A coverability artifact is ambiguous, drifted, or internally inconsistent."""


class IneligibleReason(str, Enum):
    """The only ordinary task-ineligibility outcomes, in evaluation order."""

    UNSAFE_START = "UNSAFE_START"
    ZERO_MISSION_TARGET = "ZERO_MISSION_TARGET"
    MISSION_COVERABLE_BELOW_95 = "MISSION_COVERABLE_BELOW_95"
    INITIAL_ALREADY_SUCCESS = "INITIAL_ALREADY_SUCCESS"
    NO_INITIAL_CANDIDATE = "NO_INITIAL_CANDIDATE"


def _require_bool_mask(value: np.ndarray, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.bool_)
        or value.ndim != 2
        or not value.flags.c_contiguous
        or any(dimension <= 0 for dimension in value.shape)
    ):
        raise CoverabilityError(
            f"{name} must be a non-empty C-contiguous boolean matrix"
        )
    return value


def _require_sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CoverabilityError(f"{name} must be a lowercase SHA-256 digest")
    return value


def mask_sha256(mask: np.ndarray) -> str:
    """Hash the semantic row-major boolean cells rather than packed padding."""
    checked = _require_bool_mask(mask, "mask")
    return sha256(checked.view(np.uint8).tobytes(order="C")).hexdigest()


def pack_detail_mask(mask: np.ndarray) -> np.ndarray:
    """Pack a two-dimensional boolean mask in row-major, MSB-first order."""
    checked = _require_bool_mask(mask, "detail mask")
    return np.ascontiguousarray(
        np.packbits(checked.reshape(-1), bitorder="big"),
        dtype=np.uint8,
    )


def _detail_shape(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
    ):
        raise CoverabilityError("detail mask shape must contain two positive integers")
    return value


def unpack_detail_mask(
    packed: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Unpack exact semantic bits and reject any non-zero padding bits."""
    checked_shape = _detail_shape(shape)
    cell_count = math.prod(checked_shape)
    byte_count = (cell_count + 7) // 8
    if (
        not isinstance(packed, np.ndarray)
        or packed.dtype != np.dtype(np.uint8)
        or packed.ndim != 1
        or not packed.flags.c_contiguous
        or packed.size != byte_count
    ):
        raise CoverabilityError("packed detail mask size or dtype is invalid")
    padding = byte_count * 8 - cell_count
    if padding and int(packed[-1]) & ((1 << padding) - 1):
        raise CoverabilityError("packed detail mask has non-zero padding bits")
    unpacked = np.unpackbits(packed, bitorder="big", count=cell_count)
    return np.ascontiguousarray(
        unpacked.reshape(checked_shape).astype(np.bool_, copy=False)
    )


def classify_ineligibility(
    *,
    qualified_start_cell: tuple[int, int] | None,
    mission_target_detail_cell_count: int,
    coverable_detail_cell_count: int,
    initial_coverable_fraction: float,
    initial_candidate_count: int,
) -> IneligibleReason | None:
    """Return the first ordinary eligibility failure in the frozen order."""
    for name, count in (
        ("mission target cell count", mission_target_detail_cell_count),
        ("coverable detail cell count", coverable_detail_cell_count),
        ("initial candidate count", initial_candidate_count),
    ):
        if type(count) is not int or count < 0:
            raise CoverabilityError(f"{name} must be a non-negative integer")
    if coverable_detail_cell_count > mission_target_detail_cell_count:
        raise CoverabilityError("coverable detail count exceeds mission target count")
    if (
        not isinstance(initial_coverable_fraction, float)
        or not math.isfinite(initial_coverable_fraction)
        or not 0.0 <= initial_coverable_fraction <= 1.0
    ):
        raise CoverabilityError("initial coverable fraction must be finite in [0,1]")
    if qualified_start_cell is not None and (
        not isinstance(qualified_start_cell, tuple)
        or len(qualified_start_cell) != 2
        or any(type(index) is not int or index < 0 for index in qualified_start_cell)
    ):
        raise CoverabilityError("qualified start cell is invalid")

    if qualified_start_cell is None:
        return IneligibleReason.UNSAFE_START
    if mission_target_detail_cell_count == 0:
        return IneligibleReason.ZERO_MISSION_TARGET
    if (
        coverable_detail_cell_count / mission_target_detail_cell_count
        < FORMAL_SUCCESS_COVERAGE_RATIO
    ):
        return IneligibleReason.MISSION_COVERABLE_BELOW_95
    if initial_coverable_fraction >= FORMAL_SUCCESS_COVERAGE_RATIO:
        return IneligibleReason.INITIAL_ALREADY_SUCCESS
    if initial_candidate_count == 0:
        return IneligibleReason.NO_INITIAL_CANDIDATE
    return None


@dataclass(frozen=True, slots=True)
class PlatformCoverability:
    """Validated exact cache payload for one scene/platform/fixed start."""

    platform_type: str
    qualified_start_cell: tuple[int, int] | None
    reachable_pose_mask: np.ndarray
    coverable_detail_shape: tuple[int, int]
    coverable_detail_bits: np.ndarray
    coverable_ratio: np.ndarray
    mission_target_detail_cell_count: int
    coverable_detail_cell_count: int
    mission_coverable_fraction: float
    initial_coverable_fraction: float
    initial_candidate_count: int
    reachability_algorithm_id: str
    visibility_algorithm_id: str
    reachable_mask_sha256: str
    coverable_mask_sha256: str
    exact: bool
    eligible: bool
    ineligible_reason: IneligibleReason | None

    def __post_init__(self) -> None:
        if self.platform_type not in _PLATFORM_TYPES:
            raise CoverabilityError("platform type is invalid")
        reachable = _require_bool_mask(
            self.reachable_pose_mask, "reachable pose mask"
        )
        shape = _detail_shape(self.coverable_detail_shape)
        detail = unpack_detail_mask(self.coverable_detail_bits, shape)
        if shape[0] % reachable.shape[0] or shape[1] % reachable.shape[1]:
            raise CoverabilityError(
                "detail mask shape must divide evenly into the reachable grid"
            )
        ratio = self.coverable_ratio
        if (
            not isinstance(ratio, np.ndarray)
            or ratio.dtype != np.dtype(np.float32)
            or ratio.shape != reachable.shape
            or not ratio.flags.c_contiguous
            or not np.isfinite(ratio).all()
            or ((ratio < 0.0) | (ratio > 1.0)).any()
        ):
            raise CoverabilityError("coverable ratio matrix is invalid")
        rows_per_cell = shape[0] // reachable.shape[0]
        columns_per_cell = shape[1] // reachable.shape[1]
        expected_ratio = detail.reshape(
            reachable.shape[0],
            rows_per_cell,
            reachable.shape[1],
            columns_per_cell,
        ).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)
        if not np.array_equal(ratio, expected_ratio):
            raise CoverabilityError("coverable ratio differs from exact detail bits")

        if type(self.exact) is not bool or not self.exact:
            raise CoverabilityError("coverability payload must be exact")
        if type(self.eligible) is not bool:
            raise CoverabilityError("coverability eligibility must be boolean")
        for name, value in (
            ("reachability algorithm ID", self.reachability_algorithm_id),
            ("visibility algorithm ID", self.visibility_algorithm_id),
        ):
            if not isinstance(value, str) or not value:
                raise CoverabilityError(f"{name} is missing")
        if mask_sha256(reachable) != _require_sha(
            self.reachable_mask_sha256, "reachable mask hash"
        ):
            raise CoverabilityError("reachable mask hash differs")
        if mask_sha256(detail) != _require_sha(
            self.coverable_mask_sha256, "coverable mask hash"
        ):
            raise CoverabilityError("coverable mask hash differs")

        if type(self.coverable_detail_cell_count) is not int or (
            self.coverable_detail_cell_count != int(detail.sum(dtype=np.int64))
        ):
            raise CoverabilityError("coverable detail cell count differs from mask")
        expected_reason = classify_ineligibility(
            qualified_start_cell=self.qualified_start_cell,
            mission_target_detail_cell_count=self.mission_target_detail_cell_count,
            coverable_detail_cell_count=self.coverable_detail_cell_count,
            initial_coverable_fraction=self.initial_coverable_fraction,
            initial_candidate_count=self.initial_candidate_count,
        )
        expected_fraction = (
            self.coverable_detail_cell_count
            / self.mission_target_detail_cell_count
            if self.mission_target_detail_cell_count
            else 0.0
        )
        if (
            not isinstance(self.mission_coverable_fraction, float)
            or not math.isclose(
                self.mission_coverable_fraction,
                expected_fraction,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise CoverabilityError("mission coverable fraction differs from counts")
        if self.eligible is not (expected_reason is None):
            raise CoverabilityError("coverability eligibility disagrees with gates")
        if self.ineligible_reason is not expected_reason:
            raise CoverabilityError("coverability ineligible reason is invalid")
        if self.qualified_start_cell is not None:
            row, column = self.qualified_start_cell
            if (
                row >= reachable.shape[0]
                or column >= reachable.shape[1]
                or not bool(reachable[row, column])
            ):
                raise CoverabilityError("qualified start is not reachable")

    @property
    def coverable_detail_mask(self) -> np.ndarray:
        """Return a detached exact boolean mask for runtime accounting."""
        return unpack_detail_mask(
            self.coverable_detail_bits.copy(), self.coverable_detail_shape
        )


__all__ = [
    "CoverabilityError",
    "IneligibleReason",
    "PlatformCoverability",
    "classify_ineligibility",
    "mask_sha256",
    "pack_detail_mask",
    "unpack_detail_mask",
]
