"""Training-only truth isolation and conservative sensor reveal state."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..polar_data.hazards import CanvasRatioLayer
from ..polar_data.raster import MapCanvas
from .observation_builder import LocalObservation, ObservedWorld
from .observation_builder import Pose2
from .visibility import VisibilityEstimator


def _shape(canvas: MapCanvas) -> tuple[int, int]:
    return canvas.geometry.cells, canvas.geometry.cells


def _float_grid(
    name: str,
    values: np.ndarray,
    shape: tuple[int, int],
    *,
    ratio: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} dtype must be float32")
    if array.shape != shape or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous with canvas shape")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if ratio and ((array < 0.0) | (array > 1.0)).any():
        raise ValueError(f"{name} must be in [0,1]")
    if nonnegative and (array < 0.0).any():
        raise ValueError(f"{name} must be nonnegative")
    return array


def _bool_grid(
    name: str, values: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} dtype must be bool")
    if array.shape != shape or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous with canvas shape")
    return array


def _count_grid(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.uint32):
        raise TypeError("observation count dtype must be uint32")
    if array.shape != shape or not array.flags.c_contiguous:
        raise ValueError(
            "observation count must be C-contiguous with canvas shape"
        )
    return array


@dataclass(frozen=True, slots=True)
class TrainingWorldTruth:
    """Complete terrain truth that is intentionally not policy-compatible."""

    canvas: MapCanvas
    elevation_m: np.ndarray
    physical_obstacle_ratio: np.ndarray
    elevation_variance: np.ndarray | None = None
    obstacle_variance: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.canvas, MapCanvas):
            raise TypeError("training truth requires a map canvas")
        shape = _shape(self.canvas)
        elevation = _float_grid(
            "truth elevation", self.elevation_m, shape
        ).copy()
        obstacle = _float_grid(
            "truth physical obstacle ratio",
            self.physical_obstacle_ratio,
            shape,
            ratio=True,
        ).copy()
        elevation_variance = (
            np.zeros(shape, dtype=np.float32)
            if self.elevation_variance is None
            else _float_grid(
                "truth elevation variance",
                self.elevation_variance,
                shape,
                nonnegative=True,
            ).copy()
        )
        obstacle_variance = (
            np.zeros(shape, dtype=np.float32)
            if self.obstacle_variance is None
            else _float_grid(
                "truth obstacle variance",
                self.obstacle_variance,
                shape,
                nonnegative=True,
            ).copy()
        )
        elevation.setflags(write=False)
        obstacle.setflags(write=False)
        elevation_variance.setflags(write=False)
        obstacle_variance.setflags(write=False)
        object.__setattr__(self, "elevation_m", elevation)
        object.__setattr__(self, "physical_obstacle_ratio", obstacle)
        object.__setattr__(self, "elevation_variance", elevation_variance)
        object.__setattr__(self, "obstacle_variance", obstacle_variance)


@dataclass(slots=True)
class TrainingObservedGrid:
    """Mutable observed state; unknown cells never expose truth values."""

    canvas: MapCanvas
    elevation_m: np.ndarray
    physical_obstacle_ratio: np.ndarray
    valid_mask: np.ndarray
    observation_age_s: np.ndarray
    observation_quality: np.ndarray
    elevation_variance: np.ndarray
    obstacle_variance: np.ndarray
    observation_count: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.canvas, MapCanvas):
            raise TypeError("training observed grid requires a map canvas")
        shape = _shape(self.canvas)
        self.elevation_m = _float_grid(
            "observed elevation", self.elevation_m, shape
        ).copy()
        self.physical_obstacle_ratio = _float_grid(
            "observed physical obstacle ratio",
            self.physical_obstacle_ratio,
            shape,
            ratio=True,
        ).copy()
        self.valid_mask = _bool_grid(
            "valid mask", self.valid_mask, shape
        ).copy()
        self.observation_age_s = _float_grid(
            "observation age", self.observation_age_s, shape
        ).copy()
        self.observation_quality = _float_grid(
            "observation quality",
            self.observation_quality,
            shape,
            ratio=True,
        ).copy()
        self.elevation_variance = _float_grid(
            "elevation variance",
            self.elevation_variance,
            shape,
            nonnegative=True,
        ).copy()
        self.obstacle_variance = _float_grid(
            "obstacle variance",
            self.obstacle_variance,
            shape,
            nonnegative=True,
        ).copy()
        self.observation_count = _count_grid(
            self.observation_count, shape
        ).copy()

    @classmethod
    def empty(cls, canvas: MapCanvas) -> "TrainingObservedGrid":
        shape = _shape(canvas)
        zeros = lambda: np.zeros(shape, dtype=np.float32)
        return cls(
            canvas=canvas,
            elevation_m=zeros(),
            physical_obstacle_ratio=zeros(),
            valid_mask=np.zeros(shape, dtype=np.bool_),
            observation_age_s=zeros(),
            observation_quality=zeros(),
            elevation_variance=zeros(),
            obstacle_variance=zeros(),
            observation_count=np.zeros(shape, dtype=np.uint32),
        )

    def copy(self) -> "TrainingObservedGrid":
        return TrainingObservedGrid(
            canvas=self.canvas,
            elevation_m=self.elevation_m.copy(),
            physical_obstacle_ratio=self.physical_obstacle_ratio.copy(),
            valid_mask=self.valid_mask.copy(),
            observation_age_s=self.observation_age_s.copy(),
            observation_quality=self.observation_quality.copy(),
            elevation_variance=self.elevation_variance.copy(),
            obstacle_variance=self.obstacle_variance.copy(),
            observation_count=self.observation_count.copy(),
        )

    def to_observed_world(
        self, *, local: LocalObservation
    ) -> ObservedWorld:
        """Cross the only explicit training-state to policy-world boundary."""
        return ObservedWorld(
            canvas=self.canvas,
            elevation_m=self.elevation_m,
            observed_mask=self.valid_mask,
            physical_obstacle_layer=CanvasRatioLayer(
                self.canvas, self.physical_obstacle_ratio
            ),
            local=local,
        )


@dataclass(frozen=True, slots=True)
class ObservationDelta:
    visible_cells: int
    newly_observed_cells: int
    mission_observed_delta_m2: float
    priority_observed_delta_m2: float


class SensorObservationState:
    """Own one atomic truth-to-observed reveal sequence."""

    def __init__(
        self,
        *,
        truth: TrainingWorldTruth,
        observed: TrainingObservedGrid,
        mission_roi_ratio: np.ndarray,
        mission_priority: np.ndarray,
        forbidden_mask: np.ndarray,
        visibility_estimator: VisibilityEstimator,
    ) -> None:
        if not isinstance(truth, TrainingWorldTruth):
            raise TypeError("sensor state requires TrainingWorldTruth")
        if not isinstance(observed, TrainingObservedGrid):
            raise TypeError("sensor state requires TrainingObservedGrid")
        if truth.canvas != observed.canvas:
            raise ValueError("truth and observed canvas mismatch")
        sensor = getattr(visibility_estimator, "sensor", None)
        reveal = getattr(visibility_estimator, "reveal_from_pose", None)
        if sensor is None or not callable(reveal):
            raise TypeError("sensor state requires a visibility estimator")
        resolution = getattr(visibility_estimator, "resolution_m", None)
        if (
            not isinstance(resolution, (int, float))
            or isinstance(resolution, bool)
            or not math.isclose(
                float(resolution),
                truth.canvas.geometry.resolution_m,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("visibility estimator resolution mismatch")
        shape = _shape(truth.canvas)
        self.truth = truth
        self.observed = observed
        self.mission_roi_ratio = _float_grid(
            "mission ROI ratio", mission_roi_ratio, shape, ratio=True
        ).copy()
        self.mission_priority = _float_grid(
            "mission priority", mission_priority, shape, ratio=True
        ).copy()
        self.forbidden_mask = _bool_grid(
            "forbidden mask", forbidden_mask, shape
        ).copy()
        self.mission_roi_ratio.setflags(write=False)
        self.mission_priority.setflags(write=False)
        self.forbidden_mask.setflags(write=False)
        self.visibility_estimator = visibility_estimator

    def observe(
        self, pose_cell: tuple[int, int], *, elapsed_s: float
    ) -> ObservationDelta:
        return self.observe_repeated(
            pose_cell, elapsed_steps_s=(elapsed_s,)
        )

    def observe_repeated(
        self,
        pose_cell: tuple[int, int],
        *,
        elapsed_steps_s: tuple[float, ...],
    ) -> ObservationDelta:
        """Apply ordered observations that share one exact visibility cell."""
        if (
            not isinstance(elapsed_steps_s, tuple)
            or not elapsed_steps_s
            or any(
                not isinstance(elapsed_s, (int, float))
                or isinstance(elapsed_s, bool)
                or not math.isfinite(float(elapsed_s))
                or elapsed_s < 0.0
                for elapsed_s in elapsed_steps_s
            )
        ):
            raise ValueError("observation elapsed time is invalid")
        shape = _shape(self.truth.canvas)
        if (
            not isinstance(pose_cell, tuple)
            or len(pose_cell) != 2
            or any(type(value) is not int for value in pose_cell)
            or not 0 <= pose_cell[0] < shape[0]
            or not 0 <= pose_cell[1] < shape[1]
        ):
            raise ValueError("observation pose is outside the grid")

        visible = self.visibility_estimator.reveal_from_pose(
            self.truth.physical_obstacle_ratio, pose_cell
        )
        if (
            not isinstance(visible, np.ndarray)
            or visible.dtype != np.dtype(np.bool_)
            or visible.shape != shape
            or not visible.flags.c_contiguous
        ):
            raise RuntimeError("visibility estimator returned an invalid mask")

        known_before = self.observed.valid_mask.copy()
        age = self.observed.observation_age_s.copy()
        elevation = self.observed.elevation_m.copy()
        obstacle = self.observed.physical_obstacle_ratio.copy()
        valid = known_before.copy()
        quality = self.observed.observation_quality.copy()
        elevation_variance = self.observed.elevation_variance.copy()
        obstacle_variance = self.observed.obstacle_variance.copy()
        count = self.observed.observation_count.copy()

        # Preserve the exact sequential float32 age and saturating-count
        # semantics while reusing the visibility mask and array copies.
        for elapsed_s in elapsed_steps_s:
            aged_values = (
                age[valid].astype(np.float64) + float(elapsed_s)
            )
            if (
                not np.isfinite(aged_values).all()
                or (aged_values > np.finfo(np.float32).max).any()
            ):
                raise ValueError("observation age would overflow")
            age[valid] = aged_values.astype(np.float32)
            age[visible] = 0.0
            valid[visible] = True
            incremented = np.minimum(
                count[visible].astype(np.uint64) + 1,
                np.iinfo(np.uint32).max,
            ).astype(np.uint32)
            count[visible] = incremented

        elevation[visible] = self.truth.elevation_m[visible]
        obstacle[visible] = self.truth.physical_obstacle_ratio[visible]
        quality[visible] = 1.0
        elevation_variance[visible] = self.truth.elevation_variance[visible]
        obstacle_variance[visible] = self.truth.obstacle_variance[visible]

        newly_observed = visible & ~known_before
        cell_area_m2 = self.truth.canvas.geometry.resolution_m**2
        mission_delta = float(
            self.mission_roi_ratio[newly_observed].sum(dtype=np.float64)
            * cell_area_m2
        )
        priority_delta = float(
            (
                self.mission_priority[newly_observed]
                * self.mission_roi_ratio[newly_observed]
            ).sum(dtype=np.float64)
            * cell_area_m2
        )

        self.observed.elevation_m = elevation
        self.observed.physical_obstacle_ratio = obstacle
        self.observed.valid_mask = valid
        self.observed.observation_age_s = age
        self.observed.observation_quality = quality
        self.observed.elevation_variance = elevation_variance
        self.observed.obstacle_variance = obstacle_variance
        self.observed.observation_count = count
        return ObservationDelta(
            visible_cells=(
                int(np.count_nonzero(visible)) * len(elapsed_steps_s)
            ),
            newly_observed_cells=int(np.count_nonzero(newly_observed)),
            mission_observed_delta_m2=mission_delta,
            priority_observed_delta_m2=priority_delta,
        )

    def observe_world(
        self, pose: Pose2, *, elapsed_s: float
    ) -> ObservationDelta:
        if not isinstance(pose, Pose2) or pose.frame_id != "map":
            raise ValueError("observation pose must be a map-frame Pose2")
        try:
            pose_cell = self.truth.canvas.world_to_grid(pose.x_m, pose.y_m)
        except ValueError as error:
            raise ValueError("observation pose is outside the grid") from error
        return self.observe(pose_cell, elapsed_s=elapsed_s)


__all__ = [
    "ObservationDelta",
    "SensorObservationState",
    "TrainingObservedGrid",
    "TrainingWorldTruth",
]
