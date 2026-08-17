"""Sparse 0.2 m sensor state with conservative 4 m policy aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import struct

import numpy as np

from .coverability import (
    mask_sha256,
    read_packed_detail_window,
    unpack_detail_mask,
)
from ..polar_data.multires_scene import (
    MultiResolutionScene,
    ProjectedScene,
    SceneTileProvider,
)
from ..polar_data.raster import GLOBAL_GEOMETRY, LOCAL_GEOMETRY, MapCanvas
from ..training_semantics import FORMAL_SENSOR_FOV_RAD, FORMAL_SENSOR_RANGE_M
from .observation_builder import LocalObservation, Pose2
from .sensor_observation import (
    ObservationDelta,
    SensorObservationState,
    TrainingObservedGrid,
    TrainingWorldTruth,
)
from .visibility import NativeVisibilityEstimator, SensorGeometry


_DETAIL_PER_GLOBAL = int(
    round(GLOBAL_GEOMETRY.resolution_m / LOCAL_GEOMETRY.resolution_m)
)


@dataclass(slots=True)
class _ObservedDetailTile:
    elevation_m: np.ndarray
    physical_obstacle_ratio: np.ndarray
    physical_obstacle_height_m: np.ndarray
    forbidden_ratio: np.ndarray
    valid_mask: np.ndarray
    observation_age_s: np.ndarray
    observation_quality: np.ndarray
    observation_count: np.ndarray

    @classmethod
    def empty(cls, cells: int) -> "_ObservedDetailTile":
        shape = (cells, cells)
        zeros = lambda: np.zeros(shape, dtype=np.float32)
        return cls(
            elevation_m=zeros(),
            physical_obstacle_ratio=zeros(),
            physical_obstacle_height_m=zeros(),
            forbidden_ratio=zeros(),
            valid_mask=np.zeros(shape, dtype=np.bool_),
            observation_age_s=zeros(),
            observation_quality=zeros(),
            observation_count=np.zeros(shape, dtype=np.uint32),
        )

    def copy(self) -> "_ObservedDetailTile":
        return _ObservedDetailTile(
            elevation_m=self.elevation_m.copy(),
            physical_obstacle_ratio=self.physical_obstacle_ratio.copy(),
            physical_obstacle_height_m=self.physical_obstacle_height_m.copy(),
            forbidden_ratio=self.forbidden_ratio.copy(),
            valid_mask=self.valid_mask.copy(),
            observation_age_s=self.observation_age_s.copy(),
            observation_quality=self.observation_quality.copy(),
            observation_count=self.observation_count.copy(),
        )


@dataclass(frozen=True, slots=True)
class _AgedDetailUpdate:
    tile: _ObservedDetailTile
    known_mask: np.ndarray
    observation_age_s: np.ndarray


@dataclass(frozen=True, slots=True)
class _VisibleDetailUpdate:
    key: tuple[int, int]
    tile: _ObservedDetailTile
    insert_tile: bool
    window_slice: tuple[slice, slice]
    tile_slice: tuple[slice, slice]
    visible_mask: np.ndarray
    observation_count: np.ndarray


@dataclass(frozen=True, slots=True)
class _CoarseObservationUpdate:
    row: int
    column: int
    elevation_m: np.float32
    physical_obstacle_ratio: np.float32
    physical_obstacle_height_m: np.float32
    forbidden_ratio: np.float32
    observation_age_s: np.float32
    observation_quality: np.float32
    observation_count: np.uint32


@dataclass(frozen=True, slots=True)
class DetailObservedWindow:
    canvas: MapCanvas
    elevation_m: np.ndarray
    physical_obstacle_ratio: np.ndarray
    physical_obstacle_height_m: np.ndarray
    forbidden_ratio: np.ndarray
    valid_mask: np.ndarray
    observation_age_s: np.ndarray
    observation_quality: np.ndarray
    observation_count: np.ndarray


@dataclass(frozen=True, slots=True)
class HopperTrajectoryPoint:
    """One finite map-frame pose at a monotonic executed-flight time."""

    pose_map: Pose2
    time_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.pose_map, Pose2) or self.pose_map.frame_id != "map":
            raise ValueError("hopper trajectory point must be a map-frame Pose2")
        if any(
            not math.isfinite(float(value))
            for value in (
                self.pose_map.x_m,
                self.pose_map.y_m,
                self.pose_map.yaw_rad,
                self.pose_map.elevation_m,
            )
        ):
            raise ValueError("hopper trajectory pose must be finite")
        if (
            not isinstance(self.time_s, (int, float))
            or isinstance(self.time_s, bool)
            or not math.isfinite(float(self.time_s))
            or float(self.time_s) < 0.0
        ):
            raise ValueError("hopper trajectory time must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class HopperTrajectoryTilePatch:
    """One immutable 0.2 m tile mask in a trajectory-capsule union."""

    tile_row: int
    tile_column: int
    visible_mask: np.ndarray

    def __post_init__(self) -> None:
        if type(self.tile_row) is not int or type(self.tile_column) is not int:
            raise ValueError("hopper trajectory tile identity is invalid")
        mask = np.asarray(self.visible_mask)
        if (
            mask.dtype != np.dtype(np.bool_)
            or mask.ndim != 2
            or mask.shape[0] != mask.shape[1]
            or not mask.flags.c_contiguous
            or not bool(mask.any())
        ):
            raise ValueError("hopper trajectory tile mask is invalid")
        frozen = np.ascontiguousarray(mask.copy(), dtype=np.bool_)
        frozen.setflags(write=False)
        object.__setattr__(self, "visible_mask", frozen)


@dataclass(frozen=True, slots=True)
class HopperTrajectoryObservationPatch:
    """Immutable sparse capsule raster prepared against one evidence revision."""

    scene_id: str
    evidence_generation: int
    trajectory_sha256: str
    tiles: tuple[HopperTrajectoryTilePatch, ...]
    visible_cells: int

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("hopper trajectory scene identity is invalid")
        if type(self.evidence_generation) is not int or self.evidence_generation < 0:
            raise ValueError("hopper trajectory evidence generation is invalid")
        if (
            not isinstance(self.trajectory_sha256, str)
            or len(self.trajectory_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.trajectory_sha256
            )
        ):
            raise ValueError("hopper trajectory identity is invalid")
        if not isinstance(self.tiles, tuple) or any(
            not isinstance(tile, HopperTrajectoryTilePatch) for tile in self.tiles
        ):
            raise ValueError("hopper trajectory tiles are invalid")
        identities = tuple((tile.tile_row, tile.tile_column) for tile in self.tiles)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("hopper trajectory tiles must be unique row-major tiles")
        count = sum(
            int(np.count_nonzero(tile.visible_mask)) for tile in self.tiles
        )
        if type(self.visible_cells) is not int or self.visible_cells != count:
            raise ValueError("hopper trajectory visible cell count differs")

    def detail_cell_visible(self, row: int, column: int) -> bool:
        """Return whether one global detail cell belongs to this patch."""
        if type(row) is not int or type(column) is not int or row < 0 or column < 0:
            return False
        if not self.tiles:
            return False
        tile_cells = self.tiles[0].visible_mask.shape[0]
        tile_identity = row // tile_cells, column // tile_cells
        for tile in self.tiles:
            identity = tile.tile_row, tile.tile_column
            if identity == tile_identity:
                return bool(
                    tile.visible_mask[row % tile_cells, column % tile_cells]
                )
            if identity > tile_identity:
                break
        return False


@dataclass(frozen=True, slots=True)
class _HopperObservationRollbackState:
    detail_tiles: dict[tuple[int, int], _ObservedDetailTile]
    observed: TrainingObservedGrid
    coarse_obstacle_height_m: np.ndarray
    coarse_forbidden_ratio: np.ndarray
    observed_coverable_detail_bits: np.ndarray | None
    observed_coverable_detail_cell_count: int
    observed_priority_detail_cell_count: int
    evidence_generation: int


class MultiresSensorObservationState(SensorObservationState):
    """Reveal real 0.2 m cells while exposing only sparse observed state."""

    def __init__(
        self,
        *,
        scene: MultiResolutionScene,
        tile_provider: SceneTileProvider,
        mission_roi_ratio: np.ndarray,
        mission_priority: np.ndarray,
        coverable_detail_shape: tuple[int, int] | None = None,
        coverable_detail_bits: np.ndarray | None = None,
        coverable_detail_cell_count: int | None = None,
        coverable_mask_sha256: str | None = None,
        priority_detail_shape: tuple[int, int] | None = None,
        priority_detail_bits: np.ndarray | None = None,
        priority_detail_cell_count: int | None = None,
        priority_detail_mask_sha256: str | None = None,
    ) -> None:
        if not isinstance(scene, MultiResolutionScene):
            raise TypeError("multires sensor state requires a scene")
        if not isinstance(tile_provider, SceneTileProvider):
            raise TypeError("multires sensor state requires a tile provider")
        if tile_provider.scene is not scene:
            raise ValueError("tile provider must own the same multires scene")
        geometry = tile_provider.tile_geometry
        if (
            geometry.cells != 320
            or not math.isclose(geometry.size_m, 64.0)
            or not math.isclose(geometry.resolution_m, 0.2)
        ):
            raise ValueError("formal detail tiles must be 64 m at 0.2 m")
        global_truth = scene.project(scene.base_canvas)
        safe_elevation = np.where(
            global_truth.valid_mask, global_truth.elevation_m, 0.0
        ).astype(np.float32)
        self.truth = TrainingWorldTruth(
            scene.base_canvas,
            safe_elevation,
            global_truth.physical_obstacle_ratio,
        )
        self.observed = TrainingObservedGrid.empty(scene.base_canvas)
        coarse_cells = scene.base_canvas.geometry.cells
        shape = (coarse_cells, coarse_cells)
        self.mission_roi_ratio = self._coarse_ratio(
            "mission ROI ratio", mission_roi_ratio, shape
        )
        self.mission_priority = self._coarse_ratio(
            "mission priority", mission_priority, shape
        )
        coverability_values = (
            coverable_detail_shape,
            coverable_detail_bits,
            coverable_detail_cell_count,
            coverable_mask_sha256,
        )
        if any(value is not None for value in coverability_values) and any(
            value is None for value in coverability_values
        ):
            raise ValueError("exact detail coverability must be supplied atomically")
        self.coverable_detail_shape: tuple[int, int] | None = None
        self.coverable_detail_bits: np.ndarray | None = None
        self.coverable_detail_cell_count: int | None = None
        self.coverable_mask_sha256: str | None = None
        self._observed_coverable_detail_bits: np.ndarray | None = None
        self.observed_coverable_detail_cell_count = 0
        self.observed_priority_detail_cell_count = 0
        self._coverable_priority_area_m2: float | None = None
        if coverable_detail_shape is not None:
            total = tile_provider.detail_cells_per_axis
            if coverable_detail_shape != (total, total):
                raise ValueError("coverable detail mask geometry differs from scene")
            unpacked = unpack_detail_mask(
                np.ascontiguousarray(coverable_detail_bits, dtype=np.uint8),
                coverable_detail_shape,
            )
            actual_count = int(unpacked.sum(dtype=np.int64))
            if (
                type(coverable_detail_cell_count) is not int
                or coverable_detail_cell_count != actual_count
                or mask_sha256(unpacked) != coverable_mask_sha256
            ):
                raise ValueError("coverable detail mask identity differs")
            packed = np.ascontiguousarray(coverable_detail_bits.copy(), dtype=np.uint8)
            packed.setflags(write=False)
            self.coverable_detail_shape = coverable_detail_shape
            self.coverable_detail_bits = packed
            self.coverable_detail_cell_count = coverable_detail_cell_count
            self.coverable_mask_sha256 = coverable_mask_sha256
            self._observed_coverable_detail_bits = np.zeros_like(packed)
            weighted_counts = unpacked.reshape(
                coarse_cells,
                _DETAIL_PER_GLOBAL,
                coarse_cells,
                _DETAIL_PER_GLOBAL,
            ).sum(axis=(1, 3), dtype=np.int64)
            self._coverable_priority_area_m2 = float(
                (
                    weighted_counts.astype(np.float64)
                    * self.mission_priority.astype(np.float64)
                ).sum(dtype=np.float64)
                * LOCAL_GEOMETRY.resolution_m**2
            )
        priority_values = (
            priority_detail_shape,
            priority_detail_bits,
            priority_detail_cell_count,
            priority_detail_mask_sha256,
        )
        if any(value is not None for value in priority_values) and any(
            value is None for value in priority_values
        ):
            raise ValueError("exact detail priority must be supplied atomically")
        self.priority_detail_shape: tuple[int, int] | None = None
        self.priority_detail_bits: np.ndarray | None = None
        self.priority_detail_cell_count: int | None = None
        self.priority_detail_mask_sha256: str | None = None
        if priority_detail_shape is not None:
            if self.coverable_detail_bits is None:
                raise ValueError("exact detail priority requires coverability")
            if priority_detail_shape != self.coverable_detail_shape:
                raise ValueError("priority detail mask geometry differs from scene")
            priority_unpacked = unpack_detail_mask(
                np.ascontiguousarray(priority_detail_bits, dtype=np.uint8),
                priority_detail_shape,
            )
            priority_count = int(priority_unpacked.sum(dtype=np.int64))
            coverable_unpacked = unpack_detail_mask(
                self.coverable_detail_bits,
                self.coverable_detail_shape,
            )
            if (
                type(priority_detail_cell_count) is not int
                or priority_detail_cell_count != priority_count
                or mask_sha256(priority_unpacked) != priority_detail_mask_sha256
                or bool((priority_unpacked & ~coverable_unpacked).any())
            ):
                raise ValueError("priority detail mask identity differs")
            packed_priority = np.ascontiguousarray(
                priority_detail_bits.copy(), dtype=np.uint8
            )
            packed_priority.setflags(write=False)
            self.priority_detail_shape = priority_detail_shape
            self.priority_detail_bits = packed_priority
            self.priority_detail_cell_count = priority_detail_cell_count
            self.priority_detail_mask_sha256 = priority_detail_mask_sha256
            self._coverable_priority_area_m2 = float(
                priority_count * LOCAL_GEOMETRY.resolution_m**2
            )
        self.forbidden_mask = np.ascontiguousarray(
            global_truth.forbidden_ratio > 0.0
        )
        self.mission_roi_ratio.setflags(write=False)
        self.mission_priority.setflags(write=False)
        self.forbidden_mask.setflags(write=False)
        self.visibility_estimator = NativeVisibilityEstimator(
            SensorGeometry(FORMAL_SENSOR_RANGE_M, FORMAL_SENSOR_FOV_RAD),
            resolution_m=geometry.resolution_m,
        )
        self.scene = scene
        self.tile_provider = tile_provider
        self._detail_tiles: dict[tuple[int, int], _ObservedDetailTile] = {}
        self._evidence_generation = 0
        self.coarse_obstacle_height_m = np.zeros(shape, dtype=np.float32)
        self.coarse_forbidden_ratio = np.zeros(shape, dtype=np.float32)

    @staticmethod
    def _coarse_ratio(
        name: str, value: np.ndarray, shape: tuple[int, int]
    ) -> np.ndarray:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        if (
            array.shape != shape
            or not np.isfinite(array).all()
            or ((array < 0.0) | (array > 1.0)).any()
        ):
            raise ValueError(
                f"{name} must be finite [{shape[0]},{shape[1]}] in [0,1]"
            )
        return array.copy()

    @property
    def allocated_detail_tiles(self) -> int:
        return len(self._detail_tiles)

    @property
    def evidence_generation(self) -> int:
        return self._evidence_generation

    def physical_evidence_sha256(self) -> str:
        """Hash sorted sparse tile identities and their observed layer bytes."""
        digest = sha256()
        layer_names = (
            "elevation_m",
            "physical_obstacle_ratio",
            "physical_obstacle_height_m",
            "forbidden_ratio",
            "valid_mask",
            "observation_age_s",
            "observation_quality",
            "observation_count",
        )
        for tile_identity in sorted(self._detail_tiles):
            digest.update(struct.pack("<ii", *tile_identity))
            tile = self._detail_tiles[tile_identity]
            for name in layer_names:
                values = np.ascontiguousarray(getattr(tile, name))
                if values.dtype.itemsize > 1:
                    values = np.ascontiguousarray(
                        values, dtype=values.dtype.newbyteorder("<")
                    )
                digest.update(values.tobytes(order="C"))
        return digest.hexdigest()

    def observed_coverable_mask_sha256(self) -> str:
        """Hash the exact row-major 0.2 m cells counted by coverage."""
        if (
            self.coverable_detail_shape is None
            or self._observed_coverable_detail_bits is None
        ):
            raise RuntimeError("exact observed coverability is unavailable")
        cell_count = math.prod(self.coverable_detail_shape)
        unpacked = np.unpackbits(
            self._observed_coverable_detail_bits,
            bitorder="big",
            count=cell_count,
        )
        return sha256(unpacked.tobytes(order="C")).hexdigest()

    def observed_coverable_detail_bits(self) -> np.ndarray:
        """Return a detached packed mask for offline denominator freezing."""
        if self._observed_coverable_detail_bits is None:
            raise RuntimeError("exact observed coverability is unavailable")
        return np.ascontiguousarray(
            self._observed_coverable_detail_bits.copy(), dtype=np.uint8
        )

    @property
    def remaining_coverable_detail_cell_count(self) -> int | None:
        if self.coverable_detail_cell_count is None:
            return None
        return (
            self.coverable_detail_cell_count
            - self.observed_coverable_detail_cell_count
        )

    @property
    def mission_area_m2(self) -> float | None:
        if self.coverable_detail_cell_count is None:
            return None
        return float(
            self.coverable_detail_cell_count
            * LOCAL_GEOMETRY.resolution_m**2
        )

    @property
    def priority_area_m2(self) -> float | None:
        return self._coverable_priority_area_m2

    @property
    def sensor(self) -> SensorGeometry:
        return self.visibility_estimator.sensor

    @property
    def resolution_m(self) -> float:
        return float(self.visibility_estimator.resolution_m)

    def prepare_hopper_trajectory_observation(
        self,
        points: tuple[HopperTrajectoryPoint, ...],
    ) -> HopperTrajectoryObservationPatch:
        """Rasterize an executed horizontal flight without mutating observation."""
        if not isinstance(points, tuple) or not points:
            raise ValueError("hopper trajectory must be a non-empty tuple")
        if any(not isinstance(point, HopperTrajectoryPoint) for point in points):
            raise ValueError("hopper trajectory points are invalid")
        times = tuple(float(point.time_s) for point in points)
        if any(right < left for left, right in zip(times, times[1:])):
            raise ValueError("hopper trajectory time regressed")

        xy = np.ascontiguousarray(
            [
                (float(point.pose_map.x_m), float(point.pose_map.y_m))
                for point in points
            ],
            dtype=np.float64,
        )
        radius_m = float(self.sensor.range_m)
        resolution_m = float(self.tile_provider.tile_geometry.resolution_m)
        tile_cells = self.tile_provider.tile_geometry.cells
        total = self.tile_provider.detail_cells_per_axis
        left, bottom, right, top = self.scene.base_canvas.bounds_m

        minimum_x = float(xy[:, 0].min()) - radius_m
        maximum_x = float(xy[:, 0].max()) + radius_m
        minimum_y = float(xy[:, 1].min()) - radius_m
        maximum_y = float(xy[:, 1].max()) + radius_m
        first_column = max(
            0,
            int(math.ceil((minimum_x - left) / resolution_m - 0.5)),
        )
        last_column = min(
            total - 1,
            int(math.floor((maximum_x - left) / resolution_m - 0.5)),
        )
        first_row = max(
            0,
            int(math.ceil((top - maximum_y) / resolution_m - 0.5)),
        )
        last_row = min(
            total - 1,
            int(math.floor((top - minimum_y) / resolution_m - 0.5)),
        )
        tile_patches: list[HopperTrajectoryTilePatch] = []
        if first_row <= last_row and first_column <= last_column:
            segments = tuple(zip(xy[:-1], xy[1:], strict=True))
            if not segments:
                segments = ((xy[0], xy[0]),)
            radius_squared = radius_m * radius_m
            for tile_row in range(
                first_row // tile_cells, last_row // tile_cells + 1
            ):
                global_rows = tile_row * tile_cells + np.arange(
                    tile_cells, dtype=np.int64
                )
                y = top - (
                    global_rows.astype(np.float64) + 0.5
                ) * resolution_m
                yy = y[:, None]
                for tile_column in range(
                    first_column // tile_cells,
                    last_column // tile_cells + 1,
                ):
                    global_columns = tile_column * tile_cells + np.arange(
                        tile_cells, dtype=np.int64
                    )
                    x = left + (
                        global_columns.astype(np.float64) + 0.5
                    ) * resolution_m
                    xx = x[None, :]
                    minimum_distance_squared = np.full(
                        (tile_cells, tile_cells), np.inf, dtype=np.float64
                    )
                    for start, end in segments:
                        dx = float(end[0] - start[0])
                        dy = float(end[1] - start[1])
                        length_squared = dx * dx + dy * dy
                        if length_squared == 0.0:
                            candidate = (
                                (xx - start[0]) ** 2
                                + (yy - start[1]) ** 2
                            )
                        else:
                            alpha = np.clip(
                                (
                                    (xx - start[0]) * dx
                                    + (yy - start[1]) * dy
                                )
                                / length_squared,
                                0.0,
                                1.0,
                            )
                            candidate = (
                                (xx - (start[0] + alpha * dx)) ** 2
                                + (yy - (start[1] + alpha * dy)) ** 2
                            )
                        np.minimum(
                            minimum_distance_squared,
                            candidate,
                            out=minimum_distance_squared,
                        )
                    visible = np.ascontiguousarray(
                        minimum_distance_squared <= radius_squared,
                        dtype=np.bool_,
                    )
                    if visible.any():
                        tile_patches.append(
                            HopperTrajectoryTilePatch(
                                tile_row=tile_row,
                                tile_column=tile_column,
                                visible_mask=visible,
                            )
                        )

        digest = sha256()
        digest.update(self.scene.scene_id.encode("ascii"))
        digest.update(struct.pack("<dd", radius_m, resolution_m))
        for point in points:
            digest.update(
                struct.pack(
                    "<ddddd",
                    float(point.time_s),
                    float(point.pose_map.x_m),
                    float(point.pose_map.y_m),
                    float(point.pose_map.elevation_m),
                    float(point.pose_map.yaw_rad),
                )
            )
        tiles = tuple(tile_patches)
        return HopperTrajectoryObservationPatch(
            scene_id=self.scene.scene_id,
            evidence_generation=self._evidence_generation,
            trajectory_sha256=digest.hexdigest(),
            tiles=tiles,
            visible_cells=sum(
                int(np.count_nonzero(tile.visible_mask)) for tile in tiles
            ),
        )

    def commit_hopper_trajectory_observation(
        self,
        patch: HopperTrajectoryObservationPatch,
        *,
        elapsed_s: float,
    ) -> ObservationDelta:
        """Atomically commit one prepared, unobstructed trajectory capsule."""
        if not isinstance(patch, HopperTrajectoryObservationPatch):
            raise ValueError("hopper trajectory observation patch is invalid")
        if patch.scene_id != self.scene.scene_id:
            raise ValueError("hopper trajectory observation scene differs")
        if patch.evidence_generation != self._evidence_generation:
            raise ValueError("hopper trajectory observation revision is stale")
        if (
            not isinstance(elapsed_s, (int, float))
            or isinstance(elapsed_s, bool)
            or not math.isfinite(float(elapsed_s))
            or float(elapsed_s) < 0.0
        ):
            raise ValueError("hopper trajectory elapsed time is invalid")
        elapsed = float(elapsed_s)
        next_tiles = {
            identity: tile.copy() for identity, tile in self._detail_tiles.items()
        }
        for tile in next_tiles.values():
            known = tile.valid_mask
            aged = tile.observation_age_s[known].astype(np.float64) + elapsed
            if (
                not np.isfinite(aged).all()
                or (aged > np.finfo(np.float32).max).any()
            ):
                raise ValueError("observation age would overflow")
            tile.observation_age_s[known] = np.ascontiguousarray(
                aged, dtype=np.float32
            )

        tile_cells = self.tile_provider.tile_geometry.cells
        total = self.tile_provider.detail_cells_per_axis
        visible_cells = 0
        new_global_rows: list[np.ndarray] = []
        new_global_columns: list[np.ndarray] = []
        affected_coarse: set[tuple[int, int]] = set()
        for tile_patch in patch.tiles:
            if not (
                0 <= tile_patch.tile_row < self.tile_provider.tiles_per_axis
                and 0 <= tile_patch.tile_column < self.tile_provider.tiles_per_axis
                and tile_patch.visible_mask.shape == (tile_cells, tile_cells)
            ):
                raise ValueError("hopper trajectory tile lies outside the scene")
            truth = self.tile_provider.tile(
                tile_patch.tile_row, tile_patch.tile_column
            )
            visible = np.ascontiguousarray(
                tile_patch.visible_mask & truth.valid_mask,
                dtype=np.bool_,
            )
            if not visible.any():
                continue
            visible_cells += int(np.count_nonzero(visible))
            identity = tile_patch.tile_row, tile_patch.tile_column
            tile = next_tiles.get(identity)
            if tile is None:
                tile = _ObservedDetailTile.empty(tile_cells)
                next_tiles[identity] = tile
            newly = visible & ~tile.valid_mask
            if newly.any():
                local_rows, local_columns = np.nonzero(newly)
                rows = (
                    tile_patch.tile_row * tile_cells
                    + local_rows.astype(np.int64)
                )
                columns = (
                    tile_patch.tile_column * tile_cells
                    + local_columns.astype(np.int64)
                )
                new_global_rows.append(rows)
                new_global_columns.append(columns)
            visible_rows, visible_columns = np.nonzero(visible)
            affected_coarse.update(
                zip(
                    (
                        (
                            tile_patch.tile_row * tile_cells
                            + visible_rows.astype(np.int64)
                        )
                        // _DETAIL_PER_GLOBAL
                    ).tolist(),
                    (
                        (
                            tile_patch.tile_column * tile_cells
                            + visible_columns.astype(np.int64)
                        )
                        // _DETAIL_PER_GLOBAL
                    ).tolist(),
                    strict=True,
                )
            )
            for target, source in (
                (tile.elevation_m, truth.elevation_m),
                (tile.physical_obstacle_ratio, truth.physical_obstacle_ratio),
                (
                    tile.physical_obstacle_height_m,
                    truth.physical_obstacle_height_m,
                ),
                (tile.forbidden_ratio, truth.forbidden_ratio),
            ):
                target[visible] = source[visible]
            tile.valid_mask[visible] = True
            tile.observation_quality[visible] = 1.0
            tile.observation_age_s[visible] = 0.0
            tile.observation_count[visible] = np.minimum(
                tile.observation_count[visible].astype(np.uint64) + 1,
                np.iinfo(np.uint32).max,
            ).astype(np.uint32)

        if new_global_rows:
            global_rows = np.concatenate(new_global_rows)
            global_columns = np.concatenate(new_global_columns)
        else:
            global_rows = np.zeros((0,), dtype=np.int64)
            global_columns = np.zeros((0,), dtype=np.int64)
        if (
            global_rows.size
            and (
                (global_rows < 0).any()
                or (global_columns < 0).any()
                or (global_rows >= total).any()
                or (global_columns >= total).any()
            )
        ):
            raise RuntimeError("hopper trajectory produced an invalid detail cell")

        next_observed_coverable_bits = (
            None
            if self._observed_coverable_detail_bits is None
            else self._observed_coverable_detail_bits.copy()
        )
        if self.coverable_detail_bits is None:
            coverable_new = np.ones(global_rows.shape, dtype=np.bool_)
        else:
            flat_indices = global_rows * total + global_columns
            byte_indices = flat_indices // 8
            bit_offsets = 7 - (flat_indices % 8)
            coverable_new = (
                self.coverable_detail_bits[byte_indices]
                & np.left_shift(np.uint8(1), bit_offsets.astype(np.uint8))
            ) != 0
        mission_rows = global_rows[coverable_new]
        mission_columns = global_columns[coverable_new]
        next_coverable_count = (
            self.observed_coverable_detail_cell_count + int(mission_rows.size)
        )
        if (
            self.coverable_detail_cell_count is not None
            and next_coverable_count > self.coverable_detail_cell_count
        ):
            raise RuntimeError("observed coverability exceeds frozen mask")
        if next_observed_coverable_bits is not None and mission_rows.size:
            flat_indices = mission_rows * total + mission_columns
            byte_indices = flat_indices // 8
            bit_offsets = 7 - (flat_indices % 8)
            bit_values = np.left_shift(
                np.uint8(1), bit_offsets.astype(np.uint8)
            )
            np.bitwise_or.at(
                next_observed_coverable_bits, byte_indices, bit_values
            )

        coarse_rows = mission_rows // _DETAIL_PER_GLOBAL
        coarse_columns = mission_columns // _DETAIL_PER_GLOBAL
        mission_weight = (
            np.ones(mission_rows.shape, dtype=np.float64)
            if self.coverable_detail_bits is not None
            else self.mission_roi_ratio[coarse_rows, coarse_columns].astype(
                np.float64
            )
        )
        mission_delta = float(
            mission_weight.sum(dtype=np.float64)
            * LOCAL_GEOMETRY.resolution_m**2
        )

        next_priority_count = self.observed_priority_detail_cell_count
        if self.priority_detail_bits is not None:
            flat_indices = global_rows * total + global_columns
            byte_indices = flat_indices // 8
            bit_offsets = 7 - (flat_indices % 8)
            priority_new = (
                self.priority_detail_bits[byte_indices]
                & np.left_shift(np.uint8(1), bit_offsets.astype(np.uint8))
            ) != 0
            newly_priority_count = int(np.count_nonzero(priority_new))
            next_priority_count += newly_priority_count
            if next_priority_count > self.priority_detail_cell_count:
                raise RuntimeError("observed priority exceeds frozen mask")
            priority_weighted_count = float(newly_priority_count)
        else:
            priority_weighted_count = float(
                (
                    self.mission_priority[coarse_rows, coarse_columns]
                    * mission_weight
                ).sum(dtype=np.float64)
            )
        priority_delta = float(
            priority_weighted_count * LOCAL_GEOMETRY.resolution_m**2
        )

        next_observed = self.observed.copy()
        next_obstacle_height = self.coarse_obstacle_height_m.copy()
        next_forbidden = self.coarse_forbidden_ratio.copy()
        for coarse_row, coarse_column in sorted(affected_coarse):
            update = self._coarse_update_from_tiles(
                next_tiles, coarse_row, coarse_column
            )
            if update is None:
                continue
            cell = coarse_row, coarse_column
            next_observed.elevation_m[cell] = update.elevation_m
            next_observed.physical_obstacle_ratio[cell] = (
                update.physical_obstacle_ratio
            )
            next_obstacle_height[cell] = update.physical_obstacle_height_m
            next_forbidden[cell] = update.forbidden_ratio
            next_observed.valid_mask[cell] = True
            next_observed.observation_age_s[cell] = update.observation_age_s
            next_observed.observation_quality[cell] = update.observation_quality
            next_observed.observation_count[cell] = update.observation_count

        self._detail_tiles = next_tiles
        self.observed = next_observed
        self.coarse_obstacle_height_m = next_obstacle_height
        self.coarse_forbidden_ratio = next_forbidden
        if next_observed_coverable_bits is not None:
            self._observed_coverable_detail_bits = next_observed_coverable_bits
            self.observed_coverable_detail_cell_count = next_coverable_count
        if self.priority_detail_bits is not None:
            self.observed_priority_detail_cell_count = next_priority_count
        self._evidence_generation += 1
        return ObservationDelta(
            visible_cells=visible_cells,
            newly_observed_cells=int(global_rows.size),
            mission_observed_delta_m2=mission_delta,
            priority_observed_delta_m2=priority_delta,
        )

    def _capture_hopper_observation_rollback_state(
        self,
    ) -> _HopperObservationRollbackState:
        """Capture references that remain untouched by copy-on-commit."""
        return _HopperObservationRollbackState(
            detail_tiles=self._detail_tiles,
            observed=self.observed,
            coarse_obstacle_height_m=self.coarse_obstacle_height_m,
            coarse_forbidden_ratio=self.coarse_forbidden_ratio,
            observed_coverable_detail_bits=self._observed_coverable_detail_bits,
            observed_coverable_detail_cell_count=(
                self.observed_coverable_detail_cell_count
            ),
            observed_priority_detail_cell_count=(
                self.observed_priority_detail_cell_count
            ),
            evidence_generation=self._evidence_generation,
        )

    def _restore_hopper_observation_rollback_state(
        self,
        state: _HopperObservationRollbackState,
    ) -> None:
        if not isinstance(state, _HopperObservationRollbackState):
            raise ValueError("hopper observation rollback state is invalid")
        self._detail_tiles = state.detail_tiles
        self.observed = state.observed
        self.coarse_obstacle_height_m = state.coarse_obstacle_height_m
        self.coarse_forbidden_ratio = state.coarse_forbidden_ratio
        self._observed_coverable_detail_bits = (
            state.observed_coverable_detail_bits
        )
        self.observed_coverable_detail_cell_count = (
            state.observed_coverable_detail_cell_count
        )
        self.observed_priority_detail_cell_count = (
            state.observed_priority_detail_cell_count
        )
        self._evidence_generation = state.evidence_generation

    def _coarse_update_from_tiles(
        self,
        tiles: dict[tuple[int, int], _ObservedDetailTile],
        coarse_row: int,
        coarse_column: int,
    ) -> _CoarseObservationUpdate | None:
        coarse_per_tile = self.tile_provider.tile_geometry.cells // _DETAIL_PER_GLOBAL
        tile_row, local_coarse_row = divmod(coarse_row, coarse_per_tile)
        tile_column, local_coarse_column = divmod(
            coarse_column, coarse_per_tile
        )
        tile = tiles.get((tile_row, tile_column))
        if tile is None:
            return None
        rows = slice(
            local_coarse_row * _DETAIL_PER_GLOBAL,
            (local_coarse_row + 1) * _DETAIL_PER_GLOBAL,
        )
        columns = slice(
            local_coarse_column * _DETAIL_PER_GLOBAL,
            (local_coarse_column + 1) * _DETAIL_PER_GLOBAL,
        )
        valid = tile.valid_mask[rows, columns]
        if not valid.all():
            return None
        elevation = tile.elevation_m[rows, columns]
        obstacle = tile.physical_obstacle_ratio[rows, columns]
        obstacle_height = tile.physical_obstacle_height_m[rows, columns]
        forbidden = tile.forbidden_ratio[rows, columns]
        age = tile.observation_age_s[rows, columns]
        quality = tile.observation_quality[rows, columns]
        count = tile.observation_count[rows, columns]
        return _CoarseObservationUpdate(
            row=coarse_row,
            column=coarse_column,
            elevation_m=np.float32(elevation.mean(dtype=np.float64)),
            physical_obstacle_ratio=np.float32(obstacle.max(initial=0.0)),
            physical_obstacle_height_m=np.float32(
                obstacle_height.max(initial=0.0)
            ),
            forbidden_ratio=np.float32(forbidden.max(initial=0.0)),
            observation_age_s=np.float32(age.min(initial=0.0)),
            observation_quality=np.float32(quality.max(initial=0.0)),
            observation_count=np.uint32(
                min(
                    int(count.sum(dtype=np.uint64)),
                    int(np.iinfo(np.uint32).max),
                )
            ),
        )

    def _existing_tile(
        self, tile_row: int, tile_column: int
    ) -> _ObservedDetailTile | None:
        return self._detail_tiles.get((tile_row, tile_column))

    def _window_slices(
        self,
        start_row: int,
        start_column: int,
        rows: int,
        columns: int | None = None,
    ):
        if columns is None:
            columns = rows
        tile_cells = self.tile_provider.tile_geometry.cells
        first_row = start_row // tile_cells
        last_row = (start_row + rows - 1) // tile_cells
        first_column = start_column // tile_cells
        last_column = (start_column + columns - 1) // tile_cells
        for tile_row in range(first_row, last_row + 1):
            for tile_column in range(first_column, last_column + 1):
                tile_start_row = tile_row * tile_cells
                tile_start_column = tile_column * tile_cells
                global_row0 = max(start_row, tile_start_row)
                global_row1 = min(start_row + rows, tile_start_row + tile_cells)
                global_column0 = max(start_column, tile_start_column)
                global_column1 = min(
                    start_column + columns, tile_start_column + tile_cells
                )
                window_slice = (
                    slice(global_row0 - start_row, global_row1 - start_row),
                    slice(
                        global_column0 - start_column,
                        global_column1 - start_column,
                    ),
                )
                tile_slice = (
                    slice(
                        global_row0 - tile_start_row,
                        global_row1 - tile_start_row,
                    ),
                    slice(
                        global_column0 - tile_start_column,
                        global_column1 - tile_start_column,
                    ),
                )
                yield tile_row, tile_column, window_slice, tile_slice

    def estimate_candidate_gains(
        self,
        observed_mask: np.ndarray,
        obstacle_ratio: np.ndarray,
        roi_ratio: np.ndarray,
        priority_weight: np.ndarray,
        candidate_cells: np.ndarray,
    ) -> np.ndarray:
        """Estimate gains from the sparse observed-only 0.2 m detail state."""
        self._validate_candidate_gain_grids(
            observed_mask, obstacle_ratio, roi_ratio, priority_weight
        )
        candidates = np.asarray(candidate_cells)
        if (
            candidates.dtype != np.dtype(np.int32)
            or candidates.ndim != 2
            or candidates.shape[1:] != (2,)
            or not candidates.flags.c_contiguous
        ):
            raise ValueError("candidate cells must be C-contiguous int32 [N,2]")
        if candidates.size and (
            (candidates < 0).any()
            or (candidates >= self.scene.base_canvas.geometry.cells).any()
        ):
            raise ValueError("candidate cell is outside the global grid")
        detail_candidates = np.ascontiguousarray(
            candidates * _DETAIL_PER_GLOBAL + _DETAIL_PER_GLOBAL // 2,
            dtype=np.int32,
        )
        return self._estimate_detail_candidate_gains(
            observed_mask,
            obstacle_ratio,
            roi_ratio,
            priority_weight,
            detail_candidates,
        )

    def estimate_candidate_gains_at_positions(
        self,
        observed_mask: np.ndarray,
        obstacle_ratio: np.ndarray,
        roi_ratio: np.ndarray,
        priority_weight: np.ndarray,
        candidate_positions_m: np.ndarray,
    ) -> np.ndarray:
        """Estimate gain at exact graph positions, never coarse cell centres."""
        self._validate_candidate_gain_grids(
            observed_mask, obstacle_ratio, roi_ratio, priority_weight
        )
        positions = np.asarray(candidate_positions_m)
        if (
            positions.dtype != np.dtype(np.float64)
            or positions.ndim != 2
            or positions.shape[1:] != (3,)
            or not positions.flags.c_contiguous
            or not np.isfinite(positions).all()
        ):
            raise ValueError(
                "candidate positions must be finite C-contiguous float64 [N,3]"
            )
        try:
            detail_candidates = np.ascontiguousarray(
                [
                    self.tile_provider.world_to_detail(
                        float(position[0]), float(position[1])
                    )
                    for position in positions
                ],
                dtype=np.int32,
            ).reshape((-1, 2))
        except ValueError as error:
            raise ValueError("candidate position is outside the detail grid") from error
        return self._estimate_detail_candidate_gains(
            observed_mask,
            obstacle_ratio,
            roi_ratio,
            priority_weight,
            detail_candidates,
        )

    def _validate_candidate_gain_grids(
        self,
        observed_mask: np.ndarray,
        obstacle_ratio: np.ndarray,
        roi_ratio: np.ndarray,
        priority_weight: np.ndarray,
    ) -> None:
        cells = self.scene.base_canvas.geometry.cells
        coarse_shape = (cells, cells)
        grids = (
            ("observed mask", observed_mask, np.dtype(np.bool_)),
            ("obstacle ratio", obstacle_ratio, np.dtype(np.float32)),
            ("ROI ratio", roi_ratio, np.dtype(np.float32)),
            ("priority weight", priority_weight, np.dtype(np.float32)),
        )
        for name, values, dtype in grids:
            if (
                not isinstance(values, np.ndarray)
                or values.shape != coarse_shape
                or values.dtype != dtype
                or not values.flags.c_contiguous
            ):
                raise ValueError(f"candidate {name} must match the global grid")
            if np.issubdtype(dtype, np.floating) and (
                not np.isfinite(values).all() or (values < 0.0).any()
            ):
                raise ValueError(f"candidate {name} values are invalid")

    def _estimate_detail_candidate_gains(
        self,
        observed_mask: np.ndarray,
        obstacle_ratio: np.ndarray,
        roi_ratio: np.ndarray,
        priority_weight: np.ndarray,
        detail_candidates: np.ndarray,
    ) -> np.ndarray:
        candidates = np.asarray(detail_candidates)
        if (
            candidates.dtype != np.dtype(np.int32)
            or candidates.ndim != 2
            or candidates.shape[1:] != (2,)
            or not candidates.flags.c_contiguous
        ):
            raise ValueError("detail candidate cells must be C-contiguous int32 [N,2]")
        total = self.tile_provider.detail_cells_per_axis
        if candidates.size and (
            (candidates < 0).any()
            or (candidates >= total).any()
        ):
            raise ValueError("candidate cell is outside the detail grid")
        if candidates.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        radius_cells = math.floor(self.sensor.range_m / self.resolution_m)
        start_row = max(0, int(candidates[:, 0].min()) - radius_cells)
        start_column = max(
            0, int(candidates[:, 1].min()) - radius_cells
        )
        end_row = min(
            total, int(candidates[:, 0].max()) + radius_cells + 1
        )
        end_column = min(
            total, int(candidates[:, 1].max()) + radius_cells + 1
        )
        rows = end_row - start_row
        columns = end_column - start_column
        detail_observed = np.zeros((rows, columns), dtype=np.bool_)
        detail_obstacle = np.zeros((rows, columns), dtype=np.float32)
        for tile_row, tile_column, window_slice, tile_slice in self._window_slices(
            start_row, start_column, rows, columns
        ):
            tile = self._existing_tile(tile_row, tile_column)
            if tile is None:
                continue
            valid = tile.valid_mask[tile_slice]
            detail_observed[window_slice] = valid
            destination = detail_obstacle[window_slice]
            source = tile.physical_obstacle_ratio[tile_slice]
            destination[valid] = source[valid]

        coarse_rows = np.arange(start_row, end_row) // _DETAIL_PER_GLOBAL
        coarse_columns = (
            np.arange(start_column, end_column) // _DETAIL_PER_GLOBAL
        )
        detail_roi = np.ascontiguousarray(
            roi_ratio[np.ix_(coarse_rows, coarse_columns)], dtype=np.float32
        )
        detail_priority = np.ascontiguousarray(
            priority_weight[np.ix_(coarse_rows, coarse_columns)],
            dtype=np.float32,
        )
        local_candidates = np.ascontiguousarray(
            candidates - np.asarray([start_row, start_column], np.int32),
            dtype=np.int32,
        )
        detail_gains = self.visibility_estimator.estimate_candidate_gains(
            detail_observed,
            detail_obstacle,
            detail_roi,
            detail_priority,
            local_candidates,
        )
        expected_shape = (candidates.shape[0], 2)
        if (
            not isinstance(detail_gains, np.ndarray)
            or detail_gains.shape != expected_shape
            or detail_gains.dtype != np.dtype(np.float32)
            or not detail_gains.flags.c_contiguous
            or not np.isfinite(detail_gains).all()
            or (detail_gains < 0.0).any()
        ):
            raise RuntimeError("detail candidate visibility result is invalid")
        area_scale = np.float32(
            (
                self.resolution_m
                / self.scene.base_canvas.geometry.resolution_m
            )
            ** 2
        )
        return np.ascontiguousarray(detail_gains * area_scale, dtype=np.float32)

    def _detail_window(
        self, pose: Pose2, cells: int
    ) -> tuple[int, int, int, int]:
        pose_row, pose_column = self.tile_provider.world_to_detail(
            pose.x_m, pose.y_m
        )
        total = self.tile_provider.detail_cells_per_axis
        if (
            type(cells) is not int
            or cells <= 0
            or cells > total
            or not 0 <= pose_row < total
            or not 0 <= pose_column < total
        ):
            raise ValueError("detail window lies outside the scene")
        start_row = min(max(pose_row - cells // 2, 0), total - cells)
        start_column = min(
            max(pose_column - cells // 2, 0), total - cells
        )
        return (
            start_row,
            start_column,
            pose_row - start_row,
            pose_column - start_column,
        )

    def observe(
        self, pose_cell: tuple[int, int], *, elapsed_s: float
    ) -> ObservationDelta:
        if (
            not isinstance(pose_cell, tuple)
            or len(pose_cell) != 2
            or any(type(value) is not int for value in pose_cell)
            or not 0 <= pose_cell[0] < self.scene.base_canvas.geometry.cells
            or not 0 <= pose_cell[1] < self.scene.base_canvas.geometry.cells
        ):
            raise ValueError("observation pose is outside the grid")
        x_m, y_m = self.scene.base_canvas.grid_center_world(*pose_cell)
        return self.observe_world(Pose2(x_m, y_m), elapsed_s=elapsed_s)

    def observation_cell_world(self, pose: Pose2) -> tuple[int, int]:
        """Return the authoritative 0.2 m cell used by detail visibility."""
        if not isinstance(pose, Pose2) or pose.frame_id != "map":
            raise ValueError("observation pose must be a map-frame Pose2")
        return self.tile_provider.world_to_detail(pose.x_m, pose.y_m)

    def observe_world_repeated(
        self,
        pose: Pose2,
        *,
        elapsed_steps_s: tuple[float, ...],
    ) -> ObservationDelta:
        """Preserve authoritative detail updates for one exact 0.2 m cell."""
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
        deltas = tuple(
            self.observe_world(pose, elapsed_s=float(elapsed_s))
            for elapsed_s in elapsed_steps_s
        )
        return ObservationDelta(
            visible_cells=sum(delta.visible_cells for delta in deltas),
            newly_observed_cells=sum(
                delta.newly_observed_cells for delta in deltas
            ),
            mission_observed_delta_m2=sum(
                delta.mission_observed_delta_m2 for delta in deltas
            ),
            priority_observed_delta_m2=sum(
                delta.priority_observed_delta_m2 for delta in deltas
            ),
        )

    def observe_world(self, pose: Pose2, *, elapsed_s: float) -> ObservationDelta:
        if not isinstance(pose, Pose2) or pose.frame_id != "map":
            raise ValueError("observation pose must be a map-frame Pose2")
        if (
            not isinstance(elapsed_s, (int, float))
            or isinstance(elapsed_s, bool)
            or not math.isfinite(float(elapsed_s))
            or elapsed_s < 0.0
        ):
            raise ValueError("observation elapsed time is invalid")
        elapsed = float(elapsed_s)
        window_cells = self.tile_provider.tile_geometry.cells
        start_row, start_column, pose_row, pose_column = self._detail_window(
            pose, window_cells
        )
        truth = self.tile_provider.read_window(
            start_row, start_column, cells=window_cells
        )
        revealed = self.visibility_estimator.reveal_from_pose(
            truth.physical_obstacle_ratio,
            (pose_row, pose_column),
        )
        if (
            not isinstance(revealed, np.ndarray)
            or revealed.dtype != np.dtype(np.bool_)
            or revealed.shape != truth.valid_mask.shape
        ):
            raise RuntimeError("sensor reveal result is invalid")
        visible = np.ascontiguousarray(revealed & truth.valid_mask)

        aged_updates: list[_AgedDetailUpdate] = []
        for tile in self._detail_tiles.values():
            known = tile.valid_mask
            aged = tile.observation_age_s[known].astype(np.float64) + elapsed
            if (
                not np.isfinite(aged).all()
                or (aged > np.finfo(np.float32).max).any()
            ):
                raise ValueError("observation age would overflow")
            aged_updates.append(
                _AgedDetailUpdate(
                    tile=tile,
                    known_mask=known,
                    observation_age_s=np.ascontiguousarray(
                        aged, dtype=np.float32
                    ),
                )
            )

        newly = np.zeros(visible.shape, dtype=np.bool_)
        visible_updates: list[_VisibleDetailUpdate] = []
        for tile_row, tile_column, window_slice, tile_slice in self._window_slices(
            start_row, start_column, window_cells
        ):
            visible_part = visible[window_slice]
            if not visible_part.any():
                continue
            tile = self._existing_tile(tile_row, tile_column)
            insert_tile = tile is None
            if tile is None:
                tile = _ObservedDetailTile.empty(
                    self.tile_provider.tile_geometry.cells
                )
            known_part = tile.valid_mask[tile_slice]
            newly_part = visible_part & ~known_part
            newly[window_slice] = newly_part
            count = tile.observation_count[tile_slice]
            updated_count = np.minimum(
                count[visible_part].astype(np.uint64) + 1,
                np.iinfo(np.uint32).max,
            ).astype(np.uint32)
            visible_updates.append(
                _VisibleDetailUpdate(
                    key=(tile_row, tile_column),
                    tile=tile,
                    insert_tile=insert_tile,
                    window_slice=window_slice,
                    tile_slice=tile_slice,
                    visible_mask=visible_part,
                    observation_count=updated_count,
                )
            )

        new_rows, new_columns = np.nonzero(newly)
        next_coverable_count = self.observed_coverable_detail_cell_count
        if self.coverable_detail_bits is None:
            newly_coverable = newly
        else:
            coverable = read_packed_detail_window(
                self.coverable_detail_bits,
                self.coverable_detail_shape,
                start_row=start_row,
                start_column=start_column,
                cells=window_cells,
            )
            newly_coverable = newly & coverable
            next_coverable_count += int(
                newly_coverable.sum(dtype=np.int64)
            )
            if (
                next_coverable_count > self.coverable_detail_cell_count
            ):
                raise RuntimeError("observed coverability exceeds frozen mask")
        mission_rows, mission_columns = np.nonzero(newly_coverable)
        global_rows = start_row + mission_rows
        global_columns = start_column + mission_columns
        coarse_rows = global_rows // _DETAIL_PER_GLOBAL
        coarse_columns = global_columns // _DETAIL_PER_GLOBAL
        mission_delta = float(
            (
                np.ones(mission_rows.shape, dtype=np.float64)
                if self.coverable_detail_bits is not None
                else self.mission_roi_ratio[coarse_rows, coarse_columns]
            ).sum(dtype=np.float64)
            * LOCAL_GEOMETRY.resolution_m**2
        )
        next_priority_count = self.observed_priority_detail_cell_count
        if self.priority_detail_bits is not None:
            priority_window = read_packed_detail_window(
                self.priority_detail_bits,
                self.priority_detail_shape,
                start_row=start_row,
                start_column=start_column,
                cells=window_cells,
            )
            newly_priority_count = int(
                (newly & priority_window).sum(dtype=np.int64)
            )
            next_priority_count += newly_priority_count
            if next_priority_count > self.priority_detail_cell_count:
                raise RuntimeError("observed priority exceeds frozen mask")
            priority_weighted_count = float(newly_priority_count)
        else:
            priority_weighted_count = float(
                (
                    self.mission_priority[coarse_rows, coarse_columns]
                    * (
                        1.0
                        if self.coverable_detail_bits is not None
                        else self.mission_roi_ratio[coarse_rows, coarse_columns]
                    )
                ).sum(dtype=np.float64)
            )
        priority_delta = float(
            priority_weighted_count * LOCAL_GEOMETRY.resolution_m**2
        )
        visible_rows, visible_columns = np.nonzero(visible)
        affected = set(
            zip(
                ((start_row + visible_rows) // _DETAIL_PER_GLOBAL).tolist(),
                ((start_column + visible_columns) // _DETAIL_PER_GLOBAL).tolist(),
                strict=True,
            )
        )
        coarse_updates = tuple(
            update
            for coarse_row, coarse_column in sorted(affected)
            for update in (
                self._prospective_coarse_update(
                    coarse_row,
                    coarse_column,
                    start_row=start_row,
                    start_column=start_column,
                    truth=truth,
                    visible=visible,
                    elapsed_s=elapsed,
                ),
            )
            if update is not None
        )
        delta = ObservationDelta(
            visible_cells=int(np.count_nonzero(visible)),
            newly_observed_cells=int(new_rows.size),
            mission_observed_delta_m2=mission_delta,
            priority_observed_delta_m2=priority_delta,
        )

        for update in aged_updates:
            update.tile.observation_age_s[update.known_mask] = (
                update.observation_age_s
            )
        for update in visible_updates:
            tile = update.tile
            for target, source in (
                (tile.elevation_m, truth.elevation_m),
                (
                    tile.physical_obstacle_ratio,
                    truth.physical_obstacle_ratio,
                ),
                (
                    tile.physical_obstacle_height_m,
                    truth.physical_obstacle_height_m,
                ),
                (tile.forbidden_ratio, truth.forbidden_ratio),
            ):
                target_view = target[update.tile_slice]
                source_view = source[update.window_slice]
                target_view[update.visible_mask] = source_view[
                    update.visible_mask
                ]
            tile.valid_mask[update.tile_slice][update.visible_mask] = True
            tile.observation_quality[update.tile_slice][
                update.visible_mask
            ] = 1.0
            tile.observation_age_s[update.tile_slice][
                update.visible_mask
            ] = 0.0
            tile.observation_count[update.tile_slice][
                update.visible_mask
            ] = update.observation_count
            if update.insert_tile:
                self._detail_tiles[update.key] = tile
        if self.coverable_detail_bits is not None:
            self.observed_coverable_detail_cell_count = next_coverable_count
            if mission_rows.size:
                if self._observed_coverable_detail_bits is None:
                    raise RuntimeError(
                        "observed coverability storage is unavailable"
                    )
                flat_indices = (
                    global_rows.astype(np.int64)
                    * int(self.coverable_detail_shape[1])
                    + global_columns.astype(np.int64)
                )
                byte_indices = flat_indices // 8
                bit_offsets = 7 - (flat_indices % 8)
                bit_values = np.left_shift(
                    np.uint8(1), bit_offsets.astype(np.uint8)
                )
                np.bitwise_or.at(
                    self._observed_coverable_detail_bits,
                    byte_indices,
                    bit_values,
                )
        if self.priority_detail_bits is not None:
            self.observed_priority_detail_cell_count = next_priority_count
        for update in coarse_updates:
            cell = update.row, update.column
            self.observed.elevation_m[cell] = update.elevation_m
            self.observed.physical_obstacle_ratio[cell] = (
                update.physical_obstacle_ratio
            )
            self.coarse_obstacle_height_m[cell] = (
                update.physical_obstacle_height_m
            )
            self.coarse_forbidden_ratio[cell] = update.forbidden_ratio
            self.observed.valid_mask[cell] = True
            self.observed.observation_age_s[cell] = update.observation_age_s
            self.observed.observation_quality[cell] = (
                update.observation_quality
            )
            self.observed.observation_count[cell] = update.observation_count
        self._evidence_generation += 1
        return delta

    def _prospective_coarse_update(
        self,
        row: int,
        column: int,
        *,
        start_row: int,
        start_column: int,
        truth: ProjectedScene,
        visible: np.ndarray,
        elapsed_s: float,
    ) -> _CoarseObservationUpdate | None:
        elevation = self._detail_block(row, column, "elevation_m").copy()
        obstacle = self._detail_block(
            row, column, "physical_obstacle_ratio"
        ).copy()
        obstacle_height = self._detail_block(
            row, column, "physical_obstacle_height_m"
        ).copy()
        forbidden = self._detail_block(
            row, column, "forbidden_ratio"
        ).copy()
        valid = self._detail_block(row, column, "valid_mask").copy()
        age = self._detail_block(
            row, column, "observation_age_s"
        ).copy()
        quality = self._detail_block(
            row, column, "observation_quality"
        ).copy()
        count = self._detail_count_block(row, column).copy()
        age[valid] = (
            age[valid].astype(np.float64) + elapsed_s
        ).astype(np.float32)

        block_start_row = row * _DETAIL_PER_GLOBAL
        block_start_column = column * _DETAIL_PER_GLOBAL
        overlap_row0 = max(block_start_row, start_row)
        overlap_column0 = max(block_start_column, start_column)
        overlap_row1 = min(
            block_start_row + _DETAIL_PER_GLOBAL,
            start_row + visible.shape[0],
        )
        overlap_column1 = min(
            block_start_column + _DETAIL_PER_GLOBAL,
            start_column + visible.shape[1],
        )
        if overlap_row0 < overlap_row1 and overlap_column0 < overlap_column1:
            block_slice = (
                slice(
                    overlap_row0 - block_start_row,
                    overlap_row1 - block_start_row,
                ),
                slice(
                    overlap_column0 - block_start_column,
                    overlap_column1 - block_start_column,
                ),
            )
            window_slice = (
                slice(overlap_row0 - start_row, overlap_row1 - start_row),
                slice(
                    overlap_column0 - start_column,
                    overlap_column1 - start_column,
                ),
            )
            visible_part = visible[window_slice]
            for target, source in (
                (elevation, truth.elevation_m),
                (obstacle, truth.physical_obstacle_ratio),
                (obstacle_height, truth.physical_obstacle_height_m),
                (forbidden, truth.forbidden_ratio),
            ):
                target_view = target[block_slice]
                source_view = source[window_slice]
                target_view[visible_part] = source_view[visible_part]
            valid_view = valid[block_slice]
            valid_view[visible_part] = True
            quality_view = quality[block_slice]
            quality_view[visible_part] = 1.0
            age_view = age[block_slice]
            age_view[visible_part] = 0.0
            count_view = count[block_slice]
            count_view[visible_part] = np.minimum(
                count_view[visible_part].astype(np.uint64) + 1,
                np.iinfo(np.uint32).max,
            ).astype(np.uint32)

        if not valid.all():
            return None
        return _CoarseObservationUpdate(
            row=row,
            column=column,
            elevation_m=np.float32(elevation[valid].mean(dtype=np.float64)),
            physical_obstacle_ratio=np.float32(
                obstacle[valid].max(initial=0.0)
            ),
            physical_obstacle_height_m=np.float32(
                obstacle_height[valid].max(initial=0.0)
            ),
            forbidden_ratio=np.float32(
                forbidden[valid].max(initial=0.0)
            ),
            observation_age_s=np.float32(age[valid].min(initial=0.0)),
            observation_quality=np.float32(
                quality[valid].max(initial=0.0)
            ),
            observation_count=np.uint32(
                min(
                    int(count[valid].sum(dtype=np.uint64)),
                    int(np.iinfo(np.uint32).max),
                )
            ),
        )

    def _detail_block(self, coarse_row: int, coarse_column: int, name: str) -> np.ndarray:
        if not (
            0 <= coarse_row < self.scene.base_canvas.geometry.cells
            and 0 <= coarse_column < self.scene.base_canvas.geometry.cells
        ):
            raise ValueError("coarse cell lies outside the scene")
        coarse_per_tile = (
            self.tile_provider.tile_geometry.cells // _DETAIL_PER_GLOBAL
        )
        tile_row, local_coarse_row = divmod(coarse_row, coarse_per_tile)
        tile_column, local_coarse_column = divmod(
            coarse_column, coarse_per_tile
        )
        tile = self._existing_tile(tile_row, tile_column)
        shape = (_DETAIL_PER_GLOBAL, _DETAIL_PER_GLOBAL)
        if tile is None:
            dtype = np.bool_ if name == "valid_mask" else np.float32
            return np.zeros(shape, dtype=dtype)
        row0 = local_coarse_row * _DETAIL_PER_GLOBAL
        column0 = local_coarse_column * _DETAIL_PER_GLOBAL
        return getattr(tile, name)[
            row0 : row0 + _DETAIL_PER_GLOBAL,
            column0 : column0 + _DETAIL_PER_GLOBAL,
        ]

    def detail_block_valid(
        self, coarse_row: int, coarse_column: int
    ) -> np.ndarray:
        return self._detail_block(
            coarse_row, coarse_column, "valid_mask"
        ).copy()

    def _detail_count_block(self, row: int, column: int) -> np.ndarray:
        coarse_per_tile = (
            self.tile_provider.tile_geometry.cells // _DETAIL_PER_GLOBAL
        )
        tile_row, local_row = divmod(row, coarse_per_tile)
        tile_column, local_column = divmod(column, coarse_per_tile)
        tile = self._existing_tile(tile_row, tile_column)
        if tile is None:
            return np.zeros(
                (_DETAIL_PER_GLOBAL, _DETAIL_PER_GLOBAL), dtype=np.uint32
            )
        row0 = local_row * _DETAIL_PER_GLOBAL
        column0 = local_column * _DETAIL_PER_GLOBAL
        return tile.observation_count[
            row0 : row0 + _DETAIL_PER_GLOBAL,
            column0 : column0 + _DETAIL_PER_GLOBAL,
        ]

    def _detail_value_at(self, x_m: float, y_m: float, name: str):
        row, column = self.tile_provider.world_to_detail(x_m, y_m)
        tile_cells = self.tile_provider.tile_geometry.cells
        tile_row, local_row = divmod(row, tile_cells)
        tile_column, local_column = divmod(column, tile_cells)
        tile = self._existing_tile(tile_row, tile_column)
        if tile is None:
            return False if name == "valid_mask" else 0
        return getattr(tile, name)[local_row, local_column]

    def detail_observed_at(self, x_m: float, y_m: float) -> bool:
        return bool(self._detail_value_at(x_m, y_m, "valid_mask"))

    def detail_observation_count_at(self, x_m: float, y_m: float) -> int:
        return int(self._detail_value_at(x_m, y_m, "observation_count"))

    def local_observation(self, pose: Pose2) -> LocalObservation:
        if not isinstance(pose, Pose2) or pose.frame_id != "map":
            raise ValueError("local observation pose must be map-frame")
        half = LOCAL_GEOMETRY.size_m / 2.0
        left, _, _, top = self.scene.base_canvas.bounds_m
        resolution = LOCAL_GEOMETRY.resolution_m
        start_column = math.floor(
            (pose.x_m - half + 0.5 * resolution - left) / resolution
        )
        start_row = math.floor(
            (top - (pose.y_m + half - 0.5 * resolution)) / resolution
        )
        cells = LOCAL_GEOMETRY.cells
        total = self.tile_provider.detail_cells_per_axis
        if (
            start_row < 0
            or start_column < 0
            or start_row + cells > total
            or start_column + cells > total
        ):
            raise ValueError("local observation lies outside the scene")
        elevation = np.zeros((cells, cells), dtype=np.float32)
        obstacle = np.zeros((cells, cells), dtype=np.float32)
        valid = np.zeros((cells, cells), dtype=np.bool_)
        for tile_row, tile_column, window_slice, tile_slice in self._window_slices(
            start_row, start_column, cells
        ):
            tile = self._existing_tile(tile_row, tile_column)
            if tile is None:
                continue
            local_valid = tile.valid_mask[tile_slice]
            valid[window_slice] = local_valid
            elevation_part = elevation[window_slice]
            obstacle_part = obstacle[window_slice]
            elevation_part[local_valid] = tile.elevation_m[tile_slice][local_valid]
            obstacle_part[local_valid] = tile.physical_obstacle_ratio[tile_slice][
                local_valid
            ]
        return LocalObservation(
            canvas_id=self.scene.base_canvas.identity,
            bounds_m=(
                pose.x_m - half,
                pose.y_m - half,
                pose.x_m + half,
                pose.y_m + half,
            ),
            elevation_m=elevation,
            observed_mask=valid,
            physical_obstacle_ratio=obstacle,
        )

    def planning_observation(self, pose: Pose2) -> DetailObservedWindow:
        """Return the observed-only 64 m map consumed by C++ local planning."""
        cells = self.tile_provider.tile_geometry.cells
        start_row, start_column, _, _ = self._detail_window(pose, cells)
        shape = (cells, cells)
        float_names = (
            "elevation_m",
            "physical_obstacle_ratio",
            "physical_obstacle_height_m",
            "forbidden_ratio",
            "observation_age_s",
            "observation_quality",
        )
        arrays = {name: np.zeros(shape, np.float32) for name in float_names}
        valid = np.zeros(shape, np.bool_)
        count = np.zeros(shape, np.uint32)
        for tile_row, tile_column, window_slice, tile_slice in self._window_slices(
            start_row, start_column, cells
        ):
            tile = self._existing_tile(tile_row, tile_column)
            if tile is None:
                continue
            local_valid = tile.valid_mask[tile_slice]
            valid[window_slice] = local_valid
            for name in float_names:
                destination = arrays[name][window_slice]
                source = getattr(tile, name)[tile_slice]
                destination[local_valid] = source[local_valid]
            destination_count = count[window_slice]
            source_count = tile.observation_count[tile_slice]
            destination_count[local_valid] = source_count[local_valid]
        truth_window = self.tile_provider.read_window(
            start_row, start_column, cells=cells
        )
        return DetailObservedWindow(
            canvas=truth_window.canvas,
            valid_mask=valid,
            observation_count=count,
            **arrays,
        )

__all__ = [
    "DetailObservedWindow",
    "HopperTrajectoryObservationPatch",
    "HopperTrajectoryPoint",
    "HopperTrajectoryTilePatch",
    "MultiresSensorObservationState",
]
