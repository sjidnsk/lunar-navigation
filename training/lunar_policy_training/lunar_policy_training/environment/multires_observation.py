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
from ..polar_data.multires_scene import MultiResolutionScene, SceneTileProvider
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
        shape = (GLOBAL_GEOMETRY.cells, GLOBAL_GEOMETRY.cells)
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
        self.observed_coverable_detail_cell_count = 0
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
            weighted_counts = unpacked.reshape(
                GLOBAL_GEOMETRY.cells,
                _DETAIL_PER_GLOBAL,
                GLOBAL_GEOMETRY.cells,
                _DETAIL_PER_GLOBAL,
            ).sum(axis=(1, 3), dtype=np.int64)
            self._coverable_priority_area_m2 = float(
                (
                    weighted_counts.astype(np.float64)
                    * self.mission_priority.astype(np.float64)
                ).sum(dtype=np.float64)
                * LOCAL_GEOMETRY.resolution_m**2
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
            raise ValueError(f"{name} must be finite [256,256] in [0,1]")
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

    def _tile(self, tile_row: int, tile_column: int) -> _ObservedDetailTile:
        key = tile_row, tile_column
        tile = self._detail_tiles.get(key)
        if tile is None:
            tile = _ObservedDetailTile.empty(self.tile_provider.tile_geometry.cells)
            self._detail_tiles[key] = tile
        return tile

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
            or (candidates >= GLOBAL_GEOMETRY.cells).any()
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

    @staticmethod
    def _validate_candidate_gain_grids(
        observed_mask: np.ndarray,
        obstacle_ratio: np.ndarray,
        roi_ratio: np.ndarray,
        priority_weight: np.ndarray,
    ) -> None:
        coarse_shape = (GLOBAL_GEOMETRY.cells, GLOBAL_GEOMETRY.cells)
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
            (self.resolution_m / GLOBAL_GEOMETRY.resolution_m) ** 2
        )
        return np.ascontiguousarray(detail_gains * area_scale, dtype=np.float32)

    def _detail_window(
        self, pose: Pose2, cells: int
    ) -> tuple[int, int, int, int]:
        pose_row, pose_column = self.tile_provider.world_to_detail(
            pose.x_m, pose.y_m
        )
        start_row = pose_row - cells // 2
        start_column = pose_column - cells // 2
        total = self.tile_provider.detail_cells_per_axis
        if (
            start_row < 0
            or start_column < 0
            or start_row + cells > total
            or start_column + cells > total
        ):
            raise ValueError("detail window lies outside the scene")
        return start_row, start_column, pose_row - start_row, pose_column - start_column

    def observe(
        self, pose_cell: tuple[int, int], *, elapsed_s: float
    ) -> ObservationDelta:
        if (
            not isinstance(pose_cell, tuple)
            or len(pose_cell) != 2
            or any(type(value) is not int for value in pose_cell)
            or not 0 <= pose_cell[0] < GLOBAL_GEOMETRY.cells
            or not 0 <= pose_cell[1] < GLOBAL_GEOMETRY.cells
        ):
            raise ValueError("observation pose is outside the grid")
        x_m, y_m = self.scene.base_canvas.grid_center_world(*pose_cell)
        return self.observe_world(Pose2(x_m, y_m), elapsed_s=elapsed_s)

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
        for tile in self._detail_tiles.values():
            known = tile.valid_mask
            aged = tile.observation_age_s[known].astype(np.float64) + float(
                elapsed_s
            )
            if (
                not np.isfinite(aged).all()
                or (aged > np.finfo(np.float32).max).any()
            ):
                raise ValueError("observation age would overflow")
            tile.observation_age_s[known] = aged.astype(np.float32)

        window_cells = self.tile_provider.tile_geometry.cells
        start_row, start_column, pose_row, pose_column = self._detail_window(
            pose, window_cells
        )
        truth = self.tile_provider.read_window(
            start_row, start_column, cells=window_cells
        )
        visible = self.visibility_estimator.reveal_from_pose(
            truth.physical_obstacle_ratio,
            (pose_row, pose_column),
        )
        visible &= truth.valid_mask
        newly = np.zeros(visible.shape, dtype=np.bool_)
        for tile_row, tile_column, window_slice, tile_slice in self._window_slices(
            start_row, start_column, window_cells
        ):
            visible_part = visible[window_slice]
            if not visible_part.any():
                continue
            tile = self._tile(tile_row, tile_column)
            known_part = tile.valid_mask[tile_slice]
            newly_part = visible_part & ~known_part
            newly[window_slice] = newly_part
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
                target_view = target[tile_slice]
                target_view[visible_part] = source[window_slice][visible_part]
            known_part[visible_part] = True
            quality = tile.observation_quality[tile_slice]
            quality[visible_part] = 1.0
            age = tile.observation_age_s[tile_slice]
            age[visible_part] = 0.0
            count = tile.observation_count[tile_slice]
            count[visible_part] = np.minimum(
                count[visible_part].astype(np.uint64) + 1,
                np.iinfo(np.uint32).max,
            ).astype(np.uint32)

        new_rows, new_columns = np.nonzero(newly)
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
            self.observed_coverable_detail_cell_count += int(
                newly_coverable.sum(dtype=np.int64)
            )
            if (
                self.observed_coverable_detail_cell_count
                > self.coverable_detail_cell_count
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
        priority_delta = float(
            (
                self.mission_priority[coarse_rows, coarse_columns]
                * (
                    1.0
                    if self.coverable_detail_bits is not None
                    else self.mission_roi_ratio[coarse_rows, coarse_columns]
                )
            ).sum(dtype=np.float64)
            * LOCAL_GEOMETRY.resolution_m**2
        )
        visible_rows, visible_columns = np.nonzero(visible)
        affected = set(
            zip(
                ((start_row + visible_rows) // _DETAIL_PER_GLOBAL).tolist(),
                ((start_column + visible_columns) // _DETAIL_PER_GLOBAL).tolist(),
                strict=True,
            )
        )
        for coarse_row, coarse_column in affected:
            self._aggregate_coarse_cell(coarse_row, coarse_column)
        delta = ObservationDelta(
            visible_cells=int(np.count_nonzero(visible)),
            newly_observed_cells=int(new_rows.size),
            mission_observed_delta_m2=mission_delta,
            priority_observed_delta_m2=priority_delta,
        )
        self._evidence_generation += 1
        return delta

    def _detail_block(self, coarse_row: int, coarse_column: int, name: str) -> np.ndarray:
        if not (
            0 <= coarse_row < GLOBAL_GEOMETRY.cells
            and 0 <= coarse_column < GLOBAL_GEOMETRY.cells
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

    def _aggregate_coarse_cell(self, row: int, column: int) -> None:
        valid = self._detail_block(row, column, "valid_mask")
        if not valid.all():
            return
        elevation = self._detail_block(row, column, "elevation_m")
        obstacle = self._detail_block(
            row, column, "physical_obstacle_ratio"
        )
        obstacle_height = self._detail_block(
            row, column, "physical_obstacle_height_m"
        )
        forbidden = self._detail_block(row, column, "forbidden_ratio")
        age = self._detail_block(row, column, "observation_age_s")
        quality = self._detail_block(row, column, "observation_quality")
        count = self._detail_count_block(row, column)
        self.observed.elevation_m[row, column] = np.float32(
            elevation[valid].mean(dtype=np.float64)
        )
        self.observed.physical_obstacle_ratio[row, column] = np.float32(
            obstacle[valid].max(initial=0.0)
        )
        self.coarse_obstacle_height_m[row, column] = np.float32(
            obstacle_height[valid].max(initial=0.0)
        )
        self.coarse_forbidden_ratio[row, column] = np.float32(
            forbidden[valid].max(initial=0.0)
        )
        self.observed.valid_mask[row, column] = True
        self.observed.observation_age_s[row, column] = np.float32(
            age[valid].min(initial=0.0)
        )
        self.observed.observation_quality[row, column] = np.float32(
            quality[valid].max(initial=0.0)
        )
        self.observed.observation_count[row, column] = np.uint32(
            min(
                int(count[valid].sum(dtype=np.uint64)),
                int(np.iinfo(np.uint32).max),
            )
        )

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

__all__ = ["DetailObservedWindow", "MultiresSensorObservationState"]
