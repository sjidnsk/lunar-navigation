from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lunar_policy_training.environment import formal_builder
from lunar_policy_training.polar_data.raster import GridGeometry, MapCanvas


def test_candidate_boundary_evidence_hash_materializes_before_reading_raw_hash() -> None:
    events: list[str] = []

    class SensorState:
        @staticmethod
        def materialize_all_observation_ages() -> None:
            events.append("materialize")

        @staticmethod
        def physical_evidence_sha256() -> str:
            events.append("raw")
            return "a" * 64

        @staticmethod
        def physical_evidence_identity_sha256() -> str:
            events.append("rolling")
            return "b" * 64

    assert formal_builder._candidate_boundary_evidence_sha256(SensorState()) == (
        "a" * 64
    )
    assert events == ["materialize", "raw"]


def test_planner_grid_projection_reuses_fixed_geometry_indices() -> None:
    source = MapCanvas(
        "a" * 64,
        (0.0, 0.0, 12.0, 12.0),
        GridGeometry(size_m=12.0, resolution_m=4.0, cells=3),
    )
    target = MapCanvas(
        "b" * 64,
        (0.0, 0.0, 12.0, 12.0),
        GridGeometry(size_m=12.0, resolution_m=2.0, cells=6),
    )
    values = np.arange(9, dtype=np.float32).reshape(3, 3)

    projection = formal_builder._planner_grid_projection(source, target)
    first = formal_builder._project_task_grid_to_planner_level(
        values,
        source=source,
        target=target,
        fill_value=np.float32(0.0),
        projection=projection,
    )
    second = formal_builder._project_task_grid_to_planner_level(
        values + np.float32(10.0),
        source=source,
        target=target,
        fill_value=np.float32(0.0),
        projection=projection,
    )

    np.testing.assert_array_equal(
        first, np.repeat(np.repeat(values, 2, axis=0), 2, axis=1)
    )
    np.testing.assert_array_equal(second, first + np.float32(10.0))
    assert not projection.source_rows.flags.writeable
    assert not projection.source_columns.flags.writeable


def test_planner_grid_projection_maps_dirty_source_cells_to_all_target_cells() -> None:
    source = MapCanvas(
        "c" * 64,
        (0.0, 0.0, 12.0, 12.0),
        GridGeometry(size_m=12.0, resolution_m=4.0, cells=3),
    )
    target = MapCanvas(
        "d" * 64,
        (0.0, 0.0, 12.0, 12.0),
        GridGeometry(size_m=12.0, resolution_m=2.0, cells=6),
    )
    projection = formal_builder._planner_grid_projection(source, target)

    target_flat, source_flat = (
        formal_builder._planner_patch_indices_for_source_cells(
            projection,
            np.asarray((0, 8), dtype=np.intp),
        )
    )

    np.testing.assert_array_equal(
        target_flat,
        np.asarray((0, 1, 6, 7, 28, 29, 34, 35), dtype=np.intp),
    )
    np.testing.assert_array_equal(
        source_flat,
        np.asarray((0, 0, 0, 0, 8, 8, 8, 8), dtype=np.intp),
    )


def test_persistent_planner_map_patches_only_dirty_projected_cells() -> None:
    source = MapCanvas(
        "e" * 64,
        (0.0, 0.0, 12.0, 12.0),
        GridGeometry(size_m=12.0, resolution_m=4.0, cells=3),
    )
    target = MapCanvas(
        "f" * 64,
        (0.0, 0.0, 12.0, 12.0),
        GridGeometry(size_m=12.0, resolution_m=2.0, cells=6),
    )
    elevation = np.arange(9, dtype=np.float32).reshape(3, 3)
    valid = np.ones((3, 3), dtype=np.bool_)
    obstacle = (elevation % 2).astype(np.float32)
    obstacle_height = elevation + np.float32(10.0)
    forbidden = (elevation % 3 == 0).astype(np.float32)
    age = elevation + np.float32(20.0)
    quality = elevation + np.float32(30.0)
    count = np.arange(9, dtype=np.uint32).reshape(3, 3)
    observed = SimpleNamespace(
        canvas=source,
        elevation_m=elevation,
        valid_mask=valid,
        physical_obstacle_ratio=obstacle,
        observation_age_s=age,
        observation_quality=quality,
        observation_count=count,
    )

    class SensorState:
        def __init__(self) -> None:
            self.observed = observed
            self.coarse_obstacle_height_m = obstacle_height
            self.coarse_forbidden_ratio = forbidden
            self._dirty = (0, 8)

        def consume_dirty_coarse_cell_ids(self) -> tuple[int, ...]:
            dirty = self._dirty
            self._dirty = ()
            return dirty

    sensor = SensorState()
    episode = SimpleNamespace(
        sensor_state=sensor,
        _planner_global_canvas=target,
        _planner_grid_projection=formal_builder._planner_grid_projection(
            source, target
        ),
    )
    expected = formal_builder._grid_map(
        canvas=target,
        frame_id="map",
        elevation_m=formal_builder._project_task_grid_to_planner_level(
            elevation,
            source=source,
            target=target,
            fill_value=np.float32(0.0),
            projection=episode._planner_grid_projection,
        ),
        valid_mask=formal_builder._project_task_grid_to_planner_level(
            valid,
            source=source,
            target=target,
            fill_value=False,
            projection=episode._planner_grid_projection,
        ),
        physical_obstacle_ratio=formal_builder._project_task_grid_to_planner_level(
            obstacle,
            source=source,
            target=target,
            fill_value=np.float32(0.0),
            projection=episode._planner_grid_projection,
        ),
        physical_obstacle_height_m=(
            formal_builder._project_task_grid_to_planner_level(
                obstacle_height,
                source=source,
                target=target,
                fill_value=np.float32(0.0),
                projection=episode._planner_grid_projection,
            )
        ),
        forbidden_ratio=formal_builder._project_task_grid_to_planner_level(
            forbidden,
            source=source,
            target=target,
            fill_value=np.float32(0.0),
            projection=episode._planner_grid_projection,
        ),
        observation_age_s=formal_builder._project_task_grid_to_planner_level(
            age,
            source=source,
            target=target,
            fill_value=np.float32(0.0),
            projection=episode._planner_grid_projection,
        ),
        observation_quality=formal_builder._project_task_grid_to_planner_level(
            quality,
            source=source,
            target=target,
            fill_value=np.float32(0.0),
            projection=episode._planner_grid_projection,
        ),
        observation_count=formal_builder._project_task_grid_to_planner_level(
            count,
            source=source,
            target=target,
            fill_value=np.uint32(0),
            projection=episode._planner_grid_projection,
        ),
        stamp_ns=2,
    )
    planner_map = formal_builder._grid_map(
        canvas=target,
        frame_id="map",
        elevation_m=np.flipud(
            np.asarray(expected.layers["elevation"].values).reshape(6, 6)
        ),
        valid_mask=np.flipud(
            np.asarray(expected.layers["valid_mask"].values).reshape(6, 6)
        ).astype(np.bool_),
        physical_obstacle_ratio=np.flipud(
            np.asarray(expected.layers["obstacle"].values).reshape(6, 6)
        ).astype(np.float32),
        physical_obstacle_height_m=np.flipud(
            np.asarray(expected.layers["obstacle_height"].values).reshape(6, 6)
        ),
        forbidden_ratio=np.flipud(
            np.asarray(expected.layers["forbidden"].values).reshape(6, 6)
        ).astype(np.float32),
        observation_age_s=np.flipud(
            np.asarray(expected.layers["observation_age_s"].values).reshape(6, 6)
        ),
        observation_quality=np.flipud(
            np.asarray(expected.layers["observation_quality"].values).reshape(6, 6)
        ),
        observation_count=np.flipud(
            np.asarray(expected.layers["observation_count"].values).reshape(6, 6)
        ),
        stamp_ns=1,
    )
    target_north_flat, _ = formal_builder._planner_patch_indices_for_source_cells(
        episode._planner_grid_projection,
        np.asarray((0, 8), dtype=np.intp),
    )
    north_rows = target_north_flat // target.geometry.cells
    south_flat = np.ascontiguousarray(
        (target.geometry.cells - 1 - north_rows) * target.geometry.cells
        + target_north_flat % target.geometry.cells,
        dtype=np.intp,
    )
    order = np.argsort(south_flat, kind="stable")
    corrupt_indices = np.ascontiguousarray(south_flat[order], dtype=np.uint32)
    for name in (
        "elevation",
        "valid_mask",
        "obstacle",
        "obstacle_height",
        "forbidden",
        "observation_age_s",
        "observation_quality",
        "observation_count",
    ):
        values = np.asarray(planner_map.layers[name].values)
        planner_map.patch_layer_flat_indices(
            name,
            corrupt_indices,
            np.zeros(corrupt_indices.shape, dtype=values.dtype),
        )

    formal_builder.FormalEpisode._patch_persistent_planner_global_map(
        episode, planner_map, stamp_ns=2
    )

    for name in (
        "elevation",
        "valid_mask",
        "obstacle",
        "obstacle_height",
        "forbidden",
        "observation_age_s",
        "observation_quality",
        "observation_count",
    ):
        np.testing.assert_array_equal(
            np.asarray(planner_map.layers[name].values),
            np.asarray(expected.layers[name].values),
        )
    assert planner_map.stamp.nanoseconds_since_epoch == 2
    assert sensor.consume_dirty_coarse_cell_ids() == ()
