from __future__ import annotations

import numpy as np
import pytest

from lunar_policy_training.environment.coverability import (
    build_streamed_detail_coverability,
    unpack_detail_mask,
)
from lunar_policy_training.polar_data.hazards import (
    FORMAL_GENERATOR_VERSION,
    generate_vector_hazard_scene,
)
from lunar_policy_training.polar_data.multires_scene import (
    LOCAL_DETAIL_PROVENANCE,
    MultiResolutionScene,
    SceneTileProvider,
    project_vector_scene,
)
from lunar_policy_training.polar_data.raster import GridGeometry, MapCanvas


def _canvas() -> MapCanvas:
    geometry = GridGeometry(size_m=16.0, resolution_m=2.0, cells=8)
    return MapCanvas("a" * 64, (100.0, 200.0, 116.0, 216.0), geometry)


def test_formal_vector_streams_are_independent_of_other_object_counts() -> None:
    canvas = _canvas()
    baseline = generate_vector_hazard_scene(
        canvas.window_sha256,
        408000,
        canvas=canvas,
        rock_count=3,
        crater_count=2,
        no_go_count=2,
    )

    more_rocks = generate_vector_hazard_scene(
        canvas.window_sha256,
        408000,
        canvas=canvas,
        rock_count=5,
        crater_count=2,
        no_go_count=2,
    )
    more_craters = generate_vector_hazard_scene(
        canvas.window_sha256,
        408000,
        canvas=canvas,
        rock_count=3,
        crater_count=4,
        no_go_count=2,
    )
    more_no_go = generate_vector_hazard_scene(
        canvas.window_sha256,
        408000,
        canvas=canvas,
        rock_count=3,
        crater_count=2,
        no_go_count=4,
    )

    assert baseline.generator_version == FORMAL_GENERATOR_VERSION
    assert baseline.rocks == more_rocks.rocks[:3]
    assert baseline.craters == more_rocks.craters
    assert baseline.no_go_polygons == more_rocks.no_go_polygons
    assert baseline.rocks == more_craters.rocks
    assert baseline.craters == more_craters.craters[:2]
    assert baseline.no_go_polygons == more_craters.no_go_polygons
    assert baseline.rocks == more_no_go.rocks
    assert baseline.craters == more_no_go.craters
    assert baseline.no_go_polygons == more_no_go.no_go_polygons[:2]


def test_global_and_local_projection_share_one_vector_scene_and_provenance() -> None:
    canvas = _canvas()
    scene = generate_vector_hazard_scene(
        canvas.window_sha256,
        408000,
        canvas=canvas,
        rock_count=4,
        crater_count=2,
        no_go_count=1,
    )
    rock = scene.rocks[0]
    local_geometry = GridGeometry(size_m=4.0, resolution_m=0.5, cells=8)
    left = canvas.bounds_m[0] + round((rock.x_m - 2.0 - canvas.bounds_m[0]) / 0.5) * 0.5
    bottom = canvas.bounds_m[1] + round((rock.y_m - 2.0 - canvas.bounds_m[1]) / 0.5) * 0.5
    left = min(max(left, canvas.bounds_m[0]), canvas.bounds_m[2] - 4.0)
    bottom = min(max(bottom, canvas.bounds_m[1]), canvas.bounds_m[3] - 4.0)
    local_canvas = MapCanvas(
        canvas.window_sha256,
        (left, bottom, left + 4.0, bottom + 4.0),
        local_geometry,
    )

    global_projection = project_vector_scene(scene, canvas)
    local_projection = project_vector_scene(scene, local_canvas)
    global_row, global_column = canvas.world_to_grid(rock.x_m, rock.y_m)

    assert global_projection.vector_sha256 == scene.vector_sha256
    assert local_projection.vector_sha256 == scene.vector_sha256
    assert global_projection.local_detail_provenance == LOCAL_DETAIL_PROVENANCE
    assert local_projection.local_detail_provenance == LOCAL_DETAIL_PROVENANCE
    assert global_projection.physical_obstacle_ratio[global_row, global_column] > 0.0
    assert global_projection.physical_obstacle_height_m[global_row, global_column] > 0.0
    assert np.max(local_projection.physical_obstacle_ratio) > 0.0
    assert np.max(local_projection.physical_obstacle_height_m) > 0.0


def test_overlapping_aligned_projections_are_bit_identical() -> None:
    canvas = _canvas()
    scene = generate_vector_hazard_scene(
        canvas.window_sha256,
        408001,
        canvas=canvas,
        rock_count=8,
        crater_count=4,
        no_go_count=3,
    )
    geometry = GridGeometry(size_m=8.0, resolution_m=0.5, cells=16)
    west = MapCanvas(canvas.window_sha256, (100.0, 204.0, 108.0, 212.0), geometry)
    east = MapCanvas(canvas.window_sha256, (104.0, 204.0, 112.0, 212.0), geometry)

    west_projection = project_vector_scene(scene, west)
    east_projection = project_vector_scene(scene, east)

    np.testing.assert_array_equal(
        west_projection.crater_elevation_delta_m[:, 8:],
        east_projection.crater_elevation_delta_m[:, :8],
    )
    np.testing.assert_array_equal(
        west_projection.physical_obstacle_ratio[:, 8:],
        east_projection.physical_obstacle_ratio[:, :8],
    )
    np.testing.assert_array_equal(
        west_projection.physical_obstacle_height_m[:, 8:],
        east_projection.physical_obstacle_height_m[:, :8],
    )
    np.testing.assert_array_equal(
        west_projection.forbidden_ratio[:, 8:],
        east_projection.forbidden_ratio[:, :8],
    )


def test_multires_scene_interpolates_locked_dem_before_adding_same_hazards() -> None:
    canvas = _canvas()
    base = np.add.outer(
        np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32)
    )
    vector = generate_vector_hazard_scene(
        canvas.window_sha256,
        408002,
        canvas=canvas,
        rock_count=0,
        crater_count=0,
        no_go_count=0,
    )
    multires = MultiResolutionScene(canvas, base, np.ones((8, 8), bool), vector)
    fine_geometry = GridGeometry(size_m=4.0, resolution_m=0.5, cells=8)
    fine_canvas = MapCanvas(
        canvas.window_sha256,
        (104.0, 206.0, 108.0, 210.0),
        fine_geometry,
    )

    fine = multires.project(fine_canvas)

    assert fine.elevation_m.shape == (8, 8)
    assert np.isfinite(fine.elevation_m).all()
    assert np.ptp(fine.elevation_m) > 0.0
    assert fine.local_detail_provenance == LOCAL_DETAIL_PROVENANCE


def test_aligned_projection_preserves_sparse_validity_without_recursive_erosion() -> None:
    canvas = _canvas()
    base = np.add.outer(
        np.arange(8, dtype=np.float32), np.arange(8, dtype=np.float32)
    )
    valid = np.ones((8, 8), dtype=bool)
    valid[3, 4] = False
    vector = generate_vector_hazard_scene(
        canvas.window_sha256,
        408005,
        canvas=canvas,
        rock_count=0,
        crater_count=0,
        no_go_count=0,
    )

    first = MultiResolutionScene(canvas, base, valid, vector).project(canvas)
    second = MultiResolutionScene(
        canvas,
        np.where(first.valid_mask, first.elevation_m, 0.0),
        first.valid_mask,
        vector,
    ).project(canvas)

    np.testing.assert_array_equal(first.valid_mask, valid)
    np.testing.assert_array_equal(second.valid_mask, valid)
    np.testing.assert_array_equal(first.elevation_m[valid], base[valid])
    np.testing.assert_array_equal(
        second.elevation_m[valid], first.elevation_m[valid]
    )


def test_tile_provider_is_bounded_and_regeneration_is_exact() -> None:
    geometry = GridGeometry(size_m=16.0, resolution_m=1.0, cells=16)
    canvas = MapCanvas("b" * 64, (0.0, 0.0, 16.0, 16.0), geometry)
    vector = generate_vector_hazard_scene(
        canvas.window_sha256,
        408003,
        canvas=canvas,
        rock_count=12,
        crater_count=3,
        no_go_count=2,
    )
    multires = MultiResolutionScene(
        canvas,
        np.zeros((16, 16), np.float32),
        np.ones((16, 16), bool),
        vector,
    )
    tile_geometry = GridGeometry(size_m=4.0, resolution_m=0.25, cells=16)
    provider = SceneTileProvider(multires, tile_geometry=tile_geometry, capacity=2)

    first = provider.tile(0, 0)
    provider.tile(0, 1)
    provider.tile(0, 2)
    regenerated = provider.tile(0, 0)

    assert provider.cache_size == 2
    np.testing.assert_array_equal(first.elevation_m, regenerated.elevation_m)
    np.testing.assert_array_equal(
        first.physical_obstacle_ratio,
        regenerated.physical_obstacle_ratio,
    )


def test_tile_halo_preserves_exact_tile_at_both_scene_boundaries() -> None:
    geometry = GridGeometry(size_m=16.0, resolution_m=1.0, cells=16)
    canvas = MapCanvas("c" * 64, (0.0, 0.0, 16.0, 16.0), geometry)
    elevation = np.arange(256, dtype=np.float32).reshape(16, 16)
    vector = generate_vector_hazard_scene(
        canvas.window_sha256,
        408006,
        canvas=canvas,
        rock_count=0,
        crater_count=0,
        no_go_count=0,
    )
    scene = MultiResolutionScene(
        canvas, elevation, np.ones((16, 16), bool), vector
    )
    tile_geometry = GridGeometry(size_m=8.0, resolution_m=1.0, cells=8)
    provider = SceneTileProvider(scene, tile_geometry=tile_geometry, capacity=1)

    northwest = provider.tile_with_halo(0, 0, halo_cells=1)
    southeast = provider.tile_with_halo(1, 1, halo_cells=1)

    assert provider.iter_tile_indices() == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert northwest.projected.elevation_m.shape == (10, 10)
    assert southeast.projected.elevation_m.shape == (10, 10)
    np.testing.assert_array_equal(
        northwest.projected.elevation_m[
            northwest.tile_rows, northwest.tile_columns
        ],
        provider.tile(0, 0).elevation_m,
    )
    np.testing.assert_array_equal(
        southeast.projected.elevation_m[
            southeast.tile_rows, southeast.tile_columns
        ],
        provider.tile(1, 1).elevation_m,
    )
    assert provider.cache_size == 1


def test_detail_window_is_projected_once_without_rebuilding_crossed_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = GridGeometry(size_m=16.0, resolution_m=1.0, cells=16)
    canvas = MapCanvas("e" * 64, (0.0, 0.0, 16.0, 16.0), geometry)
    vector = generate_vector_hazard_scene(
        canvas.window_sha256,
        408008,
        canvas=canvas,
        rock_count=4,
        crater_count=2,
        no_go_count=1,
    )
    scene = MultiResolutionScene(
        canvas,
        np.arange(256, dtype=np.float32).reshape(16, 16),
        np.ones((16, 16), dtype=np.bool_),
        vector,
    )
    tile_geometry = GridGeometry(size_m=8.0, resolution_m=1.0, cells=8)
    provider = SceneTileProvider(scene, tile_geometry=tile_geometry, capacity=1)
    expected_canvas = MapCanvas(
        canvas.window_sha256,
        (4.0, 4.0, 12.0, 12.0),
        tile_geometry,
    )
    expected = scene.project(expected_canvas)
    original = MultiResolutionScene.project
    projected_canvases: list[MapCanvas] = []

    def counted_projection(self, target: MapCanvas):
        projected_canvases.append(target)
        return original(self, target)

    monkeypatch.setattr(MultiResolutionScene, "project", counted_projection)

    actual = provider.read_window(4, 4, cells=8)

    assert [value.identity for value in projected_canvases] == [
        expected_canvas.identity
    ]
    assert provider.cache_size == 0
    np.testing.assert_array_equal(actual.valid_mask, expected.valid_mask)
    np.testing.assert_array_equal(actual.elevation_m, expected.elevation_m)
    np.testing.assert_array_equal(
        actual.physical_obstacle_ratio,
        expected.physical_obstacle_ratio,
    )
    np.testing.assert_array_equal(actual.forbidden_ratio, expected.forbidden_ratio)


def test_composed_detail_window_reuses_fixed_tiles_and_is_bit_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = GridGeometry(size_m=16.0, resolution_m=1.0, cells=16)
    canvas = MapCanvas("1" * 64, (0.0, 0.0, 16.0, 16.0), geometry)
    vector = generate_vector_hazard_scene(
        canvas.window_sha256,
        408010,
        canvas=canvas,
        rock_count=4,
        crater_count=2,
        no_go_count=1,
    )
    scene = MultiResolutionScene(
        canvas,
        np.arange(256, dtype=np.float32).reshape(16, 16),
        np.ones((16, 16), dtype=np.bool_),
        vector,
    )
    provider = SceneTileProvider(
        scene,
        tile_geometry=GridGeometry(size_m=8.0, resolution_m=1.0, cells=8),
        capacity=4,
    )
    expected = provider.read_window(4, 4, cells=8)
    original = MultiResolutionScene.project
    projected_canvases: list[MapCanvas] = []

    def counted_projection(self, target: MapCanvas):
        projected_canvases.append(target)
        return original(self, target)

    monkeypatch.setattr(MultiResolutionScene, "project", counted_projection)

    first = provider.compose_window_from_tiles(4, 4, cells=8)
    second = provider.compose_window_from_tiles(4, 4, cells=8)

    assert [value.geometry.size_m for value in projected_canvases] == [
        8.0,
        8.0,
        8.0,
        8.0,
    ]
    assert provider.cache_size == 4
    assert first.canvas.identity == expected.canvas.identity
    assert second.canvas.identity == expected.canvas.identity
    for name in (
        "crater_elevation_delta_m",
        "physical_obstacle_ratio",
        "physical_obstacle_height_m",
        "forbidden_ratio",
        "elevation_m",
        "valid_mask",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(expected, name))
        np.testing.assert_array_equal(getattr(second, name), getattr(expected, name))


def test_visibility_obstacle_window_matches_exact_truth_without_sampling_dem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = GridGeometry(size_m=16.0, resolution_m=1.0, cells=16)
    canvas = MapCanvas("f" * 64, (0.0, 0.0, 16.0, 16.0), geometry)
    vector = generate_vector_hazard_scene(
        canvas.window_sha256,
        408009,
        canvas=canvas,
        rock_count=12,
        crater_count=4,
        no_go_count=2,
    )
    scene = MultiResolutionScene(
        canvas,
        np.arange(256, dtype=np.float32).reshape(16, 16),
        np.ones((16, 16), dtype=np.bool_),
        vector,
    )
    tile_geometry = GridGeometry(size_m=8.0, resolution_m=0.5, cells=16)
    provider = SceneTileProvider(scene, tile_geometry=tile_geometry, capacity=1)
    expected = provider.read_window(
        8, 8, cells=16
    ).physical_obstacle_ratio

    def reject_dem_sampling(self, target: MapCanvas):
        raise AssertionError("visibility-only projection sampled the DEM")

    monkeypatch.setattr(MultiResolutionScene, "_sample_base", reject_dem_sampling)

    actual = provider.read_visibility_obstacle_window(8, 8, cells=16)

    assert actual.dtype == np.dtype(np.float32)
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)


def test_streamed_coverability_matches_exact_pose_center_visibility() -> None:
    canvas = MapCanvas(
        "d" * 64,
        (0.0, 0.0, 16.0, 16.0),
        GridGeometry(size_m=16.0, resolution_m=4.0, cells=4),
    )
    vector = generate_vector_hazard_scene(
        canvas.window_sha256,
        408007,
        canvas=canvas,
        rock_count=0,
        crater_count=0,
        no_go_count=0,
    )
    scene = MultiResolutionScene(
        canvas,
        np.zeros((4, 4), dtype=np.float32),
        np.ones((4, 4), dtype=np.bool_),
        vector,
    )
    tile_geometry = GridGeometry(size_m=8.0, resolution_m=1.0, cells=8)
    reachable = np.zeros((4, 4), dtype=np.bool_)
    reachable[1, 1] = True

    def intrinsic(projected) -> np.ndarray:
        return np.ones(projected.elevation_m.shape, dtype=np.bool_)

    def reveal(
        truth_obstacle_ratio: np.ndarray,
        pose_cell: tuple[int, int],
    ) -> np.ndarray:
        rows, columns = np.indices(truth_obstacle_ratio.shape)
        return np.ascontiguousarray(
            (rows - pose_cell[0]) ** 2 + (columns - pose_cell[1]) ** 2
            <= 3**2,
            dtype=np.bool_,
        )

    def build():
        return build_streamed_detail_coverability(
            tile_provider=SceneTileProvider(
                scene, tile_geometry=tile_geometry, capacity=1
            ),
            inside_mission_roi=np.ones((4, 4), dtype=np.bool_),
            reachable_pose_mask=reachable,
            intrinsic_terrain_feasible=intrinsic,
            reveal_from_pose=reveal,
        )

    first = build()
    second = build()
    expected = np.zeros((16, 16), dtype=np.bool_)
    rows, columns = np.indices(expected.shape)
    expected |= (rows - 6) ** 2 + (columns - 6) ** 2 <= 3**2

    assert first.detail_shape == (16, 16)
    assert first.mission_target_detail_cell_count == 256
    assert first.coverable_detail_cell_count == int(expected.sum())
    np.testing.assert_array_equal(
        unpack_detail_mask(first.coverable_detail_bits, first.detail_shape),
        expected,
    )
    np.testing.assert_array_equal(first.coverable_detail_bits, second.coverable_detail_bits)
    np.testing.assert_array_equal(first.coverable_ratio, second.coverable_ratio)
    assert first.coverable_mask_sha256 == second.coverable_mask_sha256


def test_streamed_coverability_visits_spatial_representatives_before_dense_poses() -> None:
    canvas = MapCanvas(
        "1" * 64,
        (0.0, 0.0, 64.0, 64.0),
        GridGeometry(size_m=64.0, resolution_m=4.0, cells=16),
    )
    vector = generate_vector_hazard_scene(
        canvas.window_sha256,
        408010,
        canvas=canvas,
        rock_count=0,
        crater_count=0,
        no_go_count=0,
    )
    scene = MultiResolutionScene(
        canvas,
        np.zeros((16, 16), dtype=np.float32),
        np.ones((16, 16), dtype=np.bool_),
        vector,
    )
    provider = SceneTileProvider(
        scene,
        tile_geometry=GridGeometry(size_m=32.0, resolution_m=1.0, cells=32),
        capacity=1,
    )
    reachable = np.zeros((16, 16), dtype=np.bool_)
    reachable[4:12, 4:12] = True
    roi = reachable.copy()
    reveal_calls = 0

    def reveal(
        truth_obstacle_ratio: np.ndarray,
        pose_cell: tuple[int, int],
    ) -> np.ndarray:
        nonlocal reveal_calls
        reveal_calls += 1
        return np.ones(truth_obstacle_ratio.shape, dtype=np.bool_)

    detail = build_streamed_detail_coverability(
        tile_provider=provider,
        inside_mission_roi=roi,
        reachable_pose_mask=reachable,
        intrinsic_terrain_feasible=lambda projected: np.ones(
            projected.valid_mask.shape, dtype=np.bool_
        ),
        reveal_from_pose=reveal,
    )

    assert detail.mission_target_detail_cell_count == 32 * 32
    assert detail.coverable_detail_cell_count == 32 * 32
    assert reveal_calls <= 4


def test_projection_rejects_a_target_grid_not_aligned_to_scene_origin() -> None:
    canvas = _canvas()
    scene = generate_vector_hazard_scene(
        canvas.window_sha256,
        408004,
        canvas=canvas,
        rock_count=0,
        crater_count=0,
        no_go_count=0,
    )
    geometry = GridGeometry(size_m=4.0, resolution_m=0.5, cells=8)
    unaligned = MapCanvas(
        canvas.window_sha256,
        (100.1, 200.0, 104.1, 204.0),
        geometry,
    )

    with pytest.raises(ValueError, match="aligned"):
        project_vector_scene(scene, unaligned)
