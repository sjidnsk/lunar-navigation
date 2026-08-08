from __future__ import annotations

import numpy as np
import pytest

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
    assert np.max(local_projection.physical_obstacle_ratio) > 0.0


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
