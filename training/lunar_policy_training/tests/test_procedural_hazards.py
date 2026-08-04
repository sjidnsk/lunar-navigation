from __future__ import annotations

import pathlib
import sys
from hashlib import sha256

import numpy as np
import pytest
from shapely.geometry import box


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lunar_policy_training.polar_data.hazards import (  # noqa: E402
    GENERATOR_VERSION,
    generate_hazard_scene,
    physical_obstacle_ratio,
    scene_seed,
)
from lunar_policy_training.polar_data.raster import GridGeometry  # noqa: E402
from lunar_policy_training.polar_data.raster import MapCanvas  # noqa: E402


def test_scene_seed_binds_window_scenario_and_generator_version() -> None:
    baseline = scene_seed("a" * 64, 7)
    assert baseline == sha256(f"{'a' * 64}:7:{GENERATOR_VERSION}".encode("utf-8")).hexdigest()
    assert baseline != scene_seed("b" * 64, 7)
    assert baseline != scene_seed("a" * 64, 8)


def test_scene_seed_rejects_non_lowercase_window_sha() -> None:
    with pytest.raises(ValueError, match="lowercase"):
        scene_seed("A" * 64, 7)


def test_hazard_generation_is_deterministic_and_craters_are_not_physical_obstacles() -> None:
    geometry = GridGeometry(size_m=8.0, resolution_m=1.0, cells=8)
    canvas = MapCanvas("f" * 64, (0.0, 0.0, 8.0, 8.0), geometry)
    left = generate_hazard_scene("f" * 64, 31, canvas=canvas, geometry=geometry, rock_count=2, crater_count=2)
    right = generate_hazard_scene("f" * 64, 31, canvas=canvas, geometry=geometry, rock_count=2, crater_count=2)

    assert left.rock_footprints_wkb == right.rock_footprints_wkb
    assert left.no_go_polygons_wkb == right.no_go_polygons_wkb
    assert left.no_go_polygons_wkb
    np.testing.assert_array_equal(left.crater_elevation_delta_m, right.crater_elevation_delta_m)
    assert left.physical_obstacle_ratio.shape == (8, 8)
    assert left.crater_polygons_wkb
    assert np.any(left.crater_elevation_delta_m != 0.0)


def test_rock_union_counts_overlap_once_and_uses_polygon_cell_area() -> None:
    geometry = GridGeometry(size_m=2.0, resolution_m=1.0, cells=2)
    ratio = physical_obstacle_ratio((box(0.0, 0.0, 1.0, 1.0), box(0.5, 0.0, 1.5, 1.0)), geometry)
    np.testing.assert_allclose(ratio, [[1.0, 0.5], [0.0, 0.0]])


def test_hazard_generation_requires_matching_absolute_canvas() -> None:
    with pytest.raises(TypeError, match="canvas"):
        generate_hazard_scene("e" * 64, 1, rock_count=0, crater_count=0)
    canvas = MapCanvas("e" * 64, (1_000.0, 2_000.0, 2_024.0, 3_024.0))
    assert generate_hazard_scene("e" * 64, 1, canvas=canvas, rock_count=0, crater_count=1).canvas is canvas
