from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lunar_exploration_policy.cpp_projection import CppV3Projector
from lunar_exploration_policy.grid_map_runtime import DecodedGridMap
from lunar_policy_training.environment.observation_builder import (
    LocalObservation,
    ObservedWorld,
)
from lunar_policy_training.polar_data.hazards import CanvasRatioLayer
from lunar_policy_training.polar_data.raster import MapCanvas


ROOT = Path(__file__).resolve().parents[4]
PROFILE_ROOT = ROOT / "ros2_ws/src/lunar_navigation_config/config/platform_profiles"


def _decoded(frame: str, cells: int, resolution: float, token: str):
    layers = {
        name: np.zeros((cells, cells), np.float32)
        for name in (
            "elevation", "valid_mask", "obstacle", "obstacle_height",
            "observation_age_s", "observation_quality",
            "elevation_variance", "obstacle_variance",
            "observation_count", "forbidden",
        )
    }
    layers["valid_mask"][:] = 1.0
    layers["observation_quality"][:] = 1.0
    return DecodedGridMap(
        frame, 1_000_000_000, cells, cells, resolution,
        (0.0, 0.0), layers, token * 64,
    )


def _world():
    canvas = MapCanvas("a" * 64, (0.0, 0.0, 1024.0, 1024.0))
    local = LocalObservation(
        canvas.identity,
        (508.8, 508.8, 515.2, 515.2),
        np.zeros((32, 32), np.float32),
        np.ones((32, 32), np.bool_),
        np.zeros((32, 32), np.float32),
    )
    return ObservedWorld(
        canvas,
        np.zeros((256, 256), np.float32),
        np.ones((256, 256), np.bool_),
        CanvasRatioLayer(canvas, np.zeros((256, 256), np.float32)),
        local,
    )


@pytest.mark.parametrize(
    "platform,profile",
    [
        ("WHEELED", "wheeled.yaml"),
        ("LEGGED", "legged.yaml"),
        ("HOPPER", "hopper.yaml"),
    ],
)
def test_cpp_projector_uses_selected_frozen_platform(
    platform: str, profile: str
) -> None:
    projector = CppV3Projector(ROOT, PROFILE_ROOT / profile)

    projection = projector.project(
        _world(),
        platform,
        global_map=_decoded("map", 256, 4.0, "a"),
        local_map=_decoded("odom", 32, 0.2, "b"),
    )

    assert projection.traversable_ratio.shape == (256, 256)
    assert projection.local_traversable_ratio.shape == (32, 32)
    assert projection.clearance_margin_norm.shape == (256, 256)
    assert projection.source.startswith("cpp_v3/")


def test_cpp_projector_rejects_platform_different_from_selected_profile() -> None:
    projector = CppV3Projector(ROOT, PROFILE_ROOT / "wheeled.yaml")

    with pytest.raises(ValueError, match="selected profile"):
        projector.project(
            _world(),
            "LEGGED",
            global_map=_decoded("map", 256, 4.0, "a"),
            local_map=_decoded("odom", 32, 0.2, "b"),
        )
