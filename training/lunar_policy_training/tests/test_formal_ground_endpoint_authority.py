from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import lunar_planner_training_bridge as bridge_api

from lunar_policy_training.environment.formal_builder import FormalEpisode
from lunar_policy_training.environment.multires_observation import (
    DetailObservedWindow,
)
from lunar_policy_training.environment.observation_builder import Pose2
from lunar_policy_training.polar_data.raster import (
    LOCAL_TILE_GEOMETRY,
    MapCanvas,
)


class _ObservedDetailWindows:
    def __init__(self) -> None:
        self.centres: list[tuple[float, float]] = []

    def planning_observation(self, pose: Pose2) -> DetailObservedWindow:
        self.centres.append((pose.x_m, pose.y_m))
        cells = LOCAL_TILE_GEOMETRY.cells
        shape = (cells, cells)
        canvas = MapCanvas(
            "f" * 64,
            (
                pose.x_m - 32.0,
                pose.y_m - 32.0,
                pose.x_m + 32.0,
                pose.y_m + 32.0,
            ),
            LOCAL_TILE_GEOMETRY,
        )
        zeros = np.zeros(shape, dtype=np.float32)
        return DetailObservedWindow(
            canvas=canvas,
            elevation_m=zeros.copy(),
            physical_obstacle_ratio=zeros.copy(),
            physical_obstacle_height_m=zeros.copy(),
            forbidden_ratio=zeros.copy(),
            valid_mask=np.ones(shape, dtype=np.bool_),
            observation_age_s=zeros.copy(),
            observation_quality=np.ones(shape, dtype=np.float32),
            observation_count=np.ones(shape, dtype=np.uint32),
        )


class _AllSafeProjectionBridge:
    def __init__(self) -> None:
        self.calls = 0

    def project_traversability(self, request: object) -> object:
        del request
        self.calls += 1
        cells = LOCAL_TILE_GEOMETRY.cells
        return SimpleNamespace(
            hard_feasible=np.ones((cells, cells), dtype=np.uint8)
        )


def test_formal_ground_endpoint_authority_batches_observed_detail_windows() -> None:
    episode = object.__new__(FormalEpisode)
    episode.platform_type = "WHEELED"
    episode.scene_id = "unit-scene"
    episode._revision = 5
    episode.current_pose = Pose2(0.0, 0.0)
    episode.sensor_state = _ObservedDetailWindows()
    episode._bridge = _AllSafeProjectionBridge()
    episode._base_request = (
        lambda _global_map, _local_map: bridge_api.TrainingPlanRequest()
    )
    targets = np.asarray(
        [[0.0, 0.0, 0.0], [4.0, 2.0, 0.0], [60.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    feasible = episode._ground_endpoint_feasibility(
        targets,
        planner_global_map=object(),
    )

    assert feasible.tolist() == [True, True, True]
    assert episode._bridge.calls == 2
    assert len(episode.sensor_state.centres) == 2
