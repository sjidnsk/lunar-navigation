from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import lunar_planner_training_bridge as bridge_api

from lunar_policy_training.environment.formal_builder import FormalEpisode
from lunar_policy_training.environment.multires_observation import (
    DetailObservedWindow,
)
from lunar_policy_training.environment.observation_builder import Pose2
from lunar_policy_training.environment.observation_builder import (
    LocalObservation,
    MissionRaster,
    ObservedWorld,
)
from lunar_policy_training.polar_data.hazards import CanvasRatioLayer
from lunar_policy_training.polar_data.raster import (
    LOCAL_TILE_GEOMETRY,
    MapCanvas,
)


class _ObservedDetailWindows:
    def __init__(self) -> None:
        self.centres: list[tuple[float, float]] = []
        self.resolution_m = LOCAL_TILE_GEOMETRY.resolution_m

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
            hard_feasible=np.ones((cells, cells), dtype=np.uint8),
            clearance_m=np.ones((cells, cells), dtype=np.float32),
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


def test_formal_ground_detail_frontier_candidates_use_only_observed_detail() -> None:
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
    canvas = MapCanvas.from_roi_bounds("a" * 64, (0.0, 0.0, 1024.0, 1024.0))
    shape = (canvas.geometry.cells, canvas.geometry.cells)
    observed = np.zeros(shape, dtype=np.bool_)
    observed[128, 128:130] = True
    zeros = np.zeros(shape, dtype=np.float32)
    local = LocalObservation(
        canvas.identity,
        (0.0, 0.0, 6.4, 6.4),
        np.zeros((32, 32), dtype=np.float32),
        np.ones((32, 32), dtype=np.bool_),
        np.zeros((32, 32), dtype=np.float32),
    )
    world = ObservedWorld(
        canvas,
        zeros.copy(),
        observed,
        CanvasRatioLayer(canvas, zeros.copy()),
        local,
    )

    candidates = episode._ground_detail_frontier_candidates(
        [[(128, 128)], [(128, 129)]], world, planner_global_map=object()
    )

    # Two anchors share one cached 64 m detail projection but retain their
    # complete, fixed 20-by-3 safe strips for later batch qualification.
    assert len(candidates) == 120
    segment_id, candidate = candidates[0]
    assert segment_id == 0
    assert candidate.frontier_cell == (128, 128)
    assert candidate.target_pose is not None
    assert candidate.target_pose.frame_id == "map"
    assert {segment_id for segment_id, _ in candidates} == {0, 1}
    assert sum(segment_id == 0 for segment_id, _ in candidates) == 60
    assert sum(segment_id == 1 for segment_id, _ in candidates) == 60
    assert episode._bridge.calls == 1


def test_formal_ground_detail_frontier_candidates_retain_strip_witnesses() -> None:
    """One sampled frontier anchor exposes its bounded safe-strip witnesses.

    The later batch reachability/gain stages, rather than the detail provider,
    decide which one represents this anchor in the three-action policy set.
    """
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
    canvas = MapCanvas.from_roi_bounds("c" * 64, (0.0, 0.0, 1024.0, 1024.0))
    shape = (canvas.geometry.cells, canvas.geometry.cells)
    observed = np.zeros(shape, dtype=np.bool_)
    observed[128, 128] = True
    zeros = np.zeros(shape, dtype=np.float32)
    local = LocalObservation(
        canvas.identity,
        (0.0, 0.0, 6.4, 6.4),
        np.zeros((32, 32), dtype=np.float32),
        np.ones((32, 32), dtype=np.bool_),
        np.zeros((32, 32), dtype=np.float32),
    )
    world = ObservedWorld(
        canvas,
        zeros.copy(),
        observed,
        CanvasRatioLayer(canvas, zeros.copy()),
        local,
    )

    candidates = episode._ground_detail_frontier_candidates(
        [[(128, 128)]], world, planner_global_map=object()
    )

    # The configured strip has 20 standoff cells and three lateral samples.
    assert len(candidates) == 60
    assert {segment_id for segment_id, _ in candidates} == {0}
    assert {candidate.sample_rank for _, candidate in candidates} == {0}
    assert len({candidate.target_pose for _, candidate in candidates}) == 60


def test_formal_ground_detail_frontier_candidates_face_task_roi_unknown_side() -> None:
    """Formal production wiring passes task ROI, not all map exterior unknown."""
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
    canvas = MapCanvas.from_roi_bounds("d" * 64, (0.0, 0.0, 1024.0, 1024.0))
    shape = (canvas.geometry.cells, canvas.geometry.cells)
    observed = np.zeros(shape, dtype=np.bool_)
    observed[128, 128] = True
    zeros = np.zeros(shape, dtype=np.float32)
    local = LocalObservation(
        canvas.identity,
        (0.0, 0.0, 6.4, 6.4),
        np.zeros((32, 32), dtype=np.float32),
        np.ones((32, 32), dtype=np.bool_),
        np.zeros((32, 32), dtype=np.float32),
    )
    world = ObservedWorld(
        canvas,
        zeros.copy(),
        observed,
        CanvasRatioLayer(canvas, zeros.copy()),
        local,
    )
    roi = np.zeros(shape, dtype=np.float32)
    roi[127, 128] = 1.0
    episode.mission = MissionRaster(canvas, zeros.copy(), roi)
    anchor_x, anchor_y = canvas.grid_center_world(128, 128)

    candidates = episode._ground_detail_frontier_candidates(
        [[(128, 128)]], world, planner_global_map=object()
    )

    assert len(candidates) == 60
    assert all(
        candidate.target_pose is not None
        and candidate.target_pose.y_m < anchor_y
        for _, candidate in candidates
    )


def test_formal_ground_detail_frontier_candidates_clamp_edge_window_center() -> None:
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
    canvas = MapCanvas.from_roi_bounds("b" * 64, (0.0, 0.0, 1024.0, 1024.0))
    shape = (canvas.geometry.cells, canvas.geometry.cells)
    observed = np.zeros(shape, dtype=np.bool_)
    observed[255, 255] = True
    zeros = np.zeros(shape, dtype=np.float32)
    local = LocalObservation(
        canvas.identity,
        (0.0, 0.0, 6.4, 6.4),
        np.zeros((32, 32), dtype=np.float32),
        np.ones((32, 32), dtype=np.bool_),
        np.zeros((32, 32), dtype=np.float32),
    )
    world = ObservedWorld(
        canvas,
        zeros.copy(),
        observed,
        CanvasRatioLayer(canvas, zeros.copy()),
        local,
    )

    episode._ground_detail_frontier_candidates(
        [[(255, 255)]], world, planner_global_map=object()
    )

    assert episode.sensor_state.centres == [(992.0, 32.0)]
