from __future__ import annotations

import math

import numpy as np

from lunar_policy_training.environment.candidate_builder import CandidateBuilderV2
from lunar_policy_training.environment.observation_builder import (
    LocalObservation,
    MissionRaster,
    ObservedWorld,
    PlatformProjection,
    Pose2,
)
from lunar_policy_training.environment.platform_reachability import (
    GroundGlobalSearchEvidence,
    PHYSICAL_PROJECTION_SCHEMA,
    PhysicalReachabilityResult,
)
from lunar_policy_training.environment.visibility import SensorGeometry
from lunar_policy_training.polar_data.hazards import CanvasRatioLayer
from lunar_policy_training.polar_data.raster import MapCanvas


class _PositiveGainEstimator:
    def __init__(self) -> None:
        self.sensor = SensorGeometry(30.0, 2.0 * math.pi)

    def estimate_candidate_gains(
        self,
        observed_mask: np.ndarray,
        obstacle_ratio: np.ndarray,
        roi_ratio: np.ndarray,
        priority_weight: np.ndarray,
        candidate_cells: np.ndarray,
    ) -> np.ndarray:
        del observed_mask, obstacle_ratio, roi_ratio, priority_weight
        return np.ones((len(candidate_cells), 2), dtype=np.float32)


def _fixture() -> tuple[
    ObservedWorld,
    MissionRaster,
    PlatformProjection,
    Pose2,
    PhysicalReachabilityResult,
]:
    canvas = MapCanvas.from_roi_bounds(
        "d" * 64,
        (500.0, 500.0, 524.0, 524.0),
    )
    shape = (canvas.geometry.cells, canvas.geometry.cells)
    observed = np.zeros(shape, dtype=np.bool_)
    observed[128:131, 100:111] = True
    roi = np.zeros(shape, dtype=np.float32)
    roi[127:131, 100:111] = 1.0
    zeros = np.zeros(shape, dtype=np.float32)
    local = LocalObservation(
        canvas.identity,
        (508.8, 508.8, 515.2, 515.2),
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
    mission = MissionRaster(canvas, roi.copy(), roi)
    projection = PlatformProjection(
        canvas,
        traversable_ratio=observed.astype(np.float32),
        local_traversable_ratio=np.ones((32, 32), dtype=np.float32),
        clearance_margin_norm=np.ones(shape, dtype=np.float32),
        source="test_only/proxy",
    )
    robot = (130, 105)
    robot_x_m, robot_y_m = canvas.grid_center_world(*robot)
    pose = Pose2(robot_x_m, robot_y_m)
    positions = np.asarray(
        [
            (*canvas.grid_center_world(int(row), int(column)), 0.0)
            for row, column in zip(*np.nonzero(observed), strict=True)
        ],
        dtype=np.float64,
    ).reshape((-1, 3))
    minimum_cost = np.full(shape, np.inf, dtype=np.float64)
    minimum_cost[observed] = 1.0
    minimum_cost[robot] = 0.0
    minimum_cost.setflags(write=False)
    evidence = GroundGlobalSearchEvidence(
        sampled_minimum_cost_m=minimum_cost,
        planner_tree_sha256="a" * 64,
        planner_tree_cells=canvas.geometry.cells,
        planner_start_index=robot[0] * canvas.geometry.cells + robot[1],
        search_elapsed_s=0.01,
        coarse_fine_unreachable_count=0,
    )
    physical = PhysicalReachabilityResult(
        platform_type="WHEELED",
        physical_observation_pose_mask=np.ascontiguousarray(observed),
        observation_positions_m=np.ascontiguousarray(positions),
        physical_projection_schema=PHYSICAL_PROJECTION_SCHEMA,
        physical_reachability_algorithm_id="test/ground-tree/v1",
        physical_evidence_algorithm_id="test/ground-evidence/v1",
        physical_safe_pose_count=int(observed.sum(dtype=np.int64)),
        physically_reachable_pose_count=int(observed.sum(dtype=np.int64)),
        ground_global_search=evidence,
    )
    return world, mission, projection, pose, physical


def test_ground_frontier_backfills_three_exactly_feasible_positions() -> None:
    world, mission, projection, pose, physical = _fixture()
    allowed = {(129, 101), (129, 104), (129, 109)}
    checked: list[tuple[int, int]] = []

    def exact_endpoint_feasibility(positions_m: np.ndarray) -> np.ndarray:
        cells = [
            world.canvas.world_to_grid(float(position[0]), float(position[1]))
            for position in positions_m
        ]
        checked.extend(cells)
        return np.ascontiguousarray(
            [cell in allowed for cell in cells], dtype=np.bool_
        )

    universe = CandidateBuilderV2(
        _PositiveGainEstimator()
    ).build_physical_universe(
        world,
        mission,
        pose,
        projection,
        physical_reachability=physical,
        platform_type="WHEELED",
        platform_id="unit-wheeled-1",
        capability_content_sha256="1" * 64,
        mission_revision=7,
        evidence_generation=3,
        physical_evidence_sha256="2" * 64,
        physical_reachability_algorithm_id=(
            physical.physical_reachability_algorithm_id
        ),
        goal_tolerance_mm=200,
        ground_endpoint_feasibility=exact_endpoint_feasibility,
    )

    emitted = {candidate.position_grid_key for candidate in universe.candidates}
    assert set(checked) > allowed
    assert emitted == allowed
    assert len(universe.candidates) == 3
    assert universe.decision_snapshot.frontier_segment_count == 1
    assert universe.decision_snapshot.raw_candidate_count == 3


def test_ground_frontier_never_fabricates_three_when_only_two_are_feasible() -> None:
    world, mission, projection, pose, physical = _fixture()
    allowed = {(129, 101), (129, 109)}

    def exact_endpoint_feasibility(positions_m: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            [
                world.canvas.world_to_grid(
                    float(position[0]), float(position[1])
                )
                in allowed
                for position in positions_m
            ],
            dtype=np.bool_,
        )

    universe = CandidateBuilderV2(
        _PositiveGainEstimator()
    ).build_physical_universe(
        world,
        mission,
        pose,
        projection,
        physical_reachability=physical,
        platform_type="WHEELED",
        platform_id="unit-wheeled-2",
        capability_content_sha256="1" * 64,
        mission_revision=7,
        evidence_generation=3,
        physical_evidence_sha256="2" * 64,
        physical_reachability_algorithm_id=(
            physical.physical_reachability_algorithm_id
        ),
        goal_tolerance_mm=200,
        ground_endpoint_feasibility=exact_endpoint_feasibility,
    )

    assert {candidate.position_grid_key for candidate in universe.candidates} == allowed
    assert len(universe.candidates) == 2
