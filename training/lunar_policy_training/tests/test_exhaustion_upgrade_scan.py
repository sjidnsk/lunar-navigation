from __future__ import annotations

import pathlib
import sys

import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_policy_training.environment import candidate_builder  # noqa: E402
from lunar_policy_training.environment.v3_environment import (  # noqa: E402
    TerminalReason,
    _audit_candidate_boundary,
)
from lunar_policy_training.environment.observation_builder import (  # noqa: E402
    LocalObservation,
    MissionRaster,
    ObservedWorld,
    PlatformProjection,
    Pose2,
)
from lunar_policy_training.environment.platform_reachability import (  # noqa: E402
    GroundGlobalSearchEvidence,
    PHYSICAL_PROJECTION_SCHEMA,
    PhysicalReachabilityResult,
)
from lunar_policy_training.environment.visibility import SensorGeometry  # noqa: E402
from lunar_policy_training.polar_data.hazards import CanvasRatioLayer  # noqa: E402
from lunar_policy_training.polar_data.raster import MapCanvas  # noqa: E402


def test_ground_exhaustion_snapshot_allows_zero_gain_only_after_complete_scan() -> None:
    snapshot = candidate_builder.CandidateDecisionSnapshot(
        snapshot_id="a" * 64,
        frontier_segment_count=1,
        raw_candidate_count=7,
        fine_pose_candidate_count=7,
        globally_reachable_candidate_count=7,
        positive_gain_candidate_count=0,
        selected_policy_candidate_count=0,
        untried_reserve_count=0,
        planner_rejected_current_snapshot_count=0,
        candidate_set_sha256="b" * 64,
        global_search_call_count=1,
        global_search_elapsed_s=0.01,
        candidate_refresh_elapsed_s=0.02,
        pipeline_kind="GROUND_EXHAUSTION",
    )

    assert _audit_candidate_boundary(snapshot, "WHEELED") is TerminalReason.ZERO_EXPECTED_GAIN


def test_residual_component_scan_is_four_connected_and_canonical() -> None:
    residual = np.zeros((5, 6), dtype=np.bool_)
    residual[0, 0] = True
    residual[0, 1] = True
    residual[1, 1] = True
    residual[3, 4] = True
    residual[4, 4] = True

    components = candidate_builder._ground_residual_components(residual)

    assert components == (
        ((0, 0), (0, 1), (1, 1)),
        ((3, 4), (4, 4)),
    )


def test_residual_component_scan_rejects_invalid_mask() -> None:
    with np.testing.assert_raises(ValueError):
        candidate_builder._ground_residual_components(
            np.ones((2, 2), dtype=np.float32)
        )


def test_ground_potential_gain_mask_marks_exact_chebyshev_sensor_stencil() -> None:
    unknown = np.zeros((5, 6), dtype=np.bool_)
    unknown[2, 3] = True

    result = candidate_builder._ground_potential_gain_mask(
        unknown, sensor_range_m=1.0, resolution_m=1.0
    )

    expected = np.zeros_like(unknown)
    expected[1:4, 2:5] = True
    assert np.array_equal(result, expected)


class _AllGainEstimator:
    def __init__(self) -> None:
        self.sensor = SensorGeometry(30.0, 2.0 * np.pi)
        self.calls: list[np.ndarray] = []

    def estimate_candidate_gains(
        self, observed_mask, obstacle_ratio, roi_ratio, priority_weight, candidate_cells
    ) -> np.ndarray:
        del observed_mask, obstacle_ratio, roi_ratio, priority_weight
        self.calls.append(candidate_cells.copy())
        return np.ones((len(candidate_cells), 2), dtype=np.float32)


class _ZeroGainEstimator(_AllGainEstimator):
    def estimate_candidate_gains(
        self, observed_mask, obstacle_ratio, roi_ratio, priority_weight, candidate_cells
    ) -> np.ndarray:
        del observed_mask, obstacle_ratio, roi_ratio, priority_weight
        self.calls.append(candidate_cells.copy())
        return np.zeros((len(candidate_cells), 2), dtype=np.float32)


def _world_and_mission() -> tuple[ObservedWorld, MissionRaster]:
    canvas = MapCanvas.from_roi_bounds("e" * 64, (0.0, 0.0, 25.6, 25.6))
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[120:136, 120:136] = True
    local = LocalObservation(
        canvas.identity, (9.6, 9.6, 16.0, 16.0),
        np.zeros((32, 32), dtype=np.float32), np.ones((32, 32), dtype=np.bool_),
        np.zeros((32, 32), dtype=np.float32),
    )
    world = ObservedWorld(
        canvas, np.zeros((256, 256), dtype=np.float32), observed,
        CanvasRatioLayer(canvas, np.zeros((256, 256), dtype=np.float32)), local,
    )
    roi = np.zeros((256, 256), dtype=np.float32)
    roi[120:140, 120:140] = 1.0
    return world, MissionRaster(canvas, roi.copy(), roi)


def test_isolated_ground_exhaustion_scan_batches_reachable_residual_observers() -> None:
    world, mission = _world_and_mission()
    estimator = _AllGainEstimator()
    builder = candidate_builder.CandidateBuilderV2(estimator)
    reachable = np.ascontiguousarray(world.observed_mask.copy(), dtype=np.bool_)

    result = builder.scan_ground_exhaustion_candidates(
        world, mission, reachable_pose_mask=reachable,
    )

    assert result.diagnostics.residual_component_count == 1
    assert 0 < result.diagnostics.reachable_pose_count <= int(reachable.sum())
    assert result.diagnostics.exact_gain_evaluated_pose_count == result.diagnostics.reachable_pose_count
    assert result.diagnostics.positive_pose_count == result.diagnostics.reachable_pose_count
    assert result.positive_gain_pairs == tuple(
        (1.0, 1.0) for _ in result.positive_pose_cells
    )
    assert len(estimator.calls) == 1
    assert len(estimator.calls[0]) == result.diagnostics.exact_gain_evaluated_pose_count


def test_isolated_ground_exhaustion_scan_filters_endpoint_infeasible_poses() -> None:
    world, mission = _world_and_mission()
    estimator = _AllGainEstimator()
    builder = candidate_builder.CandidateBuilderV2(estimator)
    reachable = np.ascontiguousarray(world.observed_mask.copy(), dtype=np.bool_)
    reachable_cells = np.argwhere(reachable)
    positions = np.ascontiguousarray(
        [
            (float(row) * 0.1, float(column) * 0.1, 0.0)
            for row, column in reachable_cells
        ],
        dtype=np.float64,
    )

    result = builder.scan_ground_exhaustion_candidates(
        world,
        mission,
        reachable_pose_mask=reachable,
        observation_positions_m=positions,
        ground_endpoint_feasibility=lambda candidates: np.zeros(
            len(candidates), dtype=np.bool_
        ),
    )

    assert result.diagnostics.reachable_pose_count > 0
    assert result.diagnostics.endpoint_feasible_pose_count == 0
    assert result.diagnostics.exact_gain_evaluated_pose_count == 0
    assert result.positive_pose_cells == ()
    assert estimator.calls == []


def test_ground_exhaustion_builds_formal_candidate_universe_from_positive_scan() -> None:
    world, mission = _world_and_mission()
    estimator = _AllGainEstimator()
    builder = candidate_builder.CandidateBuilderV2(estimator)
    mask = np.ascontiguousarray(world.observed_mask.copy(), dtype=np.bool_)
    positions = np.ascontiguousarray(
        [
            (*world.canvas.grid_center_world(int(row), int(column)), 0.0)
            for row, column in zip(*np.nonzero(mask), strict=True)
        ],
        dtype=np.float64,
    )
    costs = np.full(mask.shape, np.inf, dtype=np.float64)
    costs[mask] = 1.0
    costs = np.ascontiguousarray(costs)
    costs.setflags(write=False)
    physical = PhysicalReachabilityResult(
        platform_type="WHEELED",
        physical_observation_pose_mask=mask,
        observation_positions_m=positions,
        physical_projection_schema=PHYSICAL_PROJECTION_SCHEMA,
        physical_reachability_algorithm_id="test/ground/v1",
        physical_evidence_algorithm_id="test/evidence/v1",
        physical_safe_pose_count=int(mask.sum()),
        physically_reachable_pose_count=int(mask.sum()),
        ground_global_search=GroundGlobalSearchEvidence(
            sampled_minimum_cost_m=costs,
            planner_tree_sha256="c" * 64,
            planner_tree_cells=256,
            planner_start_index=0,
            search_elapsed_s=0.01,
            coarse_fine_unreachable_count=0,
        ),
    )
    projection = PlatformProjection(
        world.canvas,
        np.ones(mask.shape, dtype=np.float32),
        np.ones((32, 32), dtype=np.float32),
        np.ones(mask.shape, dtype=np.float32),
        "test_only/proxy",
    )

    universe = builder.build_ground_exhaustion_universe(
        world,
        mission,
        Pose2(12.8, 12.8),
        projection,
        physical_reachability=physical,
        platform_type="WHEELED",
        platform_id="unit-wheeled-1",
        capability_content_sha256="1" * 64,
        mission_revision=1,
        evidence_generation=1,
        physical_evidence_sha256="2" * 64,
        physical_reachability_algorithm_id="test/ground/v1",
        goal_tolerance_mm=200,
        ground_endpoint_feasibility=lambda targets: np.ones(
            len(targets), dtype=np.bool_
        ),
    )

    assert universe.candidates
    assert universe.decision_snapshot is not None
    assert universe.decision_snapshot.pipeline_kind == "GROUND_EXHAUSTION"
    assert universe.diagnostics.selected_policy_candidate_count > 0


def test_ground_normal_builder_does_not_scan_after_zero_gain_frontier(
    monkeypatch,
) -> None:
    world, mission = _world_and_mission()
    estimator = _ZeroGainEstimator()
    builder = candidate_builder.CandidateBuilderV2(estimator)
    monkeypatch.setattr(
        builder,
        "scan_ground_exhaustion_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("normal frontier path must not start an exhaustion scan")
        ),
    )
    mask = np.ascontiguousarray(world.observed_mask.copy(), dtype=np.bool_)
    positions = np.ascontiguousarray(
        [
            (*world.canvas.grid_center_world(int(row), int(column)), 0.0)
            for row, column in zip(*np.nonzero(mask), strict=True)
        ],
        dtype=np.float64,
    )
    costs = np.full(mask.shape, np.inf, dtype=np.float64)
    costs[mask] = 1.0
    costs = np.ascontiguousarray(costs)
    costs.setflags(write=False)
    physical = PhysicalReachabilityResult(
        platform_type="WHEELED",
        physical_observation_pose_mask=mask,
        observation_positions_m=positions,
        physical_projection_schema=PHYSICAL_PROJECTION_SCHEMA,
        physical_reachability_algorithm_id="test/ground/v1",
        physical_evidence_algorithm_id="test/evidence/v1",
        physical_safe_pose_count=int(mask.sum()),
        physically_reachable_pose_count=int(mask.sum()),
        ground_global_search=GroundGlobalSearchEvidence(
            sampled_minimum_cost_m=costs,
            planner_tree_sha256="c" * 64,
            planner_tree_cells=256,
            planner_start_index=0,
            search_elapsed_s=0.01,
            coarse_fine_unreachable_count=0,
        ),
    )
    projection = PlatformProjection(
        world.canvas,
        np.ones(mask.shape, dtype=np.float32),
        np.ones((32, 32), dtype=np.float32),
        np.ones(mask.shape, dtype=np.float32),
        "test_only/proxy",
    )

    universe = builder.build_physical_universe(
        world,
        mission,
        Pose2(12.8, 12.8),
        projection,
        physical_reachability=physical,
        platform_type="WHEELED",
        platform_id="unit-wheeled-1",
        capability_content_sha256="1" * 64,
        mission_revision=1,
        evidence_generation=1,
        physical_evidence_sha256="2" * 64,
        physical_reachability_algorithm_id="test/ground/v1",
        goal_tolerance_mm=200,
        ground_endpoint_feasibility=lambda targets: np.ones(
            len(targets), dtype=np.bool_
        ),
    )

    assert universe.decision_snapshot is not None
    assert universe.decision_snapshot.pipeline_kind == "GROUND_FRONTIER"
    assert universe.decision_snapshot.positive_gain_candidate_count == 0
    assert _audit_candidate_boundary(
        universe.decision_snapshot, "WHEELED"
    ) is TerminalReason.ZERO_EXPECTED_GAIN
