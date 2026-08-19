from __future__ import annotations

import math

import numpy as np

from lunar_policy_training.environment.candidate_builder import (
    CandidateBuilderV2,
    _FeasibleAnchor,
    _RawFrontierCandidate,
    _ground_target_visit_key,
    _select_narrow_frontier_strip_positions,
    _select_three_chain_options,
)
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
    # Raw is now the true bounded feasibility fan-out; three remains the
    # policy/action limit after qualification.
    assert universe.decision_snapshot.raw_candidate_count == 9


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


def test_ground_frontier_qualifies_all_strip_witnesses_but_emits_three_actions() -> None:
    """Fine-strip fan-out changes qualification, not policy cardinality."""
    world, mission, projection, pose, physical = _fixture()
    gain_batches: list[int] = []
    endpoint_batches: list[int] = []

    class RecordingGainEstimator(_PositiveGainEstimator):
        def estimate_candidate_gains(
            self,
            observed_mask: np.ndarray,
            obstacle_ratio: np.ndarray,
            roi_ratio: np.ndarray,
            priority_weight: np.ndarray,
            candidate_cells: np.ndarray,
        ) -> np.ndarray:
            gain_batches.append(len(candidate_cells))
            return super().estimate_candidate_gains(
                observed_mask,
                obstacle_ratio,
                roi_ratio,
                priority_weight,
                candidate_cells,
            )

    def provider(segments, _world):
        frontier_cell = segments[0][0]
        output = []
        for sample_rank, column in enumerate((101, 104, 109)):
            pose_cell = (129, column)
            center_x, center_y = world.canvas.grid_center_world(*pose_cell)
            for witness in range(2):
                output.append((
                    0,
                    _RawFrontierCandidate(
                        sample_rank=sample_rank,
                        frontier_cell=frontier_cell,
                        pose_cell=pose_cell,
                        target_pose=Pose2(
                            center_x + 0.2 * witness,
                            center_y,
                        ),
                    ),
                ))
        return tuple(output)

    def endpoint_feasibility(positions: np.ndarray) -> np.ndarray:
        endpoint_batches.append(len(positions))
        return np.ones(len(positions), dtype=np.bool_)

    builder = CandidateBuilderV2(RecordingGainEstimator())
    universe = builder.build_physical_universe(
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
        ground_endpoint_feasibility=endpoint_feasibility,
        ground_detail_candidate_provider=provider,
    )

    assert len(universe.candidates) == 3
    assert universe.decision_snapshot.raw_candidate_count == 6
    assert universe.decision_snapshot.fine_pose_candidate_count == 6
    assert universe.decision_snapshot.globally_reachable_candidate_count == 6
    assert endpoint_batches == [6]
    assert gain_batches == [6]
    # All detailed witnesses within this 4 m source cell share the legacy
    # coarse position key.  Visiting one 0.2 m witness must leave its sibling
    # selectable rather than suppressing the whole coarse cell.
    first = universe.candidates[0]
    result = CandidateBuilderV2(RecordingGainEstimator()).select_available(
        universe,
        canvas_id=world.canvas.identity,
        excluded_cells={
            _ground_target_visit_key(
                first.target_position_m[0], first.target_position_m[1]
            )
        },
    )
    selected_ids = {
        candidate_id
        for candidate_id, enabled in zip(
            result.batch.candidate_ids, result.batch.mask, strict=True
        )
        if enabled
    }
    assert first.candidate_id not in selected_ids
    assert selected_ids

    refreshed = builder.build_physical_universe(
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
        excluded_cells={
            _ground_target_visit_key(
                first.target_position_m[0], first.target_position_m[1]
            )
        },
        ground_endpoint_feasibility=endpoint_feasibility,
        ground_detail_candidate_provider=provider,
    )
    assert len(refreshed.candidates) == 3
    assert first.candidate_id not in {
        candidate.candidate_id for candidate in refreshed.candidates
    }


def test_narrow_frontier_strip_keeps_only_observed_safe_positions() -> None:
    """A coarse frontier retains every bounded detailed safe witness."""
    coarse_observed = np.zeros((5, 5), dtype=np.bool_)
    coarse_observed[2:, :] = True
    detail_observed = np.zeros((25, 25), dtype=np.bool_)
    detail_observed[10:, :] = True
    detail_safe = detail_observed.copy()
    clearance = np.zeros((25, 25), dtype=np.float32)
    for column in (7, 12, 17):
        clearance[15, column] = 0.8
        clearance[15, column + 1] = 1.0
        detail_safe[15, column + 1] = False

    selection = _select_narrow_frontier_strip_positions(
        [(2, column) for column in range(5)],
        coarse_observed_mask=coarse_observed,
        observed_detail_mask=detail_observed,
        physical_safe_detail_mask=detail_safe,
        clearance_detail=clearance,
        detail_cells_per_coarse=5,
        minimum_standoff_detail_cells=2,
        maximum_standoff_detail_cells=4,
        lateral_half_width_detail_cells=1,
    )

    assert len(selection.pose_cells) == 24
    assert {(15, 7), (15, 12), (15, 17)} <= set(selection.pose_cells)
    assert not {(15, 8), (15, 13), (15, 18)} & set(selection.pose_cells)
    assert selection.evaluated_detail_cell_count == 27


def test_narrow_frontier_strip_retains_all_bounded_safe_witnesses() -> None:
    """A later gain gate must see every safe cell, not only max clearance."""
    coarse_observed = np.zeros((5, 5), dtype=np.bool_)
    coarse_observed[2:, :] = True
    detail_observed = np.zeros((25, 25), dtype=np.bool_)
    detail_observed[10:, :] = True
    detail_safe = detail_observed.copy()
    clearance = np.zeros((25, 25), dtype=np.float32)
    clearance[15, 12] = 10.0

    selection = _select_narrow_frontier_strip_positions(
        [(2, 2)],
        coarse_observed_mask=coarse_observed,
        observed_detail_mask=detail_observed,
        physical_safe_detail_mask=detail_safe,
        clearance_detail=clearance,
        detail_cells_per_coarse=5,
        minimum_standoff_detail_cells=2,
        maximum_standoff_detail_cells=3,
        lateral_half_width_detail_cells=1,
    )

    assert selection.pose_cells == (
        (14, 11),
        (14, 12),
        (14, 13),
        (15, 11),
        (15, 12),
        (15, 13),
    )
    assert selection.evaluated_detail_cell_count == 6


def test_three_chain_selection_spreads_actions_across_sampled_anchors() -> None:
    """Many valid strip witnesses from one anchor stay internal alternatives."""
    def raw(sample_rank: int, column: int) -> _RawFrontierCandidate:
        return _RawFrontierCandidate(
            sample_rank=sample_rank,
            frontier_cell=(9, sample_rank),
            pose_cell=(10, column),
        )

    selected = _select_three_chain_options(
        [
            (raw(0, 1), 0, 1.0),
            (raw(0, 2), 1, 9.0),
            (raw(1, 3), 2, 2.0),
            (raw(1, 4), 3, 8.0),
            (raw(2, 5), 4, 3.0),
            (raw(2, 6), 5, 7.0),
        ],
        chain_length=50,
    )

    # Exactly three final actions: one best-gain witness for each of q1/q2/q3.
    assert selected == (1, 3, 5)


def test_narrow_frontier_strip_has_a_fixed_bounded_scan_cost() -> None:
    """Long segments cannot expand one anchor's detailed strip into map search."""
    coarse_observed = np.zeros((9, 41), dtype=np.bool_)
    coarse_observed[4:, :] = True
    detail_observed = np.zeros((45, 205), dtype=np.bool_)
    detail_observed[20:, :] = True
    detail_safe = detail_observed.copy()
    clearance = np.ones((45, 205), dtype=np.float32)

    selection = _select_narrow_frontier_strip_positions(
        [(4, column) for column in range(1, 40)],
        coarse_observed_mask=coarse_observed,
        observed_detail_mask=detail_observed,
        physical_safe_detail_mask=detail_safe,
        clearance_detail=clearance,
        detail_cells_per_coarse=5,
        minimum_standoff_detail_cells=2,
        maximum_standoff_detail_cells=4,
        lateral_half_width_detail_cells=1,
    )

    # Three sampled anchors, each with a fixed 3-by-3 detail strip.  The
    # policy cardinality is bounded later, after feasibility and gain gates;
    # the local witness scan itself must not silently throw safe positions
    # away.
    assert len(selection.pose_cells) == 27
    assert selection.evaluated_detail_cell_count == 27


def test_narrow_frontier_strip_uses_clearance_gradient_when_frontier_normal_cancels() -> None:
    """Opposing unknown neighbours choose the safest observed-side direction."""
    coarse_observed = np.ones((5, 5), dtype=np.bool_)
    coarse_observed[1, 2] = False
    coarse_observed[3, 2] = False
    detail_observed = np.ones((25, 25), dtype=np.bool_)
    detail_safe = detail_observed.copy()
    clearance = np.zeros((25, 25), dtype=np.float32)
    clearance[12, 14] = 0.9

    selection = _select_narrow_frontier_strip_positions(
        [(2, 2)],
        coarse_observed_mask=coarse_observed,
        observed_detail_mask=detail_observed,
        physical_safe_detail_mask=detail_safe,
        clearance_detail=clearance,
        detail_cells_per_coarse=5,
        minimum_standoff_detail_cells=2,
        maximum_standoff_detail_cells=2,
        lateral_half_width_detail_cells=0,
    )

    assert selection.pose_cells == ((12, 14),)
    assert selection.evaluated_detail_cell_count == 5


def test_narrow_frontier_strip_uses_task_roi_unknown_side_not_map_exterior() -> None:
    """The observed-side direction is derived from unexplored task ROI only."""
    coarse_observed = np.ones((5, 5), dtype=np.bool_)
    coarse_observed[1, 2] = False
    unknown_roi = np.zeros((5, 5), dtype=np.bool_)
    unknown_roi[1, 2] = True
    detail_observed = np.ones((25, 25), dtype=np.bool_)
    detail_safe = detail_observed.copy()
    clearance = np.ones((25, 25), dtype=np.float32)

    selection = _select_narrow_frontier_strip_positions(
        [(2, 2)],
        coarse_observed_mask=coarse_observed,
        coarse_unknown_roi_mask=unknown_roi,
        observed_detail_mask=detail_observed,
        physical_safe_detail_mask=detail_safe,
        clearance_detail=clearance,
        detail_cells_per_coarse=5,
        minimum_standoff_detail_cells=2,
        maximum_standoff_detail_cells=2,
        lateral_half_width_detail_cells=0,
    )

    # Unknown task area is above the anchor, so its safe observation strip is
    # below it.  Outside-ROI unknown cells must not reverse this direction.
    assert selection.pose_cells == ((14, 12),)


def test_ground_candidate_identity_distinguishes_exact_positions_in_one_coarse_cell() -> None:
    """A 0.2 m strip must not alias two poses through its 4 m parent cell."""
    feature = np.zeros(12, dtype=np.float32)
    feature[5] = np.float32(1.0)
    common = {
        "pose_map": Pose2(500.0, 500.0),
        "platform_type": "WHEELED",
        "platform_id": "unit-wheeled",
        "mission_revision": 7,
        "goal_tolerance_mm": 200,
    }
    left = CandidateBuilderV2._physical_candidate(
        _FeasibleAnchor(
            segment_id=0,
            point=(100, 101),
            feature=feature,
            elevation_m=1.0,
            target_position_m=(504.1, 503.9, 1.0),
        ),
        **common,
    )
    right = CandidateBuilderV2._physical_candidate(
        _FeasibleAnchor(
            segment_id=0,
            point=(100, 101),
            feature=feature,
            elevation_m=1.0,
            target_position_m=(504.3, 503.9, 1.0),
        ),
        **common,
    )

    assert left.position_grid_key == right.position_grid_key
    assert left.candidate_id != right.candidate_id
