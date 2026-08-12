from __future__ import annotations

from hashlib import sha256
import math

import numpy as np
import pytest

from lunar_policy_training.environment.candidate_builder import CandidateBuilderV2
from lunar_policy_training.environment.frontier_oracle import (
    FrontierOpportunityOracle,
)
from lunar_policy_training.environment.observation_builder import (
    LocalObservation,
    MissionRaster,
    ObservedWorld,
    Pose2,
)
from lunar_policy_training.environment.platform_reachability import (
    PHYSICAL_PROJECTION_SCHEMA,
    PhysicalReachabilityResult,
)
from lunar_policy_training.environment.primitive_reachability import (
    ObservedPrimitiveReachability,
)
from lunar_policy_training.environment.visibility import SensorGeometry
from lunar_policy_training.polar_data.hazards import CanvasRatioLayer
from lunar_policy_training.polar_data.raster import GridGeometry, MapCanvas


EMPTY_OPPORTUNITY_SET_SHA256 = sha256(b"").hexdigest()


class _GainEstimator:
    sensor = SensorGeometry(30.0, 2.0 * math.pi)
    resolution_m = 4.0

    def __init__(self, gains: dict[tuple[int, int], float]) -> None:
        self.gains = gains
        self.calls: list[np.ndarray] = []

    def estimate_candidate_gains(
        self,
        observed_mask,
        obstacle_ratio,
        roi_ratio,
        priority_weight,
        candidate_cells,
    ) -> np.ndarray:
        del observed_mask, obstacle_ratio, roi_ratio, priority_weight
        self.calls.append(candidate_cells.copy())
        return np.asarray(
            [
                (self.gains.get(tuple(cell), 0.0), 0.0)
                for cell in candidate_cells
            ],
            dtype=np.float32,
        ).reshape((-1, 2))


def _world_and_mission(
    *,
    canvas: MapCanvas | None = None,
) -> tuple[ObservedWorld, MissionRaster, Pose2]:
    resolved = canvas or MapCanvas.from_roi_bounds(
        "f" * 64, (0.0, 0.0, 1024.0, 1024.0)
    )
    shape = (resolved.geometry.cells, resolved.geometry.cells)
    observed = np.zeros(shape, dtype=np.bool_)
    observed[126:131, 126:135] = True
    roi = np.zeros(shape, dtype=np.float32)
    roi[124:133, 124:138] = 1.0
    elevation = np.full(shape, 7.0, dtype=np.float32)
    robot_x_m, robot_y_m = resolved.grid_center_world(128, 128)
    local = LocalObservation(
        resolved.identity,
        (
            robot_x_m - 3.2,
            robot_y_m - 3.2,
            robot_x_m + 3.2,
            robot_y_m + 3.2,
        ),
        np.zeros((32, 32), dtype=np.float32),
        np.ones((32, 32), dtype=np.bool_),
        np.zeros((32, 32), dtype=np.float32),
    )
    world = ObservedWorld(
        resolved,
        elevation,
        observed,
        CanvasRatioLayer(resolved, np.zeros(shape, dtype=np.float32)),
        local,
    )
    return (
        world,
        MissionRaster(resolved, roi.copy(), roi),
        Pose2(robot_x_m, robot_y_m, elevation_m=7.0),
    )


def _physical(
    world: ObservedWorld,
    cells: tuple[tuple[int, int], ...],
    *,
    positions_m: np.ndarray | None = None,
    platform_type: str = "WHEELED",
) -> PhysicalReachabilityResult:
    mask = np.zeros(world.observed_mask.shape, dtype=np.bool_)
    for cell in cells:
        mask[cell] = True
    if positions_m is None:
        positions_m = np.asarray(
            [
                (
                    *world.canvas.grid_center_world(*cell),
                    float(world.elevation_m[cell]),
                )
                for cell in sorted(cells)
            ],
            dtype=np.float64,
        ).reshape((-1, 3))
    return PhysicalReachabilityResult(
        platform_type=platform_type,
        physical_observation_pose_mask=np.ascontiguousarray(mask),
        observation_positions_m=np.ascontiguousarray(
            positions_m, dtype=np.float64
        ).reshape((-1, 3)),
        physical_projection_schema=PHYSICAL_PROJECTION_SCHEMA,
        physical_reachability_algorithm_id="test/physical-reachability/v1",
        physical_evidence_algorithm_id="test/physical-evidence/v1",
        physical_safe_pose_count=len(cells),
        physically_reachable_pose_count=len(cells),
    )


def _forbid_primitive_and_production_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("oracle called a production or primitive path")

    monkeypatch.setattr(
        ObservedPrimitiveReachability, "update", forbidden
    )
    monkeypatch.setattr(
        CandidateBuilderV2, "build_from_primitive_graph", forbidden
    )
    monkeypatch.setattr(
        CandidateBuilderV2, "build_physical_universe", forbidden
    )


def test_physical_oracle_is_independent_and_hashes_sorted_positive_opportunities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_primitive_and_production_paths(monkeypatch)
    world, mission, pose = _world_and_mission()
    cells = ((128, 130), (128, 129))
    estimator = _GainEstimator({cell: 1.0 for cell in cells})
    physical = _physical(world, cells)

    result = FrontierOpportunityOracle(estimator).evaluate_physical(
        world,
        mission,
        pose_map=pose,
        physical_reachability=physical,
    )

    expected_keys = b"128:129:518000:510000:7000\n128:130:522000:510000:7000"
    assert result.oracle_opportunity_count == 2
    assert result.opportunity_count == 2
    assert result.oracle_opportunity_set_sha256 == sha256(
        expected_keys
    ).hexdigest()
    assert result.opportunity_set_sha256 == result.oracle_opportunity_set_sha256
    assert result.platform_reachable_pose_count == 2
    assert len(estimator.calls) == 1
    np.testing.assert_array_equal(
        estimator.calls[0], np.asarray(cells[::-1], dtype=np.int32)
    )


def test_physical_oracle_does_not_turn_frontier_existence_into_positive_gain() -> None:
    world, mission, pose = _world_and_mission()
    physical = _physical(world, ((128, 129), (128, 130)))
    estimator = _GainEstimator({})

    result = FrontierOpportunityOracle(estimator).evaluate_physical(
        world,
        mission,
        pose_map=pose,
        physical_reachability=physical,
    )

    assert bool((mission.roi_ratio > 0.0).any())
    assert result.platform_reachable_pose_count == 2
    assert result.oracle_opportunity_count == 0
    assert result.oracle_opportunity_set_sha256 == EMPTY_OPPORTUNITY_SET_SHA256


def test_ground_global_oracle_keeps_positive_opportunity_beyond_sensor_range() -> None:
    world, mission, pose = _world_and_mission()
    remote = (128, 150)
    world.observed_mask[remote] = True
    mission.roi_ratio[remote] = np.float32(1.0)
    mission.priority[remote] = np.float32(1.0)
    mission.roi_ratio[128, 151] = np.float32(1.0)
    mission.priority[128, 151] = np.float32(1.0)
    assert math.dist((128, 128), remote) * world.canvas.geometry.resolution_m == 88.0
    estimator = _GainEstimator({remote: 1.0})

    result = FrontierOpportunityOracle(estimator).evaluate_physical(
        world,
        mission,
        pose_map=pose,
        physical_reachability=_physical(world, (remote,)),
    )

    assert result.platform_reachable_pose_count == 1
    assert result.oracle_opportunity_count == 1
    np.testing.assert_array_equal(
        estimator.calls[0], np.asarray((remote,), dtype=np.int32)
    )


def test_physical_oracle_requires_reachability_before_estimating_gain() -> None:
    world, mission, pose = _world_and_mission()
    estimator = _GainEstimator({(128, 129): 1.0})

    result = FrontierOpportunityOracle(estimator).evaluate_physical(
        world,
        mission,
        pose_map=pose,
        physical_reachability=_physical(world, ()),
    )

    assert result.frontier_anchor_count == int(
        world.observed_mask.sum(dtype=np.int64)
    )
    assert result.observed_safe_pose_count == (
        int(world.observed_mask.sum(dtype=np.int64)) - 1
    )
    assert result.platform_reachable_pose_count == 0
    assert result.oracle_opportunity_count == 0
    assert result.oracle_opportunity_set_sha256 == EMPTY_OPPORTUNITY_SET_SHA256
    assert estimator.calls == []


def test_physical_oracle_uses_exact_positions_at_thirty_metre_boundary() -> None:
    canvas = MapCanvas(
        "9" * 64,
        (0.0, 0.0, 256.0, 256.0),
        GridGeometry(256.0, 1.0, 256),
    )
    world, mission, _ = _world_and_mission(canvas=canvas)
    robot_x_m, robot_y_m = canvas.grid_center_world(128, 128)
    pose = Pose2(robot_x_m, robot_y_m, elevation_m=7.0)
    cells = ((128, 158), (129, 158))
    world.observed_mask[128, 158] = True
    world.observed_mask[129, 158] = True
    positions = np.asarray(
        (
            (robot_x_m + 29.9, robot_y_m, 10.0),
            (robot_x_m + 30.1, robot_y_m - 1.0, 11.0),
        ),
        dtype=np.float64,
    )
    physical = _physical(
        world,
        cells,
        positions_m=positions,
        platform_type="HOPPER",
    )
    estimator = _GainEstimator({cell: 1.0 for cell in cells})

    result = FrontierOpportunityOracle(estimator).evaluate_physical(
        world,
        mission,
        pose_map=pose,
        physical_reachability=physical,
    )

    assert result.oracle_opportunity_count == 1
    np.testing.assert_array_equal(
        estimator.calls[0], np.asarray(((128, 158),), dtype=np.int32)
    )


def test_physical_oracle_hash_is_repeatable_and_binds_exact_hopper_xyz() -> None:
    world, mission, pose = _world_and_mission()
    cells = ((128, 129),)
    x_m, y_m = world.canvas.grid_center_world(*cells[0])
    first_physical = _physical(
        world,
        cells,
        positions_m=np.asarray(((x_m + 0.02, y_m - 0.02, 31.0),)),
        platform_type="HOPPER",
    )
    changed_height = _physical(
        world,
        cells,
        positions_m=np.asarray(((x_m + 0.02, y_m - 0.02, 32.0),)),
        platform_type="HOPPER",
    )

    def evaluate(physical: PhysicalReachabilityResult):
        return FrontierOpportunityOracle(
            _GainEstimator({cells[0]: 1.0})
        ).evaluate_physical(
            world,
            mission,
            pose_map=pose,
            physical_reachability=physical,
        )

    first = evaluate(first_physical)
    repeated = evaluate(first_physical)
    changed = evaluate(changed_height)

    assert first.oracle_opportunity_count == repeated.oracle_opportunity_count == 1
    assert (
        first.oracle_opportunity_set_sha256
        == repeated.oracle_opportunity_set_sha256
    )
    assert (
        changed.oracle_opportunity_set_sha256
        != first.oracle_opportunity_set_sha256
    )
