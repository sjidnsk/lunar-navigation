from __future__ import annotations

import math

import numpy as np

from lunar_policy_training.environment.candidate_builder import CandidateBuilderV2
from lunar_policy_training.environment.frontier_oracle import (
    FrontierOpportunityOracle,
)
from lunar_policy_training.environment.observation_builder import (
    LocalObservation,
    MissionRaster,
    ObservedWorld,
    PlatformProjection,
    Pose2,
)
from lunar_policy_training.environment.platform_reachability import (
    CandidateReachabilityResult,
)
from lunar_policy_training.environment.primitive_reachability import (
    ObservedPrimitiveSnapshot,
)
from lunar_policy_training.environment.visibility import SensorGeometry
from lunar_policy_training.polar_data.hazards import CanvasRatioLayer
from lunar_policy_training.polar_data.raster import MapCanvas


class _GainEstimator:
    sensor = SensorGeometry(30.0, 2.0 * math.pi)
    resolution_m = 4.0

    def __init__(self, gains: float = 1.0) -> None:
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
        self.calls.append(candidate_cells.copy())
        return np.full(
            (candidate_cells.shape[0], 2), self.gains, dtype=np.float32
        )


class _Reachability:
    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted
        self.calls: list[np.ndarray] = []

    def filter(self, candidate_cells: np.ndarray) -> CandidateReachabilityResult:
        self.calls.append(candidate_cells.copy())
        mask = np.full(candidate_cells.shape[0], self.accepted, dtype=np.bool_)
        return CandidateReachabilityResult(
            mask,
            {
                "platform_unreachable_count": int(
                    (~mask).sum(dtype=np.int64)
                )
            },
        )


def _fixture():
    canvas = MapCanvas.from_roi_bounds(
        "f" * 64, (500.0, 500.0, 1524.0, 1524.0)
    )
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[126:130, 126:130] = True
    obstacle = CanvasRatioLayer(
        canvas, np.zeros((256, 256), dtype=np.float32)
    )
    local = LocalObservation(
        canvas.identity,
        (1008.8, 1008.8, 1015.2, 1015.2),
        np.zeros((32, 32), dtype=np.float32),
        np.ones((32, 32), dtype=np.bool_),
        np.zeros((32, 32), dtype=np.float32),
    )
    world = ObservedWorld(
        canvas,
        np.zeros((256, 256), dtype=np.float32),
        observed,
        obstacle,
        local,
    )
    roi = np.zeros((256, 256), dtype=np.float32)
    roi[120:136, 120:136] = 1.0
    mission = MissionRaster(canvas, roi.copy(), roi)
    projection = PlatformProjection(
        canvas,
        observed.astype(np.float32),
        np.ones((32, 32), dtype=np.float32),
        np.ones((256, 256), dtype=np.float32),
        "test_only/proxy",
    )
    return world, mission, projection


def _frozen(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values)
    result.setflags(write=False)
    return result


def _oracle_graph(canvas: MapCanvas) -> ObservedPrimitiveSnapshot:
    cells = ((128, 128), (128, 129), (128, 130))
    positions = np.asarray(
        [(*canvas.grid_center_world(*cell), 0.0) for cell in cells],
        dtype=np.float64,
    )
    labels = _frozen(np.ones(3, dtype=np.bool_))
    return ObservedPrimitiveSnapshot(
        platform_type="WHEELED",
        width=32,
        height=32,
        algorithm_id="test-wheel/v1",
        state_schema="test-wheel-state/v1",
        primitive_set_sha256="1" * 64,
        world_evidence_sha256="2" * 64,
        graph_sha256="3" * 64,
        revision=1,
        invalidated_edge_count=0,
        revalidated_edge_count=0,
        state_ids=_frozen(np.asarray([1, 2, 3], dtype=np.uint64)),
        positions_m=_frozen(positions),
        yaw_rad=_frozen(np.zeros(3, dtype=np.float64)),
        cells=_frozen(np.asarray(cells, dtype=np.int32)),
        yaw_bin=_frozen(np.zeros(3, dtype=np.int32)),
        motion_mode=_frozen(np.zeros(3, dtype=np.int32)),
        body_z_m=_frozen(np.zeros((3, 2), dtype=np.float64)),
        path_cost=_frozen(np.asarray([0.0, 1.0, 1.0], dtype=np.float64)),
        forward_reachable=labels,
        returnable=labels,
        observation_state=labels,
        recoverable=labels,
        direct_successor=_frozen(
            np.asarray([False, True, True], dtype=np.bool_)
        ),
        edge_source_ids=_frozen(np.asarray([1, 2, 1], dtype=np.uint64)),
        edge_target_ids=_frozen(np.asarray([2, 1, 3], dtype=np.uint64)),
        edge_primitive_indices=_frozen(np.zeros(3, dtype=np.uint32)),
        edge_primitive_ids=("forward", "reverse", "outbound"),
        edge_cost=_frozen(np.ones(3, dtype=np.float64)),
    )


def test_primitive_oracle_recomputes_returnability_from_edges(
    monkeypatch,
) -> None:
    estimator = _GainEstimator()
    world, mission, _ = _fixture()
    graph = _oracle_graph(world.canvas)
    monkeypatch.setattr(
        CandidateBuilderV2,
        "build_from_primitive_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("oracle called production graph ranking")
        ),
    )

    result = FrontierOpportunityOracle(estimator).evaluate(
        world,
        mission,
        pose_map=Pose2(*world.canvas.grid_center_world(128, 128)),
        primitive_graph=graph,
    )

    assert result.frontier_anchor_count == 2
    assert result.platform_reachable_pose_count == 1
    assert result.opportunity_count == 1
    np.testing.assert_array_equal(estimator.calls[0], np.asarray([[128, 129]]))


def test_oracle_finds_observed_safe_positive_gain_without_candidate_ranking(
    monkeypatch,
) -> None:
    estimator = _GainEstimator()
    reachability = _Reachability(True)
    world, mission, projection = _fixture()
    monkeypatch.setattr(
        CandidateBuilderV2,
        "build",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("oracle called production candidate ranking")
        ),
    )

    result = FrontierOpportunityOracle(estimator).evaluate(
        world,
        mission,
        projection,
        pose_map=Pose2(1012.0, 1012.0),
        platform_reachability=reachability,
    )

    assert result.frontier_anchor_count > 0
    assert result.observed_safe_pose_count == result.frontier_anchor_count
    assert result.platform_reachable_pose_count == result.frontier_anchor_count
    assert result.opportunity_count == result.frontier_anchor_count
    assert len(estimator.calls) == 1
    np.testing.assert_array_equal(estimator.calls[0], reachability.calls[0])


def test_oracle_requires_platform_certification_before_counting_gain() -> None:
    estimator = _GainEstimator()
    world, mission, projection = _fixture()

    result = FrontierOpportunityOracle(estimator).evaluate(
        world,
        mission,
        projection,
        pose_map=Pose2(1012.0, 1012.0),
        platform_reachability=_Reachability(False),
    )

    assert result.frontier_anchor_count > 0
    assert result.platform_reachable_pose_count == 0
    assert result.opportunity_count == 0
    assert estimator.calls == []


def test_oracle_finds_positive_gain_pose_when_frontier_is_beyond_one_hop() -> None:
    canvas = MapCanvas.from_roi_bounds(
        "a" * 64, (500.0, 500.0, 1524.0, 1524.0)
    )
    robot = (128, 128)
    robot_x_m, robot_y_m = canvas.grid_center_world(*robot)
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[127:130, 128:142] = True
    roi = np.zeros((256, 256), dtype=np.float32)
    roi[127:130, 128:143] = 1.0
    world = ObservedWorld(
        canvas,
        np.zeros((256, 256), dtype=np.float32),
        observed,
        CanvasRatioLayer(canvas, np.zeros((256, 256), dtype=np.float32)),
        LocalObservation(
            canvas.identity,
            (
                robot_x_m - 3.2,
                robot_y_m - 3.2,
                robot_x_m + 3.2,
                robot_y_m + 3.2,
            ),
            np.zeros((32, 32), dtype=np.float32),
            np.ones((32, 32), dtype=np.bool_),
            np.zeros((32, 32), dtype=np.float32),
        ),
    )
    mission = MissionRaster(canvas, roi.copy(), roi)
    projection = PlatformProjection(
        canvas,
        observed.astype(np.float32),
        np.ones((32, 32), dtype=np.float32),
        np.ones((256, 256), dtype=np.float32),
        "test_only/proxy",
    )
    estimator = _GainEstimator()

    result = FrontierOpportunityOracle(estimator).evaluate(
        world,
        mission,
        projection,
        pose_map=Pose2(robot_x_m, robot_y_m),
        platform_reachability=_Reachability(True),
    )

    assert result.opportunity_count > 0
    assert len(estimator.calls) == 1
    assert (128, 141) not in map(tuple, estimator.calls[0])
    distances_m = np.linalg.norm(
        estimator.calls[0] - np.asarray(robot), axis=1
    ) * canvas.geometry.resolution_m
    assert np.all(distances_m <= 30.0)


def test_oracle_uses_exact_pose_distance_at_sensor_boundary() -> None:
    canvas = MapCanvas.from_roi_bounds(
        "9" * 64, (500.0, 500.0, 1524.0, 1524.0)
    )
    robot = (128, 128)
    target = (133, 133)
    robot_center_x, robot_center_y = canvas.grid_center_world(*robot)
    pose = Pose2(robot_center_x - 1.9, robot_center_y + 1.9)
    target_x, target_y = canvas.grid_center_world(*target)
    assert math.hypot(target_x - pose.x_m, target_y - pose.y_m) > 30.0
    assert (
        np.linalg.norm(np.asarray(target) - np.asarray(robot))
        * canvas.geometry.resolution_m
        < 30.0
    )
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[robot] = True
    observed[target] = True
    roi = observed.astype(np.float32)
    world = ObservedWorld(
        canvas,
        np.zeros((256, 256), dtype=np.float32),
        observed,
        CanvasRatioLayer(canvas, np.zeros((256, 256), dtype=np.float32)),
        LocalObservation(
            canvas.identity,
            (
                pose.x_m - 3.2,
                pose.y_m - 3.2,
                pose.x_m + 3.2,
                pose.y_m + 3.2,
            ),
            np.zeros((32, 32), dtype=np.float32),
            np.ones((32, 32), dtype=np.bool_),
            np.zeros((32, 32), dtype=np.float32),
        ),
    )
    mission = MissionRaster(canvas, roi.copy(), roi)
    projection = PlatformProjection(
        canvas,
        observed.astype(np.float32),
        np.ones((32, 32), dtype=np.float32),
        np.ones((256, 256), dtype=np.float32),
        "test_only/proxy",
    )

    result = FrontierOpportunityOracle(_GainEstimator()).evaluate(
        world,
        mission,
        projection,
        pose_map=pose,
        platform_reachability=_Reachability(True),
    )

    assert result.opportunity_count == 0


def test_oracle_counts_certified_zero_gain_transit_to_remote_frontier() -> None:
    canvas = MapCanvas.from_roi_bounds(
        "b" * 64, (500.0, 500.0, 524.0, 524.0)
    )
    robot = (128, 128)
    robot_x_m, robot_y_m = canvas.grid_center_world(*robot)
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[127:130, 128:142] = True
    roi = np.zeros((256, 256), dtype=np.float32)
    roi[127:130, 128:143] = 1.0
    world = ObservedWorld(
        canvas,
        np.zeros((256, 256), dtype=np.float32),
        observed,
        CanvasRatioLayer(canvas, np.zeros((256, 256), dtype=np.float32)),
        LocalObservation(
            canvas.identity,
            (
                robot_x_m - 3.2,
                robot_y_m - 3.2,
                robot_x_m + 3.2,
                robot_y_m + 3.2,
            ),
            np.zeros((32, 32), dtype=np.float32),
            np.ones((32, 32), dtype=np.bool_),
            np.zeros((32, 32), dtype=np.float32),
        ),
    )
    mission = MissionRaster(canvas, roi.copy(), roi)
    projection = PlatformProjection(
        canvas,
        observed.astype(np.float32),
        np.ones((32, 32), dtype=np.float32),
        np.ones((256, 256), dtype=np.float32),
        "test_only/proxy",
    )
    estimator = _GainEstimator(gains=0.0)

    class OneStepReachability(_Reachability):
        def filter(
            self,
            candidate_cells: np.ndarray,
            *,
            target_positions_map: np.ndarray | None = None,
        ) -> CandidateReachabilityResult:
            del target_positions_map
            self.calls.append(candidate_cells.copy())
            mask = (candidate_cells[:, 0] == 128) & (
                candidate_cells[:, 1] == 129
            )
            mask = np.ascontiguousarray(mask, dtype=np.bool_)
            return CandidateReachabilityResult(
                mask,
                {
                    "platform_unreachable_count": int(
                        (~mask).sum(dtype=np.int64)
                    )
                },
            )

    result = FrontierOpportunityOracle(estimator).evaluate(
        world,
        mission,
        projection,
        pose_map=Pose2(robot_x_m, robot_y_m),
        platform_reachability=OneStepReachability(True),
        excluded_cells=set(map(tuple, np.column_stack(np.nonzero(observed)))),
        backtrack_pose=Pose2(
            *canvas.grid_center_world(128, 129), elevation_m=0.0
        ),
    )

    assert result.platform_reachable_pose_count > 0
    assert result.opportunity_count > 0
