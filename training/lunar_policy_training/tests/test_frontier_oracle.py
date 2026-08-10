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
)
from lunar_policy_training.environment.platform_reachability import (
    CandidateReachabilityResult,
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
        platform_reachability=_Reachability(False),
    )

    assert result.frontier_anchor_count > 0
    assert result.platform_reachable_pose_count == 0
    assert result.opportunity_count == 0
    assert estimator.calls == []
