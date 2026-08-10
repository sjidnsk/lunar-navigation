from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from lunar_exploration_policy.identity import DecisionIdentity, IdentityMismatch
from lunar_exploration_policy.observation_runtime import Fed9ObservationRuntime
from lunar_policy_training.environment.candidate_builder import CandidateBuilderV2
from lunar_policy_training.environment.observation_builder import (
    LocalObservation,
    MissionRaster,
    ObservedWorld,
    PlatformProjection,
    Pose2,
)
from lunar_policy_training.environment.visibility import SensorGeometry
from lunar_policy_training.polar_data.hazards import CanvasRatioLayer
from lunar_policy_training.polar_data.raster import MapCanvas


class _UnitGainEstimator:
    def __init__(self) -> None:
        self.sensor = SensorGeometry(30.0, 2.0 * math.pi)
        self.resolution_m = 4.0

    def estimate_candidate_gains(
        self,
        observed_mask,
        obstacle_ratio,
        roi_ratio,
        priority_weight,
        candidate_cells,
    ) -> np.ndarray:
        return np.ones((candidate_cells.shape[0], 2), dtype=np.float32)


def _fixture():
    canvas = MapCanvas.from_roi_bounds("d" * 64, (500.0, 500.0, 524.0, 524.0))
    observed = np.zeros((256, 256), dtype=np.bool_)
    observed[120:136, 100:128] = True
    roi = np.zeros((256, 256), dtype=np.float32)
    roi[116:140, 96:144] = 1.0
    x_m, y_m = canvas.grid_center_world(128, 120)
    pose = Pose2(x_m, y_m, 0.25)
    local = LocalObservation(
        canvas.identity,
        (x_m - 3.2, y_m - 3.2, x_m + 3.2, y_m + 3.2),
        np.zeros((32, 32), np.float32),
        np.ones((32, 32), np.bool_),
        np.zeros((32, 32), np.float32),
    )
    world = ObservedWorld(
        canvas,
        np.zeros((256, 256), np.float32),
        observed,
        CanvasRatioLayer(canvas, np.zeros((256, 256), np.float32)),
        local,
    )
    mission = MissionRaster(canvas, roi.copy(), roi)
    projection = PlatformProjection(
        canvas,
        np.ones((256, 256), np.float32),
        np.ones((32, 32), np.float32),
        np.full((256, 256), 0.3, np.float32),
        "test_only/proxy",
    )
    return world, mission, pose, projection


def _identity() -> DecisionIdentity:
    return DecisionIdentity(
        mission_revision=7,
        map_snapshot_id="map-7",
        robot_state_id="state-11",
        state_time_ns=123456,
        execution_state="DECISION_BOUNDARY",
    )


def test_runtime_reuses_fed9_candidate_and_observation_builders() -> None:
    world, mission, pose, projection = _fixture()
    runtime = Fed9ObservationRuntime(
        CandidateBuilderV2(_UnitGainEstimator())
    )

    snapshot = runtime.build(
        world,
        mission,
        pose,
        projection,
        "WHEELED",
        _identity(),
    )

    assert snapshot.candidates.count > 0
    assert snapshot.arrays["frontier_features"].shape == (1, 64, 12)
    assert snapshot.arrays["candidate_mask"].shape == (1, 64)
    np.testing.assert_allclose(
        snapshot.arrays["pose_features"],
        np.asarray(
            [[
                0.470703125,
                0.501953125,
                math.sin(0.25),
                math.cos(0.25),
                448.0 / 1152.0,
            ]],
            np.float32,
        ),
        atol=1e-7,
        rtol=0.0,
    )
    assert snapshot.identity.candidate_set_id == snapshot.candidate_set_id


@pytest.mark.parametrize(
    "field,value",
    [
        ("mission_revision", 8),
        ("map_snapshot_id", "map-8"),
        ("robot_state_id", "state-12"),
        ("state_time_ns", 123457),
        ("execution_state", "EXECUTING"),
        ("candidate_set_id", "candidates-other"),
    ],
)
def test_snapshot_rejects_any_request_identity_drift(field: str, value) -> None:
    world, mission, pose, projection = _fixture()
    snapshot = Fed9ObservationRuntime(
        CandidateBuilderV2(_UnitGainEstimator())
    ).build(world, mission, pose, projection, "LEGGED", _identity())

    with pytest.raises(IdentityMismatch, match=field):
        snapshot.require_identity(
            replace(snapshot.identity, **{field: value})
        )


def test_decision_identity_is_immutable() -> None:
    identity = _identity()

    with pytest.raises(FrozenInstanceError):
        identity.map_snapshot_id = "changed"  # type: ignore[misc]
