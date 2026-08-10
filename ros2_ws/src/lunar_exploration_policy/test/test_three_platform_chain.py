from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from lunar_exploration_policy.action_selection import DeterministicPolicy
from lunar_exploration_policy.coordinator import (
    ClosedLoopCoordinator,
    ExecutionFeedback,
    PlannerResult,
)
from lunar_exploration_policy.identity import DecisionIdentity
from lunar_exploration_policy.inference import OnnxPolicyRuntime
from lunar_exploration_policy.observation_runtime import Fed9ObservationRuntime
from lunar_exploration_policy.ros_runtime import planner_goal_to_ros
from lunar_policy_training.environment.candidate_builder import CandidateBuilderV2
from lunar_policy_training.environment.observation_builder import (
    LocalObservation,
    MissionRaster,
    ObservedWorld,
    PlatformProjection,
    Pose2,
)
from lunar_policy_training.environment.visibility import (
    NativeVisibilityEstimator,
    SensorGeometry,
)
from lunar_policy_training.polar_data.hazards import CanvasRatioLayer
from lunar_policy_training.polar_data.raster import MapCanvas


def _model_dir() -> Path:
    value = os.environ.get("LUNAR_INTERFACE_V1_MODEL_DIR")
    if not value:
        pytest.skip("real interface-v1 model directory was not supplied")
    return Path(value).resolve(strict=True)


def _snapshot(platform: str, generation: int):
    canvas = MapCanvas("c" * 64, (0.0, 0.0, 1024.0, 1024.0))
    pose = Pose2(512.0 + generation * 4.0, 512.0, 0.0)
    observed = np.zeros((256, 256), np.bool_)
    # 旧 30 m 探测包络内保留一圈 observed/unknown 前沿。
    observed[122:134, 122:134] = True
    obstacle = np.zeros((256, 256), np.float32)
    local = LocalObservation(
        canvas.identity,
        (pose.x_m - 3.2, pose.y_m - 3.2, pose.x_m + 3.2, pose.y_m + 3.2),
        np.zeros((32, 32), np.float32),
        np.ones((32, 32), np.bool_),
        np.zeros((32, 32), np.float32),
    )
    world = ObservedWorld(
        canvas,
        np.zeros((256, 256), np.float32),
        observed,
        CanvasRatioLayer(canvas, obstacle),
        local,
    )
    mission = MissionRaster(
        canvas,
        np.ones((256, 256), np.float32),
        np.ones((256, 256), np.float32),
    )
    projection = PlatformProjection(
        canvas,
        np.ones((256, 256), np.float32),
        np.ones((32, 32), np.float32),
        np.ones((256, 256), np.float32),
        "cpp_v3/chain-fixture",
    )
    runtime = Fed9ObservationRuntime(
        CandidateBuilderV2(
            NativeVisibilityEstimator(
                SensorGeometry(30.0, 2.0 * math.pi), resolution_m=4.0
            )
        )
    )
    return runtime.build(
        world,
        mission,
        pose,
        projection,
        platform,
        DecisionIdentity(
            generation,
            f"map-{generation}",
            f"state-{generation}",
            generation * 1_000_000_000,
            "DECISION_BOUNDARY",
        ),
    )


@pytest.mark.parametrize("platform", ("WHEELED", "LEGGED", "HOPPER"))
def test_real_onnx_runs_two_complete_decision_boundaries(platform: str) -> None:
    coordinator = ClosedLoopCoordinator(
        DeterministicPolicy(OnnxPolicyRuntime(_model_dir()))
    )

    first = coordinator.start_decision(_snapshot(platform, 1), mission_id="mission")
    first_message = planner_goal_to_ros(first, stamp_ns=1_000_000_000)
    coordinator.accept_planner_result(
        PlannerResult(first.request_id, True, "plan-1", "REFERENCE_AVAILABLE")
    )
    terminal = "LANDED_HOLD" if platform == "HOPPER" else "SEGMENT_COMPLETE"
    assert coordinator.accept_feedback(
        ExecutionFeedback(1, platform, "plan-1", terminal)
    )

    second = coordinator.start_decision(_snapshot(platform, 2), mission_id="mission")
    second_message = planner_goal_to_ros(second, stamp_ns=2_000_000_000)

    assert first.request_id != second.request_id
    assert first.identity.map_snapshot_id != second.identity.map_snapshot_id
    assert first_message.goal.has_yaw_constraint is (platform != "HOPPER")
    assert second_message.goal.has_yaw_constraint is (platform != "HOPPER")
    assert second.identity.robot_state_id == "state-2"
