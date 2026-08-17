from __future__ import annotations

import pytest
from lunar_planner_training_bridge import (
    PlannerOutput,
    PlanningOutcome,
    TrainingPlanRequest,
)

from lunar_policy_training.environment import formal_builder as formal_builder_module
from lunar_policy_training.environment.formal_builder import FormalEpisode
from lunar_policy_training.environment.observation_builder import Pose2


class _GroundStartQualificationHarness:
    def __init__(self, outputs: list[PlannerOutput]) -> None:
        self.platform_type = "WHEELED"
        self.scene_id = "scene-for-start-qualification"
        self._revision = 7
        self.requests: list[str] = []
        self._outputs = iter(outputs)
        self._bridge = self

    @staticmethod
    def _base_request(_global_map: object, _local_map: object) -> TrainingPlanRequest:
        return TrainingPlanRequest()

    def plan(self, request: TrainingPlanRequest) -> PlannerOutput:
        self.requests.append(request.request_id)
        return next(self._outputs)


def _output(outcome: PlanningOutcome, reason_code: str) -> PlannerOutput:
    output = PlannerOutput()
    output.outcome = outcome
    output.reason_code = reason_code
    return output


def test_ground_start_qualification_retries_one_abnormal_probe() -> None:
    """Would fail if one transient qualification probe still killed an update."""
    harness = _GroundStartQualificationHarness(
        [
            _output(
                PlanningOutcome.RESOURCE_EXHAUSTED,
                "PLANNER_CAPACITY_TRANSIENT",
            ),
            _output(
                PlanningOutcome.NEW_REFERENCE_AVAILABLE,
                "WHEEL_PLAN_AVAILABLE",
            ),
        ]
    )

    result = FormalEpisode._qualify_ground_start(
        harness,
        planner_global_map=object(),
        local_map=object(),
        pose=Pose2(12.0, 34.0, 0.5, "map", 2.0),
    )

    assert result is None
    assert harness.requests == [
        "formal-start-qualification/scene-for-start-qualification/7",
        "formal-start-qualification/scene-for-start-qualification/7/retry-1",
    ]


def test_ground_start_qualification_fails_closed_after_retry() -> None:
    """Would fail if repeated planner faults were swallowed or left unauditable."""
    harness = _GroundStartQualificationHarness(
        [
            _output(
                PlanningOutcome.RESOURCE_EXHAUSTED,
                "PLANNER_CAPACITY_TRANSIENT",
            ),
            _output(
                PlanningOutcome.INVALID_REQUEST,
                "BAD_REBUILT_REQUEST",
            ),
        ]
    )

    with pytest.raises(
        formal_builder_module.EnvironmentInvariantError,
        match=(
            "GROUND_START_QUALIFICATION_FAILED.*"
            "INVALID_REQUEST.*BAD_REBUILT_REQUEST"
        ),
    ):
        FormalEpisode._qualify_ground_start(
            harness,
            planner_global_map=object(),
            local_map=object(),
            pose=Pose2(12.0, 34.0, 0.5, "map", 2.0),
        )

    assert len(harness.requests) == 2


def test_ground_start_qualification_does_not_retry_an_unsafe_start() -> None:
    """Would fail if a valid start-resample decision were treated as transient."""
    harness = _GroundStartQualificationHarness(
        [
            _output(
                PlanningOutcome.NO_KNOWN_SAFE_ROUTE,
                "WHEEL_START_NOT_SAFE",
            ),
        ]
    )

    result = FormalEpisode._qualify_ground_start(
        harness,
        planner_global_map=object(),
        local_map=object(),
        pose=Pose2(12.0, 34.0, 0.5, "map", 2.0),
    )

    assert result == "WHEEL_START_NOT_SAFE"
    assert len(harness.requests) == 1
