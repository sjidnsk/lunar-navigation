from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from lunar_planner_training_bridge import (
    PlannerOutput,
    PlanningOutcome,
    TrainingPlanRequest,
)

from lunar_policy_training.environment import formal_builder as formal_builder_module
from lunar_policy_training.environment.formal_builder import FormalEpisode
from lunar_policy_training.environment.observation_builder import Pose2
from lunar_policy_training.policy.observation import PolicyBatch


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


def test_ground_continuation_uses_certified_endpoint_without_start_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A certified reference endpoint must not trigger an origin-to-origin plan."""
    previous = PolicyBatch(
        prior_channels=torch.zeros((1, 1), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 1), dtype=torch.float32),
        local_crop=torch.zeros((1, 1), dtype=torch.float32),
        frontier_features=torch.zeros((1, 1), dtype=torch.float32),
        pose_features=torch.zeros((1, 5), dtype=torch.float32),
        candidate_mask=torch.tensor([[True]], dtype=torch.bool),
        platform_context=torch.zeros((1, 1), dtype=torch.float32),
    )

    class _ContinuationHarness:
        platform_type = "WHEELED"
        _revision = 8
        capability = SimpleNamespace(platform_id="wheel-platform")
        sensor_state = SimpleNamespace(
            evidence_generation=9,
            physical_evidence_sha256=lambda: "a" * 64,
        )
        controller = SimpleNamespace(current_observation=previous)
        loaded = SimpleNamespace(
            scene=SimpleNamespace(
                base_canvas=SimpleNamespace(
                    bounds_m=(0.0, 0.0, 100.0, 100.0),
                    geometry=SimpleNamespace(size_m=100.0),
                )
            )
        )

        def __init__(self) -> None:
            self._snapshot = SimpleNamespace(
                candidates=object(),
                world=object(),
                projection=object(),
                physical_reachability=object(),
                candidate_universe=object(),
            )
            self.qualification_calls = 0

        def _qualify_ground_start(self, **_: object) -> str | None:
            self.qualification_calls += 1
            raise AssertionError(
                "ground continuation repeated start qualification"
            )

    harness = _ContinuationHarness()
    monkeypatch.setattr(
        formal_builder_module,
        "_physical_capability_content_sha256",
        lambda _: "b" * 64,
    )

    result = FormalEpisode._build_ground_continuation_observation(
        harness,
        Pose2(12.0, 34.0, 0.5, "map", 2.0),
        global_map=object(),
        planner_global_map=object(),
        local_map=object(),
    )

    assert harness.qualification_calls == 0
    assert harness._snapshot.ground_start_resample_reason is None
    assert torch.equal(result.candidate_mask, previous.candidate_mask)
    assert float(result.pose_features[0, 0]) == pytest.approx(0.12)
    assert float(result.pose_features[0, 1]) == pytest.approx(0.66)
