from __future__ import annotations

import pathlib
import sys
from unittest.mock import Mock


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_planner_training_bridge import (  # noqa: E402
    ExecutionDirective,
    PlannerOutput,
    PlanningOutcome,
)
import pytest  # noqa: E402

from lunar_policy_training.environment.v3_environment import (  # noqa: E402
    EnvironmentInvariantError,
    V3ExplorationEnvironment,
)
from lunar_policy_training.environment.macro_step import PolicyAction  # noqa: E402
import torch  # noqa: E402

from lunar_policy_training.policy.observation import PolicyBatch  # noqa: E402


class _Bridge:
    def __init__(self, output: PlannerOutput) -> None:
        self.output = output

    def plan(self, request: object) -> PlannerOutput:
        return self.output


def _hopper_observation() -> PolicyBatch:
    return PolicyBatch(
        prior_channels=torch.zeros((1, 7, 4, 4), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        local_crop=torch.zeros((1, 8, 4, 4), dtype=torch.float32),
        frontier_features=torch.zeros((1, 2, 22), dtype=torch.float32),
        pose_features=torch.zeros((1, 6), dtype=torch.float32),
        candidate_mask=torch.tensor([[True, False]], dtype=torch.bool),
        platform_context=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
    )


def test_hopper_does_not_request_policy_while_committed() -> None:
    """Would fail if policy could replace a committed or in-flight hop."""
    hopper_env = V3ExplorationEnvironment(
        platform_type="HOPPER",
        bridge=Mock(),
        request_builder=Mock(),
        initial_observation=_hopper_observation(),
    )
    policy_spy = Mock()

    hopper_env.begin_committed_hop()
    result = hopper_env.advance_until_decision_boundary(policy_spy)

    assert policy_spy.call_count == 0
    assert result.execution_state == "LANDED_HOLD"


def test_hopper_rejects_explicit_action_while_committed() -> None:
    """Would fail if callers could bypass the no-policy commitment boundary."""
    hopper_env = V3ExplorationEnvironment(
        platform_type="HOPPER",
        bridge=Mock(),
        request_builder=Mock(),
        initial_observation=_hopper_observation(),
    )
    hopper_env.begin_committed_hop()

    with pytest.raises(EnvironmentInvariantError, match="committed"):
        hopper_env.step(PolicyAction(frontier_index=0, theta_rad=0.0))

    assert hopper_env.rollout_discarded is True
    assert hopper_env.training_stopped is True


def test_hopper_resumes_policy_only_after_landed_hold() -> None:
    """Would fail if the policy stayed disabled after the landing boundary."""
    output = PlannerOutput()
    output.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    output.directive = ExecutionDirective.NO_SAFE_REFERENCE
    output.reason_code = "LANDED_NO_ROUTE"
    hopper_env = V3ExplorationEnvironment(
        platform_type="HOPPER",
        bridge=_Bridge(output),
        request_builder=lambda action: action,
        initial_observation=_hopper_observation(),
    )
    policy_spy = Mock(return_value=PolicyAction(frontier_index=0, theta_rad=0.0))

    hopper_env.begin_committed_hop()
    hopper_env.advance_until_decision_boundary(policy_spy)
    result = hopper_env.advance_until_decision_boundary(policy_spy)

    assert policy_spy.call_count == 1
    assert result.execution_state == "LANDED_HOLD"
    assert result.transition is not None
    assert result.transition.reason_code == "LANDED_NO_ROUTE"
