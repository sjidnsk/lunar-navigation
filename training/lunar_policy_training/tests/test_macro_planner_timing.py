from __future__ import annotations

from datetime import timedelta

from lunar_planner_training_bridge import (
    ExecutionDirective,
    PlannerOutput,
    PlanningOutcome,
)

from lunar_policy_training.environment.macro_step import PolicyAction
from lunar_policy_training.environment.v3_environment import (
    V3ExplorationEnvironment,
)
from lunar_policy_training.proxy_scenario import proxy_observation


class _Bridge:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, _request: object) -> PlannerOutput:
        self.calls += 1
        output = PlannerOutput()
        output.outcome = PlanningOutcome.NO_KNOWN_SAFE_ROUTE
        output.directive = ExecutionDirective.NO_SAFE_REFERENCE
        output.reason_code = "NO_ROUTE"
        output.diagnostics.elapsed = timedelta(milliseconds=25)
        return output


def test_macro_transition_reports_request_level_planner_time() -> None:
    bridge = _Bridge()
    environment = V3ExplorationEnvironment(
        platform_type="WHEELED",
        bridge=bridge,
        request_builder=lambda action: action,
        initial_observation=proxy_observation(0, "WHEELED", step=0),
    )

    transition = environment.step(PolicyAction(0, 0.0))

    assert bridge.calls == 1
    assert transition.execution_events.planner_call_count == 1
    assert transition.execution_events.planner_elapsed_s >= 0.0
