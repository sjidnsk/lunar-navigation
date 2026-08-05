from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch
from lunar_planner_training_bridge import (
    ExecutionDirective,
    PlanningOutcome,
    TrainingPlanRequest,
)


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_policy_training.policy.cross_attention import CrossAttentionPolicy  # noqa: E402
from lunar_policy_training.policy.observation import (  # noqa: E402
    ObservationIdentity,
    PolicyBatch,
)
from lunar_policy_training.cli import _ParallelPoolVectorEnv  # noqa: E402
from lunar_policy_training.environment.macro_step import (  # noqa: E402
    PlannerTransition,
)
from lunar_policy_training.environment.parallel_pool import (  # noqa: E402
    ParallelEnvironmentWorker,
    ParallelEnvPool,
)
from lunar_policy_training.environment.v3_environment import (  # noqa: E402
    DecisionBoundaryResult,
    PreparedPlanRequest,
    create_v3_environment,
)
from lunar_policy_training.ppo.collector import (  # noqa: E402
    CollectedRollout,
    CollectorConfig,
    CollectorError,
    EnvStep,
    collect_rollout,
)
from lunar_policy_training.ppo.rollout import RolloutBatch  # noqa: E402


def _policy_batch(step: int = 0) -> PolicyBatch:
    batch_size = 3
    frontier = torch.zeros((batch_size, 64, 12), dtype=torch.float32)
    frontier[..., 0] = float(step) / 10.0
    mask = torch.zeros((batch_size, 64), dtype=torch.bool)
    mask[:, :3] = True
    pose_features = torch.zeros((batch_size, 6), dtype=torch.float32)
    pose_features[:, 5] = 1.0
    return PolicyBatch(
        prior_channels=torch.full(
            (batch_size, 4, 256, 256), float(step) / 100.0, dtype=torch.float32
        ),
        coverage_summary=torch.zeros(
            (batch_size, 3, 256, 256), dtype=torch.float32
        ),
        local_crop=torch.zeros((batch_size, 4, 32, 32), dtype=torch.float32),
        frontier_features=frontier,
        pose_features=pose_features,
        candidate_mask=mask,
        platform_context=torch.eye(3, dtype=torch.float32),
    )


class DeterministicVectorEnv:
    env_count = 3

    def __init__(self) -> None:
        self.reset_count = 0
        self.step_count = 0
        self.actions: list[tuple[np.ndarray, np.ndarray]] = []

    def reset(self) -> PolicyBatch:
        self.reset_count += 1
        self.step_count = 0
        return _policy_batch(0)

    def step(
        self, candidate_indices: np.ndarray, thetas: np.ndarray
    ) -> EnvStep:
        self.actions.append((candidate_indices.copy(), thetas.copy()))
        done_rows = (
            np.asarray([False, True, False], dtype=np.bool_)
            if self.step_count == 0
            else np.asarray([True, False, True], dtype=np.bool_)
        )
        self.step_count += 1
        return EnvStep(
            observations=_policy_batch(self.step_count),
            rewards=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            dones=done_rows,
        )


class InvalidTransitionEnv(DeterministicVectorEnv):
    def __init__(self, mutation: str) -> None:
        super().__init__()
        self.mutation = mutation

    def step(
        self, candidate_indices: np.ndarray, thetas: np.ndarray
    ) -> EnvStep:
        valid = super().step(candidate_indices, thetas)
        if self.mutation == "nonfinite_reward":
            valid.rewards[1] = np.nan
        elif self.mutation == "reward_dtype":
            valid.rewards = valid.rewards.astype(np.float64)
        elif self.mutation == "done_dtype":
            valid.dones = valid.dones.astype(np.int64)
        elif self.mutation == "nonfinite_observation":
            valid.observations.pose_features[0, 0] = torch.nan
        return valid


def _zero_value_policy() -> CrossAttentionPolicy:
    torch.manual_seed(11)
    policy = CrossAttentionPolicy().eval()
    with torch.no_grad():
        for parameter in policy.value_mlp.parameters():
            parameter.zero_()
    return policy


def test_collect_rollout_runs_real_vector_env_actions_and_hand_computed_gae() -> None:
    """Would fail if reset/action/step/bootstrap order or done-masked GAE were bypassed."""
    environment = DeterministicVectorEnv()

    collected = collect_rollout(
        environment,
        _zero_value_policy(),
        CollectorConfig(horizon=2, deterministic=True),
        device="cpu",
    )

    assert isinstance(collected, CollectedRollout)
    assert isinstance(collected.rollout, RolloutBatch)
    assert environment.reset_count == 1
    assert environment.step_count == 2
    assert len(environment.actions) == 2
    assert all(indices.dtype == np.int64 for indices, _ in environment.actions)
    assert all(thetas.dtype == np.float32 for _, thetas in environment.actions)
    assert collected.rewards.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert collected.dones.tolist() == [
        [False, True, False],
        [True, False, True],
    ]
    np.testing.assert_allclose(
        collected.rollout.returns,
        np.asarray([1.94525, 2.0, 5.83575, 1.0, 2.0, 3.0], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        collected.rollout.platform_context,
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    assert np.isfinite(collected.rollout.advantages).all()
    assert abs(float(collected.rollout.advantages.mean())) < 1.0e-6


def test_collector_bypasses_all_false_candidate_rows_before_policy_forward() -> None:
    """Would fail if a no-candidate row could enter policy action sampling."""
    environment = DeterministicVectorEnv()
    original_reset = environment.reset

    class CountingPolicy(CrossAttentionPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.forward_calls = 0

        def forward(self, batch: PolicyBatch):
            self.forward_calls += 1
            return super().forward(batch)

    def reset_with_no_candidate() -> PolicyBatch:
        batch = original_reset()
        batch.candidate_mask[1].zero_()
        return batch

    environment.reset = reset_with_no_candidate
    policy = CountingPolicy().eval()

    with pytest.raises(
        CollectorError, match="all-false candidate rows must bypass rollout collection"
    ):
        collect_rollout(
            environment,
            policy,
            CollectorConfig(horizon=1, deterministic=True),
            device="cpu",
        )

    assert policy.forward_calls == 0


def test_collector_rejects_exhausted_budget_without_resolver_before_policy() -> None:
    """Would fail if budget-zero rows entered policy on a generic vector env."""
    environment = DeterministicVectorEnv()
    original_reset = environment.reset

    def reset_with_exhausted_budget() -> PolicyBatch:
        batch = original_reset()
        batch.pose_features[1, 5] = 0.0
        batch.pose_features[[0, 2], 5] = 1.0
        return batch

    environment.reset = reset_with_exhausted_budget
    policy = _CountingPolicy().eval()

    with pytest.raises(
        CollectorError, match="must bypass rollout collection"
    ):
        collect_rollout(
            environment,
            policy,
            CollectorConfig(horizon=1, deterministic=True),
            device="cpu",
        )

    assert policy.forward_calls == 0


def _real_v3_worker(
    worker_index: int, platform_type: str
) -> ParallelEnvironmentWorker:
    observation = PolicyBatch(
        prior_channels=torch.zeros((1, 4, 256, 256), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 3, 256, 256), dtype=torch.float32),
        local_crop=torch.zeros((1, 4, 32, 32), dtype=torch.float32),
        frontier_features=torch.zeros((1, 64, 12), dtype=torch.float32),
        pose_features=torch.zeros((1, 6), dtype=torch.float32),
        candidate_mask=torch.tensor([[True, True, True] + [False] * 61], dtype=torch.bool),
        platform_context=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        observation_identities=(
            ObservationIdentity(
                episode_id=f"collector-{worker_index}",
                mission_revision=1,
                map_snapshot_id="map-1",
                robot_state_id="robot-1",
                state_time_ns=1_000,
                execution_state="DECISION_BOUNDARY",
                candidate_set_id="candidates-1",
            ),
        ),
    )
    request = TrainingPlanRequest()
    request.request_id = f"collector-v3-{worker_index}"

    def build_request(action, identity):
        request.state_time.nanoseconds_since_epoch = identity.state_time_ns
        return PreparedPlanRequest(request=request, identity=identity)

    return ParallelEnvironmentWorker(
        environment=create_v3_environment(
            platform_type=platform_type,
            request_builder=build_request,
            initial_observation=observation,
            observation_provider=lambda: observation,
        ),
        initial_observation=observation,
    )


def _planner_transition_reward(transition: PlannerTransition) -> float:
    assert isinstance(transition, PlannerTransition)
    return 0.5 if transition.planning_outcome == PlanningOutcome.INVALID_REQUEST else 0.0


def _boundary_observation(*, all_false: bool, generation: int) -> PolicyBatch:
    mask = [False] * 64
    if not all_false:
        mask[0] = True
    return PolicyBatch(
        prior_channels=torch.zeros((1, 4, 256, 256), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 3, 256, 256), dtype=torch.float32),
        local_crop=torch.zeros((1, 4, 32, 32), dtype=torch.float32),
        frontier_features=torch.zeros((1, 64, 12), dtype=torch.float32),
        pose_features=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 1.0]], dtype=torch.float32),
        candidate_mask=torch.tensor([mask], dtype=torch.bool),
        platform_context=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        observation_identities=(
            ObservationIdentity(
                episode_id=f"boundary-{generation}",
                mission_revision=1,
                map_snapshot_id=f"map-{generation}",
                robot_state_id=f"state-{generation}",
                state_time_ns=generation,
                execution_state="DECISION_BOUNDARY",
                candidate_set_id=f"candidates-{generation}",
            ),
        ),
    )


class _BoundaryProtocolEnvironment:
    def __init__(self, observation: PolicyBatch) -> None:
        self.observation = observation

    @property
    def current_observation(self) -> PolicyBatch:
        return self.observation

    def refresh_decision_boundary(self) -> DecisionBoundaryResult:
        if not bool(self.observation.candidate_mask.any()):
            return DecisionBoundaryResult(execution_state="NO_CANDIDATES")
        if float(self.observation.pose_features[0, 5]) <= 0.0:
            return DecisionBoundaryResult(
                execution_state="DECISION_BUDGET_EXHAUSTED"
            )
        return DecisionBoundaryResult(execution_state="DECISION_READY")

    def advance_until_decision_boundary(self, policy) -> DecisionBoundaryResult:
        if not bool(self.observation.candidate_mask.any()):
            return DecisionBoundaryResult(execution_state="NO_CANDIDATES")
        policy(self.observation)
        next_observation = _boundary_observation(
            all_false=True,
            generation=self.observation.observation_identities[0].state_time_ns,
        )
        next_observation.pose_features[0, 5] = 0.875
        self.observation = next_observation
        return DecisionBoundaryResult(
            execution_state="DECISION_BOUNDARY",
            transition=PlannerTransition(
                next_observation=next_observation,
                coverage_delta=0.0,
                goal_progress=0.0,
                normalized_plan_cost=0.0,
                normalized_elapsed_time=0.0,
                repeated_visit=False,
                planning_outcome=PlanningOutcome.NO_KNOWN_SAFE_ROUTE,
                execution_directive=ExecutionDirective.NO_SAFE_REFERENCE,
                reason_code="LAST_CANDIDATE_REJECTED",
                terminated=False,
            ),
            decision_budget_consumed=1,
        )

    def advance_prepared_action(
        self, action, *, expected_identity: ObservationIdentity
    ) -> DecisionBoundaryResult:
        if expected_identity != self.observation.observation_identities[0]:
            raise AssertionError("stale prepared identity")
        return self.advance_until_decision_boundary(lambda observation: action)


class _ResettingBoundaryFactory:
    def __init__(self, *, initial_all_false: bool) -> None:
        self.initial_all_false = initial_all_false
        self.calls = 0

    def __call__(
        self, worker_index: int, platform_type: str
    ) -> ParallelEnvironmentWorker:
        self.calls += 1
        observation = _boundary_observation(
            all_false=self.initial_all_false and self.calls == 1,
            generation=self.calls,
        )
        return ParallelEnvironmentWorker(
            environment=_BoundaryProtocolEnvironment(observation),
            initial_observation=observation,
        )


class _BudgetOneFactory:
    def __init__(self, *, total: int = 1, remaining: int = 1) -> None:
        self.calls = 0
        self.total = total
        self.remaining = remaining

    def __call__(
        self, worker_index: int, platform_type: str
    ) -> ParallelEnvironmentWorker:
        self.calls += 1
        observation = _boundary_observation(all_false=False, generation=self.calls)
        observation.pose_features[0, 5] = 0.777
        request = TrainingPlanRequest()
        request.request_id = f"budget-one-{self.calls}"

        def build_request(action, identity):
            request.state_time.nanoseconds_since_epoch = identity.state_time_ns
            return PreparedPlanRequest(request=request, identity=identity)

        environment = create_v3_environment(
            platform_type=platform_type,
            request_builder=build_request,
            initial_observation=observation,
            observation_provider=lambda: observation,
            total_decision_budget=self.total,
            remaining_decision_budget=self.remaining,
        )
        return ParallelEnvironmentWorker(
            environment=environment,
            initial_observation=observation,
        )


class _CountingPolicy(CrossAttentionPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.budget_ratios: list[list[float]] = []

    def forward(self, batch: PolicyBatch):
        self.forward_calls += 1
        self.budget_ratios.append(batch.pose_features[:, 5].tolist())
        return super().forward(batch)


def _quarter_reward(transition: PlannerTransition) -> float:
    return 0.25 * transition.next_observation.observation_identities[0].state_time_ns


def test_budget_exhaustion_resolves_before_next_policy_forward() -> None:
    """Would fail if a budget-zero row reached policy before done/reset."""
    template = _boundary_observation(all_false=False, generation=1)
    policy = _CountingPolicy().eval()
    with ParallelEnvPool(
        allocation={"WHEELED": 1},
        observation_template=template,
        environment_factory=_BudgetOneFactory(),
        reward_fn=_quarter_reward,
        worker_timeout_seconds=5.0,
    ) as pool:
        collected = collect_rollout(
            _ParallelPoolVectorEnv(pool, policy_version=29),
            policy,
            CollectorConfig(horizon=2, deterministic=True),
            device="cpu",
        )

    assert policy.forward_calls == 3  # two real actions plus valid bootstrap
    assert len(collected.rollout) == 2
    assert collected.dones.tolist() == [[True], [True]]


def test_first_production_policy_forward_sees_restored_budget_ratio() -> None:
    """Would fail if pool preparation published producer's stale ratio."""
    template = _boundary_observation(all_false=False, generation=1)
    policy = _CountingPolicy().eval()
    with ParallelEnvPool(
        allocation={"WHEELED": 1},
        observation_template=template,
        environment_factory=_BudgetOneFactory(total=4, remaining=1),
        reward_fn=_quarter_reward,
        worker_timeout_seconds=5.0,
    ) as pool:
        collect_rollout(
            _ParallelPoolVectorEnv(pool, policy_version=31),
            policy,
            CollectorConfig(horizon=2, deterministic=True),
            device="cpu",
        )

    assert policy.budget_ratios[0] == pytest.approx([0.25])


@pytest.mark.parametrize("initial_all_false", [True, False])
def test_production_collector_resolves_initial_or_final_all_false_without_policy_action(
    initial_all_false: bool,
) -> None:
    """Would fail if pool resolution called policy or discarded the last action row."""
    factory = _ResettingBoundaryFactory(initial_all_false=initial_all_false)
    template = _boundary_observation(all_false=initial_all_false, generation=1)
    policy = _CountingPolicy().eval()
    with ParallelEnvPool(
        allocation={"WHEELED": 1},
        observation_template=template,
        environment_factory=factory,
        reward_fn=_quarter_reward,
        worker_timeout_seconds=5.0,
    ) as pool:
        collected = collect_rollout(
            _ParallelPoolVectorEnv(pool, policy_version=23),
            policy,
            CollectorConfig(horizon=2, deterministic=True),
            device="cpu",
        )

    assert policy.forward_calls == 3  # two actions plus one valid bootstrap
    assert len(collected.rollout) == 2
    assert collected.rewards.tolist() == (
        [[0.5], [0.75]] if initial_all_false else [[0.25], [0.5]]
    )
    assert collected.dones.tolist() == [[True], [True]]


def test_production_pool_adapter_collects_real_v3_reward_gae_at_one_policy_version() -> None:
    """Would fail if train/resume could replace the Task 2/C++ rollout with a proxy."""
    with ParallelEnvPool(
        allocation={"WHEELED": 1},
        observation_template=_real_v3_worker(0, "WHEELED").initial_observation,
        environment_factory=_real_v3_worker,
        reward_fn=_planner_transition_reward,
        worker_timeout_seconds=5.0,
    ) as pool:
        environment = _ParallelPoolVectorEnv(pool, policy_version=17)
        collected = collect_rollout(
            environment,
            _zero_value_policy(),
            CollectorConfig(horizon=2, deterministic=True),
            device="cpu",
        )

    assert collected.rewards.tolist() == [[0.5], [0.5]]
    assert environment.policy_versions == [17, 17]
    assert environment.planning_outcomes == [
        PlanningOutcome.INVALID_REQUEST,
        PlanningOutcome.INVALID_REQUEST,
    ]
    assert all(environment.reason_codes)
    assert collected.rollout.returns.tolist() == pytest.approx([0.972625, 0.5])
    assert np.isfinite(collected.rollout.advantages).all()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("nonfinite_reward", "rewards must be finite"),
        ("reward_dtype", "rewards must be float32"),
        ("done_dtype", "dones must be boolean"),
        ("nonfinite_observation", "pose_features must contain only finite values"),
    ),
)
def test_collect_rollout_rejects_invalid_transition(
    mutation: str, message: str
) -> None:
    """Would fail if an invalid environment transition entered the training batch."""
    with pytest.raises(CollectorError, match=message):
        collect_rollout(
            InvalidTransitionEnv(mutation),
            _zero_value_policy(),
            CollectorConfig(horizon=1, deterministic=True),
            device="cpu",
        )
