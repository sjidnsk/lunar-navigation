from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch
from lunar_planner_training_bridge import PlanningOutcome, TrainingPlanRequest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_policy_training.policy.cross_attention import CrossAttentionPolicy  # noqa: E402
from lunar_policy_training.policy.observation import PolicyBatch  # noqa: E402
from lunar_policy_training.cli import _ParallelPoolVectorEnv  # noqa: E402
from lunar_policy_training.environment.macro_step import PlannerTransition  # noqa: E402
from lunar_policy_training.environment.parallel_pool import (  # noqa: E402
    ParallelEnvironmentWorker,
    ParallelEnvPool,
)
from lunar_policy_training.environment.v3_environment import (  # noqa: E402
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
    return PolicyBatch(
        prior_channels=torch.full(
            (batch_size, 4, 256, 256), float(step) / 100.0, dtype=torch.float32
        ),
        coverage_summary=torch.zeros(
            (batch_size, 3, 256, 256), dtype=torch.float32
        ),
        local_crop=torch.zeros((batch_size, 4, 32, 32), dtype=torch.float32),
        frontier_features=frontier,
        pose_features=torch.zeros((batch_size, 6), dtype=torch.float32),
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
    )
    request = TrainingPlanRequest()
    request.request_id = f"collector-v3-{worker_index}"
    return ParallelEnvironmentWorker(
        environment=create_v3_environment(
            platform_type=platform_type,
            request_builder=lambda action: request,
            initial_observation=observation,
        ),
        initial_observation=observation,
    )


def _planner_transition_reward(transition: PlannerTransition) -> float:
    assert isinstance(transition, PlannerTransition)
    return 0.5 if transition.planning_outcome == PlanningOutcome.INVALID_REQUEST else 0.0


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
