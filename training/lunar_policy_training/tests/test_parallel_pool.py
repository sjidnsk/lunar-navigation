from __future__ import annotations

import pathlib
import os
import sys

import pytest
import torch

from lunar_planner_training_bridge import TrainingPlanRequest


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))

from lunar_policy_training.config import load_training_config  # noqa: E402
from lunar_policy_training.environment.parallel_pool import (  # noqa: E402
    ParallelActions,
    ParallelEnvironmentWorker,
    ParallelEnvPool,
    ParallelPoolError,
    joint_worker_allocation,
)
from lunar_policy_training.environment.v3_environment import (  # noqa: E402
    create_v3_environment,
)
from lunar_policy_training.policy.observation import PolicyBatch  # noqa: E402


@pytest.fixture
def resolved_config():
    return load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
    )


def test_joint_pool_has_eight_workers_per_platform() -> None:
    """Would fail if any platform lost its fixed share of the 24-worker pool."""
    allocation = joint_worker_allocation(total_workers=24)

    assert allocation == {"WHEELED": 8, "LEGGED": 8, "HOPPER": 8}


def test_checkpoint_interval_is_thirty_minutes(resolved_config) -> None:
    """Would fail if latest.pt were scheduled later or earlier than 30 minutes."""
    assert resolved_config.checkpoint_interval_seconds == 1800


def test_joint_candidate_interval_is_one_hour(resolved_config) -> None:
    """Would fail if immutable joint candidates did not use one-hour boundaries."""
    assert resolved_config.candidate_checkpoint_interval_seconds == 3600


def _observation(worker_index: int, platform_type: str) -> PolicyBatch:
    platform_index = {"WHEELED": 0, "LEGGED": 1, "HOPPER": 2}[platform_type]
    platform_context = torch.zeros((1, 3), dtype=torch.float32)
    platform_context[0, platform_index] = 1.0
    pose = torch.zeros((1, 6), dtype=torch.float32)
    pose[0, 0] = float(worker_index)
    return PolicyBatch(
        prior_channels=torch.zeros((1, 7, 8, 8), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 8, 8, 8), dtype=torch.float32),
        local_crop=torch.zeros((1, 8, 8, 8), dtype=torch.float32),
        frontier_features=torch.zeros((1, 2, 22), dtype=torch.float32),
        pose_features=pose,
        candidate_mask=torch.tensor([[True, False]], dtype=torch.bool),
        platform_context=platform_context,
    )


def _real_v3_worker_factory(
    worker_index: int, platform_type: str
) -> ParallelEnvironmentWorker:
    observation = _observation(worker_index, platform_type)
    request = TrainingPlanRequest()
    request.request_id = f"parallel-worker-{worker_index}"
    environment = create_v3_environment(
        platform_type=platform_type,
        request_builder=lambda action: request,
        initial_observation=observation,
        committed_hop_executor=(lambda: None) if platform_type == "HOPPER" else None,
    )
    return ParallelEnvironmentWorker(
        environment=environment,
        initial_observation=observation,
    )


def _zero_reward(transition) -> float:
    return float(transition.coverage_delta)


class _CrashEnvironment:
    def step(self, action) -> None:
        os._exit(23)


def _crash_worker_factory(
    worker_index: int, platform_type: str
) -> ParallelEnvironmentWorker:
    return ParallelEnvironmentWorker(
        environment=_CrashEnvironment(),
        initial_observation=_observation(worker_index, platform_type),
    )


def _actions(worker_count: int) -> ParallelActions:
    return ParallelActions(
        candidate_indices=torch.zeros((worker_count,), dtype=torch.int64),
        thetas=torch.zeros((worker_count,), dtype=torch.float32),
    )


def test_parallel_pool_uses_real_worker_processes_shared_double_buffers() -> None:
    """Would fail if workers shared one bridge or copied rollouts through a queue."""
    allocation = {"WHEELED": 1, "LEGGED": 1, "HOPPER": 1}
    with ParallelEnvPool(
        allocation=allocation,
        observation_template=_observation(0, "WHEELED"),
        environment_factory=_real_v3_worker_factory,
        reward_fn=_zero_reward,
        worker_timeout_seconds=5.0,
    ) as pool:
        initial = pool.reset()
        first = pool.step(_actions(3), policy_version=7)
        second = pool.step(_actions(3), policy_version=8)

        assert pool.worker_count == 3
        assert len(set(pool.worker_pids)) == 3
        assert pool.worker_thread_limits == [
            ("1", "1"),
            ("1", "1"),
            ("1", "1"),
        ]
        assert len(pool.shared_observation_buffers) == 2
        assert all(
            tensor.is_shared()
            for buffer in pool.shared_observation_buffers
            for tensor in buffer.values()
        )
        assert all(
            tensor.is_shared()
            for buffer in pool.shared_action_buffers
            for tensor in buffer.values()
        )
        if torch.cuda.is_available():
            assert pool.staging_buffers_are_pinned is True
        assert [initial.buffer_index, first.buffer_index, second.buffer_index] == [
            0,
            1,
            0,
        ]
        assert first.policy_versions.tolist() == [7, 7, 7]
        assert second.policy_versions.tolist() == [8, 8, 8]
        assert first.observations.platform_context.tolist() == [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]


def test_mixed_policy_versions_fail_closed_and_discard_rollout() -> None:
    """Would fail if samples from different policy versions reached one update."""
    pool = ParallelEnvPool(
        allocation={"WHEELED": 1},
        observation_template=_observation(0, "WHEELED"),
        environment_factory=_real_v3_worker_factory,
        reward_fn=_zero_reward,
        worker_timeout_seconds=5.0,
    )
    pool.reset()

    with pytest.raises(ParallelPoolError, match="policy version"):
        pool.validate_policy_versions(
            torch.tensor([4, 5], dtype=torch.int64), expected_policy_version=4
        )

    assert pool.rollout_discarded is True
    assert pool.training_stopped is True
    pool.close()


def test_nonfinite_action_fails_closed_before_worker_dispatch() -> None:
    """Would fail if a non-finite tensor could enter planner execution."""
    pool = ParallelEnvPool(
        allocation={"WHEELED": 1},
        observation_template=_observation(0, "WHEELED"),
        environment_factory=_real_v3_worker_factory,
        reward_fn=_zero_reward,
        worker_timeout_seconds=5.0,
    )
    pool.reset()
    actions = _actions(1)
    actions.thetas[0] = torch.nan

    with pytest.raises(ParallelPoolError, match="finite"):
        pool.step(actions, policy_version=1)

    assert pool.rollout_discarded is True
    assert pool.training_stopped is True
    pool.close()


def test_worker_crash_never_silently_reduces_worker_count() -> None:
    """Would fail if a dead worker were omitted and training continued."""
    pool = ParallelEnvPool(
        allocation={"WHEELED": 1},
        observation_template=_observation(0, "WHEELED"),
        environment_factory=_crash_worker_factory,
        reward_fn=_zero_reward,
        worker_timeout_seconds=2.0,
    )
    pool.reset()

    with pytest.raises(ParallelPoolError, match="worker"):
        pool.step(_actions(1), policy_version=2)

    assert pool.worker_count == 1
    assert pool.rollout_discarded is True
    assert pool.training_stopped is True
    pool.close()
