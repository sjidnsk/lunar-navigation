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
    DecisionBoundaryResult,
    PreparedPlanRequest,
    create_v3_environment,
)
from lunar_policy_training.policy.observation import (  # noqa: E402
    ObservationIdentity,
    PolicyBatch,
)


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
    pose = torch.zeros((1, 5), dtype=torch.float32)
    pose[0, 0] = float(worker_index)
    return PolicyBatch(
        prior_channels=torch.zeros((1, 4, 256, 256), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 3, 256, 256), dtype=torch.float32),
        local_crop=torch.zeros((1, 4, 32, 32), dtype=torch.float32),
        frontier_features=torch.zeros((1, 64, 12), dtype=torch.float32),
        pose_features=pose,
        candidate_mask=torch.tensor([[True, True, True] + [False] * 61], dtype=torch.bool),
        platform_context=platform_context,
        observation_identities=(
            ObservationIdentity(
                episode_id=f"pool-{worker_index}",
                mission_revision=1,
                map_snapshot_id="map-1",
                robot_state_id=f"robot-{worker_index}",
                state_time_ns=1_000,
                execution_state="DECISION_BOUNDARY",
                candidate_set_id="candidates-1",
            ),
        ),
    )


def _real_v3_worker_factory(
    worker_index: int, platform_type: str
) -> ParallelEnvironmentWorker:
    observation = _observation(worker_index, platform_type)
    request = TrainingPlanRequest()
    request.request_id = f"parallel-worker-{worker_index}"

    def build_request(action, identity):
        request.state_time.nanoseconds_since_epoch = identity.state_time_ns
        return PreparedPlanRequest(request=request, identity=identity)

    environment = create_v3_environment(
        platform_type=platform_type,
        request_builder=build_request,
        initial_observation=observation,
        observation_provider=lambda: observation,
        committed_hop_executor=(lambda: None) if platform_type == "HOPPER" else None,
    )
    return ParallelEnvironmentWorker(
        environment=environment,
        initial_observation=observation,
    )


def _zero_reward(transition) -> float:
    return float(transition.mission_observed_delta)


class _CrashEnvironment:
    def __init__(self, observation: PolicyBatch) -> None:
        self.current_observation = observation

    def advance_prepared_action(self, action, *, expected_identity) -> None:
        os._exit(23)


def _crash_worker_factory(
    worker_index: int, platform_type: str
) -> ParallelEnvironmentWorker:
    observation = _observation(worker_index, platform_type)
    return ParallelEnvironmentWorker(
        environment=_CrashEnvironment(observation),
        initial_observation=observation,
    )


class _PreparationOnlyEnvironment:
    def __init__(self, observation: PolicyBatch) -> None:
        self._observation = observation

    @property
    def current_observation(self) -> PolicyBatch:
        return self._observation

    def refresh_decision_boundary(self) -> DecisionBoundaryResult:
        if not bool(self._observation.candidate_mask.any()):
            return DecisionBoundaryResult(execution_state="NO_CANDIDATES")
        return DecisionBoundaryResult(execution_state="DECISION_READY")


class _NoActionThenReadyFactory:
    def __init__(self, *, mixed_workers: bool = False) -> None:
        self.mixed_workers = mixed_workers
        self.calls = 0

    def __call__(
        self, worker_index: int, platform_type: str
    ) -> ParallelEnvironmentWorker:
        self.calls += 1
        observation = _observation(worker_index + self.calls * 10, platform_type)
        selected = worker_index == 0 or not self.mixed_workers
        if selected and self.calls == 1:
            observation.candidate_mask.zero_()
        return ParallelEnvironmentWorker(
            environment=_PreparationOnlyEnvironment(observation),
            initial_observation=observation,
        )


class _EpisodeCursorFactory:
    def __call__(
        self, worker_index: int, platform_type: str
    ) -> ParallelEnvironmentWorker:
        return self.create_for_episode(
            worker_index,
            platform_type,
            0,
            platform_worker_index=worker_index,
            platform_worker_count=worker_index + 1,
        )

    def create_for_episode(
        self,
        worker_index: int,
        platform_type: str,
        episode_cursor: int,
        *,
        platform_worker_index: int,
        platform_worker_count: int,
    ) -> ParallelEnvironmentWorker:
        observation = _observation(worker_index, platform_type)
        identity = observation.observation_identities[0]
        observation.observation_identities = (
            ObservationIdentity(
                episode_id=f"cursor-{worker_index}-{episode_cursor}",
                mission_revision=identity.mission_revision,
                map_snapshot_id=f"map-{episode_cursor}",
                robot_state_id=identity.robot_state_id,
                state_time_ns=identity.state_time_ns,
                execution_state=identity.execution_state,
                candidate_set_id=identity.candidate_set_id,
            ),
        )
        return ParallelEnvironmentWorker(
            environment=_PreparationOnlyEnvironment(observation),
            initial_observation=observation,
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
        assert first.policy_decisions_consumed.tolist() == [1, 1, 1]
        assert first.observations.observation_identities is not None
        assert [
            identity.episode_id
            for identity in first.observations.observation_identities
        ] == ["pool-0", "pool-1", "pool-2"]
        assert first.observations.pose_features[:, 0].tolist() == pytest.approx(
            [0.0, 1.0, 2.0]
        )
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


def test_preserved_no_action_terminal_can_be_explicitly_reset(
) -> None:
    """Would fail if evaluation could not reset a no-action terminal row."""
    factory = _NoActionThenReadyFactory()
    with ParallelEnvPool(
        allocation={"WHEELED": 1},
        observation_template=_observation(0, "WHEELED"),
        environment_factory=factory,
        reward_fn=_zero_reward,
        worker_timeout_seconds=5.0,
        auto_reset=False,
    ) as pool:
        pool.reset()
        prepared = pool.prepare_decision_boundaries(policy_version=41)
        prepared_dones = prepared.dones.tolist()
        prepared_consumed = prepared.policy_decisions_consumed.tolist()
        prepared_outcomes = prepared.planning_outcomes
        prepared_events = prepared.execution_events
        reset = pool.reset_terminated_workers((0,), policy_version=42)
        reset_dones = reset.dones.tolist()
        actionable = pool.prepare_decision_boundaries(policy_version=42)

    assert prepared_dones == [True]
    assert prepared_consumed == [0]
    assert prepared_outcomes == ()
    assert prepared_events == ()
    assert reset_dones == [False]
    assert actionable.dones.tolist() == [False]
    assert bool(actionable.observations.candidate_mask[0].any())


def test_targeted_reset_preserves_every_unselected_worker_field() -> None:
    """Would fail if a local reset rewrote another worker's staged state."""
    factory = _NoActionThenReadyFactory(mixed_workers=True)
    with ParallelEnvPool(
        allocation={"WHEELED": 2},
        observation_template=_observation(0, "WHEELED"),
        environment_factory=factory,
        reward_fn=_zero_reward,
        worker_timeout_seconds=5.0,
        auto_reset=False,
    ) as pool:
        pool.reset()
        source = pool.prepare_decision_boundaries(policy_version=51)
        source_fields = {
            name: getattr(source.observations, name)[1].clone()
            for name in (
                "prior_channels",
                "coverage_summary",
                "local_crop",
                "frontier_features",
                "pose_features",
                "candidate_mask",
                "platform_context",
            )
        }
        source_reward = source.rewards[1].item()
        source_done = source.dones[1].item()
        source_identity = source.observations.observation_identities[1]
        source_version = source.policy_versions[1].item()

        reset = pool.reset_terminated_workers((0,), policy_version=52)

    assert reset.policy_versions.tolist() == [52, 51]
    for name, expected in source_fields.items():
        assert torch.equal(getattr(reset.observations, name)[1], expected)
    assert reset.rewards[1].item() == source_reward
    assert reset.dones[1].item() == source_done
    assert reset.observations.observation_identities[1] == source_identity
    assert source_version == 51


def test_update_boundary_rollover_advances_and_restores_each_episode_cursor() -> None:
    """Would fail if resume repeated scenes or rollover retained stale workers."""
    with ParallelEnvPool(
        allocation={"WHEELED": 2},
        observation_template=_observation(0, "WHEELED"),
        environment_factory=_EpisodeCursorFactory(),
        reward_fn=_zero_reward,
        worker_timeout_seconds=5.0,
        initial_episode_cursors=(4, 9),
    ) as pool:
        initial = pool.reset()
        first = pool.rollover_all_workers(policy_version=31)
        second = pool.rollover_all_workers(policy_version=32)

        assert pool.episode_cursors == (6, 11)

    assert [
        identity.episode_id
        for identity in initial.observations.observation_identities
    ] == ["cursor-0-4", "cursor-1-9"]
    assert [
        identity.episode_id
        for identity in first.observations.observation_identities
    ] == ["cursor-0-5", "cursor-1-10"]
    assert [
        identity.episode_id
        for identity in second.observations.observation_identities
    ] == ["cursor-0-6", "cursor-1-11"]
    assert first.rewards.tolist() == [0.0, 0.0]
    assert first.dones.tolist() == [False, False]
    assert first.policy_versions.tolist() == [31, 31]
