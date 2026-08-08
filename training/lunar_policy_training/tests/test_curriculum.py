from __future__ import annotations

import math

import torch
import pytest
from lunar_planner_training_bridge import MotionReference, PlannerBridge

from lunar_policy_training.curriculum import CurriculumSchedule
from lunar_policy_training.capability_freeze import (
    FrozenInterval,
    FrozenLeggedBodyPrimitive,
    FrozenLeggedCapability,
    FrozenObservationCapability,
    FrozenPlatformCapability,
    FrozenVec3,
)
from lunar_policy_training.environment.parallel_pool import (
    ParallelActions,
    ParallelEnvPool,
    ParallelPoolError,
)
from lunar_policy_training.environment.macro_step import PolicyAction
from lunar_policy_training.proxy_scenario import (
    ProxyEnvironmentFactory,
    _ProxyEpisode,
    proxy_environment_factory,
    proxy_observation,
)
from lunar_policy_training.reward import compute_transition_reward


def _build_episode_request(
    episode: _ProxyEpisode, action: PolicyAction
):
    identity = episode.observation.observation_identities[0]
    return episode.build_request(action, identity).request


def test_frozen_curriculum_reserves_sixteen_hours_for_joint_training() -> None:
    """Would fail if calibration or warmups could consume the joint minimum."""
    schedule = CurriculumSchedule()

    assert schedule.total_gpu_limit_s == 24 * 60 * 60
    assert schedule.calibration_limit_s == 2 * 60 * 60
    assert schedule.platform_warmup_order == ("WHEELED", "LEGGED", "HOPPER")
    assert schedule.platform_warmup_limit_s == 2 * 60 * 60
    assert schedule.joint_minimum_s == 16 * 60 * 60
    assert schedule.joint_worker_allocation == {
        "WHEELED": 8,
        "LEGGED": 8,
        "HOPPER": 8,
    }


def test_early_phase_savings_transfer_only_to_joint() -> None:
    """Would fail if early savings vanished or shortened joint training."""
    schedule = CurriculumSchedule()
    frozen = schedule.freeze(
        calibration_used_s=60 * 60,
        warmup_used_s={
            "WHEELED": 60 * 60,
            "LEGGED": 2 * 60 * 60,
            "HOPPER": 30 * 60,
        },
    )

    assert frozen.joint_budget_s == 19.5 * 60 * 60
    assert frozen.total_gpu_limit_s == 24 * 60 * 60
    assert frozen.formal_seed == 4080
    assert len(frozen.reward_calibration_seeds) == 3


def test_active_gpu_phase_order_and_allocations_prepare_formal_train() -> None:
    """Would fail if public train skipped a warmup or mixed platform workers."""
    schedule = CurriculumSchedule()
    calibration_end = 120.0

    assert schedule.phase_for(
        consumed_gpu_s=calibration_end, calibration_end_gpu_s=calibration_end
    ) == "warmup_wheeled"
    assert schedule.phase_for(
        consumed_gpu_s=calibration_end + 2 * 60 * 60,
        calibration_end_gpu_s=calibration_end,
    ) == "warmup_legged"
    assert schedule.phase_for(
        consumed_gpu_s=calibration_end + 4 * 60 * 60,
        calibration_end_gpu_s=calibration_end,
    ) == "warmup_hopper"
    assert schedule.phase_for(
        consumed_gpu_s=calibration_end + 6 * 60 * 60,
        calibration_end_gpu_s=calibration_end,
    ) == "joint"
    assert schedule.worker_allocation("warmup_legged", selected_workers=24) == {
        "LEGGED": 24
    }
    assert schedule.worker_allocation("joint", selected_workers=18) == {
        "WHEELED": 6,
        "LEGGED": 6,
        "HOPPER": 6,
    }


def test_curriculum_sampler_is_proxy_and_deterministic() -> None:
    """Would fail if scenario identity drifted or claimed real capability."""
    first = CurriculumSchedule().scenario_for(
        platform_type="LEGGED", scenario_index=2
    )
    second = CurriculumSchedule().scenario_for(
        platform_type="LEGGED", scenario_index=2
    )

    assert first == second
    assert first.proxy is True
    assert first.capability_id.startswith("proxy-")
    assert proxy_environment_factory.run_kind == "development-smoke"
    assert first.scenario_seed == second.scenario_seed
    assert CurriculumSchedule().scenario_schedule_id.startswith(
        "proxy-scenario-schedule-v1:"
    )


def test_formal_curriculum_identity_matches_injected_capability() -> None:
    """Would fail if a worker request named a capability other than its bundle."""
    capability = FrozenPlatformCapability(
        platform_type="LEGGED",
        capability_type="lunar-planner-legged-capability/v2",
        capability_version="legged-2026.08",
        platform_id="test-only-legged",
        base_frame_id="base_link",
        platform_document_path="legged/platform.yaml",
        observation_document_path="legged/observation.json",
        urdf_path="legged/rover.urdf",
        mesh_paths=("legged/body.stl",),
        observation_capability=FrozenObservationCapability(30.0, 2.0 * math.pi),
        typed_capability=FrozenLeggedCapability(
            reference_point="base_link",
            body_extent_m=FrozenVec3(0.68, 0.33, 0.35),
            platform_mass_kg=15.89,
            maximum_payload_kg=10.0,
            maximum_slope_rad=0.5235987755982988,
            maximum_step_height_m=0.5,
            maximum_gap_width_m=0.3,
            minimum_body_clearance_m=0.3,
            step_vertical_rate_mps=0.1,
            body_height_m=FrozenInterval(0.28, 0.38),
            forward_speed_mps=FrozenInterval(-1.5, 1.5),
            lateral_speed_mps=FrozenInterval(-0.8, 0.8),
            yaw_rate_radps=FrozenInterval(-1.0, 1.0),
            maximum_linear_acceleration_mps2=1.0,
            maximum_yaw_acceleration_radps2=1.0,
            motion_primitives=(
                FrozenLeggedBodyPrimitive(
                    primitive_id="test-forward",
                    kind="FORWARD",
                    body_frame_displacement_m=FrozenVec3(1.0, 0.0, 0.0),
                    yaw_change_rad=0.0,
                ),
            ),
        ),
        content_sha256="a" * 64,
        resources=(),
    )

    scenario = CurriculumSchedule().scenario_for(
        platform_type="LEGGED",
        scenario_index=4,
        capability=capability,
        scenario_schedule_id="nasa-polar-train/v1",
    )

    assert scenario.proxy is False
    assert scenario.capability_version == "legged-2026.08"
    assert scenario.capability_sha256 == "a" * 64
    assert scenario.capability_id == "legged-2026.08:" + "a" * 64
    assert scenario.scenario_schedule_id == "nasa-polar-train/v1"


def test_evaluation_schedule_exposes_every_frozen_proxy_scenario() -> None:
    """Would fail if evaluation silently ran only the first frozen terrain."""
    schedule = CurriculumSchedule()

    assert schedule.evaluation_scenario_indices == (0, 1, 2)
    assert tuple(
        schedule.scenario_for(
            platform_type="HOPPER", scenario_index=scenario_index
        ).scenario_seed
        for scenario_index in schedule.evaluation_scenario_indices
    ) == (12000, 12001, 12002)


def test_proxy_macro_step_uses_real_v3_and_changes_observation() -> None:
    """Would fail if Task 4 used an empty request or a synthetic score fixture."""
    with ParallelEnvPool(
        allocation={"WHEELED": 1, "LEGGED": 1, "HOPPER": 1},
        observation_template=proxy_observation(0, "WHEELED", step=0),
        environment_factory=proxy_environment_factory,
        reward_fn=compute_transition_reward,
        worker_timeout_seconds=30.0,
    ) as pool:
        initial = pool.reset()
        stepped = pool.step(
            ParallelActions(
                candidate_indices=torch.zeros(3, dtype=torch.int64),
                thetas=torch.zeros(3, dtype=torch.float32),
            ),
            policy_version=7,
        )

    assert not torch.equal(
        initial.observations.coverage_summary,
        stepped.observations.coverage_summary,
    )
    assert not torch.equal(
        initial.observations.pose_features,
        stepped.observations.pose_features,
    )
    assert all(outcome.name == "NEW_REFERENCE_AVAILABLE" for outcome in stepped.planning_outcomes)
    assert bool(torch.isfinite(stepped.rewards).all())
    assert stepped.decision_budget_consumed.tolist() == [1, 1, 1]
    assert stepped.observations.pose_features[:, 5].tolist() == pytest.approx(
        [0.875, 0.875, 0.875]
    )
    assert stepped.observations.observation_identities is not None
    assert len(stepped.execution_events) == 3
    assert stepped.execution_events[0].reference_samples_consumed > 1
    assert stepped.execution_events[1].reference_samples_consumed > 1
    assert stepped.execution_events[2].hopper_commitment_states == (
        "JUMP_COMMITTED",
        "IN_FLIGHT",
        "LANDED_HOLD",
    )


@pytest.mark.parametrize("platform_type", ["WHEELED", "LEGGED"])
def test_proxy_executor_consumes_cpp_trajectory_endpoint(
    platform_type: str,
) -> None:
    """Would fail if execution teleported to the requested target without a trajectory."""
    episode = _ProxyEpisode(0, platform_type, scenario_index=0)
    request = _build_episode_request(
        episode, PolicyAction(frontier_index=0, theta_rad=0.0)
    )
    output = PlannerBridge().plan(request)
    assert output.reference is not None

    executed = episode.execute_reference(output.reference)

    assert executed.mission_observed_delta > 0.0
    assert executed.execution_state == "DECISION_BOUNDARY"
    assert executed.execution_events.reference_samples_consumed > 1
    assert executed.execution_events.execution_failure_count == 0
    assert executed.next_observation.pose_features[0, 0].item() == pytest.approx(
        4.5 / 12.0
    )


@pytest.mark.parametrize("platform_type", ["WHEELED", "LEGGED"])
def test_ground_proxy_request_preserves_sampled_absolute_yaw(
    platform_type: str,
) -> None:
    """Would fail if theta were ignored or hidden behind an unconstrained goal."""
    episode = _ProxyEpisode(0, platform_type, scenario_index=0)
    sampled_theta = -1.125

    request = _build_episode_request(
        episode,
        PolicyAction(frontier_index=0, theta_rad=sampled_theta)
    )

    assert request.goal.yaw_rad == pytest.approx(sampled_theta)
    assert request.goal.yaw_tolerance_rad == pytest.approx(math.pi / 24.0)
    assert request.config.wheel.yaw_bin_count == 64
    assert request.config.legged.yaw_bin_count == 64


def test_hopper_proxy_request_ignores_theta_and_uses_exact_point() -> None:
    """Hopper theta is inactive and the current planner requires no goal yaw."""
    episode = _ProxyEpisode(0, "HOPPER", scenario_index=0)

    request = _build_episode_request(
        episode,
        PolicyAction(frontier_index=0, theta_rad=-1.125),
    )

    assert request.goal.yaw_rad is None
    assert request.goal.yaw_tolerance_rad == 0.0
    assert request.goal.target.tolerance_m == 0.0
    assert not hasattr(request, "hopper_propellant")


@pytest.mark.parametrize("platform_type", ["WHEELED", "LEGGED"])
def test_ground_proxy_capability_can_plan_non_cardinal_sampled_yaw(
    platform_type: str,
) -> None:
    """Would fail if proxy primitives could not realize the fixed 64-bin yaw."""
    episode = _ProxyEpisode(0, platform_type, scenario_index=0)
    request = _build_episode_request(
        episode,
        PolicyAction(frontier_index=0, theta_rad=-1.125)
    )

    output = PlannerBridge().plan(request)

    assert output.reference is not None, output.reason_code


@pytest.mark.parametrize("platform_type", ["WHEELED", "LEGGED"])
def test_ground_theta_changes_terminal_yaw_and_execution_time(
    platform_type: str,
) -> None:
    """The current planner gives ground theta a learnable non-coverage signal."""
    outputs = []
    for theta in (0.0, -1.125):
        episode = _ProxyEpisode(0, platform_type, scenario_index=0)
        request = _build_episode_request(
            episode,
            PolicyAction(frontier_index=0, theta_rad=theta),
        )
        output = PlannerBridge().plan(request)
        assert output.reference is not None, output.reason_code
        outputs.append(output.reference.data)

    final_points = tuple(output.points[-1] for output in outputs)
    final_yaws = tuple(
        math.atan2(
            2.0
            * (
                point.pose.orientation.w * point.pose.orientation.z
                + point.pose.orientation.x * point.pose.orientation.y
            ),
            1.0
            - 2.0
            * (
                point.pose.orientation.y * point.pose.orientation.y
                + point.pose.orientation.z * point.pose.orientation.z
            ),
        )
        for point in final_points
    )
    durations = tuple(
        point.time_from_start.total_seconds() for point in final_points
    )

    assert final_yaws[0] == pytest.approx(0.0, abs=1.0e-6)
    assert final_yaws[1] == pytest.approx(-1.125, abs=1.0e-6)
    assert durations[0] != durations[1]


def test_proxy_executor_rejects_unexecutable_reference_without_success_gain() -> None:
    """Would fail if an empty C++ reference still received synthetic coverage."""
    episode = _ProxyEpisode(0, "WHEELED", scenario_index=0)
    _build_episode_request(
        episode, PolicyAction(frontier_index=0, theta_rad=0.0)
    )
    empty_reference = MotionReference()
    empty_reference.platform_type = "WHEELED"

    executed = episode.execute_reference(empty_reference)

    assert executed.mission_observed_delta == 0.0
    assert executed.priority_observed_delta == 0.0
    assert executed.execution_events.execution_failure_count == 1
    assert executed.execution_events.safety_violation_count == 1
    assert executed.execution_events.selected_action_observed_safe is False
    assert executed.next_observation.pose_features[0, 0].item() == pytest.approx(
        2.5 / 12.0
    )


def test_hopper_proxy_consumes_hop_and_reaches_landed_decision_boundary() -> None:
    """Would fail if the hopper skipped its committed execution lifecycle."""
    episode = _ProxyEpisode(0, "HOPPER", scenario_index=0)
    request = _build_episode_request(
        episode, PolicyAction(frontier_index=0, theta_rad=0.0)
    )
    output = PlannerBridge().plan(request)
    assert output.reference is not None

    executed = episode.execute_reference(output.reference)

    assert executed.execution_state == "LANDED_HOLD"
    assert executed.execution_events.hopper_commitment_states == (
        "JUMP_COMMITTED",
        "IN_FLIGHT",
        "LANDED_HOLD",
    )
    assert executed.execution_events.hopper_commitment_violation_count == 0
    assert executed.execution_events.reference_samples_consumed == 1
    assert executed.mission_observed_delta > 0.0


def test_proxy_action_has_fixed_target_and_unique_frontiers_can_converge() -> None:
    """Would fail if step count secretly changed an action or capped coverage below 95%."""
    worker = ProxyEnvironmentFactory(scenario_index=0)(0, "WHEELED")
    environment = worker.environment

    def take_action(candidate_index: int):
        result = environment.advance_until_decision_boundary(
            lambda _observation: PolicyAction(
                frontier_index=candidate_index,
                theta_rad=0.0,
            )
        )
        assert result.transition is not None
        return result.transition

    distractor = take_action(2)
    first = take_action(0)
    repeated = take_action(0)
    second = take_action(1)

    assert distractor.mission_observed_delta == 0.0
    assert first.mission_observed_delta > 0.0
    assert repeated.mission_observed_delta == 0.0
    assert second.mission_observed_delta > 0.0
    assert worker.initial_observation.candidate_mask.shape == (1, 64)
    assert bool(worker.initial_observation.candidate_mask[0, :3].all())
    assert not bool(worker.initial_observation.candidate_mask[0, 3:].any())
    assert first.next_observation.frontier_features[0, 0, 5].item() == 0.0
    assert first.next_observation.frontier_features[0, 1, 5].item() > 0.0
    assert second.next_observation.pose_features[0, 4].item() >= 0.95


def test_parallel_pool_auto_resets_terminal_proxy_episode() -> None:
    """Would fail if a terminal worker remained stuck in repeat-only state."""
    with ParallelEnvPool(
        allocation={"WHEELED": 1},
        observation_template=proxy_observation(0, "WHEELED", step=0),
        environment_factory=ProxyEnvironmentFactory(scenario_index=0),
        reward_fn=compute_transition_reward,
        worker_timeout_seconds=30.0,
    ) as pool:
        pool.reset()
        pool.step(
            ParallelActions(
                candidate_indices=torch.tensor([0], dtype=torch.int64),
                thetas=torch.zeros(1, dtype=torch.float32),
            ),
            policy_version=1,
        )
        terminal = pool.step(
            ParallelActions(
                candidate_indices=torch.tensor([1], dtype=torch.int64),
                thetas=torch.zeros(1, dtype=torch.float32),
            ),
            policy_version=1,
        )
        continued = pool.step(
            ParallelActions(
                candidate_indices=torch.tensor([0], dtype=torch.int64),
                thetas=torch.zeros(1, dtype=torch.float32),
            ),
            policy_version=2,
        )

    assert terminal.dones.tolist() == [True]
    assert terminal.rewards[0].item() > 0.0
    assert terminal.observations.pose_features[0, 4].item() == pytest.approx(0.05)
    assert continued.dones.tolist() == [False]
    assert continued.observations.pose_features[0, 4].item() == pytest.approx(
        0.525
    )
    assert continued.planning_outcomes[0].name == "NEW_REFERENCE_AVAILABLE"


def test_parallel_pool_can_preserve_terminal_observation_for_evaluation() -> None:
    """Would fail if evaluation saw the reset episode instead of final coverage."""
    with ParallelEnvPool(
        allocation={"WHEELED": 1},
        observation_template=proxy_observation(0, "WHEELED", step=0),
        environment_factory=ProxyEnvironmentFactory(scenario_index=0),
        reward_fn=compute_transition_reward,
        worker_timeout_seconds=30.0,
        auto_reset=False,
    ) as pool:
        pool.reset()
        pool.step(
            ParallelActions(
                candidate_indices=torch.tensor([0], dtype=torch.int64),
                thetas=torch.zeros(1, dtype=torch.float32),
            ),
            policy_version=1,
        )
        terminal = pool.step(
            ParallelActions(
                candidate_indices=torch.tensor([1], dtype=torch.int64),
                thetas=torch.zeros(1, dtype=torch.float32),
            ),
            policy_version=1,
        )
        with pytest.raises(ParallelPoolError, match="terminated worker"):
            pool.step(
                ParallelActions(
                    candidate_indices=torch.tensor([0], dtype=torch.int64),
                    thetas=torch.zeros(1, dtype=torch.float32),
                ),
                policy_version=2,
            )

    assert terminal.dones.tolist() == [True]
    assert terminal.observations.pose_features[0, 4].item() == pytest.approx(1.0)
    assert terminal.observations.coverage_summary[0, 0].mean().item() == (
        pytest.approx(1.0)
    )


def test_parallel_pool_allows_explicit_reset_after_preserved_terminal() -> None:
    """Evaluation must reset a finished row before the pool accepts another action."""
    with ParallelEnvPool(
        allocation={"WHEELED": 1},
        observation_template=proxy_observation(0, "WHEELED", step=0),
        environment_factory=ProxyEnvironmentFactory(scenario_index=0),
        reward_fn=compute_transition_reward,
        worker_timeout_seconds=30.0,
        auto_reset=False,
    ) as pool:
        pool.reset()
        for candidate_index in (0, 1):
            terminal = pool.step(
                ParallelActions(
                    candidate_indices=torch.tensor(
                        [candidate_index], dtype=torch.int64
                    ),
                    thetas=torch.zeros(1, dtype=torch.float32),
                ),
                policy_version=1,
            )
        assert terminal.dones.tolist() == [True]

        reset = pool.reset_terminated_workers((0,), policy_version=2)
        continued = pool.step(
            ParallelActions(
                candidate_indices=torch.tensor([0], dtype=torch.int64),
                thetas=torch.zeros(1, dtype=torch.float32),
            ),
            policy_version=2,
        )

    assert reset.dones.tolist() == [False]
    assert reset.observations.pose_features[0, 4].item() == pytest.approx(0.05)
    assert continued.dones.tolist() == [False]
