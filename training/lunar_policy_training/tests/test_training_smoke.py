from __future__ import annotations

import json
import pathlib
import sys

import pytest
import torch
from lunar_planner_training_bridge import (
    ExecutionDirective,
    PlanningOutcome,
)


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))

from lunar_policy_training.cli import (  # noqa: E402
    ResumablePPOTrainer,
    run_cuda_interrupt_resume_smoke,
)
import lunar_policy_training.cli as training_cli  # noqa: E402
from lunar_policy_training.checkpoint import load_checkpoint  # noqa: E402
from lunar_policy_training.environment.macro_step import PlannerTransition  # noqa: E402
from lunar_policy_training.policy.cross_attention import CrossAttentionPolicy  # noqa: E402
from lunar_policy_training.policy.observation import PolicyBatch  # noqa: E402


def _transition() -> PlannerTransition:
    observation = PolicyBatch(
        prior_channels=torch.zeros((1, 7, 8, 8), dtype=torch.float32),
        coverage_summary=torch.zeros((1, 8, 8, 8), dtype=torch.float32),
        local_crop=torch.zeros((1, 8, 8, 8), dtype=torch.float32),
        frontier_features=torch.zeros((1, 2, 22), dtype=torch.float32),
        pose_features=torch.zeros((1, 6), dtype=torch.float32),
        candidate_mask=torch.tensor([[True, False]], dtype=torch.bool),
        platform_context=torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float32
        ),
    )
    return PlannerTransition(
        next_observation=observation,
        coverage_delta=0.5,
        goal_progress=0.25,
        normalized_plan_cost=0.1,
        normalized_elapsed_time=0.2,
        repeated_visit=False,
        planning_outcome=PlanningOutcome.INVALID_REQUEST,
        execution_directive=ExecutionDirective.NO_SAFE_REFERENCE,
        reason_code="TEST",
        terminated=False,
    )


def test_resumable_trainer_uses_injected_transition_reward() -> None:
    """Would fail if Task 4 rewards required replacing the Task 3 trainer."""
    trainer = ResumablePPOTrainer(
        CrossAttentionPolicy(),
        reward_fn=lambda transition: (
            transition.coverage_delta + transition.goal_progress
        ),
        device="cpu",
    )

    assert trainer.reward_transition(_transition()) == 0.75


@pytest.mark.cuda
def test_cuda_interrupt_resume_preserves_step_budget_and_allocation(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a real CUDA pause/resume reset progress or the warmup split."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    artifact_root = tmp_path / "cuda-interrupt-resume"
    monkeypatch.setattr(
        training_cli,
        "_proxy_rollout",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("public train/resume must not use proxy rollout")
        ),
    )

    evidence = run_cuda_interrupt_resume_smoke(
        config_path=(
            REPOSITORY_ROOT / "training/configs/rtx4080_super_smoke.yaml"
        ),
        artifact_root=artifact_root,
        repository_root=REPOSITORY_ROOT,
    )

    assert evidence.device_name == "NVIDIA GeForce RTX 4080 SUPER"
    assert evidence.interrupted_global_step >= 1
    assert evidence.resumed_global_step == evidence.interrupted_global_step + 1
    assert (
        evidence.resumed_consumed_gpu_seconds
        > evidence.interrupted_consumed_gpu_seconds
        > 0.0
    )
    assert evidence.signal_observed_at_update_boundary is True
    assert (artifact_root / "checkpoints/latest.pt").is_file()
    manifest = json.loads(
        (artifact_root / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["global_step"] == evidence.resumed_global_step
    assert manifest["consumed_gpu_seconds"] == (
        evidence.resumed_consumed_gpu_seconds
    )
    assert manifest["frozen_config"]["parallel"]["joint_workers"] == {
        "WHEELED": 8,
        "LEGGED": 8,
        "HOPPER": 8,
    }
    selected_workers = manifest["runtime_calibration"]["selected_workers"]
    selected_micro_batch = manifest["runtime_calibration"][
        "selected_micro_batch"
    ]
    assert all(
        measurement["optimizer_steps"] == 1
        and measurement["ipc_failures"] == 0
        for measurement in manifest["runtime_calibration"]["measurements"]
    )
    assert evidence.platform_allocation == {"WHEELED": selected_workers}
    checkpoint = load_checkpoint(artifact_root / "checkpoints/latest.pt")
    assert checkpoint.worker_allocation == evidence.platform_allocation
    assert checkpoint.curriculum_phase == "warmup_wheeled"
    assert checkpoint.micro_batch_size == selected_micro_batch
    assert checkpoint.latest_checkpoint_gpu_seconds == (
        checkpoint.consumed_gpu_seconds
    )
