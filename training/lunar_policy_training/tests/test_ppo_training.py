from __future__ import annotations

from dataclasses import fields
import pathlib
import sys

import numpy as np
import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_policy_training.policy.cross_attention import (  # noqa: E402
    CrossAttentionPolicy,
    sample_action,
)
from lunar_policy_training.policy.observation import PolicyBatch  # noqa: E402
from lunar_policy_training.ppo import rollout_core, trainer_core  # noqa: E402
from lunar_policy_training.ppo.rollout import RolloutBatch, compute_gae  # noqa: E402
from lunar_policy_training.ppo.trainer import PPOTrainer  # noqa: E402


def _policy_batch() -> PolicyBatch:
    mask = torch.zeros((2, 64), dtype=torch.bool)
    mask[:, :3] = True
    return PolicyBatch(
        prior_channels=torch.zeros((2, 4, 256, 256), dtype=torch.float32),
        coverage_summary=torch.zeros((2, 3, 256, 256), dtype=torch.float32),
        local_crop=torch.zeros((2, 4, 32, 32), dtype=torch.float32),
        frontier_features=torch.zeros((2, 64, 12), dtype=torch.float32),
        pose_features=torch.zeros((2, 6), dtype=torch.float32),
        candidate_mask=mask,
        platform_context=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
    )


def _rollout_from_current_policy(policy: CrossAttentionPolicy) -> RolloutBatch:
    policy_batch = _policy_batch()
    with torch.no_grad():
        output = policy(policy_batch)
        sample = sample_action(output, policy_batch.candidate_mask, deterministic=True)
    return RolloutBatch(
        prior_channels=policy_batch.prior_channels.numpy(),
        coverage_summary=policy_batch.coverage_summary.numpy(),
        local_crop=policy_batch.local_crop.numpy(),
        frontier_features=policy_batch.frontier_features.numpy(),
        pose_features=policy_batch.pose_features.numpy(),
        candidate_mask=policy_batch.candidate_mask.numpy(),
        platform_context=policy_batch.platform_context.numpy(),
        selected_frontier_indices=sample.selected_frontier_index.numpy(),
        selected_thetas=sample.selected_theta.numpy(),
        old_log_prob_total=sample.log_prob_total.numpy(),
        old_values=output.value.numpy(),
        advantages=np.asarray([1.0, -1.0], dtype=np.float32),
        returns=output.value.numpy() + np.float32(1.0),
    )


def test_public_gae_uses_core_and_matches_hand_computed_terminal_sequence() -> None:
    """Would fail if public rollout drifted from frozen GAE recurrence or terminal masking."""
    assert compute_gae is rollout_core.compute_gae

    result = compute_gae(
        rewards=np.asarray([[1.0], [2.0]], dtype=np.float32),
        values=np.asarray([[0.5], [1.0]], dtype=np.float32),
        dones=np.asarray([[False], [True]], dtype=np.bool_),
        last_values=np.asarray([9.0], dtype=np.float32),
    )

    np.testing.assert_allclose(
        result.raw_advantages[:, 0],
        np.asarray([2.44025, 1.0], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        result.returns[:, 0],
        np.asarray([2.94025, 2.0], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        result.normalized_advantages[:, 0],
        np.asarray([1.0, -1.0], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_rollout_batch_is_ephemeral_seven_input_storage_with_device_select() -> None:
    """Would fail if durable snapshot fields returned or select omitted platform context."""
    policy = CrossAttentionPolicy().eval()
    rollout = _rollout_from_current_policy(policy)

    assert [field.name for field in fields(RolloutBatch)] == [
        "prior_channels",
        "coverage_summary",
        "local_crop",
        "frontier_features",
        "pose_features",
        "candidate_mask",
        "platform_context",
        "selected_frontier_indices",
        "selected_thetas",
        "old_log_prob_total",
        "old_values",
        "advantages",
        "returns",
    ]
    selected = rollout.select(np.asarray([1, 0], dtype=np.int64), device="cpu")
    selected_batch = selected[0]
    assert isinstance(selected_batch, PolicyBatch)
    assert selected_batch.platform_context.shape == (2, 3)
    assert torch.equal(
        selected_batch.platform_context,
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert selected[1].dtype == torch.int64
    assert all(value.dtype == torch.float32 for value in selected[2:])


def test_single_cpu_ppo_update_uses_core_loss_and_changes_a_parameter() -> None:
    """Would fail if trainer only reported metrics without a real optimizer update."""
    torch.manual_seed(7)
    policy = CrossAttentionPolicy()
    rollout = _rollout_from_current_policy(policy)
    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }

    metrics = PPOTrainer(policy, learning_rate=1.0e-3).update(rollout)

    assert metrics.optimizer_steps == 1
    assert metrics.parameter_change_l2 > 0.0
    assert np.isfinite(metrics.total_loss)
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in policy.named_parameters()
    )
    assert PPOTrainer.loss_function is trainer_core.compute_ppo_loss_terms


def test_ppo_update_uses_the_frozen_physical_microbatch_for_real_backward() -> None:
    """Would fail if calibration selected a value that formal training ignored."""
    torch.manual_seed(19)
    policy = CrossAttentionPolicy()
    rollout = _rollout_from_current_policy(policy)
    forward_batch_sizes: list[int] = []
    handle = policy.register_forward_pre_hook(
        lambda module, args: forward_batch_sizes.append(
            int(args[0].prior_channels.shape[0])
        )
    )
    try:
        trainer = PPOTrainer(policy, learning_rate=1.0e-3)
        metrics = trainer.update(rollout, micro_batch_size=1)
    finally:
        handle.remove()

    assert forward_batch_sizes == [1, 1]
    assert trainer.last_micro_batch_size == 1
    assert metrics.optimizer_steps == 1
    assert metrics.parameter_change_l2 > 0.0
