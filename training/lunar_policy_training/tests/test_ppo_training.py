from __future__ import annotations

from dataclasses import asdict, fields, replace
import pathlib
import sys

import numpy as np
import pytest
import torch
import yaml


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_policy_training.policy.cross_attention import (  # noqa: E402
    CrossAttentionPolicy,
    PolicyOutput,
    recompute_action_log_probs,
    sample_action,
)
from lunar_policy_training.policy.observation import PolicyBatch  # noqa: E402
from lunar_policy_training.config import (  # noqa: E402
    PPOConfig,
    TrainingConfigError,
    load_training_config,
    resolve_training_config,
    validate_ppo_config,
)
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
        pose_features=torch.zeros((2, 5), dtype=torch.float32),
        candidate_mask=mask,
        platform_context=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ),
    )


def _ppo_config() -> PPOConfig:
    return load_training_config(
        pathlib.Path(__file__).resolve().parents[3]
        / "training/configs/rtx4080_super_smoke.yaml"
    ).ppo


def _rollout_from_current_policy(
    policy: CrossAttentionPolicy, *, old_log_prob_offset: float = 0.0
) -> RolloutBatch:
    policy_batch = _policy_batch()
    with torch.no_grad():
        output = policy(policy_batch)
        sample = sample_action(
            output,
            policy_batch.candidate_mask,
            policy_batch.platform_context,
            deterministic=True,
        )
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
        old_log_prob_total=(
            sample.log_prob_total.numpy() + np.float32(old_log_prob_offset)
        ),
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
    """Would fail if four configured epochs did not perform real AdamW updates."""
    torch.manual_seed(7)
    policy = CrossAttentionPolicy()
    rollout = _rollout_from_current_policy(policy)
    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }

    trainer = PPOTrainer(policy, config=_ppo_config())
    metrics = trainer.update(rollout)

    assert metrics.optimizer_steps == 4
    assert metrics.epochs_completed == 4
    assert metrics.target_kl_early_stopped is False
    assert metrics.parameter_change_l2 > 0.0
    assert np.isfinite(metrics.total_loss)
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in policy.named_parameters()
    )
    assert PPOTrainer.loss_function is trainer_core.compute_ppo_loss_terms
    assert trainer.optimizer.defaults["lr"] == 3.0e-4
    assert trainer.optimizer.defaults["weight_decay"] == 1.0e-4
    assert trainer.optimizer.defaults["eps"] == 1.0e-5


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
        trainer = PPOTrainer(policy, config=_ppo_config())
        metrics = trainer.update(rollout, micro_batch_size=1)
    finally:
        handle.remove()

    assert forward_batch_sizes == [1, 1] * 4
    assert trainer.last_micro_batch_size == 1
    assert metrics.optimizer_steps == 4
    assert metrics.parameter_change_l2 > 0.0


def test_loss_uses_separate_frontier_and_conditional_theta_entropy_terms() -> None:
    """Would fail if theta entropy were omitted or merged under the frontier weight."""
    terms = trainer_core.compute_ppo_loss_terms(
        new_log_prob_total=torch.zeros((1,), dtype=torch.float32),
        old_log_prob_total=torch.zeros((1,), dtype=torch.float32),
        normalized_advantage=torch.zeros((1,), dtype=torch.float32),
        new_value=torch.tensor([2.0], dtype=torch.float32),
        old_value=torch.zeros((1,), dtype=torch.float32),
        returns=torch.ones((1,), dtype=torch.float32),
        frontier_entropy=torch.tensor([2.0], dtype=torch.float32),
        theta_entropy=torch.tensor([3.0], dtype=torch.float32),
        theta_active=torch.ones((1,), dtype=torch.bool),
        config=_ppo_config(),
    )

    assert terms.value_loss.item() == pytest.approx(1.0)
    assert terms.frontier_entropy.item() == pytest.approx(2.0)
    assert terms.theta_entropy.item() == pytest.approx(3.0)
    assert terms.total_loss.item() == pytest.approx(0.477)


def test_mixed_platform_ppo_ignores_hopper_theta_and_averages_active_entropy() -> None:
    """Inactive hopper theta cannot change ratio, KL, or the entropy bonus."""
    platform_context = torch.eye(3, dtype=torch.float32)
    candidate_mask = torch.ones((3, 64), dtype=torch.bool)
    output = PolicyOutput(
        frontier_logits=torch.zeros((3, 64), dtype=torch.float32),
        theta_mu=torch.zeros((3, 64), dtype=torch.float32),
        theta_kappa=torch.ones((3, 64), dtype=torch.float32),
        value=torch.zeros((3,), dtype=torch.float32),
    )
    selected_indices = torch.zeros((3,), dtype=torch.int64)
    baseline = recompute_action_log_probs(
        output,
        candidate_mask,
        selected_indices,
        torch.zeros((3,), dtype=torch.float32),
        platform_context,
    )
    changed_hopper = recompute_action_log_probs(
        output,
        candidate_mask,
        selected_indices,
        torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32),
        platform_context,
    )
    changed_wheel = recompute_action_log_probs(
        output,
        candidate_mask,
        selected_indices,
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        platform_context,
    )
    common = {
        "old_log_prob_total": baseline.log_prob_total,
        "normalized_advantage": torch.ones((3,), dtype=torch.float32),
        "new_value": torch.zeros((3,), dtype=torch.float32),
        "old_value": torch.zeros((3,), dtype=torch.float32),
        "returns": torch.zeros((3,), dtype=torch.float32),
        "frontier_entropy": torch.zeros((3,), dtype=torch.float32),
        "theta_entropy": torch.tensor([2.0, 4.0, 99.0], dtype=torch.float32),
        "theta_active": baseline.theta_active,
        "config": _ppo_config(),
    }

    terms = trainer_core.compute_ppo_loss_terms(
        new_log_prob_total=changed_hopper.log_prob_total,
        **common,
    )

    assert torch.equal(terms.ratio, torch.ones((3,), dtype=torch.float32))
    assert terms.approx_kl.item() == 0.0
    assert terms.theta_entropy.item() == pytest.approx(3.0)

    wheel_terms = trainer_core.compute_ppo_loss_terms(
        new_log_prob_total=changed_wheel.log_prob_total,
        **common,
    )
    assert wheel_terms.ratio[0].item() != pytest.approx(1.0)
    assert wheel_terms.approx_kl.item() > 0.0


def test_all_hopper_ppo_batch_has_exact_zero_theta_entropy() -> None:
    zeros = torch.zeros((2,), dtype=torch.float32)

    terms = trainer_core.compute_ppo_loss_terms(
        new_log_prob_total=zeros,
        old_log_prob_total=zeros,
        normalized_advantage=zeros,
        new_value=zeros,
        old_value=zeros,
        returns=zeros,
        frontier_entropy=zeros,
        theta_entropy=torch.tensor([7.0, 9.0], dtype=torch.float32),
        theta_active=torch.zeros((2,), dtype=torch.bool),
        config=_ppo_config(),
    )

    assert terms.theta_entropy.dtype == torch.float32
    assert terms.theta_entropy.item() == 0.0


def test_real_update_clips_gradients_and_reports_target_kl_early_stop() -> None:
    """Would fail if clipping or the KL stop were metrics-only placeholders."""
    torch.manual_seed(31)
    policy = CrossAttentionPolicy()
    rollout = _rollout_from_current_policy(policy, old_log_prob_offset=-2.0)
    rollout.advantages[:] = np.float32(1.0e4)
    rollout.returns[:] += np.float32(1.0e4)

    metrics = PPOTrainer(policy, config=_ppo_config()).update(rollout)

    assert metrics.optimizer_steps == 1
    assert metrics.epochs_completed == 1
    assert metrics.target_kl_early_stopped is True
    assert metrics.gradient_norm > 0.5
    assert metrics.clipped_gradient_norm <= 0.50001


_EXPECTED_PPO = {
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "policy_clip": 0.20,
    "value_clip": 0.20,
    "learning_rate": 3.0e-4,
    "optimizer": "AdamW",
    "weight_decay": 1.0e-4,
    "adam_epsilon": 1.0e-5,
    "epochs_per_update": 4,
    "target_kl": 0.03,
    "value_loss_coefficient": 0.5,
    "frontier_entropy_coef": 0.01,
    "theta_entropy_coef": 0.001,
    "max_grad_norm": 0.5,
    "rollout_horizon": 32,
    "dtype": "float32",
}

_PPO_VALUE_DRIFTS = (
    ("gamma", 0.994),
    ("gae_lambda", 0.94),
    ("policy_clip", 0.19),
    ("value_clip", 0.19),
    ("learning_rate", 2.0e-4),
    ("optimizer", "Adam"),
    ("weight_decay", 2.0e-4),
    ("adam_epsilon", 2.0e-5),
    ("epochs_per_update", 3),
    ("target_kl", 0.04),
    ("value_loss_coefficient", 0.4),
    ("frontier_entropy_coef", 0.02),
    ("theta_entropy_coef", 0.002),
    ("max_grad_norm", 0.6),
    ("rollout_horizon", 31),
    ("dtype", "float16"),
)


class _StringSubclass(str):
    pass


_PPO_SAME_VALUE_WRONG_TYPES = (
    ("gamma", np.float64(0.995)),
    ("gae_lambda", np.float64(0.95)),
    ("policy_clip", np.float64(0.20)),
    ("value_clip", np.float64(0.20)),
    ("learning_rate", np.float64(3.0e-4)),
    ("optimizer", _StringSubclass("AdamW")),
    ("weight_decay", np.float64(1.0e-4)),
    ("adam_epsilon", np.float64(1.0e-5)),
    ("epochs_per_update", np.int64(4)),
    ("target_kl", np.float64(0.03)),
    ("value_loss_coefficient", np.float64(0.5)),
    ("frontier_entropy_coef", np.float64(0.01)),
    ("theta_entropy_coef", np.float64(0.001)),
    ("max_grad_norm", np.float64(0.5)),
    ("rollout_horizon", np.int64(32)),
    ("dtype", _StringSubclass("float32")),
)


def _unchecked_ppo_config(**changes: object) -> PPOConfig:
    values = dict(_EXPECTED_PPO)
    values.update(changes)
    config = object.__new__(PPOConfig)
    for field, value in values.items():
        object.__setattr__(config, field, value)
    return config


@pytest.mark.parametrize("rollout_horizon", [16, 32, 64])
def test_rollout_horizon_accepts_only_calibratable_batch_boundaries(
    rollout_horizon: int,
) -> None:
    config = _unchecked_ppo_config(rollout_horizon=rollout_horizon)

    assert validate_ppo_config(config) is config


@pytest.mark.parametrize(
    "filename",
    ["rtx4080_super_smoke.yaml", "rtx4080_super_v3_joint.yaml"],
)
def test_both_yaml_files_freeze_the_exact_typed_ppo_baseline(filename: str) -> None:
    """Would fail if either supported run config omitted or drifted a PPO value."""
    config = load_training_config(
        pathlib.Path(__file__).resolve().parents[3] / "training/configs" / filename
    )

    assert isinstance(config.ppo, PPOConfig)
    assert asdict(config.ppo) == _EXPECTED_PPO


@pytest.mark.parametrize(
    ("field", "drift"),
    _PPO_VALUE_DRIFTS,
)
def test_loader_rejects_every_ppo_value_drift(field: str, drift: object) -> None:
    """Would fail if one PPO scalar, dtype, or horizon bypassed the freeze."""
    raw = yaml.safe_load(
        (
            pathlib.Path(__file__).resolve().parents[3]
            / "training/configs/rtx4080_super_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["ppo"][field] = drift

    with pytest.raises(TrainingConfigError, match="PPO|frozen"):
        resolve_training_config(raw)


@pytest.mark.parametrize(("field", "drift"), _PPO_VALUE_DRIFTS)
@pytest.mark.parametrize("construction", ("direct", "replace"))
def test_typed_ppo_config_rejects_every_value_drift(
    field: str,
    drift: object,
    construction: str,
) -> None:
    """Would fail if direct construction or replace bypassed one frozen value."""
    with pytest.raises(TrainingConfigError, match="PPO|frozen"):
        if construction == "direct":
            values = dict(_EXPECTED_PPO)
            values[field] = drift
            PPOConfig(**values)
        else:
            replace(_ppo_config(), **{field: drift})


@pytest.mark.parametrize(("field", "drift"), _PPO_SAME_VALUE_WRONG_TYPES)
def test_typed_ppo_config_rejects_same_value_with_wrong_exact_type(
    field: str,
    drift: object,
) -> None:
    """Would fail if equality let a non-builtin PPO field type pass."""
    values = dict(_EXPECTED_PPO)
    values[field] = drift

    with pytest.raises(TrainingConfigError, match="PPO|type"):
        PPOConfig(**values)


@pytest.mark.parametrize("consumer", ("trainer", "loss"))
def test_ppo_consumers_revalidate_unchecked_typed_config(consumer: str) -> None:
    """Would fail if a typed object bypassing construction reached PPO math."""
    config = _unchecked_ppo_config(epochs_per_update=3)

    with pytest.raises(trainer_core.PPOTrainingError, match="PPO|frozen"):
        if consumer == "trainer":
            PPOTrainer(CrossAttentionPolicy(), config=config)
        else:
            scalar = torch.zeros((1,), dtype=torch.float32)
            trainer_core.compute_ppo_loss_terms(
                new_log_prob_total=scalar,
                old_log_prob_total=scalar,
                normalized_advantage=scalar,
                new_value=scalar,
                old_value=scalar,
                returns=scalar,
                frontier_entropy=scalar,
                theta_entropy=scalar,
                theta_active=torch.ones((1,), dtype=torch.bool),
                config=config,
            )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_loader_rejects_missing_or_extra_ppo_fields(mutation: str) -> None:
    """Would fail if YAML could rely on hidden PPO defaults or ignored keys."""
    raw = yaml.safe_load(
        (
            pathlib.Path(__file__).resolve().parents[3]
            / "training/configs/rtx4080_super_smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    if mutation == "missing":
        del raw["ppo"]["target_kl"]
    else:
        raw["ppo"]["unapproved"] = 1

    with pytest.raises(TrainingConfigError, match="PPO"):
        resolve_training_config(raw)
