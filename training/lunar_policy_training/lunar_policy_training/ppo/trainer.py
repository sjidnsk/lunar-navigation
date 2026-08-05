"""Frozen FP32 PPO optimizer over an ephemeral seven-input rollout."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ..config import PPOConfig
from ..policy.cross_attention import (
    CrossAttentionPolicy,
    recompute_action_log_probs,
)
from . import trainer_core
from .rollout import RolloutBatch


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    total_loss: float
    policy_loss: float
    value_loss: float
    frontier_entropy: float
    theta_entropy: float
    approx_kl: float
    gradient_norm: float
    clipped_gradient_norm: float
    parameter_change_l2: float
    optimizer_steps: int
    epochs_completed: int
    target_kl_early_stopped: bool


class PPOTrainer:
    """Perform up to four finite FP32 AdamW epochs per update call."""

    loss_function = staticmethod(trainer_core.compute_ppo_loss_terms)

    def __init__(
        self,
        policy: CrossAttentionPolicy,
        *,
        config: PPOConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        if not isinstance(policy, CrossAttentionPolicy):
            raise trainer_core.PPOTrainingError(
                "PPOTrainer requires CrossAttentionPolicy"
            )
        if not isinstance(config, PPOConfig):
            raise trainer_core.PPOTrainingError("PPOTrainer requires typed PPOConfig")
        if config.dtype != "float32" or config.optimizer != "AdamW":
            raise trainer_core.PPOTrainingError(
                "PPOTrainer requires the frozen FP32 AdamW baseline"
            )
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise trainer_core.PPOTrainingError("requested CUDA device is unavailable")
        self.policy = policy.to(self.device, dtype=torch.float32)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon,
        )
        self.last_micro_batch_size: int | None = None

    def update(
        self,
        rollout: RolloutBatch,
        *,
        micro_batch_size: int | None = None,
    ) -> PPOUpdateMetrics:
        if not isinstance(rollout, RolloutBatch):
            raise trainer_core.PPOTrainingError("update requires RolloutBatch")
        if micro_batch_size is None:
            micro_batch_size = len(rollout)
        if type(micro_batch_size) is not int or micro_batch_size <= 0:
            raise trainer_core.PPOTrainingError(
                "micro-batch size must be a positive integer"
            )
        self.last_micro_batch_size = micro_batch_size
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.policy.named_parameters()
        }
        totals = {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "frontier_entropy": 0.0,
            "theta_entropy": 0.0,
            "approx_kl": 0.0,
        }
        max_gradient_norm = 0.0
        max_clipped_gradient_norm = 0.0
        optimizer_steps = 0
        early_stopped = False

        for _epoch in range(self.config.epochs_per_update):
            self.optimizer.zero_grad(set_to_none=True)
            epoch_totals = {name: 0.0 for name in totals}
            for start in range(0, len(rollout), micro_batch_size):
                stop = min(start + micro_batch_size, len(rollout))
                (
                    policy_batch,
                    selected_indices,
                    selected_thetas,
                    old_log_prob_total,
                    old_values,
                    advantages,
                    returns,
                ) = rollout.select(
                    np.arange(start, stop, dtype=np.int64), device=self.device
                )
                output = self.policy(policy_batch)
                evaluation = recompute_action_log_probs(
                    output,
                    policy_batch.candidate_mask,
                    selected_indices,
                    selected_thetas,
                )
                terms = self.loss_function(
                    new_log_prob_total=evaluation.log_prob_total,
                    old_log_prob_total=old_log_prob_total,
                    normalized_advantage=advantages,
                    new_value=output.value,
                    old_value=old_values,
                    returns=returns,
                    frontier_entropy=evaluation.frontier_entropy,
                    theta_entropy=evaluation.theta_entropy,
                    config=self.config,
                )
                weight = (stop - start) / len(rollout)
                (terms.total_loss * weight).backward()
                for name in epoch_totals:
                    epoch_totals[name] += (
                        float(getattr(terms, name).detach().cpu()) * weight
                    )

            gradient_norm = trainer_core._gradient_norm(self.policy.parameters())
            if not math.isfinite(gradient_norm):
                raise trainer_core.PPOTrainingError("PPO gradients must be finite")
            max_gradient_norm = max(max_gradient_norm, gradient_norm)
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_norm=self.config.max_grad_norm
            )
            clipped_gradient_norm = trainer_core._gradient_norm(
                self.policy.parameters()
            )
            if not math.isfinite(clipped_gradient_norm):
                raise trainer_core.PPOTrainingError(
                    "clipped PPO gradients must be finite"
                )
            max_clipped_gradient_norm = max(
                max_clipped_gradient_norm, clipped_gradient_norm
            )
            self.optimizer.step()
            optimizer_steps += 1
            if any(
                not bool(torch.isfinite(parameter).all())
                for parameter in self.policy.parameters()
            ):
                raise trainer_core.PPOTrainingError(
                    "PPO parameters must remain finite"
                )
            for name, value in epoch_totals.items():
                totals[name] += value
            if epoch_totals["approx_kl"] > self.config.target_kl:
                early_stopped = True
                break

        parameter_change = trainer_core._parameter_change_l2(self.policy, before)
        if not math.isfinite(parameter_change) or parameter_change <= 0.0:
            raise trainer_core.PPOTrainingError(
                "PPO update did not change a policy parameter"
            )
        if optimizer_steps <= 0:
            raise trainer_core.PPOTrainingError("PPO update completed no optimizer epoch")
        averaged = {name: value / optimizer_steps for name, value in totals.items()}
        return PPOUpdateMetrics(
            total_loss=averaged["total_loss"],
            policy_loss=averaged["policy_loss"],
            value_loss=averaged["value_loss"],
            frontier_entropy=averaged["frontier_entropy"],
            theta_entropy=averaged["theta_entropy"],
            approx_kl=averaged["approx_kl"],
            gradient_norm=max_gradient_norm,
            clipped_gradient_norm=max_clipped_gradient_norm,
            parameter_change_l2=parameter_change,
            optimizer_steps=optimizer_steps,
            epochs_completed=optimizer_steps,
            target_kl_early_stopped=early_stopped,
        )


PPOTrainingError = trainer_core.PPOTrainingError
PPOLossTerms = trainer_core.PPOLossTerms
physical_microbatch_slices = trainer_core.physical_microbatch_slices
policy_state_sha256 = trainer_core.policy_state_sha256


__all__ = [
    "PPOLossTerms",
    "PPOTrainer",
    "PPOTrainingError",
    "PPOUpdateMetrics",
    "physical_microbatch_slices",
    "policy_state_sha256",
]
