"""Lightweight PPO optimizer over an ephemeral seven-input rollout."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

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
    approx_kl: float
    gradient_norm: float
    parameter_change_l2: float
    optimizer_steps: int


class PPOTrainer:
    """Perform one finite FP32 PPO optimizer step per update call."""

    loss_function = staticmethod(trainer_core.compute_ppo_loss_terms)

    def __init__(
        self,
        policy: CrossAttentionPolicy,
        *,
        learning_rate: float = 3.0e-4,
        device: torch.device | str = "cpu",
    ) -> None:
        if not isinstance(policy, CrossAttentionPolicy):
            raise trainer_core.PPOTrainingError(
                "PPOTrainer requires CrossAttentionPolicy"
            )
        if (
            not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or learning_rate <= 0.0
        ):
            raise trainer_core.PPOTrainingError(
                "learning_rate must be finite and positive"
            )
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise trainer_core.PPOTrainingError("requested CUDA device is unavailable")
        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=float(learning_rate)
        )

    def update(self, rollout: RolloutBatch) -> PPOUpdateMetrics:
        if not isinstance(rollout, RolloutBatch):
            raise trainer_core.PPOTrainingError("update requires RolloutBatch")
        selection = rollout.select(
            np.arange(len(rollout), dtype=np.int64), device=self.device
        )
        (
            policy_batch,
            selected_indices,
            selected_thetas,
            old_log_prob_total,
            old_values,
            advantages,
            returns,
        ) = selection
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.policy.named_parameters()
        }
        self.optimizer.zero_grad(set_to_none=True)
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
        )
        terms.total_loss.backward()
        gradient_norm = trainer_core._gradient_norm(self.policy.parameters())
        if not math.isfinite(gradient_norm):
            raise trainer_core.PPOTrainingError("PPO gradients must be finite")
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        if any(
            not bool(torch.isfinite(parameter).all())
            for parameter in self.policy.parameters()
        ):
            raise trainer_core.PPOTrainingError("PPO parameters must remain finite")
        parameter_change = trainer_core._parameter_change_l2(self.policy, before)
        if not math.isfinite(parameter_change) or parameter_change <= 0.0:
            raise trainer_core.PPOTrainingError(
                "PPO update did not change a policy parameter"
            )
        return PPOUpdateMetrics(
            total_loss=float(terms.total_loss.detach().cpu()),
            policy_loss=float(terms.policy_loss.detach().cpu()),
            value_loss=float(terms.value_loss.detach().cpu()),
            frontier_entropy=float(terms.frontier_entropy.detach().cpu()),
            approx_kl=float(terms.approx_kl.detach().cpu()),
            gradient_norm=gradient_norm,
            parameter_change_l2=parameter_change,
            optimizer_steps=1,
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
