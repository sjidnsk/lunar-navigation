"""Frozen FP32 PPO optimizer over an ephemeral seven-input rollout."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ..config import PPOConfig, TrainingConfigError, validate_ppo_config
from ..policy.cross_attention import (
    CrossAttentionPolicy,
    recompute_action_log_probs,
)
from . import trainer_core
from .rollout import RolloutBatch


@dataclass(frozen=True, slots=True)
class StratumPPOUpdateMetrics:
    platform_id: int
    scale_bucket_id: int
    sample_count: int
    logical_weight: float
    total_loss: float
    policy_loss: float
    value_loss: float
    frontier_entropy: float
    theta_entropy: float
    approx_kl: float


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
    strata: tuple[StratumPPOUpdateMetrics, ...] = ()


def stratum_loss_weights(
    rollout: RolloutBatch,
) -> dict[tuple[int, int], float]:
    """Return sample-count-independent platform and scale loss weights."""
    if not isinstance(rollout, RolloutBatch):
        raise trainer_core.PPOTrainingError(
            "stratum loss weights require RolloutBatch"
        )
    return trainer_core.logical_stratum_weights(
        rollout.platform_ids,
        rollout.scale_bucket_ids,
    )


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
        try:
            validate_ppo_config(config)
        except TrainingConfigError as error:
            raise trainer_core.PPOTrainingError(
                "PPOTrainer requires the frozen typed PPO baseline"
            ) from error
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
        logical_weights = stratum_loss_weights(rollout)
        stratum_indices = {
            stratum: np.flatnonzero(
                (rollout.platform_ids == stratum[0])
                & (rollout.scale_bucket_ids == stratum[1])
            ).astype(np.int64, copy=False)
            for stratum in logical_weights
        }
        stratum_totals = {
            stratum: {name: 0.0 for name in totals}
            for stratum in logical_weights
        }

        for _epoch in range(self.config.epochs_per_update):
            self.optimizer.zero_grad(set_to_none=True)
            epoch_totals = {name: 0.0 for name in totals}
            epoch_stratum_totals = {
                stratum: {name: 0.0 for name in totals}
                for stratum in logical_weights
            }
            for stratum in sorted(logical_weights):
                indices = stratum_indices[stratum]
                logical_weight = logical_weights[stratum]
                for start in range(0, len(indices), micro_batch_size):
                    stop = min(start + micro_batch_size, len(indices))
                    micro_indices = indices[start:stop]
                    (
                        policy_batch,
                        selected_indices,
                        selected_thetas,
                        old_log_prob_total,
                        old_values,
                        advantages,
                        returns,
                    ) = rollout.select(micro_indices, device=self.device)
                    output = self.policy(policy_batch)
                    evaluation = recompute_action_log_probs(
                        output,
                        policy_batch.candidate_mask,
                        selected_indices,
                        selected_thetas,
                        policy_batch.platform_context,
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
                        theta_active=evaluation.theta_active,
                        config=self.config,
                    )
                    physical_weight = len(micro_indices) / len(indices)
                    weighted_loss = (
                        terms.total_loss * logical_weight * physical_weight
                    )
                    weighted_loss.backward()
                    for name in epoch_stratum_totals[stratum]:
                        epoch_stratum_totals[stratum][name] += (
                            float(getattr(terms, name).detach().cpu())
                            * physical_weight
                        )
                for name, value in epoch_stratum_totals[stratum].items():
                    epoch_totals[name] += value * logical_weight

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
            for stratum, values in epoch_stratum_totals.items():
                for name, value in values.items():
                    stratum_totals[stratum][name] += value
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
        stratum_metrics = tuple(
            StratumPPOUpdateMetrics(
                platform_id=stratum[0],
                scale_bucket_id=stratum[1],
                sample_count=len(stratum_indices[stratum]),
                logical_weight=logical_weights[stratum],
                **{
                    name: value / optimizer_steps
                    for name, value in stratum_totals[stratum].items()
                },
            )
            for stratum in sorted(logical_weights)
        )
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
            strata=stratum_metrics,
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
    "StratumPPOUpdateMetrics",
    "physical_microbatch_slices",
    "policy_state_sha256",
    "stratum_loss_weights",
]
