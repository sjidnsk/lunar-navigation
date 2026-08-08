"""Shared platform-conditioned cross-attention PPO policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lunar_model_contract import ActionContractV2

from . import backbone_core
from .observation import PolicyBatch, validate_policy_batch


@dataclass(frozen=True)
class PolicyOutput:
    """The four exported shared policy outputs."""

    frontier_logits: torch.Tensor
    theta_mu: torch.Tensor
    theta_kappa: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class ActionEvaluation:
    log_prob_frontier: torch.Tensor
    log_prob_theta: torch.Tensor
    log_prob_total: torch.Tensor
    frontier_entropy: torch.Tensor
    theta_entropy: torch.Tensor


@dataclass(frozen=True)
class ActionSample:
    selected_frontier_index: torch.Tensor
    selected_theta: torch.Tensor
    log_prob_frontier: torch.Tensor
    log_prob_theta: torch.Tensor
    log_prob_total: torch.Tensor
    frontier_entropy: torch.Tensor
    theta_entropy: torch.Tensor
    value: torch.Tensor


class CrossAttentionPolicy(nn.Module):
    """One frozen backbone graph and one output head set for all platforms."""

    def __init__(self) -> None:
        super().__init__()
        self.global_encoder = backbone_core.MapEncoder(input_channels=7, output_grid=(32, 32))
        self.local_encoder = backbone_core.MapEncoder(input_channels=4, output_grid=(16, 16))
        self.pose_encoder = nn.Sequential(
            nn.Linear(6, backbone_core.TOKEN_DIM),
            nn.LayerNorm(backbone_core.TOKEN_DIM),
            nn.GELU(),
            nn.Linear(backbone_core.TOKEN_DIM, backbone_core.TOKEN_DIM),
            nn.LayerNorm(backbone_core.TOKEN_DIM),
        )
        self.platform_encoder = nn.Linear(3, backbone_core.TOKEN_DIM)
        self.frontier_encoder = nn.Linear(12, backbone_core.TOKEN_DIM)
        self.frontier_position_encoder = nn.Sequential(
            nn.Linear(4, backbone_core.TOKEN_DIM),
            nn.LayerNorm(backbone_core.TOKEN_DIM),
        )
        self.cross_attention_blocks = nn.ModuleList(
            backbone_core.CrossAttentionBlock()
            for _ in range(backbone_core.CROSS_ATTENTION_LAYERS)
        )
        self.action_output_mlp = nn.Sequential(
            nn.Linear(
                backbone_core.TOKEN_DIM * 2 + 12,
                backbone_core.ACTION_HIDDEN_DIM,
            ),
            nn.LayerNorm(backbone_core.ACTION_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(
                backbone_core.ACTION_HIDDEN_DIM,
                backbone_core.ACTION_HIDDEN_DIM,
            ),
            nn.LayerNorm(backbone_core.ACTION_HIDDEN_DIM),
            nn.GELU(),
        )
        self.frontier_logit_head = nn.Linear(backbone_core.ACTION_HIDDEN_DIM, 1)
        self.theta_sin_head = nn.Linear(backbone_core.ACTION_HIDDEN_DIM, 1)
        self.theta_cos_head = nn.Linear(backbone_core.ACTION_HIDDEN_DIM, 1)
        self.theta_kappa_head = nn.Linear(backbone_core.ACTION_HIDDEN_DIM, 1)
        self.value_mlp = nn.Sequential(
            nn.Linear(
                backbone_core.TOKEN_DIM * 3,
                backbone_core.ACTION_HIDDEN_DIM,
            ),
            nn.LayerNorm(backbone_core.ACTION_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(backbone_core.ACTION_HIDDEN_DIM, 1),
        )
        self._initialize_action_heads()

    def _initialize_action_heads(self) -> None:
        nn.init.uniform_(self.frontier_logit_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.frontier_logit_head.bias)
        for head in (self.theta_sin_head, self.theta_cos_head, self.theta_kappa_head):
            nn.init.zeros_(head.weight)
        nn.init.zeros_(self.theta_sin_head.bias)
        nn.init.ones_(self.theta_cos_head.bias)
        nn.init.constant_(self.theta_kappa_head.bias, backbone_core.INITIAL_KAPPA_RAW)

    def forward(self, batch: PolicyBatch) -> PolicyOutput:
        if not isinstance(batch, PolicyBatch):
            raise ValueError("forward requires PolicyBatch")
        validate_policy_batch(batch)
        if not bool(batch.candidate_mask.any(dim=1).all()):
            raise ValueError("all-false candidate rows must bypass policy")
        model_parameter = next(self.parameters())
        if model_parameter.dtype != torch.float32:
            raise ValueError("policy parameters must be float32")
        if model_parameter.device != batch.prior_channels.device:
            raise ValueError("policy tensors and parameters must share a device")

        global_tokens = self.global_encoder(
            torch.cat((batch.prior_channels, batch.coverage_summary), dim=1)
        )
        local_tokens = self.local_encoder(batch.local_crop)
        global_token = (
            self.pose_encoder(batch.pose_features)
            + self.platform_encoder(batch.platform_context)
        ).unsqueeze(1)
        context_tokens = torch.cat((global_tokens, local_tokens, global_token), dim=1)

        valid_mask = batch.candidate_mask.unsqueeze(-1)
        masked_features = torch.where(
            valid_mask,
            batch.frontier_features,
            torch.zeros_like(batch.frontier_features),
        )
        frontier_tokens = self.frontier_encoder(masked_features)
        frontier_tokens = frontier_tokens + self.frontier_position_encoder(
            masked_features[..., (0, 1, 3, 4)]
        )
        refined = frontier_tokens
        for block in self.cross_attention_blocks:
            refined = block(refined, context_tokens)

        action_hidden = self.action_output_mlp(
            torch.cat((frontier_tokens, refined, masked_features), dim=-1)
        )
        raw_logits = self.frontier_logit_head(action_hidden).squeeze(-1)
        frontier_logits = torch.where(
            batch.candidate_mask,
            raw_logits,
            torch.full_like(raw_logits, backbone_core.INVALID_LOGIT_VALUE),
        )
        raw_sin = self.theta_sin_head(action_hidden).squeeze(-1)
        raw_cos = self.theta_cos_head(action_hidden).squeeze(-1)
        theta_mu = backbone_core.normalize_theta(torch.atan2(raw_sin, raw_cos))
        raw_kappa = self.theta_kappa_head(action_hidden).squeeze(-1)
        theta_kappa = torch.clamp(
            torch.nn.functional.softplus(raw_kappa),
            min=ActionContractV2.theta_kappa_min,
            max=ActionContractV2.theta_kappa_max,
        )
        candidate_mean, candidate_max = backbone_core._masked_mean_max(
            refined, batch.candidate_mask
        )
        value = self.value_mlp(
            torch.cat(
                (candidate_mean, candidate_max, context_tokens.mean(dim=1)), dim=-1
            )
        ).squeeze(-1)
        return PolicyOutput(
            frontier_logits=frontier_logits,
            theta_mu=theta_mu,
            theta_kappa=theta_kappa,
            value=value,
        )


def recompute_action_log_probs(
    output: PolicyOutput,
    candidate_mask: torch.Tensor,
    selected_frontier_index: torch.Tensor,
    selected_theta: torch.Tensor,
) -> ActionEvaluation:
    """Recompute the masked categorical and conditional angle log-probability."""
    _validate_policy_output(output, candidate_mask)
    batch_size, candidate_count = candidate_mask.shape
    if (
        not isinstance(selected_frontier_index, torch.Tensor)
        or selected_frontier_index.dtype != torch.int64
        or selected_frontier_index.shape != (batch_size,)
        or selected_frontier_index.device != candidate_mask.device
    ):
        raise backbone_core.PolicyActionError(
            "selected frontier indices must be int64 [B] on the policy device"
        )
    if bool(
        ((selected_frontier_index < 0) | (selected_frontier_index >= candidate_count)).any()
    ):
        raise backbone_core.PolicyActionError("selected frontier index is out of range")
    selected_valid = backbone_core._gather_candidate(
        candidate_mask, selected_frontier_index
    )
    if not bool(selected_valid.all()):
        raise backbone_core.PolicyActionError("selected frontier index is masked")
    if (
        not isinstance(selected_theta, torch.Tensor)
        or selected_theta.dtype != torch.float32
        or selected_theta.shape != (batch_size,)
        or selected_theta.device != candidate_mask.device
    ):
        raise backbone_core.PolicyActionError(
            "selected theta must be float32 [B] on the policy device"
        )
    normalized_theta = backbone_core.normalize_theta(selected_theta)
    masked_logits = backbone_core._validated_masked_logits(
        output.frontier_logits, candidate_mask
    )
    frontier_distribution = torch.distributions.Categorical(logits=masked_logits)
    selected_mu = backbone_core._gather_candidate(
        output.theta_mu, selected_frontier_index
    )
    selected_kappa = backbone_core._gather_candidate(
        output.theta_kappa, selected_frontier_index
    )
    theta_distribution = torch.distributions.VonMises(selected_mu, selected_kappa)
    log_prob_frontier = frontier_distribution.log_prob(
        selected_frontier_index
    ).to(torch.float32)
    log_prob_theta = theta_distribution.log_prob(normalized_theta).to(torch.float32)
    evaluation = ActionEvaluation(
        log_prob_frontier=log_prob_frontier,
        log_prob_theta=log_prob_theta,
        log_prob_total=log_prob_frontier + log_prob_theta,
        frontier_entropy=frontier_distribution.entropy().to(torch.float32),
        theta_entropy=_von_mises_entropy(selected_kappa),
    )
    _require_finite_action_tensors(
        evaluation.log_prob_frontier,
        evaluation.log_prob_theta,
        evaluation.log_prob_total,
        evaluation.frontier_entropy,
        evaluation.theta_entropy,
    )
    return evaluation


def sample_action(
    output: PolicyOutput,
    candidate_mask: torch.Tensor,
    deterministic: bool,
) -> ActionSample:
    """Sample one valid candidate and its conditional angle."""
    _validate_policy_output(output, candidate_mask)
    if type(deterministic) is not bool:
        raise backbone_core.PolicyActionError("deterministic must be boolean")
    masked_logits = backbone_core._validated_masked_logits(
        output.frontier_logits, candidate_mask
    )
    frontier_distribution = torch.distributions.Categorical(logits=masked_logits)
    selected_index = (
        masked_logits.argmax(dim=-1)
        if deterministic
        else frontier_distribution.sample()
    )
    selected_mu = backbone_core._gather_candidate(output.theta_mu, selected_index)
    selected_kappa = backbone_core._gather_candidate(
        output.theta_kappa, selected_index
    )
    if deterministic:
        selected_theta = backbone_core.normalize_theta(selected_mu)
    else:
        selected_theta = backbone_core.normalize_theta(
            torch.distributions.VonMises(selected_mu, selected_kappa)
            .sample()
            .to(torch.float32)
        )
    evaluation = recompute_action_log_probs(
        output, candidate_mask, selected_index, selected_theta
    )
    sample = ActionSample(
        selected_frontier_index=selected_index,
        selected_theta=selected_theta,
        log_prob_frontier=evaluation.log_prob_frontier,
        log_prob_theta=evaluation.log_prob_theta,
        log_prob_total=evaluation.log_prob_total,
        frontier_entropy=evaluation.frontier_entropy,
        theta_entropy=evaluation.theta_entropy,
        value=output.value,
    )
    _require_finite_action_tensors(
        sample.selected_theta,
        sample.log_prob_frontier,
        sample.log_prob_theta,
        sample.log_prob_total,
        sample.frontier_entropy,
        sample.theta_entropy,
        sample.value,
    )
    return sample


def _von_mises_entropy(kappa: torch.Tensor) -> torch.Tensor:
    """Stable FP32 entropy for the selected conditional Von Mises action."""
    scaled_i0 = torch.special.i0e(kappa)
    scaled_i1 = torch.special.i1e(kappa)
    entropy = (
        torch.log(
            torch.tensor(
                2.0 * torch.pi,
                dtype=torch.float32,
                device=kappa.device,
            )
        )
        + torch.log(scaled_i0)
        + kappa
        - kappa * scaled_i1 / scaled_i0
    ).to(torch.float32)
    if not bool(torch.isfinite(entropy).all()):
        raise backbone_core.PolicyActionError("theta entropy must be finite")
    return entropy


def _validate_policy_output(
    output: PolicyOutput, candidate_mask: torch.Tensor
) -> None:
    if not isinstance(output, PolicyOutput):
        raise backbone_core.PolicyActionError("output must be PolicyOutput")
    backbone_core._validated_masked_logits(output.frontier_logits, candidate_mask)
    expected = output.frontier_logits.shape
    for name in ("theta_mu", "theta_kappa"):
        value = getattr(output, name)
        if (
            value.shape != expected
            or value.dtype != torch.float32
            or value.device != candidate_mask.device
            or not bool(torch.isfinite(value).all())
        ):
            raise backbone_core.PolicyActionError(
                f"{name} shape, dtype, device, or finiteness is invalid"
            )
    if (
        output.value.shape != (expected[0],)
        or output.value.dtype != torch.float32
        or output.value.device != candidate_mask.device
        or not bool(torch.isfinite(output.value).all())
    ):
        raise backbone_core.PolicyActionError("value shape, dtype, or device is invalid")
    if expected[1] != ActionContractV2.candidate_count:
        raise backbone_core.PolicyActionError("policy outputs must use 64 candidates")
    if bool((output.theta_kappa < ActionContractV2.theta_kappa_min).any()) or bool(
        (output.theta_kappa > ActionContractV2.theta_kappa_max).any()
    ):
        raise backbone_core.PolicyActionError("theta kappa is out of bounds")


def _require_finite_action_tensors(*tensors: torch.Tensor) -> None:
    if any(
        tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())
        for tensor in tensors
    ):
        raise backbone_core.PolicyActionError(
            "action values must be finite float32 tensors"
        )
