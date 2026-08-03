"""Shared platform-conditioned cross-attention PPO policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .observation import PolicyBatch, validate_policy_batch


@dataclass(frozen=True)
class PolicyOutput:
    """The four exported shared policy outputs."""

    frontier_logits: torch.Tensor
    theta_mu: torch.Tensor
    theta_kappa: torch.Tensor
    value: torch.Tensor


class CrossAttentionPolicy(nn.Module):
    """One network and one output head set for all three platforms."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.global_encoder = nn.Linear(15, d_model)
        self.local_encoder = nn.Linear(8, d_model)
        self.pose_encoder = nn.Linear(6, d_model)
        self.platform_encoder = nn.Linear(3, d_model)
        self.frontier_encoder = nn.Linear(22, d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.frontier_logit_head = nn.Linear(d_model, 1)
        self.theta_mu_head = nn.Linear(d_model, 1)
        self.theta_kappa_head = nn.Linear(d_model, 1)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, batch: PolicyBatch) -> PolicyOutput:
        if not isinstance(batch, PolicyBatch):
            raise ValueError("forward requires PolicyBatch")
        validate_policy_batch(batch)
        global_features = torch.cat((batch.prior_channels, batch.coverage_summary), dim=1).mean(dim=(2, 3))
        local_features = batch.local_crop.mean(dim=(2, 3))
        global_embedding = self.pose_encoder(batch.pose_features) + self.platform_encoder(batch.platform_context)
        context = torch.stack((self.global_encoder(global_features), self.local_encoder(local_features), global_embedding), dim=1)
        frontier = self.frontier_encoder(batch.frontier_features)
        refined, _ = self.cross_attention(frontier, context, context, need_weights=False)
        logits_raw = self.frontier_logit_head(refined).squeeze(-1)
        frontier_logits = torch.where(batch.candidate_mask, logits_raw, torch.full_like(logits_raw, -1.0e9))
        theta_mu = torch.tanh(self.theta_mu_head(refined).squeeze(-1)) * torch.pi
        theta_kappa = torch.nn.functional.softplus(self.theta_kappa_head(refined).squeeze(-1)) + 1.0e-3
        value = self.value_head(global_embedding).squeeze(-1)
        return PolicyOutput(frontier_logits=frontier_logits, theta_mu=theta_mu, theta_kappa=theta_kappa, value=value)
