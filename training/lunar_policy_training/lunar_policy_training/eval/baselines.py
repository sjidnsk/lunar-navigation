"""Thin evaluation actions over the extracted baseline candidate math."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ..policy.cross_attention import PolicyOutput
from . import baseline_core


@dataclass(frozen=True, slots=True)
class EvaluationAction:
    candidate_index: int
    theta: float


def select_baseline_action(
    method: str,
    frontier_features: np.ndarray,
    candidate_mask: np.ndarray,
    rng: np.random.Generator,
) -> EvaluationAction:
    """Select one valid candidate using an extracted deterministic baseline rule."""
    if method not in baseline_core.BASELINE_METHODS:
        raise baseline_core.BaselineSelectionError(
            f"unknown baseline method: {method!r}"
        )
    if not isinstance(rng, np.random.Generator):
        raise baseline_core.BaselineSelectionError(
            "baseline selection requires a numpy Generator"
        )
    features = np.asarray(frontier_features)
    mask = np.asarray(candidate_mask)
    if (
        features.ndim != 2
        or features.shape[1] <= baseline_core._REACHABLE_COST_FEATURE
        or mask.shape != (features.shape[0],)
        or mask.dtype != np.bool_
    ):
        raise baseline_core.BaselineSelectionError(
            "baseline candidate feature or mask shape is invalid"
        )
    if not np.isfinite(features).all():
        raise baseline_core.BaselineSelectionError(
            "baseline candidate features must be finite"
        )
    valid_indices = np.flatnonzero(mask)
    if valid_indices.size == 0:
        raise baseline_core.NoCandidateAction("no valid candidate")

    if method == "random_valid_frontier":
        selected = int(rng.choice(valid_indices))
    elif method == "nearest_frontier":
        selected = min(
            (int(index) for index in valid_indices),
            key=lambda index: (
                float(features[index, baseline_core._DISTANCE_FEATURE]),
                index,
            ),
        )
    elif method == "max_potential_gain_frontier":
        selected = min(
            (int(index) for index in valid_indices),
            key=lambda index: (
                -float(features[index, baseline_core._POTENTIAL_GAIN_FEATURE]),
                index,
            ),
        )
    else:
        scores: dict[int, float] = {}
        for raw_index in valid_indices:
            index = int(raw_index)
            denominator = 1.0 + float(
                features[index, baseline_core._REACHABLE_COST_FEATURE]
            )
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise baseline_core.BaselineSelectionError(
                    "gain-over-cost denominator must be positive"
                )
            scores[index] = float(
                features[index, baseline_core._POTENTIAL_GAIN_FEATURE]
            ) / denominator
        selected = min(scores, key=lambda index: (-scores[index], index))

    baseline_core.validate_selected_index(selected, mask)
    return EvaluationAction(
        candidate_index=selected,
        theta=baseline_core.reconstruct_recommended_theta(features[selected]),
    )


def select_ppo_action(
    output: PolicyOutput, candidate_mask: torch.Tensor
) -> EvaluationAction:
    """Select the valid maximum-logit candidate from a four-field policy output."""
    if not isinstance(output, PolicyOutput):
        raise baseline_core.BaselineSelectionError(
            "PPO selection requires PolicyOutput"
        )
    if (
        not isinstance(candidate_mask, torch.Tensor)
        or candidate_mask.dtype != torch.bool
        or candidate_mask.ndim != 2
        or candidate_mask.shape[0] != 1
        or output.frontier_logits.shape != candidate_mask.shape
        or output.theta_mu.shape != candidate_mask.shape
        or output.frontier_logits.device != candidate_mask.device
        or output.theta_mu.device != candidate_mask.device
    ):
        raise baseline_core.BaselineSelectionError(
            "PPO candidate tensors must match a batch of one"
        )
    if not bool(candidate_mask.any()):
        raise baseline_core.NoCandidateAction("no valid candidate")
    if not bool(torch.isfinite(output.frontier_logits).all()) or not bool(
        torch.isfinite(output.theta_mu).all()
    ):
        raise baseline_core.BaselineSelectionError(
            "PPO logits and theta means must be finite"
        )
    masked_logits = torch.where(
        candidate_mask,
        output.frontier_logits,
        torch.full_like(output.frontier_logits, float("-inf")),
    )
    selected = int(masked_logits.argmax(dim=1).item())
    baseline_core.validate_selected_index(
        selected, candidate_mask[0].detach().cpu().numpy()
    )
    theta = float(output.theta_mu[0, selected].item())
    if not math.isfinite(theta):
        raise baseline_core.BaselineSelectionError(
            "selected PPO theta mean must be finite"
        )
    return EvaluationAction(candidate_index=selected, theta=theta)


ALL_METHODS = baseline_core.ALL_METHODS
BASELINE_METHODS = baseline_core.BASELINE_METHODS
BaselineSelectionError = baseline_core.BaselineSelectionError
NoCandidateAction = baseline_core.NoCandidateAction
reconstruct_recommended_theta = baseline_core.reconstruct_recommended_theta
validate_selected_index = baseline_core.validate_selected_index


__all__ = [
    "ALL_METHODS",
    "BASELINE_METHODS",
    "BaselineSelectionError",
    "EvaluationAction",
    "NoCandidateAction",
    "reconstruct_recommended_theta",
    "select_baseline_action",
    "select_ppo_action",
    "validate_selected_index",
]
