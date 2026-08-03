"""Stage 5 observed-only baseline and deterministic PPO action selection."""

from __future__ import annotations
from typing import TYPE_CHECKING

import math
from typing import Final

import numpy as np
import torch

if TYPE_CHECKING:
    from typing import Any as EnvAction
from ..policy.cross_attention import PolicyForwardOutput
from ..policy.observation import PolicyObservation


BASELINE_METHODS: Final = (
    "random_valid_frontier",
    "nearest_frontier",
    "max_potential_gain_frontier",
    "gain_over_cost_frontier",
)
ALL_METHODS: Final = (*BASELINE_METHODS, "ppo_policy")

_DISTANCE_FEATURE: Final = 2
_POTENTIAL_GAIN_FEATURE: Final = 5
_RECOMMENDED_THETA_SIN_FEATURE: Final = 14
_RECOMMENDED_THETA_COS_FEATURE: Final = 15
_REACHABLE_COST_FEATURE: Final = 18
_MIN_DIRECTION_NORM: Final = 1.0e-12


class BaselineSelectionError(ValueError):
    """A baseline decision input violates the Stage 5 fail-closed contract."""


class NoCandidateAction(BaselineSelectionError):
    """No action may be fabricated when the shared valid set is empty."""


def reconstruct_recommended_theta(feature_row: np.ndarray) -> float:
    """Recover the exact candidate direction from its frozen sine/cosine pair."""

    row = np.asarray(feature_row)
    if row.ndim != 1 or row.size <= _RECOMMENDED_THETA_COS_FEATURE:
        raise BaselineSelectionError("recommended theta feature row is incomplete")
    sine = float(row[_RECOMMENDED_THETA_SIN_FEATURE])
    cosine = float(row[_RECOMMENDED_THETA_COS_FEATURE])
    if (
        not math.isfinite(sine)
        or not math.isfinite(cosine)
        or math.hypot(sine, cosine) <= _MIN_DIRECTION_NORM
    ):
        raise BaselineSelectionError("recommended theta is non-finite or near-zero")
    return math.atan2(sine, cosine)


def validate_selected_index(index: int, candidate_mask: np.ndarray) -> int:
    """Reject negative, masked, boolean, and out-of-range candidate indices."""

    mask = np.asarray(candidate_mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise BaselineSelectionError("candidate index mask must be one-dimensional boolean")
    if type(index) is not int or not 0 <= index < mask.size or not bool(mask[index]):
        raise BaselineSelectionError("selected candidate index is invalid")
    return index


def select_baseline_action(
    method: str,
    observation: PolicyObservation,
    rng: np.random.Generator,
) -> EnvAction:
    """Choose only among valid observed-safe reachable candidate rows."""

    if method not in BASELINE_METHODS:
        raise BaselineSelectionError(f"unknown baseline method: {method!r}")
    if not isinstance(observation, PolicyObservation):
        raise BaselineSelectionError("baseline selection requires PolicyObservation")
    if not isinstance(rng, np.random.Generator):
        raise BaselineSelectionError("baseline selection requires a numpy Generator")
    features = np.asarray(observation.frontier_features)
    mask = np.asarray(observation.candidate_mask)
    if (
        features.ndim != 2
        or features.shape[1] <= _REACHABLE_COST_FEATURE
        or mask.shape != (features.shape[0],)
        or mask.dtype != np.bool_
    ):
        raise BaselineSelectionError("baseline candidate feature or mask shape is invalid")
    if not np.isfinite(features).all():
        raise BaselineSelectionError("baseline candidate features must be finite")
    valid_indices = np.flatnonzero(mask)
    if valid_indices.size == 0:
        raise NoCandidateAction("no valid candidate; policy action must be skipped")

    if method == "random_valid_frontier":
        selected = int(rng.choice(valid_indices))
    elif method == "nearest_frontier":
        selected = min(
            (int(index) for index in valid_indices),
            key=lambda index: (float(features[index, _DISTANCE_FEATURE]), index),
        )
    elif method == "max_potential_gain_frontier":
        selected = min(
            (int(index) for index in valid_indices),
            key=lambda index: (-float(features[index, _POTENTIAL_GAIN_FEATURE]), index),
        )
    else:
        scores: dict[int, float] = {}
        for raw_index in valid_indices:
            index = int(raw_index)
            denominator = 1.0 + float(features[index, _REACHABLE_COST_FEATURE])
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise BaselineSelectionError("gain-over-cost denominator must be positive")
            scores[index] = float(features[index, _POTENTIAL_GAIN_FEATURE]) / denominator
        selected = min(scores, key=lambda index: (-scores[index], index))

    validate_selected_index(selected, mask)
    return EnvAction(
        candidate_index=selected,
        target_theta=reconstruct_recommended_theta(features[selected]),
    )


def select_ppo_action(
    output: PolicyForwardOutput,
    candidate_mask: torch.Tensor,
) -> EnvAction:
    """Apply deterministic argmax frontier and selected candidate mean theta."""

    if not isinstance(output, PolicyForwardOutput):
        raise BaselineSelectionError("PPO selection requires PolicyForwardOutput")
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
        raise BaselineSelectionError("PPO candidate tensors must match a batch of one")
    if not bool(candidate_mask.any()) :
        raise NoCandidateAction("no valid candidate; PPO action must be skipped")
    if not bool(torch.isfinite(output.frontier_logits).all()) or not bool(
        torch.isfinite(output.theta_mu).all()
    ):
        raise BaselineSelectionError("PPO logits and theta means must be finite")

    masked_logits = torch.where(
        candidate_mask,
        output.frontier_logits,
        torch.full_like(output.frontier_logits, float("-inf")),
    )
    selected = int(masked_logits.argmax(dim=1).item())
    mask_numpy = candidate_mask[0].detach().to(device="cpu").numpy()
    validate_selected_index(selected, mask_numpy)
    theta = float(output.theta_mu[0, selected].item())
    if not math.isfinite(theta):
        raise BaselineSelectionError("selected PPO theta mean must be finite")
    return EnvAction(candidate_index=selected, target_theta=theta)


__all__ = [
    "ALL_METHODS",
    "BASELINE_METHODS",
    "BaselineSelectionError",
    "NoCandidateAction",
    "reconstruct_recommended_theta",
    "select_baseline_action",
    "select_ppo_action",
    "validate_selected_index",
]
