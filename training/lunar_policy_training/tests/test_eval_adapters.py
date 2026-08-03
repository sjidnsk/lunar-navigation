from __future__ import annotations

from dataclasses import fields
import pathlib
import sys

import numpy as np
import pytest
import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_policy_training.eval import baseline_core, metrics_core  # noqa: E402
from lunar_policy_training.eval.baselines import (  # noqa: E402
    BaselineSelectionError,
    EvaluationAction,
    select_baseline_action,
    select_ppo_action,
)
from lunar_policy_training.eval.metrics import (  # noqa: E402
    bootstrap_episode_indices,
    bootstrap_indices_sha256,
)
from lunar_policy_training.policy.cross_attention import PolicyOutput  # noqa: E402


def _candidate_features() -> np.ndarray:
    features = np.zeros((3, 22), dtype=np.float32)
    features[0, 2] = 5.0
    features[0, 5] = 2.0
    features[0, 14:16] = (0.0, 1.0)
    features[0, 18] = 2.0
    features[1, 2] = 1.0
    features[1, 5] = 1.0
    features[1, 14:16] = (1.0, 0.0)
    features[1, 18] = 0.25
    features[2, 2] = 0.1
    features[2, 5] = 100.0
    features[2, 14:16] = (-1.0, 0.0)
    features[2, 18] = 0.01
    return features


def test_public_baselines_use_core_candidate_math_and_return_only_index_theta() -> None:
    """Would fail if EnvAction/old observations returned or masked candidates were considered."""
    features = _candidate_features()
    mask = np.asarray([True, True, False], dtype=np.bool_)

    nearest = select_baseline_action(
        "nearest_frontier", features, mask, np.random.default_rng(9)
    )
    max_gain = select_baseline_action(
        "max_potential_gain_frontier", features, mask, np.random.default_rng(9)
    )
    gain_over_cost = select_baseline_action(
        "gain_over_cost_frontier", features, mask, np.random.default_rng(9)
    )

    assert [field.name for field in fields(EvaluationAction)] == [
        "candidate_index",
        "theta",
    ]
    assert nearest == EvaluationAction(candidate_index=1, theta=np.pi / 2.0)
    assert max_gain == EvaluationAction(candidate_index=0, theta=0.0)
    assert gain_over_cost == EvaluationAction(candidate_index=1, theta=np.pi / 2.0)
    assert BaselineSelectionError is baseline_core.BaselineSelectionError


def test_ppo_eval_selection_masks_logits_and_uses_four_field_policy_output() -> None:
    """Would fail if eval used the old output type or selected a masked high logit."""
    output = PolicyOutput(
        frontier_logits=torch.tensor([[1.0, 99.0, 2.0]], dtype=torch.float32),
        theta_mu=torch.tensor([[0.1, 0.2, -0.3]], dtype=torch.float32),
        theta_kappa=torch.ones((1, 3), dtype=torch.float32),
        value=torch.zeros((1,), dtype=torch.float32),
    )

    action = select_ppo_action(
        output, torch.tensor([[True, False, True]], dtype=torch.bool)
    )

    assert action.candidate_index == 2
    assert action.theta == pytest.approx(-0.3)


def test_public_metrics_use_core_fixed_seed_indices_and_stable_hash() -> None:
    """Would fail if public eval changed bootstrap RNG, dtype, byte order, or hashing."""
    assert bootstrap_episode_indices is metrics_core.bootstrap_episode_indices
    assert bootstrap_indices_sha256 is metrics_core.bootstrap_indices_sha256

    indices = bootstrap_episode_indices(3, 4, 7)

    assert indices.tolist() == [
        [2, 1, 2],
        [2, 1, 2],
        [2, 0, 0],
        [0, 0, 2],
    ]
    assert bootstrap_indices_sha256(indices) == (
        "ab7d62e2de838d2e984f30ba72f6069c3626edc6d24be3ca632832f88da9d6a3"
    )
