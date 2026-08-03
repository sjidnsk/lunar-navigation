from __future__ import annotations

from dataclasses import fields
import pathlib
import sys

import pytest
import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_policy_training.policy.cross_attention import (  # noqa: E402
    CrossAttentionPolicy,
    PolicyOutput,
    recompute_action_log_probs,
    sample_action,
)
from lunar_policy_training.policy.observation import (  # noqa: E402
    PolicyBatch,
)
from lunar_policy_training.policy import backbone_core  # noqa: E402


def test_policy_public_types_and_module_graph_use_the_frozen_backbone_core() -> None:
    """Would fail if a second batch/output type or placeholder network bypassed the core."""
    policy = CrossAttentionPolicy()

    assert [field.name for field in fields(PolicyBatch)] == [
        "prior_channels",
        "coverage_summary",
        "local_crop",
        "frontier_features",
        "pose_features",
        "candidate_mask",
        "platform_context",
    ]
    assert [field.name for field in fields(PolicyOutput)] == [
        "frontier_logits",
        "theta_mu",
        "theta_kappa",
        "value",
    ]
    assert isinstance(policy.global_encoder, backbone_core.MapEncoder)
    assert isinstance(policy.local_encoder, backbone_core.MapEncoder)
    assert all(
        isinstance(block, backbone_core.CrossAttentionBlock)
        for block in policy.cross_attention_blocks
    )
    assert policy.platform_encoder.in_features == 3
    assert policy.platform_encoder.out_features == backbone_core.TOKEN_DIM


def test_sampled_joint_log_prob_equals_immediate_recomputation() -> None:
    """Would fail if sampling and PPO recomputation used different masked distributions."""
    policy = CrossAttentionPolicy().eval()
    batch = _batch()

    with torch.no_grad():
        output = policy(batch)
        sample = sample_action(output, batch.candidate_mask, deterministic=True)
        recomputed = recompute_action_log_probs(
            output,
            batch.candidate_mask,
            sample.selected_frontier_index,
            sample.selected_theta,
        )

    assert torch.equal(
        batch.candidate_mask.gather(
            1, sample.selected_frontier_index.unsqueeze(1)
        ).squeeze(1),
        torch.ones((2,), dtype=torch.bool),
    )
    assert torch.equal(sample.log_prob_frontier, recomputed.log_prob_frontier)
    assert torch.equal(sample.log_prob_theta, recomputed.log_prob_theta)
    assert torch.equal(sample.log_prob_total, recomputed.log_prob_total)
    assert torch.equal(sample.frontier_entropy, recomputed.frontier_entropy)

def _batch(device: str = "cpu") -> PolicyBatch:
    return PolicyBatch(
        prior_channels=torch.zeros((2, 7, 32, 32), dtype=torch.float32, device=device),
        coverage_summary=torch.zeros((2, 8, 32, 32), dtype=torch.float32, device=device),
        local_crop=torch.zeros((2, 8, 32, 32), dtype=torch.float32, device=device),
        frontier_features=torch.zeros((2, 4, 22), dtype=torch.float32, device=device),
        pose_features=torch.zeros((2, 6), dtype=torch.float32, device=device),
        candidate_mask=torch.tensor(
            [[True, False, True, False], [False, True, True, False]],
            dtype=torch.bool,
            device=device,
        ),
        platform_context=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
        ),
    )


def test_policy_returns_finite_shared_outputs_and_never_selects_masked_candidate() -> None:
    """Would fail if the shared policy lost an output or masked candidates leaked."""
    policy = CrossAttentionPolicy().eval()
    batch = _batch()

    with torch.no_grad():
        output = policy(batch)

    assert output.frontier_logits.shape == (2, 4)
    assert output.theta_mu.shape == (2, 4)
    assert output.theta_kappa.shape == (2, 4)
    assert output.value.shape == (2,)
    assert all(
        torch.isfinite(tensor).all()
        for tensor in (output.frontier_logits, output.theta_mu, output.theta_kappa, output.value)
    )
    selected = output.frontier_logits.argmax(dim=1)
    assert bool(batch.candidate_mask.gather(1, selected.unsqueeze(1)).all())


def test_policy_rejects_non_one_hot_platform_context() -> None:
    """Would fail if unsupported mixed platforms entered the shared policy."""
    batch = _batch()
    batch.platform_context[0] = torch.tensor([1.0, 1.0, 0.0])

    with pytest.raises(ValueError, match="platform_context must be one-hot"):
        CrossAttentionPolicy()(batch)


@pytest.mark.cuda
def test_policy_cuda_forward_is_finite_and_masks_candidates() -> None:
    """Would fail if the CUDA policy path diverged from the CPU mask contract."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    policy = CrossAttentionPolicy().cuda().eval()

    with torch.no_grad():
        output = policy(_batch("cuda"))

    assert output.frontier_logits.dtype == torch.float32
    assert torch.isfinite(output.frontier_logits).all()
    selected = output.frontier_logits.argmax(dim=1)
    assert bool(_batch("cuda").candidate_mask.gather(1, selected.unsqueeze(1)).all())
