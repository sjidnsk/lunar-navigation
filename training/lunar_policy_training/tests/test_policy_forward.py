from __future__ import annotations

import pathlib
import sys

import pytest
import torch


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "model_contract"))

from lunar_policy_training.policy.cross_attention import (  # noqa: E402
    CrossAttentionPolicy,
    PolicyBatch,
)


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
