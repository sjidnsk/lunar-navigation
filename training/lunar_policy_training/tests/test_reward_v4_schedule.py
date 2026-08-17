from __future__ import annotations

from pathlib import Path

import pytest

from lunar_policy_training.evaluation.reward_v4_schedule import (
    RewardV4EvaluationTier,
    materialize_reward_v4_evaluation_mode,
    select_reward_v4_evaluation_tier,
)


_PAYLOAD_SHA = "a" * 64


@pytest.mark.parametrize(
    ("previous_gpu_s", "candidate_gpu_s", "expected"),
    (
        (0.0, 10_800.0, RewardV4EvaluationTier.SENTINEL),
        (10_800.0, 21_600.0, RewardV4EvaluationTier.SENTINEL),
        (43_199.0, 43_200.0, RewardV4EvaluationTier.FULL),
        (43_200.0, 54_000.0, RewardV4EvaluationTier.SENTINEL),
        (86_399.0, 86_400.0, RewardV4EvaluationTier.FULL),
    ),
)
def test_reward_v4_evaluation_tier_only_promotes_on_full_interval_crossing(
    previous_gpu_s: float,
    candidate_gpu_s: float,
    expected: RewardV4EvaluationTier,
) -> None:
    assert (
        select_reward_v4_evaluation_tier(
            previous_candidate_gpu_seconds=previous_gpu_s,
            candidate_gpu_seconds=candidate_gpu_s,
        )
        is expected
    )


def test_evaluation_mode_is_durable_and_bound_to_candidate(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "candidate-33" / "evaluation-mode.json").resolve()

    first = materialize_reward_v4_evaluation_mode(
        path,
        checkpoint_payload_sha256=_PAYLOAD_SHA,
        previous_candidate_gpu_seconds=0.0,
        candidate_gpu_seconds=10_879.0,
    )
    second = materialize_reward_v4_evaluation_mode(
        path,
        checkpoint_payload_sha256=_PAYLOAD_SHA,
        previous_candidate_gpu_seconds=0.0,
        candidate_gpu_seconds=10_879.0,
    )

    assert first == second
    assert first.tier is RewardV4EvaluationTier.SENTINEL
    assert path.is_file()

    with pytest.raises(ValueError, match="checkpoint identity differs"):
        materialize_reward_v4_evaluation_mode(
            path,
            checkpoint_payload_sha256="b" * 64,
            previous_candidate_gpu_seconds=0.0,
            candidate_gpu_seconds=10_879.0,
        )


def test_evaluation_mode_rejects_corrupt_artifact(tmp_path: Path) -> None:
    path = (tmp_path / "evaluation-mode.json").resolve()
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="structure differs"):
        materialize_reward_v4_evaluation_mode(
            path,
            checkpoint_payload_sha256=_PAYLOAD_SHA,
            previous_candidate_gpu_seconds=0.0,
            candidate_gpu_seconds=10_800.0,
        )
