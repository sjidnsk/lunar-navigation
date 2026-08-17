from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from lunar_policy_training.evaluation.report import (
    build_reward_v4_evaluation_report,
)
from lunar_policy_training.reward_contract import (
    DEFAULT_REWARD_CONFIG,
    TaskScaleBucket,
)
from lunar_policy_training.reward_curriculum import (
    PlatformType,
    initial_reward_curriculum_state,
)
from lunar_policy_training.reward_evaluation import (
    RewardEpisodeMetrics,
    build_reward_v4_evaluation_manifest,
)
from lunar_policy_training.reward_update_boundary import (
    RewardUpdateBoundaryError,
    advance_reward_sentinel_update_boundary,
)


_PAYLOAD_SHA = "e" * 64


def _report(*, hard_error_count: int = 0):
    manifest = build_reward_v4_evaluation_manifest(
        platforms=(PlatformType.WHEELED,),
        scale_buckets=tuple(TaskScaleBucket),
        evaluation_seeds=(4081,),
    )
    episodes = tuple(
        RewardEpisodeMetrics(
            platform=task.platform,
            scale_bucket=task.scale_bucket,
            evaluation_seed=task.evaluation_seed,
            success_at_0_95=False,
            final_coverage=0.1,
            priority_coverage_auc_over_macro_actions=None,
            steps_to_success=None,
            normalized_executed_path_to_success=None,
            hard_error_count=hard_error_count,
            macro_action_count=1,
            priority_denominator_present=False,
        )
        for task in manifest.tasks
    )
    return build_reward_v4_evaluation_report(
        manifest=manifest,
        episodes=episodes,
        checkpoint_payload_sha256=_PAYLOAD_SHA,
        enabled_r2_platforms=(),
        bootstrap_seed=7,
        bootstrap_resample_count=1,
    )


def test_sentinel_pass_advances_only_the_update_boundary(
    tmp_path: Path,
) -> None:
    state = initial_reward_curriculum_state(DEFAULT_REWARD_CONFIG)
    best = {
        "schema_version": "lunar-best-checkpoint-state/v1",
        "accepted": [],
    }

    decision = advance_reward_sentinel_update_boundary(
        state,
        update_id=33,
        evaluation_report=_report(),
        candidate_checkpoint_path=(tmp_path / "candidate.pt").resolve(),
        candidate_checkpoint_gpu_seconds=10_879.0,
        best_checkpoint_state=best,
        config=DEFAULT_REWARD_CONFIG,
    )

    assert decision.curriculum_state == replace(state, current_update_id=33)
    assert dict(decision.best_checkpoint_state) == best
    assert decision.checkpoint_decision is None
    assert decision.rollback_checkpoint_path is None
    assert tuple(event["event"] for event in decision.curriculum_events) == (
        "SENTINEL_EVALUATION_PASSED",
    )
    assert decision.candidate_checkpoint_gpu_seconds == 10_879.0


def test_sentinel_hard_error_fails_closed(tmp_path: Path) -> None:
    state = initial_reward_curriculum_state(DEFAULT_REWARD_CONFIG)

    with pytest.raises(
        RewardUpdateBoundaryError,
        match="sentinel evaluation contains a hard error",
    ):
        advance_reward_sentinel_update_boundary(
            state,
            update_id=33,
            evaluation_report=_report(hard_error_count=1),
            candidate_checkpoint_path=(tmp_path / "candidate.pt").resolve(),
            candidate_checkpoint_gpu_seconds=10_879.0,
            best_checkpoint_state={
                "schema_version": "lunar-best-checkpoint-state/v1",
                "accepted": [],
            },
            config=DEFAULT_REWARD_CONFIG,
        )
