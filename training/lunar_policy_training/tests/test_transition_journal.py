from __future__ import annotations

from pathlib import Path

import pytest
import torch

import lunar_policy_training.recovery.transition_journal as journal_module
from lunar_policy_training.recovery.transition_journal import (
    JournalValidationError,
    MacroTransitionPayload,
    TransitionJournal,
)
from lunar_policy_training.reward import (
    RewardInputsV4,
    compute_reward_components,
)
from lunar_policy_training.reward_contract import (
    RewardStage,
    RewardTerminalClass,
    RewardWeightsV4,
    TaskScaleBucket,
)


def _full_pre_worker_state() -> dict[str, object]:
    return {
        "worker_index": 0,
        "platform_type": "WHEELED",
        "observation_identity": {"episode_id": "episode-000"},
        "platform_task_key_sha256": "a" * 64,
        "physical_snapshot_id": "b" * 64,
        "coverable_detail_cell_count": 100,
        "observed_coverable_detail_cell_count": 10,
        "observed_coverable_mask_sha256": "c" * 64,
        "candidate_decision_snapshot": {
            "candidate_refresh_elapsed_s": 0.125,
            "global_search_elapsed_s": 0.25,
            "fine_pose_candidate_count": 7,
            "globally_reachable_candidate_count": 5,
        },
        "large_runtime_only_tensor": torch.ones(2_000_000, dtype=torch.float32),
    }


def _compact_pre_worker_audit(state: dict[str, object]) -> dict[str, object]:
    builder = getattr(journal_module, "build_pre_worker_audit", None)
    assert callable(builder), "journal must expose compact pre-boundary projection"
    return dict(builder(state))


def _payload(
    *,
    update_id: int,
    slot_index: int,
    pre_worker_audit: dict[str, object],
) -> MacroTransitionPayload:
    inputs = RewardInputsV4(
        platform_type="WHEELED",
        coverage_before=0.10,
        coverage_after=0.20,
        priority_before=0.0,
        priority_after=0.0,
        path_before_m=1.0,
        path_after_m=2.0,
        task_scale_m=100.0,
        terminal_class=RewardTerminalClass.CONTINUE,
        success_first_crossing=False,
    )
    weights = RewardWeightsV4()
    return MacroTransitionPayload(
        update_id=update_id,
        worker_index=0,
        slot_index=slot_index,
        policy_version=3,
        platform_type="WHEELED",
        scale_bucket=TaskScaleBucket.M100_200,
        episode_id="episode-000",
        episode_transition_index=slot_index,
        transition_id=f"transition-{update_id}-{slot_index}",
        observation={"policy": torch.zeros(4, dtype=torch.float32)},
        action={"candidate_index": torch.tensor(0, dtype=torch.int64)},
        old_log_prob=torch.tensor(0.0, dtype=torch.float32),
        old_value=torch.tensor(0.0, dtype=torch.float32),
        reward_stage=RewardStage.R1,
        reward_weights=weights,
        reward_components=compute_reward_components(inputs, RewardStage.R1, weights),
        reward_inputs=inputs,
        done=False,
        terminal_class=RewardTerminalClass.CONTINUE,
        pre_worker_audit=pre_worker_audit,
        post_worker_state={"resume_tensor": torch.arange(16, dtype=torch.int64)},
    )


def test_persisted_transition_keeps_only_compact_pre_boundary_audit(
    tmp_path: Path,
) -> None:
    """Recovery keeps the complete post-state without duplicating the pre-state."""
    full_pre_state = _full_pre_worker_state()
    audit = _compact_pre_worker_audit(full_pre_state)
    journal = TransitionJournal(
        tmp_path / "journal",
        run_id="reward-v4-test",
        worker_count=1,
        slots_per_worker=1,
    )

    committed = journal.commit(
        _payload(update_id=1, slot_index=0, pre_worker_audit=audit)
    )

    stored = torch.load(committed.payload_path, map_location="cpu", weights_only=True)
    assert "pre_worker_state" not in stored
    assert stored["pre_worker_audit"] == audit
    assert "large_runtime_only_tensor" not in stored["pre_worker_audit"]
    assert committed.payload_path.stat().st_size < 100_000

    recovered = journal.recover_worker_boundaries(update_id=1)
    assert torch.equal(
        recovered[0].post_worker_state["resume_tensor"],
        torch.arange(16, dtype=torch.int64),
    )


def test_prune_applied_updates_keeps_newest_applied_and_open_boundary(
    tmp_path: Path,
) -> None:
    """Only manifest-safe applied history is prunable; an open update remains."""
    audit = _compact_pre_worker_audit(_full_pre_worker_state())
    journal = TransitionJournal(
        tmp_path / "journal",
        run_id="reward-v4-test",
        worker_count=1,
        slots_per_worker=1,
    )
    for update_id in (1, 2):
        journal.commit(
            _payload(
                update_id=update_id,
                slot_index=0,
                pre_worker_audit=audit,
            )
        )
        journal.seal_update(update_id=update_id, expected_slots={(0, 0)})
        journal.mark_update_applied(
            update_id=update_id,
            checkpoint_payload_sha256="d" * 64,
            metrics_record_sha256="e" * 64,
        )
    journal.commit(_payload(update_id=3, slot_index=0, pre_worker_audit=audit))

    prune = getattr(journal, "prune_applied_updates", None)
    assert callable(prune), "journal must prune superseded applied updates"
    assert prune(keep_latest=1) == (1,)

    assert not (journal.recovery_root / "update-00000001").exists()
    assert (journal.recovery_root / "update-00000002").is_dir()
    assert (journal.recovery_root / "update-00000003").is_dir()
    assert 0 in journal.recover_worker_boundaries(update_id=3)


def test_compact_pre_audit_must_match_transition_identity(tmp_path: Path) -> None:
    """A compact audit cannot be paired with another worker's transition."""
    audit = _compact_pre_worker_audit(_full_pre_worker_state())
    audit["worker_index"] = 1
    journal = TransitionJournal(
        tmp_path / "journal",
        run_id="reward-v4-test",
        worker_count=2,
        slots_per_worker=1,
    )

    with pytest.raises(JournalValidationError, match="pre worker audit identity"):
        journal.commit(_payload(update_id=1, slot_index=0, pre_worker_audit=audit))
