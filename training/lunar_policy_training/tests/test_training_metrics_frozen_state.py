from __future__ import annotations

from types import MappingProxyType

from lunar_policy_training.training_metrics import (
    RewardV4RuntimeDiagnostics,
    RuntimeWorkerIdentity,
    _thaw_journal_worker_state,
)


def test_runtime_diagnostics_thaw_journal_worker_state_sequences() -> None:
    frozen = MappingProxyType(
        {
            "task_coarse_bounds_half_open": (114, 140, 114, 140),
            "start_cell": (3, 7),
            "reveal_history": (
                MappingProxyType(
                    {
                        "path_samples": (
                            MappingProxyType({"x_m": 1.0, "y_m": 2.0}),
                        ),
                    }
                ),
            ),
        }
    )

    thawed = _thaw_journal_worker_state(frozen)

    assert thawed == {
        "task_coarse_bounds_half_open": [114, 140, 114, 140],
        "start_cell": [3, 7],
        "reveal_history": [{"path_samples": [{"x_m": 1.0, "y_m": 2.0}]}],
    }
    assert frozen["task_coarse_bounds_half_open"] == (114, 140, 114, 140)


def test_runtime_diagnostics_round_trip_internal_timing() -> None:
    identity = RuntimeWorkerIdentity(
        worker_index=0,
        platform_type="WHEELED",
        episode_ids=("episode-0",),
        task_ids=("a" * 64,),
        physical_snapshot_ids=("b" * 64,),
    )
    diagnostics = RewardV4RuntimeDiagnostics(
        update_id=1,
        worker_identities=(identity,),
        invalid_start_task_count=0,
        coarse_fine_reachability_mismatch_count=0,
        planner_suppressed_candidate_count=0,
        zero_motion_reselection_count=0,
        macro_action_elapsed_s_by_worker=(3.0,),
        slowest_worker={"worker_index": 0, "elapsed_s": 3.0},
        terminal_reason_counts={},
        invalid_task_reason_counts={},
        hard_error_reason_counts={},
        planner_call_count_by_worker=(4,),
        planner_elapsed_s_by_worker=(1.5,),
        policy_inference_elapsed_s=0.25,
        candidate_refresh_elapsed_s=0.5,
        global_search_elapsed_s=0.75,
    )

    restored = RewardV4RuntimeDiagnostics.from_dict(diagnostics.to_dict())

    assert restored.planner_call_count_by_worker == (4,)
    assert restored.planner_elapsed_s_by_worker == (1.5,)
    assert restored.policy_inference_elapsed_s == 0.25
    assert restored.candidate_refresh_elapsed_s == 0.5
    assert restored.global_search_elapsed_s == 0.75
