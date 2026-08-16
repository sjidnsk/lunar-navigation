from __future__ import annotations

from types import MappingProxyType

from lunar_policy_training.training_metrics import _thaw_journal_worker_state


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
