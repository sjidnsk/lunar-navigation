from __future__ import annotations

import pathlib
import runpy


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
TOOL_PATH = REPOSITORY_ROOT / "training/tools/qualify_task_cache_training_entry.py"


def _target_path(target: str) -> pathlib.Path:
    return REPOSITORY_ROOT / target.split("::", maxsplit=1)[0]


def test_focused_qualification_targets_exist_in_current_source() -> None:
    """The entry authority must not qualify current code through deleted tests."""
    namespace = runpy.run_path(str(TOOL_PATH))
    targets = tuple(namespace["_FOCUSED_TARGETS"])
    smokes = tuple(target for _, target in namespace["_SHORT_SMOKES"])

    assert all(_target_path(target).is_file() for target in (*targets, *smokes))
    assert (
        "training/lunar_policy_training/tests/"
        "test_reachable_constrained_ground_candidates.py"
    ) in targets
    assert all("test_prepared_hopper_action_aggregates" not in target for target in smokes)
