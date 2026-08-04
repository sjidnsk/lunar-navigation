from __future__ import annotations

import pathlib
import random
import sys

import numpy as np
import pytest
import torch


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))

from lunar_policy_training.budget import TrainingBudget  # noqa: E402
from lunar_policy_training.cli import SignalStopFlag, TrainingBoundaryLoop  # noqa: E402
from lunar_policy_training.checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    build_training_checkpoint,
    config_sha256,
    load_checkpoint,
    load_checkpoint_for_resume,
    restore_training_state,
    save_checkpoint_atomic,
)
from lunar_policy_training.ppo.checkpoint import CheckpointError  # noqa: E402


def _checkpoint(consumed_gpu_seconds: float):
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    loss = model(torch.ones((1, 3))).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=12,
        curriculum_phase="joint",
        normalization={"reward_mean": 0.25, "reward_var": 1.5},
        frozen_config={"total_gpu_budget_seconds": 86400},
        source_commit="a0cc8dfd9210e1badcbe883e6178b1b27888bd93",
        consumed_gpu_seconds=consumed_gpu_seconds,
        worker_allocation={"WHEELED": 6, "LEGGED": 6, "HOPPER": 6},
        micro_batch_size=2,
        latest_checkpoint_gpu_seconds=consumed_gpu_seconds,
        candidate_checkpoint_gpu_seconds=min(consumed_gpu_seconds, 3600.0),
    )


def test_resume_preserves_consumed_gpu_budget(tmp_path: pathlib.Path) -> None:
    """Would fail if resume reset the single cumulative 24-hour GPU budget."""
    checkpoint = _checkpoint(consumed_gpu_seconds=7200.0)
    save_checkpoint_atomic(tmp_path / "latest.pt", checkpoint)

    resumed = load_checkpoint(tmp_path / "latest.pt")
    budget = TrainingBudget.from_checkpoint(resumed)

    assert resumed.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert resumed.contract_version == "ObservationContractV1"
    assert resumed.consumed_gpu_seconds == 7200.0
    assert budget.remaining_gpu_seconds == 86400.0 - 7200.0


def test_exhausted_checkpoint_resume_never_restarts_terminal_unit(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if a persisted 24-hour terminal state retried callbacks."""
    path = tmp_path / "latest.pt"
    save_checkpoint_atomic(path, _checkpoint(consumed_gpu_seconds=86400.0))
    resumed = load_checkpoint(path)
    budget = TrainingBudget.from_checkpoint(resumed)
    events: list[str] = []
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
        initial_global_step=resumed.global_step,
        initial_latest_checkpoint_gpu_seconds=(
            resumed.latest_checkpoint_gpu_seconds
        ),
        initial_candidate_checkpoint_gpu_seconds=(
            resumed.candidate_checkpoint_gpu_seconds
        ),
        clock=lambda: 5.0,
    )

    state = loop.run(
        collect_rollout=lambda: events.append("collect"),
        update_rollout=lambda rollout: events.append("update"),
        save_checkpoint=lambda kind, saved_state: events.append(
            f"save-{kind}-step-{saved_state.global_step}"
        ),
        max_updates=1,
    )

    assert budget.exhausted is True
    assert budget.consumed_gpu_seconds == 86400.0
    assert events == ["save-latest-step-12"]
    assert state.global_step == 12


@pytest.mark.parametrize(
    ("changed_expectation", "message"),
    [
        ({"contract": "ObservationContractV0"}, "contract"),
        ({"config_hash": "0" * 64}, "config hash"),
        ({"source_commit": "1" * 40}, "source commit"),
    ],
)
def test_resume_rejects_contract_config_or_source_drift(
    tmp_path: pathlib.Path,
    changed_expectation: dict[str, str],
    message: str,
) -> None:
    """Would fail if resume accepted a different graph, config, or source tree."""
    checkpoint = _checkpoint(consumed_gpu_seconds=5.0)
    path = tmp_path / "latest.pt"
    save_checkpoint_atomic(path, checkpoint)
    expected = {
        "contract": "ObservationContractV1",
        "config_hash": checkpoint.config_hash,
        "source_commit": checkpoint.source_commit,
    }
    expected.update(changed_expectation)

    with pytest.raises(CheckpointError, match=message):
        load_checkpoint_for_resume(
            path,
            expected_contract_version=expected["contract"],
            expected_config_hash=expected["config_hash"],
            expected_source_commit=expected["source_commit"],
        )


def test_restore_recovers_complete_train_state_and_rng(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if resume restored weights but lost optimizer, scheduler, or RNG."""
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    model(torch.ones((1, 3))).sum().backward()
    optimizer.step()
    scheduler.step()
    checkpoint = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=31,
        curriculum_phase="joint",
        normalization={"mean": torch.tensor([1.0])},
        frozen_config={"total_gpu_budget_seconds": 86400},
        source_commit="a0cc8dfd9210e1badcbe883e6178b1b27888bd93",
        consumed_gpu_seconds=11.0,
        worker_allocation={"WHEELED": 8, "LEGGED": 8, "HOPPER": 8},
        micro_batch_size=4,
        latest_checkpoint_gpu_seconds=10.0,
        candidate_checkpoint_gpu_seconds=0.0,
    )
    saved_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_rng = (random.random(), float(np.random.random()), float(torch.rand(())))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)
    optimizer.param_groups[0]["lr"] = 9.0
    scheduler.step()
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    live_normalization = {"mean": torch.tensor([-9.0]), "extra": 1.0}
    restore_training_state(
        checkpoint,
        model,
        optimizer,
        scheduler,
        normalization_state=live_normalization,
    )
    actual_rng = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in saved_parameters.items()
    )
    assert optimizer.param_groups[0]["lr"] == 1.0e-3
    assert scheduler.last_epoch == 1
    assert set(live_normalization) == {"mean"}
    assert torch.equal(live_normalization["mean"], torch.tensor([1.0]))
    assert actual_rng == expected_rng


def test_atomic_checkpoint_leaves_no_temporary_file_and_candidate_is_immutable(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if a candidate could be partially written or replaced in place."""
    checkpoint = _checkpoint(consumed_gpu_seconds=1.0)
    candidate = tmp_path / "candidate-step-12.pt"

    save_checkpoint_atomic(candidate, checkpoint, overwrite=False)

    assert list(tmp_path.iterdir()) == [candidate]
    with pytest.raises(CheckpointError, match="immutable"):
        save_checkpoint_atomic(candidate, checkpoint, overwrite=False)


def test_checkpoint_rejects_nonfinite_model_state() -> None:
    """Would fail if a poisoned parameter entered a recoverable checkpoint."""
    model = torch.nn.Linear(3, 2)
    with torch.no_grad():
        model.weight[0, 0] = torch.nan
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    with pytest.raises(CheckpointError, match="non-finite"):
        build_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=0,
            curriculum_phase="warmup",
            normalization={},
            frozen_config={"total_gpu_budget_seconds": 86400},
            source_commit="a0cc8dfd9210e1badcbe883e6178b1b27888bd93",
            consumed_gpu_seconds=0.0,
            worker_allocation={"WHEELED": 6, "LEGGED": 6, "HOPPER": 6},
            micro_batch_size=1,
            latest_checkpoint_gpu_seconds=0.0,
            candidate_checkpoint_gpu_seconds=0.0,
        )


def test_config_hash_is_order_independent_and_changes_with_values() -> None:
    """Would fail if equivalent YAML ordering blocked resume or value drift passed."""
    assert config_sha256({"a": 1, "b": [2, 3]}) == config_sha256(
        {"b": [2, 3], "a": 1}
    )
    assert config_sha256({"a": 1}) != config_sha256({"a": 2})


@pytest.mark.parametrize(
    ("allocation", "micro_batch", "message"),
    [
        ({"WHEELED": 8, "LEGGED": 8, "HOPPER": 8}, 2, "allocation"),
        ({"WHEELED": 6, "LEGGED": 6, "HOPPER": 6}, 4, "micro-batch"),
    ],
)
def test_resume_rejects_frozen_runtime_identity_drift(
    tmp_path: pathlib.Path,
    allocation: dict[str, int],
    micro_batch: int,
    message: str,
) -> None:
    checkpoint = _checkpoint(consumed_gpu_seconds=7200.0)
    path = tmp_path / "latest.pt"
    save_checkpoint_atomic(path, checkpoint)

    with pytest.raises(CheckpointError, match=message):
        load_checkpoint_for_resume(
            path,
            expected_contract_version="ObservationContractV1",
            expected_config_hash=checkpoint.config_hash,
            expected_source_commit=checkpoint.source_commit,
            expected_worker_allocation=allocation,
            expected_micro_batch_size=micro_batch,
        )
