from __future__ import annotations

import copy
import hashlib
import pathlib
import random
import json
import sys

import numpy as np
import pytest
import torch


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))

from lunar_policy_training.budget import TrainingBudget  # noqa: E402
import lunar_policy_training.cli as cli_module  # noqa: E402
from lunar_policy_training.cli import SignalStopFlag, TrainingBoundaryLoop  # noqa: E402
from lunar_policy_training.checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    OBSERVATION_CONTRACT_VERSION,
    POLICY_WARM_START_PREFIXES,
    PolicyWarmStartEvidence,
    RunIdentity,
    build_training_checkpoint,
    config_sha256,
    load_checkpoint,
    load_checkpoint_for_resume,
    load_policy_warm_start,
    migrate_checkpoint_source_commit,
    restore_training_state,
    save_checkpoint_atomic,
    _body_from_checkpoint,
)
from lunar_policy_training.policy.cross_attention import (  # noqa: E402
    CrossAttentionPolicy,
)
from lunar_policy_training.ppo.checkpoint import (  # noqa: E402
    CheckpointError,
    _capture_rng_state,
    _semantic_sha256,
)
from lunar_policy_training.training_semantics import (  # noqa: E402
    training_semantics_sha256,
)
from lunar_model_contract import ObservationContractV2, ObservationContractV3  # noqa: E402


def _identity(
    *, run_kind: str = "development-smoke", **changes: str
) -> RunIdentity:
    fields = {
        "run_kind": run_kind,
        "data_sha256": "1" * 64,
        "split_sha256": "2" * 64,
        "generator_sha256": "3" * 64,
        "capability_sha256": "4" * 64,
        "reward_sha256": "5" * 64,
        "v3_sha256": "6" * 64,
        "training_semantics_sha256": training_semantics_sha256(),
    }
    fields.update(changes)
    return RunIdentity(**fields)


def test_complete_checkpoint_identity_uses_current_observation_contract() -> None:
    """Would fail if V3 tensors were mislabeled with the legacy V2 tag."""
    assert OBSERVATION_CONTRACT_VERSION == ObservationContractV3.version


def _checkpoint(
    consumed_gpu_seconds: float,
    *,
    worker_allocation: dict[str, int] | None = None,
    budget_extension_blocks: int = 0,
    run_kind: str = "development-smoke",
    environment_state: dict[str, object] | None = None,
):
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
        run_identity=_identity(run_kind=run_kind),
        source_commit="a0cc8dfd9210e1badcbe883e6178b1b27888bd93",
        consumed_gpu_seconds=consumed_gpu_seconds,
        budget_extension_blocks=budget_extension_blocks,
        total_gpu_budget_seconds=(
            86400 + budget_extension_blocks * 21600
        ),
        worker_allocation=(
            {"WHEELED": 6, "LEGGED": 6, "HOPPER": 6}
            if worker_allocation is None
            else worker_allocation
        ),
        micro_batch_size=2,
        latest_checkpoint_gpu_seconds=consumed_gpu_seconds,
        candidate_checkpoint_gpu_seconds=min(consumed_gpu_seconds, 3600.0),
        environment_state=environment_state,
    )


def test_resume_preserves_consumed_gpu_budget(tmp_path: pathlib.Path) -> None:
    """Would fail if resume reset the single cumulative 24-hour GPU budget."""
    checkpoint = _checkpoint(consumed_gpu_seconds=7200.0)
    save_checkpoint_atomic(tmp_path / "latest.pt", checkpoint)

    resumed = load_checkpoint(tmp_path / "latest.pt")
    budget = TrainingBudget.from_checkpoint(resumed)

    assert resumed.schema_version == CHECKPOINT_SCHEMA_VERSION
    assert resumed.schema_version == "lunar-ppo-checkpoint/v6"
    assert resumed.run_identity == _identity()
    assert resumed.contract_version == OBSERVATION_CONTRACT_VERSION
    assert resumed.consumed_gpu_seconds == 7200.0
    assert budget.remaining_gpu_seconds == 86400.0 - 7200.0


def test_source_migration_changes_only_commit_and_payload_hash() -> None:
    checkpoint = _checkpoint(consumed_gpu_seconds=7200.0)

    migrated = migrate_checkpoint_source_commit(
        checkpoint,
        expected_source_commit=checkpoint.source_commit,
        new_source_commit="b" * 40,
    )

    original_body = _body_from_checkpoint(checkpoint)
    migrated_body = _body_from_checkpoint(migrated)
    assert original_body.pop("source_commit") == checkpoint.source_commit
    assert migrated_body.pop("source_commit") == "b" * 40
    assert _semantic_sha256(original_body) == _semantic_sha256(migrated_body)
    assert migrated.payload_sha256 != checkpoint.payload_sha256
    assert migrated.global_step == checkpoint.global_step
    assert migrated.consumed_gpu_seconds == checkpoint.consumed_gpu_seconds


def test_source_migration_rejects_unexpected_checkpoint_commit() -> None:
    checkpoint = _checkpoint(consumed_gpu_seconds=7200.0)

    with pytest.raises(CheckpointError, match="source commit"):
        migrate_checkpoint_source_commit(
            checkpoint,
            expected_source_commit="c" * 40,
            new_source_commit="b" * 40,
        )


def _formal_worker_state(index: int) -> dict[str, object]:
    platform_index, lane = divmod(index, 8)
    platform = ("WHEELED", "LEGGED", "HOPPER")[platform_index]
    execution_state = "GROUND_HOLD" if platform == "HOPPER" else "DECISION_BOUNDARY"
    episode_id = f"scene/{platform.lower()}/{index}/episode-{index}"
    pose = {
        "x_m": 100.0 + index,
        "y_m": 200.0,
        "yaw_rad": 0.0,
        "elevation_m": 7.0,
        "frame_id": "map",
    }
    return {
        "scenario_schedule_id": "cache-sha/train/v3",
        "platform_type": platform,
        "worker_index": index,
        "platform_worker_index": lane,
        "platform_worker_count": 8,
        "episode_cursor": index,
        "scene_id": f"{index + 1:064x}",
        "scene_seed": f"{index + 101:064x}",
        "start_seed": f"{index + 201:064x}",
        "episode_seed": f"{index + 301:064x}",
        "coverability_mask_sha256": f"{index + 801:064x}",
        "start_cell": [64, 96],
        "current_pose": pose,
        "legged_body_z_m": 7.0,
        "execution_state": execution_state,
        "observation_revision": 1,
        "state_time_ns": 1_000_000_000,
        "reveal_history": [],
        "observation_identity": {
            "episode_id": episode_id,
            "mission_revision": 1,
            "map_snapshot_id": f"{index + 401:064x}",
            "robot_state_id": f"{index + 501:064x}",
            "state_time_ns": 1_000_000_000,
            "execution_state": execution_state,
            "candidate_set_id": f"{index + 601:064x}",
        },
        "policy_batch_sha256": f"{index + 701:064x}",
        "rejected_candidate_indices": [],
        "last_hop_available_delta_v_mps": 0.0,
    }


def _policy_parent_checkpoint(
    path: pathlib.Path,
    *,
    run_kind: str = "formal",
) -> tuple[object, dict[str, torch.Tensor]]:
    torch.manual_seed(91)
    model = CrossAttentionPolicy()
    with torch.no_grad():
        for name, value in model.state_dict().items():
            value.fill_(
                7.0
                if name.startswith("value_mlp.")
                else (sum(name.encode("utf-8")) % 31 + 1) / 100.0
            )
    parent_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _step: 1.0
    )
    checkpoint = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=906,
        curriculum_phase="joint",
        normalization={"reward_mean": 3.0, "reward_var": 4.0},
        frozen_config={"total_gpu_budget_seconds": 86400},
        run_identity=_identity(run_kind=run_kind),
        source_commit="a0cc8dfd9210e1badcbe883e6178b1b27888bd93",
        consumed_gpu_seconds=7200.0,
        budget_extension_blocks=0,
        total_gpu_budget_seconds=86400,
        worker_allocation={"WHEELED": 8, "LEGGED": 8, "HOPPER": 8},
        micro_batch_size=4,
        latest_checkpoint_gpu_seconds=7200.0,
        candidate_checkpoint_gpu_seconds=7200.0,
        environment_state=(
            {
                "schema_version": "lunar-formal-environment-state/v3",
                "scenario_schedule_id": "cache-sha/train/v3",
                "worker_episode_states": [
                    _formal_worker_state(index) for index in range(24)
                ],
            }
            if run_kind == "formal"
            else None
        ),
    )
    save_checkpoint_atomic(path, checkpoint)
    return checkpoint, parent_state


def test_policy_warm_start_transfers_only_approved_weights(
    tmp_path: pathlib.Path,
) -> None:
    parent_path = tmp_path / "parent.pt"
    parent, parent_state = _policy_parent_checkpoint(parent_path)
    torch.manual_seed(1234)
    target = CrossAttentionPolicy()
    optimizer = torch.optim.AdamW(target.parameters(), lr=9.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _step: 1.0
    )
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    rng_before = _semantic_sha256(_capture_rng_state())

    evidence = load_policy_warm_start(parent_path, target, value_head_seed=4080)

    target_state = target.state_dict()
    assert isinstance(evidence, PolicyWarmStartEvidence)
    assert evidence.parent_checkpoint_sha256 == hashlib.sha256(
        parent_path.read_bytes()
    ).hexdigest()
    assert evidence.parent_payload_sha256 == parent.payload_sha256
    assert evidence.parent_global_step == 906
    assert evidence.loaded_prefixes == POLICY_WARM_START_PREFIXES
    assert evidence.value_head_seed == 4080
    assert all(
        torch.equal(target_state[name], parent_state[name])
        for name in target_state
        if not name.startswith("value_mlp.")
    )
    assert all(
        not torch.equal(target_state[name], parent_state[name])
        for name in target_state
        if name.startswith("value_mlp.")
    )
    assert optimizer.state_dict() == optimizer_before
    assert scheduler.state_dict() == scheduler_before
    assert _semantic_sha256(_capture_rng_state()) == rng_before

    repeated = CrossAttentionPolicy()
    repeated_evidence = load_policy_warm_start(
        parent_path, repeated, value_head_seed=4080
    )
    assert (
        repeated_evidence.value_head_reinitialization_sha256
        == evidence.value_head_reinitialization_sha256
    )
    assert all(
        torch.equal(repeated.state_dict()[name], target_state[name])
        for name in target_state
        if name.startswith("value_mlp.")
    )


def test_policy_warm_start_keeps_obsolete_episode_state_inert(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "old-environment-parent.pt"
    _policy_parent_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    environment = payload["body"]["environment_state"]
    environment["schema_version"] = "lunar-formal-environment-state/v2"
    for worker in environment["worker_episode_states"]:
        worker.pop("coverability_mask_sha256")
    payload["body"]["run_identity"]["training_semantics_sha256"] = "8" * 64
    payload["body_sha256"] = _semantic_sha256(payload["body"])
    torch.save(payload, path)

    evidence = load_policy_warm_start(
        path, CrossAttentionPolicy(), value_head_seed=4080
    )

    assert evidence.parent_global_step == 906
    with pytest.raises(CheckpointError, match="environment state schema"):
        load_checkpoint(path)


@pytest.mark.parametrize("corruption", ("missing", "unexpected", "shape", "dtype"))
def test_policy_warm_start_rejects_architecture_drift_without_mutating_target(
    tmp_path: pathlib.Path,
    corruption: str,
) -> None:
    path = tmp_path / f"parent-{corruption}.pt"
    _policy_parent_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["body"]["model_state"]
    first_name = next(iter(state))
    if corruption == "missing":
        state.pop(first_name)
    elif corruption == "unexpected":
        state["unexpected.weight"] = torch.zeros(1)
    elif corruption == "shape":
        state[first_name] = state[first_name].reshape(-1)[:1]
    else:
        state[first_name] = state[first_name].to(torch.float64)
    payload["body_sha256"] = _semantic_sha256(payload["body"])
    torch.save(payload, path)
    target = CrossAttentionPolicy()
    before = {
        name: value.detach().clone() for name, value in target.state_dict().items()
    }

    with pytest.raises(CheckpointError, match="warm-start"):
        load_policy_warm_start(path, target, value_head_seed=4080)

    assert all(
        torch.equal(target.state_dict()[name], value)
        for name, value in before.items()
    )


def test_policy_warm_start_rejects_nonformal_parent(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "development-parent.pt"
    _policy_parent_checkpoint(path, run_kind="development-smoke")

    with pytest.raises(CheckpointError, match="formal"):
        load_policy_warm_start(path, CrossAttentionPolicy(), value_head_seed=4080)


def test_formal_checkpoint_roundtrips_exact_active_worker_states(
    tmp_path: pathlib.Path,
) -> None:
    environment_state = {
        "schema_version": "lunar-formal-environment-state/v3",
        "scenario_schedule_id": "cache-sha/train/v3",
        "worker_episode_states": [
            _formal_worker_state(index) for index in range(24)
        ],
    }
    checkpoint = _checkpoint(
        consumed_gpu_seconds=25.0,
        worker_allocation={"WHEELED": 8, "LEGGED": 8, "HOPPER": 8},
        run_kind="formal",
        environment_state=environment_state,
    )
    path = tmp_path / "formal.pt"
    save_checkpoint_atomic(path, checkpoint)

    resumed = load_checkpoint_for_resume(
        path,
        expected_contract_version=OBSERVATION_CONTRACT_VERSION,
        expected_config_hash=checkpoint.config_hash,
        expected_source_commit=checkpoint.source_commit,
        expected_run_identity=_identity(run_kind="formal"),
        expected_worker_allocation={"WHEELED": 8, "LEGGED": 8, "HOPPER": 8},
    )

    assert resumed.environment_state == environment_state


def test_formal_source_migration_preserves_original_and_records_evidence(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment_state = {
        "schema_version": "lunar-formal-environment-state/v3",
        "scenario_schedule_id": "cache-sha/train/v3",
        "worker_episode_states": [
            _formal_worker_state(index) for index in range(24)
        ],
    }
    checkpoint = _checkpoint(
        consumed_gpu_seconds=7200.0,
        worker_allocation={"WHEELED": 8, "LEGGED": 8, "HOPPER": 8},
        run_kind="formal",
        environment_state=environment_state,
    )
    root = tmp_path / "run"
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    latest = checkpoints / "latest.pt"
    save_checkpoint_atomic(latest, checkpoint)
    original_bytes = latest.read_bytes()
    (root / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "lunar-training-run/v1",
                "source_commit": checkpoint.source_commit,
                "config_hash": checkpoint.config_hash,
                "run_identity": checkpoint.run_identity.to_dict(),
                "global_step": checkpoint.global_step,
                "consumed_gpu_seconds": checkpoint.consumed_gpu_seconds,
                "platform_allocation": checkpoint.worker_allocation,
                "formal_environment": {
                    "sensor_performance_sha256": "7" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    new_commit = "b" * 40
    monkeypatch.setattr(cli_module, "_source_commit", lambda _root: new_commit)
    monkeypatch.setattr(
        cli_module, "_commit_is_ancestor", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        cli_module,
        "_commit_changed_paths",
        lambda *args, **kwargs: ("training/fix.py",),
    )
    monkeypatch.setattr(
        cli_module,
        "_validated_source_migration_sensor_reports",
        lambda **kwargs: ("7" * 64, "8" * 64),
    )
    previous_sensor = tmp_path / "previous-sensor.json"
    current_sensor = tmp_path / "current-sensor.json"
    previous_sensor.write_text("{}", encoding="utf-8")
    current_sensor.write_text("{}", encoding="utf-8")

    migrated_path = cli_module._prepare_source_migrated_resume_checkpoint(
        artifact_root=root,
        checkpoint_path=latest,
        repository_root=REPOSITORY_ROOT,
        expected_old_source_commit=checkpoint.source_commit,
        expected_global_step=12,
        previous_sensor_performance_report=previous_sensor,
        current_sensor_performance_report=current_sensor,
    )

    migrated = load_checkpoint(migrated_path)
    manifest = json.loads(
        (root / "run-manifest.json").read_text(encoding="utf-8")
    )
    backup = checkpoints / (
        f"source-checkpoint-step-12-{checkpoint.source_commit[:12]}.pt"
    )
    assert latest.read_bytes() == original_bytes
    assert backup.read_bytes() == original_bytes
    assert migrated.source_commit == new_commit
    assert migrated.global_step == checkpoint.global_step
    assert migrated.consumed_gpu_seconds == checkpoint.consumed_gpu_seconds
    assert manifest["source_commit"] == new_commit
    assert manifest["source_migrations"][-1]["from_source_commit"] == (
        checkpoint.source_commit
    )
    assert manifest["source_migrations"][-1]["to_source_commit"] == new_commit
    assert manifest["formal_environment"]["sensor_performance_sha256"] == (
        "8" * 64
    )


def test_formal_checkpoint_rejects_missing_episode_cursor_state() -> None:
    with pytest.raises(CheckpointError, match="environment state"):
        _checkpoint(
            consumed_gpu_seconds=0.0,
            worker_allocation={"WHEELED": 8, "LEGGED": 8, "HOPPER": 8},
            run_kind="formal",
        )


def test_checkpoint_roundtrips_extended_budget_without_resetting_consumed(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if extension state lived only in a mutable manifest."""
    checkpoint = _checkpoint(
        consumed_gpu_seconds=90000.0,
        budget_extension_blocks=2,
    )
    path = tmp_path / "extended.pt"

    save_checkpoint_atomic(path, checkpoint)
    resumed = load_checkpoint(path)
    budget = TrainingBudget.from_checkpoint(resumed)

    assert resumed.budget_extension_blocks == 2
    assert resumed.total_gpu_budget_seconds == 129600
    assert budget.consumed_gpu_seconds == 90000.0
    assert budget.remaining_gpu_seconds == 39600.0


def test_pre_extension_checkpoint_may_resume_against_extended_manifest_identity(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if an explicit extension made the latest older checkpoint unusable."""
    checkpoint = _checkpoint(consumed_gpu_seconds=7200.0)
    path = tmp_path / "pre-extension.pt"
    save_checkpoint_atomic(path, checkpoint)

    resumed = load_checkpoint_for_resume(
        path,
        expected_contract_version=OBSERVATION_CONTRACT_VERSION,
        expected_config_hash=checkpoint.config_hash,
        expected_source_commit=checkpoint.source_commit,
        expected_run_identity=checkpoint.run_identity,
        expected_budget_extension_blocks=2,
        expected_total_gpu_budget_seconds=129600,
    )

    assert resumed.budget_extension_blocks == 0
    assert resumed.total_gpu_budget_seconds == 86400


def test_checkpoint_cannot_claim_more_budget_than_manifest(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if stale manifest identity could silently lose an extension."""
    checkpoint = _checkpoint(
        consumed_gpu_seconds=1000.0,
        budget_extension_blocks=2,
    )
    path = tmp_path / "ahead-of-manifest.pt"
    save_checkpoint_atomic(path, checkpoint)

    with pytest.raises(CheckpointError, match="budget"):
        load_checkpoint_for_resume(
            path,
            expected_contract_version=OBSERVATION_CONTRACT_VERSION,
            expected_config_hash=checkpoint.config_hash,
            expected_source_commit=checkpoint.source_commit,
            expected_run_identity=checkpoint.run_identity,
            expected_budget_extension_blocks=1,
            expected_total_gpu_budget_seconds=108000,
        )


def test_checkpoint_roundtrips_single_platform_warmup_allocation(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if the formal warmup could not reach its first checkpoint."""
    checkpoint = _checkpoint(
        consumed_gpu_seconds=120.0,
        worker_allocation={"WHEELED": 24},
    )
    path = tmp_path / "latest.pt"

    save_checkpoint_atomic(path, checkpoint)

    assert load_checkpoint(path).worker_allocation == {"WHEELED": 24}


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
        "contract": OBSERVATION_CONTRACT_VERSION,
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
            expected_run_identity=checkpoint.run_identity,
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
        run_identity=_identity(),
        source_commit="a0cc8dfd9210e1badcbe883e6178b1b27888bd93",
        consumed_gpu_seconds=11.0,
        budget_extension_blocks=0,
        total_gpu_budget_seconds=86400,
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
            run_identity=_identity(),
            source_commit="a0cc8dfd9210e1badcbe883e6178b1b27888bd93",
            consumed_gpu_seconds=0.0,
            budget_extension_blocks=0,
            total_gpu_budget_seconds=86400,
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
            expected_contract_version=OBSERVATION_CONTRACT_VERSION,
            expected_config_hash=checkpoint.config_hash,
            expected_source_commit=checkpoint.source_commit,
            expected_run_identity=checkpoint.run_identity,
            expected_worker_allocation=allocation,
            expected_micro_batch_size=micro_batch,
        )


@pytest.mark.parametrize(
    "field",
    (
        "run_kind",
        "data_sha256",
        "split_sha256",
        "generator_sha256",
        "capability_sha256",
        "reward_sha256",
        "v3_sha256",
        "training_semantics_sha256",
    ),
)
def test_resume_rejects_each_v6_run_identity_field(
    tmp_path: pathlib.Path, field: str
) -> None:
    """Would fail if any frozen data/capability/reward/v3 identity could drift."""
    checkpoint = _checkpoint(consumed_gpu_seconds=10.0)
    path = tmp_path / "identity.pt"
    save_checkpoint_atomic(path, checkpoint)
    changed = (
        "formal"
        if field == "run_kind"
        else "f" * 64
    )
    expected = _identity(**{field: changed})

    with pytest.raises(CheckpointError, match=field.replace("_", " ")):
        load_checkpoint_for_resume(
            path,
            expected_contract_version=OBSERVATION_CONTRACT_VERSION,
            expected_config_hash=checkpoint.config_hash,
            expected_source_commit=checkpoint.source_commit,
            expected_run_identity=expected,
        )


def test_v2_checkpoint_requires_explicit_development_smoke_reader(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if legacy v2 were inferred or admitted into a formal run."""
    checkpoint = _checkpoint(consumed_gpu_seconds=10.0)
    body = _body_from_checkpoint(checkpoint)
    body.pop("environment_state")
    body.pop("run_identity")
    body["schema_version"] = "lunar-ppo-checkpoint/v2"
    body["contract_version"] = "ObservationContractV1"
    path = tmp_path / "legacy-v2.pt"
    torch.save(
        {"body": body, "body_sha256": _semantic_sha256(body)},
        path,
    )

    with pytest.raises(CheckpointError, match="explicit development-smoke"):
        load_checkpoint(path)
    with pytest.raises(CheckpointError, match="formal"):
        load_checkpoint(path, run_kind="formal")

    loaded = load_checkpoint(path, run_kind="development-smoke")
    assert loaded.schema_version == "lunar-ppo-checkpoint/v2"
    assert loaded.global_step == checkpoint.global_step


def test_v3_checkpoint_is_read_only_development_evidence(
    tmp_path: pathlib.Path,
) -> None:
    """Unknown historical semantics may be inspected but never resumed."""
    checkpoint = _checkpoint(consumed_gpu_seconds=10.0)
    body = _body_from_checkpoint(checkpoint)
    body.pop("environment_state")
    body["run_identity"].pop("training_semantics_sha256")
    body["schema_version"] = "lunar-ppo-checkpoint/v3"
    body["contract_version"] = "ObservationContractV1"
    path = tmp_path / "legacy-v3.pt"
    torch.save(
        {"body": body, "body_sha256": _semantic_sha256(body)},
        path,
    )

    with pytest.raises(CheckpointError, match="explicit development-smoke"):
        load_checkpoint(path)
    with pytest.raises(CheckpointError, match="formal"):
        load_checkpoint(path, run_kind="formal")

    loaded = load_checkpoint(path, run_kind="development-smoke")
    assert loaded.schema_version == "lunar-ppo-checkpoint/v3"
    with pytest.raises(CheckpointError, match="read-only"):
        load_checkpoint_for_resume(
            path,
            expected_contract_version=OBSERVATION_CONTRACT_VERSION,
            expected_config_hash=checkpoint.config_hash,
            expected_source_commit=checkpoint.source_commit,
            expected_run_identity=checkpoint.run_identity,
        )


def test_v5_checkpoint_is_explicit_read_only_v2_evidence(
    tmp_path: pathlib.Path,
) -> None:
    checkpoint = _checkpoint(consumed_gpu_seconds=10.0)
    body = _body_from_checkpoint(checkpoint)
    body["schema_version"] = "lunar-ppo-checkpoint/v5"
    body["contract_version"] = ObservationContractV2.version
    path = tmp_path / "legacy-v5.pt"
    torch.save(
        {"body": body, "body_sha256": _semantic_sha256(body)},
        path,
    )

    with pytest.raises(CheckpointError, match="explicit development-smoke"):
        load_checkpoint(path)
    with pytest.raises(CheckpointError, match="formal"):
        load_checkpoint(path, run_kind="formal")

    loaded = load_checkpoint(path, run_kind="development-smoke")
    assert loaded.schema_version == "lunar-ppo-checkpoint/v5"
    assert loaded.contract_version == ObservationContractV2.version
    with pytest.raises(CheckpointError, match="read-only"):
        load_checkpoint_for_resume(
            path,
            expected_contract_version=OBSERVATION_CONTRACT_VERSION,
            expected_config_hash=checkpoint.config_hash,
            expected_source_commit=checkpoint.source_commit,
            expected_run_identity=checkpoint.run_identity,
        )


def test_v4_checkpoint_is_read_only_development_evidence(
    tmp_path: pathlib.Path,
) -> None:
    checkpoint = _checkpoint(consumed_gpu_seconds=10.0)
    body = _body_from_checkpoint(checkpoint)
    body.pop("environment_state")
    body["schema_version"] = "lunar-ppo-checkpoint/v4"
    path = tmp_path / "legacy-v4.pt"
    torch.save(
        {"body": body, "body_sha256": _semantic_sha256(body)},
        path,
    )

    with pytest.raises(CheckpointError, match="explicit development-smoke"):
        load_checkpoint(path)
    with pytest.raises(CheckpointError, match="formal"):
        load_checkpoint(path, run_kind="formal")

    loaded = load_checkpoint(path, run_kind="development-smoke")
    assert loaded.schema_version == "lunar-ppo-checkpoint/v4"
    with pytest.raises(CheckpointError, match="read-only"):
        load_checkpoint_for_resume(
            path,
            expected_contract_version=OBSERVATION_CONTRACT_VERSION,
            expected_config_hash=checkpoint.config_hash,
            expected_source_commit=checkpoint.source_commit,
            expected_run_identity=checkpoint.run_identity,
        )
