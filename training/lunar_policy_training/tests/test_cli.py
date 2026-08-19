from __future__ import annotations

import ast
import inspect
import pathlib
import os
import signal
import json
import subprocess
import sys
import contextlib
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import yaml

import lunar_policy_training.cli as cli_module


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))


def test_preflight_only_source_delta_rejects_runtime_changes() -> None:
    assert cli_module._preflight_only_source_delta_paths(
        {
            "training/lunar_policy_training/lunar_policy_training/cli.py",
            "training/lunar_policy_training/lunar_policy_training/formal_preflight.py",
            "training/lunar_policy_training/tests/test_cli.py",
            "training/lunar_policy_training/tests/test_qualification_entry_targets.py",
            "training/tools/qualify_task_cache_training_entry.py",
        }
    )
    assert not cli_module._preflight_only_source_delta_paths(
        {
            "training/lunar_policy_training/lunar_policy_training/reward.py"
        }
    )


def test_train_path_propagates_the_reused_task_cache_source() -> None:
    """Fresh policy warm starts must retain the prefetch key namespace."""
    tree = ast.parse(inspect.getsource(cli_module.main))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            "_formal_environment_from_calibrated_root",
            "_formal_evaluation_batches_from_calibrated_root",
        }
    ]
    names = {node.func.id for node in calls}
    assert names == {
        "_formal_environment_from_calibrated_root",
        "_formal_evaluation_batches_from_calibrated_root",
    }
    for name in names:
        assert any(
            any(
                keyword.arg == "task_key_source_commit"
                and ast.unparse(keyword.value) == "task_runtime.source_commit"
                for keyword in call.keywords
            )
            for call in calls
            if call.func.id == name
        )


from lunar_policy_training.budget import TrainingBudget  # noqa: E402
from lunar_policy_training.cli import (  # noqa: E402
    ArtifactRootError,
    PreflightError,
    SignalStopFlag,
    TrainingBoundaryLoop,
    _ParallelPoolVectorEnv,
    _checkpoint_target,
    _formal_sensor_performance_preflight,
    _formal_run_identity,
    _freeze_task4_manifest,
    _freeze_formal_environment_manifest,
    _freeze_policy_warm_start_manifest,
    _load_calibrated_run_state,
    _latest_candidate_checkpoint,
    _operator_skips_pending_reward_v4_evaluation,
    _run_curriculum_training,
    _update_run_manifest,
    build_parser,
    validate_artifact_root,
)
from lunar_policy_training.environment.candidate_builder import (  # noqa: E402
    CandidateDiagnostics,
)
from lunar_policy_training.environment.macro_step import (  # noqa: E402
    TerminalAudit,
    TerminalReason,
)
from lunar_policy_training.environment.parallel_pool import (  # noqa: E402
    ParallelEnvPool,
    ParallelRolloutStep,
)
from lunar_policy_training.capability_freeze import FrozenCapabilityBundle  # noqa: E402
from lunar_policy_training.config import (
    TrainingConfigError,
    load_training_config,
    resolve_training_config,
    with_rollout_horizon,
)
from lunar_policy_training.checkpoint import PolicyWarmStartEvidence, RunIdentity
from lunar_policy_training.curriculum import CurriculumSchedule
from lunar_policy_training.polar_data.formal_cache import FormalCacheIdentity
from lunar_policy_training.polar_data.task_cache import TaskCommonKey
from lunar_policy_training.evaluation.release_gate import (
    GateResult,
    evaluate_release_gate,
    load_gate_rules,
)
from lunar_policy_training.evaluation.report import (
    EvaluationReport,
    FormalEvaluationBatch,
    MethodEvaluation,
    PlatformMetrics,
    REQUIRED_METHODS,
)
from lunar_policy_training.evaluation.reward_v4_schedule import (
    RewardV4EvaluationMode,
    RewardV4EvaluationTier,
)
from lunar_policy_training.proxy_scenario import proxy_observation
from lunar_policy_training.reward import reward_weights_sha256
from lunar_policy_training.reward_contract import RewardStage, TaskScaleBucket
from lunar_policy_training.reward_curriculum import PlatformType
from lunar_policy_training.reward_evaluation import (
    build_reward_v4_evaluation_manifest,
)


def test_operator_skip_pending_reward_v4_evaluation_is_update_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUNAR_SKIP_PENDING_REWARD_V4_EVALUATION_UPDATE", "320")

    assert _operator_skips_pending_reward_v4_evaluation(320) is True
    assert _operator_skips_pending_reward_v4_evaluation(321) is False


def test_reward_v4_evaluation_can_be_disabled_for_the_training_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUNAR_DISABLE_REWARD_V4_EVALUATION", "1")

    assert cli_module._reward_v4_evaluation_disabled() is True


def test_reward_v4_candidate_reuse_requires_current_source_and_journal() -> None:
    candidate = SimpleNamespace(
        source_commit="a" * 40,
        update_recovery_state=SimpleNamespace(journal_sha256="b" * 64),
    )

    assert cli_module._reward_v4_candidate_matches_current_journal(
        candidate, source_commit="a" * 40, journal_sha256="b" * 64
    ) is True
    assert cli_module._reward_v4_candidate_matches_current_journal(
        candidate, source_commit="c" * 40, journal_sha256="b" * 64
    ) is False
    assert cli_module._reward_v4_candidate_matches_current_journal(
        candidate, source_commit="a" * 40, journal_sha256="c" * 64
    ) is False


def test_reward_v4_evaluation_resolves_only_active_r2_platforms() -> None:
    curriculum = SimpleNamespace(
        platforms={
            PlatformType.WHEELED: SimpleNamespace(stage=RewardStage.R1),
            PlatformType.LEGGED: SimpleNamespace(stage=RewardStage.R2),
            PlatformType.HOPPER: SimpleNamespace(stage=RewardStage.R2),
        }
    )

    active, enabled_r2 = cli_module._reward_v4_evaluation_platforms(
        curriculum,
        {"WHEELED": 12, "LEGGED": 12},
    )

    assert active == (PlatformType.WHEELED, PlatformType.LEGGED)
    assert enabled_r2 == (PlatformType.LEGGED,)


def test_checkpoint_environment_state_reuses_terminal_journal_states() -> None:
    """A sealed journal already owns the exact worker recovery boundary."""
    def committed(worker: int, slot: int, label: str) -> SimpleNamespace:
        return SimpleNamespace(
            worker_index=worker,
            slot_index=slot,
            payload=SimpleNamespace(
                post_worker_state=MappingProxyType(
                    {
                        "worker": worker,
                        "label": label,
                        "reveal_history": (label, "boundary"),
                    }
                )
            ),
        )

    loaded = SimpleNamespace(
        state="SEALED",
        committed={
            (0, 0): committed(0, 0, "old-0"),
            (1, 0): committed(1, 0, "old-1"),
            (0, 1): committed(0, 1, "final-0"),
            (1, 1): committed(1, 1, "final-1"),
        },
    )

    environment_state = cli_module._checkpoint_environment_state_from_sealed_journal(
        loaded=loaded,
        scenario_schedule_id="fixed-schedule",
        worker_count=2,
        actions_per_worker=2,
    )

    assert environment_state == {
        "schema_version": cli_module.FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION,
        "scenario_schedule_id": "fixed-schedule",
        "worker_episode_states": [
            {
                "worker": 0,
                "label": "final-0",
                "reveal_history": ["final-0", "boundary"],
            },
            {
                "worker": 1,
                "label": "final-1",
                "reveal_history": ["final-1", "boundary"],
            },
        ],
    }


def test_checkpoint_environment_state_rejects_missing_terminal_journal_slot() -> None:
    loaded = SimpleNamespace(
        state="SEALED",
        committed={
            (0, 0): SimpleNamespace(
                worker_index=0,
                slot_index=0,
                payload=SimpleNamespace(post_worker_state={"worker": 0}),
            )
        },
    )

    with pytest.raises(cli_module.UpdateCommitError, match="terminal worker state"):
        cli_module._checkpoint_environment_state_from_sealed_journal(
            loaded=loaded,
            scenario_schedule_id="fixed-schedule",
            worker_count=1,
            actions_per_worker=2,
        )


def test_resume_task_keys_keep_first_source_namespace_after_same_step_repairs() -> None:
    checkpoint = SimpleNamespace(global_step=63, source_commit="e" * 40)
    manifest = {
        "source_migrations": [
            {
                "global_step": 63,
                "from_source_commit": "e" * 40,
                "to_source_commit": "c" * 40,
            },
            {
                "global_step": 63,
                "from_source_commit": "c" * 40,
                "to_source_commit": "d" * 40,
            },
        ]
    }

    assert cli_module._resume_task_key_source_commit(manifest, checkpoint) == (
        "e" * 40
    )


def test_resume_task_keys_read_persisted_task_namespace_after_later_repairs(
    tmp_path: pathlib.Path,
) -> None:
    persisted_source = "a" * 40
    common_key = TaskCommonKey(
        scene_id="1" * 64,
        scenario_identity_sha256="2" * 64,
        source_identity_sha256="3" * 64,
        coarse_bounds_half_open=(0, 25, 0, 25),
        detail_bounds_half_open=(0, 500, 0, 500),
        task_span_cells=25,
        scale_bucket="100_200",
        geometry_sha256="4" * 64,
        halo_contract_sha256="5" * 64,
        capability_bundle_sha256="6" * 64,
        generator_sha256="7" * 64,
        source_commit=persisted_source,
    )
    artifact = tmp_path / "tasks" / "v1" / "common" / common_key.sha256()
    artifact.mkdir(parents=True)
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "key_sha256": common_key.sha256(),
                "key": common_key.to_dict(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    checkpoint = SimpleNamespace(global_step=71, source_commit="d" * 40)
    manifest = {
        "source_migrations": [
            {
                "global_step": 71,
                "from_source_commit": "b" * 40,
                "to_source_commit": "c" * 40,
            },
            {
                "global_step": 71,
                "from_source_commit": "c" * 40,
                "to_source_commit": "d" * 40,
            },
        ]
    }

    assert cli_module._resume_task_key_source_commit(
        manifest,
        checkpoint,
        task_cache_root=tmp_path,
        task_common_key_sha256s=(common_key.sha256(),),
    ) == persisted_source


@pytest.mark.parametrize(
    ("tier", "expected_seeds", "expected_cap", "expected_report"),
    (
        (
            RewardV4EvaluationTier.SENTINEL,
            (4081,),
            1,
            "sentinel-report.json",
        ),
        (
            RewardV4EvaluationTier.FULL,
            (4081, 4082, 4083),
            None,
            "report.json",
        ),
    ),
)
def test_reward_v4_evaluation_request_separates_sentinel_from_full(
    tmp_path: pathlib.Path,
    tier: RewardV4EvaluationTier,
    expected_seeds: tuple[int, ...],
    expected_cap: int | None,
    expected_report: str,
) -> None:
    seeds = (4081, 4082, 4083)
    manifest = build_reward_v4_evaluation_manifest(
        platforms=tuple(PlatformType),
        scale_buckets=tuple(TaskScaleBucket),
        evaluation_seeds=seeds,
    )
    batch = FormalEvaluationBatch(
        split="validation",
        factory=SimpleNamespace(scenario_schedule_id="fixed-eval"),
        observation_template=proxy_observation(0, "WHEELED", step=0),
        scenario_seeds=seeds,
    )
    mode = RewardV4EvaluationMode(
        tier=tier,
        checkpoint_payload_sha256="a" * 64,
        previous_candidate_gpu_seconds=(
            0.0 if tier is RewardV4EvaluationTier.SENTINEL else 43_199.0
        ),
        candidate_gpu_seconds=(
            10_800.0 if tier is RewardV4EvaluationTier.SENTINEL else 43_200.0
        ),
    )

    request = cli_module._reward_v4_evaluation_request(
        evaluation_directory=(tmp_path / "candidate-33").resolve(),
        mode=mode,
        manifest=manifest,
        batch=batch,
    )

    assert request.tier is tier
    assert request.batch.scenario_seeds == expected_seeds
    assert tuple(
        sorted({task.evaluation_seed for task in request.manifest.tasks})
    ) == expected_seeds
    assert request.max_macro_actions_per_task == expected_cap
    assert request.report_path.name == expected_report
    assert request.progress_directory.name == f"{tier.value}-progress"


def test_parallel_adapter_preserves_terminal_audit_outside_policy_inputs() -> None:
    observation = proxy_observation(0, "WHEELED", step=0)
    audit = TerminalAudit(
        reason=TerminalReason.ZERO_GAIN,
        oracle_opportunity_count=0,
        candidate_diagnostics=CandidateDiagnostics(
            zero_gain_count=3,
        ),
        remaining_coverable_detail_cell_count=42,
    )
    prepared = ParallelRolloutStep(
        observations=observation,
        rewards=torch.zeros(1, dtype=torch.float32),
        dones=torch.ones(1, dtype=torch.bool),
        policy_versions=torch.zeros(1, dtype=torch.int64),
        policy_decisions_consumed=torch.zeros(1, dtype=torch.int64),
        success_first_crossings=torch.zeros(1, dtype=torch.bool),
        buffer_index=0,
        candidate_diagnostics=(audit.candidate_diagnostics,),
        no_candidate_terminations=(True,),
        terminal_audits=(audit,),
    )
    pool = object.__new__(ParallelEnvPool)
    pool.worker_count = 1
    pool.reset = lambda: prepared
    pool.prepare_decision_boundaries = lambda **kwargs: prepared
    adapter = _ParallelPoolVectorEnv(pool, policy_version=0)

    adapter.reset()
    result = adapter.prepare_decision_boundaries()

    assert result.dones.tolist() == [True]
    assert adapter.terminal_audits == [audit]
    assert observation.input_names == (
        "prior_channels",
        "coverage_summary",
        "local_crop",
        "frontier_features",
        "pose_features",
        "candidate_mask",
        "platform_context",
    )


def test_task_four_cli_registers_calibrate_train_resume_and_evaluate() -> None:
    """Would fail if the approved calibration/evaluation handoff were missing."""
    parser = build_parser()

    prepare = parser.parse_args(
        [
            "prepare-data",
            "--source-lock",
            "/tmp/polar-source-lock.json",
            "--split-manifest",
            "/tmp/polar-split.json",
            "--cache-root",
            "/tmp/formal-cache",
            "--materialization",
            "full",
        ]
    )

    calibrate = parser.parse_args(
        [
            "calibrate",
            "--config",
            "training/configs/rtx4080_super_v3_joint.yaml",
            "--artifact-root",
            "/tmp/lunar-task4",
            "--cache-manifest",
            "/tmp/formal-cache/cache-manifest.json",
            "--sensor-performance-report",
            "/tmp/sensor-performance.json",
        ]
    )

    train = parser.parse_args(
        [
            "train",
            "--config",
            "training/configs/rtx4080_super_smoke.yaml",
            "--artifact-root",
            "/tmp/lunar-task3",
            "--warm-start-checkpoint",
            "/tmp/parent-policy.pt",
        ]
    )
    resume = parser.parse_args(
        [
            "resume",
            "--artifact-root",
            "/tmp/lunar-task4",
            "--checkpoint",
            "/tmp/lunar-task4/checkpoints/latest.pt",
        ]
    )
    evaluate = parser.parse_args(
        [
            "evaluate",
            "--checkpoint",
            "/tmp/lunar-task4/latest.pt",
            "--gate",
            "training/configs/candidate_gate_v1.yaml",
            "--artifact-root",
            "/tmp/lunar-task4",
        ]
    )
    formal_preflight = parser.parse_args(
        [
            "formal-preflight",
            "--config",
            "training/configs/rtx4080_super_v3_joint.yaml",
            "--cache-manifest",
            "/tmp/formal-cache/cache-manifest.json",
            "--artifact-root",
            "/tmp/formal-preflight",
            "--calibration-root",
            "/tmp/lunar-task4",
            "--sensor-performance-report",
            "/tmp/sensor-performance.json",
        ]
    )
    closed_loop_gate = parser.parse_args(
        [
            "closed-loop-gate",
            "--cache-manifest",
            "/tmp/preflight-cache/cache-manifest.json",
            "--artifact-root",
            "/tmp/closed-loop-gate",
            "--max-workers",
            "8",
        ]
    )
    extension = parser.parse_args(
        [
            "extend-budget",
            "--artifact-root",
            "/tmp/lunar-task4",
            "--blocks",
            "2",
        ]
    )

    assert prepare.command == "prepare-data"
    assert prepare.materialization == "full"
    assert prepare.preflight_scenario_limit is None
    assert calibrate.command == "calibrate"
    assert calibrate.cache_manifest == "/tmp/formal-cache/cache-manifest.json"
    assert calibrate.sensor_performance_report == "/tmp/sensor-performance.json"
    assert train.command == "train"
    assert train.sensor_performance_report is None
    assert train.warm_start_checkpoint == "/tmp/parent-policy.pt"
    assert resume.command == "resume"
    assert resume.sensor_performance_report is None
    assert evaluate.command == "evaluate"
    assert formal_preflight.command == "formal-preflight"
    assert formal_preflight.cache_manifest.endswith("cache-manifest.json")
    assert formal_preflight.calibration_root == "/tmp/lunar-task4"
    assert closed_loop_gate.command == "closed-loop-gate"
    assert closed_loop_gate.minimum_scenes == 1
    assert closed_loop_gate.max_workers == 8
    assert evaluate.sensor_performance_report is None
    assert extension.command == "extend-budget"
    assert extension.blocks == 2


def test_resume_parser_accepts_fresh_episodes_at_update_boundary() -> None:
    """The non-exact mode is explicit rather than an implicit recovery fallback."""
    parsed = build_parser().parse_args(
        [
            "resume",
            "--artifact-root",
            "/tmp/lunar-task4",
            "--checkpoint",
            "/tmp/lunar-task4/checkpoints/update-00000012.pt",
            "--fresh-episodes-at-update-boundary",
        ]
    )

    assert parsed.fresh_episodes_at_update_boundary is True


def test_fresh_episode_resume_records_an_idempotent_audit_marker(
    tmp_path: pathlib.Path,
) -> None:
    """The manifest must disclose that active worker state was intentionally reset."""
    manifest_path = tmp_path / "run-manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": cli_module.RUN_MANIFEST_SCHEMA_VERSION}),
        encoding="utf-8",
    )
    checkpoint = SimpleNamespace(
        global_step=12,
        payload_sha256="a" * 64,
        update_recovery_state=SimpleNamespace(
            update_id=12,
            journal_state="SEALED",
        ),
    )
    checkpoint_path = tmp_path / "checkpoints" / "update-00000012.pt"

    first = cli_module._record_fresh_episode_resume(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
    )
    second = cli_module._record_fresh_episode_resume(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert first == second
    assert payload["fresh_episode_resume_history"] == [first]
    assert first == {
        "schema_version": "lunar-fresh-episode-resume/v1",
        "mode": "fresh-episodes-at-update-boundary",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_payload_sha256": "a" * 64,
        "global_step": 12,
        "environment_state_restored": False,
        "worker_boundary_replayed": False,
    }


def test_closed_loop_gate_cli_runs_without_starting_training(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}
    report = SimpleNamespace(
        payload={
            "closed_loop_evidence_sha256": "e" * 64,
            "scene_count": 24,
            "scene_platform_count": 72,
            "training_started": False,
        }
    )

    def run_gate(**kwargs):
        captured.update(kwargs)
        return report, tmp_path / "gate" / "closed-loop-gate.json"

    monkeypatch.setattr(cli_module, "run_closed_loop_gate", run_gate)
    monkeypatch.setattr(cli_module, "_source_commit", lambda _root: "a" * 40)

    assert (
        cli_module.main(
            [
                "closed-loop-gate",
                "--cache-manifest",
                str(tmp_path / "cache" / "cache-manifest.json"),
                "--artifact-root",
                str(tmp_path / "gate"),
                "--minimum-scenes",
                "24",
                "--max-workers",
                "6",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["training_started"] is False
    assert output["closed_loop_evidence_sha256"] == "e" * 64
    assert captured["minimum_scene_count"] == 24
    assert captured["max_workers"] == 6


def test_formal_preflight_passes_the_validated_sensor_digest_to_calibration(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if the formal-preflight handoff used an unbound alias."""
    sensor_digest = "s" * 64
    bundle = FrozenCapabilityBundle(
        schema="lunar-training-capability-freeze/v1",
        platforms=(),
        bundle_sha256="b" * 64,
        formal_eligible=True,
    )
    cache = SimpleNamespace(identity=object())
    assembly = SimpleNamespace()
    config = load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
    )
    calibrated = SimpleNamespace(
        config=config,
        selected_workers=24,
        micro_batch_size=4,
        rollout_horizon=1,
        allocation={"WHEELED": 8, "LEGGED": 8, "HOPPER": 8},
    )
    captured: dict[str, object] = {}
    task_areas: list[object] = []
    task_key_sources: list[object] = []
    monkeypatch.setattr(
        cli_module, "_formal_capability_preflight", lambda _root: bundle
    )
    monkeypatch.setattr(
        cli_module,
        "_formal_sensor_performance_preflight",
        lambda *args, **kwargs: sensor_digest,
    )
    def build_formal_environment(*args, task_area, task_key_source_commit, **kwargs):
        task_areas.append(task_area)
        task_key_sources.append(task_key_source_commit)
        return cache, assembly

    monkeypatch.setattr(
        cli_module, "_build_formal_environment", build_formal_environment
    )
    monkeypatch.setattr(
        cli_module,
        "_open_formal_task_cache_runtime",
        lambda **kwargs: contextlib.nullcontext(
            SimpleNamespace(client=object(), source_commit="e" * 40)
        ),
    )

    def validated_calibration(**kwargs):
        captured.update(kwargs)
        return calibrated

    monkeypatch.setattr(
        cli_module,
        "_validated_formal_preflight_calibration",
        validated_calibration,
    )
    monkeypatch.setattr(
        cli_module, "_formal_run_identity", lambda _identity: object()
    )
    report = SimpleNamespace(
        payload={
            "selected_workers": 24,
            "selected_micro_batch": 4,
        }
    )
    monkeypatch.setattr(
        cli_module,
        "run_formal_preflight",
        lambda **kwargs: (report, tmp_path / "preflight" / "formal-preflight.json"),
    )

    assert cli_module.main(
        [
            "formal-preflight",
            "--config",
            str(REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"),
            "--cache-manifest",
            str(tmp_path / "cache-manifest.json"),
            "--artifact-root",
            str(tmp_path / "preflight"),
            "--calibration-root",
            str(tmp_path / "calibration"),
            "--sensor-performance-report",
            str(tmp_path / "sensor-performance.json"),
        ]
    ) == 0
    assert captured["sensor_performance_sha256"] == sensor_digest
    assert task_areas == [config.task_area] * 4
    assert task_key_sources == ["e" * 40] * 4


def test_formal_calibrate_rejects_cache_before_cuda_or_artifact_creation(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "formal-run"
    cache_manifest = tmp_path / "preflight-cache" / "cache-manifest.json"
    sensor_report = tmp_path / "sensor-performance.json"
    touched: list[str] = []
    bundle = FrozenCapabilityBundle(
        schema="lunar-training-capability-freeze/v1",
        platforms=(),
        bundle_sha256="b" * 64,
        formal_eligible=True,
    )
    monkeypatch.setattr(
        cli_module, "_formal_capability_preflight", lambda _root: bundle
    )
    monkeypatch.setattr(
        cli_module,
        "_formal_sensor_performance_preflight",
        lambda *args, **kwargs: "s" * 64,
    )
    monkeypatch.setattr(
        cli_module,
        "_build_formal_environment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PreflightError("formal command requires a full cache")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: touched.append("cuda") or True
    )

    with pytest.raises(PreflightError, match="full cache"):
        cli_module.main(
            [
                "calibrate",
                "--config",
                str(REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"),
                "--artifact-root",
                str(artifact_root),
                "--cache-manifest",
                str(cache_manifest),
                "--sensor-performance-report",
                str(sensor_report),
            ]
        )

    assert touched == []
    assert not artifact_root.exists()


def test_formal_calibrate_validates_capability_before_config(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []
    monkeypatch.setattr(
        cli_module,
        "_formal_capability_preflight",
        lambda _root: (_ for _ in ()).throw(
            PreflightError("project formal capability is invalid")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "load_training_config",
        lambda _path: touched.append("config"),
    )

    with pytest.raises(PreflightError, match="capability"):
        cli_module.main(
            [
                "calibrate",
                "--config",
                str(tmp_path / "formal.yaml"),
                "--artifact-root",
                str(tmp_path / "run"),
                "--cache-manifest",
                str(tmp_path / "cache-manifest.json"),
                "--sensor-performance-report",
                str(tmp_path / "sensor.json"),
            ]
        )

    assert touched == []


@pytest.mark.parametrize(
    "bundle",
    (
        FrozenCapabilityBundle(
            schema="lunar-training-capability-freeze/v1",
            platforms=(),
            bundle_sha256="b" * 64,
            formal_eligible=True,
        ),
        FrozenCapabilityBundle(
            schema="lunar-training-capability-freeze/v1",
            platforms=(),
            bundle_sha256="b" * 64,
            formal_eligible=False,
        ),
    ),
)
def test_formal_capability_preflight_rejects_incomplete_or_proxy_bundle(
    bundle: FrozenCapabilityBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "load_project_formal_capability",
        lambda _root: bundle,
    )

    with pytest.raises(PreflightError, match="capability"):
        cli_module._formal_capability_preflight(REPOSITORY_ROOT)


def test_formal_resume_equivalence_rejects_checkpoint_v8() -> None:
    def checkpoint(schema_version: str) -> SimpleNamespace:
        return SimpleNamespace(
            schema_version=schema_version,
            model_state={"value": 1},
            optimizer_state={"value": 2},
            rng_state={"value": 3},
            environment_state={"value": 4},
        )

    legacy = checkpoint("lunar-ppo-checkpoint/v8")
    with pytest.raises(PreflightError, match="checkpoint schema"):
        cli_module._build_formal_resume_equivalence_evidence(
            checkpoint_relative_path="resume-equivalence/update-1.pt",
            checkpoint_sha256="a" * 64,
            checkpoint_roundtrip=True,
            rollout_exact=True,
            uninterrupted=legacy,
            resumed=legacy,
            uninterrupted_observation_sha256="b" * 64,
            resumed_observation_sha256="b" * 64,
            uninterrupted_candidate_sha256="c" * 64,
            resumed_candidate_sha256="c" * 64,
            uninterrupted_request_sha256="d" * 64,
            resumed_request_sha256="d" * 64,
        )

    current = checkpoint("lunar-ppo-checkpoint/v9")
    evidence = cli_module._build_formal_resume_equivalence_evidence(
        checkpoint_relative_path="resume-equivalence/update-1.pt",
        checkpoint_sha256="a" * 64,
        checkpoint_roundtrip=True,
        rollout_exact=True,
        uninterrupted=current,
        resumed=current,
        uninterrupted_observation_sha256="b" * 64,
        resumed_observation_sha256="b" * 64,
        uninterrupted_candidate_sha256="c" * 64,
        resumed_candidate_sha256="c" * 64,
        uninterrupted_request_sha256="d" * 64,
        resumed_request_sha256="d" * 64,
    )
    assert evidence["checkpoint_schema"] == "lunar-ppo-checkpoint/v9"


def test_cache_accepts_runtime_only_visibility_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module, "_commit_is_ancestor", lambda *_args: True
    )
    monkeypatch.setattr(
        cli_module,
        "_commit_changed_paths",
        lambda *_args: (
            "docs/runtime-repair.md",
            "training/lunar_policy_training/lunar_policy_training/environment/"
            "multires_observation.py",
            "ros2_ws/src/lunar_planner_training_bridge/include/"
            "lunar_planner_training_bridge/visibility.hpp",
            "ros2_ws/src/lunar_planner_training_bridge/src/visibility.cpp",
            "ros2_ws/src/lunar_planner_training_bridge/test/"
            "visibility_benchmark.cpp",
        ),
    )

    assert cli_module._cache_accepts_runtime_only_v3_repair(
        REPOSITORY_ROOT,
        cached_commit="a" * 40,
        current_commit="b" * 40,
    )


def test_cache_accepts_batched_visibility_runtime_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The V2 batch bridge may reuse static task-cache material."""
    monkeypatch.setattr(
        cli_module, "_commit_is_ancestor", lambda *_args: True
    )
    monkeypatch.setattr(
        cli_module,
        "_commit_changed_paths",
        lambda *_args: (
            "ros2_ws/src/lunar_planner_training_bridge/include/"
            "lunar_planner_training_bridge/visibility.hpp",
            "ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp",
            "ros2_ws/src/lunar_planner_training_bridge/src/visibility.cpp",
            "ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py",
            "ros2_ws/src/lunar_planner_training_bridge/test/visibility_test.cpp",
        ),
    )

    assert cli_module._cache_accepts_runtime_only_v3_repair(
        REPOSITORY_ROOT,
        cached_commit="a" * 40,
        current_commit="b" * 40,
    )


def test_cache_accepts_exact_ground_endpoint_runtime_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Endpoint certification augments runtime queries, not cached scene data."""
    monkeypatch.setattr(
        cli_module, "_commit_is_ancestor", lambda *_args: True
    )
    monkeypatch.setattr(
        cli_module,
        "_commit_changed_paths",
        lambda *_args: (
            "docs/superpowers/specs/fine-ground-endpoints.md",
            "training/lunar_policy_training/lunar_policy_training/environment/"
            "candidate_builder.py",
            "ros2_ws/src/lunar_planner_core/include/lunar_planner_core/"
            "reachability_projection.hpp",
            "ros2_ws/src/lunar_planner_core/src/shared/"
            "reachability_projection.cpp",
            "ros2_ws/src/lunar_planner_core/test/"
            "reachability_projection_test.cpp",
            "ros2_ws/src/lunar_planner_training_bridge/include/"
            "lunar_planner_training_bridge/request.hpp",
            "ros2_ws/src/lunar_planner_training_bridge/src/conversions.cpp",
            "ros2_ws/src/lunar_planner_training_bridge/src/python_bindings.cpp",
            "ros2_ws/src/lunar_planner_training_bridge/test/test_bridge.py",
        ),
    )

    assert cli_module._cache_accepts_runtime_only_v3_repair(
        REPOSITORY_ROOT,
        cached_commit="a" * 40,
        current_commit="b" * 40,
    )


def test_cache_rejects_exact_endpoint_allowlist_if_global_planner_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module, "_commit_is_ancestor", lambda *_args: True
    )
    monkeypatch.setattr(
        cli_module,
        "_commit_changed_paths",
        lambda *_args: (
            "ros2_ws/src/lunar_planner_core/src/shared/"
            "reachability_projection.cpp",
            "ros2_ws/src/lunar_planner_core/src/planner.cpp",
        ),
    )

    assert not cli_module._cache_accepts_runtime_only_v3_repair(
        REPOSITORY_ROOT,
        cached_commit="a" * 40,
        current_commit="b" * 40,
    )


def test_cache_rejects_runtime_repair_that_changes_planner_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module, "_commit_is_ancestor", lambda *_args: True
    )
    monkeypatch.setattr(
        cli_module,
        "_commit_changed_paths",
        lambda *_args: (
            "ros2_ws/src/lunar_planner_training_bridge/src/visibility.cpp",
            "ros2_ws/src/lunar_planner_training_bridge/src/bindings.cpp",
        ),
    )

    assert not cli_module._cache_accepts_runtime_only_v3_repair(
        REPOSITORY_ROOT,
        cached_commit="a" * 40,
        current_commit="b" * 40,
    )


def test_prepare_data_delegates_to_formal_cache_without_touching_cuda(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []
    bundle = FrozenCapabilityBundle(
        schema="lunar-training-capability-freeze/v1",
        platforms=(),
        bundle_sha256="b" * 64,
        formal_eligible=True,
    )
    monkeypatch.setattr(cli_module, "_formal_capability_preflight", lambda _root: bundle)
    monkeypatch.setattr(
        cli_module,
        "prepare_formal_training_cache",
        lambda **kwargs: calls.append(kwargs)
        or {
            "cache_manifest_sha256": "c" * 64,
            "materialization": "preflight",
            "scene_count": 4,
        },
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("prepare-data must not initialize CUDA"),
    )

    assert (
        cli_module.main(
            [
                "prepare-data",
                "--source-lock",
                str(tmp_path / "source.json"),
                "--split-manifest",
                str(tmp_path / "split.json"),
                "--cache-root",
                str(tmp_path / "cache"),
                "--materialization",
                "preflight",
                "--preflight-scenario-limit",
                "4",
            ]
        )
        == 0
    )
    assert calls[0]["preflight_scenario_limit"] == 4
    assert calls[0]["capability_bundle"] is bundle
    assert json.loads(capsys.readouterr().out)["scene_count"] == 4


def test_prepare_data_accepts_the_frozen_bounded_inventory(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    bundle = FrozenCapabilityBundle(
        schema="lunar-training-capability-freeze/v1",
        platforms=(),
        bundle_sha256="b" * 64,
        formal_eligible=True,
    )
    monkeypatch.setattr(
        cli_module, "_formal_capability_preflight", lambda _root: bundle
    )
    monkeypatch.setattr(
        cli_module,
        "prepare_formal_training_cache",
        lambda **kwargs: calls.append(kwargs)
        or {
            "cache_manifest_sha256": "c" * 64,
            "materialization": "bounded",
            "scene_count": 128,
        },
    )

    assert cli_module.main(
        [
            "prepare-data",
            "--source-lock",
            str(tmp_path / "source.json"),
            "--split-manifest",
            str(tmp_path / "split.json"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--materialization",
            "bounded",
            "--preflight-scenario-limit",
            "128",
        ]
    ) == 0
    assert calls[0]["materialization"] == "bounded"
    assert calls[0]["preflight_scenario_limit"] == 128


@pytest.mark.parametrize(
    "arguments,message",
    (
        (
            ["--materialization", "preflight"],
            "preflight.*requires",
        ),
        (
            [
                "--materialization",
                "full",
                "--preflight-scenario-limit",
                "1",
            ],
            "full.*rejects",
        ),
    ),
)
def test_prepare_data_rejects_ambiguous_materialization_before_capability(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "_formal_capability_preflight",
        lambda _root: pytest.fail("materialization must fail first"),
    )
    prefix = [
        "prepare-data",
        "--source-lock",
        str(tmp_path / "source.json"),
        "--split-manifest",
        str(tmp_path / "split.json"),
        "--cache-root",
        str(tmp_path / "cache"),
    ]

    with pytest.raises(PreflightError, match=message):
        cli_module.main([*prefix, *arguments])


def test_formal_sensor_performance_report_is_required_before_artifacts(
    tmp_path: pathlib.Path,
) -> None:
    bundle = FrozenCapabilityBundle(
        schema="lunar-training-capability-freeze/v1",
        platforms=(),
        bundle_sha256="b" * 64,
        formal_eligible=True,
    )

    with pytest.raises(PreflightError, match="sensor performance report"):
        _formal_sensor_performance_preflight(
            None,
            capability_bundle=bundle,
            repository_root=REPOSITORY_ROOT,
        )

    assert not tuple(tmp_path.iterdir())


def test_formal_sensor_performance_preflight_translates_strict_report_failure(
    tmp_path: pathlib.Path,
) -> None:
    report = tmp_path / "sensor-performance.json"
    report.write_text("{}\n", encoding="utf-8")
    bundle = FrozenCapabilityBundle(
        schema="lunar-training-capability-freeze/v1",
        platforms=(),
        bundle_sha256="b" * 64,
        formal_eligible=True,
    )

    with pytest.raises(PreflightError, match="sensor performance report"):
        _formal_sensor_performance_preflight(
            str(report),
            capability_bundle=bundle,
            repository_root=REPOSITORY_ROOT,
        )


def test_training_configs_explicitly_separate_formal_and_development_smoke() -> None:
    """Would fail if proxy status were inferred instead of frozen as run kind."""
    formal = load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
    )
    smoke = load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_smoke.yaml"
    )

    assert formal.run_kind == "formal"
    assert smoke.run_kind == "development-smoke"
    assert "proxy" not in formal.as_frozen_dict()
    assert "proxy" not in smoke.as_frozen_dict()

    leaked = formal.as_frozen_dict()
    leaked["proxy"] = True
    with pytest.raises(TrainingConfigError, match="exactly"):
        resolve_training_config(leaked)


def test_formal_run_identity_is_derived_only_from_verified_cache_identity() -> None:
    cache_identity = FormalCacheIdentity(
        source_lock_file_sha256="1" * 64,
        source_sha256s={
            "NASA_LOLA_87S_DEM": "2" * 64,
            "NASA_LOLA_87S_COUNT": "3" * 64,
            "JAXA_LUPEX_DATA_S1": "4" * 64,
        },
        split_manifest_file_sha256="5" * 64,
        split_sha256="6" * 64,
        scenario_manifest_sha256="7" * 64,
        generator_sha256="8" * 64,
        capability_sha256="9" * 64,
        reward_sha256=reward_weights_sha256(),
        training_semantics_sha256="b" * 64,
        v3_source_commit="c" * 40,
        v3_sha256="d" * 64,
    )

    identity = _formal_run_identity(cache_identity)

    assert identity == RunIdentity(
        run_kind="formal",
        data_sha256="1" * 64,
        split_sha256="6" * 64,
        generator_sha256="8" * 64,
        capability_sha256="9" * 64,
        reward_sha256=reward_weights_sha256(),
        v3_sha256="d" * 64,
        training_semantics_sha256="b" * 64,
    )


@pytest.mark.parametrize("command", ("train", "resume"))
def test_formal_parser_has_no_max_updates_escape_hatch(command: str) -> None:
    """Would fail if a bounded test knob remained exposed on a formal command."""
    arguments = [command]
    if command == "train":
        arguments += [
            "--config",
            "training/configs/rtx4080_super_v3_joint.yaml",
        ]
    else:
        arguments += ["--checkpoint", "/tmp/run/checkpoints/latest.pt"]
    arguments += ["--artifact-root", "/tmp/run", "--max-updates", "1"]

    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


@pytest.mark.parametrize("command", ("train", "resume", "evaluate"))
def test_formal_uses_project_capability_before_artifact_or_cuda(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """Formal entry uses the project freeze and reaches the performance gate first."""
    artifact_root = tmp_path / command
    touched: list[str] = []
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: touched.append("cuda") or True
    )
    arguments = [command]
    if command == "train":
        arguments += [
            "--config",
            str(REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"),
        ]
    elif command == "resume":
        arguments += ["--checkpoint", str(artifact_root / "checkpoints/latest.pt")]
    else:
        arguments += [
            "--checkpoint",
            str(artifact_root / "checkpoints/latest.pt"),
            "--gate",
            str(REPOSITORY_ROOT / "training/configs/candidate_gate_v1.yaml"),
        ]
    arguments += ["--artifact-root", str(artifact_root)]

    parsed = build_parser().parse_args(arguments)
    assert not hasattr(parsed, "capability_lock")

    with pytest.raises(PreflightError, match="sensor performance report"):
        cli_module.main(arguments)

    assert touched == []
    assert not artifact_root.exists()


@pytest.mark.parametrize("command", ("train", "resume", "evaluate"))
def test_formal_parser_rejects_external_capability_lock(command: str) -> None:
    arguments = [command]
    if command == "train":
        arguments += ["--config", "formal.yaml"]
    elif command == "resume":
        arguments += ["--checkpoint", "latest.pt"]
    else:
        arguments += ["--checkpoint", "latest.pt", "--gate", "gate.yaml"]
    arguments += [
        "--artifact-root",
        "/tmp/formal",
        "--capability-lock",
        "/tmp/stale-lock.json",
    ]

    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_development_smoke_is_proxy_only_and_bounded_to_two_updates(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if the test helper could become an unbounded pseudo-formal run."""
    artifact_root = tmp_path / "unbounded-smoke"
    touched: list[str] = []
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: touched.append("cuda") or True
    )

    with pytest.raises(PreflightError, match="one or two"):
        cli_module._start_training_run(
            config_path=(
                REPOSITORY_ROOT / "training/configs/rtx4080_super_smoke.yaml"
            ),
            artifact_root=artifact_root,
            repository_root=REPOSITORY_ROOT,
            max_updates=3,
            interrupt_first_update=False,
        )

    assert touched == []
    assert not artifact_root.exists()


def test_development_smoke_rejects_policy_warm_start_before_artifact_access(
    tmp_path: pathlib.Path,
) -> None:
    artifact_root = tmp_path / "smoke-warm-start"

    with pytest.raises(PreflightError, match="formal-only"):
        cli_module._start_training_run(
            config_path=(
                REPOSITORY_ROOT / "training/configs/rtx4080_super_smoke.yaml"
            ),
            artifact_root=artifact_root,
            repository_root=REPOSITORY_ROOT,
            max_updates=1,
            interrupt_first_update=False,
            warm_start_checkpoint_path=tmp_path / "parent.pt",
        )

    assert not artifact_root.exists()


def test_policy_warm_start_manifest_is_exact_immutable_step_zero_evidence(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "run-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "lunar-training-run/v1",
                "global_step": 0,
            }
        ),
        encoding="utf-8",
    )
    evidence = PolicyWarmStartEvidence(
        parent_checkpoint_sha256="1" * 64,
        parent_payload_sha256="2" * 64,
        parent_global_step=906,
        loaded_prefixes=("global_encoder", "frontier_logit_head"),
        value_head_reinitialization_sha256="3" * 64,
        value_head_seed=4080,
    )

    _freeze_policy_warm_start_manifest(path, evidence=evidence)
    _freeze_policy_warm_start_manifest(path, evidence=evidence)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["global_step"] == 0
    assert payload["policy_warm_start"] == {
        "schema_version": "lunar-policy-warm-start/v1",
        "mode": "policy-only",
        "warm_start_parent_checkpoint_sha256": "1" * 64,
        "warm_start_parent_payload_sha256": "2" * 64,
        "warm_start_parent_global_step": 906,
        "loaded_parameter_prefixes": [
            "global_encoder",
            "frontier_logit_head",
        ],
        "value_head_reinitialized": True,
        "value_head_reinitialization_sha256": "3" * 64,
        "value_head_seed": 4080,
        "fresh_training_state": {
            "global_step": 0,
            "optimizer": True,
            "scheduler": True,
            "normalization": True,
            "rng": True,
            "worker_episode_state": True,
            "metrics_journal": True,
        },
    }
    changed = PolicyWarmStartEvidence(
        parent_checkpoint_sha256="4" * 64,
        parent_payload_sha256="2" * 64,
        parent_global_step=906,
        loaded_prefixes=evidence.loaded_prefixes,
        value_head_reinitialization_sha256="3" * 64,
        value_head_seed=4080,
    )
    with pytest.raises(ArtifactRootError, match="cannot drift"):
        _freeze_policy_warm_start_manifest(path, evidence=changed)


def _perfect_development_report() -> EvaluationReport:
    metrics = PlatformMetrics(
        scenario_seeds=(101,),
        success_coverage_rate=1.0,
        safety_violation_count=0,
        invalid_action_count=0,
        output_finite_rate=1.0,
        platform_reference_mismatch_count=0,
        hopper_commitment_violation_count=0,
        selected_action_observed_safe_rate=1.0,
        deterministic_repeat_match_rate=1.0,
        planner_failure_rate=0.0,
        completion_time_s=1.0,
        theta_mean_resultant_length=0.0,
        fixed_yaw_mean_abs_delta_rad=0.0,
    )
    per_platform = {
        platform: metrics for platform in ("WHEELED", "LEGGED", "HOPPER")
    }
    return EvaluationReport(
        proxy=True,
        scenario_schedule_id="proxy-schedule",
        run_identity=RunIdentity(
            run_kind="development-smoke",
            data_sha256="1" * 64,
            split_sha256="2" * 64,
            generator_sha256="3" * 64,
            capability_sha256="4" * 64,
            reward_sha256="5" * 64,
            v3_sha256="6" * 64,
            training_semantics_sha256=(
                cli_module.training_semantics_sha256()
            ),
        ),
        reward_hash="5" * 64,
        checkpoint_sha256="7" * 64,
        methods=tuple(
            MethodEvaluation(method=method, per_platform=per_platform)
            for method in REQUIRED_METHODS
        ),
    )


def _perfect_formal_report() -> EvaluationReport:
    development = _perfect_development_report()
    return EvaluationReport(
        proxy=False,
        scenario_schedule_id="formal/evaluation",
        run_identity=RunIdentity(
            **{
                **development.run_identity.to_dict(),
                "run_kind": "formal",
            }
        ),
        reward_hash=development.reward_hash,
        checkpoint_sha256=development.checkpoint_sha256,
        methods=tuple(
            MethodEvaluation(
                method=method.method,
                per_platform=method.per_platform,
                per_split={
                    split: dict(method.per_platform)
                    for split in ("validation", "test", "holdout")
                },
            )
            for method in development.methods
        ),
    )


def test_cli_development_evaluation_artifacts_cannot_be_mistaken_for_release(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if proxy output reused formal filenames/schema/manifest keys."""
    root = tmp_path / "run"
    root.mkdir()
    (root / "run-manifest.json").write_text(
        json.dumps({"schema_version": "lunar-training-run/v1"}),
        encoding="utf-8",
    )
    report = _perfect_development_report()
    gate_result = evaluate_release_gate(
        report,
        load_gate_rules(REPOSITORY_ROOT / "training/configs/release_gate_v1.yaml"),
    )

    digest = cli_module._write_development_evaluation_artifacts(
        root=root,
        checkpoint_path=root / "checkpoints/latest.pt",
        report=report,
        gate_result=gate_result,
    )

    evaluation = root / "evaluation"
    result = json.loads(
        (evaluation / "development-result.json").read_text(encoding="utf-8")
    )
    assert (evaluation / "development-report.json").is_file()
    assert not (evaluation / "report.json").exists()
    assert not (evaluation / "gate.json").exists()
    assert result == {
        "schema_version": "lunar-policy-development-evaluation-result/v1",
        "report_sha256": digest,
        "run_kind": "development-smoke",
        "proxy": True,
        "formal_candidate_eligible": False,
        "release_gate_passed": False,
        "failed_rules": ["formal_candidate_eligible"],
    }
    manifest = json.loads(
        (root / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert "last_evaluation" not in manifest
    assert manifest["last_development_evaluation"] == {
        "checkpoint": str(root / "checkpoints/latest.pt"),
        "report_sha256": digest,
        "run_kind": "development-smoke",
        "proxy": True,
        "formal_candidate_eligible": False,
    }


def test_formal_evaluation_batches_ignore_preflight_train_assembly(
    tmp_path: pathlib.Path,
) -> None:
    scenario_path = tmp_path / "scenario-manifest.json"
    scenario_path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scene_id": "a" * 64,
                        "split": "validation",
                        "scenario_seed": 409000,
                        "scene_seed": "1" * 64,
                    },
                    {
                        "scene_id": "d" * 64,
                        "split": "validation",
                        "scenario_seed": 409001,
                        "scene_seed": "4" * 64,
                    },
                    {
                        "scene_id": "b" * 64,
                        "split": "test",
                        "scenario_seed": 410000,
                        "scene_seed": "2" * 64,
                    },
                    {
                        "scene_id": "c" * 64,
                        "split": "holdout",
                        "scenario_seed": None,
                        "scene_seed": "fedcba9876543210" + "3" * 48,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cache = SimpleNamespace(
        root=tmp_path,
        manifest={
            "scenes": [
                {
                    "scene_id": "c" * 64,
                    "split": "holdout",
                },
                {
                    "scene_id": "d" * 64,
                    "split": "validation",
                },
                {
                    "scene_id": "a" * 64,
                    "split": "validation",
                },
                {
                    "scene_id": "b" * 64,
                    "split": "test",
                },
            ],
            "exact_common_evaluation": {
                "splits": {
                    "validation": {
                        "scene_ids": ["d" * 64, "a" * 64],
                        "scenario_schedule_id": "cache/validation/v6",
                    },
                    "test": {
                        "scene_ids": ["b" * 64],
                        "scenario_schedule_id": "cache/test/v6",
                    },
                    "holdout": {
                        "scene_ids": ["c" * 64],
                        "scenario_schedule_id": "cache/holdout/v6",
                    },
                }
            },
        },
    )
    template = proxy_observation(0, "WHEELED", step=0)
    assemblies = {
        split: SimpleNamespace(
            factory=SimpleNamespace(scenario_schedule_id=f"cache/{split}/v6"),
            scenario_schedule_id=f"cache/{split}/v6",
            observation_template=template,
        )
        for split in ("train", "validation", "test", "holdout")
    }

    batches = cli_module._formal_evaluation_batches(cache, assemblies)

    assert tuple(batch.split for batch in batches) == (
        "validation",
        "test",
        "holdout",
    )
    assert batches[0].scenario_seeds == (409000, 409001)
    assert batches[1].scenario_seeds == (410000,)
    assert batches[2].scenario_seeds == (int("fedcba9876543210", 16),)


def test_formal_evaluation_artifacts_are_non_proxy_and_gate_bound(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run-manifest.json").write_text(
        json.dumps({"schema_version": "lunar-training-run/v1"}),
        encoding="utf-8",
    )
    report = _perfect_formal_report()
    gate_result = evaluate_release_gate(
        report,
        load_gate_rules(REPOSITORY_ROOT / "training/configs/release_gate_v1.yaml"),
    )

    digest = cli_module._write_formal_evaluation_artifacts(
        root=root,
        checkpoint_path=root / "checkpoints/candidate-step-42.pt",
        report=report,
        gate_result=gate_result,
        started_gpu_seconds=32400.0,
        completed_gpu_seconds=32412.5,
    )

    result = json.loads(
        (root / "evaluation/formal-result.json").read_text(encoding="utf-8")
    )
    assert (root / "evaluation/formal-report.json").is_file()
    assert (
        root / "evaluation/candidate-step-42/formal-report.json"
    ).is_file()
    immutable_result = json.loads(
        (
            root / "evaluation/candidate-step-42/formal-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["report_sha256"] == digest
    assert immutable_result["report_sha256"] == digest
    assert immutable_result["started_gpu_seconds"] == 32400.0
    assert immutable_result["completed_gpu_seconds"] == 32412.5
    assert result["run_kind"] == "formal"
    assert result["proxy"] is False
    assert result["formal_candidate_eligible"] is True
    manifest = json.loads(
        (root / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["last_evaluation"]["report_sha256"] == digest
    assert manifest["last_evaluation"]["started_gpu_seconds"] == 32400.0
    assert manifest["last_evaluation"]["completed_gpu_seconds"] == 32412.5

    with pytest.raises(PreflightError, match="already exists"):
        cli_module._write_formal_evaluation_artifacts(
            root=root,
            checkpoint_path=root / "checkpoints/candidate-step-42.pt",
            report=report,
            gate_result=gate_result,
            started_gpu_seconds=32400.0,
            completed_gpu_seconds=32412.5,
        )


def test_formal_evaluation_uses_shared_budget_and_preserves_training_position(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    candidate = checkpoints / "candidate-step-7.pt"
    candidate.write_bytes(b"checkpoint")
    report = _perfect_formal_report()
    allocation = {"WHEELED": 24}
    source_commit = "a" * 40
    budget = TrainingBudget(consumed_gpu_seconds=100.0)
    config = load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
    )
    calibrated = cli_module.CalibratedRunState(
        config=config,
        budget=budget,
        allocation=allocation,
        selected_workers=24,
        micro_batch_size=2,
        rollout_horizon=config.ppo.rollout_horizon,
        reward_hash=report.reward_hash,
        scenario_schedule_id="formal/train",
        formal_seed=4080,
        reward_calibration_seeds=(4081, 4082, 4083),
        calibration_end_gpu_seconds=600.0,
        run_identity=report.run_identity,
        cache_manifest_path=tmp_path / "cache.json",
        cache_manifest_sha256="8" * 64,
        sensor_performance_sha256="9" * 64,
    )
    (root / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "lunar-training-run/v1",
                "runtime_calibration": {},
                "budget_extension_blocks": 0,
                "total_gpu_budget_seconds": 86400,
                "consumed_gpu_seconds": 100.0,
                "global_step": 118,
                "platform_allocation": allocation,
                "run_identity": report.run_identity.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    checkpoint = SimpleNamespace(
        model_state={},
        payload_sha256=report.checkpoint_sha256,
        source_commit=source_commit,
        config_hash="b" * 64,
        global_step=7,
    )
    template = proxy_observation(0, "WHEELED", step=0)
    batches = tuple(
        FormalEvaluationBatch(
            split=split,
            factory=SimpleNamespace(scenario_schedule_id=f"formal/{split}"),
            observation_template=template,
            scenario_seeds=(1,),
        )
        for split in ("validation", "test", "holdout")
    )
    monkeypatch.setattr(
        cli_module, "_validate_formal_bundle_identity", lambda *args: None
    )
    monkeypatch.setattr(
        cli_module, "load_checkpoint_for_resume", lambda *args, **kwargs: checkpoint
    )
    monkeypatch.setattr(cli_module, "_source_commit", lambda _root: source_commit)
    monkeypatch.setattr(
        cli_module,
        "CrossAttentionPolicy",
        lambda: SimpleNamespace(load_state_dict=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        cli_module, "evaluate_formal_policy", lambda *args, **kwargs: report
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    timestamps = iter((10.0, 15.0))
    monkeypatch.setattr(cli_module.time, "monotonic", lambda: next(timestamps))
    reserved: list[float] = []
    original_begin = TrainingBudget.begin_gpu_interval

    def begin_interval(self, *, monotonic_seconds, upper_bound_gpu_seconds):
        reserved.append(float(upper_bound_gpu_seconds))
        return original_begin(
            self,
            monotonic_seconds=monotonic_seconds,
            upper_bound_gpu_seconds=upper_bound_gpu_seconds,
        )

    monkeypatch.setattr(TrainingBudget, "begin_gpu_interval", begin_interval)

    _, gate_result = cli_module._evaluate_checkpoint(
        checkpoint_path=candidate,
        gate_path=REPOSITORY_ROOT / "training/configs/candidate_gate_v1.yaml",
        artifact_root=root,
        repository_root=REPOSITORY_ROOT,
        capability_bundle=object(),
        formal_batches=batches,
        shared_budget=budget,
        evaluation_started_gpu_seconds=100.0,
        calibrated_state=calibrated,
    )

    manifest = json.loads(
        (root / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert reserved == [3600.0]
    assert budget.consumed_gpu_seconds == 105.0
    assert manifest["global_step"] == 118
    assert manifest["platform_allocation"] == allocation
    assert manifest["last_evaluation"]["started_gpu_seconds"] == 100.0
    assert manifest["last_evaluation"]["completed_gpu_seconds"] == 105.0
    assert gate_result.formal_candidate_eligible is True


@pytest.mark.parametrize("blocks", ["0", "-1", "1.5", "not-an-int"])
def test_extend_budget_cli_rejects_non_positive_integer_blocks(blocks: str) -> None:
    """Would fail if automation could pass an implicit or invalid extension."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "extend-budget",
                "--artifact-root",
                "/tmp/lunar-task4",
                "--blocks",
                blocks,
            ]
        )


def test_task_five_authoritative_checkpoint_paths_are_versioned_under_directory(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if Task 5 train/resume used the legacy root-level checkpoint."""
    assert _checkpoint_target(tmp_path, kind="latest", global_step=12) == (
        tmp_path / "checkpoints/latest.pt"
    )
    assert _checkpoint_target(tmp_path, kind="candidate", global_step=12) == (
        tmp_path / "checkpoints/candidate-step-12.pt"
    )


def test_artifact_root_must_be_explicit_absolute_and_outside_repository(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if checkpoints or logs could be written into the Git tree."""
    with pytest.raises(ArtifactRootError, match="absolute"):
        validate_artifact_root(
            pathlib.Path("relative/run"), repository_root=REPOSITORY_ROOT
        )
    with pytest.raises(ArtifactRootError, match="outside"):
        validate_artifact_root(
            REPOSITORY_ROOT / "training-output", repository_root=REPOSITORY_ROOT
        )

    assert validate_artifact_root(
        tmp_path / "run", repository_root=REPOSITORY_ROOT
    ) == tmp_path / "run"


def _write_calibrated_manifest(root: pathlib.Path, *, consumed: float = 12.5) -> None:
    root.mkdir()
    run_identity = cli_module._development_run_identity(
        cli_module._source_commit(REPOSITORY_ROOT)
    )
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "training/configs/rtx4080_super_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    (root / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "lunar-training-run/v1",
                "frozen_config": config,
                "run_identity": run_identity.to_dict(),
                "runtime_calibration": {
                    "selected_workers": 24,
                    "selected_micro_batch": 2,
                    "selected_rollout_horizon": 32,
                    "compared_workers": [18, 24, 30],
                    "measurements": [],
                    "rollout_horizon_candidates": [],
                    "horizon_transitions_per_worker": 0,
                    "horizon_measurements": [],
                },
                "consumed_gpu_seconds": consumed,
                "budget_extension_blocks": 0,
                "total_gpu_budget_seconds": 86400,
            }
        ),
        encoding="utf-8",
    )
    _freeze_task4_manifest(
        root / "run-manifest.json",
        schedule=CurriculumSchedule(),
        selected_workers=24,
        reward_hash=reward_weights_sha256(),
        reward_seed_results=(
            {
                "seed": 4081,
                "minimum_platform_score": 0.5,
                "report_sha256": "1" * 64,
            },
            {
                "seed": 4082,
                "minimum_platform_score": 0.5,
                "report_sha256": "2" * 64,
            },
            {
                "seed": 4083,
                "minimum_platform_score": 0.5,
                "report_sha256": "3" * 64,
            },
        ),
    )


def _formal_horizon_measurements() -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for horizon, throughput, update_wall in (
        (16, 90.0, 2.0),
        (32, 120.0, 3.0),
        (64, 110.0, 5.0),
    ):
        values.append(
            {
                "workers": 24,
                "micro_batch": 2,
                "rollout_horizon": horizon,
                "total_transitions": 24 * 64,
                "completed_updates": 64 // horizon,
                "throughput_transitions_per_second": throughput,
                "mean_update_wall_seconds": update_wall,
                "peak_gpu_memory_fraction": 0.5,
                "worker_wait_ratio": 0.25,
                "planner_timeouts": 0,
                "oom": False,
                "gpu_seconds": 1.0,
                "ipc_failures": 0,
                "approximate_kl": 0.01,
                "value_loss": 0.2,
                "advantages_finite": True,
                "returns_finite": True,
                "update_boundary_continuity_verified": True,
                "resume_digest": f"{horizon // 16:x}" * 64,
                "resume_verified": True,
            }
        )
    return values


def test_extend_budget_cli_atomically_updates_only_run_budget_state(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if the explicit command reset consumption or touched checkpoints."""
    root = tmp_path / "calibrated"
    _write_calibrated_manifest(root, consumed=321.0)

    assert cli_module.main(
        [
            "extend-budget",
            "--artifact-root",
            str(root),
            "--blocks",
            "2",
        ]
    ) == 0

    payload = json.loads((root / "run-manifest.json").read_text(encoding="utf-8"))
    assert payload["budget_extension_blocks"] == 2
    assert payload["total_gpu_budget_seconds"] == 129600
    assert payload["consumed_gpu_seconds"] == 321.0
    assert not (root / "checkpoints").exists()


def test_calibration_freezes_reward_schedule_and_one_shared_budget(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if train/evaluate could create a fresh Task 4 run identity."""
    root = tmp_path / "calibrated"
    _write_calibrated_manifest(root, consumed=123.5)

    state = _load_calibrated_run_state(root)

    assert state.budget.consumed_gpu_seconds == 123.5
    assert state.reward_hash == reward_weights_sha256()
    assert state.scenario_schedule_id == CurriculumSchedule().scenario_schedule_id
    assert state.formal_seed == 4080
    assert state.reward_calibration_seeds == (4081, 4082, 4083)
    assert state.allocation == {"WHEELED": 8, "LEGGED": 8, "HOPPER": 8}
    assert state.rollout_horizon == 32


def test_formal_calibration_manifest_is_non_proxy_and_cache_bound(
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "formal-calibrated"
    root.mkdir()
    cache_manifest = tmp_path / "cache-manifest.json"
    cache_manifest.write_text("{}\n", encoding="utf-8")
    config = yaml.safe_load(
        (
            REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
        ).read_text(encoding="utf-8")
    )
    cache_identity = FormalCacheIdentity(
        source_lock_file_sha256="1" * 64,
        source_sha256s={
            "NASA_LOLA_87S_DEM": "2" * 64,
            "NASA_LOLA_87S_COUNT": "3" * 64,
            "JAXA_LUPEX_DATA_S1": "4" * 64,
        },
        split_manifest_file_sha256="5" * 64,
        split_sha256="6" * 64,
        scenario_manifest_sha256="7" * 64,
        generator_sha256="8" * 64,
        capability_sha256="9" * 64,
        reward_sha256=reward_weights_sha256(),
        training_semantics_sha256="b" * 64,
        v3_source_commit="c" * 40,
        v3_sha256="d" * 64,
    )
    run_identity = _formal_run_identity(cache_identity)
    (root / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "lunar-training-run/v1",
                "frozen_config": config,
                "runtime_calibration": {
                    "selection_mode": "operator-fixed/v1",
                    "selected_workers": 24,
                    "selected_micro_batch": 4,
                    "selected_rollout_horizon": 12,
                    "compared_workers": [24],
                    "measurements": [],
                    "rollout_horizon_candidates": [],
                    "horizon_transitions_per_worker": 0,
                    "horizon_measurements": [],
                },
                "consumed_gpu_seconds": 12.5,
                "budget_extension_blocks": 0,
                "total_gpu_budget_seconds": 86400,
            }
        ),
        encoding="utf-8",
    )
    schedule_id = "cache-sha/train/v2"
    _freeze_formal_environment_manifest(
        root / "run-manifest.json",
        cache_manifest_path=cache_manifest,
        cache_manifest_sha256="e" * 64,
        scenario_schedule_id=schedule_id,
        sensor_performance_sha256="f" * 64,
    )
    _update_run_manifest(
        root / "run-manifest.json",
        source_commit=cli_module._source_commit(REPOSITORY_ROOT),
        config_hash=cli_module.config_sha256(config),
        run_identity=run_identity,
        global_step=0,
        consumed_gpu_seconds=12.5,
        platform_allocation={"WHEELED": 8, "LEGGED": 8, "HOPPER": 8},
    )
    _freeze_task4_manifest(
        root / "run-manifest.json",
        schedule=CurriculumSchedule(),
        selected_workers=24,
        reward_hash=reward_weights_sha256(),
        reward_seed_results=tuple(
            {
                "seed": seed,
                "platform_mean_rewards": {
                    "WHEELED": 0.1,
                    "LEGGED": 0.2,
                    "HOPPER": 0.3,
                },
                "rollout_sha256": str(index) * 64,
            }
            for index, seed in enumerate((4081, 4082, 4083), start=1)
        ),
        proxy=False,
        scenario_schedule_id=schedule_id,
    )

    state = _load_calibrated_run_state(root)
    manifest = json.loads(
        (root / "run-manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["task4_calibration"]["proxy"] is False
    assert manifest["task4_calibration"]["curriculum"][
        "joint_worker_allocation"
    ] == {"WHEELED": 8, "LEGGED": 8, "HOPPER": 8}
    assert manifest["resume_parent"] is None
    assert manifest["warm_start_parent"] is None
    assert state.run_identity.run_kind == "formal"
    assert state.scenario_schedule_id == schedule_id
    assert state.cache_manifest_path == cache_manifest
    assert state.cache_manifest_sha256 == "e" * 64
    assert state.sensor_performance_sha256 == "f" * 64


def test_train_consumes_existing_calibrated_root_without_recalibration(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if public train rejected or recalibrated an existing run root."""
    root = tmp_path / "calibrated"
    _write_calibrated_manifest(root)
    sentinel = object()
    monkeypatch.setattr(cli_module, "_run_updates", lambda **kwargs: sentinel)

    result = cli_module._start_training_run(
        config_path=REPOSITORY_ROOT / "training/configs/rtx4080_super_smoke.yaml",
        artifact_root=root,
        repository_root=REPOSITORY_ROOT,
        max_updates=1,
        interrupt_first_update=False,
    )

    assert result is sentinel
    payload = json.loads((root / "run-manifest.json").read_text(encoding="utf-8"))
    assert payload["task4_calibration"]["scenario_schedule_id"] == (
        CurriculumSchedule().scenario_schedule_id
    )


def test_formal_train_uses_the_calibrated_horizon_not_the_bootstrap_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail when train drifts from the fixed formal horizon."""
    root = tmp_path / "formal-calibrated"
    root.mkdir()
    requested = load_training_config(
        REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
    )
    assert requested.ppo.rollout_horizon == 12
    frozen = with_rollout_horizon(requested, 12)
    identity = RunIdentity(
        run_kind="formal",
        data_sha256="1" * 64,
        split_sha256="2" * 64,
        generator_sha256="3" * 64,
        capability_sha256="4" * 64,
        reward_sha256="5" * 64,
        v3_sha256="6" * 64,
        training_semantics_sha256="7" * 64,
    )
    calibrated = SimpleNamespace(
        config=frozen,
        rollout_horizon=12,
        run_identity=identity,
        formal_seed=4080,
    )
    sentinel = object()
    monkeypatch.setattr(
        cli_module, "_load_calibrated_run_state", lambda _root: calibrated
    )
    monkeypatch.setattr(
        cli_module, "_validate_formal_bundle_identity", lambda *args: None
    )
    monkeypatch.setattr(cli_module, "_source_commit", lambda _root: "a" * 40)
    monkeypatch.setattr(cli_module, "_seed_everything", lambda _seed: None)
    monkeypatch.setattr(
        cli_module, "_run_curriculum_training", lambda **kwargs: sentinel
    )

    result = cli_module._start_training_run(
        config_path=(
            REPOSITORY_ROOT / "training/configs/rtx4080_super_v3_joint.yaml"
        ),
        artifact_root=root,
        repository_root=REPOSITORY_ROOT,
        max_updates=None,
        interrupt_first_update=False,
        capability_bundle=object(),
    )

    assert result is sentinel
    assert calibrated.config.ppo.rollout_horizon == 12


def test_sigterm_during_update_saves_only_after_complete_update_boundary() -> None:
    """Would fail if the signal handler checkpointed inside a PPO update."""
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    events: list[str] = []
    flag = SignalStopFlag()
    budget = TrainingBudget()
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=flag,
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
    )

    def update(batch: torch.Tensor) -> None:
        events.append("update-start")
        optimizer.zero_grad(set_to_none=True)
        model(batch).sum().backward()
        optimizer.step()
        os.kill(os.getpid(), signal.SIGTERM)
        events.append("update-finished")

    def save(kind: str, state) -> None:
        events.append(f"save-{kind}-step-{state.global_step}")

    with flag.installed():
        result = loop.run(
            collect_rollout=lambda: torch.ones((1, 2)),
            update_rollout=update,
            save_checkpoint=save,
            max_updates=5,
        )

    assert events == [
        "update-start",
        "update-finished",
        "save-latest-step-1",
    ]
    assert result.global_step == 1
    assert result.rollout_discarded is False
    assert result.stop_signal == signal.SIGTERM
    assert budget.consumed_gpu_seconds > 0.0


def test_signal_after_collection_discards_unfinished_rollout_without_update() -> None:
    """Would fail if a signal-boundary partial rollout entered an optimizer step."""
    events: list[str] = []
    flag = SignalStopFlag()
    loop = TrainingBoundaryLoop(
        budget=TrainingBudget(),
        stop_flag=flag,
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
    )

    def collect() -> object:
        events.append("rollout-collected")
        os.kill(os.getpid(), signal.SIGINT)
        return object()

    with flag.installed():
        result = loop.run(
            collect_rollout=collect,
            update_rollout=lambda rollout: events.append("updated"),
            save_checkpoint=lambda kind, state: events.append(
                f"save-{kind}-step-{state.global_step}"
            ),
            max_updates=3,
        )

    assert events == ["rollout-collected", "save-latest-step-0"]
    assert result.global_step == 0
    assert result.rollout_discarded is True
    assert result.stop_signal == signal.SIGINT


def test_rollout_and_failed_update_both_settle_the_same_active_gpu_budget() -> None:
    """Would fail if policy inference or an exceptional PPO update escaped accounting."""
    timestamps = iter((10.0, 15.0))
    budget = TrainingBudget()
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
        clock=lambda: next(timestamps),
    )

    with pytest.raises(RuntimeError, match="update failed"):
        loop.run(
            collect_rollout=lambda: object(),
            update_rollout=lambda rollout: (_ for _ in ()).throw(
                RuntimeError("update failed")
            ),
            save_checkpoint=lambda kind, state: None,
            max_updates=1,
        )

    assert budget.consumed_gpu_seconds == 5.0
    assert budget.interval_active is False


def test_training_boundary_records_only_a_completed_update() -> None:
    timestamps = iter((10.0, 12.0))
    events: list[object] = []
    loop = TrainingBoundaryLoop(
        budget=TrainingBudget(),
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="warmup_legged",
        clock=lambda: next(timestamps),
    )

    state = loop.run(
        collect_rollout=lambda: "rollout",
        update_rollout=lambda rollout: {"ppo": rollout},
        record_update=lambda rollout, result, boundary: events.append(
            (rollout, result, boundary.global_step, boundary.latest_checkpoint_gpu_seconds)
        ),
        save_checkpoint=lambda kind, boundary: None,
        max_updates=1,
    )

    assert state.global_step == 1
    assert events == [("rollout", {"ppo": "rollout"}, 1, 0.0)]


def test_training_boundary_does_not_record_a_failed_update() -> None:
    timestamps = iter((10.0, 12.0))
    records: list[object] = []
    loop = TrainingBoundaryLoop(
        budget=TrainingBudget(),
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="warmup_legged",
        clock=lambda: next(timestamps),
    )

    with pytest.raises(RuntimeError, match="update failed"):
        loop.run(
            collect_rollout=lambda: "rollout",
            update_rollout=lambda rollout: (_ for _ in ()).throw(
                RuntimeError("update failed")
            ),
            record_update=lambda rollout, result, boundary: records.append(result),
            save_checkpoint=lambda kind, boundary: None,
            max_updates=1,
        )

    assert records == []


def test_training_boundary_pauses_after_crossing_joint_evaluation_target() -> None:
    timestamps = iter((10.0, 15.0))
    budget = TrainingBudget(consumed_gpu_seconds=100.0)
    events: list[str] = []
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
        pause_after_gpu_seconds=103.0,
        clock=lambda: next(timestamps),
    )

    state = loop.run(
        collect_rollout=lambda: events.append("collect") or object(),
        update_rollout=lambda rollout: events.append("update"),
        save_checkpoint=lambda kind, boundary: events.append(
            f"save-{kind}-step-{boundary.global_step}"
        ),
        max_updates=5,
    )

    assert events == ["collect", "update", "save-latest-step-1"]
    assert state.global_step == 1
    assert budget.consumed_gpu_seconds == 105.0


def test_latest_candidate_checkpoint_uses_numeric_global_step(
    tmp_path: pathlib.Path,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "candidate-step-9.pt").write_bytes(b"nine")
    expected = checkpoints / "candidate-step-120.pt"
    expected.write_bytes(b"one-twenty")
    (checkpoints / "candidate-step-invalid.pt").write_bytes(b"invalid")
    (checkpoints / "latest.pt").write_bytes(b"latest")

    assert _latest_candidate_checkpoint(tmp_path) == expected


def test_resume_continues_latest_and_candidate_rhythms_from_active_gpu_markers() -> None:
    """Would fail if pause/resume restarted the 30/60-minute checkpoint clocks."""
    timestamps = iter((0.0, 1.0))
    budget = TrainingBudget(consumed_gpu_seconds=3599.0)
    saves: list[tuple[str, object]] = []
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
        initial_latest_checkpoint_gpu_seconds=1800.0,
        initial_candidate_checkpoint_gpu_seconds=0.0,
        clock=lambda: next(timestamps),
    )

    state = loop.run(
        collect_rollout=lambda: object(),
        update_rollout=lambda rollout: None,
        save_checkpoint=lambda kind, saved_state: saves.append((kind, saved_state)),
        max_updates=1,
    )

    assert [kind for kind, _ in saves][:2] == ["latest", "candidate"]
    assert state.latest_checkpoint_gpu_seconds == 3600.0
    assert state.candidate_checkpoint_gpu_seconds == 3600.0


def test_training_allows_one_complete_update_with_partial_remaining_budget() -> None:
    """The final update may finish at a safe boundary without early reservation."""
    budget = TrainingBudget(consumed_gpu_seconds=86400.0 - 600.0 + 1.0)
    events: list[str] = []
    timestamps = iter((10.0, 609.0))
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
        initial_global_step=41,
        initial_latest_checkpoint_gpu_seconds=84000.0,
        initial_candidate_checkpoint_gpu_seconds=84000.0,
        clock=lambda: next(timestamps),
    )

    state = loop.run(
        collect_rollout=lambda: events.append("collect"),
        update_rollout=lambda rollout: events.append("update"),
        save_checkpoint=lambda kind, saved_state: events.append(
            f"save-{kind}-step-{saved_state.global_step}"
        ),
        max_updates=1,
    )

    assert events == ["collect", "update", "save-latest-step-42"]
    assert state.global_step == 42
    assert state.rollout_discarded is False
    assert state.latest_checkpoint_gpu_seconds == 86400.0
    assert budget.consumed_gpu_seconds == 86400.0
    assert budget.interval_active is False


def test_curriculum_phase_allows_one_complete_update_to_cross_boundary() -> None:
    """A long macro action completes before the curriculum switches phases."""
    budget = TrainingBudget(consumed_gpu_seconds=6599.0)
    events: list[str] = []
    timestamps = iter((10.0, 611.0))
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="warmup_wheeled",
        phase_end_gpu_seconds=7199.0,
        clock=lambda: next(timestamps),
    )

    state = loop.run(
        collect_rollout=lambda: events.append("collect"),
        update_rollout=lambda rollout: events.append("update"),
        save_checkpoint=lambda kind, saved: events.append(
            f"save-{kind}-step-{saved.global_step}"
        ),
        max_updates=10,
    )

    assert events == ["collect", "update", "save-latest-step-1"]
    assert state.global_step == 1
    assert budget.consumed_gpu_seconds == 7200.0


def test_formal_curriculum_invocation_advances_all_phases_without_restart(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if an unbounded formal invocation stalled at a phase edge."""
    root = tmp_path / "calibrated"
    _write_calibrated_manifest(root)
    calibrated = _load_calibrated_run_state(root)
    phases: list[tuple[str, dict[str, int]]] = []

    def run_phase(**kwargs):
        phase = kwargs["curriculum_phase"]
        allocation = kwargs["allocation"]
        budget = kwargs["budget"]
        phase_end = kwargs["phase_end_gpu_seconds"]
        phases.append((phase, allocation))
        if phase == "joint":
            budget.consume(budget.remaining_gpu_seconds)
        else:
            budget.consume(
                max(0.0, phase_end - budget.consumed_gpu_seconds - 600.0)
            )
        return SimpleNamespace(
            global_step=len(phases),
            consumed_gpu_seconds=budget.consumed_gpu_seconds,
            platform_allocation=allocation,
            signal_observed_at_update_boundary=False,
        )

    monkeypatch.setattr(cli_module, "_run_updates", run_phase)
    monkeypatch.setattr(
        cli_module,
        "load_checkpoint",
        lambda path: SimpleNamespace(
            global_step=len(phases), curriculum_phase=phases[-1][0]
        ),
    )

    result = _run_curriculum_training(
        calibrated=calibrated,
        artifact_root=root,
        repository_root=REPOSITORY_ROOT,
        source_commit="a" * 40,
        initial_global_step=0,
        max_updates=None,
        interrupt_first_update=False,
        restore_checkpoint=None,
    )

    assert phases == [
        ("warmup_wheeled", {"WHEELED": 24}),
        ("warmup_legged", {"LEGGED": 24}),
        ("warmup_hopper", {"HOPPER": 24}),
        ("joint", {"WHEELED": 8, "LEGGED": 8, "HOPPER": 8}),
    ]
    assert result.global_step == 4
    assert calibrated.budget.exhausted is True


def test_joint_curriculum_evaluates_latest_candidate_and_continues_same_budget(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "calibrated"
    _write_calibrated_manifest(root)
    calibrated = _load_calibrated_run_state(root)
    schedule = CurriculumSchedule()
    joint_start = (
        calibrated.calibration_end_gpu_seconds
        + 3 * schedule.platform_warmup_limit_s
        - 600.0
    )
    calibrated.budget.consume(
        joint_start - calibrated.budget.consumed_gpu_seconds
    )
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    candidate = checkpoints / "candidate-step-5.pt"
    candidate.write_bytes(b"candidate")
    calls: list[tuple[str, object]] = []
    update_calls = 0

    def run_joint(**kwargs):
        nonlocal update_calls
        update_calls += 1
        budget = kwargs["budget"]
        assert budget is calibrated.budget
        assert kwargs["curriculum_phase"] == "joint"
        if update_calls == 1:
            due = kwargs["pause_after_gpu_seconds"]
            budget.consume(due - budget.consumed_gpu_seconds)
        else:
            budget.consume(budget.remaining_gpu_seconds)
        return SimpleNamespace(
            global_step=update_calls,
            consumed_gpu_seconds=budget.consumed_gpu_seconds,
            platform_allocation=kwargs["allocation"],
            signal_observed_at_update_boundary=False,
        )

    def evaluate_candidate(checkpoint, budget, started_gpu_seconds):
        calls.append(("evaluate", checkpoint))
        assert checkpoint == candidate
        assert budget is calibrated.budget
        assert started_gpu_seconds == budget.consumed_gpu_seconds
        budget.consume(1.0)
        return GateResult(
            passed=False,
            failed_rules=("validation.WHEELED.success_coverage_rate_min",),
            run_kind="formal",
            proxy=False,
            formal_candidate_eligible=True,
        )

    monkeypatch.setattr(cli_module, "_run_updates", run_joint)
    monkeypatch.setattr(
        cli_module,
        "load_checkpoint",
        lambda path: SimpleNamespace(
            global_step=update_calls,
            curriculum_phase="joint",
        ),
    )

    result = _run_curriculum_training(
        calibrated=calibrated,
        artifact_root=root,
        repository_root=REPOSITORY_ROOT,
        source_commit="a" * 40,
        initial_global_step=0,
        max_updates=None,
        interrupt_first_update=False,
        restore_checkpoint=None,
        evaluate_candidate=evaluate_candidate,
    )

    assert calls == [("evaluate", candidate)]
    assert update_calls == 2
    assert result.global_step == 2
    assert calibrated.budget.exhausted is True


def test_run_manifest_persists_exhausted_budget_terminal_state(
    tmp_path: pathlib.Path,
) -> None:
    manifest = tmp_path / "run-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "lunar-training-run/v1",
                "runtime_calibration": {"selected_workers": 18},
                "budget_extension_blocks": 0,
                "total_gpu_budget_seconds": 86400,
            }
        ),
        encoding="utf-8",
    )

    _update_run_manifest(
        manifest,
        source_commit="a" * 40,
        config_hash="b" * 64,
        run_identity=cli_module._development_run_identity("a" * 40),
        global_step=41,
        consumed_gpu_seconds=86400.0,
        platform_allocation={"WHEELED": 6, "LEGGED": 6, "HOPPER": 6},
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["budget_state"] == "exhausted"
    assert payload["consumed_gpu_seconds"] == 86400.0
    identity = cli_module._development_run_identity("a" * 40)
    assert payload["run_identity"] == identity.to_dict()
    with pytest.raises(ArtifactRootError, match="identity cannot drift"):
        _update_run_manifest(
            manifest,
            source_commit="a" * 40,
            config_hash="b" * 64,
            run_identity=cli_module._development_run_identity("c" * 40),
            global_step=41,
            consumed_gpu_seconds=86400.0,
            platform_allocation={"WHEELED": 6, "LEGGED": 6, "HOPPER": 6},
        )


def test_run_manifest_keeps_extended_budget_active_at_initial_limit(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if terminal state ignored an explicit six-hour extension."""
    manifest = tmp_path / "run-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "lunar-training-run/v1",
                "runtime_calibration": {"selected_workers": 18},
                "budget_extension_blocks": 1,
                "total_gpu_budget_seconds": 108000,
            }
        ),
        encoding="utf-8",
    )

    _update_run_manifest(
        manifest,
        source_commit="a" * 40,
        config_hash="b" * 64,
        run_identity=cli_module._development_run_identity("a" * 40),
        global_step=41,
        consumed_gpu_seconds=86400.0,
        platform_allocation={"WHEELED": 6, "LEGGED": 6, "HOPPER": 6},
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["budget_state"] == "active"
    assert payload["budget_extension_blocks"] == 1
    assert payload["total_gpu_budget_seconds"] == 108000
    assert payload["consumed_gpu_seconds"] == 86400.0


def test_run_manifest_records_the_checkpointed_metrics_journal(
    tmp_path: pathlib.Path,
) -> None:
    manifest = tmp_path / "run-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "lunar-training-run/v1",
                "runtime_calibration": {"selected_workers": 24},
                "budget_extension_blocks": 0,
                "total_gpu_budget_seconds": 86400,
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "lunar-training-metrics-summary/v1",
        "path": "metrics/train.jsonl",
        "last_global_step": 119,
        "sha256": "c" * 64,
    }

    _update_run_manifest(
        manifest,
        source_commit="a" * 40,
        config_hash="b" * 64,
        run_identity=cli_module._development_run_identity("a" * 40),
        global_step=119,
        consumed_gpu_seconds=7300.0,
        platform_allocation={"LEGGED": 24},
        training_metrics=summary,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["training_metrics"] == summary


def test_training_overrun_saves_terminal_latest_after_complete_update() -> None:
    timestamps = iter((0.0, 15000.001))
    events: list[str] = []
    budget = TrainingBudget()
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
        initial_global_step=7,
        clock=lambda: next(timestamps),
    )

    state = loop.run(
        collect_rollout=lambda: events.append("collect") or object(),
        update_rollout=lambda rollout: events.append("update"),
        save_checkpoint=lambda kind, saved_state: events.append(
            f"save-{kind}-step-{saved_state.global_step}"
        ),
        max_updates=2,
    )

    assert events == ["collect", "update", "save-latest-step-8"]
    assert state.global_step == 8
    assert state.rollout_discarded is False
    assert state.latest_checkpoint_gpu_seconds == 86400.0
    assert budget.exhausted is True


def _training_fingerprint() -> dict[str, object]:
    return {
        "schema_version": "lunar-platform-fingerprint/v1",
        "profile": "train_amd64_rtx4080_super",
        "captured_at_utc": "2026-08-03T00:00:00Z",
        "os": {"name": "Ubuntu", "version_id": "22.04"},
        "architecture": "amd64",
        "ros": {"available": True, "distro": "humble"},
        "gpu": {
            "available": True,
            "model": "NVIDIA GeForce RTX 4080 SUPER",
        },
        "cuda": {"available": True, "release": "13.2"},
        "readiness": {"ready": True, "errors": []},
    }


def test_lock_training_stack_is_deterministic_and_portable(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if constraints embedded host paths or changed for one input."""
    fingerprint = tmp_path / "fingerprint.json"
    fingerprint.write_text(
        json.dumps(_training_fingerprint()), encoding="utf-8"
    )
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    script = REPOSITORY_ROOT / "training/tools/lock_training_stack.py"

    for output in (first, second):
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--fingerprint",
                str(fingerprint),
                "--output",
                str(output),
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
        )

    contents = first.read_text(encoding="utf-8")
    assert contents == second.read_text(encoding="utf-8")
    assert "numpy==" in contents
    assert "PyYAML==" in contents
    assert f"torch=={torch.__version__}" in contents
    assert "/home/" not in contents
    assert "/mnt/" not in contents
    assert str(fingerprint) not in contents


def test_lock_training_stack_rejects_unready_fingerprint(
    tmp_path: pathlib.Path,
) -> None:
    """Would fail if an unqualified host could generate approved constraints."""
    payload = _training_fingerprint()
    payload["readiness"] = {"ready": False, "errors": ["gpu mismatch"]}
    fingerprint = tmp_path / "unready.json"
    fingerprint.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "constraints.txt"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "training/tools/lock_training_stack.py"),
            "--fingerprint",
            str(fingerprint),
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "ready" in completed.stderr.lower()
    assert not output.exists()
