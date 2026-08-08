from __future__ import annotations

import pathlib
import os
import signal
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

import lunar_policy_training.cli as cli_module


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "model_contract"))

from lunar_policy_training.budget import TrainingBudget  # noqa: E402
from lunar_policy_training.cli import (  # noqa: E402
    ArtifactRootError,
    PreflightError,
    SignalStopFlag,
    TrainingBoundaryLoop,
    _checkpoint_target,
    _formal_sensor_performance_preflight,
    _formal_run_identity,
    _freeze_task4_manifest,
    _freeze_formal_environment_manifest,
    _load_calibrated_run_state,
    _run_curriculum_training,
    _update_run_manifest,
    build_parser,
    validate_artifact_root,
)
from lunar_policy_training.capability_freeze import FrozenCapabilityBundle  # noqa: E402
from lunar_policy_training.config import (
    TrainingConfigError,
    load_training_config,
    resolve_training_config,
)
from lunar_policy_training.checkpoint import RunIdentity
from lunar_policy_training.curriculum import CurriculumSchedule
from lunar_policy_training.polar_data.formal_cache import FormalCacheIdentity
from lunar_policy_training.evaluation.release_gate import (
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
from lunar_policy_training.proxy_scenario import proxy_observation
from lunar_policy_training.reward import reward_weights_sha256


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
            "--sensor-performance-report",
            "/tmp/sensor-performance.json",
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
    assert resume.command == "resume"
    assert resume.sensor_performance_report is None
    assert evaluate.command == "evaluate"
    assert formal_preflight.command == "formal-preflight"
    assert formal_preflight.cache_manifest.endswith("cache-manifest.json")
    assert evaluate.sensor_performance_report is None
    assert extension.command == "extend-budget"
    assert extension.blocks == 2


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
                {"scene_id": "c" * 64, "split": "holdout"},
                {"scene_id": "a" * 64, "split": "validation"},
                {"scene_id": "b" * 64, "split": "test"},
            ]
        },
    )
    template = proxy_observation(0, "WHEELED", step=0)
    assemblies = {
        split: SimpleNamespace(
            factory=SimpleNamespace(scenario_schedule_id=f"cache/{split}/v3"),
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
    assert batches[0].scenario_seeds == (409000,)
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
    development = _perfect_development_report()
    report = EvaluationReport(
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
    gate_result = evaluate_release_gate(
        report,
        load_gate_rules(REPOSITORY_ROOT / "training/configs/release_gate_v1.yaml"),
    )

    digest = cli_module._write_formal_evaluation_artifacts(
        root=root,
        checkpoint_path=root / "checkpoints/latest.pt",
        report=report,
        gate_result=gate_result,
    )

    result = json.loads(
        (root / "evaluation/formal-result.json").read_text(encoding="utf-8")
    )
    assert (root / "evaluation/formal-report.json").is_file()
    assert result["report_sha256"] == digest
    assert result["run_kind"] == "formal"
    assert result["proxy"] is False
    assert result["formal_candidate_eligible"] is True
    manifest = json.loads(
        (root / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["last_evaluation"]["report_sha256"] == digest


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
                    "compared_workers": [18, 24],
                    "measurements": [],
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
        reward_hash=reward_weights_sha256(),
        reward_seed_results=(
            {"seed": 4081, "minimum_platform_score": 0.5, "report_sha256": "1" * 64},
            {"seed": 4082, "minimum_platform_score": 0.5, "report_sha256": "2" * 64},
            {"seed": 4083, "minimum_platform_score": 0.5, "report_sha256": "3" * 64},
        ),
    )


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
                    "selected_workers": 24,
                    "selected_micro_batch": 2,
                    "compared_workers": [18, 24],
                    "measurements": [],
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


def test_training_does_not_start_callbacks_below_bounded_unit_reserve() -> None:
    """Would fail if the final partial budget launched another rollout/update."""
    budget = TrainingBudget(consumed_gpu_seconds=86400.0 - 600.0 + 1.0)
    events: list[str] = []
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="joint",
        initial_global_step=41,
        initial_latest_checkpoint_gpu_seconds=84000.0,
        initial_candidate_checkpoint_gpu_seconds=82800.0,
        clock=lambda: 10.0,
    )

    state = loop.run(
        collect_rollout=lambda: events.append("collect"),
        update_rollout=lambda rollout: events.append("update"),
        save_checkpoint=lambda kind, saved_state: events.append(
            f"save-{kind}-step-{saved_state.global_step}"
        ),
        max_updates=1,
    )

    assert events == ["save-latest-step-41"]
    assert state.global_step == 41
    assert state.rollout_discarded is False
    assert state.latest_checkpoint_gpu_seconds == 86400.0
    assert budget.consumed_gpu_seconds == 86400.0
    assert budget.interval_active is False


def test_curriculum_phase_returns_before_next_update_could_cross_boundary() -> None:
    """Would fail if a warmup could consume time reserved for its next phase."""
    budget = TrainingBudget(consumed_gpu_seconds=6599.0)
    events: list[str] = []
    loop = TrainingBoundaryLoop(
        budget=budget,
        stop_flag=SignalStopFlag(),
        checkpoint_interval_seconds=1800,
        candidate_checkpoint_interval_seconds=3600,
        curriculum_phase="warmup_wheeled",
        phase_end_gpu_seconds=7199.0,
        clock=lambda: 10.0,
    )

    state = loop.run(
        collect_rollout=lambda: events.append("collect"),
        update_rollout=lambda rollout: events.append("update"),
        save_checkpoint=lambda kind, saved: events.append(
            f"save-{kind}-step-{saved.global_step}"
        ),
        max_updates=10,
    )

    assert events == ["save-latest-step-0"]
    assert state.global_step == 0
    assert budget.consumed_gpu_seconds == 6599.0


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


def test_training_overrun_saves_terminal_latest_after_complete_update() -> None:
    timestamps = iter((0.0, 600.001))
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
