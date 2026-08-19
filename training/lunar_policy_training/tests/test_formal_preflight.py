from __future__ import annotations

import json
import hashlib
import inspect
import pathlib
from dataclasses import dataclass, replace

import pytest

from lunar_policy_training.checkpoint import RunIdentity
from lunar_policy_training.formal_preflight import (
    REQUIRED_PREFLIGHT_CHECKS,
    FormalPreflightError,
    _direct_environment_checks,
    build_formal_preflight_report,
    run_formal_preflight,
    write_formal_preflight_report,
)
from lunar_policy_training.reward import reward_weights_sha256
from lunar_policy_training.training_semantics import training_semantics_sha256


def _identity() -> RunIdentity:
    return RunIdentity(
        run_kind="formal",
        data_sha256="1" * 64,
        split_sha256="2" * 64,
        generator_sha256="3" * 64,
        capability_sha256="4" * 64,
        reward_sha256=reward_weights_sha256(),
        v3_sha256="6" * 64,
        training_semantics_sha256=training_semantics_sha256(),
    )


def _resume_boundary() -> dict[str, object]:
    return {
        "schema_version": "lunar-formal-preflight-resume-boundary/v1",
        "platforms": ["WHEELED", "LEGGED", "HOPPER"],
        "post_step_observation_sha256": "f" * 64,
        "resumed_observation_sha256": "e" * 64,
        "state_roundtrip": True,
        "evidence_sha256": "e" * 64,
    }


def _task_cache_evidence() -> dict[str, object]:
    item = {
        "task_geometry_sha256": "1" * 64,
        "task_common_key_sha256": "2" * 64,
        "task_common_artifact_sha256": "3" * 64,
        "platform_task_key_sha256": "4" * 64,
        "platform_task_artifact_sha256": "5" * 64,
        "coarse_shape": [64, 64],
        "detail_shape": [1280, 1280],
    }
    return {
        "current_ground": {
            "WHEELED": dict(item),
            "LEGGED": dict(item),
        },
        "next_ground": {
            "WHEELED": dict(item),
            "LEGGED": dict(item),
        },
        "halo_leak_count": 0,
        "full_scene_derived_call_count": 0,
    }


def test_minimal_preflight_contract_does_not_accept_full_pool_or_evaluation_inputs() -> None:
    """Preflight validates wiring, rather than replaying a training update."""
    parameters = inspect.signature(run_formal_preflight).parameters

    assert "evaluation_batches" not in parameters
    assert "worker_candidates" not in parameters


def test_direct_checks_use_exact_common_training_world(monkeypatch) -> None:
    scene_id = "f" * 64
    common_schedule_id = "common/train/v1"
    calls: list[tuple[str, bool, object]] = []

    class Worker:
        episode = type("Episode", (), {"scene_id": scene_id})()

    @dataclass(frozen=True)
    class Builder:
        scenario_schedule_id: str
        paired_evaluation: bool
        platform_scenario_schedule_ids: object

    @dataclass(frozen=True)
    class Factory:
        scenario_schedule_id: str
        builder: Builder

        def create_for_episode(
            self,
            worker_index,
            platform,
            episode_cursor,
            *,
            platform_worker_index,
            platform_worker_count,
        ):
            calls.append(
                (
                    platform,
                    self.builder.paired_evaluation,
                    self.builder.platform_scenario_schedule_ids,
                )
            )
            return Worker()

    factory = Factory(
        scenario_schedule_id="combined/train/v1",
        builder=Builder(
            scenario_schedule_id="combined/train/v1",
            paired_evaluation=False,
            platform_scenario_schedule_ids={
                platform: f"platform/{platform.lower()}"
                for platform in ("WHEELED", "LEGGED", "HOPPER")
            },
        ),
    )
    assembly = type("Assembly", (), {"factory": factory})()
    cache = type(
        "Cache",
        (),
        {
            "manifest": {
                "exact_common_evaluation": {
                    "splits": {
                        "train": {
                            "scenario_schedule_id": common_schedule_id,
                            "scene_ids": [scene_id],
                        }
                    }
                },
                "scenes": [
                    {
                        "split": "train",
                        "platform_coverability": {
                            "HOPPER": {"eligible": True}
                        },
                    }
                ],
            }
        },
    )()

    monkeypatch.setattr(
        "lunar_policy_training.formal_preflight._first_action",
        lambda worker, platform: (_ for _ in ()).throw(
            RuntimeError("past scene check")
        ),
    )
    with pytest.raises(RuntimeError, match="past scene check"):
        _direct_environment_checks(cache, assembly)

    assert calls == [
        ("WHEELED", True, None),
        ("LEGGED", True, None),
        ("HOPPER", True, None),
        ("WHEELED", True, None),
        ("LEGGED", True, None),
    ]
    assert factory.builder.scenario_schedule_id == "combined/train/v1"


def test_preflight_report_is_canonical_and_records_minimal_resume_boundary(
    tmp_path: pathlib.Path,
) -> None:
    report = build_formal_preflight_report(
        source_commit="a" * 40,
        cache_manifest_sha256="b" * 64,
        sensor_performance_sha256="c" * 64,
        run_identity=_identity(),
        scenario_schedule_ids={
            split: f"cache/{split}/v3"
            for split in ("train", "validation", "test", "holdout")
        },
        checks={name: True for name in REQUIRED_PREFLIGHT_CHECKS},
        timings_seconds={"worker_18": 1.0, "worker_24": 1.5},
        qualified_worker_candidates=(18, 24),
        selected_workers=24,
        selected_micro_batch=2,
        selected_rollout_horizon=1,
        three_platform_step_sha256="d" * 64,
        resume_boundary=_resume_boundary(),
        additional_corridor_margin_m=2.0,
        task_cache_evidence=_task_cache_evidence(),
    )

    first = write_formal_preflight_report(tmp_path, report)
    second = write_formal_preflight_report(tmp_path, report)
    payload = json.loads(first.read_text(encoding="utf-8"))

    assert first == second == tmp_path / "formal-preflight.json"
    assert payload["proxy"] is False
    assert payload["training_started"] is False
    assert payload["selected_workers"] == 24
    assert payload["rollout_horizon_candidates"] == [1]
    assert payload["selected_rollout_horizon"] == 1
    assert payload["episode_decision_limit"] is None
    assert payload["schema_version"] == "lunar-formal-training-preflight/v10"
    assert payload["three_platform_step_sha256"] == "d" * 64
    assert payload["additional_corridor_margin_m"] == 2.0
    assert "evaluation_report_sha256" not in payload
    assert payload["resume_boundary"] == _resume_boundary()
    assert len(payload["preflight_report_sha256"]) == 64
    assert not (tmp_path / "checkpoints").exists()


def test_preflight_report_rejects_an_unclosed_required_check() -> None:
    checks = {name: True for name in REQUIRED_PREFLIGHT_CHECKS}
    checks[REQUIRED_PREFLIGHT_CHECKS[0]] = False

    with pytest.raises(FormalPreflightError, match="required checks"):
        build_formal_preflight_report(
            source_commit="a" * 40,
            cache_manifest_sha256="b" * 64,
            sensor_performance_sha256="c" * 64,
            run_identity=_identity(),
            scenario_schedule_ids={
                split: f"cache/{split}/v3"
                for split in ("train", "validation", "test", "holdout")
            },
            checks=checks,
            timings_seconds={},
            qualified_worker_candidates=(18,),
            selected_workers=18,
            selected_micro_batch=1,
            selected_rollout_horizon=1,
            three_platform_step_sha256="d" * 64,
            resume_boundary=_resume_boundary(),
            additional_corridor_margin_m=2.0,
            task_cache_evidence=_task_cache_evidence(),
        )


def test_preflight_report_rejects_resume_boundary_without_state_roundtrip() -> None:
    resume = _resume_boundary()
    resume["state_roundtrip"] = False

    with pytest.raises(FormalPreflightError, match="resume equivalence"):
        build_formal_preflight_report(
            source_commit="a" * 40,
            cache_manifest_sha256="b" * 64,
            sensor_performance_sha256="c" * 64,
            run_identity=_identity(),
            scenario_schedule_ids={
                split: f"cache/{split}/v6"
                for split in ("train", "validation", "test", "holdout")
            },
            checks={name: True for name in REQUIRED_PREFLIGHT_CHECKS},
            timings_seconds={},
            qualified_worker_candidates=(18,),
            selected_workers=18,
            selected_micro_batch=1,
            selected_rollout_horizon=1,
            three_platform_step_sha256="d" * 64,
            resume_boundary=resume,
            additional_corridor_margin_m=2.0,
            task_cache_evidence=_task_cache_evidence(),
        )


def test_formal_preflight_rejects_semantics_v10() -> None:
    legacy = (
        "lunar-training-semantics/"
        "sensor-30m-360-platform-primitive-coverable-detail95-observed-"
        "incremental-primitive-candidates-option-path-observation-auditable-"
        "failure/v10"
    )
    identity = replace(
        _identity(),
        training_semantics_sha256=hashlib.sha256(
            legacy.encode("utf-8")
        ).hexdigest(),
    )

    with pytest.raises(FormalPreflightError, match="semantics"):
        build_formal_preflight_report(
            source_commit="a" * 40,
            cache_manifest_sha256="b" * 64,
            sensor_performance_sha256="c" * 64,
            run_identity=identity,
            scenario_schedule_ids={
                split: f"cache/{split}/v6"
                for split in ("train", "validation", "test", "holdout")
            },
            checks={name: True for name in REQUIRED_PREFLIGHT_CHECKS},
            timings_seconds={},
            qualified_worker_candidates=(18,),
            selected_workers=18,
            selected_micro_batch=1,
            selected_rollout_horizon=1,
            three_platform_step_sha256="d" * 64,
            resume_boundary=_resume_boundary(),
            additional_corridor_margin_m=2.0,
            task_cache_evidence=_task_cache_evidence(),
        )


def test_formal_preflight_rejects_unknown_resume_boundary_schema() -> None:
    resume = _resume_boundary()
    resume["schema_version"] = "lunar-formal-preflight-resume-boundary/v0"

    with pytest.raises(FormalPreflightError, match="resume equivalence"):
        build_formal_preflight_report(
            source_commit="a" * 40,
            cache_manifest_sha256="b" * 64,
            sensor_performance_sha256="c" * 64,
            run_identity=_identity(),
            scenario_schedule_ids={
                split: f"cache/{split}/v6"
                for split in ("train", "validation", "test", "holdout")
            },
            checks={name: True for name in REQUIRED_PREFLIGHT_CHECKS},
            timings_seconds={},
            qualified_worker_candidates=(18,),
            selected_workers=18,
            selected_micro_batch=1,
            selected_rollout_horizon=1,
            three_platform_step_sha256="d" * 64,
            resume_boundary=resume,
            additional_corridor_margin_m=2.0,
            task_cache_evidence=_task_cache_evidence(),
        )


def test_formal_preflight_rejects_non_fixed_corridor_margin() -> None:
    with pytest.raises(FormalPreflightError, match="corridor margin"):
        build_formal_preflight_report(
            source_commit="a" * 40,
            cache_manifest_sha256="b" * 64,
            sensor_performance_sha256="c" * 64,
            run_identity=_identity(),
            scenario_schedule_ids={
                split: f"cache/{split}/v6"
                for split in ("train", "validation", "test", "holdout")
            },
            checks={name: True for name in REQUIRED_PREFLIGHT_CHECKS},
            timings_seconds={},
            qualified_worker_candidates=(18,),
            selected_workers=18,
            selected_micro_batch=1,
            selected_rollout_horizon=1,
            three_platform_step_sha256="d" * 64,
            resume_boundary=_resume_boundary(),
            additional_corridor_margin_m=1.5,
            task_cache_evidence=_task_cache_evidence(),
        )
