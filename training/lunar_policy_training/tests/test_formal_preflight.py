from __future__ import annotations

import json
import hashlib
import pathlib
from dataclasses import replace

import pytest

from lunar_policy_training.checkpoint import RunIdentity
from lunar_policy_training.formal_preflight import (
    REQUIRED_PREFLIGHT_CHECKS,
    FormalPreflightError,
    build_formal_preflight_report,
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


def _resume_equivalence() -> dict[str, object]:
    return {
        "checkpoint_schema": "lunar-ppo-checkpoint/v9",
        "checkpoint_relative_path": "resume-equivalence/update-1.pt",
        "checkpoint_sha256": "f" * 64,
        "checkpoint_roundtrip": True,
        "rollout_exact": True,
        "model_exact": True,
        "optimizer_exact": True,
        "rng_exact": True,
        "environment_state_exact": True,
        "observation_exact": True,
        "candidate_exact": True,
        "first_request_exact": True,
        "uninterrupted_update": 2,
        "resumed_update": 2,
        "evidence_sha256": "e" * 64,
    }


def test_preflight_report_is_canonical_non_proxy_and_records_real_v9_resume(
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
        selected_rollout_horizon=12,
        evaluation_probe_sha256="d" * 64,
        resume_equivalence=_resume_equivalence(),
        additional_corridor_margin_m=2.0,
    )

    first = write_formal_preflight_report(tmp_path, report)
    second = write_formal_preflight_report(tmp_path, report)
    payload = json.loads(first.read_text(encoding="utf-8"))

    assert first == second == tmp_path / "formal-preflight.json"
    assert payload["proxy"] is False
    assert payload["training_started"] is False
    assert payload["selected_workers"] == 24
    assert payload["rollout_horizon_candidates"] == [12]
    assert payload["selected_rollout_horizon"] == 12
    assert payload["episode_decision_limit"] is None
    assert payload["schema_version"] == "lunar-formal-training-preflight/v6"
    assert payload["evaluation_probe_sha256"] == "d" * 64
    assert payload["additional_corridor_margin_m"] == 2.0
    assert "evaluation_report_sha256" not in payload
    assert payload["resume_equivalence"] == _resume_equivalence()
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
            selected_rollout_horizon=12,
            evaluation_probe_sha256="d" * 64,
            resume_equivalence=_resume_equivalence(),
            additional_corridor_margin_m=2.0,
        )


def test_preflight_report_rejects_claimed_resume_without_exact_update_two() -> None:
    resume = _resume_equivalence()
    resume["optimizer_exact"] = False

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
            selected_rollout_horizon=12,
            evaluation_probe_sha256="d" * 64,
            resume_equivalence=resume,
            additional_corridor_margin_m=2.0,
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
            selected_rollout_horizon=12,
            evaluation_probe_sha256="d" * 64,
            resume_equivalence=_resume_equivalence(),
            additional_corridor_margin_m=2.0,
        )


def test_formal_preflight_rejects_checkpoint_v8() -> None:
    resume = _resume_equivalence()
    resume["checkpoint_schema"] = "lunar-ppo-checkpoint/v8"

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
            selected_rollout_horizon=12,
            evaluation_probe_sha256="d" * 64,
            resume_equivalence=resume,
            additional_corridor_margin_m=2.0,
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
            selected_rollout_horizon=12,
            evaluation_probe_sha256="d" * 64,
            resume_equivalence=_resume_equivalence(),
            additional_corridor_margin_m=1.5,
        )
