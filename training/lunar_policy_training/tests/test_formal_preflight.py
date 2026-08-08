from __future__ import annotations

import json
import pathlib

import pytest

from lunar_policy_training.checkpoint import RunIdentity
from lunar_policy_training.formal_preflight import (
    REQUIRED_PREFLIGHT_CHECKS,
    FormalPreflightError,
    build_formal_preflight_report,
    write_formal_preflight_report,
)
from lunar_policy_training.reward import reward_weights_sha256


def _identity() -> RunIdentity:
    return RunIdentity(
        run_kind="formal",
        data_sha256="1" * 64,
        split_sha256="2" * 64,
        generator_sha256="3" * 64,
        capability_sha256="4" * 64,
        reward_sha256=reward_weights_sha256(),
        v3_sha256="6" * 64,
        training_semantics_sha256="7" * 64,
    )


def _resume_equivalence() -> dict[str, object]:
    return {
        "checkpoint_schema": "lunar-ppo-checkpoint/v6",
        "checkpoint_relative_path": "resume-equivalence/update-1.pt",
        "checkpoint_sha256": "f" * 64,
        "checkpoint_roundtrip": True,
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


def test_preflight_report_is_canonical_non_proxy_and_records_real_v6_resume(
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
        selected_rollout_horizon=32,
        evaluation_report_sha256="d" * 64,
        resume_equivalence=_resume_equivalence(),
    )

    first = write_formal_preflight_report(tmp_path, report)
    second = write_formal_preflight_report(tmp_path, report)
    payload = json.loads(first.read_text(encoding="utf-8"))

    assert first == second == tmp_path / "formal-preflight.json"
    assert payload["proxy"] is False
    assert payload["training_started"] is False
    assert payload["selected_workers"] == 24
    assert payload["rollout_horizon_candidates"] == [16, 32, 64]
    assert payload["selected_rollout_horizon"] == 32
    assert payload["episode_decision_limit"] is None
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
            selected_rollout_horizon=32,
            evaluation_report_sha256="d" * 64,
            resume_equivalence=_resume_equivalence(),
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
            selected_rollout_horizon=32,
            evaluation_report_sha256="d" * 64,
            resume_equivalence=resume,
        )
