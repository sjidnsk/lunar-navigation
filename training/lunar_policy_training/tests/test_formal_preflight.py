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


def test_preflight_report_is_canonical_non_proxy_and_never_a_checkpoint(
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
        evaluation_report_sha256="d" * 64,
    )

    first = write_formal_preflight_report(tmp_path, report)
    second = write_formal_preflight_report(tmp_path, report)
    payload = json.loads(first.read_text(encoding="utf-8"))

    assert first == second == tmp_path / "formal-preflight.json"
    assert payload["proxy"] is False
    assert payload["training_started"] is False
    assert payload["selected_workers"] == 24
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
            evaluation_report_sha256="d" * 64,
        )
