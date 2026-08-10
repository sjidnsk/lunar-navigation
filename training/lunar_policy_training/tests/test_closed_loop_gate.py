from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

import lunar_policy_training.closed_loop_gate as gate_module
from lunar_policy_training.closed_loop_gate import (
    CLOSED_LOOP_GATE_SCHEMA,
    CLOSED_LOOP_MINIMUM_SCENES,
    ClosedLoopGateError,
    _validate_bound_identities,
    build_closed_loop_gate_report,
    select_closed_loop_gate_cases,
    write_closed_loop_gate_report,
)


PLATFORMS = ("WHEELED", "LEGGED", "HOPPER")


def _sha(character: str) -> str:
    return character * 64


def _scene_id(index: int) -> str:
    return f"{index:064x}"


def _coverability() -> dict[str, object]:
    return {
        "exact": True,
        "eligible": True,
        "mission_coverable_fraction": 0.97,
        "reachable_mask_sha256": _sha("a"),
        "coverable_mask_sha256": _sha("b"),
    }


def _cache_documents(scene_count: int = CLOSED_LOOP_MINIMUM_SCENES):
    scenes = []
    scenarios = []
    ids = []
    for index in range(scene_count):
        scene_id = _scene_id(index + 1)
        ids.append(scene_id)
        scenes.append(
            {
                "scene_id": scene_id,
                "split": "train",
                "platform_coverability": {
                    platform: _coverability() for platform in PLATFORMS
                },
            }
        )
        scenarios.append(
            {
                "scene_id": scene_id,
                "split": "train",
                "scenario_seed": 1000 + index,
                "scene_seed": _sha("c"),
            }
        )
    manifest = {
        "schema": "lunar-formal-training-cache/v4",
        "cache_manifest_sha256": _sha("d"),
        "scenes": scenes,
        "exact_common_evaluation": {
            "scene_count": scene_count,
            "scene_ids": ids,
            "splits": {
                split: {
                    "scene_count": scene_count if split == "train" else 0,
                    "scene_ids": ids if split == "train" else [],
                    "scenario_schedule_id": f"common/{split}/v1",
                }
                for split in ("train", "validation", "test", "holdout")
            },
        },
    }
    return manifest, {"scenarios": scenarios}


def _passing_rows(cases) -> list[dict[str, object]]:
    return [
        {
            "scene_id": case.scene_id,
            "split": case.split,
            "platform": platform,
            "exact": True,
            "mission_coverable_fraction_hex": float(0.97).hex(),
            "reachable_mask_sha256": _sha("a"),
            "coverable_mask_sha256": _sha("b"),
            "final_coverage_hex": float(0.95).hex(),
            "success_first_crossing": True,
            "terminal_reason": "SUCCESS",
            "oracle_contradiction_count": 0,
            "oracle_opportunity_count": 0,
            "planner_failure_count": 0,
            "safety_violation_count": 0,
            "invalid_action_count": 0,
            "execution_failure_count": 0,
            "executed_step_count": 11,
            "candidate_sequence_sha256": _sha("e"),
            "request_sequence_sha256": _sha("f"),
            "planner_sequence_sha256": _sha("1"),
        }
        for case in cases
        for platform in PLATFORMS
    ]


def test_select_closed_loop_gate_cases_requires_one_exact_common_scene() -> None:
    manifest, scenario_document = _cache_documents(
        CLOSED_LOOP_MINIMUM_SCENES - 1
    )

    with pytest.raises(ClosedLoopGateError, match="at least 1"):
        select_closed_loop_gate_cases(manifest, scenario_document)


def test_gate_keeps_full_source_head_distinct_from_v3_cache_commit() -> None:
    expected = {
        "v3_source_commit": "9" * 40,
        "v3_sha256": _sha("1"),
    }
    identity = SimpleNamespace(**expected)

    _validate_bound_identities(
        source_commit="0" * 40,
        head_commit="0" * 40,
        cache_identity=identity,
        expected_cache_identity=expected,
    )

    with pytest.raises(ClosedLoopGateError, match="differs from HEAD"):
        _validate_bound_identities(
            source_commit="8" * 40,
            head_commit="0" * 40,
            cache_identity=identity,
            expected_cache_identity=expected,
        )


def test_checked_gate_worker_identifies_the_failed_scene_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = SimpleNamespace(
        case=SimpleNamespace(scene_id=_scene_id(99)),
        platform="HOPPER",
    )
    monkeypatch.setattr(
        gate_module,
        "_run_closed_loop_work",
        lambda _work: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(
        ClosedLoopGateError,
        match=f"{_scene_id(99)}/HOPPER: boom",
    ):
        gate_module._run_closed_loop_work_checked(work)


def test_select_closed_loop_gate_cases_binds_schedule_cursor_and_coverability() -> None:
    manifest, scenario_document = _cache_documents()

    first = select_closed_loop_gate_cases(manifest, scenario_document)
    second = select_closed_loop_gate_cases(manifest, scenario_document)

    assert first == second
    assert len(first) == CLOSED_LOOP_MINIMUM_SCENES
    assert {case.scene_id for case in first} == {
        entry["scene_id"] for entry in manifest["scenes"]
    }
    assert {case.split for case in first} == {"train"}
    assert sorted(case.episode_cursor for case in first) == list(
        range(CLOSED_LOOP_MINIMUM_SCENES)
    )
    assert all(case.scenario_schedule_id == "common/train/v1" for case in first)


def test_closed_loop_gate_report_is_canonical_and_repeat_comparable(
    tmp_path: pathlib.Path,
) -> None:
    manifest, scenario_document = _cache_documents()
    cases = select_closed_loop_gate_cases(manifest, scenario_document)
    rows = _passing_rows(cases)

    first = build_closed_loop_gate_report(
        source_commit="9" * 40,
        cache_manifest_sha256=_sha("d"),
        cases=cases,
        rows=rows,
        timings_seconds={"closed_loop": 12.5},
    )
    second = build_closed_loop_gate_report(
        source_commit="9" * 40,
        cache_manifest_sha256=_sha("d"),
        cases=tuple(reversed(cases)),
        rows=tuple(reversed(rows)),
        timings_seconds={"closed_loop": 19.0},
    )

    assert first.payload["schema_version"] == CLOSED_LOOP_GATE_SCHEMA
    assert first.payload["scene_count"] == 1
    assert first.payload["scene_platform_count"] == 3
    assert first.payload["successful_scene_platform_count"] == 3
    assert first.payload["natural_failure_scene_platform_count"] == 0
    assert first.payload["passed"] is True
    assert first.payload["closed_loop_evidence_sha256"] == second.payload[
        "closed_loop_evidence_sha256"
    ]
    path = write_closed_loop_gate_report(tmp_path, first)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path == tmp_path / "closed-loop-gate.json"
    assert len(payload["closed_loop_report_sha256"]) == 64


@pytest.mark.parametrize(
    ("terminal_reason", "oracle_opportunity_count"),
    (
        ("NO_FRONTIER_ANCHOR", 0),
        ("VISITED_EXHAUSTED", 0),
        ("PLATFORM_UNREACHABLE", 0),
        ("ZERO_GAIN", 0),
        ("PLANNER_REJECTED_ALL", 3),
    ),
)
def test_closed_loop_gate_accepts_auditable_failure_below_success_threshold(
    terminal_reason: str,
    oracle_opportunity_count: int,
) -> None:
    manifest, scenario_document = _cache_documents()
    cases = select_closed_loop_gate_cases(manifest, scenario_document)
    rows = _passing_rows(cases)
    rows[0].update(
        final_coverage_hex=float(0.31).hex(),
        success_first_crossing=False,
        terminal_reason=terminal_reason,
        oracle_opportunity_count=oracle_opportunity_count,
        planner_failure_count=2,
    )

    report = build_closed_loop_gate_report(
        source_commit="9" * 40,
        cache_manifest_sha256=_sha("d"),
        cases=cases,
        rows=rows,
        timings_seconds={},
    )

    assert report.payload["passed"] is True


@pytest.mark.parametrize(
    "field,value,message",
    (
        ("exact", False, "exact"),
        (
            "mission_coverable_fraction_hex",
            float(0.94).hex(),
            "mission coverable",
        ),
        ("oracle_contradiction_count", 1, "oracle contradiction"),
        ("safety_violation_count", 1, "safety"),
        ("invalid_action_count", 1, "invalid action"),
        ("execution_failure_count", 1, "execution failure"),
    ),
)
def test_closed_loop_gate_report_rejects_a_failed_scene_platform(
    field: str,
    value: object,
    message: str,
) -> None:
    manifest, scenario_document = _cache_documents()
    cases = select_closed_loop_gate_cases(manifest, scenario_document)
    rows = _passing_rows(cases)
    rows[0][field] = value

    with pytest.raises(ClosedLoopGateError, match=message):
        build_closed_loop_gate_report(
            source_commit="9" * 40,
            cache_manifest_sha256=_sha("d"),
            cases=cases,
            rows=rows,
            timings_seconds={},
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        (
            {
                "final_coverage_hex": float(0.949999).hex(),
                "success_first_crossing": True,
                "terminal_reason": "SUCCESS",
            },
            "successful coverage",
        ),
        (
            {
                "final_coverage_hex": float(0.95).hex(),
                "success_first_crossing": False,
                "terminal_reason": "ZERO_GAIN",
            },
            "failure coverage",
        ),
        (
            {
                "final_coverage_hex": float(0.31).hex(),
                "success_first_crossing": False,
                "terminal_reason": "HARD_FAILURE",
            },
            "terminal reason",
        ),
        (
            {
                "final_coverage_hex": float(0.31).hex(),
                "success_first_crossing": False,
                "terminal_reason": "ZERO_GAIN",
                "oracle_opportunity_count": 1,
            },
            "terminal oracle opportunity",
        ),
        (
            {
                "final_coverage_hex": float(0.31).hex(),
                "success_first_crossing": False,
                "terminal_reason": "PLANNER_REJECTED_ALL",
                "planner_failure_count": 11,
            },
            "successful execution",
        ),
    ),
)
def test_closed_loop_gate_rejects_inconsistent_terminal_evidence(
    updates: dict[str, object],
    message: str,
) -> None:
    manifest, scenario_document = _cache_documents()
    cases = select_closed_loop_gate_cases(manifest, scenario_document)
    rows = _passing_rows(cases)
    rows[0].update(updates)

    with pytest.raises(ClosedLoopGateError, match=message):
        build_closed_loop_gate_report(
            source_commit="9" * 40,
            cache_manifest_sha256=_sha("d"),
            cases=cases,
            rows=rows,
            timings_seconds={},
        )
