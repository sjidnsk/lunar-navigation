from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lunar_policy_training.closed_loop_gate as gate_module
import lunar_policy_training.environment.formal_builder as formal_builder_module
import lunar_policy_training.evaluation.report as report_module
from lunar_policy_training.closed_loop_gate import (
    CLOSED_LOOP_GATE_SCHEMA,
    CLOSED_LOOP_MINIMUM_SCENES,
    ClosedLoopGateReport,
    ClosedLoopGateError,
    _validate_bound_identities,
    build_closed_loop_gate_report,
    select_closed_loop_gate_cases,
    write_closed_loop_gate_report,
)
from lunar_policy_training.environment.candidate_builder import CandidateDiagnostics
from lunar_policy_training.environment.macro_step import (
    ExecutionEvents,
    TerminalReason,
)


PLATFORMS = ("WHEELED", "LEGGED", "HOPPER")

CANDIDATE_DIAGNOSTICS = {
    "physical_snapshot_id": "3" * 64,
    "physical_reachability_algorithm_id": "lunar-physical-reachability/test-v1",
    "physical_candidate_universe_count": 1,
    "selected_policy_candidate_count": 1,
    "available_candidate_count": 1,
    "untried_reserve_count": 0,
    "planner_failed_current_snapshot_count": 0,
    "zero_gain_count": 0,
    "visited_excluded_count": 0,
    "physical_unreachable_count": 0,
}


def _sha(character: str) -> str:
    return character * 64


def _scene_id(index: int) -> str:
    return f"{index:064x}"


def _coverability() -> dict[str, object]:
    return {
        "exact": True,
        "eligible": True,
        "mission_coverable_fraction": 0.97,
        "physical_projection_sha256": _sha("a"),
        "coverable_detail_mask_sha256": _sha("b"),
        "physical_reachability_algorithm_id": (
            "lunar-physical-reachability/test-v1"
        ),
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
        "schema": "lunar-formal-training-cache/v7",
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
            "physical_projection_sha256": _sha("a"),
            "coverable_mask_sha256": _sha("b"),
            "physical_reachability_algorithm_id": (
                "lunar-physical-reachability/test-v1"
            ),
            "physical_candidate_universe_sha256": _sha("2"),
            "physical_snapshot_id": _sha("3"),
            "oracle_opportunity_set_sha256": _sha("4"),
            "final_coverage_hex": float(0.95).hex(),
            "success_first_crossing": True,
            "terminal_reason": "SUCCESS",
            "oracle_contradiction_count": 0,
            "oracle_opportunity_count": 0,
            "planner_failure_count": 0,
            "safety_violation_count": 0,
            "invalid_action_count": 0,
            "platform_reference_mismatch_count": 0,
            "execution_failure_count": 0,
            "executed_step_count": 11,
            "planner_call_count": 11,
            "request_sequence_sha256": _sha("f"),
            "planner_sequence_sha256": _sha("1"),
            "additional_corridor_margin_m": (
                0.0 if platform == "HOPPER" else 2.0
            ),
            "search_domain_cell_count": 0 if platform == "HOPPER" else 37,
            "search_domain_sha256": "" if platform == "HOPPER" else _sha("5"),
            "planner_reason_counts": {"OK": 11},
            "candidate_diagnostics": dict(CANDIDATE_DIAGNOSTICS),
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


def test_gate_binds_domain_evidence_to_the_actual_stateful_planner_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, scenario_document = _cache_documents()
    case = select_closed_loop_gate_cases(manifest, scenario_document)[0]
    observation = SimpleNamespace(
        frontier_features=torch.zeros((1, 1, 4), dtype=torch.float32),
        candidate_mask=torch.ones((1, 1), dtype=torch.bool),
        observation_identities=(object(),),
    )

    class StatefulPlanner:
        def __init__(self) -> None:
            self.call_count = 0

        def plan(self, _request: object) -> SimpleNamespace:
            self.call_count += 1
            reason = (
                "FIRST_ACTUAL" if self.call_count == 1 else "SECOND_EXECUTION"
            )
            hierarchical = SimpleNamespace(
                additional_corridor_margin_m=2.0,
                search_domain_cell_count=37,
                search_domain_sha256=_sha(str(self.call_count)),
            )
            return SimpleNamespace(
                outcome=SimpleNamespace(name="NEW_REFERENCE_AVAILABLE"),
                directive=SimpleNamespace(name="EXECUTE_REFERENCE"),
                reason_code=reason,
                diagnostics=SimpleNamespace(hierarchical=hierarchical),
            )

    planner = StatefulPlanner()
    events = ExecutionEvents(selected_action_observed_safe=True)

    class FakeEnvironment:
        def __init__(self) -> None:
            self._bridge = planner
            self.current_observation = observation

        def refresh_decision_boundary(self) -> SimpleNamespace:
            return SimpleNamespace(execution_state="DECISION_READY")

        def advance_prepared_action(self, *_args, **_kwargs) -> SimpleNamespace:
            output = self._bridge.plan(object())
            transition = SimpleNamespace(
                planning_outcome=output.outcome,
                execution_directive=output.directive,
                reason_code=output.reason_code,
                execution_events=events,
                next_observation=observation,
                success_first_crossing=True,
                terminated=True,
                terminal_reason=TerminalReason.SUCCESS,
                oracle_opportunity_count=0,
            )
            return SimpleNamespace(
                transition=transition,
                policy_decisions_consumed=1,
            )

    snapshot = SimpleNamespace(
        candidate_universe_sha256=_sha("2"),
        candidate_universe=SimpleNamespace(physical_snapshot_id=_sha("3")),
        frontier_oracle=SimpleNamespace(
            oracle_opportunity_set_sha256=_sha("4")
        ),
    )
    worker = SimpleNamespace(
        episode=SimpleNamespace(
            scene_id=case.scene_id,
            _snapshot=snapshot,
            build_request=lambda *_args: SimpleNamespace(request=object()),
        ),
        environment=FakeEnvironment(),
        current_candidate_diagnostics=lambda: CandidateDiagnostics(
            physical_snapshot_id=_sha("3"),
            physical_reachability_algorithm_id=(
                "lunar-physical-reachability/test-v1"
            ),
            physical_candidate_universe_count=1,
            selected_policy_candidate_count=1,
            available_candidate_count=1,
        ),
    )
    factory = SimpleNamespace(
        create_for_episode=lambda *_args, **_kwargs: worker
    )
    monkeypatch.setattr(gate_module, "load_project_formal_capability", lambda *_: object())
    monkeypatch.setattr(
        formal_builder_module,
        "FormalWorkerBuilder",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        gate_module,
        "FrozenCapabilityEnvironmentFactory",
        lambda **_kwargs: factory,
    )
    monkeypatch.setattr(
        gate_module,
        "select_baseline_action",
        lambda *_args, **_kwargs: SimpleNamespace(candidate_index=0, theta=0.0),
    )
    monkeypatch.setattr(gate_module, "_request_signature", lambda _request: _sha("8"))
    monkeypatch.setattr(
        report_module,
        "mission_coverage_ratio",
        lambda _observation: np.asarray([0.95], dtype=np.float64),
    )
    monkeypatch.setattr(gate_module, "PlannerBridge", lambda: planner, raising=False)
    work = SimpleNamespace(
        cache_manifest_path="/tmp/cache-manifest.json",
        repository_root="/tmp/repository",
        case=case,
        platform="WHEELED",
        watchdog_max_steps=2,
        watchdog_seconds=5.0,
    )

    row = gate_module._run_closed_loop_work(work)

    assert planner.call_count == 1
    assert row["planner_reason_counts"] == {"FIRST_ACTUAL": 1}
    assert row["search_domain_sha256"] == _sha("1")


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
    assert all(
        binding.physical_projection_sha256 == _sha("a")
        for case in first
        for binding in case.coverability
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("physical_reachability_algorithm_id", "", "physical identity"),
        ("physical_projection_sha256", "invalid", "physical identity"),
        ("coverable_detail_mask_sha256", "invalid", "physical identity"),
    ),
)
def test_select_closed_loop_gate_cases_rejects_incomplete_physical_identity(
    field: str,
    value: object,
    message: str,
) -> None:
    manifest, scenario_document = _cache_documents()
    manifest["scenes"][0]["platform_coverability"]["WHEELED"][field] = value

    with pytest.raises(ClosedLoopGateError, match=message):
        select_closed_loop_gate_cases(manifest, scenario_document)


def test_select_closed_loop_gate_cases_rejects_cache_v6() -> None:
    manifest, scenario_document = _cache_documents()
    manifest["schema"] = "lunar-formal-training-cache/v6"

    with pytest.raises(ClosedLoopGateError, match="current formal cache"):
        select_closed_loop_gate_cases(manifest, scenario_document)


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
    assert first.payload["terminal_reason_counts"] == {"SUCCESS": 3}
    assert first.payload["planner_blocked_scene_platform_count"] == 0
    assert first.payload["hard_failure_scene_platform_count"] == 0
    assert first.payload["canceled_scene_platform_count"] == 0
    assert first.payload["platform_reference_mismatch_count"] == 0
    assert first.payload["passed"] is True
    assert first.payload["closed_loop_evidence_sha256"] == second.payload[
        "closed_loop_evidence_sha256"
    ]
    path = write_closed_loop_gate_report(tmp_path, first)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path == tmp_path / "closed-loop-gate.json"
    assert len(payload["closed_loop_report_sha256"]) == 64


def test_closed_loop_gate_report_accepts_zero_hopper_corridor_margin() -> None:
    manifest, scenario_document = _cache_documents()
    cases = select_closed_loop_gate_cases(manifest, scenario_document)
    rows = _passing_rows(cases)
    hopper = next(row for row in rows if row["platform"] == "HOPPER")

    assert hopper["additional_corridor_margin_m"] == 0.0
    report = build_closed_loop_gate_report(
        source_commit="9" * 40,
        cache_manifest_sha256=_sha("d"),
        cases=cases,
        rows=rows,
        timings_seconds={},
    )

    assert report.payload["passed"] is True


def test_closed_loop_gate_report_rejects_nonzero_hopper_corridor_margin() -> None:
    manifest, scenario_document = _cache_documents()
    cases = select_closed_loop_gate_cases(manifest, scenario_document)
    rows = _passing_rows(cases)
    hopper = next(row for row in rows if row["platform"] == "HOPPER")
    hopper["additional_corridor_margin_m"] = 2.0

    with pytest.raises(ClosedLoopGateError, match="hopper corridor margin"):
        build_closed_loop_gate_report(
            source_commit="9" * 40,
            cache_manifest_sha256=_sha("d"),
            cases=cases,
            rows=rows,
            timings_seconds={},
        )


def test_closed_loop_gate_report_rejects_fake_hopper_search_domain() -> None:
    manifest, scenario_document = _cache_documents()
    cases = select_closed_loop_gate_cases(manifest, scenario_document)
    rows = _passing_rows(cases)
    hopper = next(row for row in rows if row["platform"] == "HOPPER")
    hopper["search_domain_cell_count"] = 37
    hopper["search_domain_sha256"] = _sha("5")

    with pytest.raises(ClosedLoopGateError, match="hopper search domain"):
        build_closed_loop_gate_report(
            source_commit="9" * 40,
            cache_manifest_sha256=_sha("d"),
            cases=cases,
            rows=rows,
            timings_seconds={},
        )


def test_closed_loop_gate_rejects_v3_report(tmp_path: pathlib.Path) -> None:
    manifest, scenario_document = _cache_documents()
    cases = select_closed_loop_gate_cases(manifest, scenario_document)
    current = build_closed_loop_gate_report(
        source_commit="9" * 40,
        cache_manifest_sha256=_sha("d"),
        cases=cases,
        rows=_passing_rows(cases),
        timings_seconds={},
    )
    legacy = ClosedLoopGateReport(
        {
            **current.payload,
            "schema_version": "lunar-platform-coverable-closed-loop-gate/v3",
        }
    )

    with pytest.raises(ClosedLoopGateError, match="schema"):
        write_closed_loop_gate_report(tmp_path, legacy)


def test_closed_loop_gate_rejects_primitive_bound_report() -> None:
    manifest, scenario_document = _cache_documents()
    cases = select_closed_loop_gate_cases(manifest, scenario_document)
    rows = _passing_rows(cases)
    rows[0]["primitive_set_sha256"] = _sha("9")

    with pytest.raises(ClosedLoopGateError, match="primitive"):
        build_closed_loop_gate_report(
            source_commit="9" * 40,
            cache_manifest_sha256=_sha("d"),
            cases=cases,
            rows=rows,
            timings_seconds={},
        )


@pytest.mark.parametrize(
    ("terminal_reason", "oracle_opportunity_count"),
    (
        ("NO_RECOVERABLE_OBSERVATION_STATE", 0),
        ("VISITED_EXHAUSTED", 0),
        ("NO_TRANSIT_OPPORTUNITY", 0),
        ("ZERO_GAIN", 0),
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
        ("physical_projection_sha256", _sha("6"), "physical identity differs"),
        ("coverable_mask_sha256", _sha("7"), "physical identity differs"),
        ("additional_corridor_margin_m", 1.5, "corridor margin"),
        ("search_domain_cell_count", 0, "search domain"),
        ("search_domain_sha256", "", "search domain"),
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
            "hard failure",
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
                "terminal_reason": "PLANNER_BLOCKED_WITH_OPPORTUNITY",
                "oracle_opportunity_count": 3,
                "planner_failure_count": 11,
            },
            "planner blocked",
        ),
        (
            {
                "final_coverage_hex": float(0.31).hex(),
                "success_first_crossing": False,
                "terminal_reason": "CANCELED",
            },
            "canceled",
        ),
        (
            {"platform_reference_mismatch_count": 1},
            "reference mismatch",
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
