"""Deterministic 24-scene x three-platform exploration qualification gate."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Mapping, Sequence

import numpy as np
import torch

from .capability_freeze import FrozenCapabilityEnvironmentFactory
from .config import PLATFORMS
from .eval.baselines import select_baseline_action
from .polar_data.multires_scene import GENERATOR_SHA256
from .project_capability import load_project_formal_capability
from .training_semantics import (
    FORMAL_SUCCESS_COVERAGE_RATIO,
    training_semantics_sha256,
)


CLOSED_LOOP_GATE_SCHEMA = "lunar-platform-coverable-closed-loop-gate/v1"
CLOSED_LOOP_MINIMUM_SCENES = 24
CLOSED_LOOP_METHOD = "gain_over_cost_frontier"
_SPLITS = ("train", "validation", "test", "holdout")
_ROW_SHA_FIELDS = (
    "reachable_mask_sha256",
    "coverable_mask_sha256",
    "candidate_sequence_sha256",
    "request_sequence_sha256",
    "planner_sequence_sha256",
)


class ClosedLoopGateError(RuntimeError):
    """The fixed-scene gate cannot prove the formal exploration closure."""


@dataclass(frozen=True, slots=True)
class PlatformCoverabilityBinding:
    platform: str
    mission_coverable_fraction: float
    reachable_mask_sha256: str
    coverable_mask_sha256: str


@dataclass(frozen=True, slots=True)
class ClosedLoopGateCase:
    scene_id: str
    split: str
    scenario_seed: int
    episode_cursor: int
    scenario_schedule_id: str
    coverability: tuple[PlatformCoverabilityBinding, ...]

    def for_platform(self, platform: str) -> PlatformCoverabilityBinding:
        for binding in self.coverability:
            if binding.platform == platform:
                return binding
        raise ClosedLoopGateError("gate case has no platform coverability")


@dataclass(frozen=True, slots=True)
class ClosedLoopGateReport:
    payload: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class _ClosedLoopWork:
    cache_manifest_path: str
    repository_root: str
    case: ClosedLoopGateCase
    platform: str
    watchdog_max_steps: int
    watchdog_seconds: float


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _semantic_sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _is_sha(value: object, *, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _scenario_seed(value: Mapping[str, object]) -> int:
    scenario_seed = value.get("scenario_seed")
    if type(scenario_seed) is int:
        return scenario_seed
    scene_seed = value.get("scene_seed")
    if not isinstance(scene_seed, str) or not _is_sha(scene_seed):
        raise ClosedLoopGateError("gate scenario seed is invalid")
    return int(scene_seed[:16], 16)


def _scheduled_entries(
    entries: Sequence[Mapping[str, object]],
    *,
    scenario_schedule_id: str,
) -> tuple[Mapping[str, object], ...]:
    """Mirror the formal builder's content-addressed stable permutation."""
    if not isinstance(scenario_schedule_id, str) or not scenario_schedule_id:
        raise ClosedLoopGateError("gate scenario schedule identity is missing")
    scene_ids = [entry.get("scene_id") for entry in entries]
    if any(not _is_sha(scene_id) for scene_id in scene_ids) or len(
        set(scene_ids)
    ) != len(scene_ids):
        raise ClosedLoopGateError("gate scheduled scene IDs are invalid")

    def key(entry: Mapping[str, object]) -> tuple[str, str]:
        scene_id = str(entry["scene_id"])
        digest = hashlib.sha256(
            f"{scenario_schedule_id}\0scene\0{scene_id}".encode("utf-8")
        ).hexdigest()
        return digest, scene_id

    return tuple(sorted(entries, key=key))


def _coverability_bindings(
    entry: Mapping[str, object],
) -> tuple[PlatformCoverabilityBinding, ...]:
    raw = entry.get("platform_coverability")
    if not isinstance(raw, Mapping) or set(raw) != set(PLATFORMS):
        raise ClosedLoopGateError("gate platform coverability is incomplete")
    bindings: list[PlatformCoverabilityBinding] = []
    for platform in PLATFORMS:
        value = raw[platform]
        if not isinstance(value, Mapping):
            raise ClosedLoopGateError("gate platform coverability is invalid")
        fraction = value.get("mission_coverable_fraction")
        if (
            value.get("exact") is not True
            or value.get("eligible") is not True
            or not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not math.isfinite(float(fraction))
            or float(fraction) < FORMAL_SUCCESS_COVERAGE_RATIO
        ):
            raise ClosedLoopGateError(
                "gate scene-platform is not exact and mission-coverable"
            )
        reachable = value.get("reachable_mask_sha256")
        coverable = value.get("coverable_mask_sha256")
        if not _is_sha(reachable) or not _is_sha(coverable):
            raise ClosedLoopGateError("gate coverability mask identity is invalid")
        bindings.append(
            PlatformCoverabilityBinding(
                platform=platform,
                mission_coverable_fraction=float(fraction),
                reachable_mask_sha256=str(reachable),
                coverable_mask_sha256=str(coverable),
            )
        )
    return tuple(bindings)


def select_closed_loop_gate_cases(
    cache_manifest: Mapping[str, object],
    scenario_document: Mapping[str, object],
    *,
    minimum_scene_count: int = CLOSED_LOOP_MINIMUM_SCENES,
) -> tuple[ClosedLoopGateCase, ...]:
    """Select the first stable exact-common schedule entries across all splits."""
    if type(minimum_scene_count) is not int or minimum_scene_count < 1:
        raise ClosedLoopGateError("gate minimum scene count must be positive")
    if cache_manifest.get("schema") != "lunar-formal-training-cache/v4":
        raise ClosedLoopGateError("gate requires formal cache v4")
    entries = cache_manifest.get("scenes")
    scenarios = scenario_document.get("scenarios")
    common = cache_manifest.get("exact_common_evaluation")
    if (
        not isinstance(entries, list)
        or any(not isinstance(entry, Mapping) for entry in entries)
        or not isinstance(scenarios, list)
        or any(not isinstance(value, Mapping) for value in scenarios)
        or not isinstance(common, Mapping)
    ):
        raise ClosedLoopGateError("gate cache inventory is invalid")
    by_scenario = {str(value.get("scene_id")): value for value in scenarios}
    if len(by_scenario) != len(scenarios):
        raise ClosedLoopGateError("gate scenario IDs are not unique")
    common_splits = common.get("splits")
    if not isinstance(common_splits, Mapping):
        raise ClosedLoopGateError("gate exact-common inventory is invalid")

    cases: list[ClosedLoopGateCase] = []
    for split in _SPLITS:
        split_common = common_splits.get(split)
        if not isinstance(split_common, Mapping):
            raise ClosedLoopGateError("gate exact-common split is invalid")
        scene_ids = split_common.get("scene_ids")
        schedule_id = split_common.get("scenario_schedule_id")
        if (
            not isinstance(scene_ids, list)
            or any(not _is_sha(scene_id) for scene_id in scene_ids)
            or not isinstance(schedule_id, str)
            or not schedule_id
        ):
            raise ClosedLoopGateError("gate exact-common schedule is invalid")
        selected_ids = set(scene_ids)
        split_entries = [
            entry
            for entry in entries
            if entry.get("split") == split
            and entry.get("scene_id") in selected_ids
        ]
        scheduled = _scheduled_entries(
            split_entries,
            scenario_schedule_id=schedule_id,
        )
        if {str(entry["scene_id"]) for entry in scheduled} != selected_ids:
            raise ClosedLoopGateError("gate exact-common cache entries are incomplete")
        for episode_cursor, entry in enumerate(scheduled):
            scene_id = str(entry["scene_id"])
            scenario = by_scenario.get(scene_id)
            if scenario is None or scenario.get("split") != split:
                raise ClosedLoopGateError("gate scenario differs from cache split")
            cases.append(
                ClosedLoopGateCase(
                    scene_id=scene_id,
                    split=split,
                    scenario_seed=_scenario_seed(scenario),
                    episode_cursor=episode_cursor,
                    scenario_schedule_id=schedule_id,
                    coverability=_coverability_bindings(entry),
                )
            )
    if len(cases) < minimum_scene_count:
        raise ClosedLoopGateError(
            f"closed-loop gate requires at least {minimum_scene_count} "
            f"exact-common scenes, found {len(cases)}"
        )
    return tuple(cases[:minimum_scene_count])


def _array_signature(name: str, values: object) -> bytes:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.digest()


def _candidate_signature(observation: object) -> str:
    digest = hashlib.sha256()
    for name in ("frontier_features", "candidate_mask"):
        tensor = getattr(observation, name)
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(_array_signature(name, values))
    digest.update(repr(observation.observation_identities).encode("utf-8"))
    return digest.hexdigest()


def _request_signature(request: object) -> str:
    goal = request.goal.target
    body: dict[str, object] = {
        "request_id": request.request_id,
        "mission_id": request.mission_id,
        "mission_revision": request.mission_revision,
        "platform_id": request.platform_id,
        "capability_version": request.capability_version,
        "global_map_generation": request.global_map_generation,
        "local_map_generation": request.local_map_generation,
        "state_time_ns": request.state_time.nanoseconds_since_epoch,
        "goal": [
            goal.position_m.x,
            goal.position_m.y,
            goal.position_m.z,
            goal.tolerance_m,
        ],
        "maps": [],
    }
    for name, grid in (
        ("global", request.world.global_map),
        ("local", request.world.local_map),
    ):
        layers = []
        for layer_name in sorted(grid.layers):
            layer = grid.layers[layer_name]
            layers.append(
                [
                    layer_name,
                    _array_signature(layer_name, layer.values).hex(),
                ]
            )
        body["maps"].append(
            [
                name,
                grid.width,
                grid.height,
                grid.resolution_m,
                grid.stamp.nanoseconds_since_epoch,
                layers,
            ]
        )
    return _semantic_sha(body)


def _chain_update(digest: "hashlib._Hash", value: object) -> None:
    digest.update(_canonical_bytes(value))


def _run_closed_loop_work(work: _ClosedLoopWork) -> dict[str, object]:
    from .environment.formal_builder import FormalWorkerBuilder
    from .environment.macro_step import PolicyAction, TerminalReason
    from .evaluation.report import mission_coverage_ratio

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    bundle = load_project_formal_capability(Path(work.repository_root))
    builder = FormalWorkerBuilder(
        work.cache_manifest_path,
        work.case.split,
        True,
        work.case.scenario_schedule_id,
        None,
        True,
    )
    factory = FrozenCapabilityEnvironmentFactory(
        bundle=bundle,
        scenario_schedule_id=work.case.scenario_schedule_id,
        builder=builder,
    )
    worker = factory.create_for_episode(
        0,
        work.platform,
        work.case.episode_cursor,
        platform_worker_index=0,
        platform_worker_count=1,
    )
    if worker.episode.scene_id != work.case.scene_id:
        raise ClosedLoopGateError("gate worker selected a different physical scene")

    candidate_chain = hashlib.sha256()
    request_chain = hashlib.sha256()
    planner_chain = hashlib.sha256()
    planner_failures = 0
    safety_violations = 0
    invalid_actions = 0
    execution_failures = 0
    success_first_crossing = False
    step_count = 0
    final_coverage = float(
        mission_coverage_ratio(worker.environment.current_observation)[0]
    )
    terminal_reason: TerminalReason | None = None
    oracle_opportunity_count = 0
    deadline = time.monotonic() + work.watchdog_seconds

    while terminal_reason is None:
        if step_count >= work.watchdog_max_steps or time.monotonic() >= deadline:
            raise ClosedLoopGateError(
                "closed-loop gate watchdog expired before natural terminal"
            )
        boundary = worker.environment.refresh_decision_boundary()
        if boundary.execution_state == "NO_CANDIDATES":
            terminal_reason = boundary.terminal_reason
            oracle_opportunity_count = boundary.oracle_opportunity_count
            final_coverage = float(
                mission_coverage_ratio(worker.environment.current_observation)[0]
            )
            break
        if boundary.execution_state != "DECISION_READY":
            raise ClosedLoopGateError("gate worker has an invalid decision boundary")
        observation = worker.environment.current_observation
        identities = observation.observation_identities
        if identities is None or len(identities) != 1:
            raise ClosedLoopGateError("gate observation identity is missing")
        features = observation.frontier_features[0].detach().cpu().numpy()
        mask = observation.candidate_mask[0].detach().cpu().numpy()
        first = select_baseline_action(
            CLOSED_LOOP_METHOD,
            features,
            mask,
            np.random.Generator(np.random.PCG64(work.case.scenario_seed)),
        )
        repeated = select_baseline_action(
            CLOSED_LOOP_METHOD,
            features,
            mask,
            np.random.Generator(np.random.PCG64(work.case.scenario_seed)),
        )
        if first != repeated:
            raise ClosedLoopGateError("gate baseline action differs on repeat")
        theta = 0.0 if work.platform == "HOPPER" else first.theta
        action = PolicyAction(first.candidate_index, theta)
        candidate_signature = _candidate_signature(observation)
        first_request = worker.episode.build_request(action, identities[0]).request
        second_request = worker.episode.build_request(action, identities[0]).request
        request_signature = _request_signature(first_request)
        if request_signature != _request_signature(second_request):
            raise ClosedLoopGateError("gate request differs on repeat")
        _chain_update(
            candidate_chain,
            {
                "step": step_count,
                "candidate_sha256": candidate_signature,
                "selected_index": first.candidate_index,
                "selected_theta_hex": float(theta).hex(),
            },
        )
        _chain_update(
            request_chain,
            {"step": step_count, "request_sha256": request_signature},
        )

        result = worker.environment.advance_prepared_action(
            action,
            expected_identity=identities[0],
        )
        transition = result.transition
        if transition is None or result.policy_decisions_consumed != 1:
            raise ClosedLoopGateError("gate action produced no planner transition")
        step_count += 1
        events = transition.execution_events
        accepted = (
            transition.planning_outcome.name == "NEW_REFERENCE_AVAILABLE"
            and events.execution_failure_count == 0
        )
        planner_failures += int(not accepted)
        safety_violations += events.safety_violation_count
        invalid_actions += events.invalid_action_count
        execution_failures += events.execution_failure_count
        success_first_crossing = bool(
            success_first_crossing or transition.success_first_crossing
        )
        final_coverage = float(
            mission_coverage_ratio(transition.next_observation)[0]
        )
        _chain_update(
            planner_chain,
            {
                "step": step_count - 1,
                "outcome": transition.planning_outcome.name,
                "directive": transition.execution_directive.name,
                "reason_code": transition.reason_code,
                "events": asdict(events),
            },
        )
        if transition.terminated:
            terminal_reason = transition.terminal_reason
            oracle_opportunity_count = transition.oracle_opportunity_count

    if terminal_reason is None:
        raise ClosedLoopGateError("gate natural terminal has no reason")
    binding = work.case.for_platform(work.platform)
    return {
        "scene_id": work.case.scene_id,
        "split": work.case.split,
        "platform": work.platform,
        "exact": True,
        "mission_coverable_fraction_hex": (
            binding.mission_coverable_fraction.hex()
        ),
        "reachable_mask_sha256": binding.reachable_mask_sha256,
        "coverable_mask_sha256": binding.coverable_mask_sha256,
        "final_coverage_hex": final_coverage.hex(),
        "success_first_crossing": success_first_crossing,
        "terminal_reason": terminal_reason.value,
        "oracle_contradiction_count": 0,
        "oracle_opportunity_count": oracle_opportunity_count,
        "planner_failure_count": planner_failures,
        "safety_violation_count": safety_violations,
        "invalid_action_count": invalid_actions,
        "execution_failure_count": execution_failures,
        "executed_step_count": step_count,
        "candidate_sequence_sha256": candidate_chain.hexdigest(),
        "request_sequence_sha256": request_chain.hexdigest(),
        "planner_sequence_sha256": planner_chain.hexdigest(),
    }


def _validate_row(
    row: Mapping[str, object],
    case: ClosedLoopGateCase,
    platform: str,
) -> None:
    if row.get("exact") is not True:
        raise ClosedLoopGateError("closed-loop row is not exact")
    try:
        mission_fraction = float.fromhex(
            str(row.get("mission_coverable_fraction_hex"))
        )
        final_coverage = float.fromhex(str(row.get("final_coverage_hex")))
    except ValueError as error:
        raise ClosedLoopGateError("closed-loop coverage encoding is invalid") from error
    binding = case.for_platform(platform)
    if (
        not math.isfinite(mission_fraction)
        or mission_fraction < FORMAL_SUCCESS_COVERAGE_RATIO
        or mission_fraction != binding.mission_coverable_fraction
    ):
        raise ClosedLoopGateError("closed-loop mission coverable fraction failed")
    if (
        not math.isfinite(final_coverage)
        or final_coverage < FORMAL_SUCCESS_COVERAGE_RATIO
    ):
        raise ClosedLoopGateError("closed-loop final coverage failed")
    if row.get("success_first_crossing") is not True:
        raise ClosedLoopGateError("closed-loop success crossing is missing")
    if row.get("terminal_reason") != "SUCCESS":
        raise ClosedLoopGateError("closed-loop terminal reason is not SUCCESS")
    for field, message in (
        ("oracle_contradiction_count", "oracle contradiction"),
        ("oracle_opportunity_count", "terminal oracle opportunity"),
        ("planner_failure_count", "planner failure"),
        ("safety_violation_count", "safety violation"),
        ("invalid_action_count", "invalid action"),
        ("execution_failure_count", "execution failure"),
    ):
        if row.get(field) != 0:
            raise ClosedLoopGateError(f"closed-loop {message} count is nonzero")
    if type(row.get("executed_step_count")) is not int or int(
        row["executed_step_count"]
    ) <= 0:
        raise ClosedLoopGateError("closed-loop executed step count is invalid")
    for field in _ROW_SHA_FIELDS:
        if not _is_sha(row.get(field)):
            raise ClosedLoopGateError(f"closed-loop {field} is invalid")
    if (
        row["reachable_mask_sha256"] != binding.reachable_mask_sha256
        or row["coverable_mask_sha256"] != binding.coverable_mask_sha256
    ):
        raise ClosedLoopGateError("closed-loop coverability identity differs")


def build_closed_loop_gate_report(
    *,
    source_commit: str,
    cache_manifest_sha256: str,
    cases: Sequence[ClosedLoopGateCase],
    rows: Sequence[Mapping[str, object]],
    timings_seconds: Mapping[str, float],
) -> ClosedLoopGateReport:
    if not _is_sha(source_commit, length=40) or not _is_sha(
        cache_manifest_sha256
    ):
        raise ClosedLoopGateError("closed-loop gate input identity is invalid")
    selected_cases = tuple(cases)
    if (
        len(selected_cases) < CLOSED_LOOP_MINIMUM_SCENES
        or len({case.scene_id for case in selected_cases}) != len(selected_cases)
        or any(not isinstance(case, ClosedLoopGateCase) for case in selected_cases)
    ):
        raise ClosedLoopGateError("closed-loop gate requires 24 unique cases")
    by_case = {case.scene_id: case for case in selected_cases}
    expected = {
        (case.scene_id, platform)
        for case in selected_cases
        for platform in PLATFORMS
    }
    materialized_rows = tuple(dict(row) for row in rows)
    actual = {
        (str(row.get("scene_id")), str(row.get("platform")))
        for row in materialized_rows
    }
    if len(materialized_rows) != len(expected) or actual != expected:
        raise ClosedLoopGateError("closed-loop scene-platform rows are incomplete")
    ordered_rows = sorted(
        materialized_rows,
        key=lambda row: (
            str(row["scene_id"]),
            PLATFORMS.index(str(row["platform"])),
        ),
    )
    for row in ordered_rows:
        scene_id = str(row["scene_id"])
        platform = str(row["platform"])
        case = by_case[scene_id]
        if row.get("split") != case.split:
            raise ClosedLoopGateError("closed-loop row split differs from case")
        _validate_row(row, case, platform)
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in timings_seconds.values()
    ):
        raise ClosedLoopGateError("closed-loop timing is invalid")
    ordered_cases = sorted(selected_cases, key=lambda case: case.scene_id)
    evidence: dict[str, object] = {
        "schema_version": CLOSED_LOOP_GATE_SCHEMA,
        "source_commit": source_commit,
        "cache_manifest_sha256": cache_manifest_sha256,
        "method": CLOSED_LOOP_METHOD,
        "minimum_scene_count": CLOSED_LOOP_MINIMUM_SCENES,
        "scene_count": len(selected_cases),
        "scene_platform_count": len(ordered_rows),
        "selected_scenes": [
            {
                "scene_id": case.scene_id,
                "split": case.split,
                "scenario_seed": case.scenario_seed,
                "episode_cursor": case.episode_cursor,
                "scenario_schedule_id": case.scenario_schedule_id,
            }
            for case in ordered_cases
        ],
        "rows": ordered_rows,
        "passed": True,
        "training_started": False,
    }
    payload = {
        **evidence,
        "closed_loop_evidence_sha256": _semantic_sha(evidence),
        "timings_seconds": {
            name: float(value) for name, value in sorted(timings_seconds.items())
        },
    }
    return ClosedLoopGateReport(payload)


def write_closed_loop_gate_report(
    artifact_root: Path,
    report: ClosedLoopGateReport,
) -> Path:
    if not isinstance(report, ClosedLoopGateReport):
        raise ClosedLoopGateError("closed-loop gate report is invalid")
    artifact_root.mkdir(parents=True, exist_ok=True)
    path = artifact_root / "closed-loop-gate.json"
    body = report.to_dict()
    body["closed_loop_report_sha256"] = _semantic_sha(body)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(_canonical_bytes(body))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        if temporary.exists():
            temporary.unlink()
        raise ClosedLoopGateError("could not write closed-loop report") from error
    return path


def run_closed_loop_gate(
    *,
    cache_manifest_path: Path,
    artifact_root: Path,
    repository_root: Path,
    source_commit: str,
    minimum_scene_count: int = CLOSED_LOOP_MINIMUM_SCENES,
    max_workers: int = 8,
    watchdog_max_steps: int = 4096,
    watchdog_seconds: float = 3600.0,
    progress_stream: IO[str] | None = sys.stderr,
) -> tuple[ClosedLoopGateReport, Path]:
    """Run every selected scene-platform to a natural terminal in subprocesses."""
    from .polar_data.formal_cache import current_v3_identity, load_formal_cache
    from .reward import reward_weights_sha256

    if type(max_workers) is not int or max_workers < 1:
        raise ClosedLoopGateError("closed-loop worker count must be positive")
    if type(watchdog_max_steps) is not int or watchdog_max_steps < 1:
        raise ClosedLoopGateError("closed-loop watchdog steps must be positive")
    if (
        not isinstance(watchdog_seconds, (int, float))
        or isinstance(watchdog_seconds, bool)
        or not math.isfinite(float(watchdog_seconds))
        or float(watchdog_seconds) <= 0.0
    ):
        raise ClosedLoopGateError("closed-loop watchdog seconds must be positive")
    root = repository_root.resolve(strict=True)
    cache_path = cache_manifest_path.resolve(strict=True)
    output_root = artifact_root.resolve()
    if output_root == root or root in output_root.parents:
        raise ClosedLoopGateError("closed-loop artifacts must stay outside Git")
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ClosedLoopGateError("could not verify the gate source tree") from error
    if status.stdout:
        raise ClosedLoopGateError("closed-loop gate requires a clean source tree")
    cache = load_formal_cache(cache_path)
    bundle = load_project_formal_capability(root)
    current_commit, current_v3_sha256 = current_v3_identity(root)
    identity = cache.identity
    expected = {
        "v3_source_commit": current_commit,
        "v3_sha256": current_v3_sha256,
        "generator_sha256": GENERATOR_SHA256,
        "capability_sha256": bundle.bundle_sha256,
        "reward_sha256": reward_weights_sha256(),
        "training_semantics_sha256": training_semantics_sha256(),
    }
    if source_commit != current_commit or any(
        getattr(identity, field) != value for field, value in expected.items()
    ):
        raise ClosedLoopGateError("closed-loop cache identity differs from source")
    try:
        scenario_document = json.loads(
            (cache.root / "scenario-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ClosedLoopGateError("closed-loop scenario manifest is invalid") from error
    cases = select_closed_loop_gate_cases(
        cache.manifest,
        scenario_document,
        minimum_scene_count=minimum_scene_count,
    )
    work_items = tuple(
        _ClosedLoopWork(
            cache_manifest_path=str(cache_path),
            repository_root=str(root),
            case=case,
            platform=platform,
            watchdog_max_steps=watchdog_max_steps,
            watchdog_seconds=float(watchdog_seconds),
        )
        for case in cases
        for platform in PLATFORMS
    )
    started = time.monotonic()
    rows: list[dict[str, object]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(max_workers, len(work_items)),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(_run_closed_loop_work, work): work
            for work in work_items
        }
        completed = 0
        for future in as_completed(futures):
            work = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                raise ClosedLoopGateError(
                    "closed-loop scene-platform execution failed: "
                    f"{work.case.scene_id}/{work.platform}: {error}"
                ) from error
            completed += 1
            if progress_stream is not None:
                print(
                    json.dumps(
                        {
                            "closed_loop_completed": completed,
                            "closed_loop_total": len(work_items),
                            "scene_id": work.case.scene_id,
                            "platform": work.platform,
                        },
                        sort_keys=True,
                    ),
                    file=progress_stream,
                    flush=True,
                )
    report = build_closed_loop_gate_report(
        source_commit=source_commit,
        cache_manifest_sha256=str(cache.manifest["cache_manifest_sha256"]),
        cases=cases,
        rows=rows,
        timings_seconds={"closed_loop": time.monotonic() - started},
    )
    return report, write_closed_loop_gate_report(output_root, report)


__all__ = [
    "CLOSED_LOOP_GATE_SCHEMA",
    "CLOSED_LOOP_MINIMUM_SCENES",
    "ClosedLoopGateCase",
    "ClosedLoopGateError",
    "ClosedLoopGateReport",
    "build_closed_loop_gate_report",
    "run_closed_loop_gate",
    "select_closed_loop_gate_cases",
    "write_closed_loop_gate_report",
]
