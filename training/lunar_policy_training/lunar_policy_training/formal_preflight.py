"""Short, non-training qualification for the cache-backed formal environment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from lunar_model_contract import ObservationContractV3
from lunar_planner_training_bridge import PlannerBridge

from .checkpoint import CHECKPOINT_SCHEMA_VERSION, RunIdentity
from .config import (
    FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS,
    FORMAL_WORKER_CANDIDATES,
    PLATFORMS,
    ROLLOUT_HORIZON_CANDIDATES,
)
from .environment.formal_builder import FormalEnvironmentAssembly
from .environment.macro_step import PolicyAction
from .environment.parallel_pool import ParallelActions, ParallelEnvPool
from .evaluation.report import (
    FormalEvaluationBatch,
    formal_evaluation_probe,
)
from .policy.cross_attention import CrossAttentionPolicy
from .policy.observation import PolicyBatch
from .polar_data.formal_cache import FormalCache
from .reward import compute_transition_reward, reward_weights_sha256
from .training_semantics import training_semantics_sha256


FORMAL_PREFLIGHT_SCHEMA = "lunar-formal-training-preflight/v6"
REQUIRED_PREFLIGHT_CHECKS = (
    "cache_and_identity",
    "three_platform_worker_construction",
    "same_world",
    "deterministic_request_and_planner",
    "multiresolution_4m_global_0p2m_local",
    "hopper_no_cumulative_fuel",
    "update_boundary_resume",
    "rollout_horizon_not_episode_limit",
    "qualified_worker_configuration",
    "nonproxy_evaluation_probe",
)
_RESUME_EQUIVALENCE_FIELDS = {
    "checkpoint_schema",
    "checkpoint_relative_path",
    "checkpoint_sha256",
    "checkpoint_roundtrip",
    "rollout_exact",
    "model_exact",
    "optimizer_exact",
    "rng_exact",
    "environment_state_exact",
    "observation_exact",
    "candidate_exact",
    "first_request_exact",
    "uninterrupted_update",
    "resumed_update",
    "evidence_sha256",
}
_RESUME_EQUIVALENCE_BOOLEAN_FIELDS = _RESUME_EQUIVALENCE_FIELDS - {
    "checkpoint_schema",
    "checkpoint_relative_path",
    "checkpoint_sha256",
    "uninterrupted_update",
    "resumed_update",
    "evidence_sha256",
}


class FormalPreflightError(RuntimeError):
    """A formal environment invariant failed before training began."""


@dataclass(frozen=True, slots=True)
class FormalPreflightReport:
    payload: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


def _is_sha(value: object, *, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


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


def build_formal_preflight_report(
    *,
    source_commit: str,
    cache_manifest_sha256: str,
    sensor_performance_sha256: str,
    run_identity: RunIdentity,
    scenario_schedule_ids: Mapping[str, str],
    checks: Mapping[str, bool],
    timings_seconds: Mapping[str, float],
    qualified_worker_candidates: tuple[int, ...],
    selected_workers: int,
    selected_micro_batch: int,
    selected_rollout_horizon: int,
    evaluation_probe_sha256: str,
    resume_equivalence: Mapping[str, object],
    additional_corridor_margin_m: float,
) -> FormalPreflightReport:
    if not _is_sha(source_commit, length=40):
        raise FormalPreflightError("source commit is invalid")
    if not _is_sha(cache_manifest_sha256) or not _is_sha(
        sensor_performance_sha256
    ):
        raise FormalPreflightError("preflight input digest is invalid")
    if not isinstance(run_identity, RunIdentity) or run_identity.run_kind != "formal":
        raise FormalPreflightError("preflight requires a formal run identity")
    if run_identity.reward_sha256 != reward_weights_sha256():
        raise FormalPreflightError("preflight reward identity is stale")
    if (
        run_identity.training_semantics_sha256
        != training_semantics_sha256()
    ):
        raise FormalPreflightError("preflight training semantics are stale")
    if set(scenario_schedule_ids) != {
        "train",
        "validation",
        "test",
        "holdout",
    } or any(
        not isinstance(value, str) or not value
        for value in scenario_schedule_ids.values()
    ):
        raise FormalPreflightError("preflight scenario schedules are incomplete")
    if set(checks) != set(REQUIRED_PREFLIGHT_CHECKS) or not all(
        checks[name] for name in REQUIRED_PREFLIGHT_CHECKS
    ):
        raise FormalPreflightError("preflight required checks are not closed")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in timings_seconds.values()
    ):
        raise FormalPreflightError("preflight timings are invalid")
    if (
        not qualified_worker_candidates
        or any(
            type(value) is not int or value <= 0
            for value in qualified_worker_candidates
        )
        or selected_workers not in qualified_worker_candidates
        or type(selected_micro_batch) is not int
        or selected_micro_batch <= 0
    ):
        raise FormalPreflightError("preflight worker recommendation is invalid")
    if selected_rollout_horizon not in ROLLOUT_HORIZON_CANDIDATES:
        raise FormalPreflightError("preflight rollout horizon is not calibrated")
    if not _is_sha(evaluation_probe_sha256):
        raise FormalPreflightError("preflight evaluation digest is invalid")
    if (
        not isinstance(additional_corridor_margin_m, (int, float))
        or isinstance(additional_corridor_margin_m, bool)
        or float(additional_corridor_margin_m) != 2.0
    ):
        raise FormalPreflightError("preflight corridor margin must be fixed at 2.0 m")
    if (
        not isinstance(resume_equivalence, Mapping)
        or set(resume_equivalence) != _RESUME_EQUIVALENCE_FIELDS
        or resume_equivalence.get("checkpoint_schema")
        != CHECKPOINT_SCHEMA_VERSION
        or not isinstance(
            resume_equivalence.get("checkpoint_relative_path"), str
        )
        or Path(str(resume_equivalence["checkpoint_relative_path"])).is_absolute()
        or ".."
        in Path(str(resume_equivalence["checkpoint_relative_path"])).parts
        or any(
            resume_equivalence.get(field) is not True
            for field in _RESUME_EQUIVALENCE_BOOLEAN_FIELDS
        )
        or resume_equivalence.get("uninterrupted_update") != 2
        or resume_equivalence.get("resumed_update") != 2
        or not _is_sha(resume_equivalence.get("checkpoint_sha256"))
        or not _is_sha(resume_equivalence.get("evidence_sha256"))
    ):
        raise FormalPreflightError("preflight resume equivalence is invalid")
    payload: dict[str, object] = {
        "schema_version": FORMAL_PREFLIGHT_SCHEMA,
        "source_commit": source_commit,
        "cache_manifest_sha256": cache_manifest_sha256,
        "sensor_performance_sha256": sensor_performance_sha256,
        "run_identity": run_identity.to_dict(),
        "scenario_schedule_ids": dict(sorted(scenario_schedule_ids.items())),
        "checks": {name: bool(checks[name]) for name in REQUIRED_PREFLIGHT_CHECKS},
        "timings_seconds": {
            name: float(value) for name, value in sorted(timings_seconds.items())
        },
        "qualified_worker_candidates": list(qualified_worker_candidates),
        "selected_workers": selected_workers,
        "selected_micro_batch": selected_micro_batch,
        "rollout_horizon_candidates": list(ROLLOUT_HORIZON_CANDIDATES),
        "selected_rollout_horizon": selected_rollout_horizon,
        "episode_decision_limit": None,
        "evaluation_probe_sha256": evaluation_probe_sha256,
        "additional_corridor_margin_m": 2.0,
        "resume_equivalence": dict(sorted(resume_equivalence.items())),
        "proxy": False,
        "training_started": False,
    }
    return FormalPreflightReport(payload)


def write_formal_preflight_report(
    artifact_root: Path, report: FormalPreflightReport
) -> Path:
    if not isinstance(report, FormalPreflightReport):
        raise FormalPreflightError("formal preflight report is invalid")
    artifact_root.mkdir(parents=True, exist_ok=True)
    path = artifact_root / "formal-preflight.json"
    body = report.to_dict()
    body["preflight_report_sha256"] = hashlib.sha256(
        _canonical_bytes(body).rstrip(b"\n")
    ).hexdigest()
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
        raise FormalPreflightError("could not write formal preflight report") from error
    return path


def _batch_digest(batch: PolicyBatch) -> str:
    digest = hashlib.sha256()
    for name in ObservationContractV3.input_names:
        values = getattr(batch, name).detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    identities = batch.observation_identities
    digest.update(repr(identities).encode("utf-8"))
    return digest.hexdigest()


def _array_digest(values: object) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _request_signature(request: object) -> str:
    goal = request.goal.target
    global_map = request.world.global_map
    local_map = request.world.local_map
    body = {
        "request_id": request.request_id,
        "mission_id": request.mission_id,
        "platform_id": request.platform_id,
        "capability_version": request.capability_version,
        "target": [
            goal.position_m.x,
            goal.position_m.y,
            goal.position_m.z,
            goal.tolerance_m,
        ],
        "global": [
            global_map.width,
            global_map.height,
            global_map.resolution_m,
            _array_digest(global_map.layers["valid_mask"].values),
            _array_digest(global_map.layers["elevation"].values),
        ],
        "local": [
            local_map.width,
            local_map.height,
            local_map.resolution_m,
            _array_digest(local_map.layers["valid_mask"].values),
            _array_digest(local_map.layers["elevation"].values),
        ],
    }
    return hashlib.sha256(_canonical_bytes(body).rstrip(b"\n")).hexdigest()


def _reference_signature(reference: object | None) -> object:
    if reference is None:
        return None
    data = reference.data
    if hasattr(data, "points"):
        geometry = [
            (
                point.pose.position_m.x,
                point.pose.position_m.y,
                point.pose.position_m.z,
                point.time_from_start.total_seconds(),
            )
            for point in data.points
        ]
    elif hasattr(data, "segments"):
        geometry = [
            (
                segment.nominal_landing_point_m.x,
                segment.nominal_landing_point_m.y,
                segment.nominal_landing_point_m.z,
                segment.available_delta_v_mps,
                segment.required_delta_v_mps,
                segment.flight_time.total_seconds(),
            )
            for segment in data.segments
        ]
    else:
        raise FormalPreflightError("planner reference type is unsupported")
    return {
        "platform_type": reference.platform_type,
        "geometry": geometry,
    }


def _planner_signature(output: object) -> str:
    body = {
        "outcome": output.outcome.name,
        "directive": output.directive.name,
        "reason_code": output.reason_code,
        "reference": _reference_signature(output.reference),
    }
    return hashlib.sha256(_canonical_bytes(body).rstrip(b"\n")).hexdigest()


def _first_action(worker: object, platform: str) -> tuple[PolicyAction, object]:
    observation = worker.initial_observation
    identities = observation.observation_identities
    if identities is None or len(identities) != 1:
        raise FormalPreflightError("formal worker observation identity is missing")
    candidates = observation.candidate_mask[0].nonzero().flatten()
    if int(candidates.numel()) == 0:
        raise FormalPreflightError("formal worker has no observed-safe candidate")
    theta = 0.0
    return PolicyAction(int(candidates[0].item()), theta), identities[0]


def _direct_environment_checks(
    cache: FormalCache, assembly: FormalEnvironmentAssembly
) -> dict[str, object]:
    workers = {
        platform: assembly.factory.create_for_episode(
            0,
            platform,
            0,
            platform_worker_index=0,
            platform_worker_count=1,
        )
        for platform in PLATFORMS
    }
    scene_ids = {worker.episode.scene_id for worker in workers.values()}
    if len(scene_ids) != 1:
        raise FormalPreflightError("three platforms did not receive the same world")
    request_hashes: dict[str, str] = {}
    planner_hashes: dict[str, str] = {}
    corridor_margins: set[float] = set()
    for platform, worker in workers.items():
        action, identity = _first_action(worker, platform)
        first = worker.episode.build_request(action, identity).request
        second = worker.episode.build_request(action, identity).request
        first_hash = _request_signature(first)
        if first_hash != _request_signature(second):
            raise FormalPreflightError("repeated formal request differs")
        first_output = PlannerBridge().plan(first)
        second_output = PlannerBridge().plan(second)
        first_planner_hash = _planner_signature(first_output)
        if first_planner_hash != _planner_signature(second_output):
            raise FormalPreflightError("repeated C++ planner result differs")
        if (
            first.world.global_map.resolution_m != 4.0
            or first.world.global_map.width != 256
            or first.world.local_map.resolution_m != 0.2
            or first.world.local_map.width != 320
        ):
            raise FormalPreflightError("formal multiresolution map contract differs")
        margin = first.config.local_frontier.additional_corridor_margin_m
        if float(margin) != 2.0:
            raise FormalPreflightError(
                "formal planner corridor margin is not fixed at 2.0 m"
            )
        corridor_margins.add(float(margin))
        request_hashes[platform] = first_hash
        planner_hashes[platform] = first_planner_hash

    train_count = sum(
        entry.get("split") == "train" for entry in cache.manifest["scenes"]
    )
    hopper_available: list[float] = []
    for cursor in (0, train_count):
        worker = assembly.factory.create_for_episode(
            0,
            "HOPPER",
            cursor,
            platform_worker_index=0,
            platform_worker_count=1,
        )
        action, identity = _first_action(worker, "HOPPER")
        result = worker.environment.advance_prepared_action(
            action, expected_identity=identity
        )
        if result.transition.hard_safety_violation:
            raise FormalPreflightError("hopper preflight action was not certified")
        hopper_available.append(worker.episode.last_hop_available_delta_v_mps)
    if hopper_available[0] <= 0.0 or hopper_available[0] != hopper_available[1]:
        raise FormalPreflightError("hopper available delta-v accumulated across episodes")
    return {
        "scene_id": next(iter(scene_ids)),
        "request_sha256s": request_hashes,
        "planner_sha256s": planner_hashes,
        "hopper_available_delta_v_mps": hopper_available[0],
        "additional_corridor_margin_m": corridor_margins.pop(),
    }


def _resume_check(assembly: FormalEnvironmentAssembly) -> None:
    allocation = {platform: 1 for platform in PLATFORMS}
    with ParallelEnvPool(
        allocation=allocation,
        observation_template=assembly.observation_template,
        environment_factory=assembly.factory,
        reward_fn=compute_transition_reward,
        worker_timeout_seconds=FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS,
    ) as first:
        initial = first.reset()
        candidate_indices = torch.tensor(
            [
                int(initial.observations.candidate_mask[index].nonzero()[0])
                for index in range(3)
            ],
            dtype=torch.int64,
        )
        stepped = first.step(
            ParallelActions(
                candidate_indices=candidate_indices,
                thetas=torch.zeros((3,), dtype=torch.float32),
            ),
            policy_version=0,
        )
        states = first.snapshot_episode_states(policy_version=0)
        active_digest = _batch_digest(stepped.observations)
    with ParallelEnvPool(
        allocation=allocation,
        observation_template=assembly.observation_template,
        environment_factory=assembly.factory,
        reward_fn=compute_transition_reward,
        worker_timeout_seconds=FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS,
        initial_episode_states=states,
    ) as resumed:
        resumed_digest = _batch_digest(resumed.reset().observations)
    if active_digest != resumed_digest:
        raise FormalPreflightError(
            "resumed formal observations differ at update boundary"
        )


def _qualify_worker_candidate(
    assembly: FormalEnvironmentAssembly, workers: int
) -> float:
    allocation = {platform: workers // 3 for platform in PLATFORMS}
    if sum(allocation.values()) != workers:
        raise FormalPreflightError("worker candidate must divide across three platforms")
    started = time.monotonic()
    with ParallelEnvPool(
        allocation=allocation,
        observation_template=assembly.observation_template,
        environment_factory=assembly.factory,
        reward_fn=compute_transition_reward,
        worker_timeout_seconds=FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS,
    ) as pool:
        pool.reset()
    return time.monotonic() - started


def run_formal_preflight(
    *,
    cache: FormalCache,
    assemblies: Mapping[str, FormalEnvironmentAssembly],
    evaluation_batches: Sequence[FormalEvaluationBatch],
    run_identity: RunIdentity,
    source_commit: str,
    sensor_performance_sha256: str,
    artifact_root: Path,
    worker_candidates: tuple[int, ...] = FORMAL_WORKER_CANDIDATES,
    selected_workers: int | None = None,
    selected_micro_batch: int = 2,
    selected_rollout_horizon: int = 32,
    resume_equivalence: Mapping[str, object] | None = None,
) -> tuple[FormalPreflightReport, Path]:
    """Execute formal wiring and record the supplied V7 resume proof."""
    expected_splits = {"train", "validation", "test", "holdout"}
    if set(assemblies) != expected_splits:
        raise FormalPreflightError("formal preflight assemblies are incomplete")
    timings: dict[str, float] = {}
    started = time.monotonic()
    direct_evidence = _direct_environment_checks(cache, assemblies["train"])
    timings["direct_environment"] = time.monotonic() - started

    started = time.monotonic()
    _resume_check(assemblies["train"])
    timings["update_boundary_resume"] = time.monotonic() - started

    qualified: list[int] = []
    for workers in worker_candidates:
        timings[f"worker_{workers}"] = _qualify_worker_candidate(
            assemblies["train"], workers
        )
        qualified.append(workers)
    if not qualified:
        raise FormalPreflightError("no formal worker configuration qualified")
    if selected_workers is None:
        selected_workers = max(qualified)
    elif selected_workers not in qualified:
        raise FormalPreflightError(
            "calibrated worker selection did not pass preflight"
        )

    bounded_batches = tuple(
        FormalEvaluationBatch(
            split=batch.split,
            factory=batch.factory,
            observation_template=batch.observation_template,
            scenario_seeds=batch.scenario_seeds[:1],
        )
        for batch in evaluation_batches
    )
    started = time.monotonic()
    evaluation_probe = formal_evaluation_probe(
        CrossAttentionPolicy(),
        device="cpu",
        run_identity=run_identity,
        batches=bounded_batches,
    )
    timings["nonproxy_evaluation_probe"] = time.monotonic() - started
    if (
        evaluation_probe.get("proxy") is not False
        or evaluation_probe.get("schema_version")
        != "lunar-formal-evaluation-probe/v1"
        or evaluation_probe.get("row_count") != 27
        or not _is_sha(evaluation_probe.get("probe_sha256"))
    ):
        raise FormalPreflightError("formal preflight evaluation fell back to proxy")

    checks = {name: True for name in REQUIRED_PREFLIGHT_CHECKS}
    if resume_equivalence is None:
        raise FormalPreflightError("formal preflight resume equivalence is missing")
    report = build_formal_preflight_report(
        source_commit=source_commit,
        cache_manifest_sha256=str(cache.manifest["cache_manifest_sha256"]),
        sensor_performance_sha256=sensor_performance_sha256,
        run_identity=run_identity,
        scenario_schedule_ids={
            split: assemblies[split].scenario_schedule_id
            for split in ("train", "validation", "test", "holdout")
        },
        checks=checks,
        timings_seconds=timings,
        qualified_worker_candidates=tuple(qualified),
        selected_workers=selected_workers,
        selected_micro_batch=selected_micro_batch,
        selected_rollout_horizon=selected_rollout_horizon,
        evaluation_probe_sha256=str(evaluation_probe["probe_sha256"]),
        resume_equivalence=resume_equivalence,
        additional_corridor_margin_m=float(
            direct_evidence["additional_corridor_margin_m"]
        ),
    )
    return report, write_formal_preflight_report(artifact_root, report)


__all__ = [
    "FORMAL_PREFLIGHT_SCHEMA",
    "REQUIRED_PREFLIGHT_CHECKS",
    "FormalPreflightError",
    "FormalPreflightReport",
    "build_formal_preflight_report",
    "run_formal_preflight",
    "write_formal_preflight_report",
]
