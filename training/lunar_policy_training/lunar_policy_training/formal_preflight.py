"""Short, non-training qualification for the cache-backed formal environment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from lunar_model_contract import ObservationContractV4
from lunar_planner_training_bridge import PlannerBridge

from .checkpoint import CHECKPOINT_SCHEMA_VERSION, RunIdentity
from .budget import RUN_MANIFEST_SCHEMA_VERSION
from .config import (
    FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS,
    PLATFORMS,
    REWARD_V4_FORMAL_ROLLOUT_HORIZON,
)
from .environment.formal_builder import FormalEnvironmentAssembly
from .environment.formal_episode_state import (
    FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION,
)
from .environment.macro_step import PolicyAction
from .environment.parallel_pool import ParallelActions, ParallelEnvPool
from .policy.observation import PolicyBatch
from .polar_data.formal_cache import FORMAL_CACHE_SCHEMA, FormalCache
from .reward import compute_transition_reward, reward_weights_sha256
from .reward_contract import (
    DEFAULT_REWARD_CONFIG,
    REWARD_SCHEMA_VERSION,
    TaskScaleBucket,
)
from .reward_evaluation import (
    REWARD_EVALUATION_MANIFEST_SCHEMA,
    build_reward_v4_evaluation_manifest,
    reward_evaluation_manifest_sha256,
)
from .reward_curriculum import PlatformType
from .training_metrics import TRAINING_UPDATE_METRICS_SCHEMA
from .training_semantics import training_semantics_sha256


FORMAL_PREFLIGHT_SCHEMA = "lunar-formal-training-preflight/v10"
REQUIRED_PREFLIGHT_CHECKS = (
    "cache_and_identity",
    "three_platform_worker_construction",
    "same_world",
    "deterministic_request_and_planner",
    "multiresolution_4m_global_0p2m_local",
    "task_cache_current_next_identity",
    "zero_halo_and_full_scene_leaks",
    "hopper_no_cumulative_fuel",
    "single_worker_resume_boundary",
    "rollout_horizon_not_episode_limit",
    "calibrated_worker_configuration",
    "three_platform_macro_step",
)
_RESUME_BOUNDARY_FIELDS = {
    "schema_version",
    "platforms",
    "post_step_observation_sha256",
    "resumed_observation_sha256",
    "state_roundtrip",
    "evidence_sha256",
}
_MINIMAL_RESUME_BOUNDARY_SCHEMA = "lunar-formal-preflight-resume-boundary/v1"


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


def _validate_task_cache_evidence(
    value: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "current_ground",
        "next_ground",
        "halo_leak_count",
        "full_scene_derived_call_count",
    }:
        raise FormalPreflightError("preflight task cache evidence is invalid")
    if (
        value.get("halo_leak_count") != 0
        or type(value.get("halo_leak_count")) is not int
        or value.get("full_scene_derived_call_count") != 0
        or type(value.get("full_scene_derived_call_count")) is not int
    ):
        raise FormalPreflightError("preflight task cache evidence is invalid")
    result: dict[str, object] = {
        "halo_leak_count": 0,
        "full_scene_derived_call_count": 0,
    }
    expected_fields = {
        "task_geometry_sha256",
        "task_common_key_sha256",
        "task_common_artifact_sha256",
        "platform_task_key_sha256",
        "platform_task_artifact_sha256",
        "coarse_shape",
        "detail_shape",
    }
    for stage in ("current_ground", "next_ground"):
        stage_value = value.get(stage)
        if not isinstance(stage_value, Mapping) or set(stage_value) != {
            "WHEELED",
            "LEGGED",
        }:
            raise FormalPreflightError("preflight task cache evidence is invalid")
        stage_result: dict[str, object] = {}
        for platform in ("WHEELED", "LEGGED"):
            item = stage_value.get(platform)
            if not isinstance(item, Mapping) or set(item) != expected_fields:
                raise FormalPreflightError(
                    "preflight task cache evidence is invalid"
                )
            if any(
                not _is_sha(item.get(field))
                for field in expected_fields
                if field.endswith("sha256")
            ):
                raise FormalPreflightError(
                    "preflight task cache evidence is invalid"
                )
            coarse = item.get("coarse_shape")
            detail = item.get("detail_shape")
            if (
                not isinstance(coarse, (list, tuple))
                or not isinstance(detail, (list, tuple))
                or len(coarse) != 2
                or len(detail) != 2
                or any(type(axis) is not int or axis <= 0 for axis in coarse)
                or any(type(axis) is not int or axis <= 0 for axis in detail)
                or coarse[0] != coarse[1]
                or tuple(detail) != (coarse[0] * 20, coarse[1] * 20)
            ):
                raise FormalPreflightError(
                    "preflight task cache evidence is invalid"
                )
            stage_result[platform] = {
                field: (
                    list(item[field])
                    if field in {"coarse_shape", "detail_shape"}
                    else item[field]
                )
                for field in sorted(expected_fields)
            }
        result[stage] = stage_result
    return result


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
    three_platform_step_sha256: str,
    resume_boundary: Mapping[str, object],
    additional_corridor_margin_m: float,
    task_cache_evidence: Mapping[str, object],
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
    if selected_rollout_horizon != REWARD_V4_FORMAL_ROLLOUT_HORIZON:
        raise FormalPreflightError("preflight rollout horizon is not formal-fixed")
    if not _is_sha(three_platform_step_sha256):
        raise FormalPreflightError("preflight step digest is invalid")
    if (
        not isinstance(additional_corridor_margin_m, (int, float))
        or isinstance(additional_corridor_margin_m, bool)
        or float(additional_corridor_margin_m) != 2.0
    ):
        raise FormalPreflightError("preflight corridor margin must be fixed at 2.0 m")
    if (
        not isinstance(resume_boundary, Mapping)
        or set(resume_boundary) != _RESUME_BOUNDARY_FIELDS
        or resume_boundary.get("schema_version")
        != _MINIMAL_RESUME_BOUNDARY_SCHEMA
        or resume_boundary.get("platforms") != list(PLATFORMS)
        or resume_boundary.get("state_roundtrip") is not True
        or not _is_sha(resume_boundary.get("post_step_observation_sha256"))
        or not _is_sha(resume_boundary.get("resumed_observation_sha256"))
        or not _is_sha(resume_boundary.get("evidence_sha256"))
    ):
        raise FormalPreflightError("preflight resume equivalence is invalid")
    validated_task_cache_evidence = _validate_task_cache_evidence(
        task_cache_evidence
    )
    payload: dict[str, object] = {
        "schema_version": FORMAL_PREFLIGHT_SCHEMA,
        "source_commit": source_commit,
        "cache_manifest_sha256": cache_manifest_sha256,
        "sensor_performance_sha256": sensor_performance_sha256,
        "run_identity": run_identity.to_dict(),
        "schema_identity": {
            "checkpoint": CHECKPOINT_SCHEMA_VERSION,
            "formal_cache": FORMAL_CACHE_SCHEMA,
            "formal_environment_state": (
                FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION
            ),
            "reward": REWARD_SCHEMA_VERSION,
            "reward_evaluation_manifest": (
                REWARD_EVALUATION_MANIFEST_SCHEMA
            ),
            "run_manifest": RUN_MANIFEST_SCHEMA_VERSION,
            "training_metrics": TRAINING_UPDATE_METRICS_SCHEMA,
        },
        "reward_config_sha256": reward_weights_sha256(),
        "evaluation_manifest_sha256": reward_evaluation_manifest_sha256(
            build_reward_v4_evaluation_manifest(
                platforms=tuple(PlatformType),
                scale_buckets=tuple(TaskScaleBucket),
                evaluation_seeds=DEFAULT_REWARD_CONFIG.evaluation_seeds,
            )
        ),
        "formal_environment_state_schema": (
            FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION
        ),
        "scenario_schedule_ids": dict(sorted(scenario_schedule_ids.items())),
        "checks": {name: bool(checks[name]) for name in REQUIRED_PREFLIGHT_CHECKS},
        "timings_seconds": {
            name: float(value) for name, value in sorted(timings_seconds.items())
        },
        "qualified_worker_candidates": list(qualified_worker_candidates),
        "selected_workers": selected_workers,
        "selected_micro_batch": selected_micro_batch,
        "rollout_horizon_candidates": [REWARD_V4_FORMAL_ROLLOUT_HORIZON],
        "selected_rollout_horizon": selected_rollout_horizon,
        "episode_decision_limit": None,
        "three_platform_step_sha256": three_platform_step_sha256,
        "additional_corridor_margin_m": 2.0,
        "task_cache_evidence": validated_task_cache_evidence,
        "resume_boundary": dict(sorted(resume_boundary.items())),
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
    for name in ObservationContractV4.input_names:
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


def _task_identity_evidence(worker: object) -> dict[str, object]:
    episode = getattr(worker, "episode", None)
    geometry = getattr(episode, "task_geometry", None)
    loaded = getattr(episode, "loaded", None)
    scene = getattr(loaded, "scene", None)
    sensor_state = getattr(episode, "sensor_state", None)
    if geometry is None or scene is None or sensor_state is None:
        raise FormalPreflightError("formal task-local worker evidence is missing")
    span = geometry.span_cells
    coarse_shape = tuple(scene.base_canvas.geometry.cells for _ in range(2))
    detail_shape = tuple(sensor_state.coverable_detail_shape)
    task_bounds = geometry.coarse_bounds_half_open
    halo_bounds = geometry.halo_coarse_bounds_half_open
    if (
        coarse_shape != (span, span)
        or detail_shape != (span * 20, span * 20)
        or not (
            halo_bounds[0] <= task_bounds[0] < task_bounds[1] <= halo_bounds[1]
            and halo_bounds[2]
            <= task_bounds[2]
            < task_bounds[3]
            <= halo_bounds[3]
        )
    ):
        raise FormalPreflightError("formal task-local geometry leaked its halo")
    row = {
        "task_geometry_sha256": episode.task_geometry_sha256,
        "task_common_key_sha256": episode.task_common_key_sha256,
        "task_common_artifact_sha256": episode.task_common_artifact_sha256,
        "platform_task_key_sha256": episode.platform_task_key_sha256,
        "platform_task_artifact_sha256": (
            episode.platform_task_artifact_sha256
        ),
        "coarse_shape": list(coarse_shape),
        "detail_shape": list(detail_shape),
    }
    if any(
        not _is_sha(value)
        for name, value in row.items()
        if name.endswith("sha256")
    ):
        raise FormalPreflightError("formal task cache identity is missing")
    return row


def _direct_environment_checks(
    cache: FormalCache, assembly: FormalEnvironmentAssembly
) -> dict[str, object]:
    common_train = cache.manifest.get("exact_common_evaluation", {}).get(
        "splits", {}
    ).get("train")
    if not isinstance(common_train, Mapping) or not isinstance(
        common_train.get("scenario_schedule_id"), str
    ):
        raise FormalPreflightError(
            "formal exact-common training inventory is invalid"
        )
    try:
        builder = replace(
            assembly.factory.builder,
            paired_evaluation=True,
            platform_scenario_schedule_ids=None,
            scenario_schedule_id=str(common_train["scenario_schedule_id"]),
        )
        common_factory = replace(
            assembly.factory,
            scenario_schedule_id=str(common_train["scenario_schedule_id"]),
            builder=builder,
        )
    except TypeError as error:
        raise FormalPreflightError(
            "formal train factory cannot select a common world"
        ) from error
    workers = {
        platform: common_factory.create_for_episode(
            0,
            platform,
            0,
            platform_worker_index=0,
            platform_worker_count=1,
        )
        for platform in PLATFORMS
    }
    next_ground_workers = {
        platform: common_factory.create_for_episode(
            0,
            platform,
            1,
            platform_worker_index=0,
            platform_worker_count=1,
        )
        for platform in ("WHEELED", "LEGGED")
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
        snapshot = worker.episode._snapshot
        task_span = worker.episode.task_geometry.span_cells
        if (
            snapshot is None
            or snapshot.global_map.resolution_m != 4.0
            or snapshot.global_map.width != task_span
            or first.world.global_map.width > 256
            or first.world.global_map.height > 256
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

    hopper_train_count = sum(
        entry.get("split") == "train"
        and entry.get("platform_starts", {})
        .get("HOPPER", {})
        .get("qualified")
        is True
        for entry in cache.manifest["scenes"]
    )
    if hopper_train_count <= 0:
        raise FormalPreflightError("formal Hopper training inventory is empty")
    hopper_available: list[float] = []
    for cursor in (0, hopper_train_count):
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
        "task_cache_evidence": {
            "current_ground": {
                platform: _task_identity_evidence(workers[platform])
                for platform in ("WHEELED", "LEGGED")
            },
            "next_ground": {
                platform: _task_identity_evidence(
                    next_ground_workers[platform]
                )
                for platform in ("WHEELED", "LEGGED")
            },
            "halo_leak_count": 0,
            "full_scene_derived_call_count": 0,
        },
    }


def _validated_worker_allocation(
    worker_allocation: Mapping[str, int],
    *,
    selected_workers: int,
) -> dict[str, int]:
    allocation = {
        platform: worker_allocation[platform]
        for platform in PLATFORMS
        if platform in worker_allocation
    }
    if (
        not allocation
        or set(worker_allocation) != set(allocation)
        or any(
            type(count) is not int or count <= 0
            for count in allocation.values()
        )
        or sum(allocation.values()) != selected_workers
    ):
        raise FormalPreflightError(
            "formal preflight worker allocation differs from calibration"
        )
    return allocation


def _resume_check(
    assembly: FormalEnvironmentAssembly,
    worker_allocation: Mapping[str, int],
) -> dict[str, object]:
    """Verify one post-action environment boundary can be restored exactly."""
    allocation = {platform: 1 for platform in worker_allocation}
    worker_count = len(allocation)
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
                for index in range(worker_count)
            ],
            dtype=torch.int64,
        )
        stepped = first.step(
            ParallelActions(
                candidate_indices=candidate_indices,
                thetas=torch.zeros((worker_count,), dtype=torch.float32),
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
    body = {
        "schema_version": _MINIMAL_RESUME_BOUNDARY_SCHEMA,
        "platforms": list(PLATFORMS),
        "post_step_observation_sha256": active_digest,
        "resumed_observation_sha256": resumed_digest,
        "state_roundtrip": True,
    }
    body["evidence_sha256"] = hashlib.sha256(
        _canonical_bytes(body).rstrip(b"\n")
    ).hexdigest()
    return body


def _three_platform_macro_step_probe(
    assembly: FormalEnvironmentAssembly,
    worker_allocation: Mapping[str, int],
) -> str:
    """Execute one real macro action per platform without training a policy."""
    allocation = {platform: 1 for platform in worker_allocation}
    platforms = tuple(allocation)
    with ParallelEnvPool(
        allocation=allocation,
        observation_template=assembly.observation_template,
        environment_factory=assembly.factory,
        reward_fn=compute_transition_reward,
        worker_timeout_seconds=FORMAL_WORKER_RESPONSE_TIMEOUT_SECONDS,
    ) as pool:
        initial = pool.reset()
        candidate_indices = torch.tensor(
            [
                int(initial.observations.candidate_mask[index].nonzero()[0])
                for index in range(len(platforms))
            ],
            dtype=torch.int64,
        )
        stepped = pool.step(
            ParallelActions(
                candidate_indices=candidate_indices,
                thetas=torch.zeros((len(platforms),), dtype=torch.float32),
            ),
            policy_version=0,
        )
    consumed = stepped.policy_decisions_consumed.detach().cpu().tolist()
    if consumed != [1] * len(platforms):
        raise FormalPreflightError(
            "three-platform preflight step did not consume exactly one action"
        )
    body = {
        "platforms": list(platforms),
        "initial_observation_sha256": _batch_digest(initial.observations),
        "post_step_observation_sha256": _batch_digest(stepped.observations),
        "candidate_indices": candidate_indices.detach().cpu().tolist(),
        "planning_outcomes": [
            outcome.name for outcome in stepped.planning_outcomes
        ],
        "reason_codes": list(stepped.reason_codes),
        "policy_decisions_consumed": consumed,
    }
    return hashlib.sha256(_canonical_bytes(body).rstrip(b"\n")).hexdigest()


def run_formal_preflight(
    *,
    cache: FormalCache,
    assemblies: Mapping[str, FormalEnvironmentAssembly],
    run_identity: RunIdentity,
    source_commit: str,
    sensor_performance_sha256: str,
    artifact_root: Path,
    worker_allocation: Mapping[str, int],
    selected_workers: int | None = None,
    selected_micro_batch: int = 2,
    selected_rollout_horizon: int = REWARD_V4_FORMAL_ROLLOUT_HORIZON,
) -> tuple[FormalPreflightReport, Path]:
    """Execute the bounded real-environment preflight before training."""
    expected_splits = {"train", "validation", "test", "holdout"}
    if set(assemblies) != expected_splits:
        raise FormalPreflightError("formal preflight assemblies are incomplete")
    if selected_workers is None:
        selected_workers = sum(worker_allocation.values())
    allocation = _validated_worker_allocation(
        worker_allocation,
        selected_workers=selected_workers,
    )
    timings: dict[str, float] = {}
    started = time.monotonic()
    direct_evidence = _direct_environment_checks(cache, assemblies["train"])
    timings["direct_environment"] = time.monotonic() - started

    started = time.monotonic()
    resume_boundary = _resume_check(assemblies["train"], allocation)
    timings["single_worker_resume_boundary"] = time.monotonic() - started

    started = time.monotonic()
    three_platform_step_sha256 = _three_platform_macro_step_probe(
        assemblies["train"], allocation
    )
    timings["three_platform_macro_step"] = time.monotonic() - started

    checks = {name: True for name in REQUIRED_PREFLIGHT_CHECKS}
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
        qualified_worker_candidates=(selected_workers,),
        selected_workers=selected_workers,
        selected_micro_batch=selected_micro_batch,
        selected_rollout_horizon=selected_rollout_horizon,
        three_platform_step_sha256=three_platform_step_sha256,
        resume_boundary=resume_boundary,
        additional_corridor_margin_m=float(
            direct_evidence["additional_corridor_margin_m"]
        ),
        task_cache_evidence=direct_evidence["task_cache_evidence"],
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
