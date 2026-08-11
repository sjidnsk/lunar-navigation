"""Strict JSON state for replaying one active formal exploration episode."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ..policy.observation import ObservationIdentity, PolicyBatch


FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION = "lunar-formal-environment-state/v6"
STABLE_EXECUTION_STATES = frozenset(
    {"DECISION_BOUNDARY", "GROUND_HOLD", "LANDED_HOLD"}
)
_PLATFORMS = frozenset({"WHEELED", "LEGGED", "HOPPER"})
_POSE_FIELDS = frozenset(
    {"x_m", "y_m", "yaw_rad", "elevation_m", "frame_id"}
)
_REVEAL_FIELDS = frozenset(
    {
        "pose",
        "elapsed_s",
        "path_samples",
        "execution_state",
        "legged_body_z_m",
        "defer_candidate_rebuild",
    }
)
_PATH_SAMPLE_FIELDS = frozenset({"pose", "elapsed_s"})
_IDENTITY_FIELDS = frozenset(
    {
        "episode_id",
        "mission_revision",
        "map_snapshot_id",
        "robot_state_id",
        "state_time_ns",
        "execution_state",
        "candidate_set_id",
    }
)
_WORKER_FIELDS = frozenset(
    {
        "scenario_schedule_id",
        "platform_type",
        "worker_index",
        "platform_worker_index",
        "platform_worker_count",
        "episode_cursor",
        "scene_id",
        "scene_seed",
        "start_seed",
        "episode_seed",
        "coverability_mask_sha256",
        "start_cell",
        "current_pose",
        "legged_body_z_m",
        "execution_state",
        "observation_revision",
        "physical_snapshot_id",
        "physical_evidence_generation",
        "physical_evidence_sha256",
        "physical_candidate_universe_sha256",
        "planner_failed_candidate_ids",
        "state_time_ns",
        "reveal_history",
        "observation_identity",
        "policy_batch_sha256",
        "candidate_ids",
        "candidate_mask",
        "oracle_opportunity_count",
        "oracle_opportunity_set_sha256",
        "terminal_reason",
        "defer_candidate_rebuild",
        "last_hop_available_delta_v_mps",
        "candidate_gain_resolution_m",
    }
)
_CANDIDATE_GAIN_RESOLUTION_M = 0.2
_TERMINAL_REASONS = frozenset(
    {
        "SUCCESS",
        "NO_RECOVERABLE_OBSERVATION_STATE",
        "VISITED_EXHAUSTED",
        "ZERO_GAIN",
        "NO_TRANSIT_OPPORTUNITY",
        "PLANNER_BLOCKED_WITH_OPPORTUNITY",
        "HARD_FAILURE",
        "CANCELED",
    }
)


def _finite(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"formal {name} must be finite")
    return float(value)


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"formal {name} must be a non-negative integer")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"formal {name} must be a lowercase SHA-256")
    return value


def _physical_sha256(value: object, name: str) -> str:
    digest = _sha256(value, name)
    if digest == "0" * 64:
        raise ValueError(f"formal {name} must not use a promoted default")
    return digest


@dataclass(frozen=True, slots=True)
class FormalPoseState:
    x_m: float
    y_m: float
    yaw_rad: float
    elevation_m: float
    frame_id: str = "map"

    @classmethod
    def from_dict(cls, value: object) -> "FormalPoseState":
        if not isinstance(value, Mapping) or set(value) != _POSE_FIELDS:
            raise ValueError("formal pose structure is invalid")
        if value["frame_id"] != "map":
            raise ValueError("formal pose frame must be map")
        return cls(
            x_m=_finite(value["x_m"], "pose x"),
            y_m=_finite(value["y_m"], "pose y"),
            yaw_rad=_finite(value["yaw_rad"], "pose yaw"),
            elevation_m=_finite(value["elevation_m"], "pose elevation"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "yaw_rad": self.yaw_rad,
            "elevation_m": self.elevation_m,
            "frame_id": self.frame_id,
        }


@dataclass(frozen=True, slots=True)
class FormalPathSampleState:
    pose: FormalPoseState
    elapsed_s: float

    @classmethod
    def from_dict(cls, value: object) -> "FormalPathSampleState":
        if not isinstance(value, Mapping) or set(value) != _PATH_SAMPLE_FIELDS:
            raise ValueError("formal path sample structure is invalid")
        elapsed_s = _finite(value["elapsed_s"], "path sample elapsed time")
        if elapsed_s < 0.0:
            raise ValueError(
                "formal path sample elapsed time must be non-negative"
            )
        return cls(
            pose=FormalPoseState.from_dict(value["pose"]),
            elapsed_s=elapsed_s,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pose": self.pose.to_dict(),
            "elapsed_s": self.elapsed_s,
        }


@dataclass(frozen=True, slots=True)
class FormalRevealState:
    pose: FormalPoseState
    elapsed_s: float
    path_samples: tuple[FormalPathSampleState, ...]
    execution_state: str
    legged_body_z_m: float
    defer_candidate_rebuild: bool

    @classmethod
    def from_dict(cls, value: object) -> "FormalRevealState":
        if not isinstance(value, Mapping) or set(value) != _REVEAL_FIELDS:
            raise ValueError("formal reveal structure is invalid")
        execution_state = value["execution_state"]
        if execution_state not in STABLE_EXECUTION_STATES:
            raise ValueError("formal reveal execution state is not stable")
        if type(value["defer_candidate_rebuild"]) is not bool:
            raise ValueError("formal reveal candidate rebuild flag is invalid")
        elapsed_s = _finite(value["elapsed_s"], "reveal elapsed time")
        if elapsed_s < 0.0:
            raise ValueError("formal reveal elapsed time must be non-negative")
        samples_value = value["path_samples"]
        if not isinstance(samples_value, list):
            raise ValueError("formal path sample list is invalid")
        path_samples = tuple(
            FormalPathSampleState.from_dict(item) for item in samples_value
        )
        pose = FormalPoseState.from_dict(value["pose"])
        if path_samples and path_samples[-1].pose != pose:
            raise ValueError("formal final path sample differs from reveal pose")
        elapsed_ns = int(round(elapsed_s * 1_000_000_000.0))
        sample_elapsed_ns = sum(
            int(round(sample.elapsed_s * 1_000_000_000.0))
            for sample in path_samples
        )
        if path_samples and sample_elapsed_ns != elapsed_ns:
            raise ValueError("formal path sample elapsed time differs from reveal")
        return cls(
            pose=pose,
            elapsed_s=elapsed_s,
            path_samples=path_samples,
            execution_state=str(execution_state),
            legged_body_z_m=_finite(
                value["legged_body_z_m"], "legged body height"
            ),
            defer_candidate_rebuild=value["defer_candidate_rebuild"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pose": self.pose.to_dict(),
            "elapsed_s": self.elapsed_s,
            "path_samples": [sample.to_dict() for sample in self.path_samples],
            "execution_state": self.execution_state,
            "legged_body_z_m": self.legged_body_z_m,
            "defer_candidate_rebuild": self.defer_candidate_rebuild,
        }


def observation_identity_to_dict(
    identity: ObservationIdentity,
) -> dict[str, object]:
    if not isinstance(identity, ObservationIdentity):
        raise ValueError("formal observation identity is invalid")
    return {
        "episode_id": identity.episode_id,
        "mission_revision": identity.mission_revision,
        "map_snapshot_id": identity.map_snapshot_id,
        "robot_state_id": identity.robot_state_id,
        "state_time_ns": identity.state_time_ns,
        "execution_state": identity.execution_state,
        "candidate_set_id": identity.candidate_set_id,
    }


def observation_identity_from_dict(value: object) -> ObservationIdentity:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_FIELDS:
        raise ValueError("formal observation identity structure is invalid")
    for name in (
        "episode_id",
        "map_snapshot_id",
        "robot_state_id",
        "execution_state",
        "candidate_set_id",
    ):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"formal observation identity {name} is invalid")
    return ObservationIdentity(
        episode_id=str(value["episode_id"]),
        mission_revision=_nonnegative_int(
            value["mission_revision"], "mission revision"
        ),
        map_snapshot_id=str(value["map_snapshot_id"]),
        robot_state_id=str(value["robot_state_id"]),
        state_time_ns=_nonnegative_int(value["state_time_ns"], "state time"),
        execution_state=str(value["execution_state"]),
        candidate_set_id=str(value["candidate_set_id"]),
    )


@dataclass(frozen=True, slots=True)
class FormalWorkerState:
    scenario_schedule_id: str
    platform_type: str
    worker_index: int
    platform_worker_index: int
    platform_worker_count: int
    episode_cursor: int
    scene_id: str
    scene_seed: str
    start_seed: str
    episode_seed: str
    coverability_mask_sha256: str
    start_cell: tuple[int, int]
    current_pose: FormalPoseState
    legged_body_z_m: float
    execution_state: str
    observation_revision: int
    physical_snapshot_id: str
    physical_evidence_generation: int
    physical_evidence_sha256: str
    physical_candidate_universe_sha256: str
    planner_failed_candidate_ids: tuple[str, ...]
    state_time_ns: int
    reveal_history: tuple[FormalRevealState, ...]
    observation_identity: ObservationIdentity
    policy_batch_sha256: str
    candidate_ids: tuple[str, ...]
    candidate_mask: tuple[bool, ...]
    oracle_opportunity_count: int
    oracle_opportunity_set_sha256: str
    terminal_reason: str | None
    defer_candidate_rebuild: bool
    last_hop_available_delta_v_mps: float
    candidate_gain_resolution_m: float

    @classmethod
    def from_dict(cls, value: object) -> "FormalWorkerState":
        if not isinstance(value, Mapping) or set(value) != _WORKER_FIELDS:
            raise ValueError("formal worker state structure is invalid")
        candidate_gain_resolution_m = _finite(
            value["candidate_gain_resolution_m"],
            "candidate gain resolution",
        )
        if candidate_gain_resolution_m != _CANDIDATE_GAIN_RESOLUTION_M:
            raise ValueError("formal candidate gain resolution is invalid")
        schedule_id = value["scenario_schedule_id"]
        if not isinstance(schedule_id, str) or not schedule_id:
            raise ValueError("formal scenario schedule identity is invalid")
        platform_type = value["platform_type"]
        if platform_type not in _PLATFORMS:
            raise ValueError("formal worker platform is invalid")
        worker_index = _nonnegative_int(value["worker_index"], "worker index")
        lane = _nonnegative_int(
            value["platform_worker_index"], "platform worker index"
        )
        lane_count = _nonnegative_int(
            value["platform_worker_count"], "platform worker count"
        )
        if lane_count == 0 or lane >= lane_count:
            raise ValueError("formal platform worker lane is invalid")
        execution_state = value["execution_state"]
        if execution_state not in STABLE_EXECUTION_STATES:
            raise ValueError("formal worker execution state is not stable")
        start_cell = value["start_cell"]
        if (
            not isinstance(start_cell, list)
            or len(start_cell) != 2
            or any(type(item) is not int or item < 0 for item in start_cell)
        ):
            raise ValueError("formal start cell is invalid")
        history_value = value["reveal_history"]
        if not isinstance(history_value, list):
            raise ValueError("formal reveal history structure is invalid")
        history = tuple(FormalRevealState.from_dict(item) for item in history_value)
        revision = _nonnegative_int(
            value["observation_revision"], "observation revision"
        )
        failed_value = value["planner_failed_candidate_ids"]
        if (
            not isinstance(failed_value, list)
            or any(not isinstance(item, str) for item in failed_value)
            or any(_sha256(item, "planner failed candidate ID") != item for item in failed_value)
            or failed_value != sorted(set(failed_value))
        ):
            raise ValueError("formal planner failed candidate IDs are invalid")
        expected_revision = len(history) + len(failed_value) + 1
        if revision != expected_revision:
            raise ValueError(
                "formal observation revision does not match replay history"
            )
        evidence_generation = _nonnegative_int(
            value["physical_evidence_generation"],
            "physical evidence generation",
        )
        expected_generation = 1 + sum(
            len(item.path_samples) if item.path_samples else 1
            for item in history
        )
        if evidence_generation != expected_generation:
            raise ValueError(
                "formal physical evidence generation does not match reveal history"
            )
        state_time_ns = _nonnegative_int(value["state_time_ns"], "state time")
        expected_time_ns = 1_000_000_000 + sum(
            int(round(item.elapsed_s * 1_000_000_000.0)) for item in history
        )
        if state_time_ns != expected_time_ns:
            raise ValueError("formal state time does not match reveal history")
        identity = observation_identity_from_dict(value["observation_identity"])
        if (
            identity.state_time_ns != state_time_ns
            or identity.execution_state != execution_state
        ):
            raise ValueError("formal observation identity differs from worker state")
        current_pose = FormalPoseState.from_dict(value["current_pose"])
        if history and history[-1].pose != current_pose:
            raise ValueError("formal current pose differs from reveal history")
        candidate_ids = value["candidate_ids"]
        candidate_mask = value["candidate_mask"]
        if (
            not isinstance(candidate_ids, list)
            or len(candidate_ids) != 64
            or not isinstance(candidate_mask, list)
            or len(candidate_mask) != 64
            or any(type(item) is not bool for item in candidate_mask)
            or any(not isinstance(item, str) for item in candidate_ids)
            or any(
                (_sha256(candidate_id, "candidate ID") if enabled else candidate_id)
                != candidate_id
                for candidate_id, enabled in zip(
                    candidate_ids, candidate_mask, strict=True
                )
            )
            or any(
                not enabled and candidate_id != ""
                for candidate_id, enabled in zip(
                    candidate_ids, candidate_mask, strict=True
                )
            )
            or len(
                {
                    candidate_id
                    for candidate_id, enabled in zip(
                        candidate_ids, candidate_mask, strict=True
                    )
                    if enabled
                }
            )
            != sum(candidate_mask)
        ):
            raise ValueError("formal candidate IDs or mask are invalid")
        oracle_count = _nonnegative_int(
            value["oracle_opportunity_count"], "oracle opportunity count"
        )
        terminal_reason = value["terminal_reason"]
        if terminal_reason is not None and terminal_reason not in _TERMINAL_REASONS:
            raise ValueError("formal terminal reason is invalid")
        if bool(any(candidate_mask)) != (terminal_reason is None):
            raise ValueError("formal terminal decision differs from availability")
        if type(value["defer_candidate_rebuild"]) is not bool:
            raise ValueError("formal defer candidate rebuild flag is invalid")
        last_delta_v = _finite(
            value["last_hop_available_delta_v_mps"], "hopper delta-v"
        )
        if last_delta_v < 0.0:
            raise ValueError("formal hopper delta-v must be non-negative")
        return cls(
            scenario_schedule_id=schedule_id,
            platform_type=str(platform_type),
            worker_index=worker_index,
            platform_worker_index=lane,
            platform_worker_count=lane_count,
            episode_cursor=_nonnegative_int(value["episode_cursor"], "episode cursor"),
            scene_id=_sha256(value["scene_id"], "scene ID"),
            scene_seed=_sha256(value["scene_seed"], "scene seed"),
            start_seed=_sha256(value["start_seed"], "start seed"),
            episode_seed=_sha256(value["episode_seed"], "episode seed"),
            coverability_mask_sha256=_sha256(
                value["coverability_mask_sha256"], "coverability mask"
            ),
            start_cell=(start_cell[0], start_cell[1]),
            current_pose=current_pose,
            legged_body_z_m=_finite(
                value["legged_body_z_m"], "legged body height"
            ),
            execution_state=str(execution_state),
            observation_revision=revision,
            physical_snapshot_id=_physical_sha256(
                value["physical_snapshot_id"], "physical snapshot ID"
            ),
            physical_evidence_generation=evidence_generation,
            physical_evidence_sha256=_physical_sha256(
                value["physical_evidence_sha256"], "physical evidence digest"
            ),
            physical_candidate_universe_sha256=_physical_sha256(
                value["physical_candidate_universe_sha256"],
                "physical candidate universe digest",
            ),
            planner_failed_candidate_ids=tuple(failed_value),
            state_time_ns=state_time_ns,
            reveal_history=history,
            observation_identity=identity,
            policy_batch_sha256=_sha256(
                value["policy_batch_sha256"], "policy batch digest"
            ),
            candidate_ids=tuple(candidate_ids),
            candidate_mask=tuple(candidate_mask),
            oracle_opportunity_count=oracle_count,
            oracle_opportunity_set_sha256=_sha256(
                value["oracle_opportunity_set_sha256"],
                "oracle opportunity set digest",
            ),
            terminal_reason=terminal_reason,
            defer_candidate_rebuild=value["defer_candidate_rebuild"],
            last_hop_available_delta_v_mps=last_delta_v,
            candidate_gain_resolution_m=candidate_gain_resolution_m,
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "scenario_schedule_id": self.scenario_schedule_id,
            "platform_type": self.platform_type,
            "worker_index": self.worker_index,
            "platform_worker_index": self.platform_worker_index,
            "platform_worker_count": self.platform_worker_count,
            "episode_cursor": self.episode_cursor,
            "scene_id": self.scene_id,
            "scene_seed": self.scene_seed,
            "start_seed": self.start_seed,
            "episode_seed": self.episode_seed,
            "coverability_mask_sha256": self.coverability_mask_sha256,
            "start_cell": list(self.start_cell),
            "current_pose": self.current_pose.to_dict(),
            "legged_body_z_m": self.legged_body_z_m,
            "execution_state": self.execution_state,
            "observation_revision": self.observation_revision,
            "physical_snapshot_id": self.physical_snapshot_id,
            "physical_evidence_generation": self.physical_evidence_generation,
            "physical_evidence_sha256": self.physical_evidence_sha256,
            "physical_candidate_universe_sha256": (
                self.physical_candidate_universe_sha256
            ),
            "planner_failed_candidate_ids": list(
                self.planner_failed_candidate_ids
            ),
            "state_time_ns": self.state_time_ns,
            "reveal_history": [item.to_dict() for item in self.reveal_history],
            "observation_identity": observation_identity_to_dict(
                self.observation_identity
            ),
            "policy_batch_sha256": self.policy_batch_sha256,
            "candidate_ids": list(self.candidate_ids),
            "candidate_mask": list(self.candidate_mask),
            "oracle_opportunity_count": self.oracle_opportunity_count,
            "oracle_opportunity_set_sha256": (
                self.oracle_opportunity_set_sha256
            ),
            "terminal_reason": self.terminal_reason,
            "defer_candidate_rebuild": self.defer_candidate_rebuild,
            "last_hop_available_delta_v_mps": self.last_hop_available_delta_v_mps,
            "candidate_gain_resolution_m": self.candidate_gain_resolution_m,
        }
        return payload


def policy_batch_sha256(batch: PolicyBatch) -> str:
    """Hash all seven contract tensors without including process-local storage."""
    if not isinstance(batch, PolicyBatch):
        raise ValueError("formal policy batch digest requires PolicyBatch")
    digest = hashlib.sha256()
    for name in (
        "prior_channels",
        "coverage_summary",
        "local_crop",
        "frontier_features",
        "pose_features",
        "candidate_mask",
        "platform_context",
    ):
        tensor = getattr(batch, name)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("formal policy batch contains a non-tensor field")
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION",
    "FormalPathSampleState",
    "FormalPoseState",
    "FormalRevealState",
    "FormalWorkerState",
    "STABLE_EXECUTION_STATES",
    "observation_identity_from_dict",
    "observation_identity_to_dict",
    "policy_batch_sha256",
]
