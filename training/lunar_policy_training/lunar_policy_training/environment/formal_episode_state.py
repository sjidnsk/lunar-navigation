"""Strict JSON state for replaying one active formal exploration episode."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ..policy.observation import ObservationIdentity, PolicyBatch


FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION = "lunar-formal-environment-state/v2"
STABLE_EXECUTION_STATES = frozenset(
    {"DECISION_BOUNDARY", "GROUND_HOLD", "LANDED_HOLD"}
)
_PLATFORMS = frozenset({"WHEELED", "LEGGED", "HOPPER"})
_POSE_FIELDS = frozenset(
    {"x_m", "y_m", "yaw_rad", "elevation_m", "frame_id"}
)
_REVEAL_FIELDS = frozenset(
    {"pose", "elapsed_s", "execution_state", "legged_body_z_m"}
)
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
        "start_cell",
        "current_pose",
        "legged_body_z_m",
        "execution_state",
        "observation_revision",
        "state_time_ns",
        "reveal_history",
        "observation_identity",
        "policy_batch_sha256",
        "rejected_candidate_indices",
        "last_hop_available_delta_v_mps",
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
class FormalRevealState:
    pose: FormalPoseState
    elapsed_s: float
    execution_state: str
    legged_body_z_m: float

    @classmethod
    def from_dict(cls, value: object) -> "FormalRevealState":
        if not isinstance(value, Mapping) or set(value) != _REVEAL_FIELDS:
            raise ValueError("formal reveal structure is invalid")
        execution_state = value["execution_state"]
        if execution_state not in STABLE_EXECUTION_STATES:
            raise ValueError("formal reveal execution state is not stable")
        elapsed_s = _finite(value["elapsed_s"], "reveal elapsed time")
        if elapsed_s < 0.0:
            raise ValueError("formal reveal elapsed time must be non-negative")
        return cls(
            pose=FormalPoseState.from_dict(value["pose"]),
            elapsed_s=elapsed_s,
            execution_state=str(execution_state),
            legged_body_z_m=_finite(
                value["legged_body_z_m"], "legged body height"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pose": self.pose.to_dict(),
            "elapsed_s": self.elapsed_s,
            "execution_state": self.execution_state,
            "legged_body_z_m": self.legged_body_z_m,
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
    start_cell: tuple[int, int]
    current_pose: FormalPoseState
    legged_body_z_m: float
    execution_state: str
    observation_revision: int
    state_time_ns: int
    reveal_history: tuple[FormalRevealState, ...]
    observation_identity: ObservationIdentity
    policy_batch_sha256: str
    rejected_candidate_indices: tuple[int, ...]
    last_hop_available_delta_v_mps: float

    @classmethod
    def from_dict(cls, value: object) -> "FormalWorkerState":
        if not isinstance(value, Mapping) or set(value) != _WORKER_FIELDS:
            raise ValueError("formal worker state structure is invalid")
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
        if revision != len(history) + 1:
            raise ValueError("formal observation revision does not match reveal history")
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
        candidates = value["rejected_candidate_indices"]
        if (
            not isinstance(candidates, list)
            or any(type(item) is not int or not 0 <= item < 64 for item in candidates)
            or candidates != sorted(set(candidates))
        ):
            raise ValueError("formal rejected candidate indices are invalid")
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
            start_cell=(start_cell[0], start_cell[1]),
            current_pose=current_pose,
            legged_body_z_m=_finite(
                value["legged_body_z_m"], "legged body height"
            ),
            execution_state=str(execution_state),
            observation_revision=revision,
            state_time_ns=state_time_ns,
            reveal_history=history,
            observation_identity=identity,
            policy_batch_sha256=_sha256(
                value["policy_batch_sha256"], "policy batch digest"
            ),
            rejected_candidate_indices=tuple(candidates),
            last_hop_available_delta_v_mps=last_delta_v,
        )

    def to_dict(self) -> dict[str, object]:
        return {
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
            "start_cell": list(self.start_cell),
            "current_pose": self.current_pose.to_dict(),
            "legged_body_z_m": self.legged_body_z_m,
            "execution_state": self.execution_state,
            "observation_revision": self.observation_revision,
            "state_time_ns": self.state_time_ns,
            "reveal_history": [item.to_dict() for item in self.reveal_history],
            "observation_identity": observation_identity_to_dict(
                self.observation_identity
            ),
            "policy_batch_sha256": self.policy_batch_sha256,
            "rejected_candidate_indices": list(self.rejected_candidate_indices),
            "last_hop_available_delta_v_mps": self.last_hop_available_delta_v_mps,
        }


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
    "FormalPoseState",
    "FormalRevealState",
    "FormalWorkerState",
    "STABLE_EXECUTION_STATES",
    "observation_identity_from_dict",
    "observation_identity_to_dict",
    "policy_batch_sha256",
]
