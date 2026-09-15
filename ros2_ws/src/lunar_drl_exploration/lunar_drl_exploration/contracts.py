"""Immutable, transport-independent policy and replay contracts."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np


def freeze_array(value, *, dtype=None) -> np.ndarray:
    """Copy an array into an immutable bytes-backed NumPy view.

    ``setflags(write=False)`` alone is insufficient because a writable owner
    can later be recovered.  A bytes object has no writable backing store.
    """
    array = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    return frozen


def freeze_mapping(value: Mapping) -> Mapping:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    frame_id: str
    polygon: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "polygon", freeze_array(self.polygon, dtype=np.float32))


@dataclass(frozen=True)
class SensorSpec:
    range_m: float = 10.0
    fov_deg: float = 90.0
    offset_x_m: float = 0.0
    offset_y_m: float = 0.0
    offset_yaw_rad: float = 0.0


@dataclass(frozen=True)
class MapSnapshot:
    epoch: str
    revision: int
    resolution_m: float
    origin: tuple
    tiles: Mapping
    pose: Pose
    start_connections: np.ndarray
    local_bounds: tuple
    profile_hash: str
    start_connection_status: str = "READY"
    goal_position_tolerance_m: float = float("nan")
    goal_yaw_tolerance_rad: float = float("nan")

    def __post_init__(self):
        object.__setattr__(self, "tiles", freeze_mapping(self.tiles))
        object.__setattr__(self, "start_connections",
                           freeze_array(self.start_connections, dtype=np.int64))


@dataclass(frozen=True)
class TaskReport:
    known_area_m2: float
    new_area_m2: float
    frontier_cells: np.ndarray
    witnesses: np.ndarray
    exhausted: bool
    revision: int
    available: bool = True
    reason_code: str = "READY"

    def __post_init__(self):
        object.__setattr__(self, "frontier_cells",
                           freeze_array(self.frontier_cells, dtype=np.int64))
        object.__setattr__(self, "witnesses", freeze_array(self.witnesses, dtype=np.int64))


@dataclass(frozen=True)
class DecisionObservation:
    node_ids: np.ndarray
    positions: np.ndarray
    features: np.ndarray
    edges: np.ndarray
    edge_lengths: np.ndarray
    current_index: int
    polygon: np.ndarray
    context: np.ndarray
    action_nodes: np.ndarray
    action_yaws: np.ndarray
    goals: np.ndarray
    epoch: str
    revision: int
    schema: str = "task_graph_v1"

    def __post_init__(self):
        for name, dtype in (("node_ids", np.int64), ("positions", np.float32),
                            ("features", np.float32), ("edges", np.int64),
                            ("edge_lengths", np.float32), ("polygon", np.float32),
                            ("context", np.float32), ("action_nodes", np.int64),
                            ("action_yaws", np.float32), ("goals", np.float32)):
            object.__setattr__(self, name, freeze_array(getattr(self, name), dtype=dtype))
        if self.features.ndim != 2 or self.features.shape[1] != 19:
            raise ValueError("features must contain 19 scalars per graph node")
        if self.context.shape != (8,):
            raise ValueError("context must contain 8 platform/sensor scalars")
        if self.node_ids.shape != (self.features.shape[0],):
            raise ValueError("node_ids must align with graph features")
        if not 0 <= self.current_index < self.node_ids.size:
            raise ValueError("current_index is outside graph nodes")
        if self.goals.ndim != 2 or self.goals.shape[1] != 3:
            raise ValueError("goals must contain world x, y, yaw")
        if (self.action_nodes.ndim != 1 or self.action_yaws.ndim != 1 or
                self.goals.shape[0] != self.action_nodes.size or
                self.action_yaws.size != self.action_nodes.size):
            raise ValueError("each action requires one yaw and frozen world goal")
        if np.any(self.action_nodes < 0) or np.any(self.action_nodes >= self.node_ids.size):
            raise ValueError("action nodes must index graph nodes")


@dataclass(frozen=True)
class PrivilegedState:
    scene_id: str
    observed: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "observed", freeze_array(self.observed, dtype=np.uint8))


@dataclass(frozen=True)
class RewardParts:
    new_area_m2: float
    distance_m: float
    turn_rad: float


@dataclass(frozen=True)
class Transition:
    observation: DecisionObservation
    action: int
    reward: float
    next_observation: DecisionObservation
    privileged: PrivilegedState
    next_privileged: PrivilegedState
    parts: RewardParts
    terminated: bool
    truncated: bool
    episode_id: str
    actor_version: int
