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


def _freeze_descriptor(value):
    """Own serializable generator metadata, including nested lists/mappings."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_descriptor(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_descriptor(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("generator descriptor must contain serializable primitive metadata")


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
    remaining_area_upper_m2: float | None = None
    coverage_target: float = 1.0

    @property
    def coverage_lower_bound(self):
        if not self.available or self.remaining_area_upper_m2 is None:
            return None
        total=self.known_area_m2+self.remaining_area_upper_m2
        return self.known_area_m2/total if total>0 else None

    @property
    def completed(self):
        """Normal task end by the target bound or proved exhaustion.

        An empty observable task may be exhausted with an undefined percentage;
        completed does not claim that such a task achieved a numeric target.
        """
        lower=self.coverage_lower_bound
        return self.available and (self.exhausted or
            (lower is not None and lower>=self.coverage_target))

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
    schema: str = "task_graph_v3"

    def __post_init__(self):
        for name, dtype in (("node_ids", np.int64), ("positions", np.float32),
                            ("features", np.float32), ("edges", np.int64),
                            ("edge_lengths", np.float32), ("polygon", np.float32),
                            ("context", np.float32), ("action_nodes", np.int64),
                            ("action_yaws", np.float32), ("goals", np.float64)):
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
class PrivilegedActionContext:
    positions: np.ndarray
    action_positions: np.ndarray
    action_yaws: np.ndarray
    support_offsets: np.ndarray
    support_indices: np.ndarray
    support_distances: np.ndarray
    gains: np.ndarray

    def __post_init__(self):
        for name, dtype in (("positions", np.float64), ("action_positions", np.int64),
                            ("action_yaws", np.float64),
                            ("support_offsets", np.int64), ("support_indices", np.int64),
                            ("support_distances", np.float32), ("gains", np.float32)):
            object.__setattr__(self, name, freeze_array(getattr(self, name), dtype=dtype))
        n = len(self.positions)
        if self.positions.shape != (n, 2) or not np.isfinite(self.positions).all():
            raise ValueError("candidate positions must be finite N x 2 coordinates")
        if n and len(np.unique(self.positions, axis=0)) != n:
            raise ValueError("candidate positions must be unique")
        if (self.action_positions.ndim != 1 or self.action_yaws.shape != self.action_positions.shape or
                self.gains.shape != self.action_positions.shape or not np.isfinite(self.action_yaws).all()):
            raise ValueError("candidate action mapping, yaws and gains must align")
        if (np.any(self.action_positions < 0) or np.any(self.action_positions >= n) or
                not np.isfinite(self.gains).all() or np.any(self.gains < 0)):
            raise ValueError("invalid candidate action mapping or gain")
        if (self.support_offsets.shape != (n + 1,) or self.support_offsets[0] != 0 or
                np.any(np.diff(self.support_offsets) <= 0) or
                self.support_offsets[-1] != len(self.support_indices) or
                self.support_distances.shape != self.support_indices.shape or
                np.any(self.support_indices < 0) or not np.isfinite(self.support_distances).all() or
                np.any(self.support_distances < 0)):
            raise ValueError("invalid candidate support CSR")


@dataclass(frozen=True)
class PrivilegedState:
    scene_id: str
    observed: np.ndarray
    actions: PrivilegedActionContext | None = None

    def __post_init__(self):
        object.__setattr__(self, "observed", freeze_array(self.observed, dtype=np.uint8))


@dataclass(frozen=True)
class PrivilegedScene:
    """Owned static critic graph; packed cells use row-major little bit order.

    The CSR mapping assigns each reference cell to its nearest static graph
    node. It supports observed-fraction pooling without retaining terrain.
    """
    scene_id: str
    positions: np.ndarray
    edges: np.ndarray
    edge_lengths: np.ndarray
    reference_offsets: np.ndarray
    reference_indices: np.ndarray
    packed_reference: np.ndarray
    reference_shape: tuple
    generator_descriptor: Mapping

    def __post_init__(self):
        for name, dtype in (("positions", np.float32), ("edges", np.int64),
                            ("edge_lengths", np.float32), ("reference_offsets", np.int64),
                            ("reference_indices", np.int64), ("packed_reference", np.uint8)):
            object.__setattr__(self, name, freeze_array(getattr(self, name), dtype=dtype))
        object.__setattr__(self, "generator_descriptor", _freeze_descriptor(self.generator_descriptor))


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
