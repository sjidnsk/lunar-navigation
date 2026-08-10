"""Observed-only ownership boundary for native primitive reachability graphs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


_PLATFORMS = frozenset(("WHEELED", "LEGGED", "HOPPER"))
_SHA256_LENGTH = 64
_NATIVE_UNSAFE_START_REASONS = {
    "WHEELED": frozenset(("WHEEL_START_NOT_SAFE",)),
    "LEGGED": frozenset(("LEGGED_START_NOT_SAFE",)),
    "HOPPER": frozenset(
        (
            "HOPPER_START_NOT_SAFE",
            "HOPPER_START_LANDING_NOT_CERTIFIED",
        )
    ),
}


def native_start_failure_is_ineligible(
    platform_type: str,
    error: BaseException,
) -> bool:
    """Classify only native failures proving that one start is unusable."""
    return (
        platform_type in _NATIVE_UNSAFE_START_REASONS
        and isinstance(error, RuntimeError)
        and str(error) in _NATIVE_UNSAFE_START_REASONS[platform_type]
    )


def _sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"observed primitive {name} is invalid")
    return value


def _readonly_array(
    name: str,
    value: object,
    *,
    dtype: np.dtype,
    shape: tuple[int | None, ...],
) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype != dtype or source.ndim != len(shape):
        raise ValueError(f"observed primitive {name} has invalid dtype or rank")
    if any(
        expected is not None and source.shape[index] != expected
        for index, expected in enumerate(shape)
    ):
        raise ValueError(f"observed primitive {name} has invalid shape")
    result = np.ascontiguousarray(source.copy(), dtype=dtype)
    if np.issubdtype(dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError(f"observed primitive {name} is non-finite")
    result.setflags(write=False)
    return result


def _finite_transform(request: object) -> tuple[np.ndarray, np.ndarray]:
    transform = getattr(getattr(request, "world", None), "map_from_odom", None)
    translation = getattr(transform, "translation_m", None)
    rotation = getattr(transform, "rotation", None)
    values = np.asarray(
        [
            getattr(translation, "x", np.nan),
            getattr(translation, "y", np.nan),
            getattr(translation, "z", np.nan),
            getattr(rotation, "w", np.nan),
            getattr(rotation, "x", np.nan),
            getattr(rotation, "y", np.nan),
            getattr(rotation, "z", np.nan),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("observed primitive map transform is invalid")
    translation_xyz = values[:3]
    quaternion = values[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("observed primitive map transform is invalid")
    w, x, y, z = quaternion / norm
    rotation_child_to_parent = np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
    return translation_xyz, rotation_child_to_parent


def _close(lhs: object, rhs: object) -> bool:
    return (
        isinstance(lhs, float)
        and isinstance(rhs, float)
        and math.isfinite(lhs)
        and math.isfinite(rhs)
        and math.isclose(lhs, rhs, rel_tol=1.0e-9, abs_tol=1.0e-9)
    )


def _positions_in_map(
    positions_m: np.ndarray,
    *,
    native_snapshot: object,
    request: object,
    platform_type: str,
) -> np.ndarray:
    if platform_type == "HOPPER":
        return positions_m
    world = getattr(request, "world", None)
    global_map = getattr(world, "global_map", None)
    local_map = getattr(world, "local_map", None)
    config = getattr(request, "config", None)
    platform_config = getattr(
        config, "wheel" if platform_type == "WHEELED" else "legged", None
    )
    graph_resolution = getattr(platform_config, "xy_resolution_m", None)
    local_selected = (
        getattr(native_snapshot, "width", None) == getattr(local_map, "width", None)
        and getattr(native_snapshot, "height", None)
        == getattr(local_map, "height", None)
        and _close(getattr(local_map, "resolution_m", None), graph_resolution)
        and not _close(getattr(global_map, "resolution_m", None), graph_resolution)
    )
    if not local_selected:
        return positions_m
    translation, rotation = _finite_transform(request)
    return np.ascontiguousarray(positions_m @ rotation.T + translation)


@dataclass(frozen=True, slots=True)
class ObservedPrimitiveSnapshot:
    platform_type: str
    width: int
    height: int
    algorithm_id: str
    state_schema: str
    primitive_set_sha256: str
    world_evidence_sha256: str
    graph_sha256: str
    revision: int
    invalidated_edge_count: int
    revalidated_edge_count: int
    state_ids: np.ndarray
    positions_m: np.ndarray
    yaw_rad: np.ndarray
    cells: np.ndarray
    yaw_bin: np.ndarray
    motion_mode: np.ndarray
    body_z_m: np.ndarray
    path_cost: np.ndarray
    forward_reachable: np.ndarray
    returnable: np.ndarray
    observation_state: np.ndarray
    recoverable: np.ndarray
    direct_successor: np.ndarray
    edge_source_ids: np.ndarray
    edge_target_ids: np.ndarray
    edge_primitive_indices: np.ndarray
    edge_primitive_ids: tuple[str, ...]
    edge_cost: np.ndarray

    def __post_init__(self) -> None:
        if self.platform_type not in _PLATFORMS:
            raise ValueError("observed primitive platform is invalid")
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or self.width <= 0
            or self.height <= 0
            or type(self.revision) is not int
            or self.revision <= 0
            or type(self.invalidated_edge_count) is not int
            or self.invalidated_edge_count < 0
            or type(self.revalidated_edge_count) is not int
            or self.revalidated_edge_count < 0
        ):
            raise ValueError("observed primitive snapshot metadata is invalid")
        if not self.algorithm_id or not self.state_schema:
            raise ValueError("observed primitive algorithm identity is missing")
        _sha256("primitive set hash", self.primitive_set_sha256)
        _sha256("world evidence hash", self.world_evidence_sha256)
        _sha256("graph hash", self.graph_sha256)
        state_count = len(self.state_ids)
        state_arrays = (
            ("state ids", self.state_ids, np.dtype(np.uint64), (state_count,)),
            ("positions", self.positions_m, np.dtype(np.float64), (state_count, 3)),
            ("yaw", self.yaw_rad, np.dtype(np.float64), (state_count,)),
            ("cells", self.cells, np.dtype(np.int32), (state_count, 2)),
            ("yaw bins", self.yaw_bin, np.dtype(np.int32), (state_count,)),
            ("motion modes", self.motion_mode, np.dtype(np.int32), (state_count,)),
            ("body height", self.body_z_m, np.dtype(np.float64), (state_count, 2)),
            ("path cost", self.path_cost, np.dtype(np.float64), (state_count,)),
            (
                "forward labels",
                self.forward_reachable,
                np.dtype(np.bool_),
                (state_count,),
            ),
            ("return labels", self.returnable, np.dtype(np.bool_), (state_count,)),
            (
                "observation labels",
                self.observation_state,
                np.dtype(np.bool_),
                (state_count,),
            ),
            ("recoverable labels", self.recoverable, np.dtype(np.bool_), (state_count,)),
            (
                "direct successor labels",
                self.direct_successor,
                np.dtype(np.bool_),
                (state_count,),
            ),
        )
        for name, values, dtype, shape in state_arrays:
            if values.dtype != dtype or values.shape != shape or values.flags.writeable:
                raise ValueError(f"observed primitive {name} is not frozen")
        if state_count == 0 or len(np.unique(self.state_ids)) != state_count:
            raise ValueError("observed primitive state identities are invalid")
        if (self.path_cost < 0.0).any():
            raise ValueError("observed primitive path cost is invalid")
        if not np.array_equal(
            self.recoverable, self.forward_reachable & self.returnable
        ):
            raise ValueError("observed primitive recoverable labels differ")
        edge_count = len(self.edge_source_ids)
        edge_arrays = (
            (self.edge_source_ids, np.dtype(np.uint64)),
            (self.edge_target_ids, np.dtype(np.uint64)),
            (self.edge_primitive_indices, np.dtype(np.uint32)),
            (self.edge_cost, np.dtype(np.float64)),
        )
        if any(
            values.shape != (edge_count,)
            or values.dtype != dtype
            or values.flags.writeable
            for values, dtype in edge_arrays
        ):
            raise ValueError("observed primitive edge arrays are invalid")
        if (
            len(self.edge_primitive_ids) != edge_count
            or any(not isinstance(value, str) or not value for value in self.edge_primitive_ids)
            or (self.edge_cost < 0.0).any()
        ):
            raise ValueError("observed primitive edge metadata is invalid")
        known_states = set(int(value) for value in self.state_ids)
        if any(
            int(value) not in known_states
            for value in np.concatenate(
                (self.edge_source_ids, self.edge_target_ids)
            )
        ):
            raise ValueError("observed primitive edge references unknown state")


def _snapshot_from_native(
    native: object,
    *,
    request: object,
    platform_type: str,
) -> ObservedPrimitiveSnapshot:
    state_ids = _readonly_array(
        "state ids", native.state_ids, dtype=np.dtype(np.uint64), shape=(None,)
    )
    state_count = len(state_ids)
    positions_native = _readonly_array(
        "positions",
        native.positions_m,
        dtype=np.dtype(np.float64),
        shape=(state_count, 3),
    )
    positions_map = _positions_in_map(
        positions_native,
        native_snapshot=native,
        request=request,
        platform_type=platform_type,
    )
    positions_map = _readonly_array(
        "map positions",
        positions_map,
        dtype=np.dtype(np.float64),
        shape=(state_count, 3),
    )

    def state_array(name: str, dtype: np.dtype, trailing: Iterable[int] = ()) -> np.ndarray:
        return _readonly_array(
            name,
            getattr(native, name),
            dtype=dtype,
            shape=(state_count, *tuple(trailing)),
        )

    edge_source_ids = _readonly_array(
        "edge source ids",
        native.edge_source_ids,
        dtype=np.dtype(np.uint64),
        shape=(None,),
    )
    edge_count = len(edge_source_ids)
    return ObservedPrimitiveSnapshot(
        platform_type=str(native.platform_type),
        width=int(native.width),
        height=int(native.height),
        algorithm_id=str(native.algorithm_id),
        state_schema=str(native.state_schema),
        primitive_set_sha256=str(native.primitive_set_sha256),
        world_evidence_sha256=str(native.world_evidence_sha256),
        graph_sha256=str(native.graph_sha256),
        revision=int(native.revision),
        invalidated_edge_count=int(native.invalidated_edge_count),
        revalidated_edge_count=int(native.revalidated_edge_count),
        state_ids=state_ids,
        positions_m=positions_map,
        yaw_rad=state_array("yaw_rad", np.dtype(np.float64)),
        cells=state_array("cells", np.dtype(np.int32), (2,)),
        yaw_bin=state_array("yaw_bin", np.dtype(np.int32)),
        motion_mode=state_array("motion_mode", np.dtype(np.int32)),
        body_z_m=state_array("body_z_m", np.dtype(np.float64), (2,)),
        path_cost=state_array("path_cost", np.dtype(np.float64)),
        forward_reachable=state_array(
            "forward_reachable", np.dtype(np.bool_)
        ),
        returnable=state_array("returnable", np.dtype(np.bool_)),
        observation_state=state_array("observation_state", np.dtype(np.bool_)),
        recoverable=state_array("recoverable", np.dtype(np.bool_)),
        direct_successor=state_array("direct_successor", np.dtype(np.bool_)),
        edge_source_ids=edge_source_ids,
        edge_target_ids=_readonly_array(
            "edge target ids",
            native.edge_target_ids,
            dtype=np.dtype(np.uint64),
            shape=(edge_count,),
        ),
        edge_primitive_indices=_readonly_array(
            "edge primitive indices",
            native.edge_primitive_indices,
            dtype=np.dtype(np.uint32),
            shape=(edge_count,),
        ),
        edge_primitive_ids=tuple(str(value) for value in native.edge_primitive_ids),
        edge_cost=_readonly_array(
            "edge cost",
            native.edge_cost,
            dtype=np.dtype(np.float64),
            shape=(edge_count,),
        ),
    )


class ObservedPrimitiveReachability:
    """Own one native engine whose only accepted inputs are observed revisions."""

    def __init__(self, platform_type: str, *, native_engine: object | None = None) -> None:
        if platform_type not in _PLATFORMS:
            raise ValueError("observed primitive platform is invalid")
        if native_engine is None:
            import lunar_planner_training_bridge as bridge_api

            native_engine = bridge_api.PrimitiveReachabilityEngine()
        if not callable(getattr(native_engine, "update", None)) or not callable(
            getattr(native_engine, "reset", None)
        ):
            raise TypeError("observed primitive native engine is invalid")
        self._platform_type = platform_type
        self._engine = native_engine
        self._observation_revision = 0
        self._snapshot: ObservedPrimitiveSnapshot | None = None

    @property
    def snapshot(self) -> ObservedPrimitiveSnapshot | None:
        return self._snapshot

    def update(
        self,
        request: object,
        *,
        observation_revision: int,
        maximum_action_distance_m: float = 30.0,
    ) -> ObservedPrimitiveSnapshot:
        if (
            type(observation_revision) is not int
            or observation_revision != self._observation_revision + 1
        ):
            raise ValueError("observed primitive revision must be strictly consecutive")
        request_id = getattr(request, "request_id", None)
        expected_prefix = f"formal-observed/{self._platform_type.lower()}/"
        if not isinstance(request_id, str) or not request_id.startswith(expected_prefix):
            raise ValueError("observed-only primitive request identity is required")
        if (
            getattr(request, "global_map_generation", None) != observation_revision
            or getattr(request, "local_map_generation", None)
            != observation_revision
        ):
            raise ValueError("observed primitive map revision differs")
        if (
            not isinstance(maximum_action_distance_m, float)
            or not math.isfinite(maximum_action_distance_m)
            or maximum_action_distance_m <= 0.0
        ):
            raise ValueError("observed primitive action distance is invalid")
        native = self._engine.update(request, maximum_action_distance_m)
        snapshot = _snapshot_from_native(
            native, request=request, platform_type=self._platform_type
        )
        if snapshot.platform_type != self._platform_type:
            raise RuntimeError("observed primitive platform identity differs")
        if snapshot.revision != observation_revision:
            raise RuntimeError("observed primitive native revision differs")
        self._observation_revision = observation_revision
        self._snapshot = snapshot
        return snapshot

    def reset(self) -> None:
        self._engine.reset()
        self._observation_revision = 0
        self._snapshot = None


__all__ = [
    "ObservedPrimitiveReachability",
    "ObservedPrimitiveSnapshot",
    "native_start_failure_is_ineligible",
]
