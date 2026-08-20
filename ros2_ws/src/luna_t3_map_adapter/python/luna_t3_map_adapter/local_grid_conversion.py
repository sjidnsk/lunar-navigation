"""Fail-closed conversion of Task3 local evidence into planner layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from grid_map_msgs.msg import GridMap

from .grid_map_codec import GridMapCodecError, decode_layers


REQUIRED_PLANNER_LAYERS = (
    "elevation", "valid_mask", "obstacle", "obstacle_height",
    "observation_age_s", "observation_quality", "elevation_variance",
    "obstacle_variance", "observation_count", "forbidden",
)


@dataclass(frozen=True)
class LocalSourceGrid:
    frame_id: str
    stamp_ns: int
    origin_x_m: float
    origin_y_m: float
    resolution_m: float
    occupancy: np.ndarray
    semantic_id: np.ndarray
    elevation: np.ndarray
    roughness: np.ndarray


@dataclass(frozen=True)
class TileEvidence:
    height_range: np.ndarray
    elevation_variance: np.ndarray
    roughness: np.ndarray
    observation_count: np.ndarray
    semantic_confidence: np.ndarray


@dataclass(frozen=True)
class LocalMapPolicy:
    occupancy_obstacle_threshold: int
    semantic_obstacle_ids: frozenset[int]
    semantic_forbidden_ids: frozenset[int]
    max_age_s: float


@dataclass(frozen=True)
class CanonicalLocalGrid:
    frame_id: str
    stamp_ns: int
    origin_x_m: float
    origin_y_m: float
    resolution_m: float
    layers: Mapping[str, np.ndarray]


class ObservationLedger:
    """Request-local history for cells without a source observation count."""

    def __init__(self) -> None:
        self._last_stamp_ns: np.ndarray | None = None
        self._count: np.ndarray | None = None

    def observe(self, stamp_ns: int, valid_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if stamp_ns <= 0:
            raise ValueError("LOCAL_MAP_TIMESTAMP_INVALID")
        valid = np.asarray(valid_cells, dtype=bool)
        if self._count is None or self._count.shape != valid.shape:
            self._count = np.zeros(valid.shape, dtype=np.float32)
            self._last_stamp_ns = np.zeros(valid.shape, dtype=np.int64)
        assert self._last_stamp_ns is not None
        newly_observed = valid & (self._last_stamp_ns != stamp_ns)
        self._count[newly_observed] += 1.0
        self._last_stamp_ns[valid] = stamp_ns
        age_s = np.zeros(valid.shape, dtype=np.float32)
        return self._count.copy(), age_s


def local_source_from_grid(message: GridMap) -> LocalSourceGrid:
    """Decode the unrotated 0.2 m Task3 local GridMap without resampling."""

    if message.header.frame_id != "odom":
        raise ValueError("LOCAL_MAP_FRAME_INVALID")
    stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
    if stamp_ns <= 0:
        raise ValueError("LOCAL_MAP_TIMESTAMP_INVALID")
    if not np.isclose(message.info.resolution, 0.2):
        raise ValueError("LOCAL_MAP_RESOLUTION_INVALID")
    orientation = message.info.pose.orientation
    if not (
        np.isclose(orientation.x, 0.0)
        and np.isclose(orientation.y, 0.0)
        and np.isclose(orientation.z, 0.0)
        and np.isclose(orientation.w, 1.0)
    ):
        raise ValueError("LOCAL_MAP_GEOMETRY_INVALID")
    try:
        layers = decode_layers(message, ("occupancy", "semantic_id", "elevation", "roughness"))
    except GridMapCodecError as error:
        raise ValueError(str(error).replace("GRID_MAP", "LOCAL_MAP")) from error
    occupancy = layers["occupancy"]
    semantic_id = layers["semantic_id"]
    if not (np.all(np.isfinite(occupancy)) and np.all(np.isfinite(semantic_id))):
        raise ValueError("LOCAL_MAP_LAYER_INVALID")
    if not (
        np.all(np.equal(occupancy, np.rint(occupancy)))
        and np.all(np.equal(semantic_id, np.rint(semantic_id)))
    ):
        raise ValueError("LOCAL_MAP_LAYER_INVALID")
    return LocalSourceGrid(
        frame_id="odom",
        stamp_ns=stamp_ns,
        origin_x_m=float(message.info.pose.position.x - message.info.length_x * 0.5),
        origin_y_m=float(message.info.pose.position.y - message.info.length_y * 0.5),
        resolution_m=float(message.info.resolution),
        occupancy=occupancy.astype(np.int32),
        semantic_id=semantic_id.astype(np.int32),
        elevation=layers["elevation"],
        roughness=layers["roughness"],
    )


def _common_shape(source: LocalSourceGrid, evidence: TileEvidence) -> tuple[int, int]:
    arrays = (
        source.occupancy, source.semantic_id, source.elevation, source.roughness,
        evidence.height_range, evidence.elevation_variance, evidence.roughness,
        evidence.observation_count, evidence.semantic_confidence,
    )
    shape = np.asarray(arrays[0]).shape
    if len(shape) != 2 or not shape[0] or not shape[1] or any(np.asarray(item).shape != shape for item in arrays):
        raise ValueError("LOCAL_MAP_SHAPE_INVALID")
    return int(shape[0]), int(shape[1])


def convert_local_grid(
    source: LocalSourceGrid,
    evidence: TileEvidence,
    policy: LocalMapPolicy,
    ledger: ObservationLedger,
) -> CanonicalLocalGrid:
    """Preserve a Task3 odom grid and derive ten finite planner layers."""

    height, width = _common_shape(source, evidence)
    if source.frame_id != "odom":
        raise ValueError("LOCAL_MAP_FRAME_INVALID")
    if source.stamp_ns <= 0:
        raise ValueError("LOCAL_MAP_TIMESTAMP_INVALID")
    if not np.isclose(source.resolution_m, 0.2):
        raise ValueError("LOCAL_MAP_RESOLUTION_INVALID")
    if policy.max_age_s < 0.0:
        raise ValueError("LOCAL_MAP_POLICY_INVALID")

    occupancy = np.asarray(source.occupancy)
    semantic = np.asarray(source.semantic_id)
    elevation = np.asarray(source.elevation, dtype=np.float32)
    source_roughness = np.asarray(source.roughness, dtype=np.float32)
    height_range = np.asarray(evidence.height_range, dtype=np.float32)
    elevation_variance = np.asarray(evidence.elevation_variance, dtype=np.float32)
    evidence_roughness = np.asarray(evidence.roughness, dtype=np.float32)
    source_count = np.asarray(evidence.observation_count, dtype=np.int32)
    confidence = np.asarray(evidence.semantic_confidence, dtype=np.float32)

    valid = (
        (occupancy >= 0)
        & np.isfinite(elevation)
        & np.isfinite(source_roughness)
        & np.isfinite(height_range) & (height_range >= 0.0)
        & np.isfinite(elevation_variance) & (elevation_variance >= 0.0)
        & np.isfinite(evidence_roughness) & (evidence_roughness >= 0.0)
        & np.isfinite(confidence) & (confidence >= 0.0) & (confidence <= 1.0)
    )
    source_obstacle = (occupancy >= policy.occupancy_obstacle_threshold) | np.isin(
        semantic, tuple(policy.semantic_obstacle_ids)
    )
    policy_forbidden = np.isin(semantic, tuple(policy.semantic_forbidden_ids))
    ledger_count, ledger_age = ledger.observe(source.stamp_ns, valid)
    known_count = source_count >= 0

    layers = {
        name: np.zeros((height, width), dtype=np.float32)
        for name in REQUIRED_PLANNER_LAYERS
    }
    layers["elevation"][valid] = elevation[valid]
    layers["valid_mask"] = valid.astype(np.float32)
    layers["obstacle"] = (source_obstacle | ~valid).astype(np.float32)
    layers["obstacle_height"][valid] = height_range[valid]
    layers["observation_age_s"][valid] = ledger_age[valid]
    layers["observation_quality"][valid] = confidence[valid]
    layers["elevation_variance"][valid] = elevation_variance[valid]
    layers["obstacle_variance"][valid] = evidence_roughness[valid]
    layers["observation_count"][valid] = np.where(
        known_count[valid], source_count[valid].astype(np.float32), ledger_count[valid]
    )
    layers["forbidden"] = ((~valid) | policy_forbidden).astype(np.float32)

    return CanonicalLocalGrid(
        frame_id=source.frame_id,
        stamp_ns=source.stamp_ns,
        origin_x_m=source.origin_x_m,
        origin_y_m=source.origin_y_m,
        resolution_m=source.resolution_m,
        layers=layers,
    )
