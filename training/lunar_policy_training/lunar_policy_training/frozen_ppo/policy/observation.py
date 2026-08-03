"""Observed-only hierarchical policy observation for Stage 2."""

from __future__ import annotations
from typing import TYPE_CHECKING

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

if TYPE_CHECKING:
    from typing import Any as FrontierActionSet
if TYPE_CHECKING:
    from typing import Any as ObservedMapState
if TYPE_CHECKING:
    from typing import Any as LowResolutionPrior
if TYPE_CHECKING:
    from typing import Any as PoseXYTheta


OBSERVATION_SCHEMA_VERSION = "policy_observation/v1"
GLOBAL_PRIOR_CHANNELS = (
    "height_prior", "value_prior", "obstacle_prior", "traversability_prior",
    "current_position_marker_lowres", "heading_sin_marker_lowres", "heading_cos_marker_lowres",
)
COVERAGE_SUMMARY_CHANNELS = (
    "observed_ratio", "unknown_ratio", "observed_free_ratio", "observed_blocked_ratio",
    "frontier_count", "observed_safe_frontier_count", "reachable_frontier_count",
    "mean_uncovered_value_prior",
)
LOCAL_CROP_CHANNELS = (
    "observed_height", "coverage_mask", "obstacle", "traversability",
    "local_frontier_channel", "current_position_marker", "heading_sin_marker", "heading_cos_marker",
)
FRONTIER_FEATURE_FIELDS = (
    "x_norm", "y_norm", "distance_from_robot_norm", "bearing_sin", "bearing_cos",
    "potential_coverage_gain_norm", "visible_unknown_count_norm", "value_gain_norm",
    "frontier_segment_id_norm", "segment_length_norm", "normal_sin", "normal_cos",
    "normal_confidence", "candidate_generation_mode", "recommended_theta_sin",
    "recommended_theta_cos", "traversability", "clearance_norm",
    "reachable_prefilter_cost_norm", "region_coverage_ratio", "region_unknown_ratio",
    "same_connected_component",
)
POSE_FEATURE_FIELDS = (
    "x_norm", "y_norm", "sin(theta)", "cos(theta)", "observed_roi_ratio",
    "remaining_step_budget_norm",
)


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    prior_channels: np.ndarray
    coverage_summary: np.ndarray
    local_crop: np.ndarray
    frontier_features: np.ndarray
    pose_features: np.ndarray
    candidate_mask: np.ndarray

    schema_version: ClassVar[str] = OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for array in self.array_fields():
            array.setflags(write=False)

    @property
    def candidate_valid_mask(self) -> np.ndarray:
        return self.candidate_mask

    @property
    def global_lowres_prior_state(self) -> np.ndarray:
        return self.prior_channels

    @property
    def global_highres_coverage_summary(self) -> np.ndarray:
        return self.coverage_summary

    @property
    def local_highres_observed_crop(self) -> np.ndarray:
        return self.local_crop

    def array_fields(self) -> tuple[np.ndarray, ...]:
        return (
            self.prior_channels,
            self.coverage_summary,
            self.local_crop,
            self.frontier_features,
            self.pose_features,
            self.candidate_mask,
        )

    @classmethod
    def schema_metadata(cls) -> dict[str, object]:
        return {
            "schema_version": cls.schema_version,
            "global_prior_channels": list(GLOBAL_PRIOR_CHANNELS),
            "coverage_summary_channels": list(COVERAGE_SUMMARY_CHANNELS),
            "local_crop_channels": list(LOCAL_CROP_CHANNELS),
            "frontier_feature_fields": list(FRONTIER_FEATURE_FIELDS),
            "pose_feature_fields": list(POSE_FEATURE_FIELDS),
            "local_padding": {name: 0.0 for name in LOCAL_CROP_CHANNELS},
        }

    def save_npz(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            schema_json=np.asarray(json.dumps(self.schema_metadata(), sort_keys=True)),
            prior_channels=self.prior_channels,
            coverage_summary=self.coverage_summary,
            local_crop=self.local_crop,
            frontier_features=self.frontier_features,
            pose_features=self.pose_features,
            candidate_mask=self.candidate_mask,
        )
        return destination

    @classmethod
    def load_npz(cls, path: str | Path) -> "PolicyObservation":
        expected = {
            "schema_json", "prior_channels", "coverage_summary", "local_crop",
            "frontier_features", "pose_features", "candidate_mask",
        }
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise ValueError("observation NPZ member set is not exact")
            metadata = json.loads(str(archive["schema_json"].item()))
            if metadata != cls.schema_metadata():
                raise ValueError("observation NPZ schema drift")
            return cls(
                prior_channels=np.asarray(archive["prior_channels"], dtype=np.float32).copy(),
                coverage_summary=np.asarray(archive["coverage_summary"], dtype=np.float32).copy(),
                local_crop=np.asarray(archive["local_crop"], dtype=np.float32).copy(),
                frontier_features=np.asarray(archive["frontier_features"], dtype=np.float32).copy(),
                pose_features=np.asarray(archive["pose_features"], dtype=np.float32).copy(),
                candidate_mask=np.asarray(archive["candidate_mask"], dtype=bool).copy(),
            )


class ObservationBuilder:
    """Build policy tensors without accepting truth or coverable-mask inputs."""

    def __init__(self, *, local_crop_size: int | None = None) -> None:
        if local_crop_size is not None and local_crop_size <= 0:
            raise ValueError("local_crop_size must be positive")
        self.local_crop_size = local_crop_size

    def build(
        self,
        prior: LowResolutionPrior,
        observed_state: ObservedMapState,
        pose: PoseXYTheta,
        action_set: FrontierActionSet,
    ) -> PolicyObservation:
        low_shape = prior.channels.shape[1:]
        summary = _coverage_summary(observed_state, prior, action_set)
        size = self.local_crop_size or _profile_crop_size(observed_state.geometry.shape, low_shape)
        crop = _local_crop(observed_state, pose, action_set, size=size)
        height, width = observed_state.geometry.shape
        prior_channels = np.zeros((7, *low_shape), dtype=np.float32)
        prior_channels[:4] = np.asarray(prior.channels[:4], dtype=np.float32)
        low_x = min(low_shape[1] - 1, int(pose.cell.x * low_shape[1] / width))
        low_y = min(low_shape[0] - 1, int(pose.cell.y * low_shape[0] / height))
        prior_channels[4, low_y, low_x] = 1.0
        prior_channels[5, low_y, low_x] = math.sin(pose.theta)
        prior_channels[6, low_y, low_x] = math.cos(pose.theta)
        pose_features = np.asarray(
            (
                pose.cell.x / max(width - 1, 1),
                pose.cell.y / max(height - 1, 1),
                math.sin(pose.theta),
                math.cos(pose.theta),
                float(np.mean(observed_state.observed_mask, dtype=np.float64)),
                float(observed_state.remaining_step_budget_norm),
            ),
            dtype=np.float32,
        )
        observation = PolicyObservation(
            prior_channels=prior_channels,
            coverage_summary=summary,
            local_crop=crop,
            frontier_features=np.asarray(action_set.frontier_features, dtype=np.float32).copy(),
            pose_features=pose_features,
            candidate_mask=np.asarray(action_set.candidate_mask, dtype=bool).copy(),
        )
        if not all(np.isfinite(array).all() for array in observation.array_fields() if array.dtype != np.bool_):
            raise ValueError("policy observation contains non-finite values")
        return observation


def _profile_crop_size(high_shape: tuple[int, int], low_shape: tuple[int, int]) -> int:
    profiles = {((128, 128), (32, 32)): 64, ((256, 256), (32, 32)): 96, ((2048, 2048), (128, 128)): 192}
    try:
        return profiles[(high_shape, low_shape)]
    except KeyError as exc:
        raise ValueError("local_crop_size is required for an unknown scale profile") from exc


def _mask_or_cells(action_set: FrontierActionSet, name: str, shape: tuple[int, int]) -> np.ndarray:
    value = getattr(action_set, name)
    if value is not None:
        mask = np.asarray(value, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"{name} shape mismatch")
        return mask
    mask = np.zeros(shape, dtype=bool)
    for cell in action_set.cells:
        if 0 <= cell.x < shape[1] and 0 <= cell.y < shape[0]:
            mask[cell.y, cell.x] = True
    return mask


def _coverage_summary(
    state: ObservedMapState,
    prior: LowResolutionPrior,
    action_set: FrontierActionSet,
) -> np.ndarray:
    out_height, out_width = prior.channels.shape[1:]
    height, width = state.geometry.shape
    if height % out_height or width % out_width:
        raise ValueError("observed grid must divide evenly into the low-resolution summary")
    block_y, block_x = height // out_height, width // out_width
    observed = state.observed_mask
    blocked = observed & (state.obstacle | (state.slope_deg > 30.0))
    free = observed & ~blocked & (state.traversability >= 0.5)
    frontier = _mask_or_cells(action_set, "frontier_mask", state.geometry.shape)
    safe_frontier = _mask_or_cells(action_set, "observed_safe_frontier_mask", state.geometry.shape)
    reachable_frontier = _mask_or_cells(action_set, "reachable_frontier_mask", state.geometry.shape)
    layers = (observed, ~observed, free, blocked, frontier, safe_frontier, reachable_frontier)
    summary = np.zeros((8, out_height, out_width), dtype=np.float32)
    for index, layer in enumerate(layers):
        reshaped = layer.reshape(out_height, block_y, out_width, block_x)
        summary[index] = reshaped.mean(axis=(1, 3), dtype=np.float64)
    value_prior = np.asarray(prior.channels[1], dtype=np.float32)
    for low_y in range(out_height):
        for low_x in range(out_width):
            y0, y1 = low_y * block_y, (low_y + 1) * block_y
            x0, x1 = low_x * block_x, (low_x + 1) * block_x
            summary[7, low_y, low_x] = value_prior[low_y, low_x] if np.any(~observed[y0:y1, x0:x1]) else 0.0
    return summary


def _local_crop(
    state: ObservedMapState,
    pose: PoseXYTheta,
    action_set: FrontierActionSet,
    *,
    size: int,
) -> np.ndarray:
    crop = np.zeros((8, size, size), dtype=np.float32)
    half = size // 2
    requested_x0 = pose.cell.x - half
    requested_y0 = pose.cell.y - half
    source_x0 = max(0, requested_x0)
    source_y0 = max(0, requested_y0)
    source_x1 = min(state.geometry.width, requested_x0 + size)
    source_y1 = min(state.geometry.height, requested_y0 + size)
    destination_x0 = source_x0 - requested_x0
    destination_y0 = source_y0 - requested_y0
    destination_x1 = destination_x0 + source_x1 - source_x0
    destination_y1 = destination_y0 + source_y1 - source_y0
    observed = state.observed_mask
    frontier = _mask_or_cells(action_set, "frontier_mask", state.geometry.shape)
    source = np.stack(
        (
            np.where(observed, state.height, 0.0),
            observed,
            observed & state.obstacle,
            np.where(observed, state.traversability, 0.0),
            frontier,
            np.zeros(state.geometry.shape, dtype=np.float32),
            np.zeros(state.geometry.shape, dtype=np.float32),
            np.zeros(state.geometry.shape, dtype=np.float32),
        ),
        axis=0,
    ).astype(np.float32)
    source[5, pose.cell.y, pose.cell.x] = 1.0
    source[6, pose.cell.y, pose.cell.x] = math.sin(pose.theta)
    source[7, pose.cell.y, pose.cell.x] = math.cos(pose.theta)
    crop[:, destination_y0:destination_y1, destination_x0:destination_x1] = source[
        :, source_y0:source_y1, source_x0:source_x1
    ]
    return crop
