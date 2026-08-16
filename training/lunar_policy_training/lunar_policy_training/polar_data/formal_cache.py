"""Content-addressed external cache for formal polar training scenes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, is_dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
from types import MappingProxyType
from typing import Callable, Iterable, Iterator, Mapping, TypeVar
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from ..capability_freeze import FrozenCapabilityBundle, FrozenPlatformCapability
from ..environment.formal_start_qualification import (
    FORMAL_BOUNDARY_MARGIN_CELLS,
    build_formal_mission_roi,
    formal_safe_start_cells,
    qualify_initial_start,
)
from ..environment.primitive_reachability import (
    native_start_failure_is_ineligible,
)
from ..environment.coverability import (
    IneligibleReason,
    PHYSICAL_GRID_AXIS_CONVENTION,
    PlatformCoverability,
    StreamedDetailCoverability,
    build_streamed_detail_coverability,
    canonical_physical_positions_um,
    classify_ineligibility,
    freeze_streamed_detail_coverability,
    mask_sha256,
    pack_detail_mask,
    physical_projection_sha256,
    read_packed_detail_window,
    unpack_detail_mask,
)
from ..environment.platform_reachability import (
    PHYSICAL_PROJECTION_SCHEMA,
    PlatformReachabilityError,
    _validate_ground_reachability_projection,
)
from ..environment.task_area import DETAIL_PER_GLOBAL, FrozenTaskGeometry
from ..environment.formal_episode_state import (
    FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION,
)
from ..reward import reward_weights_sha256
from ..reward_contract import REWARD_SCHEMA_VERSION
from ..training_semantics import (
    FORMAL_TRAINING_SEMANTICS_VERSION,
    training_semantics_sha256,
)
from .hazards import (
    CraterBowl,
    NoGoPolygon,
    RockCircle,
    VectorHazardScene,
    generate_vector_hazard_scene,
)
from .multires_scene import (
    GENERATOR_SHA256,
    MultiResolutionScene,
    SceneTileProvider,
)
from .raster import GridGeometry, MapCanvas, load_polar_window
from .scenario_manifest import (
    build_scenario_manifest_document,
    load_scenario_manifest,
)
from .source_lock import load_aggregate_source_lock


FORMAL_CACHE_SCHEMA = "lunar-formal-training-cache/v11"
BOUNDED_FORMAL_SCENE_COUNT = 128
BOUNDED_FORMAL_SPLIT_COUNTS = MappingProxyType(
    {"train": 98, "validation": 12, "test": 12, "holdout": 6}
)
_SOURCE_IDS = (
    "NASA_LOLA_87S_DEM",
    "NASA_LOLA_87S_COUNT",
    "JAXA_LUPEX_DATA_S1",
)
_PLATFORMS = ("WHEELED", "LEGGED", "HOPPER")
_REQUIRED_SPLITS = ("train", "validation", "test", "holdout")
_RESIDUAL_PRIMITIVE_IDENTITY_FIELDS = frozenset(
    (
        "primitive_state_count",
        "certified_edge_count",
        "recoverable_state_count",
        "primitive_state_schema",
        "primitive_set_sha256",
        "world_evidence_sha256",
        "reachability_graph_sha256",
    )
)
_PLATFORM_COVERABILITY_FIELDS = frozenset(
    (
        "qualified_start_cell",
        "physical_observation_pose_shape",
        "physical_projection_schema",
        "physical_reachability_algorithm_id",
        "physical_evidence_algorithm_id",
        "physical_grid_axis_convention",
        "physical_safe_pose_count",
        "physically_reachable_pose_count",
        "physical_projection_sha256",
        "mission_target_detail_mask_sha256",
        "coverable_detail_shape",
        "mission_target_detail_cell_count",
        "coverable_detail_cell_count",
        "mission_coverable_fraction",
        "initial_coverable_fraction",
        "initial_candidate_count",
        "coverable_detail_mask_sha256",
        "sensor_visibility_algorithm_id",
        "capability_content_sha256",
        "start_identity_sha256",
        "exact",
        "eligible",
        "ineligible_reason",
        "stage_diagnostics",
    )
)
_SCENE_ENTRY_FIELDS = frozenset(
    (
        "scene_id",
        "source",
        "split",
        "window_id",
        "window_sha256",
        "world_bounds_m",
        "relative_path",
        "size_bytes",
        "sha256",
        "arrays",
        "platform_coverability",
    )
)
_SCENE_INDEX_FIELDS = frozenset(
    (
        "scene_id",
        "source",
        "split",
        "scene_seed",
        "window_id",
        "window_sha256",
        "world_bounds_m",
        "platform_starts",
    )
)
_PLATFORM_START_FIELDS = frozenset(
    (
        "qualified_start_cell",
        "start_identity_sha256",
        "capability_content_sha256",
        "qualification_algorithm_id",
        "safe_start_cell_count",
        "qualified",
        "ineligible_reason",
    )
)
_MINIMAL_START_QUALIFICATION_ALGORITHM_ID = (
    "formal-minimal-start-qualification/v1"
)
_SOURCE_LOCATOR_SCHEMA = "lunar-formal-source-locator/v1"
_STATIC_SCENE_BASE_ARRAYS = frozenset(
    (
        "elevation_m",
        "valid_mask",
        "physical_obstacle_ratio",
        "physical_obstacle_height_m",
        "forbidden_ratio",
        "rocks",
        "craters",
        "no_go_vertices",
    )
)
_STATIC_SCENE_ARRAYS = _STATIC_SCENE_BASE_ARRAYS | frozenset(
    f"{platform.lower()}_{suffix}"
    for platform in _PLATFORMS
    for suffix in (
        "hard_feasible",
        "clearance_margin_norm",
        "physical_observation_pose_bits",
        "coverable_detail_bits",
        "coverable_ratio",
    )
) | frozenset(("hopper_physical_observation_positions_um",))
_IDENTITY_FIELDS = (
    "source_lock_file_sha256",
    "source_sha256s",
    "split_manifest_file_sha256",
    "split_sha256",
    "scenario_manifest_sha256",
    "generator_sha256",
    "capability_sha256",
    "reward_sha256",
    "training_semantics_sha256",
    "v3_source_commit",
    "v3_sha256",
)
_PlatformResult = TypeVar("_PlatformResult")
_ParallelItem = TypeVar("_ParallelItem")
_ParallelResult = TypeVar("_ParallelResult")


class FormalCacheError(ValueError):
    """The cache is incomplete, drifted, unsafe, or not formally eligible."""


def _parallel_platform_map(
    worker: Callable[[str], _PlatformResult],
) -> dict[str, _PlatformResult]:
    """Evaluate independent platform records concurrently in frozen order."""
    with ThreadPoolExecutor(
        max_workers=len(_PLATFORMS),
        thread_name_prefix="formal-cache-platform",
    ) as executor:
        futures = {
            platform: executor.submit(worker, platform)
            for platform in _PLATFORMS
        }
        return {
            platform: futures[platform].result()
            for platform in _PLATFORMS
        }


def _ordered_bounded_process_map(
    items: Iterable[_ParallelItem],
    worker: Callable[[_ParallelItem], _ParallelResult],
    *,
    max_workers: int,
) -> Iterator[_ParallelResult]:
    """Compute bounded process work while preserving the frozen input order."""
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("process worker count must be a positive integer")
    iterator = iter(items)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pending = []
        for _ in range(max_workers):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(executor.submit(worker, item))
        while pending:
            future = pending.pop(0)
            yield future.result()
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(worker, item))


def _scene_process_worker_count(*, cpu_count: int, scene_count: int) -> int:
    """Use one process per three platform threads without unbounded fan-out."""
    if type(cpu_count) is not int or cpu_count < 1:
        raise ValueError("CPU count must be a positive integer")
    if type(scene_count) is not int or scene_count < 1:
        raise ValueError("scene count must be a positive integer")
    return min(scene_count, max(1, min(12, cpu_count // len(_PLATFORMS))))


def _formal_platform_eligibility_ready(
    materialization: str,
    platform_counts: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> bool:
    """Require a non-empty exact eligible lane for every platform and split."""
    if materialization not in {"bounded", "full"}:
        return False
    if set(platform_counts) != set(_PLATFORMS):
        return False
    for platform in _PLATFORMS:
        splits = platform_counts[platform]
        for split in _REQUIRED_SPLITS:
            counts = splits.get(split)
            if not isinstance(counts, Mapping):
                return False
            total = counts.get("total_scene_count")
            eligible = counts.get("eligible_scene_count")
            if (
                type(total) is not int
                or type(eligible) is not int
                or total <= 0
                or eligible <= 0
                or eligible > total
            ):
                return False
    return True


def platform_scenario_schedule_id(
    platform_type: str,
    split: str,
    ordered_scene_ids: Iterable[str],
) -> str:
    """Bind one deterministic scenario schedule to a platform and split."""
    if platform_type not in _PLATFORMS:
        raise FormalCacheError("platform schedule type is invalid")
    if split not in _REQUIRED_SPLITS:
        raise FormalCacheError("platform schedule split is invalid")
    scene_ids = tuple(ordered_scene_ids)
    if len(set(scene_ids)) != len(scene_ids):
        raise FormalCacheError("platform schedule contains duplicate scenes")
    for scene_id in scene_ids:
        _require_sha(scene_id, "platform schedule scene_id")
    payload = {
        "schema": "lunar-platform-scenario-schedule/v1",
        "platform_type": platform_type,
        "split": split,
        "ordered_scene_ids": list(scene_ids),
    }
    return f"lunar-platform-scenario-schedule/v1:{_semantic_sha(payload)}"


def _require_sha(value: object, name: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FormalCacheError(f"{name} must be a lowercase hexadecimal digest")
    return value


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
    return sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _reject_residual_primitive_identity(value: object) -> None:
    if isinstance(value, Mapping):
        if set(value) & _RESIDUAL_PRIMITIVE_IDENTITY_FIELDS:
            raise FormalCacheError(
                "cache contains residual primitive identity fields"
            )
        for item in value.values():
            _reject_residual_primitive_identity(item)
    elif isinstance(value, list):
        for item in value:
            _reject_residual_primitive_identity(item)


def _physical_capability_content_sha256(platform: object) -> str:
    """Hash bridge-visible physical content without planner primitive graphs."""
    typed = getattr(platform, "typed_capability", None)
    observation = getattr(platform, "observation_capability", None)
    if (
        not is_dataclass(typed)
        or isinstance(typed, type)
        or not is_dataclass(observation)
        or isinstance(observation, type)
    ):
        raise FormalCacheError("physical capability content is invalid")
    physical_content = asdict(typed)
    physical_content.pop("motion_primitives", None)
    return _semantic_sha(
        {
            "schema": "lunar-physical-capability-content/v1",
            "platform_type": getattr(platform, "platform_type", None),
            "capability_type": getattr(platform, "capability_type", None),
            "capability_version": getattr(platform, "capability_version", None),
            "platform_id": getattr(platform, "platform_id", None),
            "base_frame_id": getattr(platform, "base_frame_id", None),
            "observation_capability": asdict(observation),
            "physical_capability": physical_content,
        }
    )


def _file_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FormalCacheIdentity:
    source_lock_file_sha256: str
    source_sha256s: Mapping[str, str]
    split_manifest_file_sha256: str
    split_sha256: str
    scenario_manifest_sha256: str
    generator_sha256: str
    capability_sha256: str
    reward_sha256: str
    training_semantics_sha256: str
    v3_source_commit: str
    v3_sha256: str

    def __post_init__(self) -> None:
        for name in _IDENTITY_FIELDS:
            if name not in {"source_sha256s", "v3_source_commit"}:
                _require_sha(getattr(self, name), name)
        _require_sha(self.v3_source_commit, "v3_source_commit", length=40)
        if set(self.source_sha256s) != set(_SOURCE_IDS):
            raise FormalCacheError("source_sha256s must identify exactly three sources")
        checked = {
            source_id: _require_sha(self.source_sha256s[source_id], source_id)
            for source_id in _SOURCE_IDS
        }
        object.__setattr__(self, "source_sha256s", MappingProxyType(checked))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_lock_file_sha256": self.source_lock_file_sha256,
            "source_sha256s": dict(self.source_sha256s),
            "split_manifest_file_sha256": self.split_manifest_file_sha256,
            "split_sha256": self.split_sha256,
            "scenario_manifest_sha256": self.scenario_manifest_sha256,
            "generator_sha256": self.generator_sha256,
            "capability_sha256": self.capability_sha256,
            "reward_sha256": self.reward_sha256,
            "training_semantics_sha256": self.training_semantics_sha256,
            "v3_source_commit": self.v3_source_commit,
            "v3_sha256": self.v3_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "FormalCacheIdentity":
        if not isinstance(value, Mapping) or set(value) != set(_IDENTITY_FIELDS):
            raise FormalCacheError("cache identity fields are invalid")
        return cls(**{name: value[name] for name in _IDENTITY_FIELDS})


@dataclass(frozen=True, slots=True)
class FormalSceneIndexRecord:
    scene_id: str
    source: str
    split: str
    scene_seed: str
    window_id: str
    window_sha256: str
    world_bounds_m: tuple[float, float, float, float]
    platform_starts: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        _require_sha(self.scene_id, "scene index scene_id")
        _require_sha(self.scene_seed, "scene index scene_seed")
        _require_sha(self.window_sha256, "scene index window_sha256")
        if self.source not in {"NASA_LOLA", "JAXA_LUPEX"}:
            raise FormalCacheError("scene index source is unsupported")
        if self.split not in _REQUIRED_SPLITS:
            raise FormalCacheError("scene index split is unsupported")
        if not isinstance(self.window_id, str) or not self.window_id:
            raise FormalCacheError("scene index window_id is missing")
        _, _, bounds = _physical_grid_geometry(self.world_bounds_m)
        object.__setattr__(self, "world_bounds_m", bounds)
        if not isinstance(self.platform_starts, Mapping) or set(
            self.platform_starts
        ) != set(_PLATFORMS):
            raise FormalCacheError(
                "scene index must contain three platform starts"
            )
        checked: dict[str, Mapping[str, object]] = {}
        for platform in _PLATFORMS:
            raw = self.platform_starts[platform]
            if not isinstance(raw, Mapping) or set(raw) != _PLATFORM_START_FIELDS:
                raise FormalCacheError(
                    "scene index platform start fields are invalid"
                )
            cell_raw = raw["qualified_start_cell"]
            if cell_raw is None:
                cell = None
            elif (
                isinstance(cell_raw, (list, tuple))
                and len(cell_raw) == 2
                and all(
                    type(value) is int and 0 <= value < 256
                    for value in cell_raw
                )
            ):
                cell = int(cell_raw[0]), int(cell_raw[1])
            else:
                raise FormalCacheError(
                    "scene index qualified start cell is invalid"
                )
            qualified = raw["qualified"]
            safe_count = raw["safe_start_cell_count"]
            reason = raw["ineligible_reason"]
            if (
                type(qualified) is not bool
                or type(safe_count) is not int
                or safe_count < 0
                or (cell is not None) is not qualified
                or (safe_count > 0) is not qualified
                or (reason is None) is not qualified
                or (
                    reason is not None
                    and reason != "NO_MINIMAL_SAFE_START"
                )
            ):
                raise FormalCacheError(
                    "scene index platform start qualification differs"
                )
            if (
                raw["qualification_algorithm_id"]
                != _MINIMAL_START_QUALIFICATION_ALGORITHM_ID
            ):
                raise FormalCacheError(
                    "scene index start qualification algorithm differs"
                )
            _require_sha(
                raw["start_identity_sha256"],
                "scene index start identity",
            )
            _require_sha(
                raw["capability_content_sha256"],
                "scene index capability identity",
            )
            checked[platform] = MappingProxyType(
                {
                    "qualified_start_cell": cell,
                    "start_identity_sha256": raw["start_identity_sha256"],
                    "capability_content_sha256": raw[
                        "capability_content_sha256"
                    ],
                    "qualification_algorithm_id": raw[
                        "qualification_algorithm_id"
                    ],
                    "safe_start_cell_count": safe_count,
                    "qualified": qualified,
                    "ineligible_reason": reason,
                }
            )
        object.__setattr__(
            self,
            "platform_starts",
            MappingProxyType(checked),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "source": self.source,
            "split": self.split,
            "scene_seed": self.scene_seed,
            "window_id": self.window_id,
            "window_sha256": self.window_sha256,
            "world_bounds_m": list(self.world_bounds_m),
            "platform_starts": {
                platform: {
                    **dict(self.platform_starts[platform]),
                    "qualified_start_cell": (
                        None
                        if self.platform_starts[platform][
                            "qualified_start_cell"
                        ]
                        is None
                        else list(
                            self.platform_starts[platform][
                                "qualified_start_cell"
                            ]
                        )
                    ),
                }
                for platform in _PLATFORMS
            },
        }


def reward_v4_runtime_identity_evidence(
    identity: FormalCacheIdentity,
) -> dict[str, str]:
    """Bind reusable physical cache bytes to current Reward V4 runtime state."""
    if not isinstance(identity, FormalCacheIdentity):
        raise FormalCacheError("runtime identity requires a formal cache identity")
    if identity.reward_sha256 != reward_weights_sha256():
        raise FormalCacheError("runtime identity reward_sha256 differs")
    if identity.training_semantics_sha256 != training_semantics_sha256():
        raise FormalCacheError("runtime identity training semantics differ")
    return {
        "schema_version": "lunar-formal-runtime-identity/v1",
        "formal_cache_schema": FORMAL_CACHE_SCHEMA,
        "formal_environment_state_schema": (
            FORMAL_ENVIRONMENT_STATE_SCHEMA_VERSION
        ),
        "reward_schema": REWARD_SCHEMA_VERSION,
        "reward_sha256": identity.reward_sha256,
        "training_semantics_version": FORMAL_TRAINING_SEMANTICS_VERSION,
        "training_semantics_sha256": identity.training_semantics_sha256,
    }


def _grid(name: str, value: np.ndarray, dtype: object) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if array.shape != (256, 256):
        raise FormalCacheError(f"{name} must have shape [256,256]")
    return array


def _physical_grid_geometry(
    world_bounds_m: object,
    shape: tuple[int, int] = (256, 256),
) -> tuple[
    float,
    tuple[float, float],
    tuple[float, float, float, float],
]:
    if (
        not isinstance(world_bounds_m, (list, tuple))
        or len(world_bounds_m) != 4
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in world_bounds_m
        )
    ):
        raise FormalCacheError("scene world bounds are invalid")
    bounds = tuple(float(value) for value in world_bounds_m)
    left, bottom, right, top = bounds
    resolution_x = (right - left) / shape[1]
    resolution_y = (top - bottom) / shape[0]
    if (
        left >= right
        or bottom >= top
        or not math.isfinite(resolution_x)
        or resolution_x <= 0.0
        or not math.isclose(
            resolution_x, resolution_y, rel_tol=0.0, abs_tol=5.0e-7
        )
    ):
        raise FormalCacheError("scene physical grid geometry is invalid")
    return resolution_x, (left, top), bounds


def _ground_physical_observation_positions_m(
    physical_mask: np.ndarray,
    elevation_m: np.ndarray,
    world_bounds_m: object,
) -> np.ndarray:
    resolution, (left, top), _ = _physical_grid_geometry(
        world_bounds_m, physical_mask.shape
    )
    rows, columns = np.nonzero(physical_mask)
    return np.ascontiguousarray(
        np.column_stack(
            (
                left + (columns.astype(np.float64) + 0.5) * resolution,
                top - (rows.astype(np.float64) + 0.5) * resolution,
                np.asarray(elevation_m[physical_mask], dtype=np.float64),
            )
        ),
        dtype=np.float64,
    ).reshape((-1, 3))


def _positions_m_from_canonical_um(values: np.ndarray) -> np.ndarray:
    if (
        not isinstance(values, np.ndarray)
        or values.dtype.str != "<i8"
        or values.ndim != 2
        or values.shape[1:] != (3,)
        or not values.flags.c_contiguous
    ):
        raise FormalCacheError(
            "Hopper physical observation positions must be canonical <i8 [N,3]"
        )
    return np.ascontiguousarray(values.astype(np.float64) / 1_000_000.0)


def _validate_hopper_position_cells(
    positions_m: np.ndarray,
    physical_mask: np.ndarray,
    world_bounds_m: object,
) -> None:
    if len(positions_m) != int(physical_mask.sum(dtype=np.int64)):
        raise FormalCacheError("Hopper physical observation position count differs")
    if not len(positions_m):
        return
    resolution, (left, top), _ = _physical_grid_geometry(
        world_bounds_m, physical_mask.shape
    )
    rows, columns = np.nonzero(physical_mask)
    tolerance = 0.5e-6
    x = positions_m[:, 0]
    y = positions_m[:, 1]
    if (
        (x < left + columns * resolution - tolerance).any()
        or (x >= left + (columns + 1) * resolution + tolerance).any()
        or (y > top - rows * resolution + tolerance).any()
        or (y <= top - (rows + 1) * resolution - tolerance).any()
    ):
        raise FormalCacheError(
            "Hopper physical observation positions are not row-major mask landings"
        )


@dataclass(frozen=True)
class StaticSceneData:
    scene_id: str
    source: str
    split: str
    window_id: str
    window_sha256: str | None
    world_bounds_m: tuple[float, float, float, float]
    elevation_m: np.ndarray
    valid_mask: np.ndarray
    physical_obstacle_ratio: np.ndarray
    physical_obstacle_height_m: np.ndarray
    forbidden_ratio: np.ndarray
    rocks: np.ndarray
    craters: np.ndarray
    no_go_vertices: np.ndarray
    hard_feasible: Mapping[str, np.ndarray]
    clearance_margin_norm: Mapping[str, np.ndarray]
    coverability: Mapping[str, PlatformCoverability]
    physical_evidence_algorithm_ids: Mapping[str, str]
    hopper_physical_observation_positions_um: np.ndarray

    def __post_init__(self) -> None:
        _require_sha(self.scene_id, "scene_id")
        if self.source not in {"NASA_LOLA", "JAXA_LUPEX"}:
            raise FormalCacheError("scene source is unsupported")
        if self.split not in {"train", "validation", "test", "holdout"}:
            raise FormalCacheError("scene split is unsupported")
        if not isinstance(self.window_id, str) or not self.window_id:
            raise FormalCacheError("scene window_id is missing")
        if self.window_sha256 is not None:
            _require_sha(self.window_sha256, "window_sha256")
        resolution_m, origin_m, bounds_m = _physical_grid_geometry(
            self.world_bounds_m
        )
        object.__setattr__(self, "world_bounds_m", bounds_m)
        elevation = _grid("elevation_m", self.elevation_m, np.float32)
        valid = _grid("valid_mask", self.valid_mask, np.bool_)
        if not np.isfinite(elevation[valid]).all():
            raise FormalCacheError("scene NoData cannot be valid")
        object.__setattr__(self, "elevation_m", elevation)
        object.__setattr__(self, "valid_mask", valid)
        for name in (
            "physical_obstacle_ratio",
            "forbidden_ratio",
        ):
            values = _grid(name, getattr(self, name), np.float32)
            if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
                raise FormalCacheError(f"{name} must be finite and in [0,1]")
            object.__setattr__(self, name, values)
        height = _grid(
            "physical_obstacle_height_m",
            self.physical_obstacle_height_m,
            np.float32,
        )
        if not np.isfinite(height).all() or (height < 0.0).any():
            raise FormalCacheError("obstacle height must be finite and non-negative")
        object.__setattr__(self, "physical_obstacle_height_m", height)
        vector_shapes = {
            "rocks": (None, 4),
            "craters": (None, 4),
            "no_go_vertices": (None, 6, 2),
        }
        for name, shape in vector_shapes.items():
            values = np.ascontiguousarray(np.asarray(getattr(self, name), np.float64))
            if values.ndim != len(shape) or any(
                expected is not None and values.shape[index] != expected
                for index, expected in enumerate(shape)
            ) or not np.isfinite(values).all():
                raise FormalCacheError(f"{name} vector inventory is invalid")
            object.__setattr__(self, name, values)
        if set(self.hard_feasible) != set(_PLATFORMS) or set(
            self.clearance_margin_norm
        ) != set(_PLATFORMS):
            raise FormalCacheError("scene projections must contain three platforms")
        if set(self.coverability) != set(_PLATFORMS):
            raise FormalCacheError(
                "scene coverability must contain three platforms"
            )
        if set(self.physical_evidence_algorithm_ids) != set(_PLATFORMS):
            raise FormalCacheError(
                "scene physical evidence must contain three platforms"
            )
        evidence_algorithms = {
            platform: self.physical_evidence_algorithm_ids[platform]
            for platform in _PLATFORMS
        }
        if any(
            not isinstance(value, str) or not value
            for value in evidence_algorithms.values()
        ):
            raise FormalCacheError("scene physical evidence algorithm is invalid")
        object.__setattr__(
            self,
            "physical_evidence_algorithm_ids",
            MappingProxyType(evidence_algorithms),
        )
        hopper_positions_um = np.ascontiguousarray(
            self.hopper_physical_observation_positions_um
        )
        hopper_positions_m = _positions_m_from_canonical_um(hopper_positions_um)
        object.__setattr__(
            self,
            "hopper_physical_observation_positions_um",
            hopper_positions_um,
        )
        checked_coverability: dict[str, PlatformCoverability] = {}
        for platform in _PLATFORMS:
            payload = self.coverability[platform]
            if (
                not isinstance(payload, PlatformCoverability)
                or payload.platform_type != platform
                or payload.physical_observation_pose_mask.shape != (256, 256)
            ):
                raise FormalCacheError("scene platform coverability is invalid")
            positions_m = (
                hopper_positions_m
                if platform == "HOPPER"
                else _ground_physical_observation_positions_m(
                    payload.physical_observation_pose_mask,
                    elevation,
                    bounds_m,
                )
            )
            if platform == "HOPPER":
                _validate_hopper_position_cells(
                    positions_m,
                    payload.physical_observation_pose_mask,
                    bounds_m,
                )
            try:
                expected_projection_sha256 = physical_projection_sha256(
                    platform_type=platform,
                    physical_reachability_algorithm_id=(
                        payload.physical_reachability_algorithm_id
                    ),
                    physical_evidence_algorithm_id=(
                        evidence_algorithms[platform]
                    ),
                    physical_observation_pose_mask=(
                        payload.physical_observation_pose_mask
                    ),
                    physical_observation_positions_m=positions_m,
                    physical_grid_resolution_m=resolution_m,
                    physical_grid_origin_m=origin_m,
                    physical_grid_world_bounds_m=bounds_m,
                    physical_grid_axis_convention=(
                        PHYSICAL_GRID_AXIS_CONVENTION
                    ),
                    capability_content_sha256=(
                        payload.capability_content_sha256
                    ),
                    start_identity_sha256=payload.start_identity_sha256,
                )
            except ValueError as error:
                raise FormalCacheError(
                    "scene physical projection authority is invalid"
                ) from error
            if expected_projection_sha256 != payload.physical_projection_sha256:
                raise FormalCacheError("scene physical projection hash differs")
            cell = payload.qualified_start_cell
            if cell is None:
                checked_coverability[platform] = payload
                continue
            row, column = cell
            margin = FORMAL_BOUNDARY_MARGIN_CELLS
            hard = np.asarray(self.hard_feasible[platform], dtype=np.bool_)
            clearance = np.asarray(
                self.clearance_margin_norm[platform], dtype=np.float32
            )
            if (
                row < margin
                or row >= 256 - margin
                or column < margin
                or column >= 256 - margin
                or not bool(valid[row, column])
                or not bool(hard[row, column])
                or not float(clearance[row, column]) > 0.0
            ):
                raise FormalCacheError("qualified start cell is not physically safe")
            checked_coverability[platform] = payload
        object.__setattr__(
            self,
            "coverability",
            MappingProxyType(checked_coverability),
        )

    def arrays(self) -> dict[str, np.ndarray]:
        output = {
            "elevation_m": self.elevation_m,
            "valid_mask": self.valid_mask,
            "physical_obstacle_ratio": self.physical_obstacle_ratio,
            "physical_obstacle_height_m": self.physical_obstacle_height_m,
            "forbidden_ratio": self.forbidden_ratio,
            "rocks": self.rocks,
            "craters": self.craters,
            "no_go_vertices": self.no_go_vertices,
            "hopper_physical_observation_positions_um": (
                self.hopper_physical_observation_positions_um
            ),
        }
        for platform in _PLATFORMS:
            prefix = platform.lower()
            hard = _grid(
                f"{prefix}_hard_feasible",
                self.hard_feasible[platform],
                np.uint8,
            )
            if ((hard != 0) & (hard != 1)).any():
                raise FormalCacheError("hard feasibility must be binary")
            clearance = _grid(
                f"{prefix}_clearance_margin_norm",
                self.clearance_margin_norm[platform],
                np.float32,
            )
            if not np.isfinite(clearance).all() or (
                (clearance < 0.0) | (clearance > 1.0)
            ).any():
                raise FormalCacheError("normalized clearance must be in [0,1]")
            output[f"{prefix}_hard_feasible"] = hard
            output[f"{prefix}_clearance_margin_norm"] = clearance
            coverability = self.coverability[platform]
            output[f"{prefix}_physical_observation_pose_bits"] = pack_detail_mask(
                coverability.physical_observation_pose_mask
            )
            output[f"{prefix}_coverable_detail_bits"] = np.ascontiguousarray(
                coverability.coverable_detail_bits.copy(), dtype=np.dtype("u1")
            )
            output[f"{prefix}_coverable_ratio"] = np.ascontiguousarray(
                coverability.coverable_ratio.copy(), dtype=np.dtype("<f4")
            )
        return output


def _validate_cache_root(root: Path, repository_root: Path) -> Path:
    if not root.is_absolute():
        raise FormalCacheError("cache root must be absolute")
    repository = repository_root.resolve(strict=True)
    parent = root.parent
    if not parent.is_dir() or parent.is_symlink():
        raise FormalCacheError("cache root parent must be an existing regular directory")
    resolved_parent = parent.resolve(strict=True)
    resolved_target = resolved_parent / root.name
    if resolved_target.is_relative_to(repository):
        raise FormalCacheError("cache root must be outside the repository")
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise FormalCacheError("cache root must be a regular directory")
    return resolved_target


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        if temporary.exists():
            temporary.unlink()
        raise FormalCacheError(f"could not atomically write {path.name}") from error


def _write_source_locator(
    root: Path,
    source_lock_path: Path,
    *,
    identity: FormalCacheIdentity,
    repository_root: Path,
) -> None:
    if (
        not source_lock_path.is_absolute()
        or source_lock_path.is_symlink()
        or not source_lock_path.is_file()
        or source_lock_path.resolve(strict=True).is_relative_to(
            repository_root.resolve(strict=True)
        )
        or _file_sha(source_lock_path) != identity.source_lock_file_sha256
    ):
        raise FormalCacheError("source locator lock is invalid")
    body: dict[str, object] = {
        "schema": _SOURCE_LOCATOR_SCHEMA,
        "source_lock_path": str(source_lock_path.resolve(strict=True)),
        "source_lock_file_sha256": identity.source_lock_file_sha256,
    }
    body["locator_sha256"] = _semantic_sha(body)
    _atomic_bytes(root / "source-locator.json", _canonical_bytes(body))


def _load_source_locator(
    root: Path,
    *,
    identity: FormalCacheIdentity,
) -> tuple[Path | None, Mapping[str, Path]]:
    locator = root / "source-locator.json"
    if not locator.exists() and not locator.is_symlink():
        return None, MappingProxyType({})
    if locator.is_symlink() or not locator.is_file():
        raise FormalCacheError("source locator must be a regular file")
    try:
        value = json.loads(locator.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalCacheError("source locator is invalid UTF-8 JSON") from error
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "source_lock_path",
        "source_lock_file_sha256",
        "locator_sha256",
    }:
        raise FormalCacheError("source locator fields are invalid")
    claimed = _require_sha(value.pop("locator_sha256"), "source locator")
    if value.get("schema") != _SOURCE_LOCATOR_SCHEMA or _semantic_sha(value) != claimed:
        raise FormalCacheError("source locator identity differs")
    if value.get("source_lock_file_sha256") != identity.source_lock_file_sha256:
        raise FormalCacheError("source locator lock identity differs")
    raw_path = value.get("source_lock_path")
    if not isinstance(raw_path, str):
        raise FormalCacheError("source locator path is invalid")
    lock_path = Path(raw_path)
    if (
        not lock_path.is_absolute()
        or lock_path.is_symlink()
        or not lock_path.is_file()
        or _file_sha(lock_path) != identity.source_lock_file_sha256
    ):
        raise FormalCacheError("source locator lock differs")
    repository_root = Path(__file__).resolve().parents[4]
    data_root, locks = load_aggregate_source_lock(
        lock_path,
        repository_root=repository_root,
    )
    paths = MappingProxyType(
        {lock.source_id: data_root / lock.filename for lock in locks}
    )
    return lock_path, paths


def _deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as raw:
            with ZipFile(raw, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
                for name in sorted(arrays):
                    buffer = BytesIO()
                    np.lib.format.write_array(
                        buffer,
                        np.ascontiguousarray(arrays[name]),
                        allow_pickle=False,
                    )
                    info = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, buffer.getvalue(), compress_type=ZIP_DEFLATED, compresslevel=6)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    except (OSError, ValueError) as error:
        if temporary.exists():
            temporary.unlink()
        raise FormalCacheError("could not atomically write static scene") from error


def _array_metadata(values: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(values)
    byte_order = array.dtype.byteorder
    if byte_order == "=":
        byte_order = "little" if np.little_endian else "big"
    elif byte_order == "|":
        byte_order = "not-applicable"
    else:
        byte_order = "little" if byte_order == "<" else "big"
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "byte_order": byte_order,
        "sha256": sha256(array.tobytes(order="C")).hexdigest(),
    }


def _scene_entry(scene: StaticSceneData, path: Path, root: Path) -> dict[str, object]:
    arrays = scene.arrays()
    _deterministic_npz(path, arrays)
    platform_coverability: dict[str, dict[str, object]] = {}
    for platform in _PLATFORMS:
        payload = scene.coverability[platform]
        platform_coverability[platform] = {
            "qualified_start_cell": (
                None
                if payload.qualified_start_cell is None
                else list(payload.qualified_start_cell)
            ),
            "physical_observation_pose_shape": list(
                payload.physical_observation_pose_mask.shape
            ),
            "physical_projection_schema": payload.physical_projection_schema,
            "physical_reachability_algorithm_id": (
                payload.physical_reachability_algorithm_id
            ),
            "physical_evidence_algorithm_id": (
                scene.physical_evidence_algorithm_ids[platform]
            ),
            "physical_grid_axis_convention": PHYSICAL_GRID_AXIS_CONVENTION,
            "physical_safe_pose_count": payload.physical_safe_pose_count,
            "physically_reachable_pose_count": (
                payload.physically_reachable_pose_count
            ),
            "physical_projection_sha256": payload.physical_projection_sha256,
            "mission_target_detail_mask_sha256": (
                payload.mission_target_detail_mask_sha256
            ),
            "coverable_detail_shape": list(payload.coverable_detail_shape),
            "mission_target_detail_cell_count": (
                payload.mission_target_detail_cell_count
            ),
            "coverable_detail_cell_count": payload.coverable_detail_cell_count,
            "mission_coverable_fraction": payload.mission_coverable_fraction,
            "initial_coverable_fraction": payload.initial_coverable_fraction,
            "initial_candidate_count": payload.initial_candidate_count,
            "coverable_detail_mask_sha256": (
                payload.coverable_detail_mask_sha256
            ),
            "sensor_visibility_algorithm_id": (
                payload.sensor_visibility_algorithm_id
            ),
            "capability_content_sha256": payload.capability_content_sha256,
            "start_identity_sha256": payload.start_identity_sha256,
            "exact": payload.exact,
            "eligible": payload.eligible,
            "ineligible_reason": (
                None
                if payload.ineligible_reason is None
                else payload.ineligible_reason.value
            ),
            "stage_diagnostics": {
                "physical_safe_pose_count": payload.physical_safe_pose_count,
                "physically_reachable_pose_count": (
                    payload.physically_reachable_pose_count
                ),
                "mission_target_detail_cell_count": (
                    payload.mission_target_detail_cell_count
                ),
                "coverable_detail_cell_count": payload.coverable_detail_cell_count,
                "initial_candidate_count": payload.initial_candidate_count,
            },
        }
    return {
        "scene_id": scene.scene_id,
        "source": scene.source,
        "split": scene.split,
        "window_id": scene.window_id,
        "window_sha256": scene.window_sha256,
        "world_bounds_m": list(scene.world_bounds_m),
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha(path),
        "arrays": {
            name: _array_metadata(values) for name, values in sorted(arrays.items())
        },
        "platform_coverability": platform_coverability,
    }


def _eligibility_summary(
    entries: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Summarize either legacy coverability or v11 start qualification."""
    ordered_entries = tuple(entries)
    platform_summary: dict[str, object] = {}
    readiness: dict[str, object] = {}
    eligible_sets: dict[str, set[str]] = {}
    for platform in _PLATFORMS:
        split_summary: dict[str, object] = {}
        split_counts: dict[str, object] = {}
        eligible_by_split: dict[str, list[str]] = {}
        schedule_ids: dict[str, str] = {}
        all_eligible: set[str] = set()
        for split in _REQUIRED_SPLITS:
            split_entries = tuple(
                entry for entry in ordered_entries if entry.get("split") == split
            )
            eligible_ids: list[str] = []
            reasons: dict[str, int] = {}
            for entry in split_entries:
                payloads = entry.get("platform_starts")
                if payloads is None:
                    payloads = entry["platform_coverability"]
                payload = payloads[platform]
                eligible = payload.get("qualified", payload.get("eligible"))
                if bool(eligible):
                    eligible_ids.append(str(entry["scene_id"]))
                else:
                    reason = str(payload["ineligible_reason"])
                    reasons[reason] = reasons.get(reason, 0) + 1
            eligible_ids.sort()
            total = len(split_entries)
            eligible = len(eligible_ids)
            counts = {
                "total_scene_count": total,
                "eligible_scene_count": eligible,
                "feasibility_rate": eligible / total if total else 0.0,
                "ineligible_reason_counts": dict(sorted(reasons.items())),
            }
            split_summary[split] = counts
            split_counts[split] = {
                "total_scene_count": total,
                "eligible_scene_count": eligible,
            }
            all_eligible.update(eligible_ids)
            eligible_by_split[split] = eligible_ids
            schedule_ids[split] = platform_scenario_schedule_id(
                platform, split, eligible_ids
            )
        platform_summary[platform] = {
            "splits": split_summary,
            "eligible_scene_ids": eligible_by_split,
            "scenario_schedule_ids": schedule_ids,
        }
        readiness[platform] = split_counts
        eligible_sets[platform] = all_eligible

    common_ids = sorted(set.intersection(*(eligible_sets[p] for p in _PLATFORMS)))
    common_splits: dict[str, object] = {}
    by_id = {str(entry["scene_id"]): entry for entry in ordered_entries}
    for split in _REQUIRED_SPLITS:
        ids = [scene_id for scene_id in common_ids if by_id[scene_id]["split"] == split]
        common_splits[split] = {
            "scene_count": len(ids),
            "scene_ids": ids,
            "scenario_schedule_id": _semantic_sha(
                {
                    "schema": "lunar-common-evaluation-schedule/v1",
                    "split": split,
                    "ordered_scene_ids": ids,
                }
            ),
        }
    exact_common = {
        "scene_count": len(common_ids),
        "scene_ids": common_ids,
        "sha256": _semantic_sha(common_ids),
        "splits": common_splits,
    }
    return platform_summary, exact_common, readiness


def write_formal_cache(
    cache_root: str | Path,
    *,
    identity: FormalCacheIdentity,
    scenario_manifest: Mapping[str, object],
    materialization: str,
    scenes: Iterable[FormalSceneIndexRecord],
    repository_root: str | Path,
    source_lock_path: str | Path | None = None,
) -> dict[str, object]:
    """Atomically publish a lightweight scene index; never write scene arrays."""
    if materialization not in {"preflight", "bounded", "full"}:
        raise FormalCacheError(
            "materialization must be preflight, bounded, or full"
        )
    root = _validate_cache_root(Path(cache_root), Path(repository_root))
    manifest_path = root / "cache-manifest.json"
    if manifest_path.is_file():
        existing = load_formal_cache(
            manifest_path,
            expected_identity=identity,
            require_full=materialization in {"bounded", "full"},
        )
        if existing.manifest["materialization"] != materialization:
            raise FormalCacheError("existing cache materialization differs")
        return existing.manifest
    if root.exists() and any(root.iterdir()):
        raise FormalCacheError("incomplete cache root is not reusable")
    root.mkdir(mode=0o755, exist_ok=True)
    if source_lock_path is not None:
        _write_source_locator(
            root,
            Path(source_lock_path),
            identity=identity,
            repository_root=Path(repository_root),
        )
    scenario_payload = _canonical_bytes(dict(scenario_manifest))
    scenario_path = root / "scenario-manifest.json"
    _atomic_bytes(scenario_path, scenario_payload)
    if scenario_manifest.get("scenario_manifest_sha256") != identity.scenario_manifest_sha256:
        raise FormalCacheError("scenario manifest identity differs from cache identity")

    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for scene in scenes:
        if not isinstance(scene, FormalSceneIndexRecord):
            raise FormalCacheError(
                "v11 formal cache accepts scene index records only"
            )
        if scene.scene_id in seen:
            raise FormalCacheError("cache scene IDs must be unique")
        seen.add(scene.scene_id)
        entries.append(scene.to_dict())
    entries.sort(key=lambda value: str(value["scene_id"]))
    platform_eligibility, exact_common, readiness = _eligibility_summary(entries)
    common_ready = all(
        int(exact_common["splits"][split]["scene_count"]) > 0
        for split in _REQUIRED_SPLITS
    )
    qualification_complete = (
        _formal_platform_eligibility_ready(materialization, readiness)
        and common_ready
    )
    inventory = [
        {
            "relative_path": "scenario-manifest.json",
            "size_bytes": scenario_path.stat().st_size,
            "sha256": _file_sha(scenario_path),
        }
    ]
    body: dict[str, object] = {
        "schema": FORMAL_CACHE_SCHEMA,
        "materialization": materialization,
        "formal_eligible": (
            materialization in {"bounded", "full"}
            and qualification_complete
        ),
        "identity": identity.to_dict(),
        "scenario_manifest": inventory[0],
        "scene_count": len(entries),
        "platform_eligibility": platform_eligibility,
        "exact_common_evaluation": exact_common,
        "scenes": entries,
        "inventory": inventory,
    }
    body["cache_manifest_sha256"] = _semantic_sha(body)
    _atomic_bytes(manifest_path, _canonical_bytes(body))
    return body


def _load_verified_scene_arrays(
    root: Path, entry: Mapping[str, object]
) -> dict[str, np.ndarray]:
    path = root / str(entry["relative_path"])
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected = entry["arrays"]
            if set(archive.files) != set(expected):
                raise FormalCacheError("scene array inventory differs")
            output = {
                name: np.ascontiguousarray(archive[name])
                for name in archive.files
            }
    except FormalCacheError:
        raise
    except (OSError, ValueError) as error:
        raise FormalCacheError("scene NPZ is invalid") from error
    for name, values in output.items():
        if _array_metadata(values) != expected[name]:
            raise FormalCacheError(f"scene array {name} hash or metadata differs")
        values.setflags(write=False)
    return output


def _validate_scene_semantics(
    entry: Mapping[str, object], output: Mapping[str, np.ndarray]
) -> None:
    resolution_m, origin_m, bounds_m = _physical_grid_geometry(
        entry["world_bounds_m"]
    )
    for platform in _PLATFORMS:
        prefix = platform.lower()
        payload = entry["platform_coverability"][platform]
        physical = unpack_detail_mask(
            output[f"{prefix}_physical_observation_pose_bits"].copy(),
            (256, 256),
        )
        detail_shape = tuple(payload["coverable_detail_shape"])
        coverable = unpack_detail_mask(
            output[f"{prefix}_coverable_detail_bits"].copy(), detail_shape
        )
        if platform == "HOPPER":
            positions_m = _positions_m_from_canonical_um(
                output["hopper_physical_observation_positions_um"]
            )
            _validate_hopper_position_cells(positions_m, physical, bounds_m)
        else:
            positions_m = _ground_physical_observation_positions_m(
                physical, output["elevation_m"], bounds_m
            )
        start_raw = payload["qualified_start_cell"]
        start_cell = (
            None
            if start_raw is None
            else (int(start_raw[0]), int(start_raw[1]))
        )
        start_position_m: tuple[float, float, float] | None = None
        if start_cell is not None:
            if not bool(physical[start_cell]):
                raise FormalCacheError(
                    "scene physical start is absent from reachability"
                )
            if platform == "HOPPER":
                rows, columns = np.nonzero(physical)
                matching = np.flatnonzero(
                    (rows == start_cell[0]) & (columns == start_cell[1])
                )
                if len(matching) != 1:
                    raise FormalCacheError(
                        "scene Hopper physical start is ambiguous"
                    )
                start_position_m = tuple(
                    float(value)
                    for value in positions_m[int(matching[0])]
                )
            else:
                start_position_m = (
                    origin_m[0]
                    + (float(start_cell[1]) + 0.5) * resolution_m,
                    origin_m[1]
                    - (float(start_cell[0]) + 0.5) * resolution_m,
                    float(output["elevation_m"][start_cell]),
                )
        expected_start_sha256 = _physical_start_identity_from_position(
            platform_type=platform,
            start_cell=start_cell,
            position_m=start_position_m,
        )
        if expected_start_sha256 != payload["start_identity_sha256"]:
            raise FormalCacheError("scene physical start identity differs")
        try:
            expected_projection_sha256 = physical_projection_sha256(
                platform_type=platform,
                physical_reachability_algorithm_id=(
                    payload["physical_reachability_algorithm_id"]
                ),
                physical_evidence_algorithm_id=(
                    payload["physical_evidence_algorithm_id"]
                ),
                physical_observation_pose_mask=physical,
                physical_observation_positions_m=positions_m,
                physical_grid_resolution_m=resolution_m,
                physical_grid_origin_m=origin_m,
                physical_grid_world_bounds_m=bounds_m,
                physical_grid_axis_convention=(
                    payload["physical_grid_axis_convention"]
                ),
                capability_content_sha256=payload["capability_content_sha256"],
                start_identity_sha256=expected_start_sha256,
            )
        except ValueError as error:
            raise FormalCacheError(
                "scene physical projection authority is invalid"
            ) from error
        if (
            expected_projection_sha256
            != payload["physical_projection_sha256"]
        ):
            raise FormalCacheError("scene physical projection hash differs")
        if (
            mask_sha256(coverable)
            != payload["coverable_detail_mask_sha256"]
            or int(physical.sum(dtype=np.int64))
            != payload["physically_reachable_pose_count"]
            or int(coverable.sum(dtype=np.int64))
            != payload["coverable_detail_cell_count"]
        ):
            raise FormalCacheError("scene semantic coverability mask differs")


@dataclass(frozen=True, slots=True)
class FormalSceneIndex:
    root: Path
    manifest: Mapping[str, object]
    identity: FormalCacheIdentity
    source_lock_path: Path | None = None
    source_paths: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        paths: dict[str, Path] = {}
        if not isinstance(self.source_paths, Mapping):
            raise FormalCacheError("scene index source paths are invalid")
        for source, path in self.source_paths.items():
            if source not in _SOURCE_IDS or not isinstance(path, Path):
                raise FormalCacheError("scene index source path is invalid")
            if path.is_symlink() or not path.is_file():
                raise FormalCacheError("scene index source must be a regular file")
            paths[source] = path
        if paths and set(paths) != set(_SOURCE_IDS):
            raise FormalCacheError("scene index source inventory is incomplete")
        if self.source_lock_path is not None and (
            not isinstance(self.source_lock_path, Path)
            or self.source_lock_path.is_symlink()
            or not self.source_lock_path.is_file()
        ):
            raise FormalCacheError("scene index source lock is invalid")
        object.__setattr__(self, "source_paths", MappingProxyType(paths))

    @property
    def formal_eligible(self) -> bool:
        return bool(self.manifest["formal_eligible"])

    def record(self, scene_id: str) -> FormalSceneIndexRecord:
        _require_sha(scene_id, "scene_id")
        entry = next(
            (
                value
                for value in self.manifest["scenes"]
                if value["scene_id"] == scene_id
            ),
            None,
        )
        if entry is None:
            raise FormalCacheError("scene is not present in scene index")
        return FormalSceneIndexRecord(**dict(entry))


# Public compatibility name retained while formal callers migrate to task artifacts.
FormalCache = FormalSceneIndex


@dataclass(frozen=True, slots=True)
class TaskSourceCrop:
    """Source-locked task evidence, cropped before platform projection."""

    scene_id: str
    source_identity_sha256: str
    coarse_bounds_half_open: tuple[int, int, int, int]
    detail_bounds_half_open: tuple[int, int, int, int]
    local_world_bounds_m: tuple[float, float, float, float]
    local_halo_world_bounds_m: tuple[float, float, float, float]
    arrays: Mapping[str, np.ndarray]

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "source_identity_sha256": self.source_identity_sha256,
            "coarse_bounds_half_open": self.coarse_bounds_half_open,
            "detail_bounds_half_open": self.detail_bounds_half_open,
            "local_world_bounds_m": self.local_world_bounds_m,
            "local_halo_world_bounds_m": self.local_halo_world_bounds_m,
            "arrays": self.arrays,
        }


def _safe_relative(value: object) -> str:
    if not isinstance(value, str):
        raise FormalCacheError("cache inventory path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise FormalCacheError("cache inventory path is invalid")
    return value


def _validate_inventory(root: Path, manifest: Mapping[str, object]) -> None:
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list):
        raise FormalCacheError("cache inventory is invalid")
    expected: dict[str, Mapping[str, object]] = {}
    for item in inventory:
        if not isinstance(item, Mapping):
            raise FormalCacheError("cache inventory entry is invalid")
        relative = _safe_relative(item.get("relative_path"))
        if relative in expected:
            raise FormalCacheError("cache inventory contains duplicate paths")
        expected[relative] = item
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise FormalCacheError("cache inventory must not contain symbolic links")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative == "source-locator.json" or relative.startswith(
                "tasks/"
            ):
                continue
            actual.add(relative)
    if actual != set(expected) | {"cache-manifest.json"}:
        raise FormalCacheError("cache inventory has missing or extra files")
    for relative, item in expected.items():
        path = root / relative
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if not isinstance(size, int) or path.stat().st_size != size:
            raise FormalCacheError("cache inventory file size differs")
        if _file_sha(path) != _require_sha(digest, "inventory sha256"):
            raise FormalCacheError("cache inventory file hash differs")


def _shape(value: object, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
    ):
        raise FormalCacheError(f"{name} is invalid")
    return value[0], value[1]


def _validate_scene_index_entry(
    entry: Mapping[str, object],
) -> FormalSceneIndexRecord:
    if set(entry) != _SCENE_INDEX_FIELDS:
        raise FormalCacheError("scene index record fields are invalid")
    return FormalSceneIndexRecord(**dict(entry))


def _validate_scene_coverability(entry: Mapping[str, object]) -> None:
    platform_payloads = entry.get("platform_coverability")
    arrays = entry.get("arrays")
    if (
        set(entry) != _SCENE_ENTRY_FIELDS
        or not isinstance(platform_payloads, Mapping)
        or set(platform_payloads) != set(_PLATFORMS)
        or not isinstance(arrays, Mapping)
        or set(arrays) != _STATIC_SCENE_ARRAYS
    ):
        raise FormalCacheError("cache scene platform coverability is invalid")
    _safe_relative(entry.get("relative_path"))
    _physical_grid_geometry(entry.get("world_bounds_m"))
    elevation_metadata = arrays.get("elevation_m")
    expected_elevation_metadata = {
        "shape": [256, 256],
        "dtype": "<f4",
        "byte_order": "little",
    }
    if not isinstance(elevation_metadata, Mapping) or any(
        elevation_metadata.get(field) != value
        for field, value in expected_elevation_metadata.items()
    ):
        raise FormalCacheError(
            "cache ground projection elevation shape, dtype, or byte order differs"
        )
    for platform in _PLATFORMS:
        payload = platform_payloads[platform]
        if (
            not isinstance(payload, Mapping)
            or set(payload) != _PLATFORM_COVERABILITY_FIELDS
        ):
            raise FormalCacheError("cache platform coverability fields are invalid")
        physical_shape = _shape(
            payload.get("physical_observation_pose_shape"),
            "physical observation pose shape",
        )
        detail_shape = _shape(
            payload.get("coverable_detail_shape"), "coverable detail shape"
        )
        if physical_shape != (256, 256) or (
            detail_shape[0] % 256 or detail_shape[1] % 256
        ):
            raise FormalCacheError("cache coverability geometry is invalid")
        cell = payload.get("qualified_start_cell")
        parsed_cell: tuple[int, int] | None
        if cell is None:
            parsed_cell = None
        elif (
            isinstance(cell, list)
            and len(cell) == 2
            and all(type(value) is int and 0 <= value < 256 for value in cell)
        ):
            parsed_cell = (cell[0], cell[1])
        else:
            raise FormalCacheError("cache qualified start cell is invalid")
        target_count = payload.get("mission_target_detail_cell_count")
        coverable_count = payload.get("coverable_detail_cell_count")
        initial_candidate_count = payload.get("initial_candidate_count")
        physical_safe_pose_count = payload.get("physical_safe_pose_count")
        physically_reachable_pose_count = payload.get(
            "physically_reachable_pose_count"
        )
        initial_fraction = payload.get("initial_coverable_fraction")
        mission_fraction = payload.get("mission_coverable_fraction")
        if (
            type(target_count) is not int
            or type(coverable_count) is not int
            or type(initial_candidate_count) is not int
            or type(physical_safe_pose_count) is not int
            or type(physically_reachable_pose_count) is not int
            or physical_safe_pose_count < 0
            or physically_reachable_pose_count < 0
            or physically_reachable_pose_count > physical_safe_pose_count
            or not isinstance(initial_fraction, float)
            or not isinstance(mission_fraction, float)
        ):
            raise FormalCacheError("cache coverability counts are invalid")
        if (parsed_cell is None) is not (physically_reachable_pose_count == 0):
            raise FormalCacheError(
                "cache qualified start disagrees with physical reachability"
            )
        try:
            expected_reason = classify_ineligibility(
                qualified_start_cell=parsed_cell,
                mission_target_detail_cell_count=target_count,
                coverable_detail_cell_count=coverable_count,
                initial_coverable_fraction=initial_fraction,
                initial_candidate_count=initial_candidate_count,
            )
        except ValueError as error:
            raise FormalCacheError("cache coverability gates are invalid") from error
        expected_fraction = coverable_count / target_count if target_count else 0.0
        if not np.isclose(mission_fraction, expected_fraction, rtol=0.0, atol=1.0e-12):
            raise FormalCacheError("cache mission coverable fraction differs")
        eligible = payload.get("eligible")
        reason = payload.get("ineligible_reason")
        if (
            payload.get("exact") is not True
            or type(eligible) is not bool
            or eligible is not (expected_reason is None)
            or reason
            != (None if expected_reason is None else expected_reason.value)
        ):
            raise FormalCacheError("cache platform eligibility differs")
        for name in (
            "physical_projection_schema",
            "physical_reachability_algorithm_id",
            "physical_evidence_algorithm_id",
            "sensor_visibility_algorithm_id",
        ):
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise FormalCacheError("cache coverability algorithm is missing")
        if payload["physical_projection_schema"] != PHYSICAL_PROJECTION_SCHEMA:
            raise FormalCacheError("cache physical projection schema is unsupported")
        if (
            payload.get("physical_grid_axis_convention")
            != PHYSICAL_GRID_AXIS_CONVENTION
        ):
            raise FormalCacheError("cache physical grid axis differs")
        for name in (
            "physical_projection_sha256",
            "mission_target_detail_mask_sha256",
            "coverable_detail_mask_sha256",
            "capability_content_sha256",
            "start_identity_sha256",
        ):
            _require_sha(payload.get(name), name)
        diagnostics = payload.get("stage_diagnostics")
        if not isinstance(diagnostics, Mapping) or diagnostics != {
            "physical_safe_pose_count": physical_safe_pose_count,
            "physically_reachable_pose_count": physically_reachable_pose_count,
            "mission_target_detail_cell_count": target_count,
            "coverable_detail_cell_count": coverable_count,
            "initial_candidate_count": initial_candidate_count,
        }:
            raise FormalCacheError("cache coverability diagnostics are invalid")
        prefix = platform.lower()
        expected_arrays = {
            f"{prefix}_physical_observation_pose_bits": {
                "shape": [(256 * 256 + 7) // 8],
                "dtype": "|u1",
                "byte_order": "not-applicable",
            },
            f"{prefix}_coverable_detail_bits": {
                "shape": [(detail_shape[0] * detail_shape[1] + 7) // 8],
                "dtype": "|u1",
                "byte_order": "not-applicable",
            },
            f"{prefix}_coverable_ratio": {
                "shape": [256, 256],
                "dtype": "<f4",
                "byte_order": "little",
            },
        }
        if platform == "HOPPER":
            expected_arrays["hopper_physical_observation_positions_um"] = {
                "shape": [physically_reachable_pose_count, 3],
                "dtype": "<i8",
                "byte_order": "little",
            }
        for name, expected in expected_arrays.items():
            metadata = arrays.get(name)
            if not isinstance(metadata, Mapping):
                raise FormalCacheError("cache coverability array metadata differs")
            for field, value in expected.items():
                if metadata.get(field) != value:
                    raise FormalCacheError(
                        f"cache coverability array {field} differs"
                    )


def load_formal_cache(
    cache_manifest: str | Path,
    *,
    expected_identity: FormalCacheIdentity | None = None,
    require_full: bool = False,
) -> FormalCache:
    path = Path(cache_manifest)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise FormalCacheError("cache manifest must be an absolute regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalCacheError("cache manifest is invalid UTF-8 JSON") from error
    if not isinstance(value, dict) or value.get("schema") != FORMAL_CACHE_SCHEMA:
        raise FormalCacheError("cache manifest schema is unsupported")
    _reject_residual_primitive_identity(value)
    claimed = _require_sha(
        value.get("cache_manifest_sha256"), "cache_manifest_sha256"
    )
    body = dict(value)
    body.pop("cache_manifest_sha256")
    if _semantic_sha(body) != claimed:
        raise FormalCacheError("cache manifest identity mismatch")
    materialization = value.get("materialization")
    if materialization not in {"preflight", "bounded", "full"}:
        raise FormalCacheError("cache materialization is invalid")
    formal_eligible = value.get("formal_eligible")
    if type(formal_eligible) is not bool or (
        materialization == "preflight" and formal_eligible
    ):
        raise FormalCacheError("cache formal eligibility is invalid")
    if require_full and (
        materialization not in {"bounded", "full"} or not formal_eligible
    ):
        raise FormalCacheError(
            "formal command requires a bounded or full, start-qualified cache"
        )
    identity = FormalCacheIdentity.from_dict(value.get("identity"))
    if require_full and identity.reward_sha256 != reward_weights_sha256():
        raise FormalCacheError("cache identity reward_sha256 differs")
    if (
        require_full
        and identity.training_semantics_sha256
        != training_semantics_sha256()
    ):
        raise FormalCacheError(
            "cache identity training_semantics_sha256 differs"
        )
    if expected_identity is not None:
        actual = identity.to_dict()
        expected = expected_identity.to_dict()
        for field in _IDENTITY_FIELDS:
            if actual[field] != expected[field]:
                raise FormalCacheError(f"cache identity {field} differs")
    scenes = value.get("scenes")
    if not isinstance(scenes, list) or value.get("scene_count") != len(scenes):
        raise FormalCacheError("cache scene inventory is invalid")
    scene_ids = [entry.get("scene_id") for entry in scenes if isinstance(entry, Mapping)]
    if len(scene_ids) != len(scenes) or len(set(scene_ids)) != len(scene_ids):
        raise FormalCacheError("cache scene identity inventory is invalid")
    for entry in scenes:
        if not isinstance(entry, Mapping) or entry.get("split") not in _REQUIRED_SPLITS:
            raise FormalCacheError("cache scene split is invalid")
        _validate_scene_index_entry(entry)
    platform_eligibility, exact_common, readiness = _eligibility_summary(scenes)
    if (
        value.get("platform_eligibility") != platform_eligibility
        or value.get("exact_common_evaluation") != exact_common
    ):
        raise FormalCacheError("cache platform eligibility summary differs")
    common_ready = all(
        int(exact_common["splits"][split]["scene_count"]) > 0
        for split in _REQUIRED_SPLITS
    )
    expected_formal_eligible = (
        _formal_platform_eligibility_ready(materialization, readiness)
        and common_ready
    )
    if formal_eligible is not expected_formal_eligible:
        raise FormalCacheError("cache formal platform eligibility is invalid")
    _validate_inventory(path.parent, value)
    source_lock_path, source_paths = _load_source_locator(
        path.parent,
        identity=identity,
    )
    return FormalSceneIndex(
        path.parent,
        value,
        identity,
        source_lock_path=source_lock_path,
        source_paths=source_paths,
    )


def _read_external_json(path: Path, repository_root: Path, name: str) -> dict[str, object]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise FormalCacheError(f"{name} must be an absolute regular file")
    if path.resolve(strict=True).is_relative_to(repository_root.resolve(strict=True)):
        raise FormalCacheError(f"{name} must be outside the repository")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalCacheError(f"{name} is invalid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise FormalCacheError(f"{name} must contain a JSON object")
    return value


def _git_output(repository_root: Path, arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalCacheError("current C++ v3 source identity is unavailable") from error
    return result.stdout.strip()


def current_v3_identity(repository_root: str | Path) -> tuple[str, str]:
    """Hash the tracked planner/bridge trees and reject uncommitted v3 drift."""
    root = Path(repository_root).resolve(strict=True)
    paths = (
        "ros2_ws/src/lunar_navigation_msgs",
        "ros2_ws/src/lunar_planning_msgs",
        "ros2_ws/src/lunar_planner_core",
        "ros2_ws/src/lunar_planner_training_bridge",
    )
    status = _git_output(root, ["status", "--porcelain", "--", *paths])
    if status:
        raise FormalCacheError("current C++ v3 source has uncommitted drift")
    commit = _git_output(root, ["log", "-1", "--format=%H", "--", *paths])
    _require_sha(commit, "v3_source_commit", length=40)
    trees = {
        path: _git_output(root, ["rev-parse", f"HEAD:{path}"]) for path in paths
    }
    for path, digest in trees.items():
        _require_sha(digest, f"v3 tree {path}", length=40)
    return commit, _semantic_sha(trees)


def build_formal_cache_identity(
    *,
    source_lock_path: str | Path,
    split_manifest_path: str | Path,
    scenario_manifest: Mapping[str, object],
    capability_bundle: FrozenCapabilityBundle,
    repository_root: str | Path,
) -> FormalCacheIdentity:
    root = Path(repository_root).resolve(strict=True)
    source_lock = Path(source_lock_path)
    split_path = Path(split_manifest_path)
    _, locks = load_aggregate_source_lock(source_lock, repository_root=root)
    split = _read_external_json(split_path, root, "split manifest")
    return _cache_identity_from_verified(
        source_lock=source_lock,
        split_path=split_path,
        split=split,
        scenario_manifest=scenario_manifest,
        capability_bundle=capability_bundle,
        repository_root=root,
        source_sha256s={lock.source_id: lock.sha256 for lock in locks},
    )


def _cache_identity_from_verified(
    *,
    source_lock: Path,
    split_path: Path,
    split: Mapping[str, object],
    scenario_manifest: Mapping[str, object],
    capability_bundle: FrozenCapabilityBundle,
    repository_root: Path,
    source_sha256s: Mapping[str, str],
) -> FormalCacheIdentity:
    if split.get("split_sha256") != scenario_manifest.get("split_sha256"):
        raise FormalCacheError("split identity differs from scenario manifest")
    if not capability_bundle.formal_eligible:
        raise FormalCacheError("formal cache requires the approved capability bundle")
    v3_commit, v3_sha = current_v3_identity(repository_root)
    return FormalCacheIdentity(
        source_lock_file_sha256=_file_sha(source_lock),
        source_sha256s=source_sha256s,
        split_manifest_file_sha256=_file_sha(split_path),
        split_sha256=str(split["split_sha256"]),
        scenario_manifest_sha256=str(
            scenario_manifest["scenario_manifest_sha256"]
        ),
        generator_sha256=GENERATOR_SHA256,
        capability_sha256=capability_bundle.bundle_sha256,
        reward_sha256=reward_weights_sha256(),
        training_semantics_sha256=training_semantics_sha256(),
        v3_source_commit=v3_commit,
        v3_sha256=v3_sha,
    )


@dataclass(frozen=True)
class _VerifiedPreparationInputs:
    data_root: Path
    source_paths: Mapping[str, Path]
    split_document: dict[str, object]
    scenario_manifest: dict[str, object]
    identity: FormalCacheIdentity


def _verified_preparation_inputs(
    *,
    source_lock_path: Path,
    split_manifest_path: Path,
    capability_bundle: FrozenCapabilityBundle,
    repository_root: Path,
) -> _VerifiedPreparationInputs:
    data_root, locks = load_aggregate_source_lock(
        source_lock_path,
        repository_root=repository_root,
    )
    split = _read_external_json(split_manifest_path, repository_root, "split manifest")
    scenario = build_scenario_manifest_document(
        split_document=split,
        source_lock_file_sha256=_file_sha(source_lock_path),
        split_manifest_file_sha256=_file_sha(split_manifest_path),
        source_sha256s={lock.source_id: lock.sha256 for lock in locks},
    )
    identity = _cache_identity_from_verified(
        source_lock=source_lock_path,
        split_path=split_manifest_path,
        split=split,
        scenario_manifest=scenario,
        capability_bundle=capability_bundle,
        repository_root=repository_root,
        source_sha256s={lock.source_id: lock.sha256 for lock in locks},
    )
    return _VerifiedPreparationInputs(
        data_root=data_root,
        source_paths=MappingProxyType(
            {lock.source_id: data_root / lock.filename for lock in locks}
        ),
        split_document=split,
        scenario_manifest=scenario,
        identity=identity,
    )


def _selected_scenarios(
    scenario_manifest: Mapping[str, object],
    materialization: str,
    preflight_scenario_limit: int | None,
) -> tuple[Mapping[str, object], ...]:
    scenarios = scenario_manifest.get("scenarios")
    if not isinstance(scenarios, list) or any(
        not isinstance(value, Mapping) for value in scenarios
    ):
        raise FormalCacheError("scenario manifest inventory is invalid")
    if materialization == "full":
        if preflight_scenario_limit is not None:
            raise FormalCacheError("full materialization rejects a scenario limit")
        return tuple(scenarios)
    if materialization == "bounded":
        if preflight_scenario_limit != BOUNDED_FORMAL_SCENE_COUNT:
            raise FormalCacheError(
                "bounded materialization requires exactly 128 scenarios"
            )
        selected: list[Mapping[str, object]] = []
        for split in _REQUIRED_SPLITS:
            split_scenarios = tuple(
                item for item in scenarios if item.get("split") == split
            )
            required = BOUNDED_FORMAL_SPLIT_COUNTS[split]
            if len(split_scenarios) < required:
                raise FormalCacheError(
                    f"bounded materialization split {split} is too small"
                )
            selected.extend(split_scenarios[:required])
        return tuple(selected)
    if (
        type(preflight_scenario_limit) is not int
        or preflight_scenario_limit < 1
        or preflight_scenario_limit > len(scenarios)
    ):
        raise FormalCacheError("preflight materialization requires a bounded scenario limit")
    selected: list[Mapping[str, object]] = []
    for split in ("train", "validation", "test", "holdout"):
        match = next((item for item in scenarios if item.get("split") == split), None)
        if match is not None and len(selected) < preflight_scenario_limit:
            selected.append(match)
    for item in scenarios:
        if len(selected) >= preflight_scenario_limit:
            break
        if item not in selected:
            selected.append(item)
    return tuple(selected)


def _jaxa_canvas(
    archive_path: Path,
    scenario: Mapping[str, object],
) -> tuple[MapCanvas, str]:
    import rasterio

    member_paths = scenario.get("archive_member_paths")
    member_hashes = scenario.get("archive_member_sha256s")
    if not isinstance(member_paths, list) or not isinstance(member_hashes, list):
        raise FormalCacheError("JAXA scenario member inventory is invalid")
    dem_index = next(
        (index for index, value in enumerate(member_paths) if "/DTM" in value.upper() or "-DEM" in value.upper()),
        None,
    )
    if dem_index is None:
        raise FormalCacheError("JAXA scenario has no DTM member")
    member = str(member_paths[dem_index])
    member_sha = _require_sha(member_hashes[dem_index], "JAXA DTM member")
    uri = f"zip://{archive_path}!{member}"
    try:
        with rasterio.open(uri) as dataset:
            center_x = (dataset.bounds.left + dataset.bounds.right) / 2.0
            center_y = (dataset.bounds.bottom + dataset.bounds.top) / 2.0
    except (OSError, rasterio.errors.RasterioError) as error:
        raise FormalCacheError("JAXA DTM cannot be opened") from error
    half = 512.0
    return (
        MapCanvas(
            member_sha,
            (center_x - half, center_y - half, center_x + half, center_y + half),
        ),
        uri,
    )


def _load_base_scene(
    scenario: Mapping[str, object],
    source_paths: Mapping[str, Path],
    base_cache: dict[str, tuple[MapCanvas, np.ndarray, np.ndarray]],
) -> tuple[MapCanvas, np.ndarray, np.ndarray]:
    source = scenario.get("source")
    window_id = str(scenario.get("window_id"))
    cached = base_cache.get(window_id)
    if cached is not None:
        return cached
    if source == "NASA_LOLA":
        window_sha = _require_sha(scenario.get("window_sha256"), "NASA window")
        bounds = scenario.get("world_bounds_m")
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise FormalCacheError("NASA scene world bounds are invalid")
        canvas = MapCanvas(window_sha, tuple(float(value) for value in bounds))
        path: str | Path = source_paths["NASA_LOLA_87S_DEM"]
    elif source == "JAXA_LUPEX":
        canvas, path = _jaxa_canvas(
            source_paths["JAXA_LUPEX_DATA_S1"], scenario
        )
    else:
        raise FormalCacheError("scenario source is unsupported")
    loaded = load_polar_window(path, canvas)
    cached = (canvas, loaded.elevation_m, loaded.observed_mask)
    base_cache[window_id] = cached
    return cached


def _vector_scene(
    scenario: Mapping[str, object], canvas: MapCanvas
) -> VectorHazardScene:
    if scenario.get("procedural_overlay") is True:
        scenario_seed = scenario.get("scenario_seed")
        if type(scenario_seed) is not int:
            raise FormalCacheError("NASA scenario seed is invalid")
        scene = generate_vector_hazard_scene(
            canvas.window_sha256,
            scenario_seed,
            canvas=canvas,
        )
        if scene.seed != scenario.get("scene_seed"):
            raise FormalCacheError("generated scene seed differs from manifest")
        return scene
    if scenario.get("procedural_overlay") is not False:
        raise FormalCacheError("scenario overlay flag is invalid")
    return VectorHazardScene(
        seed=_require_sha(scenario.get("scene_seed"), "JAXA scene seed"),
        canvas=canvas,
        rocks=(),
        craters=(),
        no_go_polygons=(),
    )


def _task_world_bounds(
    canvas: MapCanvas,
    bounds: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    row0, row1, column0, column1 = bounds
    left, _, _, top = canvas.bounds_m
    resolution = canvas.geometry.resolution_m
    return (
        left + column0 * resolution,
        top - row1 * resolution,
        left + column1 * resolution,
        top - row0 * resolution,
    )


def _task_source_scenario(
    scene_index: FormalSceneIndex,
    scene_id: str,
) -> Mapping[str, object]:
    document = load_scenario_manifest(
        scene_index.root / "scenario-manifest.json"
    )
    scenarios = document.get("scenarios")
    if not isinstance(scenarios, list):
        raise FormalCacheError("scene index scenario inventory is invalid")
    matches = tuple(
        scenario
        for scenario in scenarios
        if isinstance(scenario, Mapping) and scenario.get("scene_id") == scene_id
    )
    if len(matches) != 1:
        raise FormalCacheError("scene index scenario record is missing")
    return matches[0]


def load_task_source_crop(
    *,
    scene_index: FormalSceneIndex,
    scene_id: str,
    geometry: FrozenTaskGeometry,
) -> TaskSourceCrop:
    """Read only one task plus its declared evidence halo from locked source."""
    if not isinstance(scene_index, FormalSceneIndex):
        raise TypeError("task source crop requires FormalSceneIndex")
    if not isinstance(geometry, FrozenTaskGeometry):
        raise TypeError("task source crop requires FrozenTaskGeometry")
    record = scene_index.record(scene_id)
    if set(scene_index.source_paths) != set(_SOURCE_IDS):
        raise FormalCacheError("scene index has no verified source locator")
    scenario = _task_source_scenario(scene_index, scene_id)
    if record.source == "NASA_LOLA":
        full_canvas = MapCanvas(record.window_sha256, record.world_bounds_m)
        source_path: str | Path = scene_index.source_paths[
            "NASA_LOLA_87S_DEM"
        ]
    elif record.source == "JAXA_LUPEX":
        full_canvas, source_path = _jaxa_canvas(
            scene_index.source_paths["JAXA_LUPEX_DATA_S1"],
            scenario,
        )
        if full_canvas.bounds_m != record.world_bounds_m:
            raise FormalCacheError("scene index JAXA bounds differ")
    else:  # pragma: no cover - FormalSceneIndexRecord closes this union
        raise FormalCacheError("task source scene type is unsupported")

    halo = geometry.halo_coarse_bounds_half_open
    row0, row1, column0, column1 = halo
    if (
        min(row0, column0) < 0
        or row1 > full_canvas.geometry.cells
        or column1 > full_canvas.geometry.cells
        or row1 <= row0
        or column1 <= column0
    ):
        raise FormalCacheError("task source halo bounds are invalid")
    side = max(row1 - row0, column1 - column0)
    square_row0 = max(
        0,
        min(row0, full_canvas.geometry.cells - side),
    )
    square_column0 = max(
        0,
        min(column0, full_canvas.geometry.cells - side),
    )
    square_bounds = (
        square_row0,
        square_row0 + side,
        square_column0,
        square_column0 + side,
    )
    square_world = _task_world_bounds(full_canvas, square_bounds)
    coarse_geometry = GridGeometry(
        size_m=side * full_canvas.geometry.resolution_m,
        resolution_m=full_canvas.geometry.resolution_m,
        cells=side,
    )
    local_canvas = MapCanvas(
        record.window_sha256,
        square_world,
        geometry=coarse_geometry,
    )
    loaded = load_polar_window(source_path, local_canvas)
    full_vector = _vector_scene(scenario, full_canvas)
    local_vector = VectorHazardScene(
        seed=full_vector.seed,
        canvas=local_canvas,
        rocks=full_vector.rocks,
        craters=full_vector.craters,
        no_go_polygons=full_vector.no_go_polygons,
    )
    local_scene = MultiResolutionScene(
        local_canvas,
        loaded.elevation_m,
        loaded.observed_mask,
        local_vector,
        scenario_id=scene_id,
    )
    coarse = local_scene.project(local_canvas)
    detail_geometry = GridGeometry(
        size_m=coarse_geometry.size_m,
        resolution_m=coarse_geometry.resolution_m / DETAIL_PER_GLOBAL,
        cells=side * DETAIL_PER_GLOBAL,
    )
    detail_canvas = MapCanvas(
        record.window_sha256,
        square_world,
        geometry=detail_geometry,
    )
    detail = local_scene.project(detail_canvas)

    coarse_rows = slice(row0 - square_row0, row1 - square_row0)
    coarse_columns = slice(
        column0 - square_column0,
        column1 - square_column0,
    )
    detail_rows = slice(
        (row0 - square_row0) * DETAIL_PER_GLOBAL,
        (row1 - square_row0) * DETAIL_PER_GLOBAL,
    )
    detail_columns = slice(
        (column0 - square_column0) * DETAIL_PER_GLOBAL,
        (column1 - square_column0) * DETAIL_PER_GLOBAL,
    )

    def crop(values: np.ndarray, rows: slice, columns: slice) -> np.ndarray:
        return np.ascontiguousarray(values[rows, columns])

    arrays = {
        "elevation_m": crop(
            coarse.elevation_m,
            coarse_rows,
            coarse_columns,
        ),
        "valid_mask": crop(
            coarse.valid_mask,
            coarse_rows,
            coarse_columns,
        ).astype(np.bool_, copy=False),
        "physical_obstacle_ratio": crop(
            coarse.physical_obstacle_ratio,
            coarse_rows,
            coarse_columns,
        ),
        "physical_obstacle_height_m": crop(
            coarse.physical_obstacle_height_m,
            coarse_rows,
            coarse_columns,
        ),
        "forbidden_ratio": crop(
            coarse.forbidden_ratio,
            coarse_rows,
            coarse_columns,
        ),
        "detail_elevation_m": crop(
            detail.elevation_m,
            detail_rows,
            detail_columns,
        ),
        "detail_valid_mask": crop(
            detail.valid_mask,
            detail_rows,
            detail_columns,
        ).astype(np.bool_, copy=False),
        "detail_physical_obstacle_ratio": crop(
            detail.physical_obstacle_ratio,
            detail_rows,
            detail_columns,
        ),
        "detail_physical_obstacle_height_m": crop(
            detail.physical_obstacle_height_m,
            detail_rows,
            detail_columns,
        ),
        "detail_forbidden_ratio": crop(
            detail.forbidden_ratio,
            detail_rows,
            detail_columns,
        ),
    }
    return TaskSourceCrop(
        scene_id=record.scene_id,
        source_identity_sha256=scene_index.identity.source_lock_file_sha256,
        coarse_bounds_half_open=halo,
        detail_bounds_half_open=tuple(
            value * DETAIL_PER_GLOBAL for value in halo
        ),
        local_world_bounds_m=_task_world_bounds(
            full_canvas,
            geometry.coarse_bounds_half_open,
        ),
        local_halo_world_bounds_m=_task_world_bounds(full_canvas, halo),
        arrays=MappingProxyType(arrays),
    )


def _bridge_vec3(bridge_api: object, x: float, y: float, z: float) -> object:
    value = bridge_api.Vec3()
    value.x = float(x)
    value.y = float(y)
    value.z = float(z)
    return value


def _bridge_pose(bridge_api: object, x: float, y: float, z: float) -> object:
    value = bridge_api.Pose3()
    value.position_m = _bridge_vec3(bridge_api, x, y, z)
    value.orientation.w = 1.0
    return value


def _bridge_grid_map(
    bridge_api: object,
    *,
    canvas: MapCanvas,
    elevation_m: np.ndarray,
    valid_mask: np.ndarray,
    obstacle_ratio: np.ndarray,
    obstacle_height_m: np.ndarray,
    forbidden_ratio: np.ndarray,
    frame_id: str,
) -> object:
    def layer(values: np.ndarray, dtype: object) -> object:
        south_up = np.ascontiguousarray(
            np.flipud(np.asarray(values)).astype(dtype, copy=False).reshape(-1)
        )
        return bridge_api.GridLayer(south_up)

    valid = np.asarray(valid_mask, dtype=bool)
    obstacle = (np.asarray(obstacle_ratio) > 0.0) & valid
    forbidden = (np.asarray(forbidden_ratio) > 0.0) & valid
    grid = bridge_api.GridMap()
    grid.frame_id = frame_id
    grid.stamp.nanoseconds_since_epoch = 1_000_000_000
    grid.width = canvas.geometry.cells
    grid.height = canvas.geometry.cells
    grid.resolution_m = canvas.geometry.resolution_m
    grid.origin_m = _bridge_vec3(
        bridge_api, canvas.bounds_m[0], canvas.bounds_m[1], 0.0
    )
    zeros = np.zeros(valid.shape, dtype=np.float32)
    grid.layers = {
        "elevation": layer(np.where(valid, elevation_m, 0.0), np.float32),
        "valid_mask": layer(valid, np.uint8),
        "obstacle": layer(obstacle, np.uint8),
        "obstacle_height": layer(
            np.where(obstacle, obstacle_height_m, 0.0), np.float32
        ),
        "observation_age_s": layer(zeros, np.float32),
        "observation_quality": layer(valid, np.float32),
        "elevation_variance": layer(zeros, np.float32),
        "obstacle_variance": layer(zeros, np.float32),
        "observation_count": layer(valid, np.uint32),
        "forbidden": layer(forbidden, np.uint8),
    }
    return grid


def _projection_request(
    platform: FrozenPlatformCapability,
    scene: MultiResolutionScene,
    projected: object,
    *,
    start_cell: tuple[int, int] | None = None,
    local_projected: object | None = None,
    exact_start_position_m: tuple[float, float, float] | None = None,
) -> object:
    import lunar_planner_training_bridge as bridge_api

    from ..policy.action_semantics import apply_goal_theta

    projection_canvas = projected.canvas
    if start_cell is None:
        valid_indices = np.argwhere(projected.valid_mask)
        if valid_indices.size == 0:
            raise FormalCacheError("scene has no valid source elevation")
        center = np.asarray(
            [
                (projection_canvas.geometry.cells - 1) / 2.0,
                (projection_canvas.geometry.cells - 1) / 2.0,
            ]
        )
        row, column = valid_indices[
            int(np.argmin(np.sum((valid_indices - center) ** 2, axis=1)))
        ]
    else:
        row, column = start_cell
        if (
            type(row) is not int
            or type(column) is not int
            or not 0 <= row < projection_canvas.geometry.cells
            or not 0 <= column < projection_canvas.geometry.cells
            or not bool(projected.valid_mask[row, column])
        ):
            raise FormalCacheError("projection start cell is invalid")
    x_m, y_m = projection_canvas.grid_center_world(int(row), int(column))
    z_m = float(projected.elevation_m[row, column])
    if exact_start_position_m is not None:
        if (
            platform.platform_type != "HOPPER"
            or start_cell is None
            or not isinstance(exact_start_position_m, tuple)
            or len(exact_start_position_m) != 3
            or any(type(value) is not float for value in exact_start_position_m)
        ):
            raise FormalCacheError("exact Hopper start position is invalid")
        exact = np.ascontiguousarray(
            np.asarray(exact_start_position_m, dtype=np.float64).reshape((1, 3))
        )
        canonical = _positions_m_from_canonical_um(
            canonical_physical_positions_um(exact)
        )
        if not np.array_equal(exact, canonical):
            raise FormalCacheError(
                "exact Hopper start position must be canonical micrometres"
            )
        try:
            exact_cell = projection_canvas.world_to_grid(
                float(exact[0, 0]), float(exact[0, 1])
            )
        except ValueError as error:
            raise FormalCacheError(
                "exact Hopper start position leaves the canvas"
            ) from error
        if exact_cell != (int(row), int(column)):
            raise FormalCacheError(
                "exact Hopper start position leaves its start cell"
            )
        x_m, y_m, z_m = (
            float(exact[0, 0]),
            float(exact[0, 1]),
            float(exact[0, 2]),
        )
    request = bridge_api.TrainingPlanRequest()
    request.request_id = f"formal-cache/{scene.scene_id}/{platform.platform_type}"
    request.mission_id = f"formal-cache/{scene.scene_id}"
    request.mission_revision = 1
    request.platform_id = platform.platform_id
    request.capability_version = platform.capability_version
    request.global_map_generation = 1
    request.local_map_generation = 1
    request.map_from_odom_generation = 1
    request.state_time.nanoseconds_since_epoch = 1_000_000_000
    state_z_m = z_m
    if platform.platform_type == "LEGGED":
        body_height = platform.typed_capability.body_height_m
        state_z_m += (body_height.lower + body_height.upper) / 2.0
    pose = _bridge_pose(bridge_api, x_m, y_m, state_z_m)
    if platform.platform_type == "WHEELED":
        state = bridge_api.WheeledState()
        state.pose = pose
    elif platform.platform_type == "LEGGED":
        state = bridge_api.LeggedState()
        state.body_pose = pose
    else:
        state = bridge_api.HopperState()
        state.pose = pose
    request.current_state = state
    request.capability = platform.to_bridge_capability()
    goal = bridge_api.PointGoal()
    goal.position_m = _bridge_vec3(bridge_api, x_m, y_m, z_m)
    goal.tolerance_m = 0.0 if platform.platform_type == "HOPPER" else 0.2
    request.goal.goal_id = f"formal-cache/{platform.platform_type.lower()}"
    request.goal.target = goal
    apply_goal_theta(request.goal, platform.platform_type, 0.0)
    global_map_arguments = {
        "canvas": projection_canvas,
        "elevation_m": projected.elevation_m,
        "valid_mask": projected.valid_mask,
        "obstacle_ratio": projected.physical_obstacle_ratio,
        "obstacle_height_m": projected.physical_obstacle_height_m,
        "forbidden_ratio": projected.forbidden_ratio,
    }
    request.world.global_map = _bridge_grid_map(
        bridge_api, frame_id="map", **global_map_arguments
    )
    local = projected if local_projected is None else local_projected
    local_map_arguments = {
        "canvas": local.canvas,
        "elevation_m": local.elevation_m,
        "valid_mask": local.valid_mask,
        "obstacle_ratio": local.physical_obstacle_ratio,
        "obstacle_height_m": local.physical_obstacle_height_m,
        "forbidden_ratio": local.forbidden_ratio,
    }
    request.world.local_map = _bridge_grid_map(
        bridge_api, frame_id="odom", **local_map_arguments
    )
    request.world.map_from_odom.parent_frame = "map"
    request.world.map_from_odom.child_frame = "odom"
    request.world.map_from_odom.stamp.nanoseconds_since_epoch = 1_000_000_000
    resolution = projection_canvas.geometry.resolution_m
    request.config.global_map.base_resolution_m = resolution
    request.config.wheel.xy_resolution_m = resolution
    request.config.wheel.yaw_bin_count = 64
    request.config.legged.xy_resolution_m = resolution
    request.config.legged.yaw_bin_count = 64
    return request


def _detail_intrinsic_projection(
    *,
    platform: FrozenPlatformCapability,
    scene: MultiResolutionScene,
    projected: object,
    bridge: object,
) -> np.ndarray:
    valid = np.asarray(projected.valid_mask, dtype=np.bool_)
    if not valid.any():
        return np.zeros(valid.shape, dtype=np.bool_)
    output = bridge.project_traversability(
        _projection_request(platform, scene, projected)
    )
    intrinsic = np.ascontiguousarray(
        np.flipud(output.intrinsic_feasible).astype(np.bool_)
    )
    if intrinsic.shape != projected.valid_mask.shape:
        raise FormalCacheError("detail intrinsic projection geometry differs")
    return intrinsic


def _build_platform_detail_coverability(
    *,
    platform: FrozenPlatformCapability,
    scene: MultiResolutionScene,
    mission_roi: np.ndarray,
    physical_observation_pose_mask: np.ndarray,
    physical_observation_positions_m: np.ndarray | None,
    bridge: object,
) -> StreamedDetailCoverability:
    from ..environment.visibility import NativeVisibilityEstimator, SensorGeometry

    provider = SceneTileProvider(scene, capacity=1)
    observation = platform.observation_capability
    estimator = NativeVisibilityEstimator(
        SensorGeometry(observation.sensor_range_m, observation.sensor_fov_rad),
        resolution_m=provider.tile_geometry.resolution_m,
    )
    return build_streamed_detail_coverability(
        tile_provider=provider,
        inside_mission_roi=np.ascontiguousarray(mission_roi, dtype=np.bool_),
        reachable_pose_mask=np.ascontiguousarray(
            physical_observation_pose_mask, dtype=np.bool_
        ),
        observation_positions_m=(
            None
            if physical_observation_positions_m is None
            else np.ascontiguousarray(
                physical_observation_positions_m, dtype=np.float64
            )
        ),
        intrinsic_terrain_feasible=lambda projected: (
            _detail_intrinsic_projection(
                platform=platform,
                scene=scene,
                projected=projected,
                bridge=bridge,
            )
        ),
        reveal_from_pose=estimator.reveal_from_pose,
    )


@dataclass(frozen=True, slots=True)
class _HopperTruthCoverability:
    physical_observation_pose_mask: np.ndarray
    observation_positions_m: np.ndarray
    physical_safe_pose_count: int
    physical_reachability_algorithm_id: str
    physical_evidence_algorithm_id: str
    detail: StreamedDetailCoverability
    initial_coverable_fraction: float


def _merge_hopper_trajectory_patches(
    patches: Iterable[object],
) -> object:
    """Union one closure round into a single atomic observation patch."""
    from ..environment.multires_observation import (
        HopperTrajectoryObservationPatch,
        HopperTrajectoryTilePatch,
    )

    iterator = iter(patches)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise FormalCacheError("hopper closure round has no trajectory patches") from error
    if not isinstance(first, HopperTrajectoryObservationPatch):
        raise FormalCacheError("hopper closure trajectory patch is invalid")
    scene_id = first.scene_id
    evidence_generation = first.evidence_generation
    trajectory_hashes: list[str] = []
    visible_by_tile: dict[tuple[int, int], np.ndarray] = {}

    def merge(patch: object) -> None:
        if not isinstance(patch, HopperTrajectoryObservationPatch):
            raise FormalCacheError("hopper closure trajectory patch is invalid")
        if patch.scene_id != scene_id:
            raise FormalCacheError("hopper closure trajectory scene differs")
        if patch.evidence_generation != evidence_generation:
            raise FormalCacheError(
                "hopper closure trajectory evidence generation differs"
            )
        trajectory_hashes.append(patch.trajectory_sha256)
        for tile in patch.tiles:
            identity = tile.tile_row, tile.tile_column
            existing = visible_by_tile.get(identity)
            if existing is None:
                visible_by_tile[identity] = np.ascontiguousarray(
                    tile.visible_mask.copy(), dtype=np.bool_
                )
                continue
            if existing.shape != tile.visible_mask.shape:
                raise FormalCacheError(
                    "hopper closure trajectory tile geometry differs"
                )
            existing |= tile.visible_mask

    merge(first)
    for patch in iterator:
        merge(patch)

    digest = sha256()
    digest.update(b"lunar-formal-cache-hopper-round-trajectory-union/v1\0")
    for trajectory_sha256 in sorted(trajectory_hashes):
        digest.update(bytes.fromhex(trajectory_sha256))
    tiles = tuple(
        HopperTrajectoryTilePatch(
            tile_row=row,
            tile_column=column,
            visible_mask=visible_by_tile[(row, column)],
        )
        for row, column in sorted(visible_by_tile)
    )
    return HopperTrajectoryObservationPatch(
        scene_id=scene_id,
        evidence_generation=evidence_generation,
        trajectory_sha256=digest.hexdigest(),
        tiles=tiles,
        visible_cells=sum(
            int(np.count_nonzero(tile.visible_mask)) for tile in tiles
        ),
    )


def _build_hopper_truth_coverability(
    *,
    platform: FrozenPlatformCapability,
    scene: MultiResolutionScene,
    projected: object,
    mission_roi: np.ndarray,
    start_cell: tuple[int, int],
    exact_start_position_m: tuple[float, float, float],
    bridge: object,
) -> _HopperTruthCoverability:
    """Build the truth-only safe-hop/trajectory-observation fixed point."""
    import lunar_planner_training_bridge as bridge_api

    from ..environment.multires_observation import (
        HopperTrajectoryPoint,
        MultiresSensorObservationState,
    )
    from ..environment.observation_builder import Pose2
    from ..environment.platform_reachability import (
        HopperSingleHopEnvelope,
        PlatformCandidateReachability,
    )
    from .raster import LOCAL_GEOMETRY

    coarse_shape = tuple(int(value) for value in mission_roi.shape)
    empty_physical = np.zeros(coarse_shape, dtype=np.bool_)
    target_only = _build_platform_detail_coverability(
        platform=platform,
        scene=scene,
        mission_roi=mission_roi,
        physical_observation_pose_mask=empty_physical,
        physical_observation_positions_m=np.empty((0, 3), dtype=np.float64),
        bridge=bridge,
    )
    provider = SceneTileProvider(scene, capacity=8)
    sensor_state = MultiresSensorObservationState(
        scene=scene,
        tile_provider=provider,
        mission_roi_ratio=np.ascontiguousarray(
            mission_roi.astype(np.float32), dtype=np.float32
        ),
        mission_priority=np.zeros(coarse_shape, dtype=np.float32),
        coverable_detail_shape=target_only.detail_shape,
        coverable_detail_bits=target_only.mission_target_detail_bits,
        coverable_detail_cell_count=(
            target_only.mission_target_detail_cell_count
        ),
        coverable_mask_sha256=target_only.mission_target_mask_sha256,
    )

    def canonical_position(values: object) -> tuple[float, float, float]:
        array = np.ascontiguousarray(
            np.asarray(values, dtype=np.float64).reshape((1, 3))
        )
        return tuple(
            float(value) / 1_000_000.0
            for value in canonical_physical_positions_um(array)[0]
        )

    start_position = canonical_position(exact_start_position_m)
    start_pose = Pose2(
        start_position[0], start_position[1], 0.0, "map", start_position[2]
    )
    initial_patch = sensor_state.prepare_hopper_trajectory_observation(
        (HopperTrajectoryPoint(start_pose, 0.0),)
    )
    sensor_state.commit_hopper_trajectory_observation(
        initial_patch, elapsed_s=0.0
    )
    initial_observed_count = sensor_state.observed_coverable_detail_cell_count

    reached_positions: dict[tuple[int, int], tuple[float, float, float]] = {
        start_cell: start_position
    }
    processed_edges: dict[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[float, float, float],
    ] = {}
    safe_union = np.zeros(coarse_shape, dtype=np.bool_)
    evidence_algorithm_id: str | None = None
    round_index = 0
    while True:
        observed = sensor_state.observed
        global_map = _bridge_grid_map(
            bridge_api,
            canvas=observed.canvas,
            elevation_m=observed.elevation_m,
            valid_mask=observed.valid_mask,
            obstacle_ratio=observed.physical_obstacle_ratio,
            obstacle_height_m=sensor_state.coarse_obstacle_height_m,
            forbidden_ratio=sensor_state.coarse_forbidden_ratio,
            frame_id="map",
        )
        observed_safe = np.ascontiguousarray(
            observed.valid_mask
            & (observed.physical_obstacle_ratio == 0.0)
            & (sensor_state.coarse_forbidden_ratio == 0.0)
            & mission_roi,
            dtype=np.bool_,
        )
        safe_cells = np.ascontiguousarray(
            np.column_stack(np.nonzero(observed_safe)), dtype=np.int32
        ).reshape((-1, 2))
        proposed: dict[
            tuple[tuple[int, int], tuple[int, int]],
            tuple[tuple[float, float, float], float],
        ] = {}
        for source_cell in sorted(reached_positions):
            source_position = reached_positions[source_cell]
            source_pose = Pose2(
                source_position[0],
                source_position[1],
                0.0,
                "map",
                source_position[2],
            )
            detail = sensor_state.planning_observation(source_pose)
            local_map = _bridge_grid_map(
                bridge_api,
                canvas=detail.canvas,
                elevation_m=detail.elevation_m,
                valid_mask=detail.valid_mask,
                obstacle_ratio=detail.physical_obstacle_ratio,
                obstacle_height_m=detail.physical_obstacle_height_m,
                forbidden_ratio=detail.forbidden_ratio,
                frame_id="odom",
            )
            request = _projection_request(
                platform,
                scene,
                projected,
                start_cell=source_cell,
                exact_start_position_m=source_position,
            )
            generation = round_index + 1
            request.request_id = (
                f"formal-cache/{scene.scene_id}/hopper-closure/"
                f"{generation}/{source_cell[0]}-{source_cell[1]}"
            )
            request.global_map_generation = generation
            request.local_map_generation = generation
            request.state_time.nanoseconds_since_epoch = (
                1_000_000_000 + generation * 1_000_000
            )
            request.world.global_map = global_map
            request.world.local_map = local_map
            request.world.map_from_odom.stamp.nanoseconds_since_epoch = (
                request.state_time.nanoseconds_since_epoch
            )
            request.config.global_map.base_resolution_m = (
                LOCAL_GEOMETRY.resolution_m
            )
            request.config.global_map.maximum_level = 5
            request.config.global_map.target_axis_cells = 256
            physical = PlatformCandidateReachability(
                platform_type="HOPPER",
                canvas=observed.canvas,
                pose_map=source_pose,
                observed_elevation_m=observed.elevation_m,
                bridge=bridge,
                request=request,
                maximum_edge_distance_m=None,
            ).project_physical(safe_cells)
            envelope = physical.hopper_single_hop_envelope
            if not isinstance(envelope, HopperSingleHopEnvelope):
                raise FormalCacheError("hopper closure envelope is missing")
            safe_union |= envelope.certified_mask
            if evidence_algorithm_id is None:
                evidence_algorithm_id = physical.physical_evidence_algorithm_id
            elif evidence_algorithm_id != physical.physical_evidence_algorithm_id:
                raise FormalCacheError("hopper closure evidence algorithm drifted")
            certified_rows, certified_columns = np.nonzero(
                envelope.certified_mask
            )
            position_by_cell = {
                (int(row), int(column)): canonical_position(position)
                for row, column, position in zip(
                    certified_rows,
                    certified_columns,
                    envelope.certified_positions_m,
                    strict=True,
                )
            }
            if position_by_cell.get(source_cell) != source_position:
                raise FormalCacheError(
                    "hopper closure exact start or landing position drifted"
                )
            for target_row, target_column in np.argwhere(
                envelope.eligible_mask
            ):
                target_cell = int(target_row), int(target_column)
                edge = source_cell, target_cell
                target_position = position_by_cell.get(target_cell)
                if target_position is None:
                    raise FormalCacheError(
                        "hopper closure target lacks landing evidence"
                    )
                old_position = processed_edges.get(edge)
                if old_position is not None:
                    if old_position != target_position:
                        raise FormalCacheError("hopper closure landing drifted")
                    continue
                proposed[edge] = (
                    target_position,
                    float(envelope.nominal_flight_time_s[target_cell]),
                )
        if not proposed:
            break
        before_count = sensor_state.observed_coverable_detail_cell_count
        reached_changed = False

        def round_patches() -> Iterable[object]:
            nonlocal reached_changed
            for edge in sorted(proposed):
                source_cell, target_cell = edge
                target_position, flight_time_s = proposed[edge]
                processed_edges[edge] = target_position
                old_target = reached_positions.get(target_cell)
                if old_target is not None and old_target != target_position:
                    raise FormalCacheError(
                        "hopper closure target position is ambiguous"
                    )
                reached_changed = reached_changed or old_target is None
                reached_positions[target_cell] = target_position
                source_position = reached_positions[source_cell]
                points = (
                    HopperTrajectoryPoint(
                        Pose2(
                            *source_position[:2],
                            0.0,
                            "map",
                            source_position[2],
                        ),
                        0.0,
                    ),
                    HopperTrajectoryPoint(
                        Pose2(
                            *target_position[:2],
                            0.0,
                            "map",
                            target_position[2],
                        ),
                        flight_time_s,
                    ),
                )
                yield sensor_state.prepare_hopper_trajectory_observation(points)

        round_patch = _merge_hopper_trajectory_patches(round_patches())
        sensor_state.commit_hopper_trajectory_observation(
            round_patch, elapsed_s=0.0
        )
        changed = (
            reached_changed
            or sensor_state.observed_coverable_detail_cell_count > before_count
        )
        if not changed:
            break
        round_index += 1

    if evidence_algorithm_id is None:
        raise FormalCacheError("hopper closure produced no landing evidence")
    physical_mask = np.zeros(coarse_shape, dtype=np.bool_)
    for cell in reached_positions:
        physical_mask[cell] = True
    positions = np.ascontiguousarray(
        [reached_positions[tuple(cell)] for cell in np.argwhere(physical_mask)],
        dtype=np.float64,
    ).reshape((-1, 3))
    detail = freeze_streamed_detail_coverability(
        detail_shape=target_only.detail_shape,
        coarse_shape=coarse_shape,
        mission_target_detail_bits=target_only.mission_target_detail_bits,
        coverable_detail_bits=sensor_state.observed_coverable_detail_bits(),
    )
    initial_fraction = (
        float(initial_observed_count / detail.coverable_detail_cell_count)
        if detail.coverable_detail_cell_count
        else 0.0
    )
    return _HopperTruthCoverability(
        physical_observation_pose_mask=np.ascontiguousarray(physical_mask),
        observation_positions_m=positions,
        physical_safe_pose_count=int(safe_union.sum(dtype=np.int64)),
        physical_reachability_algorithm_id=(
            "truth-hopper-safe-hop-observation-closure/v1"
        ),
        physical_evidence_algorithm_id=evidence_algorithm_id,
        detail=detail,
        initial_coverable_fraction=initial_fraction,
    )


def _finite_hopper_landing_targets(
    projected: object,
    *,
    row0: int,
    row1: int,
    column0: int,
    column1: int,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Return only valid finite landing cells in stable row-major order."""
    cells = [
        (row, column)
        for row in range(row0, row1)
        for column in range(column0, column1)
        if bool(projected.valid_mask[row, column])
    ]
    targets = np.asarray(
        [
            (
                *projected.canvas.grid_center_world(row, column),
                float(projected.elevation_m[row, column]),
            )
            for row, column in cells
        ],
        dtype=np.float64,
    ).reshape((-1, 3))
    if not np.isfinite(targets).all():
        raise FormalCacheError("valid Hopper landing target is non-finite")
    return cells, np.ascontiguousarray(targets)


@dataclass(frozen=True, slots=True)
class _HopperPhysicalEvidence:
    bridge_grid: object
    certified_pose_mask: np.ndarray
    aim_positions_m: np.ndarray
    evidence_algorithm_id: str


@dataclass(frozen=True, slots=True)
class _TruthPhysicalReachability:
    """Offline truth projection, separate from runtime single-hop authority."""

    platform_type: str
    physical_observation_pose_mask: np.ndarray
    observation_positions_m: np.ndarray
    physical_projection_schema: str
    physical_reachability_algorithm_id: str
    physical_evidence_algorithm_id: str
    physical_safe_pose_count: int
    physically_reachable_pose_count: int
    minimum_global_cost: np.ndarray | None = None
    global_parent_index: np.ndarray | None = None
    global_start_index: int | None = None
    global_search_elapsed_s: float | None = None


def _hopper_landing_evidence(
    *,
    platform: FrozenPlatformCapability,
    scene: MultiResolutionScene,
    projected: object,
    start_cell: tuple[int, int],
    bridge: object,
    exact_start_position_m: tuple[float, float, float],
) -> _HopperPhysicalEvidence:
    """Stream exact 0.2 m landing regions into one coarse evidence grid."""
    import lunar_planner_training_bridge as bridge_api

    provider = SceneTileProvider(scene, capacity=1)
    coarse_cells = projected.canvas.geometry.cells
    certified = np.zeros((coarse_cells, coarse_cells), dtype=np.bool_)
    aim = np.zeros((coarse_cells, coarse_cells, 3), dtype=np.float64)
    boundary = np.zeros((coarse_cells, coarse_cells, 4, 3), dtype=np.float64)
    area = np.zeros((coarse_cells, coarse_cells), dtype=np.float64)
    coarse_per_tile = int(
        round(
            provider.tile_geometry.size_m
            / projected.canvas.geometry.resolution_m
        )
    )
    typed = platform.typed_capability
    support_radius = float(getattr(typed, "landing_support_radius_m"))
    lateral_margin = float(getattr(typed, "landing_lateral_margin_m"))
    halo_cells = math.ceil(
        (support_radius + lateral_margin)
        / provider.tile_geometry.resolution_m
    ) + 2
    algorithm_id: str | None = None
    for tile_row, tile_column in provider.iter_tile_indices():
        window = provider.tile_with_halo(
            tile_row, tile_column, halo_cells=halo_cells
        )
        row0 = tile_row * coarse_per_tile
        column0 = tile_column * coarse_per_tile
        row1 = min(row0 + coarse_per_tile, coarse_cells)
        column1 = min(column0 + coarse_per_tile, coarse_cells)
        cells, targets = _finite_hopper_landing_targets(
            projected,
            row0=row0,
            row1=row1,
            column0=column0,
            column1=column1,
        )
        if not cells:
            continue
        result = bridge.project_hopper_landing_evidence(
            _projection_request(
                platform,
                scene,
                projected,
                start_cell=start_cell,
                local_projected=window.projected,
                exact_start_position_m=exact_start_position_m,
            ),
            targets,
        )
        if algorithm_id is None:
            algorithm_id = str(result.algorithm_id)
        elif algorithm_id != result.algorithm_id:
            raise FormalCacheError("hopper landing evidence algorithm drifted")
        if result.certified.shape != (len(cells),):
            raise FormalCacheError("hopper landing evidence geometry differs")
        for index, (row, column) in enumerate(cells):
            certified[row, column] = result.certified[index]
            aim[row, column] = result.aim_positions_m[index]
            boundary[row, column] = result.boundary_m[index]
            area[row, column] = result.area_m2[index]
    if algorithm_id is None:
        raise FormalCacheError("hopper landing evidence is empty")
    bridge_grid = bridge_api.HopperLandingEvidenceGrid(
        np.ascontiguousarray(np.flipud(certified)),
        np.ascontiguousarray(np.flipud(aim)),
        np.ascontiguousarray(np.flipud(boundary)),
        np.ascontiguousarray(np.flipud(area)),
        algorithm_id,
    )
    return _HopperPhysicalEvidence(
        bridge_grid=bridge_grid,
        certified_pose_mask=np.ascontiguousarray(certified, dtype=np.bool_),
        aim_positions_m=np.ascontiguousarray(aim, dtype=np.float64),
        evidence_algorithm_id=algorithm_id,
    )


def _build_truth_physical_reachability(
    *,
    platform: FrozenPlatformCapability,
    scene: MultiResolutionScene,
    projected: object,
    start_cell: tuple[int, int],
    bridge: object,
    exact_start_position_m: tuple[float, float, float] | None = None,
) -> _TruthPhysicalReachability:
    request = _projection_request(
        platform,
        scene,
        projected,
        start_cell=start_cell,
        exact_start_position_m=exact_start_position_m,
    )
    if platform.platform_type == "HOPPER":
        if exact_start_position_m is None:
            raise FormalCacheError("truth Hopper exact start position is missing")
        landing_evidence = _hopper_landing_evidence(
            platform=platform,
            scene=scene,
            projected=projected,
            start_cell=start_cell,
            bridge=bridge,
            exact_start_position_m=exact_start_position_m,
        )
        output = bridge.project_reachability(
            request, 30.0, landing_evidence.bridge_grid
        )
        safe = landing_evidence.certified_pose_mask
        evidence_algorithm_id = landing_evidence.evidence_algorithm_id
        physical = np.ascontiguousarray(
            np.flipud(output.reachable).astype(np.bool_)
        )
        reachability_algorithm_id = str(output.algorithm_id)
        minimum_global_cost = None
        global_parent_index = None
        global_start_index = None
        global_search_elapsed_s = None
    else:
        output = bridge.project_reachability(request, 30.0)
        traversability = bridge.project_traversability(request)
        safe = np.ascontiguousarray(
            np.flipud(traversability.hard_feasible).astype(np.bool_)
        )
        evidence_algorithm_id = "cpp-safe-traversability-projection/v1"
        try:
            (
                physical,
                reachability_algorithm_id,
                minimum_global_cost,
                global_parent_index,
                global_start_index,
                global_search_elapsed_s,
            ) = _validate_ground_reachability_projection(
                output,
                platform_type=platform.platform_type,
                cells=256,
            )
        except PlatformReachabilityError as error:
            raise FormalCacheError(
                "truth ground global cost tree is invalid"
            ) from error
    if output.platform_type != platform.platform_type:
        raise FormalCacheError("physical projection platform identity differs")
    if physical.shape != (256, 256) or safe.shape != physical.shape:
        raise FormalCacheError("physical pose projection geometry differs")
    if np.logical_and(physical, np.logical_not(safe)).any():
        raise FormalCacheError("physical reachability exceeds safe poses")
    if platform.platform_type == "HOPPER":
        positions = np.ascontiguousarray(
            landing_evidence.aim_positions_m[physical], dtype=np.float64
        )
        if not bool(physical[start_cell]):
            raise FormalCacheError("truth Hopper exact start is unreachable")
        rows, columns = np.nonzero(physical)
        matching = np.flatnonzero(
            (rows == start_cell[0]) & (columns == start_cell[1])
        )
        if len(matching) != 1:
            raise FormalCacheError("truth Hopper exact start is ambiguous")
        truth_start = np.ascontiguousarray(
            positions[int(matching[0])].reshape((1, 3)), dtype=np.float64
        )
        truth_start_canonical = _positions_m_from_canonical_um(
            canonical_physical_positions_um(truth_start)
        )[0]
        if tuple(float(value) for value in truth_start_canonical) != (
            exact_start_position_m
        ):
            raise FormalCacheError("truth Hopper exact start position drifted")
    else:
        positions = np.asarray(
            [
                (
                    *projected.canvas.grid_center_world(
                        int(row), int(column)
                    ),
                    float(projected.elevation_m[row, column]),
                )
                for row, column in np.argwhere(physical)
            ],
            dtype=np.float64,
        ).reshape((-1, 3))
        positions = np.ascontiguousarray(positions, dtype=np.float64)
    return _TruthPhysicalReachability(
        platform_type=platform.platform_type,
        physical_observation_pose_mask=physical,
        observation_positions_m=positions,
        physical_projection_schema=PHYSICAL_PROJECTION_SCHEMA,
        physical_reachability_algorithm_id=reachability_algorithm_id,
        physical_evidence_algorithm_id=evidence_algorithm_id,
        physical_safe_pose_count=int(safe.sum(dtype=np.int64)),
        physically_reachable_pose_count=int(physical.sum(dtype=np.int64)),
        minimum_global_cost=minimum_global_cost,
        global_parent_index=global_parent_index,
        global_start_index=global_start_index,
        global_search_elapsed_s=global_search_elapsed_s,
    )


def _initial_coverable_fraction(
    *,
    platform: FrozenPlatformCapability,
    scene: MultiResolutionScene,
    start_cell: tuple[int, int],
    detail: StreamedDetailCoverability,
    exact_start_position_m: tuple[float, float, float] | None = None,
) -> float:
    from ..environment.visibility import NativeVisibilityEstimator, SensorGeometry

    if detail.coverable_detail_cell_count == 0:
        return 0.0
    provider = SceneTileProvider(scene, capacity=1)
    x_m, y_m = (
        scene.base_canvas.grid_center_world(*start_cell)
        if exact_start_position_m is None
        else exact_start_position_m[:2]
    )
    pose_row, pose_column = provider.world_to_detail(x_m, y_m)
    cells = provider.tile_geometry.cells
    start_row = pose_row - cells // 2
    start_column = pose_column - cells // 2
    truth_obstacle_ratio = provider.read_visibility_obstacle_window(
        start_row, start_column, cells=cells
    )
    visible = NativeVisibilityEstimator(
        SensorGeometry(
            platform.observation_capability.sensor_range_m,
            platform.observation_capability.sensor_fov_rad,
        ),
        resolution_m=provider.tile_geometry.resolution_m,
    ).reveal_from_pose(
        truth_obstacle_ratio,
        (pose_row - start_row, pose_column - start_column),
    )
    coverable = read_packed_detail_window(
        detail.coverable_detail_bits,
        detail.detail_shape,
        start_row=start_row,
        start_column=start_column,
        cells=cells,
    )
    observed = int((visible & coverable).sum(dtype=np.int64))
    return float(observed / detail.coverable_detail_cell_count)


def _physical_start_identity_from_position(
    *,
    platform_type: str,
    start_cell: tuple[int, int] | None,
    position_m: tuple[float, float, float] | None,
) -> str:
    if (start_cell is None) is not (position_m is None):
        raise FormalCacheError("physical start identity inputs differ")
    payload: dict[str, object] = {
        "schema": "lunar-physical-start-identity/v1",
        "platform_type": platform_type,
        "qualified_start_cell": (
            None if start_cell is None else list(start_cell)
        ),
    }
    if position_m is not None:
        values = np.asarray(position_m, dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise FormalCacheError("physical start position is invalid")
        payload["position_m"] = [float(value) for value in values]
    return _semantic_sha(payload)


def _physical_start_identity_sha256(
    *,
    platform_type: str,
    projected: object,
    start_cell: tuple[int, int],
    exact_start_position_m: tuple[float, float, float] | None = None,
) -> str:
    row, column = start_cell
    x_m, y_m = projected.canvas.grid_center_world(row, column)
    position_m = (
        [
            float(x_m),
            float(y_m),
            float(projected.elevation_m[row, column]),
        ]
        if exact_start_position_m is None
        else [float(value) for value in exact_start_position_m]
    )
    return _physical_start_identity_from_position(
        platform_type=platform_type,
        start_cell=start_cell,
        position_m=tuple(position_m),
    )


@dataclass(frozen=True, slots=True)
class _PlatformCoverabilityProduct:
    coverability: PlatformCoverability
    observation_positions_m: np.ndarray
    physical_evidence_algorithm_id: str

    def __post_init__(self) -> None:
        positions = self.observation_positions_m
        if (
            not isinstance(self.coverability, PlatformCoverability)
            or not isinstance(positions, np.ndarray)
            or positions.dtype != np.dtype(np.float64)
            or positions.ndim != 2
            or positions.shape[1:] != (3,)
            or len(positions)
            != self.coverability.physically_reachable_pose_count
            or not positions.flags.c_contiguous
            or not np.isfinite(positions).all()
            or not isinstance(self.physical_evidence_algorithm_id, str)
            or not self.physical_evidence_algorithm_id
        ):
            raise FormalCacheError("physical coverability authority is invalid")


def _unsafe_platform_coverability(
    platform: FrozenPlatformCapability,
    projected: object,
    detail_shape: tuple[int, int],
) -> _PlatformCoverabilityProduct:
    physical = np.zeros((256, 256), dtype=np.bool_)
    detail = np.zeros(detail_shape, dtype=np.bool_)
    capability_sha256 = _physical_capability_content_sha256(platform)
    start_sha256 = _physical_start_identity_from_position(
        platform_type=platform.platform_type,
        start_cell=None,
        position_m=None,
    )
    algorithm_id = "not-run/unsafe-start"
    canvas = projected.canvas
    bounds = tuple(float(value) for value in canvas.bounds_m)
    coverability = PlatformCoverability(
        platform_type=platform.platform_type,
        qualified_start_cell=None,
        physical_observation_pose_mask=physical,
        physical_projection_schema=PHYSICAL_PROJECTION_SCHEMA,
        physical_reachability_algorithm_id=algorithm_id,
        physical_safe_pose_count=0,
        physically_reachable_pose_count=0,
        physical_projection_sha256=physical_projection_sha256(
            platform_type=platform.platform_type,
            physical_reachability_algorithm_id=algorithm_id,
            physical_evidence_algorithm_id=algorithm_id,
            physical_observation_pose_mask=physical,
            physical_observation_positions_m=np.empty((0, 3), dtype=np.float64),
            physical_grid_resolution_m=float(canvas.geometry.resolution_m),
            physical_grid_origin_m=(bounds[0], bounds[3]),
            physical_grid_world_bounds_m=bounds,
            physical_grid_axis_convention=PHYSICAL_GRID_AXIS_CONVENTION,
            capability_content_sha256=capability_sha256,
            start_identity_sha256=start_sha256,
        ),
        mission_target_detail_mask_sha256=mask_sha256(detail),
        coverable_detail_shape=detail_shape,
        coverable_detail_bits=pack_detail_mask(detail),
        coverable_ratio=np.zeros((256, 256), dtype=np.float32),
        mission_target_detail_cell_count=0,
        coverable_detail_cell_count=0,
        mission_coverable_fraction=0.0,
        initial_coverable_fraction=0.0,
        initial_candidate_count=0,
        coverable_detail_mask_sha256=mask_sha256(detail),
        sensor_visibility_algorithm_id="not-run/unsafe-start",
        capability_content_sha256=capability_sha256,
        start_identity_sha256=start_sha256,
        exact=True,
        eligible=False,
        ineligible_reason=IneligibleReason.UNSAFE_START,
    )
    return _PlatformCoverabilityProduct(
        coverability=coverability,
        observation_positions_m=np.empty((0, 3), dtype=np.float64),
        physical_evidence_algorithm_id=algorithm_id,
    )


def _build_scene_platform_coverability(
    *,
    platform_type: str,
    capability_bundle: FrozenCapabilityBundle,
    qualification: object,
    scene: MultiResolutionScene,
    projected: object,
    mission_roi: np.ndarray,
    detail_shape: tuple[int, int],
) -> _PlatformCoverabilityProduct:
    platform = capability_bundle.for_platform(platform_type)
    if qualification is None:
        return _unsafe_platform_coverability(platform, projected, detail_shape)
    exact_start_position_m = getattr(
        qualification, "exact_start_position_m", None
    )
    if platform_type == "HOPPER" and exact_start_position_m is None:
        raise FormalCacheError("qualified Hopper exact start position is missing")
    if platform_type != "HOPPER" and exact_start_position_m is not None:
        raise FormalCacheError("ground qualification carries a Hopper exact start")

    import lunar_planner_training_bridge as bridge_api

    bridge = bridge_api.PlannerBridge()
    try:
        if platform_type == "HOPPER":
            assert exact_start_position_m is not None
            hopper_closure = _build_hopper_truth_coverability(
                platform=platform,
                scene=scene,
                projected=projected,
                mission_roi=mission_roi,
                start_cell=qualification.cell,
                exact_start_position_m=exact_start_position_m,
                bridge=bridge,
            )
            physical = hopper_closure.physical_observation_pose_mask
            observation_positions_m = hopper_closure.observation_positions_m
            physical_projection_schema = PHYSICAL_PROJECTION_SCHEMA
            physical_reachability_algorithm_id = (
                hopper_closure.physical_reachability_algorithm_id
            )
            physical_evidence_algorithm_id = (
                hopper_closure.physical_evidence_algorithm_id
            )
            physical_safe_pose_count = hopper_closure.physical_safe_pose_count
            physically_reachable_pose_count = int(
                physical.sum(dtype=np.int64)
            )
            detail = hopper_closure.detail
            initial_fraction = hopper_closure.initial_coverable_fraction
        else:
            physical_projection = _build_truth_physical_reachability(
                platform=platform,
                scene=scene,
                projected=projected,
                start_cell=qualification.cell,
                bridge=bridge,
                exact_start_position_m=None,
            )
            physical = physical_projection.physical_observation_pose_mask
            observation_positions_m = physical_projection.observation_positions_m
            physical_projection_schema = (
                physical_projection.physical_projection_schema
            )
            physical_reachability_algorithm_id = (
                physical_projection.physical_reachability_algorithm_id
            )
            physical_evidence_algorithm_id = (
                physical_projection.physical_evidence_algorithm_id
            )
            physical_safe_pose_count = (
                physical_projection.physical_safe_pose_count
            )
            physically_reachable_pose_count = (
                physical_projection.physically_reachable_pose_count
            )
            detail = _build_platform_detail_coverability(
                platform=platform,
                scene=scene,
                mission_roi=mission_roi,
                physical_observation_pose_mask=physical,
                physical_observation_positions_m=observation_positions_m,
                bridge=bridge,
            )
            initial_fraction = _initial_coverable_fraction(
                platform=platform,
                scene=scene,
                start_cell=qualification.cell,
                detail=detail,
                exact_start_position_m=None,
            )
    except RuntimeError as error:
        if native_start_failure_is_ineligible(platform_type, error):
            return _unsafe_platform_coverability(platform, projected, detail_shape)
        raise
    if not bool(physical[qualification.cell]):
        return _unsafe_platform_coverability(platform, projected, detail_shape)
    reason = classify_ineligibility(
        qualified_start_cell=qualification.cell,
        mission_target_detail_cell_count=(
            detail.mission_target_detail_cell_count
        ),
        coverable_detail_cell_count=detail.coverable_detail_cell_count,
        initial_coverable_fraction=initial_fraction,
        initial_candidate_count=qualification.initial_candidate_count,
    )
    start_sha256 = _physical_start_identity_sha256(
        platform_type=platform_type,
        projected=projected,
        start_cell=qualification.cell,
        exact_start_position_m=exact_start_position_m,
    )
    capability_sha256 = _physical_capability_content_sha256(platform)
    canvas = projected.canvas
    bounds = tuple(float(value) for value in canvas.bounds_m)
    coverability = PlatformCoverability(
        platform_type=platform_type,
        qualified_start_cell=qualification.cell,
        physical_observation_pose_mask=physical,
        physical_projection_schema=physical_projection_schema,
        physical_reachability_algorithm_id=physical_reachability_algorithm_id,
        physical_safe_pose_count=physical_safe_pose_count,
        physically_reachable_pose_count=physically_reachable_pose_count,
        physical_projection_sha256=physical_projection_sha256(
            platform_type=platform_type,
            physical_reachability_algorithm_id=(
                physical_reachability_algorithm_id
            ),
            physical_evidence_algorithm_id=(
                physical_evidence_algorithm_id
            ),
            physical_observation_pose_mask=physical,
            physical_observation_positions_m=(
                observation_positions_m
            ),
            physical_grid_resolution_m=float(canvas.geometry.resolution_m),
            physical_grid_origin_m=(bounds[0], bounds[3]),
            physical_grid_world_bounds_m=bounds,
            physical_grid_axis_convention=PHYSICAL_GRID_AXIS_CONVENTION,
            capability_content_sha256=capability_sha256,
            start_identity_sha256=start_sha256,
        ),
        mission_target_detail_mask_sha256=detail.mission_target_mask_sha256,
        coverable_detail_shape=detail.detail_shape,
        coverable_detail_bits=detail.coverable_detail_bits,
        coverable_ratio=detail.coverable_ratio,
        mission_target_detail_cell_count=(
            detail.mission_target_detail_cell_count
        ),
        coverable_detail_cell_count=detail.coverable_detail_cell_count,
        mission_coverable_fraction=(
            detail.coverable_detail_cell_count
            / detail.mission_target_detail_cell_count
            if detail.mission_target_detail_cell_count
            else 0.0
        ),
        initial_coverable_fraction=initial_fraction,
        initial_candidate_count=qualification.initial_candidate_count,
        coverable_detail_mask_sha256=detail.coverable_mask_sha256,
        sensor_visibility_algorithm_id=(
            "hopper-unobstructed-trajectory-capsule-closure/v1"
            if platform_type == "HOPPER"
            else "two-dimensional-detail-los/v1"
        ),
        capability_content_sha256=capability_sha256,
        start_identity_sha256=start_sha256,
        exact=True,
        eligible=reason is None,
        ineligible_reason=reason,
    )
    return _PlatformCoverabilityProduct(
        coverability=coverability,
        observation_positions_m=observation_positions_m,
        physical_evidence_algorithm_id=physical_evidence_algorithm_id,
    )


def _static_scene_data(
    scenario: Mapping[str, object],
    *,
    source_paths: Mapping[str, Path],
    capability_bundle: FrozenCapabilityBundle,
    bridge: object,
    base_cache: dict[str, tuple[MapCanvas, np.ndarray, np.ndarray]],
) -> StaticSceneData:
    canvas, base_elevation, base_valid = _load_base_scene(
        scenario, source_paths, base_cache
    )
    vector = _vector_scene(scenario, canvas)
    multires = MultiResolutionScene(
        canvas,
        base_elevation,
        base_valid,
        vector,
        scenario_id=_require_sha(scenario.get("scene_id"), "scene_id"),
    )
    projected = multires.project(canvas)
    hard: dict[str, np.ndarray] = {}
    clearance: dict[str, np.ndarray] = {}
    for platform_type in _PLATFORMS:
        platform = capability_bundle.for_platform(platform_type)
        output = bridge.project_traversability(
            _projection_request(platform, multires, projected)
        )
        known = np.ascontiguousarray(np.flipud(output.known).astype(bool))
        feasible = np.ascontiguousarray(
            np.flipud(output.hard_feasible).astype(bool)
        )
        clearance_m = np.ascontiguousarray(
            np.flipud(output.clearance_m).astype(np.float32)
        )
        hard[platform_type] = (known & feasible).astype(np.uint8)
        sensor_range = platform.observation_capability.sensor_range_m
        clearance[platform_type] = np.where(
            known,
            np.clip(
                np.nan_to_num(
                    clearance_m,
                    nan=0.0,
                    posinf=sensor_range,
                    neginf=0.0,
                ),
                0.0,
                sensor_range,
            )
            / sensor_range,
            0.0,
        ).astype(np.float32)
    qualification_arrays = {
        "elevation_m": projected.elevation_m,
        "valid_mask": projected.valid_mask,
        "forbidden_ratio": projected.forbidden_ratio,
        **{
            f"{platform.lower()}_hard_feasible": hard[platform]
            for platform in _PLATFORMS
        },
        **{
            f"{platform.lower()}_clearance_margin_norm": clearance[platform]
            for platform in _PLATFORMS
        },
    }
    qualifications = {
        platform: qualify_initial_start(
            scene=multires,
            arrays=qualification_arrays,
            capability=capability_bundle.for_platform(platform),
        )
        for platform in _PLATFORMS
    }
    mission_roi = build_formal_mission_roi(qualification_arrays)
    detail_shape = (
        SceneTileProvider(multires, capacity=1).detail_cells_per_axis,
    ) * 2
    coverability_products = _parallel_platform_map(
        lambda platform_type: _build_scene_platform_coverability(
            platform_type=platform_type,
            capability_bundle=capability_bundle,
            qualification=qualifications[platform_type],
            scene=multires,
            projected=projected,
            mission_roi=mission_roi,
            detail_shape=detail_shape,
        )
    )
    coverability = {
        platform: coverability_products[platform].coverability
        for platform in _PLATFORMS
    }
    physical_evidence_algorithms = {
        platform: coverability_products[platform].physical_evidence_algorithm_id
        for platform in _PLATFORMS
    }
    hopper_positions_um = canonical_physical_positions_um(
        coverability_products["HOPPER"].observation_positions_m
    )
    rocks = np.asarray(
        [
            (item.x_m, item.y_m, item.radius_m, item.height_m)
            for item in vector.rocks
        ],
        dtype=np.float64,
    ).reshape((-1, 4))
    craters = np.asarray(
        [
            (item.x_m, item.y_m, item.radius_m, item.depth_m)
            for item in vector.craters
        ],
        dtype=np.float64,
    ).reshape((-1, 4))
    no_go = np.asarray(
        [item.vertices_m for item in vector.no_go_polygons], dtype=np.float64
    ).reshape((-1, 6, 2))
    return StaticSceneData(
        scene_id=multires.scene_id,
        source=str(scenario["source"]),
        split=str(scenario["split"]),
        window_id=str(scenario["window_id"]),
        window_sha256=(
            str(scenario["window_sha256"])
            if scenario.get("window_sha256") is not None
            else canvas.window_sha256
        ),
        world_bounds_m=canvas.bounds_m,
        elevation_m=projected.elevation_m,
        valid_mask=projected.valid_mask,
        physical_obstacle_ratio=projected.physical_obstacle_ratio,
        physical_obstacle_height_m=projected.physical_obstacle_height_m,
        forbidden_ratio=projected.forbidden_ratio,
        rocks=rocks,
        craters=craters,
        no_go_vertices=no_go,
        hard_feasible=hard,
        clearance_margin_norm=clearance,
        coverability=coverability,
        physical_evidence_algorithm_ids=physical_evidence_algorithms,
        hopper_physical_observation_positions_um=hopper_positions_um,
    )


@dataclass(frozen=True)
class _StaticSceneWork:
    scenario: dict[str, object]
    source_paths: dict[str, Path]
    capability_bundle: FrozenCapabilityBundle


def _static_scene_process_payload(scene: StaticSceneData) -> dict[str, object]:
    """Convert immutable scene mappings to process-safe materialization data."""
    payload = dict(vars(scene))
    payload["hard_feasible"] = dict(scene.hard_feasible)
    payload["clearance_margin_norm"] = dict(scene.clearance_margin_norm)
    payload["coverability"] = dict(scene.coverability)
    payload["physical_evidence_algorithm_ids"] = dict(
        scene.physical_evidence_algorithm_ids
    )
    return payload


def _static_scene_process_worker(
    work: _StaticSceneWork,
) -> dict[str, object]:
    """Build one scene in an isolated process and return a picklable payload."""
    import lunar_planner_training_bridge as bridge_api

    scene = _static_scene_data(
        work.scenario,
        source_paths=work.source_paths,
        capability_bundle=work.capability_bundle,
        bridge=bridge_api.PlannerBridge(),
        base_cache={},
    )
    return _static_scene_process_payload(scene)


def _scene_index_record_data(
    scenario: Mapping[str, object],
    *,
    source_paths: Mapping[str, Path],
    capability_bundle: FrozenCapabilityBundle,
    bridge: object,
    base_cache: dict[str, tuple[MapCanvas, np.ndarray, np.ndarray]],
) -> FormalSceneIndexRecord:
    """Build only locked source identity and minimum static start evidence."""
    canvas, base_elevation, base_valid = _load_base_scene(
        scenario, source_paths, base_cache
    )
    vector = _vector_scene(scenario, canvas)
    multires = MultiResolutionScene(
        canvas,
        base_elevation,
        base_valid,
        vector,
        scenario_id=_require_sha(scenario.get("scene_id"), "scene_id"),
    )
    projected = multires.project(canvas)
    hard: dict[str, np.ndarray] = {}
    clearance: dict[str, np.ndarray] = {}
    for platform_type in _PLATFORMS:
        platform = capability_bundle.for_platform(platform_type)
        output = bridge.project_traversability(
            _projection_request(platform, multires, projected)
        )
        known = np.ascontiguousarray(np.flipud(output.known).astype(bool))
        feasible = np.ascontiguousarray(
            np.flipud(output.hard_feasible).astype(bool)
        )
        clearance_m = np.ascontiguousarray(
            np.flipud(output.clearance_m).astype(np.float32)
        )
        hard[platform_type] = np.ascontiguousarray(known & feasible)
        sensor_range = platform.observation_capability.sensor_range_m
        clearance[platform_type] = np.where(
            known,
            np.clip(
                np.nan_to_num(
                    clearance_m,
                    nan=0.0,
                    posinf=sensor_range,
                    neginf=0.0,
                ),
                0.0,
                sensor_range,
            )
            / sensor_range,
            0.0,
        ).astype(np.float32)
    qualification_arrays = {
        "elevation_m": projected.elevation_m,
        "valid_mask": projected.valid_mask,
        "forbidden_ratio": projected.forbidden_ratio,
        **{
            f"{platform.lower()}_hard_feasible": hard[platform]
            for platform in _PLATFORMS
        },
        **{
            f"{platform.lower()}_clearance_margin_norm": clearance[platform]
            for platform in _PLATFORMS
        },
    }
    mission_roi = build_formal_mission_roi(qualification_arrays)
    platform_starts: dict[str, dict[str, object]] = {}
    for platform_type in _PLATFORMS:
        safe_cells = tuple(
            cell
            for cell in formal_safe_start_cells(
                qualification_arrays,
                platform_type,
            )
            if bool(mission_roi[cell])
        )
        cell = safe_cells[0] if safe_cells else None
        position_m: tuple[float, float, float] | None = None
        if cell is not None:
            x_m, y_m = canvas.grid_center_world(*cell)
            position_m = (
                float(x_m),
                float(y_m),
                float(projected.elevation_m[cell]),
            )
        platform = capability_bundle.for_platform(platform_type)
        platform_starts[platform_type] = {
            "qualified_start_cell": cell,
            "start_identity_sha256": _physical_start_identity_from_position(
                platform_type=platform_type,
                start_cell=cell,
                position_m=position_m,
            ),
            "capability_content_sha256": (
                _physical_capability_content_sha256(platform)
            ),
            "qualification_algorithm_id": (
                _MINIMAL_START_QUALIFICATION_ALGORITHM_ID
            ),
            "safe_start_cell_count": len(safe_cells),
            "qualified": cell is not None,
            "ineligible_reason": (
                None if cell is not None else "NO_MINIMAL_SAFE_START"
            ),
        }
    window_sha256 = scenario.get("window_sha256")
    if window_sha256 is None:
        window_sha256 = canvas.window_sha256
    return FormalSceneIndexRecord(
        scene_id=multires.scene_id,
        source=str(scenario["source"]),
        split=str(scenario["split"]),
        scene_seed=_require_sha(scenario.get("scene_seed"), "scene_seed"),
        window_id=str(scenario["window_id"]),
        window_sha256=_require_sha(window_sha256, "window_sha256"),
        world_bounds_m=canvas.bounds_m,
        platform_starts=platform_starts,
    )


@dataclass(frozen=True)
class _SceneIndexWork:
    scenario: dict[str, object]
    source_paths: dict[str, Path]
    capability_bundle: FrozenCapabilityBundle


def _scene_index_process_worker(
    work: _SceneIndexWork,
) -> dict[str, object]:
    """Build one lightweight record in an isolated process."""
    import lunar_planner_training_bridge as bridge_api

    record = _scene_index_record_data(
        work.scenario,
        source_paths=work.source_paths,
        capability_bundle=work.capability_bundle,
        bridge=bridge_api.PlannerBridge(),
        base_cache={},
    )
    return record.to_dict()


def prepare_formal_training_cache(
    *,
    source_lock_path: str | Path,
    split_manifest_path: str | Path,
    cache_root: str | Path,
    materialization: str,
    preflight_scenario_limit: int | None,
    capability_bundle: FrozenCapabilityBundle,
    repository_root: str | Path,
) -> dict[str, object]:
    """Verify locked inputs and publish a lightweight formal scene index."""
    if materialization not in {"preflight", "bounded", "full"}:
        raise FormalCacheError(
            "materialization must be preflight, bounded, or full"
        )
    root = Path(repository_root).resolve(strict=True)
    verified = _verified_preparation_inputs(
        source_lock_path=Path(source_lock_path),
        split_manifest_path=Path(split_manifest_path),
        capability_bundle=capability_bundle,
        repository_root=root,
    )
    selected = _selected_scenarios(
        verified.scenario_manifest,
        materialization,
        preflight_scenario_limit,
    )

    def records() -> Iterable[FormalSceneIndexRecord]:
        work_items = (
            _SceneIndexWork(
                scenario=dict(scenario),
                source_paths={
                    source: Path(path)
                    for source, path in verified.source_paths.items()
                },
                capability_bundle=capability_bundle,
            )
            for scenario in selected
        )

        worker_count = _scene_process_worker_count(
            cpu_count=os.cpu_count() or 1,
            scene_count=len(selected),
        )
        payloads = _ordered_bounded_process_map(
            work_items,
            _scene_index_process_worker,
            max_workers=worker_count,
        )
        for payload in payloads:
            yield FormalSceneIndexRecord(**payload)

    manifest = write_formal_cache(
        cache_root,
        identity=verified.identity,
        scenario_manifest=verified.scenario_manifest,
        materialization=materialization,
        scenes=records(),
        repository_root=root,
        source_lock_path=source_lock_path,
    )
    if materialization == "full" and manifest.get("scene_count") != 1734:
        raise FormalCacheError("full cache must contain exactly 1734 scenes")
    if (
        materialization == "bounded"
        and manifest.get("scene_count") != BOUNDED_FORMAL_SCENE_COUNT
    ):
        raise FormalCacheError("bounded cache must contain exactly 128 scenes")
    if materialization in {"bounded", "full"} and not manifest.get(
        "formal_eligible"
    ):
        raise FormalCacheError(
            "formal scene index common-start subset is below split qualification"
        )
    return manifest


__all__ = [
    "BOUNDED_FORMAL_SCENE_COUNT",
    "BOUNDED_FORMAL_SPLIT_COUNTS",
    "FORMAL_CACHE_SCHEMA",
    "FormalCache",
    "FormalCacheError",
    "FormalCacheIdentity",
    "FormalSceneIndex",
    "FormalSceneIndexRecord",
    "StaticSceneData",
    "TaskSourceCrop",
    "build_formal_cache_identity",
    "current_v3_identity",
    "load_formal_cache",
    "load_task_source_crop",
    "prepare_formal_training_cache",
    "reward_v4_runtime_identity_evidence",
    "write_formal_cache",
]
