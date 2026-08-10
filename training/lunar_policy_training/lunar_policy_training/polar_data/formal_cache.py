"""Content-addressed external cache for formal polar training scenes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
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
    qualify_initial_start,
)
from ..environment.coverability import (
    IneligibleReason,
    PlatformCoverability,
    QualifiedStartState,
    StreamedDetailCoverability,
    build_streamed_detail_coverability,
    classify_ineligibility,
    mask_sha256,
    pack_detail_mask,
    read_packed_detail_window,
    unpack_detail_mask,
)
from ..reward import reward_weights_sha256
from ..training_semantics import training_semantics_sha256
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
from .raster import MapCanvas, load_polar_window
from .scenario_manifest import (
    build_scenario_manifest_document,
    load_scenario_manifest,
)
from .source_lock import load_aggregate_source_lock


FORMAL_CACHE_SCHEMA = "lunar-formal-training-cache/v5"
_SOURCE_IDS = (
    "NASA_LOLA_87S_DEM",
    "NASA_LOLA_87S_COUNT",
    "JAXA_LUPEX_DATA_S1",
)
_PLATFORMS = ("WHEELED", "LEGGED", "HOPPER")
_REQUIRED_SPLITS = ("train", "validation", "test", "holdout")
_PLATFORM_COVERABILITY_FIELDS = frozenset(
    (
        "qualified_start_cell",
        "qualified_start_state",
        "reachable_pose_shape",
        "coverable_detail_shape",
        "mission_target_detail_cell_count",
        "coverable_detail_cell_count",
        "mission_coverable_fraction",
        "initial_coverable_fraction",
        "initial_candidate_count",
        "primitive_state_count",
        "certified_edge_count",
        "recoverable_state_count",
        "reachability_algorithm_id",
        "primitive_state_schema",
        "primitive_set_sha256",
        "world_evidence_sha256",
        "reachability_graph_sha256",
        "visibility_algorithm_id",
        "reachable_mask_sha256",
        "coverable_mask_sha256",
        "exact",
        "eligible",
        "ineligible_reason",
        "stage_diagnostics",
    )
)
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


def _formal_platform_eligibility_ready(
    materialization: str,
    platform_counts: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> bool:
    """Require a non-empty exact eligible lane for every platform and split."""
    if materialization != "full":
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


def _grid(name: str, value: np.ndarray, dtype: object) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if array.shape != (256, 256):
        raise FormalCacheError(f"{name} must have shape [256,256]")
    return array


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
        if len(self.world_bounds_m) != 4:
            raise FormalCacheError("scene world bounds are invalid")
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
        checked_coverability: dict[str, PlatformCoverability] = {}
        for platform in _PLATFORMS:
            payload = self.coverability[platform]
            if (
                not isinstance(payload, PlatformCoverability)
                or payload.platform_type != platform
                or payload.reachable_pose_mask.shape != (256, 256)
            ):
                raise FormalCacheError("scene platform coverability is invalid")
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
            output[f"{prefix}_reachable_pose_bits"] = pack_detail_mask(
                coverability.reachable_pose_mask
            )
            output[f"{prefix}_coverable_detail_bits"] = np.ascontiguousarray(
                coverability.coverable_detail_bits.copy(), dtype=np.uint8
            )
            output[f"{prefix}_coverable_ratio"] = np.ascontiguousarray(
                coverability.coverable_ratio.copy(), dtype=np.float32
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
            "qualified_start_state": (
                None
                if payload.qualified_start_state is None
                else payload.qualified_start_state.to_dict()
            ),
            "reachable_pose_shape": list(payload.reachable_pose_mask.shape),
            "coverable_detail_shape": list(payload.coverable_detail_shape),
            "mission_target_detail_cell_count": (
                payload.mission_target_detail_cell_count
            ),
            "coverable_detail_cell_count": payload.coverable_detail_cell_count,
            "mission_coverable_fraction": payload.mission_coverable_fraction,
            "initial_coverable_fraction": payload.initial_coverable_fraction,
            "initial_candidate_count": payload.initial_candidate_count,
            "primitive_state_count": payload.primitive_state_count,
            "certified_edge_count": payload.certified_edge_count,
            "recoverable_state_count": payload.recoverable_state_count,
            "reachability_algorithm_id": payload.reachability_algorithm_id,
            "primitive_state_schema": payload.primitive_state_schema,
            "primitive_set_sha256": payload.primitive_set_sha256,
            "world_evidence_sha256": payload.world_evidence_sha256,
            "reachability_graph_sha256": payload.reachability_graph_sha256,
            "visibility_algorithm_id": payload.visibility_algorithm_id,
            "reachable_mask_sha256": payload.reachable_mask_sha256,
            "coverable_mask_sha256": payload.coverable_mask_sha256,
            "exact": payload.exact,
            "eligible": payload.eligible,
            "ineligible_reason": (
                None
                if payload.ineligible_reason is None
                else payload.ineligible_reason.value
            ),
            "stage_diagnostics": {
                "reachable_pose_count": int(
                    payload.reachable_pose_mask.sum(dtype=np.int64)
                ),
                "mission_target_detail_cell_count": (
                    payload.mission_target_detail_cell_count
                ),
                "coverable_detail_cell_count": payload.coverable_detail_cell_count,
                "initial_candidate_count": payload.initial_candidate_count,
                "primitive_state_count": payload.primitive_state_count,
                "certified_edge_count": payload.certified_edge_count,
                "recoverable_state_count": payload.recoverable_state_count,
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
                payload = entry["platform_coverability"][platform]
                if bool(payload["eligible"]):
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
    scenes: Iterable[StaticSceneData],
    repository_root: str | Path,
) -> dict[str, object]:
    """Atomically publish a cache manifest last; an existing cache is read-only."""
    if materialization not in {"preflight", "full"}:
        raise FormalCacheError("materialization must be preflight or full")
    root = _validate_cache_root(Path(cache_root), Path(repository_root))
    manifest_path = root / "cache-manifest.json"
    if manifest_path.is_file():
        existing = load_formal_cache(
            manifest_path,
            expected_identity=identity,
            require_full=materialization == "full",
        )
        if existing.manifest["materialization"] != materialization:
            raise FormalCacheError("existing cache materialization differs")
        return existing.manifest
    if root.exists() and any(root.iterdir()):
        raise FormalCacheError("incomplete cache root is not reusable")
    root.mkdir(mode=0o755, exist_ok=True)
    scene_root = root / "scenes"
    scene_root.mkdir(mode=0o755)
    scenario_payload = _canonical_bytes(dict(scenario_manifest))
    scenario_path = root / "scenario-manifest.json"
    _atomic_bytes(scenario_path, scenario_payload)
    if scenario_manifest.get("scenario_manifest_sha256") != identity.scenario_manifest_sha256:
        raise FormalCacheError("scenario manifest identity differs from cache identity")

    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for scene in scenes:
        if scene.scene_id in seen:
            raise FormalCacheError("cache scene IDs must be unique")
        seen.add(scene.scene_id)
        entries.append(
            _scene_entry(scene, scene_root / f"{scene.scene_id}.npz", root)
        )
    entries.sort(key=lambda value: str(value["scene_id"]))
    platform_eligibility, exact_common, readiness = _eligibility_summary(entries)
    qualification_complete = _formal_platform_eligibility_ready(
        materialization, readiness
    )
    inventory = [
        {
            "relative_path": "scenario-manifest.json",
            "size_bytes": scenario_path.stat().st_size,
            "sha256": _file_sha(scenario_path),
        },
        *[
            {
                "relative_path": entry["relative_path"],
                "size_bytes": entry["size_bytes"],
                "sha256": entry["sha256"],
            }
            for entry in entries
        ],
    ]
    body: dict[str, object] = {
        "schema": FORMAL_CACHE_SCHEMA,
        "materialization": materialization,
        "formal_eligible": materialization == "full" and qualification_complete,
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


@dataclass(frozen=True)
class FormalCache:
    root: Path
    manifest: dict[str, object]
    identity: FormalCacheIdentity

    @property
    def formal_eligible(self) -> bool:
        return bool(self.manifest["formal_eligible"])

    def load_scene(self, scene_id: str) -> dict[str, np.ndarray]:
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
            raise FormalCacheError("scene is not present in cache")
        path = self.root / entry["relative_path"]
        with np.load(path, allow_pickle=False) as archive:
            expected = entry["arrays"]
            if set(archive.files) != set(expected):
                raise FormalCacheError("scene array inventory differs")
            output = {
                name: np.ascontiguousarray(archive[name]) for name in archive.files
            }
        for name, values in output.items():
            if _array_metadata(values) != expected[name]:
                raise FormalCacheError(f"scene array {name} hash or metadata differs")
            values.setflags(write=False)
        for platform in _PLATFORMS:
            prefix = platform.lower()
            payload = entry["platform_coverability"][platform]
            reachable = unpack_detail_mask(
                output[f"{prefix}_reachable_pose_bits"].copy(), (256, 256)
            )
            detail_shape = tuple(payload["coverable_detail_shape"])
            coverable = unpack_detail_mask(
                output[f"{prefix}_coverable_detail_bits"].copy(), detail_shape
            )
            if (
                mask_sha256(reachable) != payload["reachable_mask_sha256"]
                or mask_sha256(coverable) != payload["coverable_mask_sha256"]
                or int(reachable.sum(dtype=np.int64))
                != payload["stage_diagnostics"]["reachable_pose_count"]
                or int(coverable.sum(dtype=np.int64))
                != payload["coverable_detail_cell_count"]
            ):
                raise FormalCacheError("scene semantic coverability mask differs")
        return output


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
            actual.add(path.relative_to(root).as_posix())
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


def _validate_scene_coverability(entry: Mapping[str, object]) -> None:
    platform_payloads = entry.get("platform_coverability")
    arrays = entry.get("arrays")
    if (
        not isinstance(platform_payloads, Mapping)
        or set(platform_payloads) != set(_PLATFORMS)
        or not isinstance(arrays, Mapping)
    ):
        raise FormalCacheError("cache scene platform coverability is invalid")
    for platform in _PLATFORMS:
        payload = platform_payloads[platform]
        if (
            not isinstance(payload, Mapping)
            or set(payload) != _PLATFORM_COVERABILITY_FIELDS
        ):
            raise FormalCacheError("cache platform coverability fields are invalid")
        reachable_shape = _shape(
            payload.get("reachable_pose_shape"), "reachable pose shape"
        )
        detail_shape = _shape(
            payload.get("coverable_detail_shape"), "coverable detail shape"
        )
        if reachable_shape != (256, 256) or (
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
        start_state_value = payload.get("qualified_start_state")
        if start_state_value is None:
            parsed_start_state = None
        else:
            try:
                parsed_start_state = QualifiedStartState.from_dict(
                    start_state_value
                )
            except ValueError as error:
                raise FormalCacheError(
                    "cache qualified start state is invalid"
                ) from error
        if (parsed_cell is None) is not (parsed_start_state is None):
            raise FormalCacheError(
                "cache qualified start state disagrees with cell"
            )
        target_count = payload.get("mission_target_detail_cell_count")
        coverable_count = payload.get("coverable_detail_cell_count")
        initial_candidate_count = payload.get("initial_candidate_count")
        primitive_state_count = payload.get("primitive_state_count")
        certified_edge_count = payload.get("certified_edge_count")
        recoverable_state_count = payload.get("recoverable_state_count")
        initial_fraction = payload.get("initial_coverable_fraction")
        mission_fraction = payload.get("mission_coverable_fraction")
        if (
            type(target_count) is not int
            or type(coverable_count) is not int
            or type(initial_candidate_count) is not int
            or type(primitive_state_count) is not int
            or type(certified_edge_count) is not int
            or type(recoverable_state_count) is not int
            or primitive_state_count < 0
            or certified_edge_count < 0
            or recoverable_state_count < 0
            or recoverable_state_count > primitive_state_count
            or not isinstance(initial_fraction, float)
            or not isinstance(mission_fraction, float)
        ):
            raise FormalCacheError("cache coverability counts are invalid")
        if parsed_start_state is not None and (
            primitive_state_count == 0 or recoverable_state_count == 0
        ):
            raise FormalCacheError(
                "cache qualified start has no recoverable primitive graph"
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
            "reachability_algorithm_id",
            "primitive_state_schema",
            "visibility_algorithm_id",
        ):
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise FormalCacheError("cache coverability algorithm is missing")
        for name in (
            "primitive_set_sha256",
            "world_evidence_sha256",
            "reachability_graph_sha256",
            "reachable_mask_sha256",
            "coverable_mask_sha256",
        ):
            _require_sha(payload.get(name), name)
        diagnostics = payload.get("stage_diagnostics")
        if not isinstance(diagnostics, Mapping) or diagnostics != {
            "reachable_pose_count": diagnostics.get("reachable_pose_count"),
            "mission_target_detail_cell_count": target_count,
            "coverable_detail_cell_count": coverable_count,
            "initial_candidate_count": initial_candidate_count,
            "primitive_state_count": primitive_state_count,
            "certified_edge_count": certified_edge_count,
            "recoverable_state_count": recoverable_state_count,
        }:
            raise FormalCacheError("cache coverability diagnostics are invalid")
        if (
            type(diagnostics["reachable_pose_count"]) is not int
            or diagnostics["reachable_pose_count"] < 0
        ):
            raise FormalCacheError("cache reachable pose count is invalid")
        prefix = platform.lower()
        expected_shapes = {
            f"{prefix}_reachable_pose_bits": [(256 * 256 + 7) // 8],
            f"{prefix}_coverable_detail_bits": [
                (detail_shape[0] * detail_shape[1] + 7) // 8
            ],
            f"{prefix}_coverable_ratio": [256, 256],
        }
        for name, shape in expected_shapes.items():
            metadata = arrays.get(name)
            if not isinstance(metadata, Mapping) or metadata.get("shape") != shape:
                raise FormalCacheError("cache coverability array metadata differs")


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
    claimed = _require_sha(
        value.get("cache_manifest_sha256"), "cache_manifest_sha256"
    )
    body = dict(value)
    body.pop("cache_manifest_sha256")
    if _semantic_sha(body) != claimed:
        raise FormalCacheError("cache manifest identity mismatch")
    materialization = value.get("materialization")
    if materialization not in {"preflight", "full"}:
        raise FormalCacheError("cache materialization is invalid")
    formal_eligible = value.get("formal_eligible")
    if type(formal_eligible) is not bool or (materialization != "full" and formal_eligible):
        raise FormalCacheError("cache formal eligibility is invalid")
    if require_full and (materialization != "full" or not formal_eligible):
        raise FormalCacheError(
            "formal command requires a full, start-qualified cache"
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
        _validate_scene_coverability(entry)
    platform_eligibility, exact_common, readiness = _eligibility_summary(scenes)
    if (
        value.get("platform_eligibility") != platform_eligibility
        or value.get("exact_common_evaluation") != exact_common
    ):
        raise FormalCacheError("cache platform eligibility summary differs")
    expected_formal_eligible = _formal_platform_eligibility_ready(
        materialization, readiness
    )
    if formal_eligible is not expected_formal_eligible:
        raise FormalCacheError("cache formal platform eligibility is invalid")
    _validate_inventory(path.parent, value)
    return FormalCache(path.parent, value, identity)


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


def _global_composed_capability(
    platform: FrozenPlatformCapability,
    *,
    resolution_m: float,
    bridge_api: object,
) -> object:
    """Compose short ground primitives into one certified global-grid stride."""
    capability = platform.to_bridge_capability()
    typed = platform.typed_capability
    if platform.platform_type == "WHEELED":
        primitives = capability.motion_primitives
        for index, frozen in enumerate(typed.motion_primitives):
            if frozen.kind not in {"FORWARD", "REVERSE"}:
                continue
            displacement = frozen.relative_end_pose.position_m
            length = math.hypot(displacement.x, displacement.y)
            if length <= 0.0 or length >= resolution_m - 1.0e-12:
                continue
            repeats = resolution_m / length
            rounded = round(repeats)
            if rounded < 1 or not math.isclose(
                repeats, rounded, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise FormalCacheError(
                    "global wheel stride is not an exact primitive composition"
                )
            primitive = primitives[index]
            primitive.primitive_id = (
                f"{frozen.primitive_id}/composed-{rounded}x"
            )
            pose = _bridge_pose(
                bridge_api,
                displacement.x * rounded,
                displacement.y * rounded,
                displacement.z * rounded,
            )
            orientation = frozen.relative_end_pose.orientation
            pose.orientation.w = orientation.w
            pose.orientation.x = orientation.x
            pose.orientation.y = orientation.y
            pose.orientation.z = orientation.z
            primitive.relative_end_pose = pose
            primitives[index] = primitive
        capability.motion_primitives = primitives
    elif platform.platform_type == "LEGGED":
        primitives = capability.motion_primitives
        for index, frozen in enumerate(typed.motion_primitives):
            displacement = frozen.body_frame_displacement_m
            length = math.hypot(displacement.x, displacement.y)
            if length <= 0.0 or length >= resolution_m - 1.0e-12:
                continue
            repeats = resolution_m / length
            rounded = round(repeats)
            if rounded < 1 or not math.isclose(
                repeats, rounded, rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise FormalCacheError(
                    "global legged stride is not an exact primitive composition"
                )
            primitive = primitives[index]
            primitive.primitive_id = (
                f"{frozen.primitive_id}/composed-{rounded}x"
            )
            primitive.body_frame_displacement_m = _bridge_vec3(
                bridge_api,
                displacement.x * rounded,
                displacement.y * rounded,
                displacement.z * rounded,
            )
            primitives[index] = primitive
        capability.motion_primitives = primitives
    return capability


def _projection_request(
    platform: FrozenPlatformCapability,
    scene: MultiResolutionScene,
    projected: object,
    *,
    start_cell: tuple[int, int] | None = None,
    local_projected: object | None = None,
    compose_global_primitives: bool = False,
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
    request.capability = (
        _global_composed_capability(
            platform,
            resolution_m=projection_canvas.geometry.resolution_m,
            bridge_api=bridge_api,
        )
        if compose_global_primitives
        else platform.to_bridge_capability()
    )
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
    reachable_pose_mask: np.ndarray,
    recoverable_observation_positions_m: np.ndarray,
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
            reachable_pose_mask, dtype=np.bool_
        ),
        observation_positions_m=np.ascontiguousarray(
            recoverable_observation_positions_m, dtype=np.float64
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


def _hopper_landing_evidence(
    *,
    platform: FrozenPlatformCapability,
    scene: MultiResolutionScene,
    projected: object,
    start_cell: tuple[int, int],
    bridge: object,
) -> object:
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
    return bridge_api.HopperLandingEvidenceGrid(
        np.ascontiguousarray(np.flipud(certified)),
        np.ascontiguousarray(np.flipud(aim)),
        np.ascontiguousarray(np.flipud(boundary)),
        np.ascontiguousarray(np.flipud(area)),
        algorithm_id,
    )


def _build_truth_primitive_reachability(
    *,
    platform: FrozenPlatformCapability,
    scene: MultiResolutionScene,
    projected: object,
    start_cell: tuple[int, int],
    bridge: object,
) -> tuple[object, np.ndarray, np.ndarray, QualifiedStartState]:
    import lunar_planner_training_bridge as bridge_api

    request = _projection_request(
        platform,
        scene,
        projected,
        start_cell=start_cell,
        compose_global_primitives=True,
    )
    output = bridge_api.PrimitiveReachabilityEngine().update(request, None)
    if output.platform_type != platform.platform_type:
        raise FormalCacheError("primitive graph platform identity differs")
    reachable = np.ascontiguousarray(
        np.flipud(output.reachable).astype(np.bool_)
    )
    if reachable.shape != (256, 256):
        raise FormalCacheError("reachable pose projection geometry differs")
    recoverable = np.asarray(output.recoverable, dtype=np.bool_)
    observation = np.asarray(output.observation_state, dtype=np.bool_)
    if recoverable.shape != observation.shape or recoverable.ndim != 1:
        raise FormalCacheError("primitive graph state flags differ")
    positions = np.asarray(output.positions_m, dtype=np.float64)
    if positions.shape != (recoverable.size, 3):
        raise FormalCacheError("primitive graph state positions differ")
    recoverable_positions = np.ascontiguousarray(
        positions[recoverable & observation], dtype=np.float64
    )
    anchor_indices = np.flatnonzero(
        np.asarray(output.path_cost, dtype=np.float64) == 0.0
    )
    if anchor_indices.size != 1:
        raise FormalCacheError("primitive graph has no unique qualified anchor")
    anchor = int(anchor_indices[0])
    if not bool(recoverable[anchor]):
        raise FormalCacheError("primitive graph anchor is not recoverable")
    start_state = QualifiedStartState(
        position_m=tuple(float(value) for value in positions[anchor]),
        yaw_rad=float(output.yaw_rad[anchor]),
        motion_mode=int(output.motion_mode[anchor]),
        body_z_m=tuple(float(value) for value in output.body_z_m[anchor]),
    )
    return output, reachable, recoverable_positions, start_state


def _initial_coverable_fraction(
    *,
    platform: FrozenPlatformCapability,
    scene: MultiResolutionScene,
    start_cell: tuple[int, int],
    detail: StreamedDetailCoverability,
) -> float:
    from ..environment.visibility import NativeVisibilityEstimator, SensorGeometry

    if detail.coverable_detail_cell_count == 0:
        return 0.0
    provider = SceneTileProvider(scene, capacity=1)
    x_m, y_m = scene.base_canvas.grid_center_world(*start_cell)
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


def _unsafe_platform_coverability(
    platform_type: str,
    detail_shape: tuple[int, int],
) -> PlatformCoverability:
    reachable = np.zeros((256, 256), dtype=np.bool_)
    detail = np.zeros(detail_shape, dtype=np.bool_)
    return PlatformCoverability(
        platform_type=platform_type,
        qualified_start_cell=None,
        qualified_start_state=None,
        reachable_pose_mask=reachable,
        coverable_detail_shape=detail_shape,
        coverable_detail_bits=pack_detail_mask(detail),
        coverable_ratio=np.zeros((256, 256), dtype=np.float32),
        mission_target_detail_cell_count=0,
        coverable_detail_cell_count=0,
        mission_coverable_fraction=0.0,
        initial_coverable_fraction=0.0,
        initial_candidate_count=0,
        primitive_state_count=0,
        certified_edge_count=0,
        recoverable_state_count=0,
        reachability_algorithm_id="not-run/unsafe-start",
        primitive_state_schema="not-run/unsafe-start",
        primitive_set_sha256=_semantic_sha(
            ["unsafe-start", platform_type, "primitive-set"]
        ),
        world_evidence_sha256=_semantic_sha(
            ["unsafe-start", platform_type, "world-evidence"]
        ),
        reachability_graph_sha256=_semantic_sha(
            ["unsafe-start", platform_type, "primitive-graph"]
        ),
        visibility_algorithm_id="not-run/unsafe-start",
        reachable_mask_sha256=mask_sha256(reachable),
        coverable_mask_sha256=mask_sha256(detail),
        exact=True,
        eligible=False,
        ineligible_reason=IneligibleReason.UNSAFE_START,
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
) -> PlatformCoverability:
    if qualification is None:
        return _unsafe_platform_coverability(platform_type, detail_shape)

    import lunar_planner_training_bridge as bridge_api

    platform = capability_bundle.for_platform(platform_type)
    bridge = bridge_api.PlannerBridge()
    (
        primitive_graph,
        reachable,
        recoverable_observation_positions,
        qualified_start_state,
    ) = _build_truth_primitive_reachability(
        platform=platform,
        scene=scene,
        projected=projected,
        start_cell=qualification.cell,
        bridge=bridge,
    )
    if not bool(reachable[qualification.cell]):
        zero = _unsafe_platform_coverability(platform_type, detail_shape)
        return PlatformCoverability(
            platform_type=platform_type,
            qualified_start_cell=None,
            qualified_start_state=None,
            reachable_pose_mask=reachable,
            coverable_detail_shape=zero.coverable_detail_shape,
            coverable_detail_bits=zero.coverable_detail_bits,
            coverable_ratio=zero.coverable_ratio,
            mission_target_detail_cell_count=0,
            coverable_detail_cell_count=0,
            mission_coverable_fraction=0.0,
            initial_coverable_fraction=0.0,
            initial_candidate_count=0,
            primitive_state_count=len(primitive_graph.state_ids),
            certified_edge_count=len(primitive_graph.edge_source_ids),
            recoverable_state_count=int(
                primitive_graph.recoverable.sum(dtype=np.int64)
            ),
            reachability_algorithm_id=primitive_graph.algorithm_id,
            primitive_state_schema=primitive_graph.state_schema,
            primitive_set_sha256=primitive_graph.primitive_set_sha256,
            world_evidence_sha256=primitive_graph.world_evidence_sha256,
            reachability_graph_sha256=primitive_graph.graph_sha256,
            visibility_algorithm_id="not-run/unsafe-start",
            reachable_mask_sha256=mask_sha256(reachable),
            coverable_mask_sha256=zero.coverable_mask_sha256,
            exact=True,
            eligible=False,
            ineligible_reason=IneligibleReason.UNSAFE_START,
        )
    detail = _build_platform_detail_coverability(
        platform=platform,
        scene=scene,
        mission_roi=mission_roi,
        reachable_pose_mask=reachable,
        recoverable_observation_positions_m=(
            recoverable_observation_positions
        ),
        bridge=bridge,
    )
    initial_fraction = _initial_coverable_fraction(
        platform=platform,
        scene=scene,
        start_cell=qualification.cell,
        detail=detail,
    )
    reason = classify_ineligibility(
        qualified_start_cell=qualification.cell,
        mission_target_detail_cell_count=(
            detail.mission_target_detail_cell_count
        ),
        coverable_detail_cell_count=detail.coverable_detail_cell_count,
        initial_coverable_fraction=initial_fraction,
        initial_candidate_count=qualification.initial_candidate_count,
    )
    return PlatformCoverability(
        platform_type=platform_type,
        qualified_start_cell=qualification.cell,
        qualified_start_state=qualified_start_state,
        reachable_pose_mask=reachable,
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
        primitive_state_count=len(primitive_graph.state_ids),
        certified_edge_count=len(primitive_graph.edge_source_ids),
        recoverable_state_count=int(
            primitive_graph.recoverable.sum(dtype=np.int64)
        ),
        reachability_algorithm_id=primitive_graph.algorithm_id,
        primitive_state_schema=primitive_graph.state_schema,
        primitive_set_sha256=primitive_graph.primitive_set_sha256,
        world_evidence_sha256=primitive_graph.world_evidence_sha256,
        reachability_graph_sha256=primitive_graph.graph_sha256,
        visibility_algorithm_id="two-dimensional-detail-los/v1",
        reachable_mask_sha256=mask_sha256(reachable),
        coverable_mask_sha256=detail.coverable_mask_sha256,
        exact=True,
        eligible=reason is None,
        ineligible_reason=reason,
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
    coverability = _parallel_platform_map(
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
    )


@dataclass(frozen=True)
class _StaticSceneWork:
    scenario: dict[str, object]
    source_paths: dict[str, Path]
    capability_bundle: FrozenCapabilityBundle


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
    payload = dict(vars(scene))
    payload["hard_feasible"] = dict(scene.hard_feasible)
    payload["clearance_margin_norm"] = dict(scene.clearance_margin_norm)
    payload["coverability"] = dict(scene.coverability)
    return payload


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
    """Verify locked inputs and materialize the current formal static world cache."""
    if materialization not in {"preflight", "full"}:
        raise FormalCacheError("materialization must be preflight or full")
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

    def records() -> Iterable[StaticSceneData]:
        work_items = (
            _StaticSceneWork(
                scenario=dict(scenario),
                source_paths={
                    source: Path(path)
                    for source, path in verified.source_paths.items()
                },
                capability_bundle=capability_bundle,
            )
            for scenario in selected
        )

        cpu_count = os.cpu_count() or 1
        worker_count = min(
            len(selected),
            max(1, min(8, cpu_count // len(_PLATFORMS))),
        )
        payloads = _ordered_bounded_process_map(
            work_items,
            _static_scene_process_worker,
            max_workers=worker_count,
        )
        for payload in payloads:
            yield StaticSceneData(**payload)

    manifest = write_formal_cache(
        cache_root,
        identity=verified.identity,
        scenario_manifest=verified.scenario_manifest,
        materialization=materialization,
        scenes=records(),
        repository_root=root,
    )
    if materialization == "full" and manifest.get("scene_count") != 1734:
        raise FormalCacheError("full cache must contain exactly 1734 scenes")
    if materialization == "full" and not manifest.get("formal_eligible"):
        raise FormalCacheError(
            "full cache common-start subset is below the frozen split qualification"
        )
    return manifest


__all__ = [
    "FORMAL_CACHE_SCHEMA",
    "FormalCache",
    "FormalCacheError",
    "FormalCacheIdentity",
    "StaticSceneData",
    "build_formal_cache_identity",
    "current_v3_identity",
    "load_formal_cache",
    "prepare_formal_training_cache",
    "write_formal_cache",
]
