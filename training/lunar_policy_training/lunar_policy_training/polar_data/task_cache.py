"""Content-addressed, task-local formal training artifacts.

The scene index is deliberately small.  Derived terrain, reachability and
coverability arrays live here under identities that include the frozen task
geometry and every platform algorithm authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

import numpy as np

if TYPE_CHECKING:
    from .hopper_task_closure import HopperClosureJournal

from ..environment.task_area import DETAIL_PER_GLOBAL, FrozenTaskGeometry
from ..reward_contract import TaskScaleBucket


TASK_COVERABILITY_SCHEMA = "lunar-formal-task-coverability/v1"
TASK_COMMON_SCHEMA = "lunar-formal-task-common/v1"
_JOURNAL_SCHEMA = "lunar-formal-task-build-journal/v1"
_PLATFORMS = frozenset(("WHEELED", "LEGGED", "HOPPER"))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_ARRAY_FIELDS = frozenset(
    (
        "relative_path",
        "encoding",
        "shape",
        "local_origin_index",
        "dtype",
        "logical_dtype",
        "storage_dtype",
        "size_bytes",
        "sha256",
    )
)


class TaskCacheError(ValueError):
    """A task key or artifact is malformed, corrupt, or unsafe."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TaskCacheError("task cache JSON is not canonical") from error


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TaskCacheError(f"{name} must be a lowercase SHA-256")
    return value


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TaskCacheError(f"{name} is invalid")
    return value


def _bounds(value: object, name: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise TaskCacheError(f"{name} is invalid")
    row0, row1, column0, column1 = (int(item) for item in value)
    if min(row0, column0) < 0 or row1 <= row0 or column1 <= column0:
        raise TaskCacheError(f"{name} is not a positive half-open box")
    return row0, row1, column0, column1


def _world_bounds(value: object, name: str) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise TaskCacheError(f"{name} is invalid")
    left, bottom, right, top = (float(item) for item in value)
    if right <= left or top <= bottom:
        raise TaskCacheError(f"{name} is not a positive half-open box")
    return left, bottom, right, top


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TaskCacheError("task cache JSON mapping key is invalid")
            output[key] = _json_value(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TaskCacheError("task cache JSON contains a non-finite value")
        return value
    raise TaskCacheError("task cache JSON value is unsupported")


@dataclass(frozen=True, slots=True)
class TaskCommonKey:
    scene_id: str
    scenario_identity_sha256: str
    source_identity_sha256: str
    coarse_bounds_half_open: tuple[int, int, int, int]
    detail_bounds_half_open: tuple[int, int, int, int]
    task_span_cells: int
    scale_bucket: str
    geometry_sha256: str
    halo_contract_sha256: str
    capability_bundle_sha256: str
    generator_sha256: str
    source_commit: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.scene_id, "task common scene identity"),
            (self.scenario_identity_sha256, "task common scenario identity"),
            (self.source_identity_sha256, "task common source identity"),
            (self.geometry_sha256, "task common geometry identity"),
            (self.halo_contract_sha256, "task common halo identity"),
            (self.capability_bundle_sha256, "task common capability identity"),
            (self.generator_sha256, "task common generator identity"),
        ):
            _require_sha256(value, name)
        coarse = _bounds(
            self.coarse_bounds_half_open,
            "task common coarse bounds",
        )
        detail = _bounds(
            self.detail_bounds_half_open,
            "task common detail bounds",
        )
        object.__setattr__(self, "coarse_bounds_half_open", coarse)
        object.__setattr__(self, "detail_bounds_half_open", detail)
        if (
            type(self.task_span_cells) is not int
            or self.task_span_cells <= 0
            or coarse[1] - coarse[0] != self.task_span_cells
            or coarse[3] - coarse[2] != self.task_span_cells
            or detail[1] - detail[0]
            != self.task_span_cells * DETAIL_PER_GLOBAL
            or detail[3] - detail[2]
            != self.task_span_cells * DETAIL_PER_GLOBAL
            or tuple(item * DETAIL_PER_GLOBAL for item in coarse) != detail
        ):
            raise TaskCacheError("task common bounds and span differ")
        if self.scale_bucket not in {bucket.value for bucket in TaskScaleBucket}:
            raise TaskCacheError("task common scale bucket is invalid")
        if not isinstance(self.source_commit, str) or _GIT_SHA.fullmatch(
            self.source_commit
        ) is None:
            raise TaskCacheError("task common source commit is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "scenario_identity_sha256": self.scenario_identity_sha256,
            "source_identity_sha256": self.source_identity_sha256,
            "coarse_bounds_half_open": list(self.coarse_bounds_half_open),
            "detail_bounds_half_open": list(self.detail_bounds_half_open),
            "task_span_cells": self.task_span_cells,
            "scale_bucket": self.scale_bucket,
            "geometry_sha256": self.geometry_sha256,
            "halo_contract_sha256": self.halo_contract_sha256,
            "capability_bundle_sha256": self.capability_bundle_sha256,
            "generator_sha256": self.generator_sha256,
            "source_commit": self.source_commit,
        }

    def sha256(self) -> str:
        return _json_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class PlatformTaskKey:
    common_key_sha256: str
    platform_type: str
    qualified_start_identity_sha256: str
    platform_capability_sha256: str
    reachability_algorithm_id: str
    physical_evidence_algorithm_id: str
    sensor_algorithm_id: str
    visibility_algorithm_id: str
    training_semantics_sha256: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.common_key_sha256, "platform task common key"),
            (
                self.qualified_start_identity_sha256,
                "platform task start identity",
            ),
            (
                self.platform_capability_sha256,
                "platform task capability identity",
            ),
            (
                self.training_semantics_sha256,
                "platform task training semantics",
            ),
        ):
            _require_sha256(value, name)
        if self.platform_type not in _PLATFORMS:
            raise TaskCacheError("platform task type is invalid")
        for value, name in (
            (self.reachability_algorithm_id, "reachability algorithm"),
            (self.physical_evidence_algorithm_id, "physical evidence algorithm"),
            (self.sensor_algorithm_id, "sensor algorithm"),
            (self.visibility_algorithm_id, "visibility algorithm"),
        ):
            _require_identifier(value, name)

    def to_dict(self) -> dict[str, object]:
        return {
            "common_key_sha256": self.common_key_sha256,
            "platform_type": self.platform_type,
            "qualified_start_identity_sha256": (
                self.qualified_start_identity_sha256
            ),
            "platform_capability_sha256": self.platform_capability_sha256,
            "reachability_algorithm_id": self.reachability_algorithm_id,
            "physical_evidence_algorithm_id": (
                self.physical_evidence_algorithm_id
            ),
            "sensor_algorithm_id": self.sensor_algorithm_id,
            "visibility_algorithm_id": self.visibility_algorithm_id,
            "training_semantics_sha256": self.training_semantics_sha256,
        }

    def sha256(self) -> str:
        return _json_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class TaskCommonPayload:
    geometry: FrozenTaskGeometry
    local_world_bounds_m: tuple[float, float, float, float]
    local_halo_world_bounds_m: tuple[float, float, float, float]
    arrays: Mapping[str, np.ndarray]
    source_identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, FrozenTaskGeometry):
            raise TaskCacheError("task common geometry is invalid")
        local = _world_bounds(self.local_world_bounds_m, "task local world bounds")
        halo = _world_bounds(
            self.local_halo_world_bounds_m,
            "task halo world bounds",
        )
        if (
            halo[0] > local[0]
            or halo[1] > local[1]
            or halo[2] < local[2]
            or halo[3] < local[3]
        ):
            raise TaskCacheError("task halo world bounds do not enclose task")
        object.__setattr__(self, "local_world_bounds_m", local)
        object.__setattr__(self, "local_halo_world_bounds_m", halo)
        _require_sha256(self.source_identity_sha256, "task common source identity")
        _validate_array_mapping(self.arrays)


@dataclass(frozen=True, slots=True)
class PlatformTaskPayload:
    platform_type: str
    common_artifact_sha256: str
    arrays: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.platform_type not in _PLATFORMS:
            raise TaskCacheError("platform task payload type is invalid")
        _require_sha256(
            self.common_artifact_sha256,
            "platform task common artifact",
        )
        _validate_array_mapping(self.arrays)
        if not isinstance(self.diagnostics, Mapping):
            raise TaskCacheError("platform task diagnostics are invalid")
        _canonical_json(_json_value(self.diagnostics))


@dataclass(frozen=True, slots=True)
class TaskBuildJournal:
    task_key_sha256: str
    source_identity_sha256: str
    algorithm_identities: Mapping[str, str]
    completed_stage: str
    committed_blob_sha256: Mapping[str, str]
    journal_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.task_key_sha256, "task journal key")
        _require_sha256(self.source_identity_sha256, "task journal source")
        algorithms = _string_mapping(
            self.algorithm_identities,
            "task journal algorithm identities",
            sha_values=False,
        )
        blobs = _string_mapping(
            self.committed_blob_sha256,
            "task journal committed blobs",
            sha_values=True,
        )
        _require_identifier(self.completed_stage, "task journal completed stage")
        object.__setattr__(self, "algorithm_identities", MappingProxyType(algorithms))
        object.__setattr__(self, "committed_blob_sha256", MappingProxyType(blobs))
        _require_sha256(self.journal_sha256, "task journal identity")
        if self.journal_sha256 != _json_sha256(self._body()):
            raise TaskCacheError("task journal identity differs")

    @classmethod
    def create(
        cls,
        *,
        task_key_sha256: str,
        source_identity_sha256: str,
        algorithm_identities: Mapping[str, str],
        completed_stage: str,
        committed_blob_sha256: Mapping[str, str],
    ) -> "TaskBuildJournal":
        body = {
            "schema": _JOURNAL_SCHEMA,
            "task_key_sha256": task_key_sha256,
            "source_identity_sha256": source_identity_sha256,
            "algorithm_identities": dict(algorithm_identities),
            "completed_stage": completed_stage,
            "committed_blob_sha256": dict(committed_blob_sha256),
        }
        return cls(
            task_key_sha256=task_key_sha256,
            source_identity_sha256=source_identity_sha256,
            algorithm_identities=algorithm_identities,
            completed_stage=completed_stage,
            committed_blob_sha256=committed_blob_sha256,
            journal_sha256=_json_sha256(body),
        )

    def _body(self) -> dict[str, object]:
        return {
            "schema": _JOURNAL_SCHEMA,
            "task_key_sha256": self.task_key_sha256,
            "source_identity_sha256": self.source_identity_sha256,
            "algorithm_identities": dict(self.algorithm_identities),
            "completed_stage": self.completed_stage,
            "committed_blob_sha256": dict(self.committed_blob_sha256),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "journal_sha256": self.journal_sha256}

    def matches_resume_authority(
        self,
        *,
        task_key_sha256: str,
        source_identity_sha256: str,
        algorithm_identities: Mapping[str, str],
        committed_blob_sha256: Mapping[str, str],
    ) -> bool:
        return (
            self.task_key_sha256 == task_key_sha256
            and self.source_identity_sha256 == source_identity_sha256
            and dict(self.algorithm_identities) == dict(algorithm_identities)
            and dict(self.committed_blob_sha256)
            == dict(committed_blob_sha256)
        )


@dataclass(frozen=True, slots=True)
class TaskCommonArtifact:
    root: Path
    key: TaskCommonKey
    artifact_sha256: str
    geometry: FrozenTaskGeometry
    local_world_bounds_m: tuple[float, float, float, float]
    local_halo_world_bounds_m: tuple[float, float, float, float]
    arrays: Mapping[str, np.ndarray]
    source_identity_sha256: str
    manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PlatformTaskArtifact:
    root: Path
    key: PlatformTaskKey
    artifact_sha256: str
    platform_type: str
    common_artifact_sha256: str
    arrays: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, object]
    manifest: Mapping[str, object]


def _string_mapping(
    value: object,
    name: str,
    *,
    sha_values: bool,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TaskCacheError(f"{name} are invalid")
    output: dict[str, str] = {}
    for key, item in sorted(value.items()):
        if not isinstance(key, str) or _SAFE_NAME.fullmatch(key) is None:
            raise TaskCacheError(f"{name} key is invalid")
        if sha_values:
            output[key] = _require_sha256(item, f"{name} value")
        else:
            output[key] = _require_identifier(item, f"{name} value")
    return output


def _validate_array_mapping(value: object) -> None:
    if not isinstance(value, Mapping) or not value:
        raise TaskCacheError("task artifact arrays are empty or invalid")
    for name, array in value.items():
        if not isinstance(name, str) or _SAFE_NAME.fullmatch(name) is None:
            raise TaskCacheError("task artifact array name is unsafe")
        values = np.asarray(array)
        if (
            values.ndim < 1
            or values.size < 1
            or values.dtype.hasobject
            or values.dtype.fields is not None
            or values.dtype.kind not in "biuf"
        ):
            raise TaskCacheError("task artifact array dtype or shape is invalid")


def _geometry_to_dict(value: FrozenTaskGeometry) -> dict[str, object]:
    return {
        "episode_seed": value.episode_seed,
        "scale_bucket": value.scale_bucket.value,
        "span_cells": value.span_cells,
        "coarse_bounds_half_open": list(value.coarse_bounds_half_open),
        "detail_bounds_half_open": list(value.detail_bounds_half_open),
        "halo_coarse_bounds_half_open": list(
            value.halo_coarse_bounds_half_open
        ),
        "local_start_cell": list(value.local_start_cell),
        "geometry_sha256": value.geometry_sha256,
    }


def _geometry_from_dict(value: object) -> FrozenTaskGeometry:
    fields = frozenset(
        (
            "episode_seed",
            "scale_bucket",
            "span_cells",
            "coarse_bounds_half_open",
            "detail_bounds_half_open",
            "halo_coarse_bounds_half_open",
            "local_start_cell",
            "geometry_sha256",
        )
    )
    if not isinstance(value, Mapping) or set(value) != fields:
        raise TaskCacheError("task artifact geometry fields are invalid")
    try:
        return FrozenTaskGeometry(
            episode_seed=str(value["episode_seed"]),
            scale_bucket=TaskScaleBucket(str(value["scale_bucket"])),
            span_cells=int(value["span_cells"]),
            coarse_bounds_half_open=tuple(value["coarse_bounds_half_open"]),
            detail_bounds_half_open=tuple(value["detail_bounds_half_open"]),
            halo_coarse_bounds_half_open=tuple(
                value["halo_coarse_bounds_half_open"]
            ),
            local_start_cell=tuple(value["local_start_cell"]),
            geometry_sha256=str(value["geometry_sha256"]),
        )
    except (TypeError, ValueError) as error:
        raise TaskCacheError("task artifact geometry is invalid") from error


def _local_origin(
    shape: tuple[int, ...],
    geometry: FrozenTaskGeometry,
) -> list[int]:
    if len(shape) != 2:
        return [0 for _ in shape]
    coarse = geometry.coarse_bounds_half_open
    halo = geometry.halo_coarse_bounds_half_open
    coarse_shape = (coarse[1] - coarse[0], coarse[3] - coarse[2])
    detail_shape = tuple(value * DETAIL_PER_GLOBAL for value in coarse_shape)
    halo_shape = (halo[1] - halo[0], halo[3] - halo[2])
    halo_detail_shape = tuple(value * DETAIL_PER_GLOBAL for value in halo_shape)
    if shape in {coarse_shape, detail_shape}:
        return [0, 0]
    if shape == halo_shape:
        return [halo[0] - coarse[0], halo[2] - coarse[2]]
    if shape == halo_detail_shape:
        return [
            (halo[0] - coarse[0]) * DETAIL_PER_GLOBAL,
            (halo[2] - coarse[2]) * DETAIL_PER_GLOBAL,
        ]
    return [0, 0]


def _array_blob(
    name: str,
    array: np.ndarray,
    geometry: FrozenTaskGeometry | None,
) -> tuple[bytes, dict[str, object]]:
    values = np.asarray(array)
    shape = tuple(int(item) for item in values.shape)
    if values.dtype.kind == "b":
        logical = np.ascontiguousarray(values, dtype=np.bool_)
        storage = np.packbits(logical.reshape(-1), bitorder="little")
        raw = storage.tobytes(order="C")
        encoding = "bitpack-little"
        dtype = "|b1"
        storage_dtype = "|u1"
    else:
        dtype_value = values.dtype.newbyteorder("<")
        logical = np.ascontiguousarray(values, dtype=dtype_value)
        raw = logical.tobytes(order="C")
        encoding = "raw-c-order"
        dtype = logical.dtype.str
        storage_dtype = dtype
    relative = f"blobs/{name}.bin"
    return raw, {
        "relative_path": relative,
        "encoding": encoding,
        "shape": list(shape),
        "local_origin_index": (
            [0 for _ in shape]
            if geometry is None
            else _local_origin(shape, geometry)
        ),
        "dtype": dtype,
        "logical_dtype": dtype,
        "storage_dtype": storage_dtype,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _prepared_arrays(
    arrays: Mapping[str, np.ndarray],
    geometry: FrozenTaskGeometry | None,
) -> tuple[dict[str, bytes], dict[str, dict[str, object]]]:
    blobs: dict[str, bytes] = {}
    metadata: dict[str, dict[str, object]] = {}
    for name in sorted(arrays):
        raw, entry = _array_blob(name, arrays[name], geometry)
        blobs[name] = raw
        metadata[name] = entry
    return blobs, metadata


def _safe_relative(value: object, expected_name: str) -> PurePosixPath:
    if not isinstance(value, str) or "\\" in value:
        raise TaskCacheError("task artifact relative path is unsafe")
    path = PurePosixPath(value)
    expected = PurePosixPath("blobs") / f"{expected_name}.bin"
    if path.is_absolute() or path != expected or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise TaskCacheError("task artifact relative path is unsafe")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise TaskCacheError("task artifact blob cannot be read") from error
    return digest.hexdigest()


def _assert_plain_directory(path: Path, name: str) -> None:
    if path.is_symlink():
        raise TaskCacheError(f"{name} symlink is forbidden")
    if path.exists() and not path.is_dir():
        raise TaskCacheError(f"{name} is not a directory")


def _assert_plain_file(path: Path, name: str) -> None:
    if path.is_symlink():
        raise TaskCacheError(f"{name} symlink is forbidden")
    if not path.is_file():
        raise TaskCacheError(f"{name} is missing")


def _read_manifest(path: Path) -> dict[str, object]:
    _assert_plain_file(path, "task artifact manifest")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TaskCacheError("task artifact manifest is invalid") from error
    if not isinstance(value, dict):
        raise TaskCacheError("task artifact manifest root is invalid")
    if raw != _canonical_json(value) + b"\n":
        raise TaskCacheError("task artifact manifest is not canonical")
    return value


def _validate_array_entries(
    root: Path,
    value: object,
) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping) or not value:
        raise TaskCacheError("task artifact arrays are invalid")
    arrays: dict[str, np.ndarray] = {}
    for name, raw_entry in sorted(value.items()):
        if not isinstance(name, str) or _SAFE_NAME.fullmatch(name) is None:
            raise TaskCacheError("task artifact array name is unsafe")
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != _ARRAY_FIELDS:
            raise TaskCacheError("task artifact array metadata differs")
        relative = _safe_relative(raw_entry["relative_path"], name)
        shape_raw = raw_entry["shape"]
        origin_raw = raw_entry["local_origin_index"]
        if (
            not isinstance(shape_raw, list)
            or not shape_raw
            or any(type(item) is not int or item <= 0 for item in shape_raw)
            or not isinstance(origin_raw, list)
            or len(origin_raw) != len(shape_raw)
            or any(type(item) is not int for item in origin_raw)
        ):
            raise TaskCacheError("task artifact array geometry is invalid")
        shape = tuple(shape_raw)
        size = raw_entry["size_bytes"]
        if type(size) is not int or size <= 0:
            raise TaskCacheError("task artifact blob size is invalid")
        expected_sha = _require_sha256(
            raw_entry["sha256"],
            "task artifact blob identity",
        )
        blob = root.joinpath(*relative.parts)
        _assert_plain_file(blob, "task artifact blob")
        try:
            actual_size = blob.stat().st_size
        except OSError as error:
            raise TaskCacheError("task artifact blob cannot be stat'ed") from error
        if actual_size != size or _file_sha256(blob) != expected_sha:
            raise TaskCacheError("task artifact blob size or identity differs")
        encoding = raw_entry["encoding"]
        dtype = raw_entry["dtype"]
        logical_dtype = raw_entry["logical_dtype"]
        storage_dtype = raw_entry["storage_dtype"]
        if encoding == "bitpack-little":
            if (
                dtype != "|b1"
                or logical_dtype != "|b1"
                or storage_dtype != "|u1"
                or size != (math.prod(shape) + 7) // 8
            ):
                raise TaskCacheError("task artifact Boolean encoding differs")
            packed = np.memmap(blob, dtype=np.uint8, mode="r", shape=(size,))
            unpacked = np.unpackbits(
                packed,
                bitorder="little",
                count=math.prod(shape),
            ).astype(np.bool_, copy=False).reshape(shape)
            unpacked.setflags(write=False)
            arrays[name] = unpacked
        elif encoding == "raw-c-order":
            if (
                not isinstance(dtype, str)
                or logical_dtype != dtype
                or storage_dtype != dtype
            ):
                raise TaskCacheError("task artifact numeric encoding differs")
            try:
                parsed_dtype = np.dtype(dtype)
            except TypeError as error:
                raise TaskCacheError("task artifact numeric dtype is invalid") from error
            if (
                parsed_dtype.kind not in "iuf"
                or parsed_dtype.byteorder == ">"
                or parsed_dtype.itemsize * math.prod(shape) != size
            ):
                raise TaskCacheError("task artifact numeric dtype differs")
            arrays[name] = np.memmap(
                blob,
                dtype=parsed_dtype,
                mode="r",
                shape=shape,
                order="C",
            )
        else:
            raise TaskCacheError("task artifact array encoding is unsupported")
    return MappingProxyType(arrays)


def _artifact_hash(value: Mapping[str, object]) -> str:
    body = dict(value)
    body.pop("artifact_sha256", None)
    return _json_sha256(body)


class TaskCacheStore:
    """Publish and load immutable common/platform task artifacts."""

    def __init__(self, cache_root: str | Path) -> None:
        self.root = Path(cache_root)
        if self.root.is_symlink():
            raise TaskCacheError("task cache root symlink is forbidden")

    def _artifact_parent(self, kind: str, *, create: bool) -> Path:
        parent = self.root / "tasks" / "v1" / kind
        for candidate, name in (
            (self.root, "task cache root"),
            (self.root / "tasks", "task cache tasks directory"),
            (self.root / "tasks" / "v1", "task cache version directory"),
            (parent, "task cache artifact parent"),
        ):
            if create:
                candidate.mkdir(parents=True, exist_ok=True)
            _assert_plain_directory(candidate, name)
        return parent

    def load_common(self, key: TaskCommonKey) -> TaskCommonArtifact | None:
        if not isinstance(key, TaskCommonKey):
            raise TypeError("load_common requires TaskCommonKey")
        parent = self._artifact_parent("common", create=False)
        if not parent.exists():
            return None
        target = parent / key.sha256()
        if not target.exists() and not target.is_symlink():
            return None
        return self._load_common_at(target, key)

    def load_platform(
        self,
        key: PlatformTaskKey,
    ) -> PlatformTaskArtifact | None:
        if not isinstance(key, PlatformTaskKey):
            raise TypeError("load_platform requires PlatformTaskKey")
        parent = self._artifact_parent("platform", create=False)
        if not parent.exists():
            return None
        target = parent / key.sha256()
        if not target.exists() and not target.is_symlink():
            return None
        return self._load_platform_at(target, key)

    def commit_common(
        self,
        key: TaskCommonKey,
        payload: TaskCommonPayload,
    ) -> TaskCommonArtifact:
        if not isinstance(key, TaskCommonKey) or not isinstance(
            payload, TaskCommonPayload
        ):
            raise TypeError("commit_common requires common key and payload")
        geometry = payload.geometry
        if (
            key.geometry_sha256 != geometry.geometry_sha256
            or key.source_identity_sha256 != payload.source_identity_sha256
            or key.coarse_bounds_half_open != geometry.coarse_bounds_half_open
            or key.detail_bounds_half_open != geometry.detail_bounds_half_open
            or key.task_span_cells != geometry.span_cells
            or key.scale_bucket != geometry.scale_bucket.value
        ):
            raise TaskCacheError("task common key and payload differ")
        blobs, entries = _prepared_arrays(payload.arrays, geometry)
        body: dict[str, object] = {
            "schema": TASK_COMMON_SCHEMA,
            "kind": "common",
            "key_sha256": key.sha256(),
            "key": key.to_dict(),
            "geometry": _geometry_to_dict(geometry),
            "local_world_bounds_m": list(payload.local_world_bounds_m),
            "local_halo_world_bounds_m": list(payload.local_halo_world_bounds_m),
            "source_identity_sha256": payload.source_identity_sha256,
            "arrays": entries,
        }
        manifest = {**body, "artifact_sha256": _json_sha256(body)}
        parent = self._artifact_parent("common", create=True)
        target = parent / key.sha256()
        self._publish(target, blobs, manifest)
        artifact = self._load_common_at(target, key)
        if artifact.artifact_sha256 != manifest["artifact_sha256"]:
            raise TaskCacheError("task common key collision or nondeterminism")
        return artifact

    def commit_platform(
        self,
        key: PlatformTaskKey,
        payload: PlatformTaskPayload,
    ) -> PlatformTaskArtifact:
        if not isinstance(key, PlatformTaskKey) or not isinstance(
            payload, PlatformTaskPayload
        ):
            raise TypeError("commit_platform requires platform key and payload")
        if key.platform_type != payload.platform_type:
            raise TaskCacheError("platform task key and payload differ")
        blobs, entries = _prepared_arrays(payload.arrays, None)
        body: dict[str, object] = {
            "schema": TASK_COVERABILITY_SCHEMA,
            "kind": "platform",
            "key_sha256": key.sha256(),
            "key": key.to_dict(),
            "platform_type": payload.platform_type,
            "common_artifact_sha256": payload.common_artifact_sha256,
            "diagnostics": _json_value(payload.diagnostics),
            "arrays": entries,
        }
        manifest = {**body, "artifact_sha256": _json_sha256(body)}
        parent = self._artifact_parent("platform", create=True)
        target = parent / key.sha256()
        self._publish(target, blobs, manifest)
        artifact = self._load_platform_at(target, key)
        if artifact.artifact_sha256 != manifest["artifact_sha256"]:
            raise TaskCacheError("platform task key collision or nondeterminism")
        return artifact

    def _publish(
        self,
        target: Path,
        blobs: Mapping[str, bytes],
        manifest: Mapping[str, object],
    ) -> None:
        _assert_plain_directory(target.parent, "task cache artifact parent")
        if target.is_symlink():
            raise TaskCacheError("task artifact directory symlink is forbidden")
        if target.exists():
            return
        temporary = Path(
            tempfile.mkdtemp(
                dir=target.parent,
                prefix=f".{target.name}.tmp-",
            )
        )
        blob_root = temporary / "blobs"
        blob_root.mkdir()
        for name in sorted(blobs):
            path = blob_root / f"{name}.bin"
            try:
                with path.open("xb") as stream:
                    stream.write(blobs[name])
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as error:
                raise TaskCacheError("task artifact blob write failed") from error
        self._fsync_directory(blob_root)
        manifest_path = temporary / "manifest.json"
        try:
            with manifest_path.open("xb") as stream:
                stream.write(_canonical_json(dict(manifest)) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise TaskCacheError("task artifact manifest write failed") from error
        self._fsync_directory(temporary)
        try:
            os.rename(temporary, target)
        except FileExistsError:
            if not target.is_dir() or target.is_symlink():
                raise TaskCacheError("task artifact publish target differs")
        except OSError as error:
            raise TaskCacheError("task artifact atomic publish failed") from error
        self._fsync_directory(target.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as error:
            raise TaskCacheError("task cache directory fsync failed") from error

    def _load_common_at(
        self,
        target: Path,
        expected_key: TaskCommonKey,
    ) -> TaskCommonArtifact:
        _assert_plain_directory(target, "task common artifact")
        manifest = _read_manifest(target / "manifest.json")
        expected_fields = frozenset(
            (
                "schema",
                "kind",
                "key_sha256",
                "key",
                "geometry",
                "local_world_bounds_m",
                "local_halo_world_bounds_m",
                "source_identity_sha256",
                "arrays",
                "artifact_sha256",
            )
        )
        if (
            set(manifest) != expected_fields
            or manifest["schema"] != TASK_COMMON_SCHEMA
            or manifest["kind"] != "common"
        ):
            raise TaskCacheError("task common artifact schema differs")
        try:
            key = TaskCommonKey(**manifest["key"])
        except (TypeError, ValueError) as error:
            raise TaskCacheError("task common artifact key is invalid") from error
        if (
            key != expected_key
            or manifest["key_sha256"] != key.sha256()
            or target.name != key.sha256()
        ):
            raise TaskCacheError("task common artifact key differs")
        artifact_sha = _require_sha256(
            manifest["artifact_sha256"],
            "task common artifact identity",
        )
        if artifact_sha != _artifact_hash(manifest):
            raise TaskCacheError("task common artifact identity differs")
        geometry = _geometry_from_dict(manifest["geometry"])
        if (
            geometry.geometry_sha256 != key.geometry_sha256
            or geometry.coarse_bounds_half_open != key.coarse_bounds_half_open
            or geometry.detail_bounds_half_open != key.detail_bounds_half_open
            or geometry.span_cells != key.task_span_cells
            or geometry.scale_bucket.value != key.scale_bucket
        ):
            raise TaskCacheError("task common artifact geometry differs")
        local = _world_bounds(
            manifest["local_world_bounds_m"],
            "task local world bounds",
        )
        halo = _world_bounds(
            manifest["local_halo_world_bounds_m"],
            "task halo world bounds",
        )
        source_identity = _require_sha256(
            manifest["source_identity_sha256"],
            "task common source identity",
        )
        if source_identity != key.source_identity_sha256:
            raise TaskCacheError("task common source identity differs")
        arrays = _validate_array_entries(target, manifest["arrays"])
        return TaskCommonArtifact(
            root=target,
            key=key,
            artifact_sha256=artifact_sha,
            geometry=geometry,
            local_world_bounds_m=local,
            local_halo_world_bounds_m=halo,
            arrays=arrays,
            source_identity_sha256=source_identity,
            manifest=MappingProxyType(manifest),
        )

    def _load_platform_at(
        self,
        target: Path,
        expected_key: PlatformTaskKey,
    ) -> PlatformTaskArtifact:
        _assert_plain_directory(target, "platform task artifact")
        manifest = _read_manifest(target / "manifest.json")
        expected_fields = frozenset(
            (
                "schema",
                "kind",
                "key_sha256",
                "key",
                "platform_type",
                "common_artifact_sha256",
                "diagnostics",
                "arrays",
                "artifact_sha256",
            )
        )
        if (
            set(manifest) != expected_fields
            or manifest["schema"] != TASK_COVERABILITY_SCHEMA
            or manifest["kind"] != "platform"
        ):
            raise TaskCacheError("platform task artifact schema differs")
        try:
            key = PlatformTaskKey(**manifest["key"])
        except (TypeError, ValueError) as error:
            raise TaskCacheError("platform task artifact key is invalid") from error
        if (
            key != expected_key
            or manifest["key_sha256"] != key.sha256()
            or target.name != key.sha256()
            or manifest["platform_type"] != key.platform_type
        ):
            raise TaskCacheError("platform task artifact key differs")
        artifact_sha = _require_sha256(
            manifest["artifact_sha256"],
            "platform task artifact identity",
        )
        if artifact_sha != _artifact_hash(manifest):
            raise TaskCacheError("platform task artifact identity differs")
        common_sha = _require_sha256(
            manifest["common_artifact_sha256"],
            "platform task common artifact identity",
        )
        diagnostics = _json_value(manifest["diagnostics"])
        if not isinstance(diagnostics, dict):
            raise TaskCacheError("platform task diagnostics are invalid")
        arrays = _validate_array_entries(target, manifest["arrays"])
        return PlatformTaskArtifact(
            root=target,
            key=key,
            artifact_sha256=artifact_sha,
            platform_type=key.platform_type,
            common_artifact_sha256=common_sha,
            arrays=arrays,
            diagnostics=MappingProxyType(diagnostics),
            manifest=MappingProxyType(manifest),
        )

    def load_build_journal(
        self,
        task_key_sha256: str,
    ) -> TaskBuildJournal | None:
        key = _require_sha256(task_key_sha256, "task journal key")
        parent = self._artifact_parent("journals", create=False)
        if not parent.exists():
            return None
        path = parent / f"{key}.json"
        if not path.exists() and not path.is_symlink():
            return None
        if path.is_symlink():
            raise TaskCacheError("task journal symlink is forbidden")
        try:
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TaskCacheError("task journal is invalid") from error
        fields = frozenset(
            (
                "schema",
                "task_key_sha256",
                "source_identity_sha256",
                "algorithm_identities",
                "completed_stage",
                "committed_blob_sha256",
                "journal_sha256",
            )
        )
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or value["schema"] != _JOURNAL_SCHEMA
            or raw != _canonical_json(value) + b"\n"
            or value["task_key_sha256"] != key
        ):
            raise TaskCacheError("task journal schema or key differs")
        return TaskBuildJournal(
            task_key_sha256=value["task_key_sha256"],
            source_identity_sha256=value["source_identity_sha256"],
            algorithm_identities=value["algorithm_identities"],
            completed_stage=value["completed_stage"],
            committed_blob_sha256=value["committed_blob_sha256"],
            journal_sha256=value["journal_sha256"],
        )

    def commit_build_journal(self, journal: TaskBuildJournal) -> None:
        if not isinstance(journal, TaskBuildJournal):
            raise TypeError("commit_build_journal requires TaskBuildJournal")
        parent = self._artifact_parent("journals", create=True)
        path = parent / f"{journal.task_key_sha256}.json"
        if path.is_symlink():
            raise TaskCacheError("task journal symlink is forbidden")
        descriptor, temporary_name = tempfile.mkstemp(
            dir=parent,
            prefix=f".{journal.task_key_sha256}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_canonical_json(journal.to_dict()) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
            self._fsync_directory(parent)
        except OSError as error:
            raise TaskCacheError("task journal atomic write failed") from error

    def load_hopper_closure_journal(
        self, task_key_sha256: str
    ) -> HopperClosureJournal | None:
        """Load the latest atomically committed HOPPER closure round."""
        from .hopper_task_closure import HopperClosureJournal

        key = _require_sha256(task_key_sha256, "hopper closure journal key")
        parent = self._artifact_parent(
            "hopper-closure-journals", create=False
        )
        if not parent.exists():
            return None
        path = parent / f"{key}.json"
        if not path.exists() and not path.is_symlink():
            return None
        if path.is_symlink():
            raise TaskCacheError(
                "hopper closure journal symlink is forbidden"
            )
        try:
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TaskCacheError("hopper closure journal is invalid") from error
        if (
            not isinstance(value, dict)
            or raw != _canonical_json(value) + b"\n"
            or value.get("task_key_sha256") != key
        ):
            raise TaskCacheError(
                "hopper closure journal schema or key differs"
            )
        try:
            return HopperClosureJournal.from_dict(value)
        except (TypeError, ValueError) as error:
            raise TaskCacheError(
                "hopper closure journal authority differs"
            ) from error

    def commit_hopper_closure_journal(
        self, journal: HopperClosureJournal
    ) -> None:
        """Replace the latest HOPPER round only after its bytes are durable."""
        from .hopper_task_closure import HopperClosureJournal

        if not isinstance(journal, HopperClosureJournal):
            raise TypeError(
                "commit_hopper_closure_journal requires HopperClosureJournal"
            )
        parent = self._artifact_parent(
            "hopper-closure-journals", create=True
        )
        path = parent / f"{journal.task_key_sha256}.json"
        if path.is_symlink():
            raise TaskCacheError(
                "hopper closure journal symlink is forbidden"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=parent,
            prefix=f".{journal.task_key_sha256}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_canonical_json(journal.to_dict()) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
            self._fsync_directory(parent)
        except OSError as error:
            raise TaskCacheError(
                "hopper closure journal atomic write failed"
            ) from error


__all__ = [
    "TASK_COMMON_SCHEMA",
    "TASK_COVERABILITY_SCHEMA",
    "PlatformTaskArtifact",
    "PlatformTaskKey",
    "PlatformTaskPayload",
    "TaskBuildJournal",
    "TaskCacheError",
    "TaskCacheStore",
    "TaskCommonArtifact",
    "TaskCommonKey",
    "TaskCommonPayload",
]
