"""Deterministic external capability closure for formal Volume 3 runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeAlias


CAPABILITY_FREEZE_SCHEMA = "lunar-training-capability-freeze/v1"
PLATFORMS = ("WHEELED", "LEGGED", "HOPPER")
RUN_KINDS = ("formal", "development-smoke")
CAPABILITY_TYPES = {
    "WHEELED": "lunar-planner-wheeled-capability/v1",
    "LEGGED": "lunar-planner-legged-capability/v1",
    "HOPPER": "lunar-planner-hopper-capability/v1",
}
_RESOURCE_KINDS = frozenset(("document", "urdf", "mesh"))
_LOCK_FIELDS = frozenset(
    ("schema", "formal_eligible", "test_only", "proxy", "platforms")
)
_PLATFORM_FIELDS = frozenset(
    (
        "platform_type",
        "capability_type",
        "capability_version",
        "content_path",
        "content_file_sha256",
        "resources",
    )
)
_CONTENT_FIELDS = frozenset(
    ("schema", "platform_type", "capability_version", "content")
)
_RESOURCE_FIELDS = frozenset(("kind", "path", "sha256"))


class CapabilityFreezeError(ValueError):
    """An external capability closure is incomplete, mutable, or mislabeled."""


FrozenJsonScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class FrozenJsonObject:
    """Pickle-safe immutable JSON mapping used after the one source parse."""

    items: tuple[tuple[str, "FrozenJsonValue"], ...]


FrozenJsonValue: TypeAlias = FrozenJsonScalar | FrozenJsonObject | tuple["FrozenJsonValue", ...]


@dataclass(frozen=True, slots=True)
class FrozenCapabilityResource:
    kind: str
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class FrozenPlatformCapability:
    platform_type: str
    capability_type: str
    capability_version: str
    content: FrozenJsonObject
    content_sha256: str
    resources: tuple[FrozenCapabilityResource, ...]


@dataclass(frozen=True, slots=True)
class FrozenCapabilityBundle:
    schema: str
    platforms: tuple[FrozenPlatformCapability, ...]
    bundle_sha256: str
    formal_eligible: bool

    def for_platform(self, platform_type: str) -> FrozenPlatformCapability:
        if platform_type not in PLATFORMS:
            raise CapabilityFreezeError("unknown capability platform")
        return self.platforms[PLATFORMS.index(platform_type)]


@dataclass(frozen=True, slots=True)
class ScenarioIdentity:
    platform_type: str
    scenario_schedule_id: str
    worker_index: int
    capability_version: str
    capability_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenCapabilityEnvironmentFactory:
    """Bind parsed capabilities to workers without reopening closure files."""

    bundle: FrozenCapabilityBundle
    scenario_schedule_id: str
    builder: Callable[[int, str, FrozenPlatformCapability, ScenarioIdentity], object]

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, FrozenCapabilityBundle):
            raise CapabilityFreezeError("environment factory bundle is invalid")
        if not self.bundle.formal_eligible:
            raise CapabilityFreezeError("formal environment requires formal capability bundle")
        if not isinstance(self.scenario_schedule_id, str) or not self.scenario_schedule_id:
            raise CapabilityFreezeError("scenario schedule identity is missing")
        if not callable(self.builder):
            raise CapabilityFreezeError("capability environment builder is not callable")

    def __call__(self, worker_index: int, platform_type: str) -> object:
        if type(worker_index) is not int or worker_index < 0:
            raise CapabilityFreezeError("worker index must be non-negative")
        capability = self.bundle.for_platform(platform_type)
        scenario = ScenarioIdentity(
            platform_type=platform_type,
            scenario_schedule_id=self.scenario_schedule_id,
            worker_index=worker_index,
            capability_version=capability.capability_version,
            capability_sha256=capability.content_sha256,
        )
        return self.builder(worker_index, platform_type, capability, scenario)


def load_frozen_capability_bundle(
    lock_path: str | Path, *, run_kind: str
) -> FrozenCapabilityBundle:
    """Verify and parse one external closure into an immutable process payload."""
    if run_kind not in RUN_KINDS:
        raise CapabilityFreezeError("run kind must be formal or development-smoke")
    target = Path(lock_path)
    if not target.is_absolute():
        raise CapabilityFreezeError("capability lock path must be absolute")
    if target.is_symlink() or not target.is_file():
        raise CapabilityFreezeError("capability lock must be a regular file")
    root = target.parent.resolve(strict=True)
    raw = _read_json(target, "capability lock")
    if set(raw) != _LOCK_FIELDS:
        raise CapabilityFreezeError("capability lock schema fields are invalid")
    if raw["schema"] != CAPABILITY_FREEZE_SCHEMA:
        raise CapabilityFreezeError("capability lock schema is unsupported")
    for flag in ("formal_eligible", "test_only", "proxy"):
        if type(raw[flag]) is not bool:
            raise CapabilityFreezeError(f"capability lock {flag} must be boolean")
    formal_eligible = raw["formal_eligible"]
    if run_kind == "formal":
        if not formal_eligible or raw["test_only"] or raw["proxy"]:
            raise CapabilityFreezeError(
                "formal capability bundle must be eligible, non-test and non-proxy"
            )
    elif formal_eligible or not (raw["test_only"] or raw["proxy"]):
        raise CapabilityFreezeError(
            "development-smoke capability bundle must be test-only or proxy"
        )
    entries = raw["platforms"]
    if not isinstance(entries, list) or len(entries) != len(PLATFORMS):
        raise CapabilityFreezeError("capability platforms must contain exactly three entries")
    seen: set[str] = set()
    parsed: list[FrozenPlatformCapability] = []
    for entry in entries:
        platform = _parse_platform_entry(root, entry)
        if platform.platform_type in seen:
            raise CapabilityFreezeError("duplicate capability platform")
        seen.add(platform.platform_type)
        parsed.append(platform)
    if seen != set(PLATFORMS):
        raise CapabilityFreezeError("capability platforms must be WHEELED, LEGGED and HOPPER")
    ordered = tuple(sorted(parsed, key=lambda item: PLATFORMS.index(item.platform_type)))
    identity = {
        "schema": CAPABILITY_FREEZE_SCHEMA,
        "platforms": [
            {
                "platform_type": item.platform_type,
                "capability_type": item.capability_type,
                "capability_version": item.capability_version,
                "content_sha256": item.content_sha256,
                "resources": [
                    {
                        "kind": resource.kind,
                        "path": resource.relative_path,
                        "sha256": resource.sha256,
                    }
                    for resource in item.resources
                ],
            }
            for item in ordered
        ],
    }
    return FrozenCapabilityBundle(
        schema=CAPABILITY_FREEZE_SCHEMA,
        platforms=ordered,
        bundle_sha256=_semantic_sha256(identity),
        formal_eligible=formal_eligible,
    )


def _parse_platform_entry(
    root: Path, entry: object
) -> FrozenPlatformCapability:
    if not isinstance(entry, Mapping) or set(entry) != _PLATFORM_FIELDS:
        raise CapabilityFreezeError("capability platform entry schema is invalid")
    platform_type = entry["platform_type"]
    if platform_type not in PLATFORMS:
        raise CapabilityFreezeError("unknown capability platform type")
    capability_type = entry["capability_type"]
    if capability_type != CAPABILITY_TYPES[platform_type]:
        raise CapabilityFreezeError("capability type is unsupported for platform")
    capability_version = entry["capability_version"]
    if not isinstance(capability_version, str) or not capability_version:
        raise CapabilityFreezeError("capability version is missing")
    content_path = _resolve_relative(root, entry["content_path"], "content")
    expected_file_hash = _digest(entry["content_file_sha256"], "content")
    content_bytes = content_path.read_bytes()
    if _sha256_bytes(content_bytes) != expected_file_hash:
        raise CapabilityFreezeError("capability content file hash mismatch")
    content_raw = _decode_json(content_bytes, "capability content")
    if set(content_raw) != _CONTENT_FIELDS:
        raise CapabilityFreezeError("capability content schema fields are invalid")
    if content_raw["schema"] != capability_type:
        raise CapabilityFreezeError("capability content type is unsupported")
    if content_raw["platform_type"] != platform_type:
        raise CapabilityFreezeError("capability content platform mismatch")
    if content_raw["capability_version"] != capability_version:
        raise CapabilityFreezeError("capability content version mismatch")
    parameters = content_raw["content"]
    if not isinstance(parameters, Mapping) or not parameters:
        raise CapabilityFreezeError("capability content must be a non-empty mapping")
    _validate_json_values(content_raw)
    resources_raw = entry["resources"]
    if not isinstance(resources_raw, list) or not resources_raw:
        raise CapabilityFreezeError("capability resources must be a non-empty list")
    resources: list[FrozenCapabilityResource] = []
    seen_paths: set[str] = set()
    for resource in resources_raw:
        if not isinstance(resource, Mapping) or set(resource) != _RESOURCE_FIELDS:
            raise CapabilityFreezeError("capability resource schema is invalid")
        kind = resource["kind"]
        if kind not in _RESOURCE_KINDS:
            raise CapabilityFreezeError("capability resource kind is unsupported")
        path = _normalized_relative(resource["path"])
        if path in seen_paths:
            raise CapabilityFreezeError("duplicate capability resource path")
        seen_paths.add(path)
        resolved = _resolve_relative(root, path, "resource")
        digest = _digest(resource["sha256"], "resource")
        if _sha256_bytes(resolved.read_bytes()) != digest:
            raise CapabilityFreezeError("capability resource hash mismatch")
        resources.append(
            FrozenCapabilityResource(kind=kind, relative_path=path, sha256=digest)
        )
    resources.sort(key=lambda item: (item.relative_path, item.kind))
    canonical_content = {
        "schema": capability_type,
        "platform_type": platform_type,
        "capability_version": capability_version,
        "content": parameters,
    }
    return FrozenPlatformCapability(
        platform_type=platform_type,
        capability_type=capability_type,
        capability_version=capability_version,
        content=_freeze_json(canonical_content),
        content_sha256=_semantic_sha256(canonical_content),
        resources=tuple(resources),
    )


def _read_json(path: Path, name: str) -> dict[str, object]:
    try:
        return _decode_json(path.read_bytes(), name)
    except OSError as error:
        raise CapabilityFreezeError(f"{name} could not be read") from error


def _decode_json(data: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CapabilityFreezeError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CapabilityFreezeError(f"{name} must be a JSON object")
    return value


def _resolve_relative(root: Path, value: object, name: str) -> Path:
    relative = _normalized_relative(value)
    try:
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError) as error:
        raise CapabilityFreezeError(f"capability {name} path escapes closure or is missing") from error
    if candidate.is_symlink() or not candidate.is_file():
        raise CapabilityFreezeError(f"capability {name} path must be a regular file")
    return candidate


def _normalized_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CapabilityFreezeError("capability path must be normalized relative POSIX")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in ("", ".", "..") for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise CapabilityFreezeError("capability path must be normalized relative POSIX")
    return value


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CapabilityFreezeError(f"capability {name} hash is invalid")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _semantic_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CapabilityFreezeError("capability identity is not canonical JSON") from error
    return _sha256_bytes(encoded)


def _validate_json_values(value: object) -> None:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise CapabilityFreezeError("capability content keys must be non-empty strings")
        for item in value.values():
            _validate_json_values(item)
    elif isinstance(value, list):
        for item in value:
            _validate_json_values(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise CapabilityFreezeError("capability content numbers must be finite")
    elif not isinstance(value, (str, int, float, bool)) and value is not None:
        raise CapabilityFreezeError("capability content contains unsupported JSON value")


def _freeze_json(value: object) -> FrozenJsonValue:
    if isinstance(value, Mapping):
        return FrozenJsonObject(
            tuple(
                (key, _freeze_json(item))
                for key, item in sorted(value.items())
            )
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


__all__ = [
    "CAPABILITY_FREEZE_SCHEMA",
    "CAPABILITY_TYPES",
    "CapabilityFreezeError",
    "FrozenCapabilityBundle",
    "FrozenCapabilityEnvironmentFactory",
    "FrozenCapabilityResource",
    "FrozenJsonObject",
    "FrozenPlatformCapability",
    "RUN_KINDS",
    "ScenarioIdentity",
    "load_frozen_capability_bundle",
]
