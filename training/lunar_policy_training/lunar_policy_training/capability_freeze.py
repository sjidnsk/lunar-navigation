"""Deterministic, typed external capability closure for formal Volume 3 runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
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
_PROJECT_MAXIMUM_SLOPE_RAD = math.pi / 6.0
_INT64_MAX = (1 << 63) - 1
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
_COMMON_CONTENT_FIELDS = frozenset(
    (
        "platform_id",
        "base_frame_id",
        "platform_document_path",
        "observation_document_path",
        "urdf_path",
        "mesh_paths",
        "observation",
        "capability",
    )
)
_OBSERVATION_FIELDS = frozenset(("sensor_range_m", "sensor_fov_deg"))
_RESOURCE_FIELDS = frozenset(("kind", "path", "sha256"))
_POSE_FIELDS = frozenset(("position_m", "orientation_wxyz"))
_WHEEL_PRIMITIVE_FIELDS = frozenset(
    ("primitive_id", "kind", "relative_end_pose", "nominal_duration_s")
)
_LEGGED_PRIMITIVE_FIELDS = frozenset(
    (
        "primitive_id",
        "kind",
        "body_frame_displacement_m",
        "yaw_change_rad",
        "nominal_duration_s",
    )
)
_HOPPER_PRIMITIVE_FIELDS = frozenset(("primitive_id",))
_PROFILE_FIELDS = frozenset(("profile_id",))
_WHEELED_FIELDS = frozenset(
    (
        "footprint_xy_m",
        "minimum_body_z_m",
        "maximum_body_z_m",
        "maximum_slope_rad",
        "maximum_obstacle_height_m",
        "maximum_forward_speed_mps",
        "maximum_reverse_speed_mps",
        "maximum_spin_rate_radps",
        "maximum_acceleration_mps2",
        "maximum_braking_deceleration_mps2",
        "maximum_yaw_acceleration_radps2",
        "maximum_lateral_acceleration_mps2",
        "maximum_curvature_per_m",
        "minimum_clearance_m",
        "motion_primitives",
    )
)
_LEGGED_FIELDS = frozenset(
    (
        "reference_point",
        "body_half_extent_m",
        "maximum_slope_rad",
        "maximum_roughness_m",
        "maximum_step_height_m",
        "maximum_gap_width_m",
        "minimum_confidence",
        "minimum_body_clearance_m",
        "body_height_m",
        "forward_speed_mps",
        "lateral_speed_mps",
        "vertical_speed_mps",
        "yaw_rate_radps",
        "maximum_linear_acceleration_mps2",
        "maximum_yaw_acceleration_radps2",
        "motion_primitives",
    )
)
_HOPPER_FIELDS = frozenset(
    (
        "body_half_extent_m",
        "platform_mass_kg",
        "gravity_mps2",
        "maximum_landing_slope_rad",
        "maximum_landing_roughness_m",
        "maximum_plane_residual_m",
        "minimum_overhead_clearance_m",
        "minimum_lateral_clearance_m",
        "minimum_landing_region_area_m2",
        "maximum_launch_speed_mps",
        "maximum_launch_impulse_newton_seconds",
        "minimum_flight_time_s",
        "maximum_flight_time_s",
        "maximum_landing_speed_mps",
        "minimum_downward_impact_speed_mps",
        "minimum_landing_clearance_m",
        "maximum_angular_speed_radps",
        "maximum_angular_acceleration_radps2",
        "maximum_initial_angular_speed_radps",
        "minimum_settle_guard_s",
        "actuator_or_impulse_profile",
        "motion_primitives",
    )
)
_WHEEL_KINDS = frozenset(
    (
        "FORWARD",
        "REVERSE",
        "FORWARD_ARC",
        "REVERSE_ARC",
        "SPIN_CLOCKWISE",
        "SPIN_COUNTERCLOCKWISE",
        "STOP_AND_SWITCH",
    )
)
_LEGGED_KINDS = frozenset(
    ("FORWARD", "BACKWARD", "LATERAL_LEFT", "LATERAL_RIGHT", "SPIN", "COUPLED")
)


class CapabilityFreezeError(ValueError):
    """An external capability closure is incomplete, mutable, or mislabeled."""


@dataclass(frozen=True, slots=True)
class FrozenVec2:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class FrozenVec3:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class FrozenQuaternion:
    w: float
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class FrozenPose3:
    position_m: FrozenVec3
    orientation: FrozenQuaternion


@dataclass(frozen=True, slots=True)
class FrozenInterval:
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class FrozenObservationCapability:
    sensor_range_m: float
    sensor_fov_rad: float


@dataclass(frozen=True, slots=True)
class FrozenWheelMotionPrimitive:
    primitive_id: str
    kind: str
    relative_end_pose: FrozenPose3
    nominal_duration_ns: int


@dataclass(frozen=True, slots=True)
class FrozenWheeledCapability:
    footprint_xy_m: tuple[FrozenVec2, ...]
    minimum_body_z_m: float
    maximum_body_z_m: float
    maximum_obstacle_height_m: float
    maximum_forward_speed_mps: float
    maximum_reverse_speed_mps: float
    maximum_spin_rate_radps: float
    maximum_acceleration_mps2: float
    maximum_braking_deceleration_mps2: float
    maximum_yaw_acceleration_radps2: float
    maximum_lateral_acceleration_mps2: float
    maximum_curvature_per_m: float
    maximum_slope_rad: float
    minimum_clearance_m: float
    motion_primitives: tuple[FrozenWheelMotionPrimitive, ...]


@dataclass(frozen=True, slots=True)
class FrozenLeggedBodyPrimitive:
    primitive_id: str
    kind: str
    body_frame_displacement_m: FrozenVec3
    yaw_change_rad: float
    nominal_duration_ns: int


@dataclass(frozen=True, slots=True)
class FrozenLeggedCapability:
    reference_point: str
    body_half_extent_m: FrozenVec3
    maximum_slope_rad: float
    maximum_roughness_m: float
    maximum_step_height_m: float
    maximum_gap_width_m: float
    minimum_confidence: float
    minimum_body_clearance_m: float
    body_height_m: FrozenInterval
    forward_speed_mps: FrozenInterval
    lateral_speed_mps: FrozenInterval
    vertical_speed_mps: FrozenInterval
    yaw_rate_radps: FrozenInterval
    maximum_linear_acceleration_mps2: float
    maximum_yaw_acceleration_radps2: float
    motion_primitives: tuple[FrozenLeggedBodyPrimitive, ...]


@dataclass(frozen=True, slots=True)
class FrozenHopperCapability:
    body_half_extent_m: FrozenVec3
    platform_mass_kg: float
    gravity_mps2: FrozenVec3
    maximum_landing_slope_rad: float
    maximum_landing_roughness_m: float
    maximum_plane_residual_m: float
    minimum_overhead_clearance_m: float
    minimum_lateral_clearance_m: float
    minimum_landing_region_area_m2: float
    maximum_launch_speed_mps: float
    maximum_launch_impulse_newton_seconds: float
    minimum_flight_time_ns: int
    maximum_flight_time_ns: int
    maximum_landing_speed_mps: float
    minimum_downward_impact_speed_mps: float
    minimum_landing_clearance_m: float
    maximum_angular_speed_radps: float
    maximum_angular_acceleration_radps2: float
    maximum_initial_angular_speed_radps: float
    minimum_settle_guard_ns: int
    actuator_profile_id: str
    source_motion_primitive_ids: tuple[str, ...]


FrozenTypedCapability: TypeAlias = (
    FrozenWheeledCapability | FrozenLeggedCapability | FrozenHopperCapability
)


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
    platform_id: str
    base_frame_id: str
    platform_document_path: str
    observation_document_path: str
    urdf_path: str
    mesh_paths: tuple[str, ...]
    observation_capability: FrozenObservationCapability
    typed_capability: FrozenTypedCapability
    content_sha256: str
    resources: tuple[FrozenCapabilityResource, ...]

    @property
    def source_motion_primitive_ids(self) -> tuple[str, ...]:
        typed = self.typed_capability
        if isinstance(typed, FrozenHopperCapability):
            return typed.source_motion_primitive_ids
        return tuple(item.primitive_id for item in typed.motion_primitives)

    def to_bridge_capability(self) -> object:
        """Create the C++ v3 bridge value from the parsed in-memory payload."""
        import lunar_planner_training_bridge as bridge_api

        typed = self.typed_capability
        if isinstance(typed, FrozenWheeledCapability):
            value = bridge_api.WheeledCapability()
            value.footprint_xy_m = [_bridge_vec2(item, bridge_api) for item in typed.footprint_xy_m]
            for field in (
                "minimum_body_z_m",
                "maximum_body_z_m",
                "maximum_forward_speed_mps",
                "maximum_reverse_speed_mps",
                "maximum_spin_rate_radps",
                "maximum_acceleration_mps2",
                "maximum_braking_deceleration_mps2",
                "maximum_yaw_acceleration_radps2",
                "maximum_lateral_acceleration_mps2",
                "maximum_curvature_per_m",
                "maximum_slope_rad",
                "minimum_clearance_m",
            ):
                setattr(value, field, getattr(typed, field))
            primitives = []
            for frozen in typed.motion_primitives:
                primitive = bridge_api.WheelMotionPrimitive()
                primitive.primitive_id = frozen.primitive_id
                primitive.kind = getattr(bridge_api.WheelPrimitiveKind, frozen.kind)
                primitive.relative_end_pose = _bridge_pose(frozen.relative_end_pose, bridge_api)
                primitive.nominal_duration = _bridge_duration(frozen.nominal_duration_ns)
                primitives.append(primitive)
            value.motion_primitives = primitives
            return value
        if isinstance(typed, FrozenLeggedCapability):
            value = bridge_api.LeggedCapability()
            value.body_half_extent_m = _bridge_vec3(typed.body_half_extent_m, bridge_api)
            for field in (
                "maximum_slope_rad",
                "maximum_roughness_m",
                "maximum_step_height_m",
                "maximum_gap_width_m",
                "minimum_confidence",
                "minimum_body_clearance_m",
                "maximum_linear_acceleration_mps2",
                "maximum_yaw_acceleration_radps2",
            ):
                setattr(value, field, getattr(typed, field))
            for field in (
                "body_height_m",
                "forward_speed_mps",
                "lateral_speed_mps",
                "vertical_speed_mps",
                "yaw_rate_radps",
            ):
                frozen_interval = getattr(typed, field)
                interval = getattr(value, field)
                interval.lower = frozen_interval.lower
                interval.upper = frozen_interval.upper
            primitives = []
            for frozen in typed.motion_primitives:
                primitive = bridge_api.LeggedBodyPrimitive()
                primitive.primitive_id = frozen.primitive_id
                primitive.kind = getattr(bridge_api.LeggedPrimitiveKind, frozen.kind)
                primitive.body_frame_displacement_m = _bridge_vec3(
                    frozen.body_frame_displacement_m, bridge_api
                )
                primitive.yaw_change_rad = frozen.yaw_change_rad
                primitive.nominal_duration = _bridge_duration(frozen.nominal_duration_ns)
                primitives.append(primitive)
            value.motion_primitives = primitives
            return value
        value = bridge_api.HopperCapability()
        value.body_half_extent_m = _bridge_vec3(typed.body_half_extent_m, bridge_api)
        value.gravity_mps2 = _bridge_vec3(typed.gravity_mps2, bridge_api)
        for field in (
            "platform_mass_kg",
            "maximum_landing_slope_rad",
            "maximum_landing_roughness_m",
            "maximum_plane_residual_m",
            "minimum_overhead_clearance_m",
            "minimum_lateral_clearance_m",
            "minimum_landing_region_area_m2",
            "maximum_launch_speed_mps",
            "maximum_launch_impulse_newton_seconds",
            "maximum_landing_speed_mps",
            "minimum_downward_impact_speed_mps",
            "minimum_landing_clearance_m",
            "maximum_angular_speed_radps",
            "maximum_angular_acceleration_radps2",
            "maximum_initial_angular_speed_radps",
        ):
            setattr(value, field, getattr(typed, field))
        value.minimum_flight_time = _bridge_duration(typed.minimum_flight_time_ns)
        value.maximum_flight_time = _bridge_duration(typed.maximum_flight_time_ns)
        value.minimum_settle_guard = _bridge_duration(typed.minimum_settle_guard_ns)
        return value


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


@dataclass(frozen=True, slots=True)
class _ParsedContent:
    platform_id: str
    base_frame_id: str
    platform_document_path: str
    observation_document_path: str
    urdf_path: str
    mesh_paths: tuple[str, ...]
    observation_capability: FrozenObservationCapability
    typed_capability: FrozenTypedCapability

    @property
    def resource_references(self) -> tuple[tuple[str, str], ...]:
        return (
            ("document", self.platform_document_path),
            ("document", self.observation_document_path),
            ("urdf", self.urdf_path),
            *(("mesh", path) for path in self.mesh_paths),
        )


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
    _require_exact_object(raw, _LOCK_FIELDS, "capability lock")
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


def _parse_platform_entry(root: Path, entry: object) -> FrozenPlatformCapability:
    platform_entry = _require_exact_object(
        entry, _PLATFORM_FIELDS, "capability platform entry"
    )
    platform_type = platform_entry["platform_type"]
    if platform_type not in PLATFORMS:
        raise CapabilityFreezeError("unknown capability platform type")
    capability_type = platform_entry["capability_type"]
    if capability_type != CAPABILITY_TYPES[platform_type]:
        raise CapabilityFreezeError("capability type is unsupported for platform")
    capability_version = _nonempty_string(
        platform_entry["capability_version"], "capability version"
    )
    content_path = _resolve_relative(
        root, platform_entry["content_path"], "content"
    )
    expected_file_hash = _digest(platform_entry["content_file_sha256"], "content")
    content_bytes = content_path.read_bytes()
    if _sha256_bytes(content_bytes) != expected_file_hash:
        raise CapabilityFreezeError("capability content file hash mismatch")
    content_raw = _decode_json(content_bytes, "capability content")
    _require_exact_object(content_raw, _CONTENT_FIELDS, "capability content")
    if content_raw["schema"] != capability_type:
        raise CapabilityFreezeError("capability content type is unsupported")
    if content_raw["platform_type"] != platform_type:
        raise CapabilityFreezeError("capability content platform mismatch")
    if content_raw["capability_version"] != capability_version:
        raise CapabilityFreezeError("capability content version mismatch")
    parsed_content = _parse_typed_content(content_raw["content"], platform_type)

    resources_raw = platform_entry["resources"]
    if not isinstance(resources_raw, list) or not resources_raw:
        raise CapabilityFreezeError("capability resources must be a non-empty list")
    resources: list[FrozenCapabilityResource] = []
    seen_paths: set[str] = set()
    for raw_resource in resources_raw:
        resource = _require_exact_object(
            raw_resource, _RESOURCE_FIELDS, "capability resource"
        )
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
        resources.append(FrozenCapabilityResource(kind, path, digest))
    expected_references = parsed_content.resource_references
    if len(set(expected_references)) != len(expected_references):
        raise CapabilityFreezeError("capability content resource references must be unique")
    if {(item.kind, item.relative_path) for item in resources} != set(expected_references):
        raise CapabilityFreezeError(
            "capability resource closure does not exactly match content references"
        )
    resources.sort(key=lambda item: (item.relative_path, item.kind))
    canonical_content = {
        "schema": capability_type,
        "platform_type": platform_type,
        "capability_version": capability_version,
        "content": content_raw["content"],
    }
    return FrozenPlatformCapability(
        platform_type=platform_type,
        capability_type=capability_type,
        capability_version=capability_version,
        platform_id=parsed_content.platform_id,
        base_frame_id=parsed_content.base_frame_id,
        platform_document_path=parsed_content.platform_document_path,
        observation_document_path=parsed_content.observation_document_path,
        urdf_path=parsed_content.urdf_path,
        mesh_paths=parsed_content.mesh_paths,
        observation_capability=parsed_content.observation_capability,
        typed_capability=parsed_content.typed_capability,
        content_sha256=_semantic_sha256(canonical_content),
        resources=tuple(resources),
    )


def _parse_typed_content(value: object, platform_type: str) -> _ParsedContent:
    content = _require_exact_object(
        value, _COMMON_CONTENT_FIELDS, "capability content fields"
    )
    platform_id = _nonempty_string(content["platform_id"], "platform_id")
    base_frame_id = _nonempty_string(content["base_frame_id"], "base_frame_id")
    if base_frame_id != "base_link":
        raise CapabilityFreezeError("base_frame_id must be base_link")
    platform_document_path = _normalized_relative(content["platform_document_path"])
    observation_document_path = _normalized_relative(
        content["observation_document_path"]
    )
    urdf_path = _normalized_relative(content["urdf_path"])
    mesh_values = content["mesh_paths"]
    if not isinstance(mesh_values, list) or not mesh_values:
        raise CapabilityFreezeError("capability geometry requires mesh paths")
    mesh_paths = tuple(_normalized_relative(item) for item in mesh_values)
    if len(set(mesh_paths)) != len(mesh_paths):
        raise CapabilityFreezeError("capability mesh paths must be unique")
    observation = _require_exact_object(
        content["observation"], _OBSERVATION_FIELDS, "observation capability"
    )
    sensor_range = _positive(observation["sensor_range_m"], "sensor_range_m")
    sensor_fov_deg = _positive(observation["sensor_fov_deg"], "sensor_fov_deg")
    if sensor_fov_deg > 360.0:
        raise CapabilityFreezeError("sensor_fov_deg exceeds 360 degrees")
    if platform_type == "WHEELED":
        typed = _parse_wheeled(content["capability"])
    elif platform_type == "LEGGED":
        typed = _parse_legged(content["capability"])
    else:
        typed = _parse_hopper(content["capability"])
    return _ParsedContent(
        platform_id=platform_id,
        base_frame_id=base_frame_id,
        platform_document_path=platform_document_path,
        observation_document_path=observation_document_path,
        urdf_path=urdf_path,
        mesh_paths=mesh_paths,
        observation_capability=FrozenObservationCapability(
            sensor_range_m=sensor_range,
            sensor_fov_rad=sensor_fov_deg * math.pi / 180.0,
        ),
        typed_capability=typed,
    )


def _parse_wheeled(value: object) -> FrozenWheeledCapability:
    node = _require_exact_object(value, _WHEELED_FIELDS, "wheeled capability")
    footprint_raw = node["footprint_xy_m"]
    if not isinstance(footprint_raw, list) or len(footprint_raw) < 3:
        raise CapabilityFreezeError("wheeled footprint requires at least three vertices")
    footprint = tuple(
        _vec2(item, f"footprint_xy_m[{index}]")
        for index, item in enumerate(footprint_raw)
    )
    if len({(item.x, item.y) for item in footprint}) != len(footprint):
        raise CapabilityFreezeError("wheeled footprint vertices must be unique")
    minimum_body_z = _finite(node["minimum_body_z_m"], "minimum_body_z_m")
    maximum_body_z = _finite(node["maximum_body_z_m"], "maximum_body_z_m")
    if minimum_body_z > maximum_body_z:
        raise CapabilityFreezeError("minimum_body_z_m exceeds maximum_body_z_m")
    primitives_raw = node["motion_primitives"]
    if not isinstance(primitives_raw, list) or not primitives_raw:
        raise CapabilityFreezeError("wheeled motion primitives must be non-empty")
    primitives: list[FrozenWheelMotionPrimitive] = []
    seen: set[str] = set()
    for index, raw in enumerate(primitives_raw):
        primitive = _require_exact_object(
            raw, _WHEEL_PRIMITIVE_FIELDS, "wheel motion primitive"
        )
        primitive_id = _unique_primitive_id(primitive["primitive_id"], seen)
        kind = _enum_string(primitive["kind"], _WHEEL_KINDS, "wheel primitive kind")
        pose_node = _require_exact_object(
            primitive["relative_end_pose"], _POSE_FIELDS, "wheel relative end pose"
        )
        primitives.append(
            FrozenWheelMotionPrimitive(
                primitive_id=primitive_id,
                kind=kind,
                relative_end_pose=FrozenPose3(
                    position_m=_vec3(
                        pose_node["position_m"],
                        f"motion_primitives[{index}].position_m",
                    ),
                    orientation=_quaternion(
                        pose_node["orientation_wxyz"],
                        f"motion_primitives[{index}].orientation_wxyz",
                    ),
                ),
                nominal_duration_ns=_duration_ns(
                    primitive["nominal_duration_s"],
                    "motion_primitives.nominal_duration_s",
                ),
            )
        )
    return FrozenWheeledCapability(
        footprint_xy_m=footprint,
        minimum_body_z_m=minimum_body_z,
        maximum_body_z_m=maximum_body_z,
        maximum_obstacle_height_m=_nonnegative(
            node["maximum_obstacle_height_m"], "maximum_obstacle_height_m"
        ),
        maximum_forward_speed_mps=_positive(
            node["maximum_forward_speed_mps"], "maximum_forward_speed_mps"
        ),
        maximum_reverse_speed_mps=_nonnegative(
            node["maximum_reverse_speed_mps"], "maximum_reverse_speed_mps"
        ),
        maximum_spin_rate_radps=_positive(
            node["maximum_spin_rate_radps"], "maximum_spin_rate_radps"
        ),
        maximum_acceleration_mps2=_positive(
            node["maximum_acceleration_mps2"], "maximum_acceleration_mps2"
        ),
        maximum_braking_deceleration_mps2=_positive(
            node["maximum_braking_deceleration_mps2"],
            "maximum_braking_deceleration_mps2",
        ),
        maximum_yaw_acceleration_radps2=_positive(
            node["maximum_yaw_acceleration_radps2"],
            "maximum_yaw_acceleration_radps2",
        ),
        maximum_lateral_acceleration_mps2=_positive(
            node["maximum_lateral_acceleration_mps2"],
            "maximum_lateral_acceleration_mps2",
        ),
        maximum_curvature_per_m=_positive(
            node["maximum_curvature_per_m"], "maximum_curvature_per_m"
        ),
        maximum_slope_rad=_slope(node["maximum_slope_rad"], "maximum_slope_rad"),
        minimum_clearance_m=_nonnegative(
            node["minimum_clearance_m"], "minimum_clearance_m"
        ),
        motion_primitives=tuple(primitives),
    )


def _parse_legged(value: object) -> FrozenLeggedCapability:
    node = _require_exact_object(value, _LEGGED_FIELDS, "legged capability")
    extent = _vec3(node["body_half_extent_m"], "body_half_extent_m")
    if min(extent.x, extent.y, extent.z) <= 0.0:
        raise CapabilityFreezeError("body_half_extent_m must be positive")
    primitives_raw = node["motion_primitives"]
    if not isinstance(primitives_raw, list) or not primitives_raw:
        raise CapabilityFreezeError("legged motion primitives must be non-empty")
    primitives: list[FrozenLeggedBodyPrimitive] = []
    seen: set[str] = set()
    for raw in primitives_raw:
        primitive = _require_exact_object(
            raw, _LEGGED_PRIMITIVE_FIELDS, "legged motion primitive"
        )
        primitives.append(
            FrozenLeggedBodyPrimitive(
                primitive_id=_unique_primitive_id(primitive["primitive_id"], seen),
                kind=_enum_string(
                    primitive["kind"], _LEGGED_KINDS, "legged primitive kind"
                ),
                body_frame_displacement_m=_vec3(
                    primitive["body_frame_displacement_m"],
                    "motion_primitives.body_frame_displacement_m",
                ),
                yaw_change_rad=_finite(
                    primitive["yaw_change_rad"], "motion_primitives.yaw_change_rad"
                ),
                nominal_duration_ns=_duration_ns(
                    primitive["nominal_duration_s"],
                    "motion_primitives.nominal_duration_s",
                ),
            )
        )
    body_height = _interval(node["body_height_m"], "body_height_m")
    if body_height.lower < 0.0:
        raise CapabilityFreezeError("body_height_m must be non-negative")
    return FrozenLeggedCapability(
        reference_point=_nonempty_string(node["reference_point"], "reference_point"),
        body_half_extent_m=extent,
        maximum_slope_rad=_slope(node["maximum_slope_rad"], "maximum_slope_rad"),
        maximum_roughness_m=_nonnegative(
            node["maximum_roughness_m"], "maximum_roughness_m"
        ),
        maximum_step_height_m=_nonnegative(
            node["maximum_step_height_m"], "maximum_step_height_m"
        ),
        maximum_gap_width_m=_nonnegative(
            node["maximum_gap_width_m"], "maximum_gap_width_m"
        ),
        minimum_confidence=_unit_interval(
            node["minimum_confidence"], "minimum_confidence"
        ),
        minimum_body_clearance_m=_nonnegative(
            node["minimum_body_clearance_m"], "minimum_body_clearance_m"
        ),
        body_height_m=body_height,
        forward_speed_mps=_interval(
            node["forward_speed_mps"], "forward_speed_mps", require_zero=True
        ),
        lateral_speed_mps=_interval(
            node["lateral_speed_mps"], "lateral_speed_mps", require_zero=True
        ),
        vertical_speed_mps=_interval(
            node["vertical_speed_mps"], "vertical_speed_mps", require_zero=True
        ),
        yaw_rate_radps=_interval(
            node["yaw_rate_radps"], "yaw_rate_radps", require_zero=True
        ),
        maximum_linear_acceleration_mps2=_positive(
            node["maximum_linear_acceleration_mps2"],
            "maximum_linear_acceleration_mps2",
        ),
        maximum_yaw_acceleration_radps2=_positive(
            node["maximum_yaw_acceleration_radps2"],
            "maximum_yaw_acceleration_radps2",
        ),
        motion_primitives=tuple(primitives),
    )


def _parse_hopper(value: object) -> FrozenHopperCapability:
    node = _require_exact_object(value, _HOPPER_FIELDS, "hopper capability")
    extent = _vec3(node["body_half_extent_m"], "body_half_extent_m")
    if min(extent.x, extent.y, extent.z) <= 0.0:
        raise CapabilityFreezeError("body_half_extent_m must be positive")
    gravity = _vec3(node["gravity_mps2"], "gravity_mps2")
    if gravity.z >= 0.0 or math.hypot(gravity.x, gravity.y, gravity.z) <= 0.0:
        raise CapabilityFreezeError("gravity_mps2 must point downward")
    minimum_flight = _duration_ns(node["minimum_flight_time_s"], "minimum_flight_time_s")
    maximum_flight = _duration_ns(node["maximum_flight_time_s"], "maximum_flight_time_s")
    if minimum_flight > maximum_flight:
        raise CapabilityFreezeError("minimum_flight_time_s exceeds maximum_flight_time_s")
    profile = _require_exact_object(
        node["actuator_or_impulse_profile"], _PROFILE_FIELDS, "actuator profile"
    )
    primitives_raw = node["motion_primitives"]
    if not isinstance(primitives_raw, list) or not primitives_raw:
        raise CapabilityFreezeError("hopper motion primitives must be non-empty")
    primitive_ids: list[str] = []
    seen: set[str] = set()
    for raw in primitives_raw:
        primitive = _require_exact_object(
            raw, _HOPPER_PRIMITIVE_FIELDS, "hopper motion primitive"
        )
        primitive_ids.append(_unique_primitive_id(primitive["primitive_id"], seen))
    return FrozenHopperCapability(
        body_half_extent_m=extent,
        platform_mass_kg=_positive(node["platform_mass_kg"], "platform_mass_kg"),
        gravity_mps2=gravity,
        maximum_landing_slope_rad=_slope(
            node["maximum_landing_slope_rad"], "maximum_landing_slope_rad"
        ),
        maximum_landing_roughness_m=_nonnegative(
            node["maximum_landing_roughness_m"], "maximum_landing_roughness_m"
        ),
        maximum_plane_residual_m=_nonnegative(
            node["maximum_plane_residual_m"], "maximum_plane_residual_m"
        ),
        minimum_overhead_clearance_m=_nonnegative(
            node["minimum_overhead_clearance_m"], "minimum_overhead_clearance_m"
        ),
        minimum_lateral_clearance_m=_nonnegative(
            node["minimum_lateral_clearance_m"], "minimum_lateral_clearance_m"
        ),
        minimum_landing_region_area_m2=_positive(
            node["minimum_landing_region_area_m2"], "minimum_landing_region_area_m2"
        ),
        maximum_launch_speed_mps=_positive(
            node["maximum_launch_speed_mps"], "maximum_launch_speed_mps"
        ),
        maximum_launch_impulse_newton_seconds=_positive(
            node["maximum_launch_impulse_newton_seconds"],
            "maximum_launch_impulse_newton_seconds",
        ),
        minimum_flight_time_ns=minimum_flight,
        maximum_flight_time_ns=maximum_flight,
        maximum_landing_speed_mps=_positive(
            node["maximum_landing_speed_mps"], "maximum_landing_speed_mps"
        ),
        minimum_downward_impact_speed_mps=_nonnegative(
            node["minimum_downward_impact_speed_mps"],
            "minimum_downward_impact_speed_mps",
        ),
        minimum_landing_clearance_m=_nonnegative(
            node["minimum_landing_clearance_m"], "minimum_landing_clearance_m"
        ),
        maximum_angular_speed_radps=_positive(
            node["maximum_angular_speed_radps"], "maximum_angular_speed_radps"
        ),
        maximum_angular_acceleration_radps2=_positive(
            node["maximum_angular_acceleration_radps2"],
            "maximum_angular_acceleration_radps2",
        ),
        maximum_initial_angular_speed_radps=_nonnegative(
            node["maximum_initial_angular_speed_radps"],
            "maximum_initial_angular_speed_radps",
        ),
        minimum_settle_guard_ns=_duration_ns(
            node["minimum_settle_guard_s"], "minimum_settle_guard_s", allow_zero=True
        ),
        actuator_profile_id=_nonempty_string(profile["profile_id"], "profile_id"),
        source_motion_primitive_ids=tuple(primitive_ids),
    )


def _require_exact_object(
    value: object, fields: frozenset[str], name: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise CapabilityFreezeError(f"{name} schema fields are invalid")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CapabilityFreezeError(f"{field} must be a non-empty string")
    return value


def _finite(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise CapabilityFreezeError(f"{field} must be a finite number")
    return float(value)


def _positive(value: object, field: str) -> float:
    parsed = _finite(value, field)
    if parsed <= 0.0:
        raise CapabilityFreezeError(f"{field} must be positive")
    return parsed


def _nonnegative(value: object, field: str) -> float:
    parsed = _finite(value, field)
    if parsed < 0.0:
        raise CapabilityFreezeError(f"{field} must be non-negative")
    return parsed


def _unit_interval(value: object, field: str) -> float:
    parsed = _finite(value, field)
    if not 0.0 <= parsed <= 1.0:
        raise CapabilityFreezeError(f"{field} must be in [0, 1]")
    return parsed


def _slope(value: object, field: str) -> float:
    parsed = _finite(value, field)
    if parsed <= 0.0 or parsed >= math.pi / 2.0:
        raise CapabilityFreezeError(f"{field} is outside the loader slope range")
    return min(parsed, _PROJECT_MAXIMUM_SLOPE_RAD)


def _vec2(value: object, field: str) -> FrozenVec2:
    if not isinstance(value, list) or len(value) != 2:
        raise CapabilityFreezeError(f"{field} must be vec2")
    return FrozenVec2(_finite(value[0], f"{field}[0]"), _finite(value[1], f"{field}[1]"))


def _vec3(value: object, field: str) -> FrozenVec3:
    if not isinstance(value, list) or len(value) != 3:
        raise CapabilityFreezeError(f"{field} must be vec3")
    return FrozenVec3(
        _finite(value[0], f"{field}[0]"),
        _finite(value[1], f"{field}[1]"),
        _finite(value[2], f"{field}[2]"),
    )


def _quaternion(value: object, field: str) -> FrozenQuaternion:
    if not isinstance(value, list) or len(value) != 4:
        raise CapabilityFreezeError(f"{field} must be quaternion wxyz")
    items = tuple(_finite(item, f"{field}[{index}]") for index, item in enumerate(value))
    norm = math.sqrt(sum(item * item for item in items))
    if not math.isfinite(norm) or abs(norm - 1.0) > 1.0e-3:
        raise CapabilityFreezeError(f"{field} quaternion must be unit length")
    return FrozenQuaternion(*(item / norm for item in items))


def _interval(
    value: object, field: str, *, require_zero: bool = False
) -> FrozenInterval:
    if not isinstance(value, list) or len(value) != 2:
        raise CapabilityFreezeError(f"{field} must be an interval")
    lower = _finite(value[0], f"{field}[0]")
    upper = _finite(value[1], f"{field}[1]")
    if lower > upper or (
        require_zero and (lower >= upper or lower > 0.0 or upper < 0.0)
    ):
        raise CapabilityFreezeError(f"{field} interval bounds are invalid")
    return FrozenInterval(lower, upper)


def _duration_ns(value: object, field: str, *, allow_zero: bool = False) -> int:
    seconds = _finite(value, field)
    if seconds < 0.0 or (seconds == 0.0 and not allow_zero):
        raise CapabilityFreezeError(f"{field} duration is invalid")
    nanoseconds = seconds * 1_000_000_000.0
    if nanoseconds > _INT64_MAX:
        raise CapabilityFreezeError(f"{field} duration exceeds int64 nanoseconds")
    return round(nanoseconds)


def _enum_string(value: object, allowed: frozenset[str], field: str) -> str:
    parsed = _nonempty_string(value, field)
    if parsed not in allowed:
        raise CapabilityFreezeError(f"{field} is unknown")
    return parsed


def _unique_primitive_id(value: object, seen: set[str]) -> str:
    primitive_id = _nonempty_string(value, "motion primitive id")
    if primitive_id in seen:
        raise CapabilityFreezeError("duplicate motion primitive id")
    seen.add(primitive_id)
    return primitive_id


def _bridge_vec2(value: FrozenVec2, bridge_api):
    result = bridge_api.Vec2()
    result.x = value.x
    result.y = value.y
    return result


def _bridge_vec3(value: FrozenVec3, bridge_api):
    result = bridge_api.Vec3()
    result.x = value.x
    result.y = value.y
    result.z = value.z
    return result


def _bridge_pose(value: FrozenPose3, bridge_api):
    result = bridge_api.Pose3()
    result.position_m = _bridge_vec3(value.position_m, bridge_api)
    orientation = bridge_api.Quaternion()
    orientation.w = value.orientation.w
    orientation.x = value.orientation.x
    orientation.y = value.orientation.y
    orientation.z = value.orientation.z
    result.orientation = orientation
    return result


def _bridge_duration(nanoseconds: int) -> timedelta:
    return timedelta(seconds=nanoseconds / 1_000_000_000.0)


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
        raise CapabilityFreezeError(
            f"capability {name} path escapes closure or is missing"
        ) from error
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


__all__ = [
    "CAPABILITY_FREEZE_SCHEMA",
    "CAPABILITY_TYPES",
    "CapabilityFreezeError",
    "FrozenCapabilityBundle",
    "FrozenCapabilityEnvironmentFactory",
    "FrozenCapabilityResource",
    "FrozenHopperCapability",
    "FrozenInterval",
    "FrozenLeggedBodyPrimitive",
    "FrozenLeggedCapability",
    "FrozenObservationCapability",
    "FrozenPlatformCapability",
    "FrozenPose3",
    "FrozenQuaternion",
    "FrozenTypedCapability",
    "FrozenVec2",
    "FrozenVec3",
    "FrozenWheelMotionPrimitive",
    "FrozenWheeledCapability",
    "RUN_KINDS",
    "ScenarioIdentity",
    "load_frozen_capability_bundle",
]
