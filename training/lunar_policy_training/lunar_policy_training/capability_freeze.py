"""Typed capabilities plus legacy external-bundle compatibility for smoke tests."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeAlias

import yaml

from .training_semantics import (
    FORMAL_SENSOR_FOV_RAD,
    FORMAL_SENSOR_RANGE_M,
)


CAPABILITY_FREEZE_SCHEMA = "lunar-training-capability-freeze/v1"
_PLATFORM_SOURCE_SCHEMA = "platform-control-capability-source/v2"
PLATFORMS = ("WHEELED", "LEGGED", "HOPPER")
RUN_KINDS = ("formal", "development-smoke")
CAPABILITY_TYPES = {
    "WHEELED": "lunar-planner-wheeled-capability/v2",
    "LEGGED": "lunar-planner-legged-capability/v2",
    "HOPPER": "lunar-planner-hopper-capability/v2",
}
_PROJECT_MAXIMUM_SLOPE_RAD = math.pi / 6.0
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
    ("primitive_id", "kind", "relative_end_pose")
)
_LEGGED_PRIMITIVE_FIELDS = frozenset(
    (
        "primitive_id",
        "kind",
        "body_frame_displacement_m",
        "yaw_change_rad",
    )
)
_SOURCE_PLATFORM_FIELDS = frozenset(
    ("platform_id", "platform_type", "capability_version", "base_frame_id")
)
_GEOMETRY_SOURCE_FIELDS = frozenset(("urdf_file",))
_WHEELED_FIELDS = frozenset(
    (
        "reference_point",
        "footprint_xy_m",
        "body_extent_m",
        "wheel_diameter_m",
        "wheel_width_m",
        "wheelbase_m",
        "track_width_m",
        "minimum_underbody_clearance_m",
        "maximum_local_obstacle_relief_m",
        "allow_unsupported_gap",
        "maximum_slope_rad",
        "maximum_forward_speed_mps",
        "maximum_reverse_speed_mps",
        "maximum_spin_rate_radps",
        "maximum_acceleration_mps2",
        "maximum_braking_deceleration_mps2",
        "maximum_yaw_acceleration_radps2",
        "maximum_lateral_acceleration_mps2",
        "maximum_curvature_per_m",
        "minimum_clearance_m",
        "roughness_handling",
        "motion_primitives",
    )
)
_LEGACY_LEGGED_FIELDS = frozenset(
    (
        "reference_point",
        "body_extent_m",
        "platform_mass_kg",
        "maximum_payload_kg",
        "maximum_slope_rad",
        "maximum_step_height_m",
        "maximum_gap_width_m",
        "minimum_body_clearance_m",
        "step_vertical_rate_mps",
        "body_height_m",
        "forward_speed_mps",
        "lateral_speed_mps",
        "yaw_rate_radps",
        "maximum_linear_acceleration_mps2",
        "maximum_yaw_acceleration_radps2",
        "roughness_handling",
        "motion_primitives",
    )
)
_LEGGED_METADATA_FIELDS = frozenset(
    ("nominal_body_height_m", "nominal_payload_kg", "unknown_is_traversable")
)
_LEGGED_FIELDS = _LEGACY_LEGGED_FIELDS | _LEGGED_METADATA_FIELDS
# A legacy bundle did not claim a nominal payload. Zero preserves that absence
# without overstating payload capability; unknown terrain remains fail-closed.
_LEGACY_NOMINAL_PAYLOAD_KG = 0.0
_LEGACY_UNKNOWN_IS_TRAVERSABLE = False
_LEGACY_HOPPER_FIELDS = frozenset(
    (
        "specific_impulse_s",
        "reference_total_mass_kg",
        "reference_propellant_mass_kg",
        "landing_support_radius_m",
        "flight_collision_radius_m",
        "maximum_landing_plane_residual_m",
        "landing_lateral_margin_m",
        "flight_map_margin_m",
        "reachability_delta_v_margin_ratio",
        "standard_gravity_mps2",
        "maximum_landing_slope_rad",
    )
)
_HOPPER_FIELDS = _LEGACY_HOPPER_FIELDS | frozenset(
    (
        "gravity_mps2",
        "reference_horizontal_range_m",
        "reference_elevation_delta_m",
        "runtime_fallback_allowed",
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
    """A capability payload is incomplete, mutable, or mislabeled."""


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


@dataclass(frozen=True, slots=True)
class FrozenWheeledCapability:
    reference_point: str
    footprint_xy_m: tuple[FrozenVec2, ...]
    body_extent_m: FrozenVec3
    wheel_diameter_m: float
    wheel_width_m: float
    wheelbase_m: float
    track_width_m: float
    minimum_underbody_clearance_m: float
    maximum_local_obstacle_relief_m: float
    allow_unsupported_gap: bool
    minimum_body_z_m: float
    maximum_body_z_m: float
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


@dataclass(frozen=True, slots=True)
class FrozenLeggedCapability:
    reference_point: str
    body_extent_m: FrozenVec3
    nominal_body_height_m: float
    platform_mass_kg: float
    nominal_payload_kg: float
    maximum_payload_kg: float
    maximum_slope_rad: float
    maximum_step_height_m: float
    maximum_gap_width_m: float
    minimum_body_clearance_m: float
    step_vertical_rate_mps: float
    body_height_m: FrozenInterval
    forward_speed_mps: FrozenInterval
    lateral_speed_mps: FrozenInterval
    yaw_rate_radps: FrozenInterval
    maximum_linear_acceleration_mps2: float
    maximum_yaw_acceleration_radps2: float
    unknown_is_traversable: bool
    motion_primitives: tuple[FrozenLeggedBodyPrimitive, ...]


@dataclass(frozen=True, slots=True)
class FrozenHopperCapability:
    specific_impulse_s: float
    reference_total_mass_kg: float
    reference_propellant_mass_kg: float
    landing_support_radius_m: float
    flight_collision_radius_m: float
    maximum_landing_plane_residual_m: float
    landing_lateral_margin_m: float
    flight_map_margin_m: float
    reachability_delta_v_margin_ratio: float
    standard_gravity_mps2: float
    maximum_landing_slope_rad: float
    gravity_mps2: FrozenVec3 = FrozenVec3(0.0, 0.0, -1.62)
    reference_horizontal_range_m: float = 100.0
    reference_elevation_delta_m: float = 0.0
    runtime_fallback_allowed: bool = False


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
    def nominal_global_cost_per_m(self) -> float:
        """Derived base route cost used by candidate cost normalization."""
        return 1.0

    @property
    def source_motion_primitive_ids(self) -> tuple[str, ...]:
        typed = self.typed_capability
        if isinstance(typed, FrozenHopperCapability):
            return ()
        return tuple(item.primitive_id for item in typed.motion_primitives)

    def to_bridge_capability(self) -> object:
        """Create the C++ v3 bridge value from the parsed in-memory payload."""
        import lunar_planner_training_bridge as bridge_api

        typed = self.typed_capability
        if isinstance(typed, FrozenWheeledCapability):
            value = bridge_api.WheeledCapability()
            value.footprint_xy_m = [
                _bridge_vec2(item, bridge_api) for item in typed.footprint_xy_m
            ]
            value.body_extent_m = _bridge_vec3(typed.body_extent_m, bridge_api)
            for field in (
                "wheel_diameter_m",
                "wheel_width_m",
                "wheelbase_m",
                "track_width_m",
                "minimum_underbody_clearance_m",
                "maximum_local_obstacle_relief_m",
                "allow_unsupported_gap",
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
                primitive.relative_end_pose = _bridge_pose(
                    frozen.relative_end_pose, bridge_api
                )
                primitives.append(primitive)
            value.motion_primitives = primitives
            return value
        if isinstance(typed, FrozenLeggedCapability):
            value = bridge_api.LeggedCapability()
            value.body_extent_m = _bridge_vec3(typed.body_extent_m, bridge_api)
            for field in (
                "nominal_body_height_m",
                "platform_mass_kg",
                "nominal_payload_kg",
                "maximum_payload_kg",
                "maximum_slope_rad",
                "maximum_step_height_m",
                "maximum_gap_width_m",
                "minimum_body_clearance_m",
                "step_vertical_rate_mps",
                "maximum_linear_acceleration_mps2",
                "maximum_yaw_acceleration_radps2",
                "unknown_is_traversable",
            ):
                setattr(value, field, getattr(typed, field))
            for field in (
                "body_height_m",
                "forward_speed_mps",
                "lateral_speed_mps",
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
                primitives.append(primitive)
            value.motion_primitives = primitives
            return value
        value = bridge_api.HopperCapability()
        for field in (
            "specific_impulse_s",
            "reference_total_mass_kg",
            "reference_propellant_mass_kg",
            "reference_horizontal_range_m",
            "reference_elevation_delta_m",
            "runtime_fallback_allowed",
            "landing_support_radius_m",
            "flight_collision_radius_m",
            "maximum_landing_plane_residual_m",
            "landing_lateral_margin_m",
            "flight_map_margin_m",
            "reachability_delta_v_margin_ratio",
            "standard_gravity_mps2",
            "maximum_landing_slope_rad",
        ):
            setattr(value, field, getattr(typed, field))
        value.gravity_mps2 = _bridge_vec3(typed.gravity_mps2, bridge_api)
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
    episode_cursor: int
    platform_worker_index: int
    platform_worker_count: int
    capability_version: str
    capability_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenCapabilityEnvironmentFactory:
    """Bind one validated capability bundle to formal sensor-closed workers."""

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
        if getattr(self.builder, "sensor_closed_loop", None) is not True:
            raise CapabilityFreezeError(
                "formal environment builder must declare the sensor-closed loop"
            )

    def __call__(self, worker_index: int, platform_type: str) -> object:
        return self.create_for_episode(worker_index, platform_type, 0)

    def task_inventory_count(self, platform_type: str) -> int:
        provider = getattr(self.builder, "task_inventory_count", None)
        if not callable(provider):
            raise CapabilityFreezeError(
                "formal environment builder does not publish task inventory"
            )
        count = provider(platform_type)
        if type(count) is not int or count <= 0:
            raise CapabilityFreezeError("formal task inventory is invalid")
        return count

    def create_for_episode(
        self,
        worker_index: int,
        platform_type: str,
        episode_cursor: int,
        *,
        platform_worker_index: int | None = None,
        platform_worker_count: int | None = None,
    ) -> object:
        if type(worker_index) is not int or worker_index < 0:
            raise CapabilityFreezeError("worker index must be non-negative")
        if type(episode_cursor) is not int or episode_cursor < 0:
            raise CapabilityFreezeError("episode cursor must be non-negative")
        if platform_worker_index is None:
            platform_worker_index = worker_index
        if platform_worker_count is None:
            platform_worker_count = platform_worker_index + 1
        if (
            type(platform_worker_index) is not int
            or type(platform_worker_count) is not int
            or platform_worker_index < 0
            or platform_worker_count <= platform_worker_index
        ):
            raise CapabilityFreezeError("platform worker lane is invalid")
        capability = self.bundle.for_platform(platform_type)
        scenario = ScenarioIdentity(
            platform_type=platform_type,
            scenario_schedule_id=self.scenario_schedule_id,
            worker_index=worker_index,
            episode_cursor=episode_cursor,
            platform_worker_index=platform_worker_index,
            platform_worker_count=platform_worker_count,
            capability_version=capability.capability_version,
            capability_sha256=capability.content_sha256,
        )
        worker = self.builder(worker_index, platform_type, capability, scenario)
        environment = getattr(worker, "environment", None)
        if getattr(environment, "sensor_closed_loop", None) is not True:
            raise CapabilityFreezeError(
                "formal environment must use the sensor-closed observation loop"
            )
        return worker

    def restore_for_episode(
        self,
        *,
        worker_index: int,
        platform_type: str,
        episode_cursor: int,
        platform_worker_index: int,
        platform_worker_count: int,
        state: object,
    ) -> object:
        """Restore one active formal worker through its bound replay builder."""
        restore = getattr(self.builder, "restore", None)
        if not callable(restore):
            raise CapabilityFreezeError(
                "formal environment builder does not support active restore"
            )
        if (
            type(worker_index) is not int
            or worker_index < 0
            or type(episode_cursor) is not int
            or episode_cursor < 0
            or type(platform_worker_index) is not int
            or type(platform_worker_count) is not int
            or platform_worker_index < 0
            or platform_worker_count <= platform_worker_index
        ):
            raise CapabilityFreezeError("formal restore worker identity is invalid")
        capability = self.bundle.for_platform(platform_type)
        scenario = ScenarioIdentity(
            platform_type=platform_type,
            scenario_schedule_id=self.scenario_schedule_id,
            worker_index=worker_index,
            episode_cursor=episode_cursor,
            platform_worker_index=platform_worker_index,
            platform_worker_count=platform_worker_count,
            capability_version=capability.capability_version,
            capability_sha256=capability.content_sha256,
        )
        worker = restore(
            worker_index,
            platform_type,
            capability,
            scenario,
            state,
        )
        if getattr(getattr(worker, "environment", None), "sensor_closed_loop", None) is not True:
            raise CapabilityFreezeError(
                "restored formal environment must use the sensor-closed loop"
            )
        return worker


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
    """Load the legacy external-bundle format used by compatibility tests.

    Production formal commands use ``load_project_formal_capability`` instead;
    this parser remains available for development-smoke and migration evidence.
    """
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
    if run_kind == "formal":
        expected_observation = FrozenObservationCapability(
            sensor_range_m=FORMAL_SENSOR_RANGE_M,
            sensor_fov_rad=FORMAL_SENSOR_FOV_RAD,
        )
        if any(
            platform.observation_capability != expected_observation
            for platform in ordered
        ):
            raise CapabilityFreezeError(
                "formal observation capability must be shared 30 m/360 degrees"
            )
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
    resource_payloads: dict[tuple[str, str], bytes] = {}
    seen_paths: set[str] = set()
    for raw_resource in resources_raw:
        resource = _require_exact_object(
            raw_resource, _RESOURCE_FIELDS, "capability resource"
        )
        kind = resource["kind"]
        if not isinstance(kind, str) or kind not in _RESOURCE_KINDS:
            raise CapabilityFreezeError("capability resource kind is unsupported")
        path = _normalized_relative(resource["path"])
        if path in seen_paths:
            raise CapabilityFreezeError("duplicate capability resource path")
        seen_paths.add(path)
        resolved = _resolve_relative(root, path, "resource")
        digest = _digest(resource["sha256"], "resource")
        payload = resolved.read_bytes()
        if _sha256_bytes(payload) != digest:
            raise CapabilityFreezeError("capability resource hash mismatch")
        resource_payloads[(kind, path)] = payload
        resources.append(FrozenCapabilityResource(kind, path, digest))
    expected_references = parsed_content.resource_references
    if len(set(expected_references)) != len(expected_references):
        raise CapabilityFreezeError("capability content resource references must be unique")
    if {(item.kind, item.relative_path) for item in resources} != set(expected_references):
        raise CapabilityFreezeError(
            "capability resource closure does not exactly match content references"
        )
    _validate_authoritative_sources(
        content=content_raw["content"],
        parsed=parsed_content,
        platform_type=platform_type,
        capability_version=capability_version,
        resource_payloads=resource_payloads,
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
    expected_base_frame = (
        "base_footprint" if platform_type == "WHEELED" else "base_link"
    )
    if base_frame_id != expected_base_frame:
        raise CapabilityFreezeError(
            f"base_frame_id must be {expected_base_frame} for {platform_type}"
        )
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


def _validate_authoritative_sources(
    *,
    content: object,
    parsed: _ParsedContent,
    platform_type: str,
    capability_version: str,
    resource_payloads: Mapping[tuple[str, str], bytes],
) -> None:
    content_object = _require_exact_object(
        content, _COMMON_CONTENT_FIELDS, "capability content fields"
    )
    platform_document = _yaml_object(
        resource_payloads[("document", parsed.platform_document_path)],
        "platform document",
    )
    platform_key = platform_type.lower()
    _require_exact_object(
        platform_document,
        frozenset(
            ("schema_version", "platform", "geometry_source", platform_key)
        ),
        "platform document",
    )
    if platform_document["schema_version"] != _PLATFORM_SOURCE_SCHEMA:
        raise CapabilityFreezeError("platform document schema is unsupported")
    source_platform = _require_exact_object(
        platform_document["platform"],
        _SOURCE_PLATFORM_FIELDS,
        "platform document platform",
    )
    expected_common = {
        "platform_id": parsed.platform_id,
        "platform_type": platform_type,
        "capability_version": capability_version,
        "base_frame_id": parsed.base_frame_id,
    }
    if source_platform != expected_common:
        raise CapabilityFreezeError(
            "platform document common identity mismatch"
        )
    geometry = _require_exact_object(
        platform_document["geometry_source"],
        _GEOMETRY_SOURCE_FIELDS,
        "platform document geometry source",
    )
    if _normalized_relative(geometry["urdf_file"]) != parsed.urdf_path:
        raise CapabilityFreezeError("platform document URDF path mismatch")
    source_capability = platform_document[platform_key]
    if source_capability != content_object["capability"]:
        raise CapabilityFreezeError(
            "platform document typed capability mismatch"
        )
    if platform_type == "WHEELED":
        source_typed = _parse_wheeled(source_capability)
    elif platform_type == "LEGGED":
        source_typed = _parse_legged(source_capability)
    else:
        source_typed = _parse_hopper(source_capability)
    if source_typed != parsed.typed_capability:
        raise CapabilityFreezeError(
            "platform document parsed capability mismatch"
        )

    observation_document = _yaml_object(
        resource_payloads[("document", parsed.observation_document_path)],
        "observation document",
    )
    observation = _require_exact_object(
        observation_document,
        _OBSERVATION_FIELDS,
        "observation document",
    )
    if observation != content_object["observation"]:
        raise CapabilityFreezeError("observation document mismatch")
    sensor_range = _positive(observation["sensor_range_m"], "sensor_range_m")
    sensor_fov_deg = _positive(observation["sensor_fov_deg"], "sensor_fov_deg")
    if sensor_fov_deg > 360.0:
        raise CapabilityFreezeError("sensor_fov_deg exceeds 360 degrees")
    source_observation = FrozenObservationCapability(
        sensor_range_m=sensor_range,
        sensor_fov_rad=sensor_fov_deg * math.pi / 180.0,
    )
    if source_observation != parsed.observation_capability:
        raise CapabilityFreezeError("observation document parsed value mismatch")

    urdf_mesh_paths = _parse_urdf_mesh_paths(
        resource_payloads[("urdf", parsed.urdf_path)],
        urdf_path=parsed.urdf_path,
        base_frame_id=parsed.base_frame_id,
    )
    if set(urdf_mesh_paths) != set(parsed.mesh_paths):
        raise CapabilityFreezeError(
            "URDF mesh closure does not match capability content"
        )


def _yaml_object(data: bytes, name: str) -> dict[str, object]:
    try:
        value = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise CapabilityFreezeError(f"{name} is not valid UTF-8 YAML") from error
    if not isinstance(value, dict):
        raise CapabilityFreezeError(f"{name} must be a YAML object")
    return value


def _parse_urdf_mesh_paths(
    data: bytes,
    *,
    urdf_path: str,
    base_frame_id: str,
) -> tuple[str, ...]:
    try:
        document = data.decode("utf-8")
    except UnicodeError as error:
        raise CapabilityFreezeError("URDF is not valid UTF-8") from error
    try:
        from lunar_planner_training_bridge import validate_urdf_geometry
    except ImportError as error:
        raise CapabilityFreezeError("URDF validator is unavailable") from error

    try:
        mesh_filenames = validate_urdf_geometry(document, base_frame_id)
    except (RuntimeError, ValueError) as error:
        raise CapabilityFreezeError("URDF model or geometry is invalid") from error
    mesh_paths: set[str] = set()
    urdf_parent = PurePosixPath(urdf_path).parent
    for filename in mesh_filenames:
        relative_mesh = _normalized_relative(filename)
        closure_path = _normalized_relative(
            (urdf_parent / PurePosixPath(relative_mesh)).as_posix()
        )
        mesh_paths.add(closure_path)
    return tuple(sorted(mesh_paths))


def _parse_wheeled(value: object) -> FrozenWheeledCapability:
    node = _require_exact_object(value, _WHEELED_FIELDS, "wheeled capability")
    reference_point = _nonempty_string(node["reference_point"], "reference_point")
    if reference_point != "base_footprint":
        raise CapabilityFreezeError("wheeled reference_point must be base_footprint")
    if node["roughness_handling"] != "COST_SPEED_AND_LOCAL_RECHECK":
        raise CapabilityFreezeError("wheeled roughness_handling is unsupported")
    footprint_raw = node["footprint_xy_m"]
    if not isinstance(footprint_raw, list) or len(footprint_raw) < 3:
        raise CapabilityFreezeError("wheeled footprint requires at least three vertices")
    footprint = tuple(
        _vec2(item, f"footprint_xy_m[{index}]")
        for index, item in enumerate(footprint_raw)
    )
    if len({(item.x, item.y) for item in footprint}) != len(footprint):
        raise CapabilityFreezeError("wheeled footprint vertices must be unique")
    body_extent = _vec3(node["body_extent_m"], "body_extent_m")
    if min(body_extent.x, body_extent.y, body_extent.z) <= 0.0:
        raise CapabilityFreezeError("body_extent_m must be positive")
    wheel_width = _positive(node["wheel_width_m"], "wheel_width_m")
    wheelbase = _positive(node["wheelbase_m"], "wheelbase_m")
    track_width = _positive(node["track_width_m"], "track_width_m")
    if (
        wheel_width >= body_extent.y
        or wheelbase >= body_extent.x
        or abs(track_width - (body_extent.y - wheel_width)) > 1.0e-6
    ):
        raise CapabilityFreezeError("wheeled wheel geometry is inconsistent")
    allow_unsupported_gap = node["allow_unsupported_gap"]
    if type(allow_unsupported_gap) is not bool:
        raise CapabilityFreezeError("allow_unsupported_gap must be boolean")
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
            )
        )
    return FrozenWheeledCapability(
        reference_point=reference_point,
        footprint_xy_m=footprint,
        body_extent_m=body_extent,
        wheel_diameter_m=_positive(node["wheel_diameter_m"], "wheel_diameter_m"),
        wheel_width_m=wheel_width,
        wheelbase_m=wheelbase,
        track_width_m=track_width,
        minimum_underbody_clearance_m=_positive(
            node["minimum_underbody_clearance_m"],
            "minimum_underbody_clearance_m",
        ),
        maximum_local_obstacle_relief_m=_nonnegative(
            node["maximum_local_obstacle_relief_m"],
            "maximum_local_obstacle_relief_m",
        ),
        allow_unsupported_gap=allow_unsupported_gap,
        minimum_body_z_m=0.0,
        maximum_body_z_m=body_extent.z,
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
    if not isinstance(value, dict):
        raise CapabilityFreezeError("legged capability schema fields are invalid")
    fields = set(value)
    if fields == _LEGGED_FIELDS:
        has_nominal_metadata = True
    elif fields == _LEGACY_LEGGED_FIELDS:
        has_nominal_metadata = False
    else:
        raise CapabilityFreezeError("legged capability schema fields are invalid")
    node = value
    reference_point = _nonempty_string(node["reference_point"], "reference_point")
    if reference_point != "base_link":
        raise CapabilityFreezeError("legged reference_point must be base_link")
    if node["roughness_handling"] != "DIAGNOSTIC_ONLY":
        raise CapabilityFreezeError("legged roughness_handling is unsupported")
    extent = _vec3(node["body_extent_m"], "body_extent_m")
    if min(extent.x, extent.y, extent.z) <= 0.0:
        raise CapabilityFreezeError("body_extent_m must be positive")
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
            )
        )
    body_height = _interval(node["body_height_m"], "body_height_m")
    if body_height.lower < 0.0:
        raise CapabilityFreezeError("body_height_m must be non-negative")
    nominal_body_height = (
        _positive(node["nominal_body_height_m"], "nominal_body_height_m")
        if has_nominal_metadata
        else 0.5 * (body_height.lower + body_height.upper)
    )
    nominal_payload = (
        _positive(node["nominal_payload_kg"], "nominal_payload_kg")
        if has_nominal_metadata
        else _LEGACY_NOMINAL_PAYLOAD_KG
    )
    unknown_is_traversable = (
        _boolean(node["unknown_is_traversable"], "unknown_is_traversable")
        if has_nominal_metadata
        else _LEGACY_UNKNOWN_IS_TRAVERSABLE
    )
    return FrozenLeggedCapability(
        reference_point=reference_point,
        body_extent_m=extent,
        nominal_body_height_m=nominal_body_height,
        platform_mass_kg=_positive(node["platform_mass_kg"], "platform_mass_kg"),
        nominal_payload_kg=nominal_payload,
        maximum_payload_kg=_positive(
            node["maximum_payload_kg"], "maximum_payload_kg"
        ),
        maximum_slope_rad=_slope(node["maximum_slope_rad"], "maximum_slope_rad"),
        maximum_step_height_m=_nonnegative(
            node["maximum_step_height_m"], "maximum_step_height_m"
        ),
        maximum_gap_width_m=_nonnegative(
            node["maximum_gap_width_m"], "maximum_gap_width_m"
        ),
        minimum_body_clearance_m=_nonnegative(
            node["minimum_body_clearance_m"], "minimum_body_clearance_m"
        ),
        step_vertical_rate_mps=_positive(
            node["step_vertical_rate_mps"], "step_vertical_rate_mps"
        ),
        body_height_m=body_height,
        forward_speed_mps=_interval(
            node["forward_speed_mps"], "forward_speed_mps", require_zero=True
        ),
        lateral_speed_mps=_interval(
            node["lateral_speed_mps"], "lateral_speed_mps", require_zero=True
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
        unknown_is_traversable=unknown_is_traversable,
        motion_primitives=tuple(primitives),
    )


def _parse_hopper(value: object) -> FrozenHopperCapability:
    if not isinstance(value, dict):
        raise CapabilityFreezeError("hopper capability schema fields are invalid")
    legacy = set(value) == _LEGACY_HOPPER_FIELDS
    node = _require_exact_object(
        value,
        _LEGACY_HOPPER_FIELDS if legacy else _HOPPER_FIELDS,
        "hopper capability",
    )
    parsed = FrozenHopperCapability(
        specific_impulse_s=_positive(
            node["specific_impulse_s"], "specific_impulse_s"
        ),
        reference_total_mass_kg=_positive(
            node["reference_total_mass_kg"], "reference_total_mass_kg"
        ),
        reference_propellant_mass_kg=_positive(
            node["reference_propellant_mass_kg"],
            "reference_propellant_mass_kg",
        ),
        gravity_mps2=(
            FrozenVec3(0.0, 0.0, -1.62)
            if legacy
            else _vec3(node["gravity_mps2"], "gravity_mps2")
        ),
        reference_horizontal_range_m=(
            100.0
            if legacy
            else _positive(
                node["reference_horizontal_range_m"], "reference_horizontal_range_m"
            )
        ),
        reference_elevation_delta_m=(
            0.0
            if legacy
            else _finite(
                node["reference_elevation_delta_m"], "reference_elevation_delta_m"
            )
        ),
        runtime_fallback_allowed=(
            False
            if legacy
            else _boolean(
                node["runtime_fallback_allowed"], "runtime_fallback_allowed")
        ),
        landing_support_radius_m=_positive(
            node["landing_support_radius_m"], "landing_support_radius_m"
        ),
        flight_collision_radius_m=_positive(
            node["flight_collision_radius_m"], "flight_collision_radius_m"
        ),
        maximum_landing_plane_residual_m=_nonnegative(
            node["maximum_landing_plane_residual_m"],
            "maximum_landing_plane_residual_m",
        ),
        landing_lateral_margin_m=_nonnegative(
            node["landing_lateral_margin_m"], "landing_lateral_margin_m"
        ),
        flight_map_margin_m=_nonnegative(
            node["flight_map_margin_m"], "flight_map_margin_m"
        ),
        reachability_delta_v_margin_ratio=_nonnegative(
            node["reachability_delta_v_margin_ratio"],
            "reachability_delta_v_margin_ratio",
        ),
        standard_gravity_mps2=_positive(
            node["standard_gravity_mps2"], "standard_gravity_mps2"
        ),
        maximum_landing_slope_rad=_slope(
            node["maximum_landing_slope_rad"], "maximum_landing_slope_rad"
        ),
    )
    if parsed.reference_propellant_mass_kg >= parsed.reference_total_mass_kg:
        raise CapabilityFreezeError(
            "reference_propellant_mass_kg must be less than reference_total_mass_kg"
        )
    if parsed.gravity_mps2 != FrozenVec3(0.0, 0.0, -1.62):
        raise CapabilityFreezeError("gravity_mps2 must be the approved lunar vector")
    if parsed.runtime_fallback_allowed:
        raise CapabilityFreezeError("runtime_fallback_allowed must be false")
    return parsed


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


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise CapabilityFreezeError(f"{field} must be a boolean")
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
