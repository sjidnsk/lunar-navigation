"""Canonical adapter for the project's approved three-platform capability."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import yaml

from .capability_freeze import (
    CAPABILITY_FREEZE_SCHEMA,
    CAPABILITY_TYPES,
    CapabilityFreezeError,
    FrozenCapabilityBundle,
    FrozenObservationCapability,
    FrozenPlatformCapability,
    _parse_hopper,
    _parse_legged,
    _parse_wheeled,
)
from .training_semantics import FORMAL_SENSOR_FOV_RAD, FORMAL_SENSOR_RANGE_M


APPROVED_PROJECT_CAPABILITY_SHA256 = (
    "60e258be85edd779d9acdc282bbde3d5cb914bce98c86c244a46a772fda5ee95"
)
_CONFIG_ROOT = Path("ros2_ws/src/lunar_navigation_config/config")
_SCHEMA_NAME = "platform_capability_schema_v2.yaml"
_FREEZE_NAME = "three_platform_capability_freeze_v1.yaml"
_PLATFORM_ORDER = ("WHEELED", "LEGGED", "HOPPER")
_FREEZE_FIELDS = frozenset(
    (
        "schema_version",
        "capability_schema",
        "freeze_version",
        "ownership",
        "platforms",
        "freeze_digest_sha256",
    )
)


def project_capability_paths(
    repository_root: str | Path,
) -> tuple[Path, Path]:
    root = Path(repository_root).resolve(strict=True)
    config = root / _CONFIG_ROOT
    schema = _regular_file(config / _SCHEMA_NAME, root)
    freeze = _regular_file(config / _FREEZE_NAME, root)
    return schema, freeze


def load_project_formal_capability(
    repository_root: str | Path,
) -> FrozenCapabilityBundle:
    """Load the user-approved repository freeze as the sole formal authority."""
    schema_path, freeze_path = project_capability_paths(repository_root)
    schema = _read_yaml_mapping(schema_path, "project capability schema")
    freeze = _read_yaml_mapping(freeze_path, "project capability freeze")
    _validate_headers(schema, freeze)
    platforms = freeze.get("platforms")
    if not isinstance(platforms, Mapping) or set(platforms) != set(_PLATFORM_ORDER):
        raise CapabilityFreezeError(
            "project capability platforms must be WHEELED, LEGGED and HOPPER"
        )
    digest = _semantic_sha256(platforms)
    if freeze.get("freeze_digest_sha256") != digest:
        raise CapabilityFreezeError(
            "freeze_digest_sha256 does not match platform payload"
        )
    if digest != APPROVED_PROJECT_CAPABILITY_SHA256:
        raise CapabilityFreezeError(
            "project capability digest is not the approved formal baseline"
        )
    try:
        converted = tuple(
            _convert_platform(platform, platforms[platform])
            for platform in _PLATFORM_ORDER
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CapabilityFreezeError(
            "project capability platform schema fields are invalid"
        ) from error
    return FrozenCapabilityBundle(
        schema=CAPABILITY_FREEZE_SCHEMA,
        platforms=converted,
        bundle_sha256=digest,
        formal_eligible=True,
    )


def _validate_headers(schema: Mapping[str, object], freeze: Mapping[str, object]) -> None:
    if schema.get("schema_version") != "platform-control-capability-source/v2":
        raise CapabilityFreezeError("project capability schema version is invalid")
    if schema.get("freeze_schema_version") != "lunar-platform-capability-freeze/v1":
        raise CapabilityFreezeError("project capability freeze schema is invalid")
    if set(freeze) != _FREEZE_FIELDS:
        raise CapabilityFreezeError("project capability freeze fields are invalid")
    if freeze.get("schema_version") != "lunar-platform-capability-freeze/v1":
        raise CapabilityFreezeError("project capability freeze version is invalid")
    if freeze.get("capability_schema") != schema.get("schema_version"):
        raise CapabilityFreezeError("project capability schema identity mismatch")
    if freeze.get("freeze_version") != "three-platform-capability-freeze-v1":
        raise CapabilityFreezeError("project capability version is invalid")
    ownership = freeze.get("ownership")
    if not isinstance(ownership, Mapping) or set(ownership) != {
        "producer",
        "consumer",
        "repository_role",
    }:
        raise CapabilityFreezeError("project capability ownership is invalid")


def _convert_platform(
    platform_type: str,
    raw: object,
) -> FrozenPlatformCapability:
    if not isinstance(raw, Mapping):
        raise CapabilityFreezeError("project capability platform must be a mapping")
    if raw.get("platform_type") != platform_type:
        raise CapabilityFreezeError("project capability platform type mismatch")
    platform_id = _required_text(raw, "platform_id")
    capability_version = _required_text(raw, "capability_version")
    base_frame_id = _required_text(raw, "base_frame_id")
    provenance = _required_mapping(raw, "provenance")
    design_document = _required_text(provenance, "design_document")
    if platform_type == "WHEELED":
        typed = _parse_wheeled(_wheel_payload(raw))
    elif platform_type == "LEGGED":
        typed = _parse_legged(_legged_payload(raw))
    else:
        typed = _parse_hopper(_hopper_payload(raw))
    observation = FrozenObservationCapability(
        sensor_range_m=FORMAL_SENSOR_RANGE_M,
        sensor_fov_rad=FORMAL_SENSOR_FOV_RAD,
    )
    return FrozenPlatformCapability(
        platform_type=platform_type,
        capability_type=CAPABILITY_TYPES[platform_type],
        capability_version=capability_version,
        platform_id=platform_id,
        base_frame_id=base_frame_id,
        platform_document_path=design_document,
        observation_document_path=(
            "docs/superpowers/specs/"
            "2026-08-08-sensor-observation-capability-design.md"
        ),
        urdf_path="",
        mesh_paths=(),
        observation_capability=observation,
        typed_capability=typed,
        content_sha256=_semantic_sha256(raw),
        resources=(),
    )


def _wheel_payload(raw: Mapping[str, object]) -> dict[str, object]:
    geometry = _required_mapping(raw, "geometry")
    kinematics = _required_mapping(raw, "kinematics")
    terrain = _required_mapping(raw, "terrain")
    motion = _required_mapping(raw, "motion")
    step = _positive_number(motion, "xy_resolution_m")
    radius = _positive_number(motion, "arc_radius_m")
    yaw = _positive_number(motion, "arc_yaw_change_rad")
    arc_x = radius * math.sin(yaw)
    arc_y = radius * (1.0 - math.cos(yaw))
    return {
        "reference_point": geometry["reference_point"],
        "footprint_xy_m": geometry["footprint_xy_m"],
        "body_extent_m": geometry["body_extent_m"],
        "wheel_diameter_m": geometry["wheel_diameter_m"],
        "wheel_width_m": geometry["wheel_width_m"],
        "wheelbase_m": geometry["wheelbase_m"],
        "track_width_m": geometry["track_width_m"],
        "minimum_underbody_clearance_m": geometry[
            "minimum_underbody_clearance_m"
        ],
        "maximum_local_obstacle_relief_m": terrain[
            "maximum_local_obstacle_relief_m"
        ],
        "allow_unsupported_gap": terrain["allow_unsupported_gap"],
        "maximum_slope_rad": terrain["maximum_surface_slope_rad"],
        "maximum_forward_speed_mps": kinematics["maximum_forward_speed_mps"],
        "maximum_reverse_speed_mps": kinematics["maximum_reverse_speed_mps"],
        "maximum_spin_rate_radps": kinematics["maximum_spin_rate_radps"],
        "maximum_acceleration_mps2": kinematics["maximum_acceleration_mps2"],
        "maximum_braking_deceleration_mps2": kinematics[
            "maximum_braking_deceleration_mps2"
        ],
        "maximum_yaw_acceleration_radps2": kinematics[
            "maximum_yaw_acceleration_radps2"
        ],
        "maximum_lateral_acceleration_mps2": kinematics[
            "maximum_lateral_acceleration_mps2"
        ],
        "maximum_curvature_per_m": kinematics["maximum_curvature_per_m"],
        "minimum_clearance_m": geometry["minimum_clearance_m"],
        "roughness_handling": terrain["roughness_handling"],
        "motion_primitives": [
            _wheel_primitive("forward", "FORWARD", step, 0.0, 0.0),
            _wheel_primitive("reverse", "REVERSE", -step, 0.0, 0.0),
            _wheel_primitive(
                "forward-arc-left", "FORWARD_ARC", arc_x, arc_y, yaw
            ),
            _wheel_primitive(
                "forward-arc-right", "FORWARD_ARC", arc_x, -arc_y, -yaw
            ),
            _wheel_primitive(
                "reverse-arc-left", "REVERSE_ARC", -arc_x, arc_y, -yaw
            ),
            _wheel_primitive(
                "reverse-arc-right", "REVERSE_ARC", -arc_x, -arc_y, yaw
            ),
            _wheel_primitive(
                "spin-left", "SPIN_COUNTERCLOCKWISE", 0.0, 0.0, yaw
            ),
            _wheel_primitive(
                "spin-right", "SPIN_CLOCKWISE", 0.0, 0.0, -yaw
            ),
            _wheel_primitive("stop-switch", "STOP_AND_SWITCH", 0.0, 0.0, 0.0),
        ],
    }


def _legged_payload(raw: Mapping[str, object]) -> dict[str, object]:
    geometry = _required_mapping(raw, "geometry")
    physical = _required_mapping(raw, "physical")
    kinematics = _required_mapping(raw, "kinematics")
    terrain = _required_mapping(raw, "terrain")
    motion = _required_mapping(raw, "motion")
    step = _positive_number(motion, "local_xy_resolution_m")
    yaw = math.pi / 16.0
    primitives = (
        ("forward", "FORWARD", (step, 0.0, 0.0), 0.0),
        ("backward", "BACKWARD", (-step, 0.0, 0.0), 0.0),
        ("left", "LATERAL_LEFT", (0.0, step, 0.0), 0.0),
        ("right", "LATERAL_RIGHT", (0.0, -step, 0.0), 0.0),
        ("spin-left", "SPIN", (0.0, 0.0, 0.0), yaw),
        ("spin-right", "SPIN", (0.0, 0.0, 0.0), -yaw),
    )
    return {
        "reference_point": geometry["reference_point"],
        "body_extent_m": geometry["body_extent_m"],
        "platform_mass_kg": physical["platform_mass_kg"],
        "maximum_payload_kg": physical["maximum_payload_kg"],
        "maximum_slope_rad": terrain["maximum_surface_slope_rad"],
        "maximum_step_height_m": terrain["maximum_step_height_m"],
        "maximum_gap_width_m": terrain["maximum_gap_width_m"],
        "minimum_body_clearance_m": terrain["minimum_body_clearance_m"],
        "step_vertical_rate_mps": terrain["step_vertical_rate_mps"],
        "body_height_m": geometry["body_height_m"],
        "forward_speed_mps": [
            -float(kinematics["maximum_reverse_speed_mps"]),
            kinematics["maximum_forward_speed_mps"],
        ],
        "lateral_speed_mps": [
            -float(kinematics["maximum_lateral_speed_mps"]),
            kinematics["maximum_lateral_speed_mps"],
        ],
        "yaw_rate_radps": [
            -float(kinematics["maximum_yaw_rate_radps"]),
            kinematics["maximum_yaw_rate_radps"],
        ],
        "maximum_linear_acceleration_mps2": kinematics[
            "maximum_linear_acceleration_mps2"
        ],
        "maximum_yaw_acceleration_radps2": kinematics[
            "maximum_yaw_acceleration_radps2"
        ],
        "roughness_handling": terrain["roughness_handling"],
        "motion_primitives": [
            {
                "primitive_id": identifier,
                "kind": kind,
                "body_frame_displacement_m": list(displacement),
                "yaw_change_rad": yaw_change,
            }
            for identifier, kind, displacement, yaw_change in primitives
        ],
    }


def _hopper_payload(raw: Mapping[str, object]) -> dict[str, object]:
    propulsion = _required_mapping(raw, "propulsion")
    landing = _required_mapping(raw, "landing")
    safety = _required_mapping(raw, "safety")
    environment = _required_mapping(raw, "environment")
    reference = _required_mapping(raw, "reference_conditions")
    # The approved freeze keeps its historical source-field spelling so its
    # digest stays stable. It is adapted once into a non-decrementing reference
    # envelope quantity; it is never exposed as runtime remaining fuel.
    return {
        "specific_impulse_s": propulsion["specific_impulse_s"],
        "reference_total_mass_kg": reference["reference_total_mass_kg"],
        "reference_propellant_mass_kg": reference[
            "reference_remaining_usable_fuel_mass_kg"
        ],
        "landing_support_radius_m": landing["landing_support_radius_m"],
        "flight_collision_radius_m": safety["flight_collision_radius_m"],
        "maximum_landing_slope_rad": landing["maximum_landing_slope_rad"],
        "maximum_landing_plane_residual_m": landing[
            "maximum_landing_plane_residual_m"
        ],
        "landing_lateral_margin_m": safety["landing_lateral_margin_m"],
        "flight_map_margin_m": safety["flight_map_margin_m"],
        "reachability_delta_v_margin_ratio": safety[
            "reachability_delta_v_margin_ratio"
        ],
        "standard_gravity_mps2": environment["standard_gravity_mps2"],
    }


def _wheel_primitive(
    identifier: str,
    kind: str,
    x: float,
    y: float,
    yaw: float,
) -> dict[str, object]:
    return {
        "primitive_id": identifier,
        "kind": kind,
        "relative_end_pose": {
            "position_m": [x, y, 0.0],
            "orientation_wxyz": [
                math.cos(0.5 * yaw),
                0.0,
                0.0,
                math.sin(0.5 * yaw),
            ],
        },
    }


def _regular_file(path: Path, root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise CapabilityFreezeError("project capability path is missing or unsafe") from error
    if path.is_symlink() or not resolved.is_file():
        raise CapabilityFreezeError("project capability path must be a regular file")
    return resolved


def _read_yaml_mapping(path: Path, label: str) -> Mapping[str, object]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CapabilityFreezeError(f"{label} is invalid YAML") from error
    if not isinstance(document, Mapping):
        raise CapabilityFreezeError(f"{label} must be a mapping")
    return document


def _required_mapping(
    value: Mapping[str, object], key: str
) -> Mapping[str, object]:
    selected = value.get(key)
    if not isinstance(selected, Mapping):
        raise CapabilityFreezeError(f"project capability {key} must be a mapping")
    return selected


def _required_text(value: Mapping[str, object], key: str) -> str:
    selected = value.get(key)
    if not isinstance(selected, str) or not selected:
        raise CapabilityFreezeError(f"project capability {key} must be text")
    return selected


def _positive_number(value: Mapping[str, object], key: str) -> float:
    selected = value.get(key)
    if (
        not isinstance(selected, (int, float))
        or isinstance(selected, bool)
        or not math.isfinite(float(selected))
        or float(selected) <= 0.0
    ):
        raise CapabilityFreezeError(f"project capability {key} must be positive")
    return float(selected)


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
        raise CapabilityFreezeError(
            "project capability is not canonicalizable"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "APPROVED_PROJECT_CAPABILITY_SHA256",
    "load_project_formal_capability",
    "project_capability_paths",
]
