#!/usr/bin/env python3
"""校验可直接替换的单平台运行 profile 及其正式数值等价性。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


PROFILE_SCHEMA = "lunar-platform-profile/v1"
PLATFORM_TYPES = ("WHEELED", "LEGGED", "HOPPER")
PLATFORM_SECTIONS = {
    "WHEELED": "wheeled",
    "LEGGED": "legged",
    "HOPPER": "hopper",
}
BASE_KEYS = {
    "schema_version",
    "ownership",
    "platform",
    "observation",
    "assets",
    "sources",
}

PROFILE_TO_FREEZE_PATHS: dict[str, dict[str, str]] = {
    "WHEELED": {
        "reference_point": "geometry.reference_point",
        "body_extent_m": "geometry.body_extent_m",
        "footprint_xy_m": "geometry.footprint_xy_m",
        "wheel_count": "geometry.wheel_count",
        "wheel_diameter_m": "geometry.wheel_diameter_m",
        "wheel_width_m": "geometry.wheel_width_m",
        "wheelbase_m": "geometry.wheelbase_m",
        "track_width_m": "geometry.track_width_m",
        "wheel_center_xy_m": "geometry.wheel_center_xy_m",
        "minimum_underbody_clearance_m":
            "geometry.minimum_underbody_clearance_m",
        "minimum_clearance_m": "geometry.minimum_clearance_m",
        "maximum_forward_speed_mps": "kinematics.maximum_forward_speed_mps",
        "maximum_reverse_speed_mps": "kinematics.maximum_reverse_speed_mps",
        "maximum_spin_rate_radps": "kinematics.maximum_spin_rate_radps",
        "maximum_acceleration_mps2": "kinematics.maximum_acceleration_mps2",
        "maximum_braking_deceleration_mps2":
            "kinematics.maximum_braking_deceleration_mps2",
        "maximum_yaw_acceleration_radps2":
            "kinematics.maximum_yaw_acceleration_radps2",
        "maximum_lateral_acceleration_mps2":
            "kinematics.maximum_lateral_acceleration_mps2",
        "maximum_curvature_per_m": "kinematics.maximum_curvature_per_m",
        "maximum_surface_slope_rad": "terrain.maximum_surface_slope_rad",
        "maximum_local_obstacle_relief_m":
            "terrain.maximum_local_obstacle_relief_m",
        "allow_unsupported_gap": "terrain.allow_unsupported_gap",
        "roughness_handling": "terrain.roughness_handling",
        "xy_resolution_m": "motion.xy_resolution_m",
        "yaw_bin_count": "motion.yaw_bin_count",
        "arc_radius_m": "motion.arc_radius_m",
        "arc_yaw_change_rad": "motion.arc_yaw_change_rad",
    },
    "LEGGED": {
        "reference_point": "geometry.reference_point",
        "body_extent_m": "geometry.body_extent_m",
        "nominal_body_height_m": "geometry.nominal_body_height_m",
        "body_height_m": "geometry.body_height_m",
        "platform_mass_kg": "physical.platform_mass_kg",
        "nominal_payload_kg": "physical.nominal_payload_kg",
        "maximum_payload_kg": "physical.maximum_payload_kg",
        "maximum_forward_speed_mps": "kinematics.maximum_forward_speed_mps",
        "maximum_reverse_speed_mps": "kinematics.maximum_reverse_speed_mps",
        "maximum_lateral_speed_mps": "kinematics.maximum_lateral_speed_mps",
        "maximum_yaw_rate_radps": "kinematics.maximum_yaw_rate_radps",
        "maximum_linear_acceleration_mps2":
            "kinematics.maximum_linear_acceleration_mps2",
        "maximum_yaw_acceleration_radps2":
            "kinematics.maximum_yaw_acceleration_radps2",
        "maximum_surface_slope_rad": "terrain.maximum_surface_slope_rad",
        "maximum_step_height_m": "terrain.maximum_step_height_m",
        "maximum_gap_width_m": "terrain.maximum_gap_width_m",
        "minimum_body_clearance_m": "terrain.minimum_body_clearance_m",
        "step_vertical_rate_mps": "terrain.step_vertical_rate_mps",
        "roughness_handling": "terrain.roughness_handling",
        "unknown_is_traversable": "terrain.unknown_is_traversable",
        "local_xy_resolution_m": "motion.local_xy_resolution_m",
        "output_semantics": "motion.output_semantics",
    },
    "HOPPER": {
        "specific_impulse_s": "propulsion.specific_impulse_s",
        "reference_total_mass_kg":
            "reference_conditions.reference_total_mass_kg",
        "reference_propellant_mass_kg":
            "reference_conditions.reference_remaining_usable_fuel_mass_kg",
        "reference_horizontal_range_m":
            "reference_conditions.reference_horizontal_range_m",
        "reference_elevation_delta_m":
            "reference_conditions.reference_elevation_delta_m",
        "runtime_fallback_allowed":
            "reference_conditions.runtime_fallback_allowed",
        "landing_support_radius_m": "landing.landing_support_radius_m",
        "flight_collision_radius_m": "safety.flight_collision_radius_m",
        "maximum_landing_slope_rad": "landing.maximum_landing_slope_rad",
        "maximum_landing_plane_residual_m":
            "landing.maximum_landing_plane_residual_m",
        "landing_lateral_margin_m": "safety.landing_lateral_margin_m",
        "flight_map_margin_m": "safety.flight_map_margin_m",
        "reachability_delta_v_margin_ratio":
            "safety.reachability_delta_v_margin_ratio",
        "gravity_mps2": "environment.gravity_mps2",
        "standard_gravity_mps2": "environment.standard_gravity_mps2",
    },
}

EXTRA_SOURCE_VALUES: dict[str, dict[str, str]] = {
    "WHEELED": {
        "sensor_range_m": "user_approved_planning_policy",
        "sensor_fov_deg": "user_approved_planning_policy",
        "xy_resolution_m": "user_approved_planning_policy",
        "yaw_bin_count": "user_approved_planning_policy",
        "arc_radius_m": "user_approved_planning_policy",
        "arc_yaw_change_rad": "user_approved_planning_policy",
    },
    "LEGGED": {
        "sensor_range_m": "user_approved_planning_policy",
        "sensor_fov_deg": "user_approved_planning_policy",
        "motion_primitives": "user_approved_planning_policy",
        "local_xy_resolution_m": "planning_policy",
        "output_semantics": "planning_policy",
    },
    "HOPPER": {
        "sensor_range_m": "user_approved_planning_policy",
        "sensor_fov_deg": "user_approved_planning_policy",
        "reference_propellant_mass_kg": "project_engineering_baseline",
        "runtime_fallback_allowed": "planning_safety_baseline",
    },
}


def load_json_mapping(path: Path, label: str) -> tuple[Mapping[str, Any] | None, list[str]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, [f"{label}: unable to load JSON: {error}"]
    if not isinstance(document, Mapping):
        return None, [f"{label}: document must be an object"]
    return document, []


def load_yaml_mapping(path: Path, label: str) -> tuple[Mapping[str, Any] | None, list[str]]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        return None, [f"{label}: unable to load YAML: {error}"]
    if not isinstance(document, Mapping):
        return None, [f"{label}: document must be a mapping"]
    return document, []


def get_path(document: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = document
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def equal_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12)
        )
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return (
            isinstance(actual, Sequence)
            and not isinstance(actual, (str, bytes))
            and len(actual) == len(expected)
            and all(equal_value(item, wanted) for item, wanted in zip(actual, expected))
        )
    return actual == expected


def resolve_ref(root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise ValueError(f"unsupported schema reference: {reference}")
    current: Any = root
    for component in reference[2:].split("/"):
        if not isinstance(current, Mapping) or component not in current:
            raise ValueError(f"unknown schema reference: {reference}")
        current = current[component]
    if not isinstance(current, Mapping):
        raise ValueError(f"schema reference is not an object: {reference}")
    return current


def validate_schema_value(
    value: Any,
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    label: str,
) -> list[str]:
    if "$ref" in schema:
        return validate_schema_value(value, resolve_ref(root, str(schema["$ref"])), root, label)
    if "oneOf" in schema:
        alternatives = schema["oneOf"]
        outcomes = [validate_schema_value(value, item, root, label) for item in alternatives]
        return [] if sum(not errors for errors in outcomes) == 1 else [f"{label} does not match exactly one schema alternative"]

    errors: list[str] = []
    if "const" in schema and value != schema["const"]:
        errors.append(f"{label} must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{label} must be one of {schema['enum']!r}")

    kind = schema.get("type")
    valid_type = True
    if kind == "object":
        valid_type = isinstance(value, Mapping)
    elif kind == "array":
        valid_type = isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    elif kind == "string":
        valid_type = isinstance(value, str)
    elif kind == "number":
        valid_type = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    elif kind == "integer":
        valid_type = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "boolean":
        valid_type = isinstance(value, bool)
    elif kind == "null":
        valid_type = value is None
    if kind is not None and not valid_type:
        return [f"{label} must have type {kind}"]

    if isinstance(value, Mapping) and kind == "object":
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{label} missing required key: {key}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            child_label = f"{label}.{key}"
            if key in properties:
                errors.extend(validate_schema_value(item, properties[key], root, child_label))
            elif additional is False:
                errors.append(f"{label} has unexpected key: {key}")
            elif isinstance(additional, Mapping):
                errors.extend(validate_schema_value(item, additional, root, child_label))
        minimum_properties = schema.get("minProperties")
        if isinstance(minimum_properties, int) and len(value) < minimum_properties:
            errors.append(f"{label} must contain at least {minimum_properties} properties")

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and kind == "array":
        minimum_items = schema.get("minItems")
        maximum_items = schema.get("maxItems")
        if isinstance(minimum_items, int) and len(value) < minimum_items:
            errors.append(f"{label} must contain at least {minimum_items} items")
        if isinstance(maximum_items, int) and len(value) > maximum_items:
            errors.append(f"{label} must contain at most {maximum_items} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                errors.extend(validate_schema_value(item, item_schema, root, f"{label}[{index}]"))

    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        pattern = schema.get("pattern")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            errors.append(f"{label} must contain at least {minimum_length} characters")
        if isinstance(pattern, str) and re.fullmatch(pattern, value) is None:
            errors.append(f"{label} does not match required pattern")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and float(value) < float(schema["minimum"]):
            errors.append(f"{label} must be at least {schema['minimum']}")
        if "exclusiveMinimum" in schema and float(value) <= float(schema["exclusiveMinimum"]):
            errors.append(f"{label} must be greater than {schema['exclusiveMinimum']}")
        if "maximum" in schema and float(value) > float(schema["maximum"]):
            errors.append(f"{label} must be at most {schema['maximum']}")
    return errors


def wheel_primitives(frozen: Mapping[str, Any]) -> list[dict[str, Any]]:
    step = float(get_path(frozen, "motion.xy_resolution_m"))
    radius = float(get_path(frozen, "motion.arc_radius_m"))
    yaw = float(get_path(frozen, "motion.arc_yaw_change_rad"))
    arc_x = radius * math.sin(yaw)
    arc_y = radius * (1.0 - math.cos(yaw))

    def primitive(identifier: str, kind: str, x: float, y: float, angle: float) -> dict[str, Any]:
        return {
            "primitive_id": identifier,
            "kind": kind,
            "relative_end_pose": {
                "position_m": [x, y, 0.0],
                "orientation_wxyz": [
                    math.cos(0.5 * angle),
                    0.0,
                    0.0,
                    math.sin(0.5 * angle),
                ],
            },
        }

    return [
        primitive("forward", "FORWARD", step, 0.0, 0.0),
        primitive("reverse", "REVERSE", -step, 0.0, 0.0),
        primitive("forward-arc-left", "FORWARD_ARC", arc_x, arc_y, yaw),
        primitive("forward-arc-right", "FORWARD_ARC", arc_x, -arc_y, -yaw),
        primitive("reverse-arc-left", "REVERSE_ARC", -arc_x, arc_y, -yaw),
        primitive("reverse-arc-right", "REVERSE_ARC", -arc_x, -arc_y, yaw),
        primitive("spin-left", "SPIN_COUNTERCLOCKWISE", 0.0, 0.0, yaw),
        primitive("spin-right", "SPIN_CLOCKWISE", 0.0, 0.0, -yaw),
        primitive("stop-switch", "STOP_AND_SWITCH", 0.0, 0.0, 0.0),
    ]


def legged_primitives(frozen: Mapping[str, Any]) -> list[dict[str, Any]]:
    step = float(get_path(frozen, "motion.local_xy_resolution_m"))
    yaw = math.pi / 16.0
    entries = (
        ("forward", "FORWARD", [step, 0.0, 0.0], 0.0),
        ("backward", "BACKWARD", [-step, 0.0, 0.0], 0.0),
        ("left", "LATERAL_LEFT", [0.0, step, 0.0], 0.0),
        ("right", "LATERAL_RIGHT", [0.0, -step, 0.0], 0.0),
        ("spin-left", "SPIN", [0.0, 0.0, 0.0], yaw),
        ("spin-right", "SPIN", [0.0, 0.0, 0.0], -yaw),
    )
    return [
        {
            "primitive_id": identifier,
            "kind": kind,
            "body_frame_displacement_m": displacement,
            "yaw_change_rad": yaw_change,
        }
        for identifier, kind, displacement, yaw_change in entries
    ]


def expected_sources(platform_type: str, frozen: Mapping[str, Any]) -> dict[str, str]:
    raw_sources = frozen.get("sources")
    if not isinstance(raw_sources, Mapping):
        return {}
    result = {str(key): str(value) for key, value in raw_sources.items()}
    if platform_type == "HOPPER":
        result["reference_propellant_mass_kg"] = result.pop(
            "reference_remaining_usable_fuel_mass_kg"
        )
    result.update(EXTRA_SOURCE_VALUES[platform_type])
    return result


def validate_equivalence(
    profile: Mapping[str, Any],
    frozen: Mapping[str, Any],
    platform_type: str,
) -> list[str]:
    errors: list[str] = []
    platform = profile.get("platform")
    section_name = PLATFORM_SECTIONS[platform_type]
    section = profile.get(section_name)
    if not isinstance(platform, Mapping) or not isinstance(section, Mapping):
        return errors

    for key in ("platform_id", "platform_type", "capability_version", "base_frame_id"):
        if not equal_value(platform.get(key), frozen.get(key)):
            errors.append(f"platform.{key} must equal approved freeze")
    for profile_key, freeze_path in PROFILE_TO_FREEZE_PATHS[platform_type].items():
        if not equal_value(section.get(profile_key), get_path(frozen, freeze_path)):
            errors.append(f"{section_name}.{profile_key} must equal approved freeze")

    if platform_type == "WHEELED" and not equal_value(
        section.get("motion_primitives"), wheel_primitives(frozen)
    ):
        errors.append("wheeled.motion_primitives must equal approved derived primitives")
    if platform_type == "LEGGED" and not equal_value(
        section.get("motion_primitives"), legged_primitives(frozen)
    ):
        errors.append("legged.motion_primitives must equal approved derived primitives")

    observation = profile.get("observation")
    if isinstance(observation, Mapping):
        if not equal_value(observation.get("sensor_range_m"), 30.0):
            errors.append("observation.sensor_range_m must equal approved 30.0")
        if not equal_value(observation.get("sensor_fov_deg"), 360.0):
            errors.append("observation.sensor_fov_deg must equal approved 360.0")

    sources = profile.get("sources")
    wanted_sources = expected_sources(platform_type, frozen)
    if not isinstance(sources, Mapping) or dict(sources) != wanted_sources:
        errors.append("sources must cover the complete approved runtime profile")
    return errors


def validate_profile(
    profile: Mapping[str, Any],
    schema: Mapping[str, Any],
    freeze_platforms: Mapping[str, Any],
) -> tuple[list[str], str | None]:
    errors = validate_schema_value(profile, schema, schema, "profile")
    if profile.get("schema_version") != PROFILE_SCHEMA:
        errors.append(f"schema_version must be {PROFILE_SCHEMA}")
    platform = profile.get("platform")
    platform_type = platform.get("platform_type") if isinstance(platform, Mapping) else None
    if platform_type not in PLATFORM_TYPES:
        errors.append("platform.platform_type must be WHEELED, LEGGED, or HOPPER")
        return errors, None

    section = PLATFORM_SECTIONS[str(platform_type)]
    expected_keys = BASE_KEYS | {section}
    if set(profile) != expected_keys:
        missing = sorted(expected_keys - set(profile))
        unexpected = sorted(set(profile) - expected_keys)
        if missing:
            errors.append(f"profile missing active section: {', '.join(missing)}")
        if unexpected:
            errors.append(f"profile has inactive section: {', '.join(unexpected)}")

    frozen = freeze_platforms.get(platform_type)
    if not isinstance(frozen, Mapping):
        errors.append(f"approved freeze does not contain {platform_type}")
    else:
        errors.extend(validate_equivalence(profile, frozen, str(platform_type)))
    return errors, str(platform_type)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--profile", type=Path, action="append", required=True)
    arguments = parser.parse_args()

    schema, schema_errors = load_json_mapping(arguments.schema, "schema")
    freeze, freeze_errors = load_yaml_mapping(arguments.freeze, "freeze")
    errors = [*schema_errors, *freeze_errors]
    freeze_platforms: Mapping[str, Any] = {}
    if freeze is not None:
        candidate = freeze.get("platforms")
        if isinstance(candidate, Mapping):
            freeze_platforms = candidate
        else:
            errors.append("freeze.platforms must be a mapping")

    identities: dict[str, str] = {}
    if schema is not None:
        for path in arguments.profile:
            profile, profile_errors = load_yaml_mapping(path, str(path))
            errors.extend(profile_errors)
            if profile is None:
                continue
            profile_errors, platform_type = validate_profile(
                profile, schema, freeze_platforms
            )
            errors.extend(f"{path}: {error}" for error in profile_errors)
            if platform_type is not None:
                if platform_type in identities:
                    errors.append(f"duplicate platform profile: {platform_type}")
                else:
                    identities[platform_type] = file_sha256(path)

    if errors:
        for error in errors:
            print(f"platform profile error: {error}")
        return 1
    summary = " ".join(
        f"{platform_type}={identities[platform_type]}"
        for platform_type in PLATFORM_TYPES
        if platform_type in identities
    )
    print(f"platform profiles: OK {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
