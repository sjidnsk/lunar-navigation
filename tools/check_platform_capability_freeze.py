#!/usr/bin/env python3
"""Validate the approved three-platform capability consumer freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


EXPECTED_PLATFORMS = ("LEGGED", "HOPPER", "WHEELED")
EXPECTED_VALUES: dict[str, dict[str, Any]] = {
    "WHEELED": {
        "platform_id": "wheeled-lunar-explorer",
        "platform_type": "WHEELED",
        "capability_version": "wheeled-engineering-baseline-v1",
        "base_frame_id": "base_footprint",
        "geometry.reference_point": "base_footprint",
        "geometry.body_extent_m": [1.182, 0.818, 1.29996],
        "geometry.footprint_xy_m": [
            [0.591, 0.409],
            [0.591, -0.409],
            [-0.591, -0.409],
            [-0.591, 0.409],
        ],
        "geometry.wheel_count": 4,
        "geometry.wheel_diameter_m": 0.319,
        "geometry.wheel_width_m": 0.148,
        "geometry.wheelbase_m": 0.8175,
        "geometry.track_width_m": 0.67,
        "geometry.wheel_center_xy_m": [
            [0.40875, 0.335],
            [0.40875, -0.335],
            [-0.40875, -0.335],
            [-0.40875, 0.335],
        ],
        "geometry.minimum_underbody_clearance_m": 0.21,
        "geometry.minimum_clearance_m": 0.2,
        "kinematics.maximum_forward_speed_mps": 1.5,
        "kinematics.maximum_reverse_speed_mps": 1.5,
        "kinematics.maximum_spin_rate_radps": 1.0,
        "kinematics.maximum_acceleration_mps2": 0.5,
        "kinematics.maximum_braking_deceleration_mps2": 0.5,
        "kinematics.maximum_yaw_acceleration_radps2": 0.5,
        "kinematics.maximum_lateral_acceleration_mps2": 0.5,
        "kinematics.maximum_curvature_per_m": 1.0,
        "terrain.maximum_surface_slope_rad": 0.3490658503988659,
        "terrain.maximum_local_obstacle_relief_m": 0.2,
        "terrain.allow_unsupported_gap": False,
        "terrain.roughness_handling": "COST_SPEED_AND_LOCAL_RECHECK",
        "motion.xy_resolution_m": 0.2,
        "motion.yaw_bin_count": 32,
        "motion.arc_radius_m": 1.0,
        "motion.arc_yaw_change_rad": 0.19634954084936207,
        "motion.primitive_types": [
            "FORWARD",
            "REVERSE",
            "FORWARD_ARC",
            "REVERSE_ARC",
            "SPIN_CLOCKWISE",
            "SPIN_COUNTERCLOCKWISE",
            "STOP_AND_SWITCH",
        ],
    },
    "LEGGED": {
        "platform_id": "yobotics-quad48",
        "platform_type": "LEGGED",
        "capability_version": "quad48-approved-baseline-v1",
        "base_frame_id": "base_link",
        "geometry.reference_point": "base_link",
        "geometry.body_extent_m": [0.68, 0.33, 0.35],
        "geometry.nominal_body_height_m": 0.33,
        "geometry.body_height_m": [0.28, 0.38],
        "physical.platform_mass_kg": 15.89,
        "physical.nominal_payload_kg": 8.0,
        "physical.maximum_payload_kg": 10.0,
        "kinematics.maximum_forward_speed_mps": 1.5,
        "kinematics.maximum_reverse_speed_mps": 1.5,
        "kinematics.maximum_lateral_speed_mps": 0.8,
        "kinematics.maximum_yaw_rate_radps": 1.0,
        "kinematics.maximum_linear_acceleration_mps2": 1.0,
        "kinematics.maximum_yaw_acceleration_radps2": 1.0,
        "terrain.maximum_surface_slope_rad": 0.5235987755982988,
        "terrain.maximum_step_height_m": 0.5,
        "terrain.maximum_gap_width_m": 0.3,
        "terrain.minimum_body_clearance_m": 0.3,
        "terrain.step_vertical_rate_mps": 0.1,
        "terrain.roughness_handling": "DIAGNOSTIC_ONLY",
        "terrain.unknown_is_traversable": False,
        "motion.local_xy_resolution_m": 0.2,
        "motion.output_semantics": "LEGGED_BODY_REFERENCE",
    },
    "HOPPER": {
        "platform_id": "hopper-lunar-explorer",
        "platform_type": "HOPPER",
        "capability_version": "hopper-engineering-baseline-v1",
        "base_frame_id": "base_link",
        "propulsion.specific_impulse_s": 301.0,
        "landing.landing_support_radius_m": 0.45,
        "landing.maximum_landing_slope_rad": 0.17453292519943295,
        "landing.maximum_landing_plane_residual_m": 0.05,
        "safety.flight_collision_radius_m": 0.55,
        "safety.landing_lateral_margin_m": 0.2,
        "safety.flight_map_margin_m": 0.2,
        "safety.reachability_delta_v_margin_ratio": 0.1,
        "environment.gravity_mps2": [0.0, 0.0, -1.62],
        "environment.standard_gravity_mps2": 9.80665,
        "reference_conditions.reference_total_mass_kg": 20.0,
        "reference_conditions.reference_remaining_usable_fuel_mass_kg": 0.2,
        "reference_conditions.reference_horizontal_range_m": 100.0,
        "reference_conditions.reference_elevation_delta_m": 0.0,
        "reference_conditions.runtime_fallback_allowed": False,
    },
}


def load_mapping(path: Path, label: str) -> tuple[Mapping[str, Any] | None, list[str]]:
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
    if isinstance(expected, list):
        return (
            isinstance(actual, Sequence)
            and not isinstance(actual, (str, bytes))
            and len(actual) == len(expected)
            and all(equal_value(item, wanted) for item, wanted in zip(actual, expected))
        )
    return actual == expected


def nested_keys(document: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(document, Mapping):
        for key, value in document.items():
            keys.add(str(key))
            keys.update(nested_keys(value))
    elif isinstance(document, Sequence) and not isinstance(document, (str, bytes)):
        for value in document:
            keys.update(nested_keys(value))
    return keys


def platform_digest(platforms: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        platforms,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate(schema: Mapping[str, Any], freeze: Mapping[str, Any]) -> tuple[list[str], str]:
    errors: list[str] = []
    if schema.get("schema_version") != "platform-control-capability-source/v2":
        errors.append("schema_version must be platform-control-capability-source/v2")
    if schema.get("freeze_schema_version") != "lunar-platform-capability-freeze/v1":
        errors.append("freeze_schema_version must be lunar-platform-capability-freeze/v1")
    if freeze.get("schema_version") != "lunar-platform-capability-freeze/v1":
        errors.append("freeze schema_version must be lunar-platform-capability-freeze/v1")
    if freeze.get("capability_schema") != schema.get("schema_version"):
        errors.append("freeze capability_schema does not match schema_version")

    schema_platforms = schema.get("platforms")
    platforms = freeze.get("platforms")
    if not isinstance(schema_platforms, Mapping):
        errors.append("schema platforms must be a mapping")
        schema_platforms = {}
    if not isinstance(platforms, Mapping):
        errors.append("freeze platforms must be a mapping")
        platforms = {}
    if set(platforms) != set(EXPECTED_PLATFORMS):
        errors.append("platform set must be exactly LEGGED, HOPPER, WHEELED")

    allowed_source_types = schema.get("allowed_source_types")
    if not isinstance(allowed_source_types, Sequence) or isinstance(
        allowed_source_types, (str, bytes)
    ):
        errors.append("allowed_source_types must be a sequence")
        allowed_sources: set[str] = set()
    else:
        allowed_sources = {
            source for source in allowed_source_types if isinstance(source, str)
        }

    versions: list[str] = []
    for platform_name in EXPECTED_PLATFORMS:
        platform = platforms.get(platform_name)
        platform_schema = schema_platforms.get(platform_name)
        if not isinstance(platform, Mapping) or not isinstance(platform_schema, Mapping):
            continue
        allowed_keys = platform_schema.get("allowed_keys", [])
        unexpected = sorted(set(platform) - set(allowed_keys))
        if unexpected:
            errors.append(
                f"{platform_name} has unexpected keys: {', '.join(unexpected)}"
            )

        retired_fields = set(platform_schema.get("retired_fields", []))
        present_retired = sorted(nested_keys(platform) & retired_fields)
        if present_retired:
            errors.append(
                f"{platform_name} contains retired fields: {', '.join(present_retired)}"
            )

        for dotted_path, expected in EXPECTED_VALUES[platform_name].items():
            actual = get_path(platform, dotted_path)
            if not equal_value(actual, expected):
                errors.append(
                    f"{platform_name}.{dotted_path} must be {expected!r}"
                )

        expected_unknown = platform_schema.get("exact_unknown_fields", [])
        if platform.get("unknown_fields") != expected_unknown:
            errors.append(
                f"{platform_name} unknown_fields must be {expected_unknown!r}"
            )

        required_sources = set(platform_schema.get("required_sources", []))
        sources = platform.get("sources")
        if not isinstance(sources, Mapping):
            errors.append(f"{platform_name} sources must be a mapping")
        else:
            missing_sources = sorted(required_sources - set(sources))
            extra_sources = sorted(set(sources) - required_sources)
            if missing_sources:
                errors.append(
                    f"{platform_name} sources missing: {', '.join(missing_sources)}"
                )
            if extra_sources:
                errors.append(
                    f"{platform_name} sources unexpected: {', '.join(extra_sources)}"
                )
            for field, source in sources.items():
                if source not in allowed_sources:
                    errors.append(
                        f"{platform_name}.{field} has unsupported source type: {source!r}"
                    )

        provenance = platform.get("provenance")
        if not isinstance(provenance, Mapping):
            errors.append(f"{platform_name} provenance must be a mapping")
        else:
            design_document = provenance.get("design_document")
            if not isinstance(design_document, str) or not Path(design_document).is_file():
                errors.append(f"{platform_name} design_document is not available")

        version = platform.get("capability_version")
        if isinstance(version, str):
            versions.append(version)

    if len(versions) != len(set(versions)):
        errors.append("capability versions must be unique")

    try:
        digest = platform_digest(platforms)
    except (TypeError, ValueError) as error:
        errors.append(f"platform payload is not canonicalizable: {error}")
        digest = ""
    if freeze.get("freeze_digest_sha256") != digest:
        errors.append(
            "freeze_digest_sha256 does not match platform payload: "
            f"expected {digest}"
        )
    return errors, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    arguments = parser.parse_args()

    schema, schema_errors = load_mapping(arguments.schema, "schema")
    freeze, freeze_errors = load_mapping(arguments.freeze, "freeze")
    errors = [*schema_errors, *freeze_errors]
    digest = ""
    if schema is not None and freeze is not None:
        validation_errors, digest = validate(schema, freeze)
        errors.extend(validation_errors)
    if errors:
        for error in errors:
            print(f"capability freeze error: {error}")
        return 1
    print(f"capability freeze: OK sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
