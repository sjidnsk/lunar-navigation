from pathlib import Path
import math
import yaml


ROOT = Path(__file__).resolve().parents[2]
CAPABILITY = ROOT / "deployment/config/wheel.yaml"


def test_wheel_v1_is_complete_parametric_runtime_capability() -> None:
    document = yaml.safe_load(CAPABILITY.read_text(encoding="utf-8"))
    assert document["schema_version"] == "platform-control-capability-source/v2"
    assert document["platform"] == {
        "platform_id": "wheel",
        "platform_type": "WHEELED",
        "capability_version": "wheel-v1",
        "base_frame_id": "base_footprint",
        "provenance": {
            "level": "approved_user_parameter_baseline",
            "design_document": "docs/superpowers/specs/2026-08-20-wheel-parametric-capability-design.md",
        },
        "unknown_fields": [
            "bare_mass_kg", "nominal_payload_kg", "maximum_payload_kg",
            "center_of_mass_height_m", "suspension_type", "maximum_drive_effort",
            "manufacturer_rated_speed_mps", "longitudinal_traction_coefficient",
            "lateral_traction_coefficient", "verified_cross_slope_limit_rad",
        ],
    }
    assert document["geometry_source"] == {"type": "parametric_envelope"}
    wheel = document["wheeled"]
    assert wheel["body_extent_m"] == [1.301, 0.808, 1.363]
    assert wheel["footprint_xy_m"] == [
        [0.6505, 0.404], [0.6505, -0.404],
        [-0.6505, -0.404], [-0.6505, 0.404],
    ]
    assert wheel["wheel_count"] == 4
    assert wheel["wheel_center_xy_m"] == [
        [0.4075, 0.3115], [0.4075, -0.3115],
        [-0.4075, -0.3115], [-0.4075, 0.3115],
    ]
    assert wheel["maximum_curvature_per_m"] == 5.0
    assert math.isclose(wheel["maximum_spin_rate_radps"], 0.389923188554511, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(wheel["maximum_yaw_acceleration_radps2"], 0.389923188554511, rel_tol=0.0, abs_tol=1e-15)
    assert [item["kind"] for item in wheel["motion_primitives"]] == [
        "FORWARD", "REVERSE", "FORWARD_ARC", "FORWARD_ARC",
        "REVERSE_ARC", "REVERSE_ARC", "SPIN_COUNTERCLOCKWISE",
        "SPIN_CLOCKWISE", "STOP_AND_SWITCH",
    ]
    required_sources = {
        "reference_point", "body_extent_m", "footprint_xy_m", "wheel_count",
        "wheel_diameter_m", "wheel_width_m", "wheelbase_m", "track_width_m",
        "wheel_center_xy_m", "minimum_underbody_clearance_m", "minimum_clearance_m",
        "maximum_forward_speed_mps", "maximum_reverse_speed_mps",
        "maximum_spin_rate_radps", "maximum_acceleration_mps2",
        "maximum_braking_deceleration_mps2", "maximum_yaw_acceleration_radps2",
        "maximum_lateral_acceleration_mps2", "maximum_curvature_per_m",
        "maximum_surface_slope_rad", "maximum_local_obstacle_relief_m",
        "allow_unsupported_gap", "roughness_handling", "motion_primitives",
    }
    assert set(document["sources"]) == required_sources
