from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CAPABILITY = ROOT / "deployment/config/legged.yaml"


def test_legged_v1_is_complete_parametric_runtime_capability() -> None:
    document = yaml.safe_load(CAPABILITY.read_text(encoding="utf-8"))

    assert document == {
        "schema_version": "platform-control-capability-source/v2",
        "platform": {
            "platform_id": "legged",
            "platform_type": "LEGGED",
            "capability_version": "legged-v1",
            "base_frame_id": "base_link",
            "provenance": {
                "level": "approved_mixed_evidence_baseline",
                "design_document": (
                    "docs/superpowers/specs/"
                    "2026-08-06-quad48-legged-platform-capability-design.md"
                ),
                "upstream_commit": "4cc726374ee423c64c6682ef419e16ec2cb49a00",
                "user_spec_sha256": (
                    "1e3db711caf733c685eceef8893a189af4f34476573c38bd91298c06fd6107e5"
                ),
            },
            "unknown_fields": [],
        },
        "geometry_source": {"type": "parametric_envelope"},
        "legged": {
            "reference_point": "base_link",
            "body_extent_m": [0.68, 0.33, 0.35],
            "nominal_body_height_m": 0.33,
            "body_height_m": [0.28, 0.38],
            "platform_mass_kg": 15.89,
            "nominal_payload_kg": 8.0,
            "maximum_payload_kg": 10.0,
            "maximum_forward_speed_mps": 1.5,
            "maximum_reverse_speed_mps": 1.5,
            "maximum_lateral_speed_mps": 0.8,
            "maximum_yaw_rate_radps": 1.0,
            "maximum_linear_acceleration_mps2": 1.0,
            "maximum_yaw_acceleration_radps2": 1.0,
            "maximum_slope_rad": 0.5235987755982988,
            "maximum_step_height_m": 0.5,
            "maximum_gap_width_m": 0.3,
            "minimum_body_clearance_m": 0.3,
            "step_vertical_rate_mps": 0.1,
            "roughness_handling": "DIAGNOSTIC_ONLY",
            "unknown_is_traversable": False,
            "motion_primitives": [
                {
                    "primitive_id": "forward",
                    "kind": "FORWARD",
                    "body_frame_displacement_m": [0.2, 0.0, 0.0],
                    "yaw_change_rad": 0.0,
                },
                {
                    "primitive_id": "backward",
                    "kind": "BACKWARD",
                    "body_frame_displacement_m": [-0.2, 0.0, 0.0],
                    "yaw_change_rad": 0.0,
                },
                {
                    "primitive_id": "lateral-left",
                    "kind": "LATERAL_LEFT",
                    "body_frame_displacement_m": [0.0, 0.2, 0.0],
                    "yaw_change_rad": 0.0,
                },
                {
                    "primitive_id": "lateral-right",
                    "kind": "LATERAL_RIGHT",
                    "body_frame_displacement_m": [0.0, -0.2, 0.0],
                    "yaw_change_rad": 0.0,
                },
                {
                    "primitive_id": "spin-left",
                    "kind": "SPIN",
                    "body_frame_displacement_m": [0.0, 0.0, 0.0],
                    "yaw_change_rad": 0.09817477042468103,
                },
                {
                    "primitive_id": "spin-right",
                    "kind": "SPIN",
                    "body_frame_displacement_m": [0.0, 0.0, 0.0],
                    "yaw_change_rad": -0.09817477042468103,
                },
            ],
        },
        "sources": {
            "reference_point": "upstream_model",
            "body_extent_m": "user_spec_material",
            "platform_mass_kg": "derived",
            "nominal_payload_kg": "user_spec_material",
            "maximum_payload_kg": "user_spec_material",
            "nominal_body_height_m": "derived",
            "body_height_m": "planning_policy",
            "maximum_forward_speed_mps": "planning_policy",
            "maximum_reverse_speed_mps": "planning_policy",
            "maximum_lateral_speed_mps": "upstream_config",
            "maximum_yaw_rate_radps": "upstream_config",
            "maximum_linear_acceleration_mps2": "planning_policy",
            "maximum_yaw_acceleration_radps2": "planning_policy",
            "maximum_surface_slope_rad": "user_spec_material",
            "maximum_step_height_m": "user_confirmed_upgrade",
            "maximum_gap_width_m": "user_spec_material",
            "minimum_body_clearance_m": "planning_policy",
            "step_vertical_rate_mps": "planning_policy",
            "roughness_handling": "planning_policy",
            "unknown_is_traversable": "planning_policy",
            "motion_primitives": "user_approved_planning_policy",
        },
    }
