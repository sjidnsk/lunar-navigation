from __future__ import annotations

import math
import pathlib

from lunar_policy_training.environment.coverability import QualifiedStartState
from lunar_policy_training.polar_data.formal_cache import (
    _global_composed_capability,
)
from lunar_policy_training.project_capability import (
    load_project_formal_capability,
)


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_qualified_start_state_round_trips_exact_platform_fields() -> None:
    start = QualifiedStartState(
        position_m=(12.5, 7.5, 1.25),
        yaw_rad=0.0,
        motion_mode=1,
        body_z_m=(0.28, 0.38),
    )

    assert QualifiedStartState.from_dict(start.to_dict()) == start


def test_ground_global_edges_are_exact_compositions_of_frozen_primitives() -> None:
    import lunar_planner_training_bridge as bridge_api

    bundle = load_project_formal_capability(REPOSITORY_ROOT)
    wheel = _global_composed_capability(
        bundle.for_platform("WHEELED"),
        resolution_m=4.0,
        bridge_api=bridge_api,
    )
    legged = _global_composed_capability(
        bundle.for_platform("LEGGED"),
        resolution_m=4.0,
        bridge_api=bridge_api,
    )

    wheel_translations = [
        primitive
        for primitive in wheel.motion_primitives
        if primitive.kind
        in {
            bridge_api.WheelPrimitiveKind.FORWARD,
            bridge_api.WheelPrimitiveKind.REVERSE,
        }
    ]
    assert wheel_translations
    assert all("/composed-" in value.primitive_id for value in wheel_translations)
    assert all(
        math.isclose(
            math.hypot(
                value.relative_end_pose.position_m.x,
                value.relative_end_pose.position_m.y,
            ),
            4.0,
        )
        for value in wheel_translations
    )
    legged_translations = [
        primitive
        for primitive in legged.motion_primitives
        if math.hypot(
            primitive.body_frame_displacement_m.x,
            primitive.body_frame_displacement_m.y,
        )
        > 0.0
    ]
    assert legged_translations
    assert all("/composed-" in value.primitive_id for value in legged_translations)
    assert all(
        math.isclose(
            math.hypot(
                value.body_frame_displacement_m.x,
                value.body_frame_displacement_m.y,
            ),
            4.0,
        )
        for value in legged_translations
    )
