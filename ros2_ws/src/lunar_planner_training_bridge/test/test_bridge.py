from __future__ import annotations

from datetime import timedelta
import threading

import numpy as np
import pytest

import lunar_planner_training_bridge as bridge_api


def _vec2(x: float, y: float) -> bridge_api.Vec2:
    value = bridge_api.Vec2()
    value.x = x
    value.y = y
    return value


def _vec3(x: float, y: float, z: float) -> bridge_api.Vec3:
    value = bridge_api.Vec3()
    value.x = x
    value.y = y
    value.z = z
    return value


def _pose(x: float, y: float, z: float) -> bridge_api.Pose3:
    value = bridge_api.Pose3()
    value.position_m = _vec3(x, y, z)
    return value


def _flat_map(
    frame_id: str, *, width: int = 12, height: int = 8, resolution_m: float = 1.0
) -> bridge_api.GridMap:
    cell_count = width * height
    grid = bridge_api.GridMap()
    grid.frame_id = frame_id
    grid.stamp.nanoseconds_since_epoch = 1_000_000_000
    grid.width = width
    grid.height = height
    grid.resolution_m = resolution_m
    grid.layers = {
        "elevation": bridge_api.GridLayer(np.zeros(cell_count, dtype=np.float32)),
        "valid_mask": bridge_api.GridLayer(np.ones(cell_count, dtype=np.uint8)),
        "obstacle": bridge_api.GridLayer(np.zeros(cell_count, dtype=np.uint8)),
        "obstacle_height": bridge_api.GridLayer(
            np.zeros(cell_count, dtype=np.float32)
        ),
        "observation_age_s": bridge_api.GridLayer(
            np.zeros(cell_count, dtype=np.float32)
        ),
        "observation_quality": bridge_api.GridLayer(
            np.ones(cell_count, dtype=np.float32)
        ),
        "elevation_variance": bridge_api.GridLayer(
            np.zeros(cell_count, dtype=np.float32)
        ),
        "obstacle_variance": bridge_api.GridLayer(
            np.zeros(cell_count, dtype=np.float32)
        ),
        "observation_count": bridge_api.GridLayer(
            np.ones(cell_count, dtype=np.uint32)
        ),
        "forbidden": bridge_api.GridLayer(np.zeros(cell_count, dtype=np.uint8)),
    }
    return grid


def _wheel_capability() -> bridge_api.WheeledCapability:
    capability = bridge_api.WheeledCapability()
    capability.footprint_xy_m = [
        _vec2(-0.2, -0.2),
        _vec2(0.2, -0.2),
        _vec2(0.2, 0.2),
        _vec2(-0.2, 0.2),
    ]
    capability.minimum_body_z_m = -0.1
    capability.maximum_body_z_m = 0.5
    capability.maximum_forward_speed_mps = 1.0
    capability.maximum_reverse_speed_mps = 0.8
    capability.maximum_spin_rate_radps = 1.0
    capability.maximum_acceleration_mps2 = 1.0
    capability.maximum_braking_deceleration_mps2 = 1.0
    capability.maximum_yaw_acceleration_radps2 = 1.0
    capability.maximum_lateral_acceleration_mps2 = 1.0
    capability.maximum_curvature_per_m = 1.0
    capability.maximum_slope_rad = 0.5
    primitive = bridge_api.WheelMotionPrimitive()
    primitive.primitive_id = "forward"
    primitive.kind = bridge_api.WheelPrimitiveKind.FORWARD
    primitive.relative_end_pose = _pose(1.0, 0.0, 0.0)
    primitive.nominal_duration = timedelta(seconds=1)
    capability.motion_primitives = [primitive]
    return capability


def _legged_capability() -> bridge_api.LeggedCapability:
    capability = bridge_api.LeggedCapability()
    capability.body_half_extent_m = _vec3(0.2, 0.2, 0.3)
    capability.maximum_slope_rad = 0.4
    capability.maximum_roughness_m = 0.2
    capability.maximum_step_height_m = 0.3
    capability.maximum_gap_width_m = 0.4
    capability.minimum_confidence = 0.8
    capability.minimum_body_clearance_m = 0.1
    capability.body_height_m.lower = 0.4
    capability.body_height_m.upper = 0.6
    capability.forward_speed_mps.lower = -0.5
    capability.forward_speed_mps.upper = 0.5
    capability.lateral_speed_mps.lower = -0.5
    capability.lateral_speed_mps.upper = 0.5
    capability.vertical_speed_mps.lower = -0.1
    capability.vertical_speed_mps.upper = 0.1
    capability.yaw_rate_radps.lower = -1.0
    capability.yaw_rate_radps.upper = 1.0
    capability.maximum_linear_acceleration_mps2 = 0.5
    capability.maximum_yaw_acceleration_radps2 = 1.0
    primitive = bridge_api.LeggedBodyPrimitive()
    primitive.primitive_id = "forward"
    primitive.kind = bridge_api.LeggedPrimitiveKind.FORWARD
    primitive.body_frame_displacement_m = _vec3(1.0, 0.0, 0.0)
    primitive.nominal_duration = timedelta(seconds=2)
    capability.motion_primitives = [primitive]
    return capability


def _hopper_capability() -> bridge_api.HopperCapability:
    capability = bridge_api.HopperCapability()
    capability.body_half_extent_m = _vec3(0.35, 0.25, 0.5)
    capability.platform_mass_kg = 10.0
    capability.gravity_mps2 = _vec3(0.0, 0.0, -1.62)
    capability.maximum_landing_slope_rad = 0.4
    capability.maximum_landing_roughness_m = 0.1
    capability.maximum_plane_residual_m = 0.05
    capability.minimum_landing_region_area_m2 = 0.2
    capability.maximum_launch_speed_mps = 8.0
    capability.maximum_launch_impulse_newton_seconds = 100.0
    capability.minimum_flight_time = timedelta(milliseconds=500)
    capability.maximum_flight_time = timedelta(seconds=10)
    capability.maximum_landing_speed_mps = 8.0
    capability.minimum_downward_impact_speed_mps = 0.1
    capability.maximum_angular_speed_radps = 2.0
    capability.maximum_angular_acceleration_radps2 = 4.0
    capability.maximum_initial_angular_speed_radps = 0.2
    capability.minimum_settle_guard = timedelta(milliseconds=100)
    return capability


@pytest.fixture
def easy_request():
    def make(platform_type: str) -> bridge_api.TrainingPlanRequest:
        request = bridge_api.TrainingPlanRequest()
        request.request_id = f"training-{platform_type.lower()}"
        request.state_time.nanoseconds_since_epoch = 1_000_000_000

        goal = bridge_api.PointGoal()
        goal.tolerance_m = 0.2
        if platform_type == "HOPPER":
            goal.position_m = _vec3(4.0, 3.0, 0.0)
            goal.tolerance_m = 0.5
        else:
            goal.position_m = _vec3(4.5, 3.5, 0.0)
        request.goal.goal_id = "goal"
        request.goal.target = goal

        if platform_type == "WHEELED":
            state = bridge_api.WheeledState()
            state.pose = _pose(2.5, 3.5, 0.0)
            request.current_state = state
            request.capability = _wheel_capability()
        elif platform_type == "LEGGED":
            state = bridge_api.LeggedState()
            state.body_pose = _pose(2.5, 3.5, 0.5)
            request.current_state = state
            request.capability = _legged_capability()
        else:
            state = bridge_api.HopperState()
            state.pose = _pose(3.0, 3.0, 0.5)
            request.current_state = state
            request.capability = _hopper_capability()
            request.goal.yaw_rad = 0.0
            request.goal.yaw_tolerance_rad = 0.1

        map_kwargs = (
            {"width": 16, "height": 12, "resolution_m": 0.5}
            if platform_type == "HOPPER"
            else {}
        )
        request.world.global_map = _flat_map("map", **map_kwargs)
        request.world.local_map = _flat_map("odom", **map_kwargs)
        request.world.map_from_odom.parent_frame = "map"
        request.world.map_from_odom.child_frame = "odom"
        request.world.map_from_odom.stamp.nanoseconds_since_epoch = 1_000_000_000
        request.config.wheel.xy_resolution_m = 1.0
        request.config.legged.xy_resolution_m = 1.0
        return request

    return make


@pytest.fixture
def bridge() -> bridge_api.PlannerBridge:
    return bridge_api.PlannerBridge()


@pytest.mark.parametrize("platform_type", ["WHEELED", "LEGGED", "HOPPER"])
def test_bridge_uses_matching_v3_planner(bridge, easy_request, platform_type):
    """Would fail if the bridge bypassed C++ v3 or returned another platform."""
    request = easy_request(platform_type)
    output = bridge.plan(request)
    if output.reference is not None:
        assert output.reference.platform_type == platform_type
    assert output.diagnostics.planner_name == "cpp_v3"


@pytest.mark.parametrize("platform_type", ["WHEELED", "LEGGED", "HOPPER"])
def test_bridge_projects_traversability_with_exact_array_contract(
    bridge, easy_request, platform_type
) -> None:
    """Would fail if projection crossed Python with wrong platform/shape/dtype."""
    request = easy_request(platform_type)

    projection = bridge.project_traversability(request)

    expected_shape = (
        request.world.local_map.height,
        request.world.local_map.width,
    )
    assert projection.platform_type == platform_type
    assert projection.width == expected_shape[1]
    assert projection.height == expected_shape[0]
    expected_dtypes = {
        "known": np.uint8,
        "hard_feasible": np.uint8,
        "clearance_m": np.float32,
        "slope_rad": np.float32,
        "roughness_m": np.float32,
        "traversal_cost": np.float32,
        "connected_component": np.int32,
    }
    for field, dtype in expected_dtypes.items():
        values = getattr(projection, field)
        assert values.shape == expected_shape
        assert values.dtype == dtype
        assert values.flags.c_contiguous


def test_bridge_projection_rejects_invalid_map_with_stable_reason(
    bridge, easy_request
) -> None:
    """Would fail if projection errors lost the C++ map reason code."""
    request = easy_request("WHEELED")
    layers = request.world.local_map.layers
    del layers["forbidden"]
    request.world.local_map.layers = layers

    with pytest.raises(RuntimeError, match="MISSING_MAP_LAYER_FORBIDDEN"):
        bridge.project_traversability(request)


def test_bridge_projection_releases_gil(bridge, easy_request) -> None:
    """Would fail if a real C++ projection blocked every Python thread."""
    request = easy_request("WHEELED")
    request.world.local_map = _flat_map("odom", width=640, height=640)
    started = threading.Event()
    stop = threading.Event()
    counter = [0]

    def count_python_work() -> None:
        started.set()
        while not stop.is_set():
            counter[0] += 1

    worker = threading.Thread(target=count_python_work)
    worker.start()
    assert started.wait(timeout=1.0)
    before = counter[0]
    try:
        bridge.project_traversability(request)
    finally:
        stop.set()
        worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert counter[0] > before


def test_grid_layer_rejects_non_contiguous_numpy() -> None:
    """Would fail if a strided view crossed the zero-copy-sensitive boundary."""
    values = np.zeros((4, 4), dtype=np.float32)[:, ::2]

    with pytest.raises(ValueError, match="C-contiguous"):
        bridge_api.GridLayer(values)


def test_grid_layer_rejects_inexact_numpy_dtype() -> None:
    """Would fail if float64 were silently narrowed at the planner boundary."""
    with pytest.raises(TypeError, match="float32, uint8, or uint32"):
        bridge_api.GridLayer(np.zeros(4, dtype=np.float64))


@pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.uint32])
def test_grid_layer_preserves_exact_numpy_dtype(dtype) -> None:
    """Would fail if accepted layer values were silently widened or narrowed."""
    values = np.arange(6, dtype=dtype)

    copied = bridge_api.GridLayer(values).values

    assert copied.dtype == values.dtype
    assert copied.flags.c_contiguous
    assert np.array_equal(copied, values)


def test_committed_hopper_request_uses_execution_context(
    bridge, easy_request
) -> None:
    """Would fail if the bridge dropped protected hopper execution state."""
    request = easy_request("HOPPER")
    execution = bridge_api.HopperExecutionContext()
    execution.state = bridge_api.HopperExecutionState.JUMP_COMMITTED
    execution.active_plan_id = "active-plan"
    execution.active_segment_id = "active-hop"
    request.previous_execution = execution

    output = bridge.plan(request)

    assert output.outcome == bridge_api.PlanningOutcome.SAFE_FRONTIER_REFERENCE_AVAILABLE
    assert output.directive == bridge_api.ExecutionDirective.CONTINUE_COMMITTED_HOP
    assert output.reason_code == "COMMITTED_HOP_CONTINUES"
    assert output.reference is None


def test_public_outcome_and_directive_names_are_exact() -> None:
    """Would fail if Python reward/execution routing saw renamed enum members."""
    assert set(bridge_api.PlanningOutcome.__members__) == {
        "NEW_REFERENCE_AVAILABLE",
        "SAFE_FRONTIER_REFERENCE_AVAILABLE",
        "NO_KNOWN_SAFE_ROUTE",
        "GOAL_INFEASIBLE",
        "INVALID_REQUEST",
        "STALE_INPUT",
        "NUMERICAL_FAILURE",
        "RESOURCE_EXHAUSTED",
        "ACTIVE_REFERENCE_INVALIDATED",
        "CANCELED",
    }
    assert set(bridge_api.ExecutionDirective.__members__) == {
        "ACTIVATE_NEW_REFERENCE",
        "CONTINUE_ACTIVE_REFERENCE",
        "HOLD_POSITION",
        "CONTINUE_COMMITTED_HOP",
        "NO_SAFE_REFERENCE",
    }
