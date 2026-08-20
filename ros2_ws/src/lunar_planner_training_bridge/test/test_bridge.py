from __future__ import annotations

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
        _vec2(-0.591, -0.409),
        _vec2(0.591, -0.409),
        _vec2(0.591, 0.409),
        _vec2(-0.591, 0.409),
    ]
    capability.body_extent_m = _vec3(1.182, 0.818, 1.29996)
    capability.wheel_diameter_m = 0.319
    capability.wheel_width_m = 0.148
    capability.wheelbase_m = 0.8175
    capability.track_width_m = 0.67
    capability.minimum_underbody_clearance_m = 0.21
    capability.maximum_local_obstacle_relief_m = 0.2
    capability.allow_unsupported_gap = False
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
    capability.motion_primitives = [primitive]
    return capability


def _legged_capability() -> bridge_api.LeggedCapability:
    capability = bridge_api.LeggedCapability()
    capability.body_extent_m = _vec3(0.68, 0.33, 0.35)
    capability.nominal_body_height_m = 0.33
    capability.platform_mass_kg = 15.89
    capability.nominal_payload_kg = 8.0
    capability.maximum_payload_kg = 10.0
    capability.maximum_slope_rad = 0.5235987755982988
    capability.maximum_step_height_m = 0.5
    capability.maximum_gap_width_m = 0.3
    capability.minimum_body_clearance_m = 0.3
    capability.step_vertical_rate_mps = 0.1
    capability.body_height_m.lower = 0.28
    capability.body_height_m.upper = 0.38
    capability.forward_speed_mps.lower = -1.5
    capability.forward_speed_mps.upper = 1.5
    capability.lateral_speed_mps.lower = -0.8
    capability.lateral_speed_mps.upper = 0.8
    capability.yaw_rate_radps.lower = -1.0
    capability.yaw_rate_radps.upper = 1.0
    capability.maximum_linear_acceleration_mps2 = 0.5
    capability.maximum_yaw_acceleration_radps2 = 1.0
    capability.unknown_is_traversable = False
    primitive = bridge_api.LeggedBodyPrimitive()
    primitive.primitive_id = "forward"
    primitive.kind = bridge_api.LeggedPrimitiveKind.FORWARD
    primitive.body_frame_displacement_m = _vec3(1.0, 0.0, 0.0)
    capability.motion_primitives = [primitive]
    return capability


def _hopper_capability() -> bridge_api.HopperCapability:
    capability = bridge_api.HopperCapability()
    capability.specific_impulse_s = 301.0
    capability.reference_total_mass_kg = 20.0
    capability.reference_propellant_mass_kg = 0.2
    capability.landing_support_radius_m = 0.45
    capability.flight_collision_radius_m = 0.55
    capability.maximum_landing_plane_residual_m = 0.05
    capability.landing_lateral_margin_m = 0.2
    capability.flight_map_margin_m = 0.2
    capability.reachability_delta_v_margin_ratio = 0.1
    capability.standard_gravity_mps2 = 9.80665
    capability.maximum_landing_slope_rad = 0.17453292519943295
    return capability


def test_bridge_exposes_capability_v2_without_retired_limits_or_durations() -> None:
    """The imported Volume 3 bridge must follow the current planner contract."""
    wheel = bridge_api.WheelMotionPrimitive()
    legged = bridge_api.LeggedBodyPrimitive()
    hopper = bridge_api.HopperCapability()
    assert not hasattr(wheel, "nominal_duration")
    assert not hasattr(legged, "nominal_duration")
    assert not hasattr(hopper, "maximum_launch_impulse_newton_seconds")
    assert not hasattr(bridge_api, "SearchResourceLimits")
    assert not hasattr(bridge_api.WheelPlannerConfig(), "maximum_terminal_candidates")
    assert not hasattr(bridge_api.HopperPlannerConfig(), "maximum_graph_nodes")


def test_urdf_validation_uses_authoritative_model_parser() -> None:
    """Would fail if freeze accepted XML that urdf::Model rejects."""
    valid = (
        '<robot name="test"><link name="base_link"><visual><geometry>'
        '<mesh filename="body.stl" scale="1 1 1"/>'
        '</geometry></visual></link></robot>'
    )
    missing_name = valid.replace(' name="test"', "", 1)

    assert bridge_api.validate_urdf_geometry(valid, "base_link") == (
        "body.stl",
    )
    with pytest.raises(ValueError, match="URDF"):
        bridge_api.validate_urdf_geometry(missing_name, "base_link")


def test_visibility_binding_preserves_exact_batch_and_reveal_contract() -> None:
    kernel = bridge_api.VisibilityKernel(1.0, 3.0)
    observed = np.zeros((7, 7), dtype=np.bool_)
    obstacle = np.zeros((7, 7), dtype=np.float32)
    roi = np.zeros((7, 7), dtype=np.float32)
    priority = np.zeros((7, 7), dtype=np.float32)
    observed[3, 3:5] = True
    roi[3, 5] = 0.25
    priority[3, 5] = 0.75
    roi[3, 6] = 1.0
    priority[3, 6] = 1.0
    candidates = np.asarray(((3, 3), (1, 1)), dtype=np.int32)

    gains = kernel.estimate_candidate_gains(
        observed, obstacle, roi, priority, candidates
    )
    known_block = obstacle.copy()
    known_block[3, 4] = 0.001
    blocked_gains = kernel.estimate_candidate_gains(
        observed, known_block, roi, priority, candidates
    )
    obstacle[3, 5] = 0.001
    visible = kernel.reveal_from_pose(obstacle, 3, 3)

    assert kernel.resolution_m == 1.0
    assert kernel.range_m == 3.0
    assert gains.shape == (2, 2)
    assert gains.dtype == np.float32
    assert gains.flags.c_contiguous
    np.testing.assert_array_equal(gains[0], (1.25, 1.75))
    np.testing.assert_array_equal(gains[1], (0.0, 0.0))
    np.testing.assert_array_equal(blocked_gains[0], (0.0, 0.0))
    assert visible.shape == (7, 7)
    assert visible.dtype == np.bool_
    assert visible.flags.c_contiguous
    assert visible[3, 5]
    assert not visible[3, 6]


def test_visibility_binding_reveals_batch_exactly_as_ordered_single_poses() -> None:
    kernel = bridge_api.VisibilityKernel(1.0, 3.0)
    truth = np.zeros((3, 7, 7), dtype=np.float32)
    truth[1, 3, 5] = 0.001
    truth[2, 0, 1] = 0.001
    poses = np.asarray(((3, 3), (3, 3), (0, 0)), dtype=np.int32)

    batched = kernel.reveal_from_poses(truth, poses)

    assert batched.shape == truth.shape
    assert batched.dtype == np.bool_
    assert batched.flags.c_contiguous
    for index, (row, column) in enumerate(poses):
        np.testing.assert_array_equal(
            batched[index],
            kernel.reveal_from_pose(truth[index], int(row), int(column)),
        )


def test_grid_map_patches_sorted_flat_indices_without_rebuilding_layers() -> None:
    grid = bridge_api.GridMap()
    grid.width = 3
    grid.height = 2
    grid.layers = {
        "observation_age_s": bridge_api.GridLayer(
            np.zeros((6,), dtype=np.float32)
        ),
    }
    indices = np.asarray((1, 5), dtype=np.uint32)
    values = np.asarray((0.25, 1.5), dtype=np.float32)

    grid.patch_layer_flat_indices("observation_age_s", indices, values)

    np.testing.assert_array_equal(
        grid.layers["observation_age_s"].values,
        np.asarray((0.0, 0.25, 0.0, 0.0, 0.0, 1.5), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        grid.patch_layer_flat_indices(
            "observation_age_s",
            np.asarray((1, 1), dtype=np.uint32),
            values,
        )
    with pytest.raises(TypeError, match="float32"):
        grid.patch_layer_flat_indices(
            "observation_age_s",
            indices,
            values.astype(np.float64),
        )


@pytest.mark.parametrize(
    ("argument", "values", "message"),
    (
        ("observed", np.zeros((7, 7), dtype=np.uint8), "bool"),
        ("obstacle", np.zeros((7, 7), dtype=np.float64), "float32"),
        ("candidates", np.zeros((2, 2), dtype=np.int64), "int32"),
        (
            "roi",
            np.zeros((7, 14), dtype=np.float32)[:, ::2],
            "C-contiguous",
        ),
    ),
)
def test_visibility_binding_rejects_inexact_or_strided_arrays(
    argument: str, values: np.ndarray, message: str
) -> None:
    kernel = bridge_api.VisibilityKernel(1.0, 3.0)
    arguments = {
        "observed": np.zeros((7, 7), dtype=np.bool_),
        "obstacle": np.zeros((7, 7), dtype=np.float32),
        "roi": np.zeros((7, 7), dtype=np.float32),
        "priority": np.zeros((7, 7), dtype=np.float32),
        "candidates": np.asarray(((3, 3),), dtype=np.int32),
    }
    arguments[argument] = values

    with pytest.raises((TypeError, ValueError), match=message):
        kernel.estimate_candidate_gains(*arguments.values())


@pytest.fixture
def easy_request():
    def make(platform_type: str) -> bridge_api.TrainingPlanRequest:
        request = bridge_api.TrainingPlanRequest()
        request.request_id = f"training-{platform_type.lower()}"
        request.mission_id = "training-mission"
        request.mission_revision = 1
        request.platform_id = f"training-{platform_type.lower()}"
        request.capability_version = "capability-v2"
        request.global_map_generation = 1
        request.local_map_generation = 1
        request.map_from_odom_generation = 1
        request.state_time.nanoseconds_since_epoch = 1_000_000_000

        goal = bridge_api.PointGoal()
        goal.tolerance_m = 0.2
        if platform_type == "HOPPER":
            goal.position_m = _vec3(4.0, 3.0, 0.0)
            goal.tolerance_m = 0.0
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
        request.config.global_map.base_resolution_m = request.world.global_map.resolution_m
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
    assert output.diagnostics.planner_name == "cpp_v3_hierarchical"


def test_bridge_exposes_candidate_disposition_and_complete_hierarchical_metrics(
    bridge, easy_request
) -> None:
    """Would fail if Task 3 metrics or structured disposition stopped at C++."""
    request = easy_request("WHEELED")
    request.world.local_map = _flat_map("odom", width=24, height=16)
    request.world.local_map.origin_m.x = -4.0

    output = bridge.plan(request)

    assert output.outcome == bridge_api.PlanningOutcome.NEW_REFERENCE_AVAILABLE
    assert output.candidate_disposition == bridge_api.CandidateDisposition.KEEP
    metrics = output.diagnostics.hierarchical
    assert metrics is not None
    expected_fields = {
        "global_level",
        "global_resolution_m",
        "global_cells",
        "global_elapsed",
        "local_elapsed",
        "global_expanded_states",
        "local_expanded_states",
        "global_open_peak",
        "estimated_work_memory_bytes",
        "raw_route_points",
        "simplified_route_points",
        "local_frontier_distance_m",
        "additional_corridor_margin_m",
        "corridor_half_width_m",
        "search_domain_cell_count",
        "search_domain_sha256",
        "local_frontier_attempts",
        "local_search_runs",
        "global_replans",
        "physical_goal_feasible",
        "local_attempts",
        "corridor_width_m",
        "global_projection_cache_hits",
        "local_projection_cache_hits",
        "hopper_graph_nodes",
        "hopper_graph_edges",
        "hopper_route_hops",
        "hopper_certification_attempts",
        "landing_field_elapsed",
        "spatial_index_elapsed",
        "ballistic_solve_elapsed",
        "flight_tube_certification_elapsed",
        "safe_landing_nodes",
        "candidate_edges_evaluated",
        "coarse_edges_rejected",
        "full_edges_certified",
        "full_edges_invalidated",
        "edge_certificate_cache_hits",
        "route_reused",
        "route_cursor",
        "rolling_request_count",
    }
    assert all(hasattr(metrics, field) for field in expected_fields)
    assert metrics.local_search_runs > 0
    assert metrics.search_domain_cell_count > 0
    assert len(metrics.search_domain_sha256) == 64
    assert metrics.physical_goal_feasible


def test_bridge_carries_an_opaque_ground_route_continuation(bridge, easy_request) -> None:
    request = easy_request("WHEELED")
    request.world.global_map = _flat_map("map", width=24, height=8)
    request.world.local_map = _flat_map("odom", width=24, height=16)
    request.world.local_map.origin_m.x = -4.0
    request.goal.target.position_m = _vec3(18.5, 3.5, 0.0)

    first = bridge.plan(request)

    assert first.reference is not None
    assert first.continuation is not None
    assert not hasattr(first.continuation, "global_route")
    request.request_id = "training-wheeled-continuation"
    request.local_map_generation += 1
    request.current_state.pose = _pose(2.6, 3.5, 0.0)
    request.continuation = first.continuation

    second = bridge.plan(request)

    assert second.diagnostics.hierarchical.route_reused is True
    assert second.diagnostics.hierarchical.global_expanded_states == 0


def test_bridge_suppresses_only_an_exhausted_physical_target(
    bridge, easy_request
) -> None:
    request = easy_request("WHEELED")
    for row in range(request.world.global_map.height):
        _set_map_byte(request.world.global_map, "forbidden", row, 3, 1)

    exhausted = bridge.plan(request)

    assert exhausted.outcome == bridge_api.PlanningOutcome.NO_KNOWN_SAFE_ROUTE
    assert exhausted.candidate_disposition == (
        bridge_api.CandidateDisposition.SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT
    )

    invalid = easy_request("WHEELED")
    layers = invalid.world.local_map.layers
    del layers["forbidden"]
    invalid.world.local_map.layers = layers

    rejected = bridge.plan(invalid)

    assert rejected.outcome == bridge_api.PlanningOutcome.INVALID_REQUEST
    assert rejected.candidate_disposition == bridge_api.CandidateDisposition.KEEP

    numerical = easy_request("WHEELED")
    numerical.config.global_search.slope_weight = float("nan")

    failed = bridge.plan(numerical)

    assert failed.outcome == bridge_api.PlanningOutcome.NUMERICAL_FAILURE
    assert failed.candidate_disposition == bridge_api.CandidateDisposition.KEEP


def test_hopper_request_has_repeatable_delta_v_and_no_dynamic_propellant_contract(
    bridge, easy_request
) -> None:
    request = easy_request("HOPPER")

    assert not hasattr(request, "hopper_propellant")
    assert not hasattr(bridge_api, "HopperPropellantState")
    first = bridge.plan(request)
    request.request_id = "training-hopper-repeat"
    second = bridge.plan(request)
    assert first.reference is not None, first.reason_code
    assert second.reference is not None, second.reason_code
    first_hop = first.reference.data.segments[0]
    second_hop = second.reference.data.segments[0]
    assert first_hop.available_delta_v_mps == second_hop.available_delta_v_mps
    assert first_hop.required_delta_v_mps == second_hop.required_delta_v_mps
    assert not hasattr(first_hop, "expected_remaining_usable_fuel_kg")


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
        "intrinsic_feasible": np.uint8,
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


@pytest.mark.parametrize("platform_type", ["WHEELED", "LEGGED", "HOPPER"])
def test_bridge_projects_start_bound_reachability_with_exact_array_contract(
    bridge, easy_request, platform_type
) -> None:
    """Would fail if Python reused one generic ground reachability rule."""
    request = easy_request(platform_type)

    projection = bridge.project_reachability(request, 2.0)

    assert projection.platform_type == platform_type
    assert projection.reachable.shape == (
        request.world.global_map.height,
        request.world.global_map.width,
    )
    assert projection.reachable.dtype == np.uint8
    assert projection.reachable.flags.c_contiguous
    assert projection.reachable.any()
    assert projection.maximum_edge_distance_m == 2.0
    if platform_type == "HOPPER":
        assert projection.algorithm_id == "cpp-hopper-certified-bidirectional-bfs/v3"
        assert projection.candidate_edges_evaluated > 0
    else:
        assert projection.algorithm_id == "cpp-ground-start-connected-component/v1"
        assert projection.candidate_edges_evaluated == 0


def test_bridge_queries_exact_ground_endpoints_from_one_context(
    bridge, easy_request
) -> None:
    """A detailed target must not be represented by the candidate canvas parent."""
    request = easy_request("WHEELED")
    context = bridge.project_ground_endpoint_context(request, 30.0)
    targets = np.ascontiguousarray(
        ((4.9, 3.5, 0.0), (-0.1, 3.5, 0.0)), dtype=np.float64
    )

    result = bridge.query_ground_exact_endpoints(context, targets, 0.2)

    assert context.projection.platform_type == "WHEELED"
    assert context.projection.minimum_cost.flags.writeable is False
    assert result.reachable.dtype == np.uint8
    assert result.reachable.flags.c_contiguous
    assert result.minimum_cost_m.dtype == np.float64
    assert result.minimum_cost_m.flags.c_contiguous
    assert result.reachable.tolist() == [1, 0]
    assert np.isfinite(result.minimum_cost_m[0])
    assert np.isposinf(result.minimum_cost_m[1])
    assert result.reason_codes == (
        "GROUND_ENDPOINT_REACHABLE",
        "GROUND_ENDPOINT_OUTSIDE_GLOBAL_MAP",
    )


def test_bridge_streams_hopper_landing_evidence_into_reachability(
    bridge, easy_request
) -> None:
    request = easy_request("HOPPER")
    grid = request.world.global_map
    targets = np.asarray(
        [
            (
                grid.origin_m.x + (column + 0.5) * grid.resolution_m,
                grid.origin_m.y + (row + 0.5) * grid.resolution_m,
                0.0,
            )
            for row in range(grid.height)
            for column in range(grid.width)
        ],
        dtype=np.float64,
    )

    projected = bridge.project_hopper_landing_evidence(request, targets)
    shape = (grid.height, grid.width)
    evidence = bridge_api.HopperLandingEvidenceGrid(
        projected.certified.reshape(shape),
        projected.aim_positions_m.reshape((*shape, 3)),
        projected.boundary_m.reshape((*shape, 4, 3)),
        projected.area_m2.reshape(shape),
        projected.algorithm_id,
    )
    internal = bridge.project_reachability(request, 2.0)
    external = bridge.project_reachability(request, 2.0, evidence)
    direct = bridge.project_direct_hopper_reachability(
        request, 2.0, evidence
    )
    positive = np.zeros(shape, dtype=np.bool_)
    positive.reshape(-1)[-1] = True
    opportunity_context = bridge.project_hopper_opportunity_context(
        request, 2.0, evidence
    )
    opportunity = bridge.query_hopper_opportunity_distance(
        opportunity_context, positive, False
    )
    primitive = bridge_api.PrimitiveReachabilityEngine().update(
        request, 2.0, evidence
    )

    assert projected.algorithm_id == "cpp-hopper-detail-landing-regions/v1"
    assert projected.candidates_evaluated == targets.shape[0]
    assert projected.certified_count == int(projected.certified.sum())
    assert projected.certified.dtype == np.bool_
    assert projected.aim_positions_m.shape == (targets.shape[0], 3)
    assert projected.boundary_m.shape == (targets.shape[0], 4, 3)
    assert evidence.width == grid.width
    assert evidence.height == grid.height
    np.testing.assert_array_equal(external.reachable, internal.reachable)
    assert direct.algorithm_id == (
        "cpp-hopper-certified-bidirectional-direct/v1"
    )
    assert direct.reachable.shape == external.reachable.shape
    assert direct.reachable.dtype == np.uint8
    assert direct.reachable.flags.c_contiguous
    assert opportunity.algorithm_id == "cpp-hopper-opportunity-distance/v1"
    assert opportunity.direct_progress.shape == shape
    assert opportunity.direct_progress.dtype == np.bool_
    assert opportunity.direct_progress.flags.c_contiguous
    assert opportunity.reachable_opportunities.shape == shape
    assert opportunity.reachable_opportunities.dtype == np.bool_
    assert opportunity.reachable_opportunities.flags.c_contiguous
    assert isinstance(opportunity.current_hop_distance, int)
    assert isinstance(opportunity.has_reachable_opportunity, bool)
    assert opportunity_context.algorithm_id == (
        "cpp-hopper-opportunity-connectivity/v1"
    )
    assert opportunity_context.reachable.dtype == np.bool_
    assert opportunity_context.hop_distance_from_current.dtype == np.int32
    assert opportunity_context.candidate_edges_evaluated >= 0
    assert primitive.platform_type == "HOPPER"
    assert primitive.reachable.any()

    with pytest.raises(TypeError, match="bool"):
        bridge_api.HopperLandingEvidenceGrid(
            projected.certified.astype(np.uint8).reshape(shape),
            projected.aim_positions_m.reshape((*shape, 3)),
            projected.boundary_m.reshape((*shape, 4, 3)),
            projected.area_m2.reshape(shape),
            projected.algorithm_id,
        )
    with pytest.raises(TypeError, match="bool"):
        bridge.query_hopper_opportunity_distance(
            opportunity_context, positive.astype(np.uint8)
        )


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


def test_primitive_reachability_engine_exposes_readonly_exact_arrays(
    easy_request,
) -> None:
    request = easy_request("WHEELED")
    engine = bridge_api.PrimitiveReachabilityEngine()

    snapshot = engine.update(request, 30.0)

    assert snapshot.platform_type == "WHEELED"
    assert snapshot.algorithm_id == "cpp-wheel-motion-primitive-recoverable-graph/v1"
    assert snapshot.state_schema == "wheel-lattice-state/v1"
    assert snapshot.revision == 1
    assert len(snapshot.primitive_set_sha256) == 64
    assert len(snapshot.world_evidence_sha256) == 64
    assert len(snapshot.graph_sha256) == 64
    arrays = {
        "state_ids": (np.uint64, (len(snapshot.state_ids),)),
        "positions_m": (np.float64, (len(snapshot.state_ids), 3)),
        "yaw_rad": (np.float64, (len(snapshot.state_ids),)),
        "cells": (np.int32, (len(snapshot.state_ids), 2)),
        "yaw_bin": (np.int32, (len(snapshot.state_ids),)),
        "motion_mode": (np.int32, (len(snapshot.state_ids),)),
        "body_z_m": (np.float64, (len(snapshot.state_ids), 2)),
        "path_cost": (np.float64, (len(snapshot.state_ids),)),
        "forward_reachable": (np.bool_, (len(snapshot.state_ids),)),
        "returnable": (np.bool_, (len(snapshot.state_ids),)),
        "observation_state": (np.bool_, (len(snapshot.state_ids),)),
        "direct_successor": (np.bool_, (len(snapshot.state_ids),)),
        "recoverable": (np.bool_, (len(snapshot.state_ids),)),
        "edge_source_ids": (np.uint64, (len(snapshot.edge_source_ids),)),
        "edge_target_ids": (np.uint64, (len(snapshot.edge_source_ids),)),
        "edge_primitive_indices": (np.uint32, (len(snapshot.edge_source_ids),)),
        "edge_cost": (np.float64, (len(snapshot.edge_source_ids),)),
        "reachable": (
            np.uint8,
            (request.world.global_map.height, request.world.global_map.width),
        ),
    }
    for name, (dtype, shape) in arrays.items():
        values = getattr(snapshot, name)
        assert values.dtype == dtype, name
        assert values.shape == shape, name
        assert values.flags.c_contiguous, name
        assert not values.flags.writeable, name
    assert len(snapshot.edge_primitive_ids) == len(snapshot.edge_source_ids)
    np.testing.assert_array_equal(
        snapshot.recoverable,
        snapshot.forward_reachable & snapshot.returnable,
    )

    repeated = engine.update(request, None)
    assert repeated.revision == 2
    assert repeated.world_evidence_sha256 == snapshot.world_evidence_sha256
    assert repeated.graph_sha256 == snapshot.graph_sha256
    engine.reset()
    reset = engine.update(request, 30.0)
    assert reset.revision == 1


def _set_map_byte(
    grid: bridge_api.GridMap, layer_name: str, row: int, column: int, value: int
) -> None:
    layers = grid.layers
    values = layers[layer_name].values
    values[row * grid.width + column] = value
    layers[layer_name] = bridge_api.GridLayer(values)
    grid.layers = layers


def test_primitive_reachability_engine_reports_content_change_revalidation(
    easy_request,
) -> None:
    request = easy_request("WHEELED")
    engine = bridge_api.PrimitiveReachabilityEngine()
    baseline = engine.update(request, 30.0)
    unchanged = engine.update(request, 30.0)
    assert unchanged.invalidated_edge_count == 0
    assert unchanged.revalidated_edge_count == 0

    _set_map_byte(request.world.global_map, "obstacle", 3, 3, 1)
    request.global_map_generation += 1
    blocked = engine.update(request, 30.0)
    assert blocked.world_evidence_sha256 != baseline.world_evidence_sha256
    assert blocked.invalidated_edge_count > 0
    assert len(blocked.state_ids) < len(baseline.state_ids)

    newly_observed_request = easy_request("WHEELED")
    _set_map_byte(newly_observed_request.world.global_map, "valid_mask", 3, 3, 0)
    newly_observed_engine = bridge_api.PrimitiveReachabilityEngine()
    unknown = newly_observed_engine.update(newly_observed_request, 30.0)
    _set_map_byte(newly_observed_request.world.global_map, "valid_mask", 3, 3, 1)
    newly_observed_request.global_map_generation += 1
    observed = newly_observed_engine.update(newly_observed_request, 30.0)
    assert observed.revalidated_edge_count > 0
    assert len(observed.state_ids) > len(unknown.state_ids)


def test_primitive_reachability_engine_resets_identity_on_contract_drift(
    easy_request,
) -> None:
    engine = bridge_api.PrimitiveReachabilityEngine()
    wheel_request = easy_request("WHEELED")
    wheel = engine.update(wheel_request, 30.0)

    changed_capability = _wheel_capability()
    changed_primitives = changed_capability.motion_primitives
    changed_primitives[0].relative_end_pose = _pose(2.0, 0.0, 0.0)
    changed_capability.motion_primitives = changed_primitives
    wheel_request.capability = changed_capability
    capability_drift = engine.update(wheel_request, 30.0)

    assert capability_drift.revision == 1
    assert capability_drift.primitive_set_sha256 != wheel.primitive_set_sha256
    assert capability_drift.invalidated_edge_count == 0
    assert capability_drift.revalidated_edge_count == 0

    legged_request = easy_request("LEGGED")
    legged_state = bridge_api.LeggedState()
    legged_state.body_pose = _pose(2.5, 3.5, 0.33)
    legged_request.current_state = legged_state
    legged = engine.update(legged_request, 30.0)
    assert legged.platform_type == "LEGGED"
    assert legged.revision == 1


def test_primitive_reachability_engine_propagates_hard_failure_without_revision(
    easy_request,
) -> None:
    request = easy_request("WHEELED")
    engine = bridge_api.PrimitiveReachabilityEngine()
    first = engine.update(request, 30.0)

    with pytest.raises(
        RuntimeError, match="PRIMITIVE_REACHABILITY_DISTANCE_INVALID"
    ):
        engine.update(request, 0.0)

    after_failure = engine.update(request, 30.0)
    assert first.revision == 1
    assert after_failure.revision == 2


def test_truth_primitive_graph_is_not_truncated_at_thirty_metres(
    easy_request,
) -> None:
    request = easy_request("WHEELED")
    request.world.global_map = _flat_map(
        "map", width=80, height=8, resolution_m=1.0
    )
    request.world.local_map = _flat_map(
        "odom", width=80, height=8, resolution_m=1.0
    )
    capability = _wheel_capability()
    primitives = capability.motion_primitives
    reverse = bridge_api.WheelMotionPrimitive()
    reverse.primitive_id = "reverse"
    reverse.kind = bridge_api.WheelPrimitiveKind.REVERSE
    reverse.relative_end_pose = _pose(-1.0, 0.0, 0.0)
    primitives.append(reverse)
    stop = bridge_api.WheelMotionPrimitive()
    stop.primitive_id = "stop-and-switch"
    stop.kind = bridge_api.WheelPrimitiveKind.STOP_AND_SWITCH
    stop.relative_end_pose = _pose(0.0, 0.0, 0.0)
    primitives.append(stop)
    capability.motion_primitives = primitives
    request.capability = capability
    request.config.wheel.xy_resolution_m = 1.0
    request.config.global_map.base_resolution_m = 1.0

    snapshot = bridge_api.PrimitiveReachabilityEngine().update(request, None)
    recoverable_positions = snapshot.positions_m[snapshot.recoverable]

    assert recoverable_positions[:, 0].max() - 2.5 > 30.0


def test_ground_graph_uses_local_primitives_when_global_start_is_partial(
    easy_request,
) -> None:
    request = easy_request("WHEELED")
    _set_map_byte(request.world.global_map, "valid_mask", 3, 2, 0)
    request.world.local_map = _flat_map(
        "odom", width=40, height=40, resolution_m=0.2
    )
    capability = _wheel_capability()
    primitives = capability.motion_primitives
    primitives[0].relative_end_pose = _pose(0.2, 0.0, 0.0)
    reverse = bridge_api.WheelMotionPrimitive()
    reverse.primitive_id = "reverse"
    reverse.kind = bridge_api.WheelPrimitiveKind.REVERSE
    reverse.relative_end_pose = _pose(-0.2, 0.0, 0.0)
    stop = bridge_api.WheelMotionPrimitive()
    stop.primitive_id = "stop-and-switch"
    stop.kind = bridge_api.WheelPrimitiveKind.STOP_AND_SWITCH
    stop.relative_end_pose = _pose(0.0, 0.0, 0.0)
    capability.motion_primitives = [*primitives, reverse, stop]
    request.capability = capability
    request.config.wheel.xy_resolution_m = 0.2

    snapshot = bridge_api.PrimitiveReachabilityEngine().update(request, 30.0)

    assert (snapshot.width, snapshot.height) == (40, 40)
    assert snapshot.recoverable.any()
    np.testing.assert_allclose(
        snapshot.positions_m[snapshot.path_cost == 0.0][0],
        (2.5, 3.5, 0.0),
    )


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
    assert output.candidate_disposition == bridge_api.CandidateDisposition.KEEP


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
    assert set(bridge_api.CandidateDisposition.__members__) == {
        "KEEP",
        "SUPPRESS_FOR_CURRENT_PHYSICAL_SNAPSHOT",
    }


def test_candidate_disposition_defaults_non_target_outputs_to_keep() -> None:
    output = bridge_api.PlannerOutput()
    for outcome in (
        bridge_api.PlanningOutcome.NEW_REFERENCE_AVAILABLE,
        bridge_api.PlanningOutcome.SAFE_FRONTIER_REFERENCE_AVAILABLE,
        bridge_api.PlanningOutcome.INVALID_REQUEST,
        bridge_api.PlanningOutcome.STALE_INPUT,
        bridge_api.PlanningOutcome.NUMERICAL_FAILURE,
        bridge_api.PlanningOutcome.RESOURCE_EXHAUSTED,
        bridge_api.PlanningOutcome.ACTIVE_REFERENCE_INVALIDATED,
        bridge_api.PlanningOutcome.CANCELED,
    ):
        output.outcome = outcome
        assert output.candidate_disposition == bridge_api.CandidateDisposition.KEEP
