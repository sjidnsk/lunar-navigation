#include "hopper/hopper_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hierarchical/frame_transform.hpp"
#include "hierarchical/global_route.hpp"
#include "hierarchical/hopper_route_planner.hpp"
#include "hierarchical/local_planning_problem.hpp"
#include "hierarchical/map_level.hpp"
#include "hierarchical/route_continuation.hpp"
#include "hopper/commitment_state_machine.hpp"
#include "hopper/flight_tube_certifier.hpp"
#include "hopper/hop_certifier.hpp"
#include "hopper/landing_region.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::planning::hopper {
namespace {

constexpr std::string_view kPlannerName = "cpp_v3_native_hopper";

[[nodiscard]] bool IsFinite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

[[nodiscard]] bool IsFinite(const Quaternion value) noexcept {
  return std::isfinite(value.w) && std::isfinite(value.x) &&
      std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] double Norm(const Vec3 value) noexcept {
  return std::hypot(std::hypot(value.x, value.y), value.z);
}

[[nodiscard]] double PositionError(const Vec3 lhs, const Vec3 rhs) noexcept {
  return std::hypot(std::hypot(lhs.x - rhs.x, lhs.y - rhs.y), lhs.z - rhs.z);
}

[[nodiscard]] double PlanarGoalError(const GoalRegion& goal,
                                     const Vec3 point) noexcept {
  if (const auto* target = std::get_if<PointGoal>(&goal.target)) {
    return std::hypot(point.x - target->position_m.x,
                      point.y - target->position_m.y);
  }
  const auto* region = std::get_if<PlanarRegionGoal>(&goal.target);
  if (region == nullptr || region->boundary_m.empty()) {
    return 0.0;
  }
  double error = std::numeric_limits<double>::infinity();
  for (const Vec3& vertex : region->boundary_m) {
    error = std::min(error,
                     std::hypot(point.x - vertex.x, point.y - vertex.y));
  }
  return error;
}

[[nodiscard]] bool ValidCapability(
    const HopperCapability& capability) noexcept {
  const double minimum_flight_s = std::chrono::duration<double>(
      capability.minimum_flight_time).count();
  const double maximum_flight_s = std::chrono::duration<double>(
      capability.maximum_flight_time).count();
  const double settle_s = std::chrono::duration<double>(
      capability.minimum_settle_guard).count();
  return IsFinite(capability.body_half_extent_m) &&
      capability.body_half_extent_m.x > 0.0 &&
      capability.body_half_extent_m.y > 0.0 &&
      capability.body_half_extent_m.z > 0.0 &&
      std::isfinite(capability.platform_mass_kg) &&
      capability.platform_mass_kg > 0.0 &&
      IsFinite(capability.gravity_mps2) &&
      Norm(capability.gravity_mps2) > 1.0e-9 &&
      std::isfinite(capability.maximum_landing_slope_rad) &&
      capability.maximum_landing_slope_rad > 0.0 &&
      capability.maximum_landing_slope_rad < std::numbers::pi / 2.0 &&
      std::isfinite(capability.maximum_landing_roughness_m) &&
      capability.maximum_landing_roughness_m >= 0.0 &&
      std::isfinite(capability.maximum_plane_residual_m) &&
      capability.maximum_plane_residual_m >= 0.0 &&
      std::isfinite(capability.minimum_overhead_clearance_m) &&
      capability.minimum_overhead_clearance_m >= 0.0 &&
      std::isfinite(capability.minimum_lateral_clearance_m) &&
      capability.minimum_lateral_clearance_m >= 0.0 &&
      std::isfinite(capability.minimum_landing_region_area_m2) &&
      capability.minimum_landing_region_area_m2 > 0.0 &&
      std::isfinite(capability.maximum_launch_speed_mps) &&
      capability.maximum_launch_speed_mps > 0.0 &&
      std::isfinite(capability.maximum_launch_impulse_newton_seconds) &&
      capability.maximum_launch_impulse_newton_seconds > 0.0 &&
      std::isfinite(minimum_flight_s) && minimum_flight_s > 0.0 &&
      std::isfinite(maximum_flight_s) && maximum_flight_s >= minimum_flight_s &&
      std::isfinite(capability.maximum_landing_speed_mps) &&
      capability.maximum_landing_speed_mps > 0.0 &&
      std::isfinite(capability.minimum_downward_impact_speed_mps) &&
      capability.minimum_downward_impact_speed_mps >= 0.0 &&
      std::isfinite(capability.minimum_landing_clearance_m) &&
      capability.minimum_landing_clearance_m >= 0.0 &&
      std::isfinite(capability.maximum_angular_speed_radps) &&
      capability.maximum_angular_speed_radps > 0.0 &&
      std::isfinite(capability.maximum_angular_acceleration_radps2) &&
      capability.maximum_angular_acceleration_radps2 > 0.0 &&
      std::isfinite(capability.maximum_initial_angular_speed_radps) &&
      capability.maximum_initial_angular_speed_radps >= 0.0 &&
      std::isfinite(settle_s) && settle_s >= 0.0;
}

[[nodiscard]] bool ValidState(const HopperState& state) noexcept {
  if (!IsFinite(state.pose.position_m) ||
      !IsFinite(state.pose.orientation) ||
      !IsFinite(state.velocity.linear_mps) ||
      !IsFinite(state.velocity.angular_radps)) {
    return false;
  }
  const double quaternion_norm = std::hypot(
      std::hypot(state.pose.orientation.w, state.pose.orientation.x),
      std::hypot(state.pose.orientation.y, state.pose.orientation.z));
  return std::isfinite(quaternion_norm) &&
      std::abs(quaternion_norm - 1.0) <= 1.0e-6;
}

[[nodiscard]] bool ValidResources(const HopperPlannerConfig& config) noexcept {
  return config.maximum_flight_tube_sections >= 2U &&
      config.maximum_authorized_hops > 0U;
}

[[nodiscard]] HierarchicalPlannerMetrics HopperGlobalMetrics(
    const PlannerInput& input,
    const hierarchical::HopperRoutePlanResult& global) {
  const hierarchical::GlobalRoute* route =
      global.route.has_value() ? &*global.route : nullptr;
  return HierarchicalPlannerMetrics{
      .global_level = global.global_level.value_or(0U),
      .global_resolution_m = input.world.global_map.resolution_m,
      .global_cells = input.world.global_map.CellCount(),
      .global_elapsed = global.elapsed,
      .global_expanded_states = global.expanded_nodes,
      .global_open_peak = global.open_peak,
      .estimated_work_memory_bytes =
          route == nullptr ? 0U : route->estimated_work_memory_bytes,
      .raw_route_points = route == nullptr ? 0U : route->raw_cells.size(),
      .simplified_route_points =
          route == nullptr ? 0U : route->simplified_cells.size(),
      .hopper_graph_nodes = global.graph_nodes,
      .hopper_graph_edges = global.graph_edges,
      .hopper_route_hops = global.route_hops,
      .landing_field_elapsed = global.landing_field_elapsed,
      .spatial_index_elapsed = global.spatial_index_elapsed,
      .ballistic_solve_elapsed = global.ballistic_solve_elapsed,
      .flight_tube_certification_elapsed =
          global.flight_tube_certification_elapsed,
      .safe_landing_nodes = global.safe_landing_nodes,
      .candidate_edges_evaluated = global.evaluated_edge_pairs,
      .coarse_edges_rejected = global.coarse_edges_rejected,
      .full_edges_certified = global.full_edges_certified,
      .full_edges_invalidated = global.full_edges_invalidated,
      .edge_certificate_cache_hits = global.edge_certificate_cache_hits,
  };
}

[[nodiscard]] PlannerOutput Failure(
    const PlanningOutcome outcome,
    const ExecutionDirective directive,
    std::string reason_code,
    const std::chrono::steady_clock::time_point started,
    const std::uint64_t expanded_states = 0U,
    std::optional<double> best_cost = std::nullopt,
    std::vector<std::string> warning_codes = {},
    std::optional<HierarchicalPlannerMetrics> hierarchical = std::nullopt) {
  return PlannerOutput{
      .outcome = outcome,
      .directive = directive,
      .reason_code = std::move(reason_code),
      .reference = std::nullopt,
      .diagnostics = PlannerDiagnostics{
          .planner_name = std::string{kPlannerName},
          .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started),
          .expanded_states = expanded_states,
          .best_cost = best_cost,
          .warning_codes = std::move(warning_codes),
          .hierarchical = std::move(hierarchical),
      },
  };
}

[[nodiscard]] PlannerOutput Canceled(
    const std::chrono::steady_clock::time_point started,
    const std::uint64_t expanded_states = 0U) {
  return Failure(
      PlanningOutcome::kCanceled,
      ExecutionDirective::kHoldPosition,
      "REQUEST_CANCELED", started, expanded_states);
}

[[nodiscard]] PlannerOutput PromoteCertifiedHop(
    const PlannerInput& input, const HopperState& state,
    const RouteContinuation& previous,
    const hierarchical::HopperHopPromotionResult& promotion,
    const std::chrono::steady_clock::time_point started) {
  if (!promotion.ok() || !promotion.hop.has_value()) {
    return Failure(
        PlanningOutcome::kNumericalFailure,
        ExecutionDirective::kNoSafeReference,
        "HOPPER_PROMOTION_RESULT_INVALID", started);
  }
  const CertifiedHopPreview& preview = *promotion.hop;
  const auto actual_map = hierarchical::TransformPose(
      state.pose, input.world.map_from_odom,
      hierarchical::TransformDirection::kChildToParent);
  RigidTransform vector_rotation = input.world.map_from_odom;
  vector_rotation.translation_m = {};
  const auto launch_velocity_odom = hierarchical::TransformPoint(
      preview.launch_velocity_mps, vector_rotation,
      hierarchical::TransformDirection::kParentToChild);
  std::vector<Vec3> landing_region_odom;
  landing_region_odom.reserve(preview.landing_region_map.size());
  for (const Vec3 vertex_map : preview.landing_region_map) {
    const auto vertex_odom = hierarchical::TransformPoint(
        vertex_map, input.world.map_from_odom,
        hierarchical::TransformDirection::kParentToChild);
    if (!vertex_odom.has_value()) {
      return Failure(
          PlanningOutcome::kInvalidRequest,
          ExecutionDirective::kNoSafeReference,
          "FRAME_TRANSFORM_INVALID", started);
    }
    landing_region_odom.push_back(*vertex_odom);
  }
  if (!actual_map.has_value() || !launch_velocity_odom.has_value() ||
      landing_region_odom.size() < 3U || preview.flight_time.count() <= 0 ||
      !std::isfinite(preview.flight_tube_radius_m) ||
      preview.flight_tube_radius_m <= 0.0) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        "FRAME_TRANSFORM_INVALID", started);
  }

  const HopSegment segment{
      .segment_id = preview.segment_id,
      .launch_pose = state.pose,
      .landing_region_boundary_m = std::move(landing_region_odom),
      .flight_time = preview.flight_time,
      .launch_velocity_mps = *launch_velocity_odom,
      .flight_tube_radius_m = preview.flight_tube_radius_m,
  };
  std::vector<Pose3> route_preview;
  route_preview.reserve(previous.certified_hops().size() -
                        promotion.route_cursor + 1U);
  route_preview.push_back(*actual_map);
  for (std::size_t index = promotion.route_cursor;
       index < previous.certified_hops().size(); ++index) {
    route_preview.push_back(previous.certified_hops()[index].landing_pose_map);
  }
  const std::string plan_id = "hopper/" + input.request_id;
  auto continuation = std::make_shared<const RouteContinuation>(
      previous.route_id(), plan_id, input, previous.global_route(),
      previous.certified_hops(), promotion.route_cursor, 0.0,
      previous.rolling_request_count() + 1U);
  const double size_x_m = static_cast<double>(input.world.global_map.width) *
                          input.world.global_map.resolution_m;
  const double size_y_m = static_cast<double>(input.world.global_map.height) *
                          input.world.global_map.resolution_m;
  const hierarchical::ExpectedMapLevelResult map_level =
      hierarchical::ExpectedGlobalMapLevel(size_x_m, size_y_m,
                                           input.config.global_map);
  const double local_distance = std::hypot(
      preview.landing_pose_map.position_m.x - actual_map->position_m.x,
      preview.landing_pose_map.position_m.y - actual_map->position_m.y);
  std::vector<std::string> warnings{"HOPPER_FIRST_HOP_ONLY"};
  if (promotion.route_cursor + 1U < previous.certified_hops().size()) {
    warnings.emplace_back("HOPPER_REMAINING_HOPS_PREVIEW_ONLY");
  }
  return PlannerOutput{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .directive = ExecutionDirective::kActivateNewReference,
      .reason_code = "HOPPER_NEXT_HOP_AVAILABLE",
      .reference = MotionReference{
          .plan_id = plan_id,
          .platform_type = PlatformType::kHopper,
          .input_time = input.state_time,
          .preview = GlobalRoutePreview{
              .poses_map = hierarchical::ThinRoutePreview(
                  route_preview,
                  input.config.global_search.maximum_preview_points)},
          .data = HopReference{.segments = {segment}},
      },
      .diagnostics = PlannerDiagnostics{
          .planner_name = std::string{kPlannerName},
          .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started),
          .expanded_states = 0U,
          .best_cost = previous.global_route().cost,
          .warning_codes = std::move(warnings),
          .hierarchical = HierarchicalPlannerMetrics{
              .global_level = map_level.level.value_or(0U),
              .global_resolution_m = input.world.global_map.resolution_m,
              .global_cells = input.world.global_map.CellCount(),
              .global_elapsed = {},
              .local_elapsed =
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() - started),
              .global_expanded_states = 0U,
              .local_expanded_states = 0U,
              .global_open_peak = 0U,
              .estimated_work_memory_bytes = 0U,
              .raw_route_points = previous.global_route().raw_cells.size(),
              .simplified_route_points =
                  previous.global_route().simplified_cells.size(),
              .local_frontier_distance_m = local_distance,
              .local_attempts = 1U,
              .hopper_route_hops = previous.certified_hops().size(),
              .route_reused = true,
              .route_cursor = promotion.route_cursor,
              .rolling_request_count =
                  previous.rolling_request_count() + 1U,
          },
          .local_trajectory = LocalTrajectoryDiagnostics{
              .trajectory_mode = TrajectoryMode::kCertifiedHop,
              .start_anchor_error_m = 0.0,
              .endpoint_error_m = 0.0,
              .maximum_curvature_per_m = 0.0,
              .collision_validation = CollisionValidation::kCertified,
          },
      },
      .certified_hops = previous.certified_hops(),
      .continuation = std::move(continuation),
  };
}

}  // namespace

PlannerOutput HopperPlanner::Plan(const PlannerInput& input) const {
  const auto started = std::chrono::steady_clock::now();
  const auto* state = std::get_if<HopperState>(&input.current_state);
  const auto* capability = std::get_if<HopperCapability>(&input.capability);
  if (state == nullptr || capability == nullptr) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        "HOPPER_PLATFORM_TYPE_MISMATCH", started);
  }

  const PlanningProtection protection =
      EvaluatePlanningProtection(input.previous_execution);
  if (protection.decision ==
      PlanningProtectionDecision::kContinueCommittedHop) {
    return Failure(
        PlanningOutcome::kSafeFrontierReferenceAvailable,
        ExecutionDirective::kContinueCommittedHop,
        protection.reason_code, started);
  }
  if (protection.decision ==
      PlanningProtectionDecision::kActiveReferenceInvalidated) {
    return Failure(
        PlanningOutcome::kActiveReferenceInvalidated,
        ExecutionDirective::kNoSafeReference,
        protection.reason_code, started);
  }
  if (input.stop_token.stop_requested()) {
    return Canceled(started);
  }
  if (input.request_id.empty() || !ValidState(*state) ||
      !ValidCapability(*capability)) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        "HOPPER_REQUEST_INVALID", started);
  }
  if (Norm(state->velocity.angular_radps) >
      capability->maximum_initial_angular_speed_radps + 1.0e-9) {
    return Failure(
        PlanningOutcome::kNoKnownSafeRoute,
        ExecutionDirective::kNoSafeReference,
        "HOPPER_ATTITUDE_NOT_CERTIFIED", started);
  }
  if (!ValidResources(input.config.hopper)) {
    return Failure(
        PlanningOutcome::kResourceExhausted,
        ExecutionDirective::kNoSafeReference,
        "HOPPER_RESOURCE_LIMIT_INVALID", started);
  }

  if (input.continuation != nullptr) {
    const hierarchical::HopperHopPromotionResult promotion =
        hierarchical::TryPromoteHopperHop(input, *input.continuation);
    if (promotion.ok()) {
      return PromoteCertifiedHop(input, *state, *input.continuation,
                                 promotion, started);
    }
  }

  const hierarchical::HopperRoutePlanResult global =
      hierarchical::PlanHopperGlobalRoute(input);
  if (!global.ok()) {
    ExecutionDirective directive = ExecutionDirective::kNoSafeReference;
    if (global.outcome == PlanningOutcome::kCanceled ||
        global.outcome == PlanningOutcome::kGoalInfeasible) {
      directive = ExecutionDirective::kHoldPosition;
    }
    return Failure(
        global.outcome, directive, global.reason_code, started,
        global.expanded_nodes, std::nullopt, {},
        HopperGlobalMetrics(input, global));
  }
  if (!global.route.has_value() || global.route->poses_map.size() < 2U) {
    return Failure(
        PlanningOutcome::kNumericalFailure,
        ExecutionDirective::kNoSafeReference,
        "HOPPER_GLOBAL_ROUTE_RESULT_INVALID", started,
        global.expanded_nodes);
  }

  const Pose3& next_pose_map = global.route->poses_map[1U];
  const auto next_pose_odom = hierarchical::TransformPose(
      next_pose_map, input.world.map_from_odom,
      hierarchical::TransformDirection::kParentToChild);
  if (!next_pose_odom.has_value()) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        "FRAME_TRANSFORM_INVALID", started, global.expanded_nodes);
  }
  const auto final_goal_odom = hierarchical::TransformGoal(
      input.goal_map, input.world.map_from_odom,
      hierarchical::TransformDirection::kParentToChild);
  if (!final_goal_odom.has_value()) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        "FRAME_TRANSFORM_INVALID", started, global.expanded_nodes);
  }
  GoalRegion local_goal{
      .goal_id = input.goal_map.goal_id + "/first-hop",
      .target = PointGoal{
          .position_m = Vec3{
              .x = next_pose_odom->position_m.x,
              .y = next_pose_odom->position_m.y,
              .z = next_pose_odom->position_m.z -
                  capability->body_half_extent_m.z,
          },
          .tolerance_m = std::max(
              0.1, 0.25 * input.world.local_map.resolution_m),
      },
      .yaw_rad = std::nullopt,
      .yaw_tolerance_rad = 0.0,
  };
  if (global.route_hops == 1U) {
    local_goal = *final_goal_odom;
  }
  const hierarchical::LocalPlanningProblem local_problem{
      .request_id = input.request_id,
      .state_time = input.state_time,
      .current_state = input.current_state,
      .goal_odom = std::move(local_goal),
      .local_map_view = input.world.local_map,
      .capability = input.capability,
      .config = input.config,
      .previous_execution = input.previous_execution,
      .stop_token = input.stop_token,
  };

  const shared::MapSnapshotBuildResult map =
      shared::MapSnapshot::Create(local_problem.local_map_view);
  if (!map.ok()) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        map.reason_code, started);
  }
  const shared::SafeProjectionBuildResult projection =
      shared::BuildSafeProjection(
          map.snapshot, local_problem.capability,
          local_problem.config.map_safety, local_problem.stop_token);
  if (!projection.ok()) {
    if (projection.reason_code == "REQUEST_CANCELED") {
      return Canceled(started);
    }
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        projection.reason_code, started);
  }

  std::uint64_t final_goal_work = 0U;
  if (global.route_hops > 1U) {
    const auto final_pose_odom = hierarchical::TransformPose(
        global.route->poses_map.back(), input.world.map_from_odom,
        hierarchical::TransformDirection::kParentToChild);
    if (!final_pose_odom.has_value()) {
      return Failure(
          PlanningOutcome::kInvalidRequest,
          ExecutionDirective::kNoSafeReference,
          "FRAME_TRANSFORM_INVALID", started, global.expanded_nodes);
    }
    const auto final_cell = map.snapshot->PositionToCell(Vec2{
        .x = final_pose_odom->position_m.x,
        .y = final_pose_odom->position_m.y,
    });
    if (final_cell.has_value()) {
      const LandingRegionResult final_region = CertifyLandingRegion(
          *projection.projection, *final_goal_odom, *capability,
          local_problem.config.map_safety, local_problem.stop_token);
      final_goal_work = final_region.inspected_cells;
      if (!final_region.ok()) {
        if (final_region.status == LandingRegionStatus::kCanceled) {
          return Canceled(
              started, global.expanded_nodes + final_goal_work);
        }
        if (final_region.status == LandingRegionStatus::kInvalidRequest) {
          return Failure(
              PlanningOutcome::kInvalidRequest,
              ExecutionDirective::kNoSafeReference,
              final_region.reason_code, started,
              global.expanded_nodes + final_goal_work);
        }
        if (final_region.status == LandingRegionStatus::kResourceExhausted) {
          return Failure(
              PlanningOutcome::kResourceExhausted,
              ExecutionDirective::kNoSafeReference,
              final_region.reason_code, started,
              global.expanded_nodes + final_goal_work);
        }
        return Failure(
            PlanningOutcome::kGoalInfeasible,
            ExecutionDirective::kHoldPosition,
            final_region.reason_code, started,
            global.expanded_nodes + final_goal_work);
      }
    }
  }

  const LandingRegionResult source = CertifyHoldingRegion(
      *projection.projection, state->pose.position_m,
      *capability, local_problem.config.map_safety,
      local_problem.stop_token);
  if (!source.ok()) {
    if (source.status == LandingRegionStatus::kCanceled) {
      return Canceled(started, source.inspected_cells);
    }
    if (source.status == LandingRegionStatus::kInvalidRequest) {
      return Failure(
          PlanningOutcome::kInvalidRequest,
          ExecutionDirective::kNoSafeReference,
          source.reason_code, started, source.inspected_cells);
    }
    return Failure(
        PlanningOutcome::kNoKnownSafeRoute,
        ExecutionDirective::kNoSafeReference,
        source.reason_code, started, source.inspected_cells);
  }
  const double expected_launch_height =
      source.region->aim_position_on_surface_m.z +
      capability->body_half_extent_m.z;
  if (std::abs(state->pose.position_m.z - expected_launch_height) >
      std::max(0.05, capability->minimum_landing_clearance_m)) {
    return Failure(
        PlanningOutcome::kNoKnownSafeRoute,
        ExecutionDirective::kNoSafeReference,
        "HOPPER_START_HEIGHT_NOT_CERTIFIED", started,
        source.inspected_cells);
  }

  const LandingRegionResult target = CertifyLandingRegion(
      *projection.projection, local_problem.goal_odom, *capability,
      local_problem.config.map_safety, local_problem.stop_token);
  const std::uint64_t landing_work = final_goal_work +
      static_cast<std::uint64_t>(
          source.inspected_cells + target.inspected_cells);
  if (!target.ok()) {
    if (target.status == LandingRegionStatus::kCanceled) {
      return Canceled(started, landing_work);
    }
    if (target.status == LandingRegionStatus::kInvalidRequest) {
      return Failure(
          PlanningOutcome::kInvalidRequest,
          ExecutionDirective::kNoSafeReference,
          target.reason_code, started, landing_work);
    }
    if (target.status == LandingRegionStatus::kResourceExhausted) {
      return Failure(
          PlanningOutcome::kResourceExhausted,
          ExecutionDirective::kNoSafeReference,
          target.reason_code, started, landing_work);
    }
    if (global.route_hops == 1U) {
      return Failure(
          PlanningOutcome::kGoalInfeasible,
          ExecutionDirective::kHoldPosition,
          target.reason_code, started,
          global.expanded_nodes + landing_work);
    }
    return Failure(
        PlanningOutcome::kNoKnownSafeRoute,
        ExecutionDirective::kNoSafeReference,
        "LOCAL_MAP_COVERAGE_INSUFFICIENT", started,
        global.expanded_nodes + landing_work);
  }

  if (global.certified_hops.size() != global.route_hops ||
      global.nominal_hops.size() != global.route_hops ||
      global.certified_hops.empty()) {
    return Failure(
        PlanningOutcome::kNumericalFailure,
        ExecutionDirective::kNoSafeReference,
        "HOPPER_GLOBAL_ROUTE_RESULT_INVALID", started,
        global.expanded_nodes + landing_work);
  }
  const CertifiedHopPreview& first_preview = global.certified_hops.front();
  const hierarchical::NominalHopEdge& first_edge = global.nominal_hops.front();
  const auto launch_pose_odom = hierarchical::TransformPose(
      first_preview.launch_pose_map, input.world.map_from_odom,
      hierarchical::TransformDirection::kParentToChild);
  const auto landing_pose_odom = hierarchical::TransformPose(
      first_preview.landing_pose_map, input.world.map_from_odom,
      hierarchical::TransformDirection::kParentToChild);
  RigidTransform vector_rotation = input.world.map_from_odom;
  vector_rotation.translation_m = {};
  const auto launch_velocity_odom = hierarchical::TransformPoint(
      first_preview.launch_velocity_mps, vector_rotation,
      hierarchical::TransformDirection::kParentToChild);
  const auto gravity_odom = hierarchical::TransformPoint(
      first_edge.arc.gravity_mps2, vector_rotation,
      hierarchical::TransformDirection::kParentToChild);
  std::vector<Vec3> landing_region_odom;
  landing_region_odom.reserve(first_preview.landing_region_map.size());
  bool landing_region_transform_ok = true;
  for (const Vec3 vertex_map : first_preview.landing_region_map) {
    const auto vertex_odom = hierarchical::TransformPoint(
        vertex_map, input.world.map_from_odom,
        hierarchical::TransformDirection::kParentToChild);
    if (!vertex_odom.has_value()) {
      landing_region_transform_ok = false;
      break;
    }
    landing_region_odom.push_back(*vertex_odom);
  }
  if (!launch_pose_odom.has_value() || !landing_pose_odom.has_value() ||
      !launch_velocity_odom.has_value() || !gravity_odom.has_value() ||
      !landing_region_transform_ok || landing_region_odom.size() < 3U ||
      first_preview.flight_time.count() <= 0 ||
      PositionError(launch_pose_odom->position_m, state->pose.position_m) >
          1.0e-6) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference,
        "FRAME_TRANSFORM_INVALID", started,
        global.expanded_nodes + landing_work);
  }
  const double certified_flight_s =
      std::chrono::duration<double>(first_preview.flight_time).count();
  const BallisticArc local_arc{
      .launch_position_m = state->pose.position_m,
      .landing_position_m = landing_pose_odom->position_m,
      .gravity_mps2 = *gravity_odom,
      .launch_velocity_mps = *launch_velocity_odom,
      .landing_velocity_mps =
          Vec3{
              launch_velocity_odom->x + gravity_odom->x * certified_flight_s,
              launch_velocity_odom->y + gravity_odom->y * certified_flight_s,
              launch_velocity_odom->z + gravity_odom->z * certified_flight_s,
          },
      .flight_time_s = certified_flight_s,
  };
  CertifiedLandingRegion exact_target_region = *target.region;
  exact_target_region.aim_position_on_surface_m = Vec3{
      .x = landing_pose_odom->position_m.x,
      .y = landing_pose_odom->position_m.y,
      .z = landing_pose_odom->position_m.z - capability->body_half_extent_m.z,
  };
  exact_target_region.boundary_m = landing_region_odom;
  const FlightTubeCertificationResult local_tube = CertifyFlightTube(
      local_arc, *map.snapshot, *source.region, exact_target_region,
      *capability, local_problem.config, local_problem.stop_token,
      first_edge.tube_expansion_margin_m);
  HopCertificationResult certified;
  certified.examined_intervals = local_tube.overlapped_cell_count;
  if (local_tube.canceled) {
    certified.status = HopCertificationStatus::kCanceled;
    certified.reason_code = "REQUEST_CANCELED";
  } else if (!local_tube.certified) {
    certified.reason_code = local_tube.reason_code;
    if (local_tube.reason_code == "HOPPER_FLIGHT_TUBE_RESOURCE_EXHAUSTED") {
      certified.status = HopCertificationStatus::kResourceExhausted;
    } else if (local_tube.reason_code ==
               "HOPPER_FLIGHT_TUBE_INPUT_INVALID") {
      certified.status = HopCertificationStatus::kInvalid;
    } else if (local_tube.reason_code ==
               "HOPPER_FLIGHT_TUBE_NUMERICAL_INDETERMINATE") {
      certified.status = HopCertificationStatus::kNumericalIndeterminate;
    } else {
      certified.status = HopCertificationStatus::kInfeasible;
    }
  } else {
    certified.status = HopCertificationStatus::kCertified;
    certified.segment = HopSegment{
        .segment_id = first_preview.segment_id,
        .launch_pose = state->pose,
        .landing_region_boundary_m = std::move(landing_region_odom),
        .flight_time = first_preview.flight_time,
        .launch_velocity_mps = *launch_velocity_odom,
        .flight_tube_radius_m = local_tube.radius_m,
    };
    certified.cost = first_edge.cost;
  }
  const std::uint64_t total_work = global.expanded_nodes + landing_work +
      static_cast<std::uint64_t>(certified.examined_intervals);
  if (!certified.ok()) {
    switch (certified.status) {
      case HopCertificationStatus::kCanceled:
        return Canceled(started, total_work);
      case HopCertificationStatus::kResourceExhausted:
        return Failure(
          PlanningOutcome::kResourceExhausted,
          ExecutionDirective::kNoSafeReference,
          certified.reason_code, started, total_work);
      case HopCertificationStatus::kInvalid:
        return Failure(
            PlanningOutcome::kInvalidRequest,
            ExecutionDirective::kNoSafeReference,
            certified.reason_code, started, total_work);
      case HopCertificationStatus::kNumericalIndeterminate:
        return Failure(
            PlanningOutcome::kNumericalFailure,
            ExecutionDirective::kNoSafeReference,
            certified.reason_code, started, total_work);
      case HopCertificationStatus::kInfeasible:
        return Failure(
            PlanningOutcome::kNoKnownSafeRoute,
            ExecutionDirective::kNoSafeReference,
            certified.reason_code, started, total_work);
      case HopCertificationStatus::kCertified:
        return Failure(
            PlanningOutcome::kNumericalFailure,
            ExecutionDirective::kNoSafeReference,
            "HOPPER_CERTIFICATION_RESULT_INVALID", started, total_work);
    }
  }

  std::vector<std::string> warnings{"HOPPER_FIRST_HOP_ONLY"};
  if (global.route_hops > 1U) {
    warnings.emplace_back("HOPPER_REMAINING_HOPS_PREVIEW_ONLY");
  }
  if (input.config.hopper.maximum_authorized_hops > 1U) {
    warnings.emplace_back("HOPPER_AUTHORIZATION_CLAMPED_TO_ONE");
  }
  const HopSegment& first_hop = *certified.segment;
  const double flight_s =
      std::chrono::duration<double>(first_hop.flight_time).count();
  const Vec3 landing_position{
      .x = first_hop.launch_pose.position_m.x +
           first_hop.launch_velocity_mps.x * flight_s +
           0.5 * capability->gravity_mps2.x * flight_s * flight_s,
      .y = first_hop.launch_pose.position_m.y +
           first_hop.launch_velocity_mps.y * flight_s +
           0.5 * capability->gravity_mps2.y * flight_s * flight_s,
      .z = first_hop.launch_pose.position_m.z +
           first_hop.launch_velocity_mps.z * flight_s +
           0.5 * capability->gravity_mps2.z * flight_s * flight_s,
  };
  const LocalTrajectoryDiagnostics local_diagnostics{
      .trajectory_mode = TrajectoryMode::kCertifiedHop,
      .start_anchor_error_m =
          PositionError(first_hop.launch_pose.position_m,
                        state->pose.position_m),
      .endpoint_error_m =
          PlanarGoalError(local_problem.goal_odom, landing_position),
      .maximum_curvature_per_m = 0.0,
      .collision_validation = CollisionValidation::kCertified,
      .smoothing_elapsed_s = 0.0,
      .landing_field_elapsed_s =
          std::chrono::duration<double>(global.landing_field_elapsed).count(),
  };
  if (!std::isfinite(local_diagnostics.start_anchor_error_m) ||
      !std::isfinite(local_diagnostics.endpoint_error_m) ||
      !std::isfinite(local_diagnostics.landing_field_elapsed_s)) {
    return Failure(PlanningOutcome::kNumericalFailure,
                   ExecutionDirective::kNoSafeReference,
                   "HOPPER_TRAJECTORY_DIAGNOSTICS_NONFINITE", started,
                   total_work, global.route->cost, std::move(warnings));
  }
  HopReference reference{
      .segments = {std::move(*certified.segment)},
  };
  const std::string plan_id = "hopper/" + input.request_id;
  auto continuation = std::make_shared<const RouteContinuation>(
      "hopper-route/" + input.request_id, plan_id, input, *global.route,
      global.certified_hops, 0U);
  return PlannerOutput{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .directive = ExecutionDirective::kActivateNewReference,
      .reason_code = "HOPPER_FIRST_HOP_AVAILABLE",
      .reference = MotionReference{
          .plan_id = plan_id,
          .platform_type = PlatformType::kHopper,
          .input_time = input.state_time,
          .preview = GlobalRoutePreview{
              .poses_map = hierarchical::ThinRoutePreview(
                  global.route->poses_map,
                  input.config.global_search.maximum_preview_points),
          },
          .data = std::move(reference),
      },
      .diagnostics = PlannerDiagnostics{
          .planner_name = std::string{kPlannerName},
          .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started),
          .expanded_states = total_work,
          .best_cost = global.route->cost,
          .warning_codes = std::move(warnings),
          .hierarchical = HierarchicalPlannerMetrics{
              .global_level = global.global_level.value_or(0U),
              .global_resolution_m = input.world.global_map.resolution_m,
              .global_cells = input.world.global_map.CellCount(),
              .global_elapsed = global.elapsed,
              .local_elapsed = std::chrono::duration_cast<
                  std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - started) -
                  global.elapsed,
              .global_expanded_states = global.expanded_nodes,
              .local_expanded_states = landing_work +
                  static_cast<std::uint64_t>(
                      certified.examined_intervals),
              .global_open_peak = global.route->open_peak,
              .estimated_work_memory_bytes =
                  global.route->estimated_work_memory_bytes,
              .raw_route_points = global.route->raw_cells.size(),
              .simplified_route_points =
                  global.route->simplified_cells.size(),
              .local_frontier_distance_m = std::hypot(
                  next_pose_odom->position_m.x - state->pose.position_m.x,
                  next_pose_odom->position_m.y - state->pose.position_m.y),
              .local_attempts = 1U,
              .hopper_graph_nodes = global.graph_nodes,
              .hopper_graph_edges = global.graph_edges,
              .hopper_route_hops = global.route_hops,
              .hopper_certification_attempts =
                  certified.examined_intervals,
              .landing_field_elapsed = global.landing_field_elapsed,
              .spatial_index_elapsed = global.spatial_index_elapsed,
              .ballistic_solve_elapsed = global.ballistic_solve_elapsed,
              .flight_tube_certification_elapsed =
                  global.flight_tube_certification_elapsed,
              .safe_landing_nodes = global.safe_landing_nodes,
              .candidate_edges_evaluated = global.evaluated_edge_pairs,
              .coarse_edges_rejected = global.coarse_edges_rejected,
              .full_edges_certified = global.full_edges_certified,
              .full_edges_invalidated = global.full_edges_invalidated,
              .edge_certificate_cache_hits =
                  global.edge_certificate_cache_hits,
          },
          .local_trajectory = local_diagnostics,
      },
      .certified_hops = global.certified_hops,
      .continuation = std::move(continuation),
  };
}

}  // namespace lunar::planning::hopper
