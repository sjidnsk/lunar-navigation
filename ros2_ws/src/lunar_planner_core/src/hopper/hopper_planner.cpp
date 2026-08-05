#include "hopper/hopper_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hierarchical/frame_transform.hpp"
#include "hierarchical/hopper_route_planner.hpp"
#include "hierarchical/local_planning_problem.hpp"
#include "hopper/commitment_state_machine.hpp"
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
  return config.maximum_landing_regions > 0U &&
      config.maximum_graph_nodes > 0U &&
      config.maximum_graph_out_degree > 0U &&
      config.maximum_nominal_aim_points_per_region > 0U &&
      config.maximum_certification_attempts > 0U &&
      config.maximum_flight_tube_sections >= 2U &&
      config.maximum_authorized_hops > 0U;
}

[[nodiscard]] PlannerOutput Failure(
    const PlanningOutcome outcome,
    const ExecutionDirective directive,
    std::string reason_code,
    const std::chrono::steady_clock::time_point started,
    const std::uint64_t expanded_states = 0U,
    std::optional<double> best_cost = std::nullopt,
    std::vector<std::string> warning_codes = {}) {
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
  if (!ValidResources(input.config.hopper)) {
    return Failure(
        PlanningOutcome::kResourceExhausted,
        ExecutionDirective::kNoSafeReference,
        "HOPPER_RESOURCE_LIMIT_INVALID", started);
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
        global.expanded_nodes);
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

  HopCertificationResult certified = CertifyFirstHop(
      local_problem, *source.region, *target.region);
  const std::uint64_t total_work = global.expanded_nodes + landing_work +
      static_cast<std::uint64_t>(certified.attempted_candidates);
  if (!certified.ok()) {
    if (certified.canceled) {
      return Canceled(started, total_work);
    }
    if (certified.resource_exhausted) {
      return Failure(
          PlanningOutcome::kResourceExhausted,
          ExecutionDirective::kNoSafeReference,
          certified.reason_code, started, total_work);
    }
    return Failure(
        PlanningOutcome::kNoKnownSafeRoute,
        ExecutionDirective::kNoSafeReference,
        certified.reason_code, started, total_work);
  }

  std::vector<std::string> warnings{"HOPPER_FIRST_HOP_ONLY"};
  if (global.route_hops > 1U) {
    warnings.emplace_back("HOPPER_REMAINING_HOPS_PREVIEW_ONLY");
  }
  if (input.config.hopper.maximum_authorized_hops > 1U) {
    warnings.emplace_back("HOPPER_AUTHORIZATION_CLAMPED_TO_ONE");
  }
  HopReference reference{
      .segments = {std::move(*certified.segment)},
  };
  return PlannerOutput{
      .outcome = PlanningOutcome::kNewReferenceAvailable,
      .directive = ExecutionDirective::kActivateNewReference,
      .reason_code = "HOPPER_FIRST_HOP_AVAILABLE",
      .reference = MotionReference{
          .plan_id = "hopper/" + input.request_id,
          .platform_type = PlatformType::kHopper,
          .input_time = input.state_time,
          .preview = GlobalRoutePreview{
              .poses_map = global.route->poses_map,
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
                      certified.attempted_candidates),
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
                  certified.attempted_candidates,
          },
      },
  };
}

}  // namespace lunar::planning::hopper
