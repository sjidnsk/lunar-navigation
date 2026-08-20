#include "hopper/hopper_planner.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numbers>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hierarchical/frame_transform.hpp"
#include "hopper/ballistic_kinematics.hpp"
#include "hopper/commitment_state_machine.hpp"
#include "hopper/flight_tube_certifier.hpp"
#include "hopper/hop_certifier.hpp"
#include "hopper/landing_region.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::planning::hopper {
namespace {

constexpr std::string_view kPlannerName = "cpp_v3_native_hopper";

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

[[nodiscard]] bool Finite(const Quaternion value) noexcept {
  return std::isfinite(value.w) && std::isfinite(value.x) &&
      std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] bool ValidState(const HopperState& state) noexcept {
  if (!Finite(state.pose.position_m) || !Finite(state.pose.orientation) ||
      !Finite(state.velocity.linear_mps) ||
      !Finite(state.velocity.angular_radps)) {
    return false;
  }
  const double norm = std::hypot(
      std::hypot(state.pose.orientation.w, state.pose.orientation.x),
      std::hypot(state.pose.orientation.y, state.pose.orientation.z));
  return std::isfinite(norm) && std::abs(norm - 1.0) <= 1.0e-6;
}

[[nodiscard]] bool ValidCapability(
    const HopperCapability& capability) noexcept {
  return std::isfinite(capability.specific_impulse_s) &&
      capability.specific_impulse_s > 0.0 &&
      std::isfinite(capability.reference_total_mass_kg) &&
      capability.reference_total_mass_kg > 0.0 &&
      std::isfinite(capability.reference_propellant_mass_kg) &&
      capability.reference_propellant_mass_kg > 0.0 &&
      capability.reference_propellant_mass_kg <
          capability.reference_total_mass_kg &&
      Finite(capability.gravity_mps2) &&
      capability.gravity_mps2.x == 0.0 && capability.gravity_mps2.y == 0.0 &&
      capability.gravity_mps2.z < 0.0 &&
      std::isfinite(capability.reference_horizontal_range_m) &&
      capability.reference_horizontal_range_m > 0.0 &&
      std::isfinite(capability.reference_elevation_delta_m) &&
      !capability.runtime_fallback_allowed &&
      std::isfinite(capability.landing_support_radius_m) &&
      capability.landing_support_radius_m > 0.0 &&
      std::isfinite(capability.flight_collision_radius_m) &&
      capability.flight_collision_radius_m > 0.0 &&
      std::isfinite(capability.maximum_landing_slope_rad) &&
      capability.maximum_landing_slope_rad > 0.0 &&
      capability.maximum_landing_slope_rad < std::numbers::pi / 2.0 &&
      std::isfinite(capability.maximum_landing_plane_residual_m) &&
      capability.maximum_landing_plane_residual_m >= 0.0 &&
      std::isfinite(capability.landing_lateral_margin_m) &&
      capability.landing_lateral_margin_m >= 0.0 &&
      std::isfinite(capability.flight_map_margin_m) &&
      capability.flight_map_margin_m >= 0.0 &&
      std::isfinite(capability.reachability_delta_v_margin_ratio) &&
      capability.reachability_delta_v_margin_ratio >= 0.0 &&
      std::isfinite(capability.standard_gravity_mps2) &&
      capability.standard_gravity_mps2 > 0.0;
}

[[nodiscard]] PlannerOutput Failure(
    const PlanningOutcome outcome, const ExecutionDirective directive,
    const CandidateDisposition candidate_disposition, std::string reason_code,
    const std::chrono::steady_clock::time_point started,
    const std::uint64_t expanded_states = 0U,
    std::optional<LocalTrajectoryDiagnostics> local = std::nullopt,
    std::optional<HierarchicalPlannerMetrics> hierarchical = std::nullopt) {
  return {
      .outcome = outcome,
      .directive = directive,
      .candidate_disposition = candidate_disposition,
      .reason_code = std::move(reason_code),
      .diagnostics = PlannerDiagnostics{
          .planner_name = std::string{kPlannerName},
          .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started),
          .expanded_states = expanded_states,
          .hierarchical = std::move(hierarchical),
          .local_trajectory = std::move(local),
      },
  };
}

[[nodiscard]] PlannerOutput Canceled(
    const std::chrono::steady_clock::time_point started,
    const std::uint64_t expanded_states = 0U) {
  return Failure(
      PlanningOutcome::kCanceled, ExecutionDirective::kHoldPosition,
      CandidateDisposition::kKeep, "REQUEST_CANCELED", started,
      expanded_states);
}

[[nodiscard]] std::optional<Vec3> RotateMapVectorToOdom(
    const Vec3 vector_map, const RigidTransform& map_from_odom) noexcept {
  RigidTransform rotation = map_from_odom;
  rotation.translation_m = {};
  return hierarchical::TransformPoint(
      vector_map, rotation,
      hierarchical::TransformDirection::kParentToChild);
}

[[nodiscard]] std::vector<Vec3> TransformBoundary(
    const std::vector<Vec3>& boundary, const RigidTransform& map_from_odom,
    const hierarchical::TransformDirection direction, bool& ok) {
  std::vector<Vec3> transformed;
  transformed.reserve(boundary.size());
  for (const Vec3 point : boundary) {
    const auto value =
        hierarchical::TransformPoint(point, map_from_odom, direction);
    if (!value.has_value()) {
      ok = false;
      return {};
    }
    transformed.push_back(*value);
  }
  ok = true;
  return transformed;
}

[[nodiscard]] HierarchicalPlannerMetrics Metrics(
    const PlannerInput& input, const std::size_t landing_work,
    const std::size_t ballistic_work,
    const std::chrono::nanoseconds elapsed) {
  return {
      .global_resolution_m = input.world.global_map.resolution_m,
      .global_cells = input.world.global_map.CellCount(),
      .global_elapsed = std::chrono::nanoseconds{0},
      .local_elapsed = elapsed,
      .local_expanded_states = landing_work + ballistic_work,
      .local_attempts = 1U,
      .hopper_route_hops = 1U,
      .hopper_certification_attempts = ballistic_work,
      .full_edges_certified = 1U,
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
        ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
        "HOPPER_PLATFORM_TYPE_MISMATCH", started);
  }

  const PlanningProtection protection =
      EvaluatePlanningProtection(input.previous_execution);
  if (protection.decision ==
      PlanningProtectionDecision::kContinueCommittedHop) {
    return Failure(
        PlanningOutcome::kSafeFrontierReferenceAvailable,
        ExecutionDirective::kContinueCommittedHop, CandidateDisposition::kKeep,
        protection.reason_code, started);
  }
  if (protection.decision ==
      PlanningProtectionDecision::kActiveReferenceInvalidated) {
    return Failure(
        PlanningOutcome::kActiveReferenceInvalidated,
        ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
        protection.reason_code, started);
  }
  if (input.stop_token.stop_requested()) {
    return Canceled(started);
  }
  if (input.request_id.empty() || !ValidState(*state)) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
        "HOPPER_REQUEST_INVALID", started);
  }
  if (!ValidCapability(*capability)) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
        "HOPPER_CAPABILITY_INVALID", started);
  }
  const auto* point = std::get_if<PointGoal>(&input.goal_map.target);
  if (point == nullptr || point->tolerance_m != 0.0 ||
      input.goal_map.yaw_rad.has_value() ||
      input.goal_map.yaw_tolerance_rad != 0.0 ||
      !Finite(point->position_m)) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
        "HOPPER_EXACT_POINT_REQUIRED", started);
  }
  const AvailableSingleHopDeltaVResult available =
      AvailableSingleHopDeltaV(*capability);
  if (!available.ok()) {
    return Failure(
        PlanningOutcome::kInvalidRequest,
        ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
        available.reason_code, started);
  }

  try {
    const shared::MapSnapshotBuildResult global_map =
        shared::MapSnapshot::Create(input.world.global_map);
    const shared::MapSnapshotBuildResult local_map =
        shared::MapSnapshot::Create(input.world.local_map);
    if (!global_map.ok() || !local_map.ok()) {
      return Failure(
          PlanningOutcome::kInvalidRequest,
          ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
          global_map.ok() ? local_map.reason_code : global_map.reason_code,
          started);
    }
    const auto local_goal = hierarchical::TransformGoal(
        input.goal_map, input.world.map_from_odom,
        hierarchical::TransformDirection::kParentToChild);
    const auto launch_pose_map = hierarchical::TransformPose(
        state->pose, input.world.map_from_odom,
        hierarchical::TransformDirection::kChildToParent);
    if (!local_goal.has_value() || !launch_pose_map.has_value()) {
      return Failure(
          PlanningOutcome::kInvalidRequest,
          ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
          "FRAME_TRANSFORM_INVALID", started);
    }

    LandingRegionResult landing = CertifyExactLandingRegion(
        *local_map.snapshot, *local_goal, *capability,
        input.config.map_safety, input.stop_token);
    if (!landing.ok()) {
      if (landing.status == LandingRegionStatus::kCanceled) {
        return Canceled(started, landing.inspected_cells);
      }
      const bool numerical =
          landing.reason_code.find("NUMERICAL") != std::string::npos;
      const PlanningOutcome outcome =
          landing.status == LandingRegionStatus::kInvalidRequest
              ? (numerical ? PlanningOutcome::kNumericalFailure
                           : PlanningOutcome::kInvalidRequest)
              : PlanningOutcome::kGoalInfeasible;
      return Failure(
          outcome,
          outcome == PlanningOutcome::kGoalInfeasible
              ? ExecutionDirective::kHoldPosition
              : ExecutionDirective::kNoSafeReference,
          CandidateDisposition::kKeep, landing.reason_code, started,
          landing.inspected_cells);
    }

    const auto landing_map = hierarchical::TransformPoint(
        landing.region->aim_position_on_surface_m,
        input.world.map_from_odom,
        hierarchical::TransformDirection::kChildToParent);
    bool map_boundary_ok = false;
    std::vector<Vec3> landing_boundary_map = TransformBoundary(
        landing.region->boundary_m, input.world.map_from_odom,
        hierarchical::TransformDirection::kChildToParent, map_boundary_ok);
    if (!landing_map.has_value() || !map_boundary_ok) {
      return Failure(
          PlanningOutcome::kInvalidRequest,
          ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
          "FRAME_TRANSFORM_INVALID", started, landing.inspected_cells);
    }

    const SingleHopCertificationResult hop = CertifySingleHop(
        SingleHopCertificationProblem{
            .launch_position_m = launch_pose_map->position_m,
            .landing_position_m = *landing_map,
            .gravity_mps2 = capability->gravity_mps2,
            .flight_map = global_map.snapshot.get(),
            .capability = capability,
            .map_safety = &input.config.map_safety,
            .stop_token = input.stop_token,
        });
    std::uint64_t total_work = landing.inspected_cells +
        hop.examined_intervals;
    if (!hop.ok()) {
      switch (hop.status) {
        case HopCertificationStatus::kCanceled:
          return Canceled(started, total_work);
        case HopCertificationStatus::kInvalid:
          return Failure(
              PlanningOutcome::kInvalidRequest,
              ExecutionDirective::kNoSafeReference,
              CandidateDisposition::kKeep, hop.reason_code, started,
              total_work);
        case HopCertificationStatus::kNumericalIndeterminate:
          return Failure(
              PlanningOutcome::kNumericalFailure,
              ExecutionDirective::kNoSafeReference,
              CandidateDisposition::kKeep, hop.reason_code, started,
              total_work);
        case HopCertificationStatus::kResourceExhausted:
          return Failure(
              PlanningOutcome::kResourceExhausted,
              ExecutionDirective::kNoSafeReference,
              CandidateDisposition::kKeep, hop.reason_code, started,
              total_work);
        case HopCertificationStatus::kInfeasible: {
          return Failure(
              PlanningOutcome::kNoKnownSafeRoute,
              ExecutionDirective::kNoSafeReference,
              CandidateDisposition::kKeep, hop.reason_code, started,
              total_work);
        }
        case HopCertificationStatus::kCertified:
          break;
      }
      return Failure(
          PlanningOutcome::kNumericalFailure,
          ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
          "HOPPER_CERTIFICATION_RESULT_INVALID", started, total_work);
    }

    const CertifiedSingleHop& certified = *hop.certification;
    const double minimum_region_radius = local_map.snapshot->resolution_m() *
        std::sqrt(std::numeric_limits<double>::epsilon());
    while (true) {
      double region_radius = 0.0;
      bool corners_envelope_certified = true;
      for (const Vec3 vertex : landing_boundary_map) {
        region_radius = std::max(
            region_radius,
            std::hypot(vertex.x - landing_map->x,
                       vertex.y - landing_map->y));
        const BallisticSolveResult vertex_arc = SolveBallisticArc(
            launch_pose_map->position_m, vertex, capability->gravity_mps2,
            certified.arc.flight_time_s);
        if (!vertex_arc.ok()) {
          return Failure(
              PlanningOutcome::kNumericalFailure,
              ExecutionDirective::kNoSafeReference,
              CandidateDisposition::kKeep,
              "HOPPER_LANDING_REGION_NUMERICAL_INDETERMINATE", started,
              total_work);
        }
        const SingleHopEnvelopeResult vertex_envelope =
            EvaluateSingleHopEnvelope(*vertex_arc.arc, *capability);
        if (!vertex_envelope.ok()) {
          if (vertex_envelope.reason_code !=
              "HOPPER_SINGLE_HOP_ENVELOPE_EXCEEDED") {
            return Failure(
                PlanningOutcome::kNumericalFailure,
                ExecutionDirective::kNoSafeReference,
                CandidateDisposition::kKeep,
                "HOPPER_LANDING_REGION_NUMERICAL_INDETERMINATE", started,
                total_work);
          }
          corners_envelope_certified = false;
          break;
        }
      }
      FlightTubeCertificationResult region_tube;
      if (corners_envelope_certified) {
        region_tube = CertifyFlightTube(
            certified.arc, *global_map.snapshot, *capability,
            input.config.map_safety, input.stop_token, region_radius);
        total_work += region_tube.overlapped_cell_count;
        if (region_tube.canceled) {
          return Canceled(started, total_work);
        }
        if (region_tube.reason_code.find("NUMERICAL") != std::string::npos) {
          return Failure(
              PlanningOutcome::kNumericalFailure,
              ExecutionDirective::kNoSafeReference,
              CandidateDisposition::kKeep,
              "HOPPER_LANDING_REGION_NUMERICAL_INDETERMINATE", started,
              total_work);
        }
      }
      if (corners_envelope_certified && region_tube.certified) {
        break;
      }
      if (!(region_radius > minimum_region_radius)) {
        return Failure(
            PlanningOutcome::kGoalInfeasible,
            ExecutionDirective::kHoldPosition, CandidateDisposition::kKeep,
            "LANDING_REGION_NOT_CERTIFIABLE", started, total_work);
      }
      const Vec3 center = landing.region->aim_position_on_surface_m;
      for (Vec3& vertex : landing.region->boundary_m) {
        vertex.x = std::midpoint(vertex.x, center.x);
        vertex.y = std::midpoint(vertex.y, center.y);
        vertex.z = std::midpoint(vertex.z, center.z);
      }
      landing.region->area_m2 *= 0.25;
      landing_boundary_map = TransformBoundary(
          landing.region->boundary_m, input.world.map_from_odom,
          hierarchical::TransformDirection::kChildToParent, map_boundary_ok);
      if (!map_boundary_ok) {
        return Failure(
            PlanningOutcome::kInvalidRequest,
            ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
            "FRAME_TRANSFORM_INVALID", started, total_work);
      }
    }
    const auto launch_velocity_odom = RotateMapVectorToOdom(
        certified.arc.launch_velocity_mps, input.world.map_from_odom);
    if (!launch_velocity_odom.has_value()) {
      return Failure(
          PlanningOutcome::kInvalidRequest,
          ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
          "FRAME_TRANSFORM_INVALID", started, total_work);
    }
    const auto flight_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>{certified.arc.flight_time_s});
    if (flight_time.count() <= 0) {
      return Failure(
          PlanningOutcome::kNumericalFailure,
          ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
          "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE", started, total_work);
    }

    const std::string segment_id = input.request_id + "/single-hop";
    HopSegment segment{
        .segment_id = segment_id,
        .launch_pose = state->pose,
        .landing_region_boundary_m = landing.region->boundary_m,
        .flight_time = flight_time,
        .launch_velocity_mps = *launch_velocity_odom,
        .flight_tube_radius_m = certified.flight_tube.radius_m,
        .nominal_landing_point_m =
            landing.region->aim_position_on_surface_m,
        .required_delta_v_mps =
            certified.envelope.required_delta_v_mps,
        .available_delta_v_mps =
            certified.envelope.available_delta_v_mps,
        .capability_version = input.capability_version,
        .global_map_generation = input.global_map_generation,
        .local_map_generation = input.local_map_generation,
    };
    CertifiedHopPreview preview{
        .segment_id = segment_id,
        .launch_pose_map = *launch_pose_map,
        .landing_pose_map = Pose3{.position_m = *landing_map},
        .launch_velocity_mps = certified.arc.launch_velocity_mps,
        .flight_time = flight_time,
        .flight_tube_radius_m = certified.flight_tube.radius_m,
        .landing_region_map = landing_boundary_map,
        .promotion_region_map = {},
        .position_uncertainty_m = input.position_uncertainty_m,
        .velocity_uncertainty_mps = input.velocity_uncertainty_mps,
    };
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started);
    const LocalTrajectoryDiagnostics local_diagnostics{
        .trajectory_mode = TrajectoryMode::kCertifiedHop,
        .start_anchor_error_m = 0.0,
        .endpoint_error_m = 0.0,
        .collision_validation = CollisionValidation::kCertified,
        .landing_field_elapsed_s = 0.0,
    };
    return {
        .outcome = PlanningOutcome::kNewReferenceAvailable,
        .directive = ExecutionDirective::kActivateNewReference,
        .candidate_disposition = CandidateDisposition::kKeep,
        .reason_code = "HOPPER_SINGLE_HOP_AVAILABLE",
        .reference = MotionReference{
            .plan_id = "hopper/" + input.request_id,
            .platform_type = PlatformType::kHopper,
            .input_time = input.state_time,
            .preview = GlobalRoutePreview{
                .poses_map = {
                    *launch_pose_map,
                    Pose3{.position_m = *landing_map},
                },
            },
            .data = HopReference{.segments = {std::move(segment)}},
        },
        .diagnostics = PlannerDiagnostics{
            .planner_name = std::string{kPlannerName},
            .elapsed = elapsed,
            .expanded_states = total_work,
            .best_cost = certified.envelope.required_delta_v_mps,
            .hierarchical = Metrics(
                input, landing.inspected_cells, hop.examined_intervals,
                elapsed),
            .local_trajectory = local_diagnostics,
        },
        .certified_hops = {std::move(preview)},
        .continuation = nullptr,
    };
  } catch (const std::bad_alloc&) {
    return Failure(
        PlanningOutcome::kResourceExhausted,
        ExecutionDirective::kNoSafeReference, CandidateDisposition::kKeep,
        "HOPPER_SEARCH_ALLOCATION_FAILED", started);
  }
}

}  // namespace lunar::planning::hopper
