#include "lunar_path_planner/v3/hopper/hop_certifier.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] Eigen::Vector3d ToEigen(const Vec3 value) {
  return {value.x, value.y, value.z};
}

[[nodiscard]] Vec3 ToContract(const Eigen::Vector3d& value) {
  return {value.x(), value.y(), value.z()};
}

[[nodiscard]] Eigen::Quaterniond ToEigen(
    const Quaternion value) {
  return {value.w, value.x, value.y, value.z};
}

[[nodiscard]] HopperKinematicState ContractState(
    const HopperState& state) {
  return {
      .position_m = state.position_m,
      .orientation_body_to_frame =
          state.orientation_body_to_frame,
      .linear_velocity_mps = state.linear_velocity_mps,
      .angular_velocity_radps = state.angular_velocity_radps,
  };
}

[[nodiscard]] DurationNanoseconds DurationFromSeconds(
    const double seconds) {
  return DurationNanoseconds{
      std::chrono::nanoseconds{
          static_cast<std::int64_t>(
              std::llround(seconds * 1.0e9))}};
}

[[nodiscard]] double Seconds(
    const DurationNanoseconds duration) {
  return std::chrono::duration<double>(duration.value).count();
}

[[nodiscard]] NextLandingRegion ContractRegion(
    const TerrainCertifiedLandingRegion& region) {
  return {
      .region_id = region.region_id,
      .frame_id = region.frame_id,
      .landing_plane = region.landing_plane,
      .convex_polygon =
          to_contract_polygon(region.vertices_uv_ccw),
      .allowed_yaw_interval = region.allowed_yaw_interval,
      .terrain_certification_ref =
          region.terrain_certification.source_snapshot_ref,
      .inward_safety_margin_m =
          region.inward_safety_margin_m,
  };
}

[[nodiscard]] bool BasicRegionValid(
    const TerrainCertifiedLandingRegion& region,
    const HopCertificationContext& context) {
  return context.map != nullptr &&
         region.frame_id == context.map->frame_id() &&
         region.terrain_certification.source_snapshot_ref ==
             context.map->snapshot_ref() &&
         validate_landing_plane(region.landing_plane).ok() &&
         validate_convex_polygon(region.vertices_uv_ccw).ok() &&
         validate_circular_yaw_interval(
             region.allowed_yaw_interval)
             .ok() &&
         std::isfinite(region.inward_safety_margin_m) &&
         region.inward_safety_margin_m >= 0.0;
}

[[nodiscard]] bool Better(
    const BallisticCandidate& lhs,
    const BallisticCandidate& rhs,
    const DurationNanoseconds tolerance) {
  const double tolerance_s = Seconds(tolerance);
  if (lhs.expected_execution_time_s + tolerance_s <
      rhs.expected_execution_time_s) {
    return true;
  }
  if (rhs.expected_execution_time_s + tolerance_s <
      lhs.expected_execution_time_s) {
    return false;
  }
  return std::tuple{
             lhs.estimated_energy_j,
             -lhs.landing_margin_m,
             -lhs.nominal_minimum_clearance_m,
             lhs.candidate_id} <
         std::tuple{
             rhs.estimated_energy_j,
             -rhs.landing_margin_m,
             -rhs.nominal_minimum_clearance_m,
             rhs.candidate_id};
}

void AppendIssues(
    ValidationReport& destination,
    const ValidationReport& source) {
  destination.issues.insert(
      destination.issues.end(), source.issues.begin(),
      source.issues.end());
}

}  // namespace

HopCertifier::HopCertifier(HopCertificationContext context)
    : context_(std::move(context)) {}

FirstEdgeCertificationResult HopCertifier::certify_first_edge(
    const LandingGraphNode& from,
    const LandingGraphNode& to) {
  diagnostics_ = {};
  if (context_.map == nullptr) {
    return {
        .rejection_reason = "missing_map",
    };
  }
  if (!BasicRegionValid(from.region, context_) ||
      !BasicRegionValid(to.region, context_)) {
    return {
        .rejection_reason =
            "terrain_landing_region_not_certified",
    };
  }
  if (to.aim_points.empty()) {
    return {
        .rejection_reason = "no_nominal_aim_point",
    };
  }
  const Eigen::Quaterniond initial_orientation =
      ToEigen(context_.launch_state.orientation_body_to_frame);
  const Eigen::Vector3d initial_angular_velocity =
      ToEigen(context_.launch_state.angular_velocity_radps);
  const AttitudeCertifier attitude_certifier{
      attitude_capability_view(context_.capability)};
  const auto required_attitude_time =
      attitude_certifier.conservative_required_time_s(
          initial_orientation, initial_angular_velocity,
          context_.launch_state.error_bounds.orientation_bound,
          context_.launch_state.error_bounds
              .angular_velocity_bound_radps,
          to.region.landing_plane,
          to.region.allowed_yaw_interval);
  if (!IsOk(required_attitude_time)) {
    return {
        .rejection_reason =
            "attitude_time_lower_bound_not_certified",
    };
  }
  const FlightTimeSearchInput search_input{
      .launch_position_m =
          ToEigen(context_.launch_state.position_m),
      .frame_id = context_.capability.frame_id,
      .aim_points = to.aim_points,
      .gravity =
          gravity_model_view(context_.capability.gravity_model),
      .minimum_attitude_time_s =
          std::get<double>(required_attitude_time) +
          Seconds(
              context_.capability.attitude_envelope
                  .minimum_settle_guard),
      .launch_preparation_time_s =
          Seconds(
              context_.capability.actuator_or_impulse_profile
                  .launch_preparation_time),
      .landing_settle_time_s =
          Seconds(
              context_.capability.actuator_or_impulse_profile
                  .landing_settle_time),
  };
  const auto solve =
      BallisticTimeSolver{
          context_.capability, context_.limits}
          .solve(search_input);
  diagnostics_.ballistic_candidate_count =
      solve.candidates.size();
  if (solve.candidates.empty()) {
    return {
        .rejection_reason = solve.diagnostics.termination_reason,
    };
  }

  struct Certified final {
    BallisticCandidate candidate;
    AimPointCandidate aim_point;
    CertifiedFlightTube tube;
    AttitudeBoundary attitude;
    PredictedLandingFootprint footprint;
  };
  std::optional<Certified> selected;
  for (const BallisticCandidate& candidate : solve.candidates) {
    if (diagnostics_.fully_attempted_candidate_count >=
        context_.limits.maximum_full_certification_attempts) {
      break;
    }
    ++diagnostics_.fully_attempted_candidate_count;
    const auto aim_iterator = std::find_if(
        to.aim_points.begin(), to.aim_points.end(),
        [&candidate](const AimPointCandidate& aim) {
          return aim.aim_point_id ==
                 candidate.arc.aim_point_id;
        });
    if (aim_iterator == to.aim_points.end()) {
      diagnostics_.candidate_rejection_reasons.push_back(
          "aim_point_identity_mismatch");
      continue;
    }
    const auto tube = FlightTubeCertifier{
        context_.capability, context_.limits}
                          .certify(
                              {
                                  .arc = candidate.arc,
                                  .map = context_.map,
                                  .source_region = from.region,
                                  .target_region = to.region,
                                  .initial_position_error_m =
                                      context_.capability.error_model
                                          .initial_position_error_m,
                                  .initial_velocity_error_mps =
                                      context_.capability.error_model
                                          .initial_velocity_error_mps,
                                  .launch_execution_velocity_error_mps =
                                      context_.capability.error_model
                                          .launch_execution_velocity_error_mps,
                                  .gravity_error_mps2 =
                                      context_.capability.error_model
                                          .gravity_error_mps2,
                                  .arbitrary_attitude_body_envelope =
                                      context_.capability
                                          .body_rotation_envelope
                                          .arbitrary_attitude_body_envelope,
                                  .source_snapshot_ref =
                                      context_.map->snapshot_ref(),
                                  .body_rotation_envelope_ref =
                                      context_.capability
                                          .body_rotation_envelope
                                          .content_ref,
                                  .error_model_ref =
                                      context_.source_error_model_ref,
                              });
    if (!tube.certified_tube.has_value()) {
      diagnostics_.candidate_rejection_reasons.push_back(
          tube.diagnostics.rejection_reason);
      continue;
    }
    const auto attitude = attitude_certifier.certify(
        {
            .initial_orientation_body_to_frame =
                initial_orientation,
            .initial_angular_velocity_radps =
                initial_angular_velocity,
            .initial_orientation_error_set =
                context_.launch_state.error_bounds
                    .orientation_bound,
            .initial_angular_velocity_error_set_radps =
                context_.launch_state.error_bounds
                    .angular_velocity_bound_radps,
            .landing_plane = to.region.landing_plane,
            .allowed_yaw_interval =
                to.region.allowed_yaw_interval,
            .flight_time_s = candidate.arc.flight_time_s,
        });
    if (!attitude.boundary.has_value()) {
      diagnostics_.candidate_rejection_reasons.push_back(
          attitude.rejection_reason);
      continue;
    }
    const auto landing =
        LandingSetPropagator{
            context_.capability, context_.limits}
            .propagate(
                {
                    .arc = candidate.arc,
                    .landing_plane =
                        to.region.landing_plane,
                    .initial_position_error_m =
                        context_.capability.error_model
                            .initial_position_error_m,
                    .initial_velocity_error_mps =
                        context_.capability.error_model
                            .initial_velocity_error_mps,
                    .gravity_error_mps2 =
                        context_.capability.error_model
                            .gravity_error_mps2,
                    .launch_execution_velocity_error_mps =
                        context_.capability.error_model
                            .launch_execution_velocity_error_mps,
                    .landing_plane_error =
                        {
                            .origin_error_m =
                                context_.capability.error_model
                                    .landing_plane_origin_error_m,
                            .normal_error =
                                context_.capability.error_model
                                    .landing_plane_normal_error,
                            .residual_error_m =
                                context_.capability.error_model
                                    .landing_plane_residual_error_m,
                        },
                    .certified_landing_yaw_interval =
                        to.region.allowed_yaw_interval,
                    .source_error_model_ref =
                        context_.source_error_model_ref,
                });
    if (!landing.footprint.has_value()) {
      diagnostics_.candidate_rejection_reasons.push_back(
          landing.diagnostics.rejection_reason);
      continue;
    }
    const NextLandingRegion next_region =
        ContractRegion(to.region);
    const auto containment = validate_landing_containment(
        *landing.footprint, next_region, *attitude.boundary);
    if (!containment.ok()) {
      diagnostics_.candidate_rejection_reasons.push_back(
          "landing_set_not_contained");
      continue;
    }
    Certified complete{
        .candidate = candidate,
        .aim_point = *aim_iterator,
        .tube = *tube.certified_tube,
        .attitude = *attitude.boundary,
        .footprint = *landing.footprint,
    };
    ++diagnostics_.certified_candidate_count;
    if (!selected.has_value() ||
        Better(
            complete.candidate, selected->candidate,
            context_.limits.time_equivalence_tolerance)) {
      selected = std::move(complete);
    }
  }
  if (!selected.has_value()) {
    return {
        .rejection_reason =
            diagnostics_.candidate_rejection_reasons.empty()
                ? "no_fully_certified_candidate"
                : diagnostics_.candidate_rejection_reasons.back(),
    };
  }

  diagnostics_.selected_candidate_id =
      selected->candidate.candidate_id;
  HopperKinematicState hold_state =
      ContractState(context_.launch_state);
  hold_state.linear_velocity_mps = {0.0, 0.0, 0.0};
  hold_state.angular_velocity_radps = {0.0, 0.0, 0.0};
  HopperKinematicState launch_state =
      ContractState(context_.launch_state);
  launch_state.linear_velocity_mps =
      ToContract(selected->candidate.arc.launch_velocity_mps);
  const DurationNanoseconds flight_time =
      DurationFromSeconds(
          selected->candidate.arc.flight_time_s);
  HopperReference reference{
      .reference_id =
          context_.source_request_id + "-hopper-" +
          selected->candidate.candidate_id,
      .reference_hash = std::string(64U, '0'),
      .reference_time_origin = context_.reference_time_origin,
      .ground_hold_anchor =
          {
              .anchor_id =
                  context_.source_request_id + "-ground-hold",
              .hold_state = hold_state,
              .allowed_hold_state_error_set =
                  context_.launch_state.error_bounds,
              .terrain_certification_ref =
                  from.region.terrain_certification
                      .source_snapshot_ref,
          },
      .next_landing_region = ContractRegion(to.region),
      .jump_boundary =
          {
              .boundary_id =
                  context_.source_request_id + "-jump-" +
                  selected->candidate.candidate_id,
              .nominal_launch_state = launch_state,
              .allowed_launch_state_error_set =
                  context_.capability.certified_state_error_bounds,
              .gravity_model_ref =
                  context_.capability.gravity_model.content_ref,
              .ballistic_flight_time = flight_time,
              .actuator_or_impulse_profile_ref =
                  context_.capability
                      .actuator_or_impulse_profile.content_ref,
          },
      .predicted_landing_footprint =
          selected->footprint,
      .certified_flight_tube = selected->tube,
      .attitude_boundary = selected->attitude,
      .nominal_aim_point =
          {
              .position_m =
                  ToContract(selected->aim_point.position_m),
          },
      .physical_certification_ref =
          context_.capability.source_safety_capability_ref,
      .future_route_preview =
          FutureRoutePreview{
              .future_viability = FutureViability::kUnknown,
              .reason_code = "FUTURE_ROUTE_NON_AUTHORITATIVE",
          },
  };
  const auto hash =
      CanonicalReferenceHash(PlatformReference{reference});
  if (!IsOk(hash)) {
    return {
        .rejection_reason =
            "canonical_reference_hash_failed",
    };
  }
  reference.reference_hash =
      std::get<Sha256Digest>(hash);
  if (!validate_final_hopper_reference(reference, context_).ok()) {
    return {
        .rejection_reason =
            "final_hopper_reference_validation_failed",
    };
  }
  return {
      .certified_candidate =
          FirstEdgeCertificationResult::CertifiedCandidate{
              .reference = std::move(reference),
              .expected_execution_time =
                  DurationFromSeconds(
                      selected->candidate
                          .expected_execution_time_s),
              .secondary_costs =
                  {
                      .energy =
                          selected->candidate
                              .estimated_energy_j,
                  },
          },
  };
}

const HopCertificationDiagnostics& HopCertifier::diagnostics()
    const noexcept {
  return diagnostics_;
}

ValidationReport validate_final_hopper_reference(
    const HopperReference& reference,
    const HopCertificationContext& context) {
  ValidationReport report =
      SemanticValidator{}.Validate(
          PlatformReference{reference});
  AppendIssues(
      report,
      validate_landing_containment(
          reference.predicted_landing_footprint,
          reference.next_landing_region,
          reference.attitude_boundary));
  if (context.map == nullptr ||
      reference.certified_flight_tube.source_map_snapshot_ref !=
          context.map->snapshot_ref()) {
    report.issues.push_back(
        {
            "certified_flight_tube.source_map_snapshot_ref",
            "map_snapshot_mismatch",
            "flight tube must reference certification map",
        });
  }
  if (reference.jump_boundary.gravity_model_ref !=
      context.capability.gravity_model.content_ref) {
    report.issues.push_back(
        {
            "jump_boundary.gravity_model_ref",
            "gravity_model_mismatch",
            "jump boundary must use bound gravity model",
        });
  }
  const auto expected_hash =
      CanonicalReferenceHash(PlatformReference{reference});
  if (!IsOk(expected_hash) ||
      std::get<Sha256Digest>(expected_hash) !=
          reference.reference_hash) {
    report.issues.push_back(
        {
            "reference_hash",
            "canonical_hash_mismatch",
            "reference hash must use shared canonical hashing",
        });
  }
  return report;
}

}  // namespace lunar::planning::v3
