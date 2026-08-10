#include "lunar_planner_core/reachability_projection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <new>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "hierarchical/frame_transform.hpp"
#include "hopper/ballistic_envelope.hpp"
#include "hopper/ballistic_kinematics.hpp"
#include "hopper/flight_tube_certifier.hpp"
#include "hopper/hop_certifier.hpp"
#include "hopper/landing_region.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::planning {
namespace {

constexpr Vec3 kLunarGravityMps2{0.0, 0.0, -1.62};
constexpr double kDistanceToleranceM = 1.0e-9;
constexpr std::string_view kHopperLandingEvidenceAlgorithm =
    "cpp-hopper-detail-landing-regions/v1";

[[nodiscard]] ReachabilityProjectionResult Failure(std::string reason_code) {
  return ReachabilityProjectionResult{
      .projection = std::nullopt,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] const Pose3* StatePose(const PlatformState& state,
                                     const PlatformType platform) noexcept {
  switch (platform) {
    case PlatformType::kWheeled: {
      const auto* value = std::get_if<WheeledState>(&state);
      return value == nullptr ? nullptr : &value->pose;
    }
    case PlatformType::kLegged: {
      const auto* value = std::get_if<LeggedState>(&state);
      return value == nullptr ? nullptr : &value->body_pose;
    }
    case PlatformType::kHopper: {
      const auto* value = std::get_if<HopperState>(&state);
      return value == nullptr ? nullptr : &value->pose;
    }
  }
  return nullptr;
}

[[nodiscard]] std::optional<shared::GridCell> StartCell(
    const PlannerInput& input, const shared::MapSnapshot& global_map,
    const PlatformType platform) noexcept {
  const Pose3* pose = StatePose(input.current_state, platform);
  if (pose == nullptr) {
    return std::nullopt;
  }
  const auto pose_map = hierarchical::TransformPose(
      *pose, input.world.map_from_odom,
      hierarchical::TransformDirection::kChildToParent);
  if (!pose_map.has_value()) {
    return std::nullopt;
  }
  return global_map.PositionToCell(
      Vec2{pose_map->position_m.x, pose_map->position_m.y});
}

[[nodiscard]] std::optional<hopper::CertifiedLandingRegion>
CertifiedLandingAtPosition(const Vec3 center_map,
                           const shared::MapSnapshot& local_map,
                           const PlannerInput& input,
                           const HopperCapability& capability,
                           std::string& fatal_reason) {
  const auto center_local = hierarchical::TransformPoint(
      center_map, input.world.map_from_odom,
      hierarchical::TransformDirection::kParentToChild);
  if (!center_local.has_value()) {
    fatal_reason = "FRAME_TRANSFORM_INVALID";
    return std::nullopt;
  }
  GoalRegion goal{
      .goal_id = "reachability-landing",
      .target = PointGoal{
          .position_m = *center_local,
          .tolerance_m = 0.0,
      },
      .yaw_rad = std::nullopt,
      .yaw_tolerance_rad = 0.0,
  };
  hopper::LandingRegionResult landing = hopper::CertifyExactLandingRegion(
      local_map, goal, capability, input.config.map_safety,
      input.stop_token);
  if (!landing.ok()) {
    switch (landing.status) {
      case hopper::LandingRegionStatus::kInfeasible:
        return std::nullopt;
      case hopper::LandingRegionStatus::kCanceled:
        fatal_reason = "REQUEST_CANCELED";
        return std::nullopt;
      case hopper::LandingRegionStatus::kResourceExhausted:
        fatal_reason = "REACHABILITY_RESOURCE_EXHAUSTED";
        return std::nullopt;
      case hopper::LandingRegionStatus::kInvalidRequest:
        fatal_reason = landing.reason_code.empty()
            ? "HOPPER_LANDING_REGION_INVALID"
            : landing.reason_code;
        return std::nullopt;
      case hopper::LandingRegionStatus::kCertified:
        fatal_reason = "HOPPER_LANDING_RESULT_INVALID";
        return std::nullopt;
    }
  }
  hopper::CertifiedLandingRegion transformed = std::move(*landing.region);
  const auto aim_map = hierarchical::TransformPoint(
      transformed.aim_position_on_surface_m, input.world.map_from_odom,
      hierarchical::TransformDirection::kChildToParent);
  if (!aim_map.has_value()) {
    fatal_reason = "FRAME_TRANSFORM_INVALID";
    return std::nullopt;
  }
  transformed.aim_position_on_surface_m = *aim_map;
  for (Vec3& vertex : transformed.boundary_m) {
    const auto value = hierarchical::TransformPoint(
        vertex, input.world.map_from_odom,
        hierarchical::TransformDirection::kChildToParent);
    if (!value.has_value()) {
      fatal_reason = "FRAME_TRANSFORM_INVALID";
      return std::nullopt;
    }
    vertex = *value;
  }
  return transformed;
}

[[nodiscard]] std::optional<hopper::CertifiedLandingRegion>
CertifiedLandingAt(const shared::MapSnapshot& global_map,
                   const shared::MapSnapshot& local_map,
                   const shared::GridCell cell,
                   const PlannerInput& input,
                   const HopperCapability& capability,
                   std::string& fatal_reason) {
  return CertifiedLandingAtPosition(
      global_map.CellCenter(cell), local_map, input, capability,
      fatal_reason);
}

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

struct ExternalLandingBuildResult final {
  std::optional<
      std::vector<std::optional<hopper::CertifiedLandingRegion>>> landings;
  std::string reason_code;
};

[[nodiscard]] ExternalLandingBuildResult BuildExternalLandings(
    const shared::MapSnapshot& global_map,
    const HopperLandingEvidenceGrid& evidence) {
  if (evidence.width != global_map.width() ||
      evidence.height != global_map.height() ||
      evidence.landings.size() != global_map.cell_count()) {
    return {.reason_code = "HOPPER_LANDING_EVIDENCE_GEOMETRY_INVALID"};
  }
  if (evidence.algorithm_id != kHopperLandingEvidenceAlgorithm) {
    return {.reason_code = "HOPPER_LANDING_EVIDENCE_ALGORITHM_INVALID"};
  }
  std::vector<std::optional<hopper::CertifiedLandingRegion>> output(
      evidence.landings.size());
  for (std::size_t index = 0U; index < evidence.landings.size(); ++index) {
    const HopperLandingEvidence& landing = evidence.landings[index];
    if (landing.certified > 1U) {
      return {.reason_code = "HOPPER_LANDING_EVIDENCE_VALUE_INVALID"};
    }
    if (landing.certified == 0U) {
      continue;
    }
    if (!Finite(landing.aim_position_on_surface_m) ||
        !std::isfinite(landing.area_m2) || landing.area_m2 <= 0.0 ||
        !std::ranges::all_of(landing.boundary_m, Finite)) {
      return {.reason_code = "HOPPER_LANDING_EVIDENCE_VALUE_INVALID"};
    }
    const auto cell = global_map.PositionToCell(Vec2{
        landing.aim_position_on_surface_m.x,
        landing.aim_position_on_surface_m.y,
    });
    const shared::GridCell expected{
        .x = static_cast<std::int32_t>(index % global_map.width()),
        .y = static_cast<std::int32_t>(index / global_map.width()),
    };
    if (!cell.has_value() || cell->x != expected.x ||
        cell->y != expected.y) {
      return {.reason_code = "HOPPER_LANDING_EVIDENCE_CELL_INVALID"};
    }
    output[index] = hopper::CertifiedLandingRegion{
        .seed_cell = expected,
        .aim_position_on_surface_m = landing.aim_position_on_surface_m,
        .boundary_m = std::vector<Vec3>(
            landing.boundary_m.begin(), landing.boundary_m.end()),
        .area_m2 = landing.area_m2,
    };
  }
  return {.landings = std::move(output)};
}

struct LandingRegionHopResult final {
  hopper::HopCertificationStatus status{
      hopper::HopCertificationStatus::kInvalid};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return status == hopper::HopCertificationStatus::kCertified &&
        reason_code.empty();
  }
};

[[nodiscard]] LandingRegionHopResult CertifyLandingRegionHop(
    const hopper::CertifiedSingleHop& center_hop,
    hopper::CertifiedLandingRegion landing_region,
    const shared::MapSnapshot& flight_map,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety,
    const std::stop_token stop_token) {
  const double minimum_region_radius = flight_map.resolution_m() *
      std::sqrt(std::numeric_limits<double>::epsilon());
  while (true) {
    if (stop_token.stop_requested()) {
      return {
          .status = hopper::HopCertificationStatus::kCanceled,
          .reason_code = "REQUEST_CANCELED",
      };
    }
    double region_radius = 0.0;
    bool corners_envelope_certified = true;
    for (const Vec3 vertex : landing_region.boundary_m) {
      region_radius = std::max(
          region_radius,
          std::hypot(
              vertex.x - landing_region.aim_position_on_surface_m.x,
              vertex.y - landing_region.aim_position_on_surface_m.y));
      const hopper::BallisticSolveResult vertex_arc =
          hopper::SolveBallisticArc(
              center_hop.arc.launch_position_m, vertex,
              center_hop.arc.gravity_mps2, center_hop.arc.flight_time_s);
      if (!vertex_arc.ok()) {
        return {
            .status =
                hopper::HopCertificationStatus::kNumericalIndeterminate,
            .reason_code =
                "HOPPER_LANDING_REGION_NUMERICAL_INDETERMINATE",
        };
      }
      const hopper::SingleHopEnvelopeResult vertex_envelope =
          hopper::EvaluateSingleHopEnvelope(*vertex_arc.arc, capability);
      if (!vertex_envelope.ok()) {
        if (vertex_envelope.reason_code !=
            "HOPPER_SINGLE_HOP_ENVELOPE_EXCEEDED") {
          return {
              .status =
                  hopper::HopCertificationStatus::kNumericalIndeterminate,
              .reason_code =
                  "HOPPER_LANDING_REGION_NUMERICAL_INDETERMINATE",
          };
        }
        corners_envelope_certified = false;
        break;
      }
    }
    hopper::FlightTubeCertificationResult region_tube;
    if (corners_envelope_certified) {
      region_tube = hopper::CertifyFlightTube(
          center_hop.arc, flight_map, capability, map_safety, stop_token,
          region_radius);
      if (region_tube.canceled) {
        return {
            .status = hopper::HopCertificationStatus::kCanceled,
            .reason_code = "REQUEST_CANCELED",
        };
      }
      if (region_tube.reason_code == "HOPPER_FLIGHT_TUBE_INPUT_INVALID" ||
          region_tube.reason_code.find("NUMERICAL") != std::string::npos) {
        return {
            .status =
                hopper::HopCertificationStatus::kNumericalIndeterminate,
            .reason_code =
                "HOPPER_LANDING_REGION_NUMERICAL_INDETERMINATE",
        };
      }
    }
    if (corners_envelope_certified && region_tube.certified) {
      return {.status = hopper::HopCertificationStatus::kCertified};
    }
    if (!(region_radius > minimum_region_radius)) {
      return {
          .status = hopper::HopCertificationStatus::kInfeasible,
          .reason_code = "LANDING_REGION_NOT_CERTIFIABLE",
      };
    }
    const Vec3 center = landing_region.aim_position_on_surface_m;
    for (Vec3& vertex : landing_region.boundary_m) {
      vertex.x = std::midpoint(vertex.x, center.x);
      vertex.y = std::midpoint(vertex.y, center.y);
      vertex.z = std::midpoint(vertex.z, center.z);
    }
    landing_region.area_m2 *= 0.25;
  }
}

[[nodiscard]] LandingRegionHopResult CertifyDirectedLandingRegionEdge(
    const hopper::CertifiedLandingRegion& source,
    const hopper::CertifiedLandingRegion& target,
    const shared::MapSnapshot& flight_map,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety,
    const std::stop_token stop_token) {
  const hopper::SingleHopCertificationResult hop = hopper::CertifySingleHop(
      hopper::SingleHopCertificationProblem{
          .launch_position_m = source.aim_position_on_surface_m,
          .landing_position_m = target.aim_position_on_surface_m,
          .gravity_mps2 = kLunarGravityMps2,
          .flight_map = &flight_map,
          .capability = &capability,
          .map_safety = &map_safety,
          .stop_token = stop_token,
      });
  if (!hop.ok()) {
    return {.status = hop.status, .reason_code = hop.reason_code};
  }
  return CertifyLandingRegionHop(
      *hop.certification, target, flight_map, capability, map_safety,
      stop_token);
}

[[nodiscard]] std::string EdgeFailureReason(
    const LandingRegionHopResult& edge) {
  switch (edge.status) {
    case hopper::HopCertificationStatus::kCanceled:
      return "REQUEST_CANCELED";
    case hopper::HopCertificationStatus::kResourceExhausted:
      return "REACHABILITY_RESOURCE_EXHAUSTED";
    case hopper::HopCertificationStatus::kInvalid:
    case hopper::HopCertificationStatus::kNumericalIndeterminate:
      return edge.reason_code.empty()
          ? "HOPPER_REACHABILITY_CERTIFICATION_INVALID"
          : edge.reason_code;
    case hopper::HopCertificationStatus::kCertified:
      return "HOPPER_REACHABILITY_CERTIFICATION_INVALID";
    case hopper::HopCertificationStatus::kInfeasible:
      return {};
  }
  return "HOPPER_REACHABILITY_CERTIFICATION_INVALID";
}

[[nodiscard]] ReachabilityProjectionResult ProjectGround(
    const PlannerInput& input,
    const std::shared_ptr<const shared::MapSnapshot>& global_map,
    const shared::SafeProjection& safe,
    const shared::GridCell start,
    const double maximum_edge_distance_m) {
  ReachabilityProjection projection{
      .platform_type = safe.platform_type(),
      .width = global_map->width(),
      .height = global_map->height(),
      .reachable = std::vector<std::uint8_t>(global_map->cell_count(), 0U),
      .algorithm_id = "cpp-ground-start-connected-component/v1",
      .maximum_edge_distance_m = maximum_edge_distance_m,
  };
  if (!safe.HardFeasible(start)) {
    return ReachabilityProjectionResult{
        .projection = std::move(projection),
        .reason_code = {},
    };
  }
  const std::int32_t component = safe.ConnectedComponent(start);
  if (component < 0) {
    return ReachabilityProjectionResult{
        .projection = std::move(projection),
        .reason_code = {},
    };
  }
  for (std::size_t index = 0U; index < global_map->cell_count(); ++index) {
    if (input.stop_token.stop_requested()) {
      return Failure("REQUEST_CANCELED");
    }
    const shared::GridCell cell{
        .x = static_cast<std::int32_t>(index % global_map->width()),
        .y = static_cast<std::int32_t>(index / global_map->width()),
    };
    projection.reachable[index] = static_cast<std::uint8_t>(
        safe.HardFeasible(cell) &&
        safe.ConnectedComponent(cell) == component);
  }
  return ReachabilityProjectionResult{
      .projection = std::move(projection),
      .reason_code = {},
  };
}

[[nodiscard]] ReachabilityProjectionResult ProjectHopper(
    const PlannerInput& input,
    const std::shared_ptr<const shared::MapSnapshot>& global_map,
    const std::shared_ptr<const shared::MapSnapshot>& local_map,
    const shared::SafeProjection& safe,
    const shared::GridCell start,
    const double maximum_edge_distance_m,
    const HopperLandingEvidenceGrid* const external_evidence,
    const bool direct_only) {
  const auto* capability = std::get_if<HopperCapability>(&input.capability);
  if (capability == nullptr) {
    return Failure("REACHABILITY_PLATFORM_STATE_MISMATCH");
  }
  ReachabilityProjection projection{
      .platform_type = PlatformType::kHopper,
      .width = global_map->width(),
      .height = global_map->height(),
      .reachable = std::vector<std::uint8_t>(global_map->cell_count(), 0U),
      .algorithm_id = direct_only
          ? "cpp-hopper-certified-bidirectional-direct/v1"
          : "cpp-hopper-certified-bidirectional-bfs/v3",
      .maximum_edge_distance_m = maximum_edge_distance_m,
  };

  std::vector<std::optional<hopper::CertifiedLandingRegion>> landings;
  if (external_evidence != nullptr) {
    ExternalLandingBuildResult built =
        BuildExternalLandings(*global_map, *external_evidence);
    if (!built.landings.has_value()) {
      return Failure(std::move(built.reason_code));
    }
    landings = std::move(*built.landings);
  } else {
    landings.resize(global_map->cell_count());
    for (std::size_t index = 0U; index < global_map->cell_count(); ++index) {
      if (input.stop_token.stop_requested()) {
        return Failure("REQUEST_CANCELED");
      }
      const shared::GridCell cell{
          .x = static_cast<std::int32_t>(index % global_map->width()),
          .y = static_cast<std::int32_t>(index / global_map->width()),
      };
      if (!safe.HardFeasible(cell)) {
        continue;
      }
      std::string fatal_reason;
      landings[index] = CertifiedLandingAt(
          *global_map, *local_map, cell, input, *capability, fatal_reason);
      if (!fatal_reason.empty()) {
        return Failure(std::move(fatal_reason));
      }
    }
  }
  if (external_evidence == nullptr) {
    for (std::size_t index = 0U; index < landings.size(); ++index) {
      const shared::GridCell cell{
          .x = static_cast<std::int32_t>(index % global_map->width()),
          .y = static_cast<std::int32_t>(index / global_map->width()),
      };
      if (!safe.HardFeasible(cell)) {
        landings[index].reset();
      }
    }
  }

  const std::size_t start_index = global_map->Index(start);
  if (start_index >= landings.size() || !landings[start_index].has_value()) {
    return ReachabilityProjectionResult{
        .projection = std::move(projection),
        .reason_code = {},
    };
  }
  projection.reachable[start_index] = 1U;

  const auto certify_recoverable_edge =
      [&](const std::size_t source_index,
          const std::size_t target_index) -> std::pair<bool, std::string> {
    ++projection.candidate_edges_evaluated;
    const LandingRegionHopResult forward =
        CertifyDirectedLandingRegionEdge(
            *landings[source_index], *landings[target_index], *global_map,
            *capability, input.config.map_safety, input.stop_token);
    if (!forward.ok()) {
      if (forward.status == hopper::HopCertificationStatus::kInfeasible) {
        ++projection.rejected_edges;
        return {false, {}};
      }
      return {false, EdgeFailureReason(forward)};
    }
    ++projection.certified_edges;
    ++projection.candidate_edges_evaluated;
    const LandingRegionHopResult reverse =
        CertifyDirectedLandingRegionEdge(
            *landings[target_index], *landings[source_index], *global_map,
            *capability, input.config.map_safety, input.stop_token);
    if (!reverse.ok()) {
      if (reverse.status == hopper::HopCertificationStatus::kInfeasible) {
        ++projection.rejected_edges;
        return {false, {}};
      }
      return {false, EdgeFailureReason(reverse)};
    }
    ++projection.certified_edges;
    return {true, {}};
  };

  if (direct_only) {
    if (external_evidence == nullptr) {
      return Failure("HOPPER_DIRECT_REACHABILITY_REQUIRES_LANDING_EVIDENCE");
    }
    const Vec3 source_position =
        landings[start_index]->aim_position_on_surface_m;
    for (std::size_t target_index = 0U;
         target_index < landings.size(); ++target_index) {
      if (target_index == start_index || !landings[target_index].has_value()) {
        continue;
      }
      const Vec3 target_position =
          landings[target_index]->aim_position_on_surface_m;
      const double distance_m = std::hypot(
          target_position.x - source_position.x,
          target_position.y - source_position.y);
      if (distance_m > maximum_edge_distance_m + kDistanceToleranceM) {
        continue;
      }
      const auto [certified, fatal_reason] =
          certify_recoverable_edge(start_index, target_index);
      if (!fatal_reason.empty()) {
        return Failure(fatal_reason);
      }
      if (!certified) {
        continue;
      }
      projection.reachable[target_index] = 1U;
      projection.maximum_certified_edge_distance_m = std::max(
          projection.maximum_certified_edge_distance_m, distance_m);
    }
    return ReachabilityProjectionResult{
        .projection = std::move(projection),
        .reason_code = {},
    };
  }

  std::deque<std::size_t> queue{start_index};

  const auto cell_radius = static_cast<std::int32_t>(
      std::ceil(maximum_edge_distance_m / global_map->resolution_m()));
  std::vector<std::pair<std::int32_t, std::int32_t>> offsets;
  for (std::int32_t dy = -cell_radius; dy <= cell_radius; ++dy) {
    for (std::int32_t dx = -cell_radius; dx <= cell_radius; ++dx) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const double distance = std::hypot(
          static_cast<double>(dx) * global_map->resolution_m(),
          static_cast<double>(dy) * global_map->resolution_m());
      if (distance <= maximum_edge_distance_m + kDistanceToleranceM) {
        offsets.emplace_back(dx, dy);
      }
    }
  }

  while (!queue.empty()) {
    if (input.stop_token.stop_requested()) {
      return Failure("REQUEST_CANCELED");
    }
    const std::size_t source_index = queue.front();
    queue.pop_front();
    const shared::GridCell source{
        .x = static_cast<std::int32_t>(source_index % global_map->width()),
        .y = static_cast<std::int32_t>(source_index / global_map->width()),
    };
    for (const auto [dx, dy] : offsets) {
      const shared::GridCell target{
          .x = source.x + dx,
          .y = source.y + dy,
      };
      if (!global_map->InBounds(target)) {
        continue;
      }
      const std::size_t target_index = global_map->Index(target);
      if (projection.reachable[target_index] != 0U ||
          !landings[target_index].has_value()) {
        continue;
      }
      const auto [certified, fatal_reason] =
          certify_recoverable_edge(source_index, target_index);
      if (!fatal_reason.empty()) {
        return Failure(fatal_reason);
      }
      if (!certified) {
        continue;
      }
      projection.reachable[target_index] = 1U;
      projection.maximum_certified_edge_distance_m = std::max(
          projection.maximum_certified_edge_distance_m,
          std::hypot(
              static_cast<double>(dx) * global_map->resolution_m(),
              static_cast<double>(dy) * global_map->resolution_m()));
      queue.push_back(target_index);
    }
  }
  return ReachabilityProjectionResult{
      .projection = std::move(projection),
      .reason_code = {},
  };
}

}  // namespace

namespace {

[[nodiscard]] ReachabilityProjectionResult ProjectReachabilityImpl(
    const PlannerInput& input, const double maximum_edge_distance_m,
    const HopperLandingEvidenceGrid* const hopper_landing_evidence,
    const bool direct_hopper) {
  if (!std::isfinite(maximum_edge_distance_m) ||
      maximum_edge_distance_m <= 0.0) {
    return Failure("REACHABILITY_MAXIMUM_EDGE_DISTANCE_INVALID");
  }
  if (input.stop_token.stop_requested()) {
    return Failure("REQUEST_CANCELED");
  }
  try {
    const shared::MapSnapshotBuildResult global =
        shared::MapSnapshot::Create(input.world.global_map);
    if (!global.ok()) {
      return Failure(global.reason_code);
    }
    const shared::MapSnapshotBuildResult local =
        shared::MapSnapshot::Create(input.world.local_map);
    if (!local.ok()) {
      return Failure(local.reason_code);
    }
    const shared::SafeProjectionBuildResult built =
        shared::BuildSafeProjection(
            global.snapshot, input.capability, input.config.map_safety,
            input.stop_token);
    if (!built.ok()) {
      return Failure(built.reason_code);
    }
    const PlatformType platform = CapabilityPlatform(input.capability);
    const auto start = StartCell(input, *global.snapshot, platform);
    if (!start.has_value()) {
      return Failure("REACHABILITY_START_OUTSIDE_GLOBAL_MAP");
    }
    if (platform == PlatformType::kHopper) {
      return ProjectHopper(
          input, global.snapshot, local.snapshot, *built.projection, *start,
          maximum_edge_distance_m, hopper_landing_evidence, direct_hopper);
    }
    if (direct_hopper) {
      return Failure("HOPPER_DIRECT_REACHABILITY_PLATFORM_MISMATCH");
    }
    if (hopper_landing_evidence != nullptr) {
      return Failure("HOPPER_LANDING_EVIDENCE_PLATFORM_MISMATCH");
    }
    if (StatePose(input.current_state, platform) == nullptr) {
      return Failure("REACHABILITY_PLATFORM_STATE_MISMATCH");
    }
    return ProjectGround(
        input, global.snapshot, *built.projection, *start,
        maximum_edge_distance_m);
  } catch (const std::bad_alloc&) {
    return Failure("REACHABILITY_RESOURCE_EXHAUSTED");
  } catch (...) {
    return Failure("REACHABILITY_INTERNAL_FAILURE");
  }
}

[[nodiscard]] HopperLandingEvidenceProjectionResult LandingFailure(
    std::string reason_code) {
  return HopperLandingEvidenceProjectionResult{
      .projection = std::nullopt,
      .reason_code = std::move(reason_code),
  };
}

}  // namespace

ReachabilityProjectionResult ProjectReachability(
    const PlannerInput& input, const double maximum_edge_distance_m) {
  return ProjectReachabilityImpl(
      input, maximum_edge_distance_m, nullptr, false);
}

ReachabilityProjectionResult ProjectReachability(
    const PlannerInput& input, const double maximum_edge_distance_m,
    const HopperLandingEvidenceGrid& hopper_landing_evidence) {
  return ProjectReachabilityImpl(
      input, maximum_edge_distance_m, &hopper_landing_evidence, false);
}

ReachabilityProjectionResult ProjectDirectHopperReachability(
    const PlannerInput& input, const double maximum_edge_distance_m,
    const HopperLandingEvidenceGrid& hopper_landing_evidence) {
  return ProjectReachabilityImpl(
      input, maximum_edge_distance_m, &hopper_landing_evidence, true);
}

HopperLandingEvidenceProjectionResult ProjectHopperLandingEvidence(
    const PlannerInput& input,
    const std::span<const Vec3> target_positions_map) {
  if (input.stop_token.stop_requested()) {
    return LandingFailure("REQUEST_CANCELED");
  }
  const auto* capability = std::get_if<HopperCapability>(&input.capability);
  if (capability == nullptr) {
    return LandingFailure("HOPPER_LANDING_EVIDENCE_PLATFORM_MISMATCH");
  }
  try {
    const shared::MapSnapshotBuildResult local =
        shared::MapSnapshot::Create(input.world.local_map);
    if (!local.ok()) {
      return LandingFailure(local.reason_code);
    }
    HopperLandingEvidenceProjection projection{
        .algorithm_id = std::string{kHopperLandingEvidenceAlgorithm},
        .candidates_evaluated = target_positions_map.size(),
    };
    projection.landings.reserve(target_positions_map.size());
    for (const Vec3 target : target_positions_map) {
      if (input.stop_token.stop_requested()) {
        return LandingFailure("REQUEST_CANCELED");
      }
      if (!Finite(target)) {
        return LandingFailure("HOPPER_LANDING_EVIDENCE_TARGET_INVALID");
      }
      std::string fatal_reason;
      const auto certified = CertifiedLandingAtPosition(
          target, *local.snapshot, input, *capability, fatal_reason);
      if (!fatal_reason.empty()) {
        return LandingFailure(std::move(fatal_reason));
      }
      HopperLandingEvidence evidence;
      if (certified.has_value()) {
        if (certified->boundary_m.size() != evidence.boundary_m.size()) {
          return LandingFailure("HOPPER_LANDING_EVIDENCE_BOUNDARY_INVALID");
        }
        evidence.certified = 1U;
        evidence.aim_position_on_surface_m =
            certified->aim_position_on_surface_m;
        std::copy(
            certified->boundary_m.begin(), certified->boundary_m.end(),
            evidence.boundary_m.begin());
        evidence.area_m2 = certified->area_m2;
        ++projection.certified_count;
      }
      projection.landings.push_back(std::move(evidence));
    }
    return HopperLandingEvidenceProjectionResult{
        .projection = std::move(projection),
        .reason_code = {},
    };
  } catch (const std::bad_alloc&) {
    return LandingFailure("REACHABILITY_RESOURCE_EXHAUSTED");
  } catch (...) {
    return LandingFailure("HOPPER_LANDING_EVIDENCE_INTERNAL_FAILURE");
  }
}

}  // namespace lunar::planning
