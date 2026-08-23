#include "lunar_pure_planner_core/reachability_projection.hpp"

#include <algorithm>
#include <chrono>
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
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "hierarchical/frame_transform.hpp"
#include "hierarchical/grid_search.hpp"
#include "hopper/ballistic_envelope.hpp"
#include "hopper/ballistic_kinematics.hpp"
#include "hopper/flight_tube_certifier.hpp"
#include "hopper/hop_certifier.hpp"
#include "hopper/landing_evidence.hpp"
#include "hopper/landing_region.hpp"
#include "shared/map_snapshot.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::pure_planning {

struct HopperOpportunityContextStorage final {
  std::shared_ptr<const shared::MapSnapshot> map;
  HopperCapability capability;
  MapSafetyConfig map_safety;
  std::stop_token stop_token;
  std::vector<std::optional<hopper::CertifiedLandingRegion>> landings;
  std::vector<std::pair<std::int32_t, std::int32_t>> offsets;
  std::deque<std::size_t> pending;
  std::unordered_map<std::size_t, std::uint8_t> edge_cache;
  std::size_t start_index{};
};

struct GroundEndpointReachabilityContextStorage final {
  std::shared_ptr<const shared::MapSnapshot> global_map;
  shared::SafeProjection safe;
  hierarchical::GlobalGridCostTree tree;
};

namespace {

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

struct LandingRegionHopResult final {
  hopper::HopCertificationStatus status{
      hopper::HopCertificationStatus::kInvalid};
  double nominal_flight_time_s{};
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
      return {
          .status = hopper::HopCertificationStatus::kCertified,
          .nominal_flight_time_s = center_hop.arc.flight_time_s,
      };
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

[[nodiscard]] LandingRegionHopResult CertifyCurrentToLandingRegionEdge(
    const Vec3 source_position_m,
    const hopper::CertifiedLandingRegion& target,
    const shared::MapSnapshot& flight_map,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety,
    const std::stop_token stop_token) {
  const hopper::SingleHopCertificationResult hop = hopper::CertifySingleHop(
      hopper::SingleHopCertificationProblem{
          .launch_position_m = source_position_m,
          .landing_position_m = target.aim_position_on_surface_m,
          .gravity_mps2 = capability.gravity_mps2,
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

[[nodiscard]] double PointRectangleDistance(
    const Vec3 point, const double x0, const double x1,
    const double y0, const double y1) noexcept {
  const double nearest_x = std::clamp(point.x, x0, x1);
  const double nearest_y = std::clamp(point.y, y0, y1);
  return std::hypot(point.x - nearest_x, point.y - nearest_y);
}

[[nodiscard]] bool SegmentIntersectsRectangle(
    const Vec3 begin, const Vec3 end, const double x0, const double x1,
    const double y0, const double y1) noexcept {
  double lower = 0.0;
  double upper = 1.0;
  const double dx = end.x - begin.x;
  const double dy = end.y - begin.y;
  const auto clip = [&](const double p, const double q) {
    if (std::abs(p) <= std::numeric_limits<double>::epsilon()) {
      return q >= 0.0;
    }
    const double ratio = q / p;
    if (p < 0.0) {
      lower = std::max(lower, ratio);
    } else {
      upper = std::min(upper, ratio);
    }
    return lower <= upper;
  };
  return clip(-dx, begin.x - x0) && clip(dx, x1 - begin.x) &&
      clip(-dy, begin.y - y0) && clip(dy, y1 - begin.y);
}

[[nodiscard]] double PlanarPointSegmentDistance(
    const Vec3 point, const Vec3 begin, const Vec3 end) noexcept {
  const double dx = end.x - begin.x;
  const double dy = end.y - begin.y;
  const double squared_length = dx * dx + dy * dy;
  if (squared_length <= std::numeric_limits<double>::epsilon()) {
    return std::hypot(point.x - begin.x, point.y - begin.y);
  }
  const double projection = std::clamp(
      ((point.x - begin.x) * dx + (point.y - begin.y) * dy) /
          squared_length,
      0.0, 1.0);
  return std::hypot(
      point.x - (begin.x + projection * dx),
      point.y - (begin.y + projection * dy));
}

[[nodiscard]] double SegmentRectangleDistance(
    const Vec3 begin, const Vec3 end, const double x0, const double x1,
    const double y0, const double y1) noexcept {
  if (SegmentIntersectsRectangle(begin, end, x0, x1, y0, y1)) {
    return 0.0;
  }
  double distance = std::min(
      PointRectangleDistance(begin, x0, x1, y0, y1),
      PointRectangleDistance(end, x0, x1, y0, y1));
  for (const Vec3 corner : {
           Vec3{x0, y0, 0.0}, Vec3{x1, y0, 0.0},
           Vec3{x1, y1, 0.0}, Vec3{x0, y1, 0.0}}) {
    distance = std::min(
        distance, PlanarPointSegmentDistance(corner, begin, end));
  }
  return distance;
}

[[nodiscard]] bool KnownForFlightEvidence(
    const shared::MapSnapshot& map, const std::size_t index,
    const MapSafetyConfig& config) noexcept {
  return map.ByteLayer("valid_mask")[index] != 0U &&
      static_cast<double>(map.FloatLayer("obstacle_variance")[index]) <=
          config.maximum_obstacle_variance_m2 + kDistanceToleranceM &&
      static_cast<double>(map.FloatLayer("observation_age_s")[index]) <=
          config.maximum_observation_age_s + kDistanceToleranceM &&
      static_cast<double>(map.FloatLayer("observation_quality")[index]) +
              kDistanceToleranceM >=
          config.minimum_observation_quality &&
      map.CountLayer("observation_count")[index] >=
          config.minimum_observation_count;
}

struct HopperEdgeDependencyResult final {
  std::vector<std::size_t> indices;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept { return reason_code.empty(); }
};

[[nodiscard]] HopperEdgeDependencyResult MissingFlightEvidenceTiles(
    const shared::MapSnapshot& map,
    const Vec3 source,
    const hopper::CertifiedLandingRegion& target,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety) {
  double landing_radius = 0.0;
  for (const Vec3 vertex : target.boundary_m) {
    landing_radius = std::max(
        landing_radius,
        std::hypot(
            vertex.x - target.aim_position_on_surface_m.x,
            vertex.y - target.aim_position_on_surface_m.y));
  }
  const double radius = capability.flight_collision_radius_m +
      capability.flight_map_margin_m + landing_radius;
  if (!Finite(source) || !Finite(target.aim_position_on_surface_m) ||
      !std::isfinite(radius) || radius <= 0.0) {
    return {.reason_code = "HOPPER_EDGE_DEPENDENCY_GEOMETRY_INVALID"};
  }
  const double minimum_x =
      std::min(source.x, target.aim_position_on_surface_m.x) - radius;
  const double maximum_x =
      std::max(source.x, target.aim_position_on_surface_m.x) + radius;
  const double minimum_y =
      std::min(source.y, target.aim_position_on_surface_m.y) - radius;
  const double maximum_y =
      std::max(source.y, target.aim_position_on_surface_m.y) + radius;
  const auto cell_coordinate = [&](const double value, const double origin) {
    return static_cast<long long>(
        std::floor((value - origin) / map.resolution_m()));
  };
  const long long x0 = std::max<long long>(
      0, cell_coordinate(minimum_x, map.origin_m().x));
  const long long x1 = std::min<long long>(
      static_cast<long long>(map.width()) - 1,
      cell_coordinate(maximum_x, map.origin_m().x));
  const long long y0 = std::max<long long>(
      0, cell_coordinate(minimum_y, map.origin_m().y));
  const long long y1 = std::min<long long>(
      static_cast<long long>(map.height()) - 1,
      cell_coordinate(maximum_y, map.origin_m().y));
  HopperEdgeDependencyResult result;
  if (x0 > x1 || y0 > y1) {
    return result;
  }
  for (long long y = y0; y <= y1; ++y) {
    for (long long x = x0; x <= x1; ++x) {
      const double cell_x0 = map.origin_m().x +
          static_cast<double>(x) * map.resolution_m();
      const double cell_y0 = map.origin_m().y +
          static_cast<double>(y) * map.resolution_m();
      if (SegmentRectangleDistance(
              source, target.aim_position_on_surface_m,
              cell_x0, cell_x0 + map.resolution_m(),
              cell_y0, cell_y0 + map.resolution_m()) >
          radius + kDistanceToleranceM) {
        continue;
      }
      const shared::GridCell cell{
          .x = static_cast<std::int32_t>(x),
          .y = static_cast<std::int32_t>(y),
      };
      const std::size_t index = map.Index(cell);
      if (!KnownForFlightEvidence(map, index, map_safety)) {
        result.indices.push_back(index);
      }
    }
  }
  return result;
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
          .gravity_mps2 = capability.gravity_mps2,
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
  const double infinity = std::numeric_limits<double>::infinity();
  const std::size_t no_parent = std::numeric_limits<std::size_t>::max();
  ReachabilityProjection projection{
      .platform_type = safe.platform_type(),
      .width = global_map->width(),
      .height = global_map->height(),
      .reachable = std::vector<std::uint8_t>(global_map->cell_count(), 0U),
      .algorithm_id = "cpp-ground-global-cost-tree/v1",
      .maximum_edge_distance_m = maximum_edge_distance_m,
      .minimum_cost =
          std::vector<double>(global_map->cell_count(), infinity),
      .parent_index =
          std::vector<std::size_t>(global_map->cell_count(), no_parent),
      .start_index = global_map->Index(start),
      .search_elapsed_s = 0.0,
  };
  if (!safe.HardFeasible(start)) {
    return ReachabilityProjectionResult{
        .projection = std::move(projection),
        .reason_code = {},
    };
  }
  const auto started = std::chrono::steady_clock::now();
  hierarchical::GlobalGridCostTreeResult tree =
      hierarchical::SearchGlobalGridCostTree(
          hierarchical::GlobalGridSearchProblem{
              .projection = safe,
              .start = start,
              .goal_mask = {},
              .excluded_mask = {},
              .maximum_speed_mps = 1.0,
              .config = input.config.global_search,
              .stop_token = input.stop_token,
          });
  projection.search_elapsed_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  if (!tree.ok()) {
    return Failure(tree.reason_code.empty()
                       ? "GLOBAL_SEARCH_RESULT_INVALID"
                       : std::move(tree.reason_code));
  }
  if (tree.tree->minimum_cost.size() != global_map->cell_count() ||
      tree.tree->parent_index.size() != global_map->cell_count() ||
      tree.tree->start_index != projection.start_index ||
      !std::isfinite(projection.search_elapsed_s) ||
      projection.search_elapsed_s < 0.0) {
    return Failure("GLOBAL_SEARCH_RESULT_INVALID");
  }
  for (std::size_t index = 0U; index < global_map->cell_count(); ++index) {
    const double cost = tree.tree->minimum_cost[index];
    const std::size_t parent = tree.tree->parent_index[index];
    if (!std::isfinite(cost)) {
      if (!std::isinf(cost) || cost < 0.0 || parent != no_parent) {
        return Failure("GLOBAL_SEARCH_RESULT_INVALID");
      }
      continue;
    }
    if (cost < 0.0 || parent >= global_map->cell_count()) {
      return Failure("GLOBAL_SEARCH_RESULT_INVALID");
    }
    if (index == projection.start_index) {
      if (cost != 0.0 || parent != index) {
        return Failure("GLOBAL_SEARCH_RESULT_INVALID");
      }
    } else if (!std::isfinite(tree.tree->minimum_cost[parent]) ||
               !(tree.tree->minimum_cost[parent] < cost)) {
      return Failure("GLOBAL_SEARCH_RESULT_INVALID");
    }
    projection.reachable[index] = 1U;
  }
  projection.minimum_cost = std::move(tree.tree->minimum_cost);
  projection.parent_index = std::move(tree.tree->parent_index);
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
    hopper::ExternalLandingBuildResult built =
        hopper::BuildExternalLandings(*global_map, *external_evidence);
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

GroundEndpointReachabilityContextResult
ProjectGroundEndpointReachabilityContext(
    const PlannerInput& input, const double maximum_edge_distance_m) {
  const auto failure = [](std::string reason_code) {
    return GroundEndpointReachabilityContextResult{
        .context = std::nullopt,
        .reason_code = std::move(reason_code),
    };
  };
  if (!std::isfinite(maximum_edge_distance_m) ||
      maximum_edge_distance_m <= 0.0) {
    return failure("REACHABILITY_MAXIMUM_EDGE_DISTANCE_INVALID");
  }
  if (input.stop_token.stop_requested()) {
    return failure("REQUEST_CANCELED");
  }
  try {
    const PlatformType platform = CapabilityPlatform(input.capability);
    if (platform == PlatformType::kHopper) {
      return failure("GROUND_ENDPOINT_CONTEXT_PLATFORM_MISMATCH");
    }
    if (StatePose(input.current_state, platform) == nullptr) {
      return failure("REACHABILITY_PLATFORM_STATE_MISMATCH");
    }
    const shared::MapSnapshotBuildResult global =
        shared::MapSnapshot::Create(input.world.global_map);
    if (!global.ok()) {
      return failure(global.reason_code);
    }
    const auto start = StartCell(input, *global.snapshot, platform);
    if (!start.has_value()) {
      return failure("REACHABILITY_START_OUTSIDE_GLOBAL_MAP");
    }
    const shared::SafeProjectionBuildResult built =
        shared::BuildSafeProjection(
            global.snapshot, input.capability, input.config.map_safety,
            input.stop_token);
    if (!built.ok()) {
      return failure(built.reason_code);
    }
    const double infinity = std::numeric_limits<double>::infinity();
    const std::size_t no_parent = std::numeric_limits<std::size_t>::max();
    const std::size_t start_index = global.snapshot->Index(*start);
    GroundEndpointReachabilityContextStorage storage{
        .global_map = global.snapshot,
        .safe = std::move(*built.projection),
        .tree = hierarchical::GlobalGridCostTree{
            .minimum_cost = std::vector<double>(
                global.snapshot->cell_count(), infinity),
            .parent_index = std::vector<std::size_t>(
                global.snapshot->cell_count(), no_parent),
            .start_index = start_index,
            .expanded_states = 0U,
        },
    };
    double search_elapsed_s = 0.0;
    if (storage.safe.HardFeasible(*start)) {
      const auto started = std::chrono::steady_clock::now();
      hierarchical::GlobalGridCostTreeResult tree =
          hierarchical::SearchGlobalGridCostTree(
              hierarchical::GlobalGridSearchProblem{
                  .projection = storage.safe,
                  .start = *start,
                  .goal_mask = {},
                  .excluded_mask = {},
                  .maximum_speed_mps = 1.0,
                  .config = input.config.global_search,
                  .stop_token = input.stop_token,
              });
      search_elapsed_s = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count();
      if (!tree.ok()) {
        return failure(tree.reason_code.empty()
                           ? "GLOBAL_SEARCH_RESULT_INVALID"
                           : std::move(tree.reason_code));
      }
      storage.tree = std::move(*tree.tree);
    }
    if (storage.tree.minimum_cost.size() != global.snapshot->cell_count() ||
        storage.tree.parent_index.size() != global.snapshot->cell_count() ||
        storage.tree.start_index != start_index ||
        !std::isfinite(search_elapsed_s) || search_elapsed_s < 0.0) {
      return failure("GLOBAL_SEARCH_RESULT_INVALID");
    }
    ReachabilityProjection projection{
        .platform_type = platform,
        .width = global.snapshot->width(),
        .height = global.snapshot->height(),
        .reachable = std::vector<std::uint8_t>(
            global.snapshot->cell_count(), 0U),
        .algorithm_id = "cpp-ground-global-cost-tree/v1",
        .maximum_edge_distance_m = maximum_edge_distance_m,
        .minimum_cost = storage.tree.minimum_cost,
        .parent_index = storage.tree.parent_index,
        .start_index = start_index,
        .search_elapsed_s = search_elapsed_s,
    };
    for (std::size_t index = 0U; index < projection.minimum_cost.size();
         ++index) {
      const double cost = projection.minimum_cost[index];
      const std::size_t parent = projection.parent_index[index];
      if (!std::isfinite(cost)) {
        if (!std::isinf(cost) || cost < 0.0 || parent != no_parent) {
          return failure("GLOBAL_SEARCH_RESULT_INVALID");
        }
        continue;
      }
      if (cost < 0.0 || parent >= projection.minimum_cost.size()) {
        return failure("GLOBAL_SEARCH_RESULT_INVALID");
      }
      if (index == start_index) {
        if (cost != 0.0 || parent != index) {
          return failure("GLOBAL_SEARCH_RESULT_INVALID");
        }
      } else if (!std::isfinite(projection.minimum_cost[parent]) ||
                 !(projection.minimum_cost[parent] < cost)) {
        return failure("GLOBAL_SEARCH_RESULT_INVALID");
      }
      projection.reachable[index] = 1U;
    }
    return GroundEndpointReachabilityContextResult{
        .context = GroundEndpointReachabilityContext{
            .projection = std::move(projection),
            .storage = std::make_shared<GroundEndpointReachabilityContextStorage>(
                std::move(storage)),
        },
        .reason_code = {},
    };
  } catch (const std::bad_alloc&) {
    return failure("REACHABILITY_RESOURCE_EXHAUSTED");
  } catch (...) {
    return failure("REACHABILITY_INTERNAL_FAILURE");
  }
}

GroundExactEndpointProjectionResult QueryGroundExactEndpoints(
    const GroundEndpointReachabilityContext& context,
    const std::span<const Vec3> target_positions_map,
    const double tolerance_m) {
  const auto failure = [](std::string reason_code) {
    return GroundExactEndpointProjectionResult{
        .projection = std::nullopt,
        .reason_code = std::move(reason_code),
    };
  };
  if (!std::isfinite(tolerance_m) || tolerance_m < 0.0) {
    return failure("GROUND_ENDPOINT_TOLERANCE_INVALID");
  }
  if (context.storage == nullptr ||
      context.storage->global_map == nullptr ||
      context.projection.platform_type == PlatformType::kHopper) {
    return failure("GROUND_ENDPOINT_CONTEXT_INVALID");
  }
  try {
    const auto& storage = *context.storage;
    const auto& map = *storage.global_map;
    const std::size_t cell_count = map.cell_count();
    if (context.projection.width != map.width() ||
        context.projection.height != map.height() ||
        context.projection.minimum_cost.size() != cell_count ||
        context.projection.parent_index.size() != cell_count ||
        context.projection.reachable.size() != cell_count ||
        storage.tree.minimum_cost.size() != cell_count ||
        storage.tree.parent_index.size() != cell_count) {
      return failure("GROUND_ENDPOINT_CONTEXT_INVALID");
    }
    const double infinity = std::numeric_limits<double>::infinity();
    GroundExactEndpointProjection projection{
        .reachable = std::vector<std::uint8_t>(
            target_positions_map.size(), 0U),
        .minimum_cost_m = std::vector<double>(
            target_positions_map.size(), infinity),
        .reason_codes = std::vector<std::string>(
            target_positions_map.size(), "GROUND_ENDPOINT_UNREACHABLE"),
    };
    const double resolution_m = map.resolution_m();
    const Vec3 origin_m = map.origin_m();
    const std::int32_t radius_cells = static_cast<std::int32_t>(
        std::ceil(tolerance_m / resolution_m)) + 1;
    for (std::size_t target_index = 0U;
         target_index < target_positions_map.size(); ++target_index) {
      const Vec3 target = target_positions_map[target_index];
      if (!std::isfinite(target.x) || !std::isfinite(target.y) ||
          !std::isfinite(target.z)) {
        projection.reason_codes[target_index] =
            "GROUND_ENDPOINT_POSITION_INVALID";
        continue;
      }
      const auto center = map.PositionToCell({.x = target.x, .y = target.y});
      if (!center.has_value()) {
        projection.reason_codes[target_index] =
            "GROUND_ENDPOINT_OUTSIDE_GLOBAL_MAP";
        continue;
      }
      bool any_hard_feasible = false;
      bool found_reachable = false;
      double best_cost = infinity;
      std::size_t best_cell_index = cell_count;
      const double tolerance_squared = tolerance_m * tolerance_m;
      for (std::int32_t y = center->y - radius_cells;
           y <= center->y + radius_cells; ++y) {
        for (std::int32_t x = center->x - radius_cells;
             x <= center->x + radius_cells; ++x) {
          const shared::GridCell cell{.x = x, .y = y};
          if (!map.InBounds(cell)) {
            continue;
          }
          const double lower_x = origin_m.x +
              static_cast<double>(cell.x) * resolution_m;
          const double upper_x = lower_x + resolution_m;
          const double lower_y = origin_m.y +
              static_cast<double>(cell.y) * resolution_m;
          const double upper_y = lower_y + resolution_m;
          const double dx = std::max(
              {lower_x - target.x, 0.0, target.x - upper_x});
          const double dy = std::max(
              {lower_y - target.y, 0.0, target.y - upper_y});
          if (dx * dx + dy * dy > tolerance_squared) {
            continue;
          }
          if (!storage.safe.HardFeasible(cell)) {
            continue;
          }
          any_hard_feasible = true;
          const std::size_t cell_index = map.Index(cell);
          const double cost = storage.tree.minimum_cost[cell_index];
          if (!std::isfinite(cost)) {
            continue;
          }
          if (!found_reachable || cost < best_cost ||
              (cost == best_cost && cell_index < best_cell_index)) {
            found_reachable = true;
            best_cost = cost;
            best_cell_index = cell_index;
          }
        }
      }
      if (!any_hard_feasible) {
        projection.reason_codes[target_index] =
            "GROUND_ENDPOINT_NOT_HARD_FEASIBLE";
        continue;
      }
      if (!found_reachable) {
        projection.reason_codes[target_index] =
            "GROUND_ENDPOINT_UNREACHABLE";
        continue;
      }
      projection.reachable[target_index] = 1U;
      projection.minimum_cost_m[target_index] = best_cost;
      projection.reason_codes[target_index] = "GROUND_ENDPOINT_REACHABLE";
    }
    return GroundExactEndpointProjectionResult{
        .projection = std::move(projection),
        .reason_code = {},
    };
  } catch (const std::bad_alloc&) {
    return failure("REACHABILITY_RESOURCE_EXHAUSTED");
  } catch (...) {
    return failure("REACHABILITY_INTERNAL_FAILURE");
  }
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

HopperOpportunityContextResult
ProjectHopperOpportunityContext(
    const PlannerInput& input, const double maximum_edge_distance_m,
    const HopperLandingEvidenceGrid& hopper_landing_evidence) {
  const auto failure = [](std::string reason_code) {
    return HopperOpportunityContextResult{
        .context = std::nullopt,
        .reason_code = std::move(reason_code),
    };
  };
  if (!std::isfinite(maximum_edge_distance_m) ||
      maximum_edge_distance_m <= 0.0) {
    return failure("REACHABILITY_MAXIMUM_EDGE_DISTANCE_INVALID");
  }
  if (input.stop_token.stop_requested()) {
    return failure("REQUEST_CANCELED");
  }
  const auto* capability = std::get_if<HopperCapability>(&input.capability);
  if (capability == nullptr) {
    return failure("HOPPER_OPPORTUNITY_DISTANCE_PLATFORM_MISMATCH");
  }
  try {
    const auto global = shared::MapSnapshot::Create(input.world.global_map);
    if (!global.ok()) {
      return failure(global.reason_code);
    }
    const auto start = StartCell(
        input, *global.snapshot, PlatformType::kHopper);
    if (!start.has_value()) {
      return failure("REACHABILITY_START_OUTSIDE_GLOBAL_MAP");
    }
    auto built = hopper::BuildExternalLandings(
        *global.snapshot, hopper_landing_evidence);
    if (!built.landings.has_value()) {
      return failure(std::move(built.reason_code));
    }
    HopperOpportunityContext context{
        .width = global.snapshot->width(),
        .height = global.snapshot->height(),
        .direct =
            std::vector<std::uint8_t>(global.snapshot->cell_count(), 0U),
        .reachable =
            std::vector<std::uint8_t>(global.snapshot->cell_count(), 0U),
        .hop_distance_from_current =
            std::vector<std::int32_t>(global.snapshot->cell_count(), -1),
        .algorithm_id = "cpp-hopper-opportunity-connectivity/v1",
        .storage = std::make_shared<HopperOpportunityContextStorage>(),
    };
    const std::size_t start_index = global.snapshot->Index(*start);
    if (start_index >= built.landings->size() ||
        !(*built.landings)[start_index].has_value()) {
      return HopperOpportunityContextResult{
          .context = std::move(context), .reason_code = {}};
    }
    context.storage->map = global.snapshot;
    context.storage->capability = *capability;
    context.storage->map_safety = input.config.map_safety;
    context.storage->stop_token = input.stop_token;
    context.storage->landings = std::move(*built.landings);
    context.storage->start_index = start_index;
    context.reachable[start_index] = 1U;
    context.hop_distance_from_current[start_index] = 0;
    const auto cell_radius = static_cast<std::int32_t>(
        std::ceil(maximum_edge_distance_m / global.snapshot->resolution_m()));
    for (std::int32_t dy = -cell_radius; dy <= cell_radius; ++dy) {
      for (std::int32_t dx = -cell_radius; dx <= cell_radius; ++dx) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        const double distance = std::hypot(
            static_cast<double>(dx) * global.snapshot->resolution_m(),
            static_cast<double>(dy) * global.snapshot->resolution_m());
        if (distance <= maximum_edge_distance_m + kDistanceToleranceM) {
          context.storage->offsets.emplace_back(dx, dy);
        }
      }
    }
    const auto certify = [&](const std::size_t target_index) {
      ++context.candidate_edges_evaluated;
      const auto forward = CertifyDirectedLandingRegionEdge(
          *context.storage->landings[start_index],
          *context.storage->landings[target_index], *global.snapshot,
          *capability, input.config.map_safety, input.stop_token);
      if (!forward.ok()) {
        if (forward.status == hopper::HopCertificationStatus::kInfeasible) {
          ++context.rejected_edges;
          return std::pair{false, std::string{}};
        }
        return std::pair{false, EdgeFailureReason(forward)};
      }
      ++context.certified_edges;
      ++context.candidate_edges_evaluated;
      const auto reverse = CertifyDirectedLandingRegionEdge(
          *context.storage->landings[target_index],
          *context.storage->landings[start_index], *global.snapshot,
          *capability, input.config.map_safety, input.stop_token);
      if (!reverse.ok()) {
        if (reverse.status == hopper::HopCertificationStatus::kInfeasible) {
          ++context.rejected_edges;
          return std::pair{false, std::string{}};
        }
        return std::pair{false, EdgeFailureReason(reverse)};
      }
      ++context.certified_edges;
      return std::pair{true, std::string{}};
    };
    const shared::GridCell start_cell = *start;
    const std::size_t cell_count = global.snapshot->cell_count();
    for (const auto [dx, dy] : context.storage->offsets) {
      const shared::GridCell target{
          .x = start_cell.x + dx, .y = start_cell.y + dy};
      if (!global.snapshot->InBounds(target)) {
        continue;
      }
      const std::size_t target_index = global.snapshot->Index(target);
      if (!context.storage->landings[target_index].has_value()) {
        continue;
      }
      const auto [certified, fatal_reason] = certify(target_index);
      if (!fatal_reason.empty()) {
        return failure(fatal_reason);
      }
      const std::size_t edge_key = std::min(start_index, target_index) *
              cell_count + std::max(start_index, target_index);
      context.storage->edge_cache.emplace(
          edge_key, certified ? 1U : 0U);
      if (certified) {
        context.direct[target_index] = 1U;
        context.reachable[target_index] = 1U;
        context.hop_distance_from_current[target_index] = 1;
        context.storage->pending.push_back(target_index);
      }
    }
    return HopperOpportunityContextResult{
        .context = std::move(context), .reason_code = {}};
  } catch (const std::bad_alloc&) {
    return failure("REACHABILITY_RESOURCE_EXHAUSTED");
  } catch (...) {
    return failure("REACHABILITY_INTERNAL_FAILURE");
  }
}

HopperOpportunityDistanceProjectionResult QueryHopperOpportunityDistance(
    HopperOpportunityContext& context,
    const std::span<const std::uint8_t> positive_opportunities,
    const bool enumerate_all_reachable_opportunities) {
  const auto failure = [](std::string reason_code) {
    return HopperOpportunityDistanceProjectionResult{
        .projection = std::nullopt,
        .reason_code = std::move(reason_code),
    };
  };
  if (context.storage == nullptr || context.storage->map == nullptr ||
      positive_opportunities.size() != context.reachable.size() ||
      std::any_of(positive_opportunities.begin(),
                  positive_opportunities.end(),
                  [](const std::uint8_t value) { return value > 1U; })) {
    return failure("HOPPER_OPPORTUNITY_MASK_INVALID");
  }
  try {
    auto& storage = *context.storage;
    const std::size_t cell_count = context.reachable.size();
    std::size_t undiscovered_positive = 0U;
    std::int32_t nearest_discovered_positive =
        std::numeric_limits<std::int32_t>::max();
    for (std::size_t index = 0U; index < cell_count; ++index) {
      if (positive_opportunities[index] != 0U) {
        if (context.reachable[index] == 0U) {
          ++undiscovered_positive;
        } else {
          nearest_discovered_positive = std::min(
              nearest_discovered_positive,
              context.hop_distance_from_current[index]);
        }
      }
    }
    const auto certify = [&](const std::size_t source_index,
                             const std::size_t target_index) {
      const std::size_t edge_key = std::min(source_index, target_index) *
              cell_count + std::max(source_index, target_index);
      if (const auto cached = storage.edge_cache.find(edge_key);
          cached != storage.edge_cache.end()) {
        return std::pair{cached->second != 0U, std::string{}};
      }
      ++context.candidate_edges_evaluated;
      const auto forward = CertifyDirectedLandingRegionEdge(
          *storage.landings[source_index], *storage.landings[target_index],
          *storage.map, storage.capability, storage.map_safety,
          storage.stop_token);
      if (!forward.ok()) {
        if (forward.status == hopper::HopCertificationStatus::kInfeasible) {
          ++context.rejected_edges;
          storage.edge_cache.emplace(edge_key, 0U);
          return std::pair{false, std::string{}};
        }
        return std::pair{false, EdgeFailureReason(forward)};
      }
      ++context.certified_edges;
      ++context.candidate_edges_evaluated;
      const auto reverse = CertifyDirectedLandingRegionEdge(
          *storage.landings[target_index], *storage.landings[source_index],
          *storage.map, storage.capability, storage.map_safety,
          storage.stop_token);
      if (!reverse.ok()) {
        if (reverse.status == hopper::HopCertificationStatus::kInfeasible) {
          ++context.rejected_edges;
          storage.edge_cache.emplace(edge_key, 0U);
          return std::pair{false, std::string{}};
        }
        return std::pair{false, EdgeFailureReason(reverse)};
      }
      ++context.certified_edges;
      storage.edge_cache.emplace(edge_key, 1U);
      return std::pair{true, std::string{}};
    };
    std::int32_t completed_positive_parent_level =
        !enumerate_all_reachable_opportunities &&
                nearest_discovered_positive !=
                    std::numeric_limits<std::int32_t>::max()
            ? nearest_discovered_positive - 1
            : -1;
    while (!storage.pending.empty() &&
           (completed_positive_parent_level < 0 ||
            context.hop_distance_from_current[storage.pending.front()] <=
                completed_positive_parent_level ||
            (enumerate_all_reachable_opportunities &&
             undiscovered_positive > 0U))) {
      if (storage.stop_token.stop_requested()) {
        return failure("REQUEST_CANCELED");
      }
      const std::size_t source_index = storage.pending.front();
      storage.pending.pop_front();
      const shared::GridCell source{
          .x = static_cast<std::int32_t>(source_index % context.width),
          .y = static_cast<std::int32_t>(source_index / context.width),
      };
      for (const auto [dx, dy] : storage.offsets) {
        const shared::GridCell target{
            .x = source.x + dx, .y = source.y + dy};
        if (!storage.map->InBounds(target)) {
          continue;
        }
        const std::size_t target_index = storage.map->Index(target);
        if (context.hop_distance_from_current[target_index] >= 0 ||
            !storage.landings[target_index].has_value()) {
          continue;
        }
        const auto [certified, fatal_reason] =
            certify(source_index, target_index);
        if (!fatal_reason.empty()) {
          return failure(fatal_reason);
        }
        if (certified) {
          context.reachable[target_index] = 1U;
          context.hop_distance_from_current[target_index] =
              context.hop_distance_from_current[source_index] + 1;
          storage.pending.push_back(target_index);
          if (positive_opportunities[target_index] != 0U) {
            --undiscovered_positive;
            const std::int32_t parent_level =
                context.hop_distance_from_current[target_index] - 1;
            completed_positive_parent_level =
                completed_positive_parent_level < 0
                ? parent_level
                : enumerate_all_reachable_opportunities
                ? std::max(completed_positive_parent_level, parent_level)
                : completed_positive_parent_level;
          }
        }
      }
    }
    HopperOpportunityDistanceProjection projection{
        .width = context.width,
        .height = context.height,
        .direct_progress = std::vector<std::uint8_t>(cell_count, 0U),
        .reachable_opportunities =
            std::vector<std::uint8_t>(cell_count, 0U),
        .direct_progress_total_cost = std::vector<double>(
            cell_count, std::numeric_limits<double>::infinity()),
        .represented_opportunity_index = std::vector<std::size_t>(
            cell_count, std::numeric_limits<std::size_t>::max()),
        .algorithm_id = "cpp-hopper-opportunity-distance/v2",
    };
    std::int32_t nearest = std::numeric_limits<std::int32_t>::max();
    for (std::size_t index = 0U; index < cell_count; ++index) {
      if (context.reachable[index] == 0U ||
          positive_opportunities[index] == 0U) {
        continue;
      }
      projection.reachable_opportunities[index] = 1U;
      nearest = std::min(
          nearest, context.hop_distance_from_current[index]);
    }
    if (nearest == std::numeric_limits<std::int32_t>::max()) {
      return HopperOpportunityDistanceProjectionResult{
          .projection = std::move(projection), .reason_code = {}};
    }
    projection.current_hop_distance = nearest;
    projection.has_reachable_opportunity = 1U;
    if (nearest == 0) {
      return HopperOpportunityDistanceProjectionResult{
          .projection = std::move(projection), .reason_code = {}};
    }
    const double infinity = std::numeric_limits<double>::infinity();
    const std::size_t no_target = std::numeric_limits<std::size_t>::max();
    const auto edge_cost = [&](const std::size_t source_index,
                               const std::size_t target_index)
        -> std::optional<double> {
      if (source_index >= storage.landings.size() ||
          target_index >= storage.landings.size() ||
          !storage.landings[source_index].has_value() ||
          !storage.landings[target_index].has_value()) {
        return std::nullopt;
      }
      const Vec3 source =
          storage.landings[source_index]->aim_position_on_surface_m;
      const Vec3 target =
          storage.landings[target_index]->aim_position_on_surface_m;
      const double value = std::hypot(
          source.x - target.x, source.y - target.y, source.z - target.z);
      if (!std::isfinite(value) || value <= 0.0) {
        return std::nullopt;
      }
      return value;
    };
    std::vector<std::uint8_t> shortest_path(cell_count, 0U);
    std::vector<double> remaining_cost(cell_count, infinity);
    std::vector<std::size_t> represented_target(cell_count, no_target);
    for (std::size_t index = 0U; index < cell_count; ++index) {
      if (positive_opportunities[index] != 0U &&
          context.hop_distance_from_current[index] == nearest) {
        shortest_path[index] = 1U;
        remaining_cost[index] = 0.0;
        represented_target[index] = index;
      }
    }
    for (std::int32_t level = nearest; level > 1; --level) {
      for (std::size_t target_index = 0U; target_index < cell_count;
           ++target_index) {
        if (shortest_path[target_index] == 0U ||
            context.hop_distance_from_current[target_index] != level) {
          continue;
        }
        const shared::GridCell target{
            .x = static_cast<std::int32_t>(target_index % context.width),
            .y = static_cast<std::int32_t>(target_index / context.width),
        };
        for (const auto [dx, dy] : storage.offsets) {
          const shared::GridCell source{
              .x = target.x + dx, .y = target.y + dy};
          if (!storage.map->InBounds(source)) {
            continue;
          }
          const std::size_t source_index = storage.map->Index(source);
          if (context.hop_distance_from_current[source_index] != level - 1 ||
              !storage.landings[source_index].has_value()) {
            continue;
          }
          const auto [certified, fatal_reason] =
              certify(source_index, target_index);
          if (!fatal_reason.empty()) {
            return failure(fatal_reason);
          }
          if (certified) {
            shortest_path[source_index] = 1U;
            const auto cost = edge_cost(source_index, target_index);
            if (!cost.has_value() ||
                !std::isfinite(remaining_cost[target_index]) ||
                represented_target[target_index] == no_target) {
              return failure("HOPPER_OPPORTUNITY_COST_INVALID");
            }
            const double candidate_cost =
                *cost + remaining_cost[target_index];
            if (!std::isfinite(candidate_cost)) {
              return failure("HOPPER_OPPORTUNITY_COST_INVALID");
            }
            if (candidate_cost < remaining_cost[source_index] ||
                (candidate_cost == remaining_cost[source_index] &&
                 represented_target[target_index] <
                     represented_target[source_index])) {
              remaining_cost[source_index] = candidate_cost;
              represented_target[source_index] =
                  represented_target[target_index];
            }
          }
        }
      }
    }
    for (std::size_t index = 0U; index < cell_count; ++index) {
      if (context.direct[index] == 0U || shortest_path[index] == 0U) {
        continue;
      }
      const auto direct_cost = edge_cost(storage.start_index, index);
      if (!direct_cost.has_value() || !std::isfinite(remaining_cost[index]) ||
          represented_target[index] == no_target) {
        return failure("HOPPER_OPPORTUNITY_COST_INVALID");
      }
      const double total_cost = *direct_cost + remaining_cost[index];
      if (!std::isfinite(total_cost) || total_cost <= 0.0) {
        return failure("HOPPER_OPPORTUNITY_COST_INVALID");
      }
      projection.direct_progress[index] = 1U;
      projection.direct_progress_total_cost[index] = total_cost;
      projection.represented_opportunity_index[index] =
          represented_target[index];
    }
    return HopperOpportunityDistanceProjectionResult{
        .projection = std::move(projection), .reason_code = {}};
  } catch (const std::bad_alloc&) {
    return failure("REACHABILITY_RESOURCE_EXHAUSTED");
  } catch (...) {
    return failure("REACHABILITY_INTERNAL_FAILURE");
  }
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

HopperIncrementalEdgeProjectionResult ProjectHopperIncrementalEdges(
    const PlannerInput& input,
    const HopperLandingEvidenceGrid& hopper_landing_evidence,
    const std::span<const std::uint8_t> task_target_mask) {
  const auto failure = [](std::string reason_code) {
    return HopperIncrementalEdgeProjectionResult{
        .projection = std::nullopt,
        .reason_code = std::move(reason_code),
    };
  };
  if (input.stop_token.stop_requested()) {
    return failure("REQUEST_CANCELED");
  }
  const auto* capability = std::get_if<HopperCapability>(&input.capability);
  const auto* state = std::get_if<HopperState>(&input.current_state);
  if (capability == nullptr || state == nullptr) {
    return failure("HOPPER_INCREMENTAL_EDGE_PLATFORM_MISMATCH");
  }
  try {
    const auto global = shared::MapSnapshot::Create(input.world.global_map);
    if (!global.ok()) {
      return failure(global.reason_code);
    }
    if (task_target_mask.size() != global.snapshot->cell_count() ||
        std::any_of(
            task_target_mask.begin(), task_target_mask.end(),
            [](const std::uint8_t value) { return value > 1U; })) {
      return failure("HOPPER_INCREMENTAL_EDGE_TASK_MASK_INVALID");
    }
    hopper::ExternalLandingBuildResult built =
        hopper::BuildExternalLandings(
            *global.snapshot, hopper_landing_evidence);
    if (!built.ok()) {
      return failure(std::move(built.reason_code));
    }
    const auto pose_map = hierarchical::TransformPose(
        state->pose, input.world.map_from_odom,
        hierarchical::TransformDirection::kChildToParent);
    if (!pose_map.has_value() || !Finite(pose_map->position_m)) {
      return failure("FRAME_TRANSFORM_INVALID");
    }
    HopperIncrementalEdgeProjection projection{
        .width = global.snapshot->width(),
        .height = global.snapshot->height(),
        .algorithm_id = "cpp-hopper-incremental-edge-evidence/v1",
    };
    const double same_pose_tolerance = std::max(
        kDistanceToleranceM,
        global.snapshot->resolution_m() *
            std::sqrt(std::numeric_limits<double>::epsilon()));
    for (std::size_t index = 0U; index < task_target_mask.size(); ++index) {
      if (input.stop_token.stop_requested()) {
        return failure("REQUEST_CANCELED");
      }
      if (task_target_mask[index] == 0U ||
          !(*built.landings)[index].has_value()) {
        continue;
      }
      const auto& target = *(*built.landings)[index];
      const double displacement = std::hypot(
          std::hypot(
              target.aim_position_on_surface_m.x - pose_map->position_m.x,
              target.aim_position_on_surface_m.y - pose_map->position_m.y),
          target.aim_position_on_surface_m.z - pose_map->position_m.z);
      if (!std::isfinite(displacement)) {
        return failure("HOPPER_LANDING_EVIDENCE_VALUE_INVALID");
      }
      if (displacement <= same_pose_tolerance) {
        continue;
      }
      HopperIncrementalEdgeDiagnostic diagnostic{
          .target_index = index,
          .exact_target_position_m = target.aim_position_on_surface_m,
      };
      const auto envelope = hopper::EvaluateMinimumSingleHopEnvelope(
          pose_map->position_m, target.aim_position_on_surface_m,
          capability->gravity_mps2, global.snapshot->resolution_m(), *capability);
      if (!envelope.ok()) {
        if (envelope.reason_code ==
            "HOPPER_SINGLE_HOP_ENVELOPE_EXCEEDED") {
          diagnostic.disposition =
              HopperEdgeEvidenceDisposition::kStablePhysicalRejection;
          diagnostic.reason_code = envelope.reason_code;
          projection.edges.push_back(std::move(diagnostic));
          continue;
        }
        return failure(
            envelope.reason_code.empty()
                ? "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"
                : envelope.reason_code);
      }
      diagnostic.nominal_flight_time_s =
          envelope.evidence->arc.flight_time_s;
      const HopperEdgeDependencyResult dependencies =
          MissingFlightEvidenceTiles(
              *global.snapshot, pose_map->position_m, target, *capability,
              input.config.map_safety);
      if (!dependencies.ok()) {
        return failure(dependencies.reason_code);
      }
      if (!dependencies.indices.empty()) {
        diagnostic.disposition =
            HopperEdgeEvidenceDisposition::kWaitingEvidence;
        diagnostic.dependency_tile_indices = dependencies.indices;
        diagnostic.reason_code = "HOPPER_FLIGHT_TUBE_UNKNOWN";
        projection.edges.push_back(std::move(diagnostic));
        continue;
      }
      const LandingRegionHopResult edge =
          CertifyCurrentToLandingRegionEdge(
              pose_map->position_m, target, *global.snapshot, *capability,
              input.config.map_safety, input.stop_token);
      if (edge.ok()) {
        if (!std::isfinite(edge.nominal_flight_time_s) ||
            edge.nominal_flight_time_s <= 0.0) {
          return failure("HOPPER_INCREMENTAL_EDGE_RESULT_INVALID");
        }
        diagnostic.disposition =
            HopperEdgeEvidenceDisposition::kCertified;
        diagnostic.nominal_flight_time_s = edge.nominal_flight_time_s;
        projection.edges.push_back(std::move(diagnostic));
        continue;
      }
      if (edge.status == hopper::HopCertificationStatus::kInfeasible) {
        diagnostic.disposition =
            HopperEdgeEvidenceDisposition::kStablePhysicalRejection;
        diagnostic.nominal_flight_time_s = 0.0;
        diagnostic.reason_code = edge.reason_code.empty()
            ? "HOPPER_EDGE_PHYSICALLY_INFEASIBLE"
            : edge.reason_code;
        projection.edges.push_back(std::move(diagnostic));
        continue;
      }
      return failure(EdgeFailureReason(edge));
    }
    return HopperIncrementalEdgeProjectionResult{
        .projection = std::move(projection),
        .reason_code = {},
    };
  } catch (const std::bad_alloc&) {
    return failure("REACHABILITY_RESOURCE_EXHAUSTED");
  } catch (...) {
    return failure("REACHABILITY_INTERNAL_FAILURE");
  }
}

HopperSingleHopEnvelopeProjectionResult ProjectHopperSingleHopEnvelope(
    const PlannerInput& input,
    const HopperLandingEvidenceGrid& hopper_landing_evidence) {
  const auto failure = [](std::string reason_code) {
    return HopperSingleHopEnvelopeProjectionResult{
        .projection = std::nullopt,
        .reason_code = std::move(reason_code),
    };
  };
  if (input.stop_token.stop_requested()) {
    return failure("REQUEST_CANCELED");
  }
  const auto* capability = std::get_if<HopperCapability>(&input.capability);
  const auto* state = std::get_if<HopperState>(&input.current_state);
  if (capability == nullptr || state == nullptr) {
    return failure("HOPPER_SINGLE_HOP_ENVELOPE_PLATFORM_MISMATCH");
  }
  try {
    const auto global =
        shared::MapSnapshot::Create(input.world.global_map);
    if (!global.ok()) {
      return failure(global.reason_code);
    }
    hopper::ExternalLandingBuildResult built =
        hopper::BuildExternalLandings(
            *global.snapshot, hopper_landing_evidence);
    if (!built.ok()) {
      return failure(std::move(built.reason_code));
    }
    const auto pose_map = hierarchical::TransformPose(
        state->pose, input.world.map_from_odom,
        hierarchical::TransformDirection::kChildToParent);
    if (!pose_map.has_value() || !Finite(pose_map->position_m)) {
      return failure("FRAME_TRANSFORM_INVALID");
    }
    const std::size_t cell_count = global.snapshot->cell_count();
    const double infinity = std::numeric_limits<double>::infinity();
    HopperSingleHopEnvelopeProjection projection{
        .width = global.snapshot->width(),
        .height = global.snapshot->height(),
        .eligible = std::vector<std::uint8_t>(cell_count, 0U),
        .required_delta_v_mps =
            std::vector<double>(cell_count, infinity),
        .nominal_flight_time_s =
            std::vector<double>(cell_count, 0.0),
        .algorithm_id = "cpp-hopper-single-hop-envelope/v1",
        .complete = 1U,
    };
    const double same_pose_tolerance = std::max(
        kDistanceToleranceM,
        global.snapshot->resolution_m() *
            std::sqrt(std::numeric_limits<double>::epsilon()));
    for (std::size_t index = 0U; index < cell_count; ++index) {
      if (input.stop_token.stop_requested()) {
        return failure("REQUEST_CANCELED");
      }
      const auto& landing = (*built.landings)[index];
      if (!landing.has_value()) {
        continue;
      }
      ++projection.raw_known_landing_count;
      const Vec3 target = landing->aim_position_on_surface_m;
      const double displacement = std::hypot(
          std::hypot(
              target.x - pose_map->position_m.x,
              target.y - pose_map->position_m.y),
          target.z - pose_map->position_m.z);
      if (!std::isfinite(displacement)) {
        return failure("HOPPER_LANDING_EVIDENCE_VALUE_INVALID");
      }
      if (displacement <= same_pose_tolerance) {
        continue;
      }
      ++projection.candidates_evaluated;
      const auto envelope = hopper::EvaluateMinimumSingleHopEnvelope(
          pose_map->position_m, target, capability->gravity_mps2,
          global.snapshot->resolution_m(), *capability);
      if (!envelope.ok()) {
        if (envelope.reason_code ==
            "HOPPER_SINGLE_HOP_ENVELOPE_EXCEEDED") {
          continue;
        }
        return failure(envelope.reason_code.empty()
                           ? "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE"
                           : envelope.reason_code);
      }
      if (!std::isfinite(
              envelope.evidence->envelope.required_delta_v_mps) ||
          envelope.evidence->envelope.required_delta_v_mps <= 0.0 ||
          !std::isfinite(envelope.evidence->arc.flight_time_s) ||
          envelope.evidence->arc.flight_time_s <= 0.0) {
        return failure("HOPPER_SINGLE_HOP_ENVELOPE_RESULT_INVALID");
      }
      projection.eligible[index] = 1U;
      projection.required_delta_v_mps[index] =
          envelope.evidence->envelope.required_delta_v_mps;
      projection.nominal_flight_time_s[index] =
          envelope.evidence->arc.flight_time_s;
      ++projection.eligible_count;
    }
    if (projection.eligible_count != static_cast<std::size_t>(
            std::count(projection.eligible.begin(),
                       projection.eligible.end(), 1U)) ||
        projection.candidates_evaluated >
            projection.raw_known_landing_count) {
      return failure("HOPPER_SINGLE_HOP_ENVELOPE_RESULT_INVALID");
    }
    return HopperSingleHopEnvelopeProjectionResult{
        .projection = std::move(projection),
        .reason_code = {},
    };
  } catch (const std::bad_alloc&) {
    return failure("REACHABILITY_RESOURCE_EXHAUSTED");
  } catch (...) {
    return failure("REACHABILITY_INTERNAL_FAILURE");
  }
}

}  // namespace lunar::pure_planning
