#include "hopper/hopper_reachability_graph.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <new>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "hierarchical/frame_transform.hpp"
#include "hopper/ballistic_envelope.hpp"
#include "hopper/ballistic_kinematics.hpp"
#include "hopper/flight_tube_certifier.hpp"
#include "hopper/hop_certifier.hpp"
#include "hopper/landing_evidence.hpp"
#include "hopper/landing_region.hpp"

namespace lunar::pure_planning::hopper {
namespace {

constexpr double kDistanceToleranceM = 1.0e-9;
constexpr double kDefaultPrimitiveDistanceM = 30.0;
constexpr std::size_t kNoState = std::numeric_limits<std::size_t>::max();

[[nodiscard]] shared::PrimitiveGraphBuildResult Failure(
    std::string reason_code) {
  return shared::PrimitiveGraphBuildResult{
      .reason_code = std::move(reason_code),
  };
}

struct DirectedEdgeResult final {
  HopCertificationStatus status{HopCertificationStatus::kInvalid};
  double cost{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return status == HopCertificationStatus::kCertified &&
        std::isfinite(cost) && cost > 0.0 && reason_code.empty();
  }
};

[[nodiscard]] std::optional<CertifiedLandingRegion>
CertifiedLandingAtPosition(
    const Vec3 center_map, const shared::MapSnapshot& local_map,
    const PlannerInput& input, const HopperCapability& capability,
    std::string& fatal_reason) {
  const auto center_local = hierarchical::TransformPoint(
      center_map, input.world.map_from_odom,
      hierarchical::TransformDirection::kParentToChild);
  if (!center_local.has_value()) {
    fatal_reason = "FRAME_TRANSFORM_INVALID";
    return std::nullopt;
  }
  const LandingRegionResult landing = CertifyExactLandingRegion(
      local_map,
      GoalRegion{
          .goal_id = "primitive-reachability-landing",
          .target = PointGoal{
              .position_m = *center_local,
              .tolerance_m = 0.0,
          },
      },
      capability, input.config.map_safety, input.stop_token);
  if (!landing.ok()) {
    switch (landing.status) {
      case LandingRegionStatus::kInfeasible:
        return std::nullopt;
      case LandingRegionStatus::kCanceled:
        fatal_reason = "REQUEST_CANCELED";
        return std::nullopt;
      case LandingRegionStatus::kResourceExhausted:
        fatal_reason = "REACHABILITY_RESOURCE_EXHAUSTED";
        return std::nullopt;
      case LandingRegionStatus::kInvalidRequest:
        if (landing.reason_code ==
            "HOPPER_LANDING_REGION_NUMERICAL_INDETERMINATE") {
          return std::nullopt;
        }
        fatal_reason = landing.reason_code.empty()
            ? "HOPPER_LANDING_REGION_INVALID"
            : landing.reason_code;
        return std::nullopt;
      case LandingRegionStatus::kCertified:
        fatal_reason = "HOPPER_LANDING_RESULT_INVALID";
        return std::nullopt;
    }
  }
  CertifiedLandingRegion transformed = *landing.region;
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

[[nodiscard]] DirectedEdgeResult CertifyLandingRegionHop(
    const CertifiedSingleHop& center_hop,
    CertifiedLandingRegion landing_region,
    const shared::MapSnapshot& flight_map,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety,
    const std::stop_token stop_token) {
  const double minimum_region_radius = flight_map.resolution_m() *
      std::sqrt(std::numeric_limits<double>::epsilon());
  while (true) {
    if (stop_token.stop_requested()) {
      return {
          .status = HopCertificationStatus::kCanceled,
          .reason_code = "REQUEST_CANCELED",
      };
    }
    double region_radius = 0.0;
    bool corners_certified = true;
    for (const Vec3 vertex : landing_region.boundary_m) {
      region_radius = std::max(
          region_radius,
          std::hypot(
              vertex.x - landing_region.aim_position_on_surface_m.x,
              vertex.y - landing_region.aim_position_on_surface_m.y));
      const BallisticSolveResult vertex_arc = SolveBallisticArc(
          center_hop.arc.launch_position_m, vertex,
          center_hop.arc.gravity_mps2, center_hop.arc.flight_time_s);
      if (!vertex_arc.ok()) {
        return {
            .status = HopCertificationStatus::kNumericalIndeterminate,
            .reason_code =
                "HOPPER_LANDING_REGION_NUMERICAL_INDETERMINATE",
        };
      }
      const SingleHopEnvelopeResult envelope =
          EvaluateSingleHopEnvelope(*vertex_arc.arc, capability);
      if (!envelope.ok()) {
        if (envelope.reason_code != "HOPPER_SINGLE_HOP_ENVELOPE_EXCEEDED") {
          return {
              .status = HopCertificationStatus::kNumericalIndeterminate,
              .reason_code =
                  "HOPPER_LANDING_REGION_NUMERICAL_INDETERMINATE",
          };
        }
        corners_certified = false;
        break;
      }
    }
    FlightTubeCertificationResult tube;
    if (corners_certified) {
      tube = CertifyFlightTube(
          center_hop.arc, flight_map, capability, map_safety, stop_token,
          region_radius);
      if (tube.canceled) {
        return {
            .status = HopCertificationStatus::kCanceled,
            .reason_code = "REQUEST_CANCELED",
        };
      }
      if (tube.reason_code == "HOPPER_FLIGHT_TUBE_INPUT_INVALID" ||
          tube.reason_code.find("NUMERICAL") != std::string::npos) {
        return {
            .status = HopCertificationStatus::kNumericalIndeterminate,
            .reason_code =
                "HOPPER_LANDING_REGION_NUMERICAL_INDETERMINATE",
        };
      }
    }
    if (corners_certified && tube.certified) {
      return {
          .status = HopCertificationStatus::kCertified,
          .cost = center_hop.arc.flight_time_s +
              0.1 * center_hop.envelope.required_delta_v_mps,
      };
    }
    if (!(region_radius > minimum_region_radius)) {
      return {
          .status = HopCertificationStatus::kInfeasible,
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

[[nodiscard]] DirectedEdgeResult CertifyDirectedEdge(
    const CertifiedLandingRegion& source,
    const CertifiedLandingRegion& target,
    const shared::MapSnapshot& flight_map,
    const HopperCapability& capability,
    const MapSafetyConfig& map_safety,
    const std::stop_token stop_token) {
  const SingleHopCertificationResult hop = CertifySingleHop(
      SingleHopCertificationProblem{
          .launch_position_m = source.aim_position_on_surface_m,
          .landing_position_m = target.aim_position_on_surface_m,
          .gravity_mps2 = capability.gravity_mps2,
          .flight_map = &flight_map,
          .capability = &capability,
          .map_safety = &map_safety,
          .stop_token = stop_token,
      });
  if (!hop.ok()) {
    return {
        .status = hop.status,
        .reason_code = hop.reason_code,
    };
  }
  return CertifyLandingRegionHop(
      *hop.certification, target, flight_map, capability, map_safety,
      stop_token);
}

[[nodiscard]] std::string FatalEdgeReason(
    const DirectedEdgeResult& edge) {
  switch (edge.status) {
    case HopCertificationStatus::kCanceled:
      return "REQUEST_CANCELED";
    case HopCertificationStatus::kResourceExhausted:
      return "REACHABILITY_RESOURCE_EXHAUSTED";
    case HopCertificationStatus::kInvalid:
    case HopCertificationStatus::kNumericalIndeterminate:
      return edge.reason_code.empty()
          ? "HOPPER_REACHABILITY_CERTIFICATION_INVALID"
          : edge.reason_code;
    case HopCertificationStatus::kCertified:
      return edge.ok() ? std::string{}
                       : "HOPPER_REACHABILITY_CERTIFICATION_INVALID";
    case HopCertificationStatus::kInfeasible:
      return {};
  }
  return "HOPPER_REACHABILITY_CERTIFICATION_INVALID";
}

void AppendUint64(
    std::vector<std::uint8_t>& output, const std::uint64_t value) {
  for (std::size_t index = 0U; index < 8U; ++index) {
    output.push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
  }
}

void AppendDouble(std::vector<std::uint8_t>& output, double value) {
  if (value == 0.0) {
    value = 0.0;
  }
  AppendUint64(output, std::bit_cast<std::uint64_t>(value));
}

void AppendString(
    std::vector<std::uint8_t>& output, const std::string& value) {
  AppendUint64(output, static_cast<std::uint64_t>(value.size()));
  output.insert(output.end(), value.begin(), value.end());
}

[[nodiscard]] std::vector<std::uint8_t> PrimitiveBytes(
    const HopperCapability& capability, const double edge_distance_m) {
  std::vector<std::uint8_t> output;
  AppendString(output, "hopper-certified-directed-hop/v4");
  AppendDouble(output, edge_distance_m);
  AppendDouble(output, capability.specific_impulse_s);
  AppendDouble(output, capability.reference_total_mass_kg);
  AppendDouble(output, capability.reference_propellant_mass_kg);
  AppendDouble(output, capability.gravity_mps2.x);
  AppendDouble(output, capability.gravity_mps2.y);
  AppendDouble(output, capability.gravity_mps2.z);
  AppendDouble(output, capability.reference_horizontal_range_m);
  AppendDouble(output, capability.reference_elevation_delta_m);
  output.push_back(capability.runtime_fallback_allowed ? 1U : 0U);
  AppendDouble(output, capability.landing_support_radius_m);
  AppendDouble(output, capability.flight_collision_radius_m);
  AppendDouble(output, capability.maximum_landing_plane_residual_m);
  AppendDouble(output, capability.landing_lateral_margin_m);
  AppendDouble(output, capability.flight_map_margin_m);
  AppendDouble(output, capability.reachability_delta_v_margin_ratio);
  AppendDouble(output, capability.standard_gravity_mps2);
  AppendDouble(output, capability.maximum_landing_slope_rad);
  return output;
}

}  // namespace

shared::PrimitiveGraphBuildResult BuildHopperPrimitiveGraph(
    const PlannerInput& input,
    const std::shared_ptr<const shared::MapSnapshot>& global_map,
    const std::shared_ptr<const shared::MapSnapshot>& local_map,
    const shared::SafeProjection& safe,
    const std::optional<double> maximum_edge_distance_m,
    const HopperLandingEvidenceGrid* const landing_evidence) try {
  if (input.stop_token.stop_requested()) {
    return Failure("REQUEST_CANCELED");
  }
  const auto* capability = std::get_if<HopperCapability>(&input.capability);
  const auto* current_state = std::get_if<HopperState>(&input.current_state);
  const double edge_distance =
      maximum_edge_distance_m.value_or(kDefaultPrimitiveDistanceM);
  if (capability == nullptr || current_state == nullptr ||
      global_map == nullptr || local_map == nullptr ||
      safe.source_map() == nullptr || !std::isfinite(edge_distance) ||
      edge_distance <= 0.0) {
    return Failure("HOPPER_PRIMITIVE_GRAPH_REQUEST_INVALID");
  }
  const auto pose_map = hierarchical::TransformPose(
      current_state->pose, input.world.map_from_odom,
      hierarchical::TransformDirection::kChildToParent);
  if (!pose_map.has_value()) {
    return Failure("FRAME_TRANSFORM_INVALID");
  }
  const auto start_cell = global_map->PositionToCell(
      Vec2{.x = pose_map->position_m.x, .y = pose_map->position_m.y});
  if (!start_cell.has_value() || !safe.HardFeasible(*start_cell)) {
    return Failure("HOPPER_START_NOT_SAFE");
  }

  std::vector<std::optional<CertifiedLandingRegion>> landings;
  if (landing_evidence != nullptr) {
    ExternalLandingBuildResult built =
        BuildExternalLandings(*global_map, *landing_evidence);
    if (!built.ok()) {
      return Failure(std::move(built.reason_code));
    }
    landings = std::move(*built.landings);
  } else {
    landings.resize(global_map->cell_count());
    for (std::size_t index = 0U; index < landings.size(); ++index) {
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
      landings[index] = CertifiedLandingAtPosition(
          global_map->CellCenter(cell), *local_map, input, *capability,
          fatal_reason);
      if (!fatal_reason.empty()) {
        return Failure(std::move(fatal_reason));
      }
    }
  }
  const std::size_t start_grid_index = global_map->Index(*start_cell);
  if (!landings[start_grid_index].has_value()) {
    return Failure("HOPPER_START_LANDING_NOT_CERTIFIED");
  }

  shared::PrimitiveGraphBuildResult graph{
      .platform_type = PlatformType::kHopper,
      .width = global_map->width(),
      .height = global_map->height(),
      .anchor_state_index = 0U,
      .algorithm_id = "cpp-hopper-certified-recoverable-state-graph/v4",
      .state_schema = "hopper-landing-state/v2",
      .primitive_set_canonical_bytes =
          PrimitiveBytes(*capability, edge_distance),
  };
  std::vector<std::size_t> state_by_grid(global_map->cell_count(), kNoState);
  std::vector<std::size_t> grid_by_state;
  const auto add_state = [&](const std::size_t grid_index,
                             const double path_cost) {
    const CertifiedLandingRegion& landing = *landings[grid_index];
    const std::size_t state_index = graph.states.size();
    const shared::GridCell cell{
        .x = static_cast<std::int32_t>(grid_index % global_map->width()),
        .y = static_cast<std::int32_t>(grid_index / global_map->width()),
    };
    graph.states.push_back(PrimitiveReachabilityState{
        .position_m = landing.aim_position_on_surface_m,
        .yaw_rad = 0.0,
        .cell_x = cell.x,
        .cell_y = cell.y,
        .yaw_bin = 0,
        .motion_mode = 0,
        .body_z_m = Interval{},
        .path_cost = path_cost,
        .observation_state = 1U,
    });
    state_by_grid[grid_index] = state_index;
    grid_by_state.push_back(grid_index);
    return state_index;
  };
  add_state(start_grid_index, 0.0);
  std::deque<std::size_t> queue{0U};

  const auto cell_radius = static_cast<std::int32_t>(
      std::ceil(edge_distance / global_map->resolution_m()));
  std::vector<std::pair<std::int32_t, std::int32_t>> offsets;
  for (std::int32_t dy = -cell_radius; dy <= cell_radius; ++dy) {
    for (std::int32_t dx = -cell_radius; dx <= cell_radius; ++dx) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const double distance = std::hypot(
          static_cast<double>(dx) * global_map->resolution_m(),
          static_cast<double>(dy) * global_map->resolution_m());
      if (distance <= edge_distance + kDistanceToleranceM) {
        offsets.emplace_back(dx, dy);
      }
    }
  }
  std::stable_sort(
      offsets.begin(), offsets.end(),
      [](const auto lhs, const auto rhs) {
        const auto lhs_squared = lhs.first * lhs.first + lhs.second * lhs.second;
        const auto rhs_squared = rhs.first * rhs.first + rhs.second * rhs.second;
        return std::tie(lhs_squared, lhs.second, lhs.first) <
            std::tie(rhs_squared, rhs.second, rhs.first);
      });

  while (!queue.empty()) {
    if (input.stop_token.stop_requested()) {
      return Failure("REQUEST_CANCELED");
    }
    const std::size_t source_state_index = queue.front();
    queue.pop_front();
    const std::size_t source_grid_index = grid_by_state[source_state_index];
    const shared::GridCell source_cell{
        .x = static_cast<std::int32_t>(
            source_grid_index % global_map->width()),
        .y = static_cast<std::int32_t>(
            source_grid_index / global_map->width()),
    };
    for (const auto [dx, dy] : offsets) {
      const shared::GridCell target_cell{
          .x = source_cell.x + dx,
          .y = source_cell.y + dy,
      };
      if (!global_map->InBounds(target_cell)) {
        continue;
      }
      const std::size_t target_grid_index = global_map->Index(target_cell);
      if (!landings[target_grid_index].has_value()) {
        continue;
      }
      const DirectedEdgeResult forward = CertifyDirectedEdge(
          *landings[source_grid_index], *landings[target_grid_index],
          *global_map, *capability, input.config.map_safety,
          input.stop_token);
      if (!forward.ok()) {
        const std::string fatal = FatalEdgeReason(forward);
        if (!fatal.empty()) {
          return Failure(fatal);
        }
        continue;
      }
      std::size_t target_state_index = state_by_grid[target_grid_index];
      if (target_state_index == kNoState) {
        target_state_index = add_state(
            target_grid_index,
            graph.states[source_state_index].path_cost + forward.cost);
        queue.push_back(target_state_index);
      }
      graph.potential_edges.push_back(shared::PrimitiveGraphPotentialEdge{
          .source_state_index = source_state_index,
          .target_state_index = target_state_index,
          .primitive_index = 0U,
          .primitive_id = "certified-directed-hop",
          .cost = forward.cost,
          .certified = true,
      });
    }
  }
  return graph;
} catch (const std::bad_alloc&) {
  return Failure("REACHABILITY_RESOURCE_EXHAUSTED");
}

}  // namespace lunar::pure_planning::hopper
