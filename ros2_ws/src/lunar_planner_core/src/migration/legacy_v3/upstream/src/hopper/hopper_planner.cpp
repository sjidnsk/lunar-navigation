#include "lunar_path_planner/v3/hopper/hopper_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <numbers>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/hopper/safe_pose_mask.hpp"
#include "lunar_path_planner/v3/hopper/swept_footprint.hpp"

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] Error Failure(
    const ErrorCode code, std::string field,
    std::string message) {
  return {
      .code = code,
      .field_path = std::move(field),
      .message = std::move(message),
  };
}

[[nodiscard]] Eigen::Vector3d ToEigen(const Vec3 value) {
  return {value.x, value.y, value.z};
}

[[nodiscard]] Eigen::Vector3d RegionCenter(
    const TerrainCertifiedLandingRegion& region) {
  Eigen::Vector2d center = Eigen::Vector2d::Zero();
  for (const Eigen::Vector2d& vertex :
       region.vertices_uv_ccw.vertices_ccw) {
    center += vertex;
  }
  center /= static_cast<double>(
      region.vertices_uv_ccw.vertices_ccw.size());
  return plane_uv_to_world(region.landing_plane, center);
}

[[nodiscard]] bool RegionContainsPosition(
    const TerrainCertifiedLandingRegion& region,
    const Eigen::Vector3d& position_m) {
  return point_in_convex_polygon(
      world_to_plane_uv(region.landing_plane, position_m),
      region.vertices_uv_ccw, region.inward_safety_margin_m);
}

[[nodiscard]] std::vector<CircularYawInterval>
PartitionYaw(
    const CircularYawInterval& interval,
    const std::size_t partition_count) {
  if (!validate_circular_yaw_interval(interval).ok() ||
      partition_count == 0U) {
    return {};
  }
  if (interval.span_rad <= 1.0e-15) {
    return {interval};
  }
  std::vector<CircularYawInterval> result;
  result.reserve(partition_count);
  const double span =
      interval.span_rad /
      static_cast<double>(partition_count);
  for (std::size_t index = 0U; index < partition_count;
       ++index) {
    result.push_back(
        {
            .start_rad = canonical_yaw(
                interval.start_rad +
                span * static_cast<double>(index)),
            .span_rad = span,
        });
  }
  return result;
}

[[nodiscard]] std::optional<std::size_t> ClosestRegion(
    const std::vector<TerrainCertifiedLandingRegion>& regions,
    const Eigen::Vector3d& target) {
  if (regions.empty()) {
    return std::nullopt;
  }
  std::size_t best = 0U;
  double best_distance = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0U; index < regions.size(); ++index) {
    const double distance =
        (RegionCenter(regions[index]) - target).squaredNorm();
    if (distance < best_distance - 1.0e-12 ||
        (std::abs(distance - best_distance) <= 1.0e-12 &&
         regions[index].region_id <
             regions[best].region_id)) {
      best = index;
      best_distance = distance;
    }
  }
  return best;
}

[[nodiscard]] Eigen::Vector3d GoalPoint(
    const GoalRegion& goal) {
  return std::visit(
      [](const auto& target) -> Eigen::Vector3d {
        using Target = std::decay_t<decltype(target)>;
        if constexpr (std::is_same_v<Target, PointGoal>) {
          return ToEigen(target.position_m);
        } else {
          Eigen::Vector2d center = Eigen::Vector2d::Zero();
          for (const Vec2 vertex :
               target.polygon.vertices_uv) {
            center += Eigen::Vector2d{vertex.x, vertex.y};
          }
          center /= static_cast<double>(
              target.polygon.vertices_uv.size());
          return plane_uv_to_world(target.plane, center);
        }
      },
      goal.target);
}

[[nodiscard]] std::optional<Eigen::Vector3d>
MissionDirection(const GoalRegion& goal) {
  if (!goal.mission_direction_hint.has_value()) {
    return std::nullopt;
  }
  return ToEigen(*goal.mission_direction_hint);
}

}  // namespace

Result<HopperReference> HopperPlanner::Plan(
    const PlanningRequest& request,
    const ResolvedTerminalSet& terminal_set) const noexcept {
  try {
    if (request.platform_type != PlatformType::kHopper ||
        !std::holds_alternative<HopperState>(
            request.current_state) ||
        request.map_snapshot == nullptr ||
        request.safety_capability == nullptr ||
        request.algorithm_config == nullptr ||
        request.frame_id.empty() ||
        request.map_snapshot->frame_id() != request.frame_id) {
      return Failure(
          ErrorCode::kInvalidArgument, "request",
          "hopper planning request is incomplete or mismatched");
    }
    if (terminal_set.kind == TerminalKind::kGoalInfeasible ||
        terminal_set.kind == TerminalKind::kNoKnownSafeRoute ||
        terminal_set.candidates.empty()) {
      return Failure(
          ErrorCode::kNoKnownSafeRoute, "terminal_set",
          "resolved terminal set contains no safe target");
    }
    const auto capability_result = bind_hopper_capability(
        *request.safety_capability,
        request.capability_bindings);
    if (!IsOk(capability_result)) {
      return std::get<Error>(capability_result);
    }
    const auto limits_result =
        bind_hopper_limits(*request.algorithm_config);
    if (!IsOk(limits_result)) {
      return std::get<Error>(limits_result);
    }
    const HopperCapabilityView capability =
        std::get<HopperCapabilityView>(capability_result);
    const HopperPlannerLimits limits =
        std::get<HopperPlannerLimits>(limits_result);
    if (capability.frame_id != request.frame_id) {
      return Failure(
          ErrorCode::kInvalidArgument,
          "safety_capability.frame_id",
          "hopper capability frame does not match request");
    }

    const CircularYawInterval requested_yaw =
        request.goal.optional_yaw_interval.value_or(
            CircularYawInterval{
                .start_rad = -std::numbers::pi,
                .span_rad = 2.0 * std::numbers::pi,
            });
    const auto yaw_partitions =
        PartitionYaw(requested_yaw, limits.yaw_partition_count);
    if (yaw_partitions.empty()) {
      return Failure(
          ErrorCode::kInvalidArgument,
          "goal.optional_yaw_interval",
          "goal yaw interval is invalid");
    }

    std::vector<TerrainCertifiedLandingRegion> regions;
    std::set<std::string> region_keys;
    for (std::size_t yaw_index = 0U;
         yaw_index < yaw_partitions.size(); ++yaw_index) {
      const auto footprint =
          outer_approximate_rotated_footprint(
              capability.landing_footprint_body_xy,
              yaw_partitions[yaw_index],
              make_uniform_unit_directions(
                  limits.support_direction_count),
              capability.landing_terrain_thresholds
                  .minimum_lateral_clearance_m);
      if (!IsOk(footprint)) {
        continue;
      }
      const auto mask = build_safe_pose_mask(
          *request.map_snapshot, capability,
          std::get<SweptFootprintEnvelope>(footprint));
      if (!IsOk(mask)) {
        continue;
      }
      const auto& safe_mask = std::get<SafePoseMask>(mask);
      const auto seeds = select_landing_seeds(
          safe_mask, limits.maximum_landing_regions);
      auto generated =
          LandingRegionGenerator{capability, limits}.generate(
              *request.map_snapshot, safe_mask, seeds);
      for (auto& region : generated) {
        const std::string key =
            region.region_id + "-yaw-" +
            std::to_string(yaw_index);
        if (!region_keys.insert(key).second) {
          continue;
        }
        region.region_id = key;
        region.terrain_certification.certification_id =
            key + "-terrain-certification";
        regions.push_back(std::move(region));
        if (regions.size() >= limits.maximum_graph_nodes) {
          break;
        }
      }
      if (regions.size() >= limits.maximum_graph_nodes) {
        break;
      }
    }
    if (regions.empty()) {
      return Failure(
          ErrorCode::kNoKnownSafeRoute, "landing_regions",
          "no terrain-certified landing region");
    }
    std::sort(
        regions.begin(), regions.end(),
        [](const auto& lhs, const auto& rhs) {
          return lhs.region_id < rhs.region_id;
        });

    const HopperState& launch_state =
        std::get<HopperState>(request.current_state);
    const Eigen::Vector3d launch_position =
        ToEigen(launch_state.position_m);
    const auto source_iterator = std::find_if(
        regions.begin(), regions.end(),
        [&launch_position](const auto& region) {
          return RegionContainsPosition(region, launch_position);
        });
    if (source_iterator == regions.end()) {
      return Failure(
          ErrorCode::kNoKnownSafeRoute,
          "current_state.position_m",
          "current hold is not inside a certified region");
    }
    const TerrainCertifiedLandingRegion source_region =
        *source_iterator;

    std::vector<std::size_t> target_region_indices;
    for (std::size_t region_index = 0U;
         region_index < regions.size(); ++region_index) {
      for (const TerminalCandidate& terminal :
           terminal_set.candidates) {
        if (RegionContainsPosition(
                regions[region_index],
                ToEigen(terminal.position_m))) {
          target_region_indices.push_back(region_index);
          break;
        }
      }
    }
    if (target_region_indices.empty()) {
      Eigen::Vector3d target = GoalPoint(request.goal);
      if (!terminal_set.candidates.empty()) {
        target = ToEigen(
            terminal_set.candidates.front().position_m);
      }
      const auto closest = ClosestRegion(regions, target);
      if (closest.has_value()) {
        target_region_indices.push_back(*closest);
      }
    }
    if (target_region_indices.empty()) {
      return Failure(
          ErrorCode::kNoKnownSafeRoute, "terminal_set",
          "no certified region intersects safe terminal");
    }
    std::sort(
        target_region_indices.begin(),
        target_region_indices.end());
    target_region_indices.erase(
        std::unique(
            target_region_indices.begin(),
            target_region_indices.end()),
        target_region_indices.end());

    const AimPointGenerator aim_generator{capability, limits};
    std::vector<LandingGraphNode> nodes;
    nodes.push_back(
        {
            .node_id = "hopper-start",
            .region = source_region,
        });
    std::vector<std::size_t> terminal_nodes;
    for (const std::size_t region_index :
         target_region_indices) {
      auto aim_points = aim_generator.generate(
          regions[region_index], request.goal,
          MissionDirection(request.goal));
      if (aim_points.empty()) {
        continue;
      }
      terminal_nodes.push_back(nodes.size());
      nodes.push_back(
          {
              .node_id =
                  "hopper-node-" +
                  regions[region_index].region_id,
              .region = regions[region_index],
              .aim_points = std::move(aim_points),
          });
      if (nodes.size() >= limits.maximum_graph_nodes) {
        break;
      }
    }
    if (terminal_nodes.empty()) {
      return Failure(
          ErrorCode::kNoKnownSafeRoute, "aim_points",
          "no finite interior aim point in terminal region");
    }

    const auto lower_bound =
        capability.launch_limits.minimum_flight_time.value +
        capability.actuator_or_impulse_profile
            .launch_preparation_time.value +
        capability.actuator_or_impulse_profile
            .landing_settle_time.value;
    std::vector<LandingGraphEdge> edges;
    const std::size_t edge_count = std::min(
        terminal_nodes.size(), limits.maximum_graph_out_degree);
    for (std::size_t index = 0U; index < edge_count; ++index) {
      const std::size_t target_node = terminal_nodes[index];
      edges.push_back(
          {
              .edge_id =
                  "hopper-start-edge-" +
                  nodes[target_node].node_id,
              .from_node = 0U,
              .to_node = target_node,
              .state = LandingEdgeState::kCheapPossible,
              .lower_bound_execution_time =
                  DurationNanoseconds{lower_bound},
              .estimated_energy_lower_bound_j = 0.0,
          });
    }
    if (edges.empty()) {
      return Failure(
          ErrorCode::kNoKnownSafeRoute, "landing_graph",
          "no cheap possible first edge");
    }

    HopCertifier certifier{
        {
            .reference_time_origin = request.request_time,
            .launch_state = launch_state,
            .map = request.map_snapshot.get(),
            .capability = capability,
            .limits = limits,
            .source_request_id = request.request_id,
            .source_error_model_ref =
                capability.error_model.content_ref,
        }};
    const auto selection =
        LazyLandingGraphPlanner{}.select(
            std::move(nodes), std::move(edges), 0U,
            terminal_nodes, certifier, limits);
    if (!IsOk(selection)) {
      return std::get<Error>(selection);
    }
    HopperReference reference =
        std::get<LazyNextHopSelection>(selection)
            .certified_next_hop;
    reference.future_route_preview =
        std::get<LazyNextHopSelection>(selection).future_preview;
    reference.reference_hash = std::string(64U, '0');
    const auto hash =
        CanonicalReferenceHash(PlatformReference{reference});
    if (!IsOk(hash)) {
      return std::get<Error>(hash);
    }
    reference.reference_hash =
        std::get<Sha256Digest>(hash);
    const HopCertificationContext final_context{
        .reference_time_origin = request.request_time,
        .launch_state = launch_state,
        .map = request.map_snapshot.get(),
        .capability = capability,
        .limits = limits,
        .source_request_id = request.request_id,
        .source_error_model_ref =
            capability.error_model.content_ref,
    };
    if (!validate_final_hopper_reference(
             reference, final_context)
             .ok()) {
      return Failure(
          ErrorCode::kNumericalFailure,
          "hopper_reference",
          "final Hopper reference failed validation");
    }
    return reference;
  } catch (const std::bad_alloc&) {
    return Failure(
        ErrorCode::kResourceLimit, "hopper_planner",
        "resource allocation failed");
  } catch (const std::exception& exception) {
    return Failure(
        ErrorCode::kNumericalFailure, "hopper_planner",
        exception.what());
  } catch (...) {
    return Failure(
        ErrorCode::kNumericalFailure, "hopper_planner",
        "unknown Hopper planning failure");
  }
}

}  // namespace lunar::planning::v3
