#include "lunar_path_planner/v3/map/safe_projection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <queue>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace lunar::planning::v3 {
namespace {

constexpr double kComparisonTolerance = 1.0e-12;

[[nodiscard]] Error Invalid(std::string field_path,
                            std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] Error Missing(std::string field_path,
                            std::string message) {
  return {
      .code = ErrorCode::kMissingRegistryObject,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool IsFinite(const Vec2& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y);
}

[[nodiscard]] bool IsFinite(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool IsFiniteInterval(const Interval& value) noexcept {
  return std::isfinite(value.lower) && std::isfinite(value.upper) &&
         value.lower <= value.upper;
}

[[nodiscard]] bool IsPositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool IsNonNegative(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] bool ValidSlopeLimit(double value) noexcept {
  return IsNonNegative(value) &&
         value <= std::numbers::pi_v<double> / 2.0;
}

[[nodiscard]] bool ValidWheelEnvelope(
    const ExtrudedConvexFootprint& envelope) noexcept {
  return envelope.vertices_xy_m.size() >= 3U &&
         std::ranges::all_of(
             envelope.vertices_xy_m,
             [](const Vec2& vertex) { return IsFinite(vertex); }) &&
         std::isfinite(envelope.minimum_z_m) &&
         std::isfinite(envelope.maximum_z_m) &&
         envelope.minimum_z_m <= envelope.maximum_z_m;
}

[[nodiscard]] bool ValidBodyEnvelope(
    const BodyConvexPolytope& envelope) noexcept {
  const auto& halfspaces = envelope.body_frame_halfspaces.halfspaces;
  return halfspaces.size() >= 4U &&
         std::ranges::all_of(
             halfspaces, [](const Halfspace3& halfspace) {
               return IsFinite(halfspace.normal) &&
                      std::isfinite(halfspace.offset_m) &&
                      std::hypot(halfspace.normal.x,
                                 halfspace.normal.y,
                                 halfspace.normal.z) >
                          0.0;
             });
}

[[nodiscard]] double MaximumAbsolute(
    const Interval& bounds) noexcept {
  return std::max(std::abs(bounds.lower), std::abs(bounds.upper));
}

[[nodiscard]] std::string CapabilityFrame(
    const SafetyCapabilityProfile& capability) {
  return std::visit(
      [](const auto& concrete) { return concrete.frame_id; },
      capability.content);
}

[[nodiscard]] std::size_t CellIndex(const GridGeometry& geometry,
                                    Cell cell) noexcept {
  return static_cast<std::size_t>(cell.y) * geometry.width +
         static_cast<std::size_t>(cell.x);
}

void DistanceTransform1D(const std::vector<double>& source,
                         std::vector<double>& destination) {
  const std::size_t count = source.size();
  std::vector<std::size_t> sites(count, 0U);
  std::vector<double> boundaries(
      count + 1U, std::numeric_limits<double>::infinity());
  std::size_t envelope_size = 0U;
  boundaries[0] = -std::numeric_limits<double>::infinity();
  boundaries[1] = std::numeric_limits<double>::infinity();

  for (std::size_t q = 1U; q < count; ++q) {
    double intersection = 0.0;
    for (;;) {
      const std::size_t site = sites[envelope_size];
      const double q_value = static_cast<double>(q);
      const double site_value = static_cast<double>(site);
      intersection =
          ((source[q] + q_value * q_value) -
           (source[site] + site_value * site_value)) /
          (2.0 * (q_value - site_value));
      if (intersection > boundaries[envelope_size] ||
          envelope_size == 0U) {
        break;
      }
      --envelope_size;
    }
    ++envelope_size;
    sites[envelope_size] = q;
    boundaries[envelope_size] = intersection;
    boundaries[envelope_size + 1U] =
        std::numeric_limits<double>::infinity();
  }

  std::size_t active = 0U;
  for (std::size_t q = 0U; q < count; ++q) {
    const double q_value = static_cast<double>(q);
    while (boundaries[active + 1U] < q_value) {
      ++active;
    }
    const double delta =
        q_value - static_cast<double>(sites[active]);
    destination[q] =
        delta * delta + source[sites[active]];
  }
}

[[nodiscard]] Result<std::vector<float>> ComputeEsdf(
    const ImmutableMapSnapshot& map) {
  const GridGeometry& geometry = map.geometry();
  if (geometry.width >
          std::numeric_limits<std::size_t>::max() - 2U ||
      geometry.height >
          std::numeric_limits<std::size_t>::max() - 2U) {
    return Invalid("safe_projection.geometry",
                   "padded ESDF geometry overflows size_t");
  }
  const std::size_t padded_width = geometry.width + 2U;
  const std::size_t padded_height = geometry.height + 2U;
  if (padded_height >
      std::numeric_limits<std::size_t>::max() / padded_width) {
    return Invalid("safe_projection.geometry",
                   "padded ESDF cell count overflows size_t");
  }

  constexpr double kUnoccupiedSquaredDistance = 1.0e30;
  std::vector<double> initial(padded_width * padded_height, 0.0);
  const auto known = map.KnownMask();
  const auto obstacle = map.HardObstacleMask();
  for (std::size_t y = 0U; y < geometry.height; ++y) {
    for (std::size_t x = 0U; x < geometry.width; ++x) {
      const std::size_t source_index = y * geometry.width + x;
      const std::size_t padded_index =
          (y + 1U) * padded_width + (x + 1U);
      initial[padded_index] =
          known[source_index] != 0U &&
                  obstacle[source_index] == 0U
              ? kUnoccupiedSquaredDistance
              : 0.0;
    }
  }

  std::vector<double> horizontal(initial.size(), 0.0);
  std::vector<double> row_source(padded_width, 0.0);
  std::vector<double> row_result(padded_width, 0.0);
  for (std::size_t y = 0U; y < padded_height; ++y) {
    const auto row_begin = initial.begin() +
                           static_cast<std::ptrdiff_t>(
                               y * padded_width);
    std::copy_n(row_begin, padded_width, row_source.begin());
    DistanceTransform1D(row_source, row_result);
    std::copy(row_result.begin(), row_result.end(),
              horizontal.begin() + static_cast<std::ptrdiff_t>(
                                       y * padded_width));
  }

  std::vector<double> squared_distance(initial.size(), 0.0);
  std::vector<double> column_source(padded_height, 0.0);
  std::vector<double> column_result(padded_height, 0.0);
  for (std::size_t x = 0U; x < padded_width; ++x) {
    for (std::size_t y = 0U; y < padded_height; ++y) {
      column_source[y] = horizontal[y * padded_width + x];
    }
    DistanceTransform1D(column_source, column_result);
    for (std::size_t y = 0U; y < padded_height; ++y) {
      squared_distance[y * padded_width + x] = column_result[y];
    }
  }

  std::vector<float> clearance(geometry.CellCount(), 0.0F);
  for (std::size_t y = 0U; y < geometry.height; ++y) {
    for (std::size_t x = 0U; x < geometry.width; ++x) {
      const double cells = std::sqrt(
          squared_distance[(y + 1U) * padded_width + (x + 1U)]);
      clearance[y * geometry.width + x] =
          static_cast<float>(cells * geometry.resolution_m);
    }
  }
  return clearance;
}

[[nodiscard]] bool PassesStepLimit(
    std::size_t x, std::size_t y,
    const ImmutableMapSnapshot& map,
    double maximum_step_height_m) noexcept {
  const auto& geometry = map.geometry();
  const auto elevation = map.ElevationMeters();
  const auto known = map.KnownMask();
  const auto obstacle = map.HardObstacleMask();
  const std::size_t index = y * geometry.width + x;
  constexpr std::int32_t kDx[] = {-1, 1, 0, 0};
  constexpr std::int32_t kDy[] = {0, 0, 1, -1};
  for (std::size_t neighbor = 0U; neighbor < 4U; ++neighbor) {
    const auto next_x =
        static_cast<std::int64_t>(x) + kDx[neighbor];
    const auto next_y =
        static_cast<std::int64_t>(y) + kDy[neighbor];
    if (next_x < 0 || next_y < 0 ||
        next_x >= static_cast<std::int64_t>(geometry.width) ||
        next_y >= static_cast<std::int64_t>(geometry.height)) {
      continue;
    }
    const std::size_t next_index =
        static_cast<std::size_t>(next_y) * geometry.width +
        static_cast<std::size_t>(next_x);
    if (known[next_index] == 0U || obstacle[next_index] != 0U) {
      continue;
    }
    const double step = std::abs(
        static_cast<double>(elevation[index]) -
        static_cast<double>(elevation[next_index]));
    if (step > maximum_step_height_m + kComparisonTolerance) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] double TerrainSpeed(
    double base_speed, double slope_rad, double roughness_m,
    double resolution_m,
    std::optional<float> static_limit) noexcept {
  const double slope_factor = std::max(0.0, std::cos(slope_rad));
  const double roughness_scale =
      std::max(resolution_m,
               std::numeric_limits<double>::min());
  const double roughness_factor =
      1.0 / (1.0 + roughness_m / roughness_scale);
  double speed = base_speed * slope_factor * roughness_factor;
  if (static_limit.has_value()) {
    speed = std::min(speed, static_cast<double>(*static_limit));
  }
  return speed;
}

void LabelConnectedComponents(SafeProjection& projection) {
  const std::size_t count = projection.geometry.CellCount();
  projection.connected_component.assign(count, -1);
  std::int32_t component = 0;
  std::vector<Cell> queue;
  queue.reserve(count);
  constexpr std::int32_t kDx[] = {-1, 1, 0, 0};
  constexpr std::int32_t kDy[] = {0, 0, 1, -1};

  for (std::size_t y = 0U; y < projection.geometry.height; ++y) {
    for (std::size_t x = 0U; x < projection.geometry.width; ++x) {
      const std::size_t seed_index =
          y * projection.geometry.width + x;
      if (projection.hard_feasible_mask[seed_index] == 0U ||
          projection.connected_component[seed_index] >= 0) {
        continue;
      }

      queue.clear();
      queue.push_back(
          {static_cast<std::int32_t>(x),
           static_cast<std::int32_t>(y)});
      projection.connected_component[seed_index] = component;
      std::size_t head = 0U;
      while (head < queue.size()) {
        const Cell current = queue[head++];
        for (std::size_t neighbor = 0U; neighbor < 4U; ++neighbor) {
          const Cell next{
              .x = current.x + kDx[neighbor],
              .y = current.y + kDy[neighbor],
          };
          if (!projection.InBounds(next)) {
            continue;
          }
          const std::size_t next_index =
              CellIndex(projection.geometry, next);
          if (projection.hard_feasible_mask[next_index] == 0U ||
              projection.connected_component[next_index] >= 0) {
            continue;
          }
          projection.connected_component[next_index] = component;
          queue.push_back(next);
        }
      }
      ++component;
    }
  }
}

[[nodiscard]] std::optional<Cell> PositionCell(
    const GridGeometry& geometry, const Vec3& position) noexcept {
  if (!IsFinite(position)) {
    return std::nullopt;
  }
  const double grid_x =
      (position.x - geometry.origin_m.x) / geometry.resolution_m;
  const double grid_y =
      (position.y - geometry.origin_m.y) / geometry.resolution_m;
  if (!std::isfinite(grid_x) || !std::isfinite(grid_y) ||
      grid_x < 0.0 || grid_y < 0.0 ||
      grid_x >= static_cast<double>(geometry.width) ||
      grid_y >= static_cast<double>(geometry.height)) {
    return std::nullopt;
  }
  const double floored_x = std::floor(grid_x);
  const double floored_y = std::floor(grid_y);
  if (floored_x >
          static_cast<double>(
              std::numeric_limits<std::int32_t>::max()) ||
      floored_y >
          static_cast<double>(
              std::numeric_limits<std::int32_t>::max())) {
    return std::nullopt;
  }
  return Cell{
      .x = static_cast<std::int32_t>(floored_x),
      .y = static_cast<std::int32_t>(floored_y),
  };
}

}  // namespace

Result<SafetyProjectionLimits> ResolveProjectionLimits(
    const SafetyCapabilityProfile& capability) {
  return std::visit(
      [&](const auto& concrete) -> Result<SafetyProjectionLimits> {
        using Capability = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<Capability,
                                     WheeledCapability>) {
          const auto& limits = concrete.hard_limits;
          if (!ValidWheelEnvelope(concrete.collision_envelope) ||
              concrete.frame_id.empty() ||
              !IsPositive(limits.maximum_forward_speed_mps) ||
              !IsPositive(limits.maximum_reverse_speed_mps) ||
              !IsPositive(limits.maximum_spin_rate_radps) ||
              !IsPositive(
                  limits.maximum_braking_deceleration_mps2) ||
              !IsPositive(
                  limits.maximum_yaw_acceleration_radps2) ||
              !IsPositive(
                  limits.maximum_forward_acceleration_mps2) ||
              !IsPositive(
                  limits.maximum_lateral_acceleration_mps2) ||
              !IsPositive(
                  limits.maximum_drive_curvature_per_m) ||
              !ValidSlopeLimit(limits.maximum_slope_rad) ||
              !IsNonNegative(limits.minimum_clearance_m)) {
            return Invalid(
                "safety_capability.content",
                "wheeled projection hard limits are incomplete");
          }
          return SafetyProjectionLimits{
              .platform_type = PlatformType::kWheeled,
              .maximum_slope_rad = limits.maximum_slope_rad,
              .maximum_roughness_m = std::nullopt,
              .maximum_step_height_m = std::nullopt,
              .minimum_clearance_m = limits.minimum_clearance_m,
              .minimum_confidence = 0.0,
              .maximum_linear_speed_mps =
                  std::max(limits.maximum_forward_speed_mps,
                           limits.maximum_reverse_speed_mps),
              .maximum_linear_deceleration_mps2 =
                  limits.maximum_braking_deceleration_mps2,
              .maximum_yaw_rate_radps =
                  limits.maximum_spin_rate_radps,
              .maximum_yaw_deceleration_radps2 =
                  limits.maximum_yaw_acceleration_radps2,
              .supports_safe_stop_anchor = true,
          };
        } else if constexpr (std::is_same_v<Capability,
                                            LeggedCapability>) {
          const auto& terrain = concrete.terrain_thresholds;
          const auto& velocity = concrete.body_velocity_limits;
          if (!ValidBodyEnvelope(concrete.collision_envelope) ||
              concrete.frame_id.empty() ||
              !ValidSlopeLimit(terrain.maximum_slope_rad) ||
              !IsNonNegative(terrain.maximum_roughness_m) ||
              !IsNonNegative(terrain.maximum_step_height_m) ||
              !IsNonNegative(terrain.maximum_gap_width_m) ||
              !IsNonNegative(terrain.minimum_body_clearance_m) ||
              !std::isfinite(terrain.minimum_confidence) ||
              terrain.minimum_confidence < 0.0 ||
              terrain.minimum_confidence > 1.0 ||
              !std::isfinite(terrain.minimum_body_height_m) ||
              !std::isfinite(terrain.maximum_body_height_m) ||
              terrain.minimum_body_height_m >
                  terrain.maximum_body_height_m ||
              !IsFiniteInterval(velocity.forward_mps) ||
              !IsFiniteInterval(velocity.lateral_mps) ||
              !IsFiniteInterval(velocity.vertical_mps) ||
              !IsFiniteInterval(velocity.yaw_rate_radps) ||
              !IsPositive(velocity.linear_acceleration_mps2) ||
              !IsPositive(velocity.yaw_acceleration_radps2)) {
            return Invalid(
                "safety_capability.content",
                "legged projection hard limits are incomplete");
          }
          const double maximum_linear_speed = std::hypot(
              MaximumAbsolute(velocity.forward_mps),
              MaximumAbsolute(velocity.lateral_mps),
              MaximumAbsolute(velocity.vertical_mps));
          const double maximum_yaw_rate =
              MaximumAbsolute(velocity.yaw_rate_radps);
          if (!IsPositive(maximum_linear_speed) ||
              !IsPositive(maximum_yaw_rate)) {
            return Invalid(
                "safety_capability.content.body_velocity_limits",
                "legged velocity bounds must certify motion and stop");
          }
          return SafetyProjectionLimits{
              .platform_type = PlatformType::kLegged,
              .maximum_slope_rad = terrain.maximum_slope_rad,
              .maximum_roughness_m =
                  terrain.maximum_roughness_m,
              .maximum_step_height_m =
                  terrain.maximum_step_height_m,
              .minimum_clearance_m =
                  terrain.minimum_body_clearance_m,
              .minimum_confidence = terrain.minimum_confidence,
              .maximum_linear_speed_mps = maximum_linear_speed,
              .maximum_linear_deceleration_mps2 =
                  velocity.linear_acceleration_mps2,
              .maximum_yaw_rate_radps = maximum_yaw_rate,
              .maximum_yaw_deceleration_radps2 =
                  velocity.yaw_acceleration_radps2,
              .supports_safe_stop_anchor = true,
          };
        } else {
          const auto& terrain =
              concrete.landing_terrain_thresholds;
          const auto& launch = concrete.launch_limits;
          if (!ValidBodyEnvelope(concrete.collision_envelope) ||
              concrete.frame_id.empty() ||
              !ValidSlopeLimit(terrain.maximum_slope_rad) ||
              !IsNonNegative(terrain.maximum_roughness_m) ||
              !IsNonNegative(terrain.maximum_plane_residual_m) ||
              !IsNonNegative(
                  terrain.minimum_overhead_clearance_m) ||
              !IsNonNegative(
                  terrain.minimum_lateral_clearance_m) ||
              !IsPositive(terrain.minimum_landing_region_area_m2) ||
              !IsPositive(launch.maximum_launch_speed_mps) ||
              !IsPositive(
                  launch.maximum_launch_impulse_newton_seconds) ||
              launch.minimum_flight_time.value.count() < 0 ||
              launch.maximum_flight_time.value.count() <= 0 ||
              launch.minimum_flight_time.value >
                  launch.maximum_flight_time.value ||
              !IsPositive(launch.maximum_landing_speed_mps) ||
              !IsNonNegative(
                  launch.minimum_downward_impact_speed_mps) ||
              !IsPositive(
                  concrete.attitude_envelope
                      .maximum_angular_speed_radps) ||
              !IsPositive(
                  concrete.attitude_envelope
                      .maximum_angular_acceleration_radps2) ||
              !IsNonNegative(
                  concrete.attitude_envelope
                      .maximum_initial_angular_speed_radps) ||
              concrete.attitude_envelope.minimum_settle_guard.value
                      .count() <
                  0 ||
              !IsNonNegative(launch.minimum_landing_clearance_m)) {
            return Invalid(
                "safety_capability.content",
                "hopper projection hard limits are incomplete");
          }
          return SafetyProjectionLimits{
              .platform_type = PlatformType::kHopper,
              .maximum_slope_rad = terrain.maximum_slope_rad,
              .maximum_roughness_m =
                  terrain.maximum_roughness_m,
              .maximum_step_height_m = std::nullopt,
              .minimum_clearance_m =
                  std::max(
                      {terrain.minimum_overhead_clearance_m,
                       terrain.minimum_lateral_clearance_m,
                       launch.minimum_landing_clearance_m}),
              .minimum_confidence = 0.0,
              .maximum_linear_speed_mps =
                  launch.maximum_launch_speed_mps,
              .maximum_linear_deceleration_mps2 = 0.0,
              .maximum_yaw_rate_radps =
                  concrete.attitude_envelope
                      .maximum_angular_speed_radps,
              .maximum_yaw_deceleration_radps2 =
                  concrete.attitude_envelope
                      .maximum_angular_acceleration_radps2,
              .supports_safe_stop_anchor = false,
          };
        }
      },
      capability.content);
}

Result<SafeProjection> BuildSafeProjection(
    const SafeProjectionRequest& request) {
  if (!request.map) {
    return Missing("safe_projection.map",
                   "immutable map snapshot is required");
  }
  if (!request.capability) {
    return Missing("safe_projection.capability",
                   "safety capability is required");
  }
  if (!request.algorithm_config) {
    return Missing("safe_projection.algorithm_config",
                   "planner algorithm config is required");
  }
  if (CapabilityFrame(*request.capability) !=
      request.map->frame_id()) {
    return Invalid("safe_projection.capability.content.frame_id",
                   "capability and map frames must match");
  }
  auto resolved_limits =
      ResolveProjectionLimits(*request.capability);
  if (!IsOk(resolved_limits)) {
    return std::get<Error>(std::move(resolved_limits));
  }
  const auto limits =
      std::get<SafetyProjectionLimits>(std::move(resolved_limits));

  auto esdf_result = ComputeEsdf(*request.map);
  if (!IsOk(esdf_result)) {
    return std::get<Error>(std::move(esdf_result));
  }

  SafeProjection projection;
  projection.geometry = request.map->geometry();
  projection.source_map_ = request.map;
  projection.platform_type_ = limits.platform_type;
  projection.capability_ref_ = request.capability->content_ref;
  projection.known_mask.assign(request.map->KnownMask().begin(),
                               request.map->KnownMask().end());
  projection.esdf_clearance_m =
      std::get<std::vector<float>>(std::move(esdf_result));
  const std::size_t count = projection.geometry.CellCount();
  projection.hard_feasible_mask.assign(count, 0U);
  projection.analytic_time_cost_s.assign(
      count, std::numeric_limits<float>::infinity());
  projection.analytic_energy.assign(count, 0.0F);
  projection.analytic_nonfatal_risk.assign(count, 1.0F);
  projection.conservative_speed_limit_mps.assign(count, 0.0F);
  projection.safe_stop_candidate_mask.assign(count, 0U);

  const auto obstacle = request.map->HardObstacleMask();
  const auto normals = request.map->SurfaceNormals();
  const auto roughness = request.map->RoughnessMeters();
  const auto confidence = request.map->Confidence();
  const auto static_speed = request.map->StaticSpeedLimitMps();
  for (std::size_t y = 0U; y < projection.geometry.height; ++y) {
    for (std::size_t x = 0U; x < projection.geometry.width; ++x) {
      const std::size_t index = y * projection.geometry.width + x;
      bool hard_feasible =
          projection.known_mask[index] != 0U &&
          obstacle[index] == 0U && confidence[index] > 0.0F &&
          static_cast<double>(confidence[index]) +
                  kComparisonTolerance >=
              limits.minimum_confidence;
      const double slope = std::acos(std::clamp(
          static_cast<double>(normals.z[index]), -1.0, 1.0));
      hard_feasible =
          hard_feasible && std::isfinite(slope) &&
          slope <= limits.maximum_slope_rad + kComparisonTolerance &&
          slope < std::numbers::pi_v<double> / 2.0 -
                      kComparisonTolerance;
      if (limits.maximum_roughness_m.has_value()) {
        hard_feasible =
            hard_feasible &&
            static_cast<double>(roughness[index]) <=
                *limits.maximum_roughness_m + kComparisonTolerance;
      }
      if (hard_feasible &&
          limits.maximum_step_height_m.has_value()) {
        hard_feasible = PassesStepLimit(
            x, y, *request.map, *limits.maximum_step_height_m);
      }
      hard_feasible =
          hard_feasible &&
          static_cast<double>(projection.esdf_clearance_m[index]) +
                  kComparisonTolerance >=
              limits.minimum_clearance_m;

      const std::optional<float> map_speed =
          static_speed.empty()
              ? std::nullopt
              : std::optional<float>{static_speed[index]};
      const double speed = TerrainSpeed(
          limits.maximum_linear_speed_mps, slope,
          static_cast<double>(roughness[index]),
          projection.geometry.resolution_m, map_speed);
      hard_feasible =
          hard_feasible && IsPositive(speed);
      if (!hard_feasible) {
        continue;
      }

      projection.hard_feasible_mask[index] = 1U;
      projection.conservative_speed_limit_mps[index] =
          static_cast<float>(speed);
      projection.analytic_time_cost_s[index] =
          static_cast<float>(
              projection.geometry.resolution_m / speed);
      const double energy =
          projection.geometry.resolution_m *
          (1.0 + slope * slope +
           static_cast<double>(roughness[index]));
      projection.analytic_energy[index] =
          static_cast<float>(energy);
      const double slope_risk =
          limits.maximum_slope_rad > 0.0
              ? slope / limits.maximum_slope_rad
              : 0.0;
      const double roughness_risk =
          limits.maximum_roughness_m.has_value() &&
                  *limits.maximum_roughness_m > 0.0
              ? static_cast<double>(roughness[index]) /
                    *limits.maximum_roughness_m
              : 0.0;
      projection.analytic_nonfatal_risk[index] =
          static_cast<float>(
              slope_risk + roughness_risk +
              (1.0 - static_cast<double>(confidence[index])));
      projection.safe_stop_candidate_mask[index] =
          static_cast<std::uint8_t>(
              limits.supports_safe_stop_anchor);
    }
  }

  auto soft_cost = ComposeBoundedSoftCost(
      projection.analytic_energy,
      projection.analytic_nonfatal_risk, request.learned_cost,
      *request.map, *request.algorithm_config);
  if (!IsOk(soft_cost)) {
    return std::get<Error>(std::move(soft_cost));
  }
  auto resolved =
      std::get<ResolvedSoftCost>(std::move(soft_cost));
  projection.resolved_energy = std::move(resolved.energy);
  projection.resolved_nonfatal_risk =
      std::move(resolved.nonfatal_risk);
  projection.soft_cost_source = resolved.source;
  projection.soft_cost_fallback_reason_code =
      std::move(resolved.fallback_reason_code);
  LabelConnectedComponents(projection);
  return projection;
}

const std::shared_ptr<const ImmutableMapSnapshot>&
SafeProjection::source_map() const noexcept {
  return source_map_;
}

std::span<const float>
SafeProjection::ElevationMeters() const noexcept {
  return source_map_ ? source_map_->ElevationMeters()
                     : std::span<const float>{};
}

std::span<const float>
SafeProjection::RoughnessMeters() const noexcept {
  return source_map_ ? source_map_->RoughnessMeters()
                     : std::span<const float>{};
}

SurfaceNormalGridView
SafeProjection::SurfaceNormals() const noexcept {
  return source_map_ ? source_map_->SurfaceNormals()
                     : SurfaceNormalGridView{};
}

bool SafeProjection::InBounds(Cell cell) const noexcept {
  return cell.x >= 0 && cell.y >= 0 &&
         static_cast<std::size_t>(cell.x) < geometry.width &&
         static_cast<std::size_t>(cell.y) < geometry.height;
}

bool SafeProjection::Known(Cell cell) const noexcept {
  return InBounds(cell) &&
         CellIndex(geometry, cell) < known_mask.size() &&
         known_mask[CellIndex(geometry, cell)] != 0U;
}

bool SafeProjection::HardFeasible(Cell cell) const noexcept {
  return InBounds(cell) &&
         CellIndex(geometry, cell) < hard_feasible_mask.size() &&
         hard_feasible_mask[CellIndex(geometry, cell)] != 0U;
}

float SafeProjection::ClearanceMeters(Cell cell) const noexcept {
  if (!InBounds(cell) ||
      CellIndex(geometry, cell) >= esdf_clearance_m.size()) {
    return 0.0F;
  }
  return esdf_clearance_m[CellIndex(geometry, cell)];
}

PlatformType SafeProjection::platform_type() const noexcept {
  return platform_type_;
}

const ContentRef& SafeProjection::capability_ref() const noexcept {
  return capability_ref_;
}

Result<SafeStopAnchor> ResolveSafeStopAnchor(
    const SafeProjection& projection,
    const WheeledOrLeggedState& state,
    const SafetyCapabilityProfile& capability) {
  if (!projection.source_map()) {
    return Invalid("safe_stop.projection",
                   "projection must retain its immutable source map");
  }
  auto resolved_limits = ResolveProjectionLimits(capability);
  if (!IsOk(resolved_limits)) {
    return std::get<Error>(std::move(resolved_limits));
  }
  const auto limits =
      std::get<SafetyProjectionLimits>(std::move(resolved_limits));
  if (limits.platform_type != projection.platform_type() ||
      capability.content_ref != projection.capability_ref()) {
    return Invalid(
        "safe_stop.capability",
        "safe-stop capability must exactly match the projection");
  }
  if (!limits.supports_safe_stop_anchor) {
    return Invalid("safe_stop.capability",
                   "platform has no ground safe-stop contract");
  }
  if (!IsFinite(state.position_m) || !std::isfinite(state.yaw_rad) ||
      !IsFinite(state.linear_velocity_mps) ||
      !std::isfinite(state.yaw_rate_radps)) {
    return Invalid("safe_stop.state",
                   "safe-stop state values must be finite");
  }
  const double speed =
      std::hypot(state.linear_velocity_mps.x,
                 state.linear_velocity_mps.y,
                 state.linear_velocity_mps.z);
  const double yaw_rate = std::abs(state.yaw_rate_radps);
  if (speed >
          limits.maximum_linear_speed_mps + kComparisonTolerance ||
      yaw_rate >
          limits.maximum_yaw_rate_radps + kComparisonTolerance) {
    return Invalid(
        "safe_stop.state",
        "current velocity exceeds certified stopping applicability");
  }
  if (!IsPositive(limits.maximum_linear_deceleration_mps2) ||
      !IsPositive(limits.maximum_yaw_deceleration_radps2)) {
    return Invalid(
        "safe_stop.capability",
        "certified linear and yaw braking bounds are required");
  }
  const double stopping_distance =
      speed * speed /
      (2.0 * limits.maximum_linear_deceleration_mps2);
  const double yaw_stopping_angle =
      yaw_rate * yaw_rate /
      (2.0 * limits.maximum_yaw_deceleration_radps2);
  if (!std::isfinite(stopping_distance) ||
      !std::isfinite(yaw_stopping_angle)) {
    return Invalid("safe_stop.state",
                   "stopping bounds are not finite");
  }

  const auto start =
      PositionCell(projection.geometry, state.position_m);
  if (!start.has_value() || !projection.HardFeasible(*start)) {
    return Invalid("safe_stop.state.position_m",
                   "current state is not in hard-feasible terrain");
  }
  const std::size_t count = projection.geometry.CellCount();
  if (projection.safe_stop_candidate_mask.size() != count ||
      projection.hard_feasible_mask.size() != count ||
      projection.connected_component.size() != count) {
    return Invalid("safe_stop.projection",
                   "projection layer shapes are inconsistent");
  }

  constexpr std::int32_t kDx[] = {-1, 1, 0, 0};
  constexpr std::int32_t kDy[] = {0, 0, 1, -1};
  std::vector<std::size_t> distance_edges(
      count, std::numeric_limits<std::size_t>::max());
  std::queue<Cell> queue;
  const std::size_t start_index =
      CellIndex(projection.geometry, *start);
  distance_edges[start_index] = 0U;
  queue.push(*start);
  std::optional<Cell> selected;

  while (!queue.empty()) {
    const Cell current = queue.front();
    queue.pop();
    const std::size_t current_index =
        CellIndex(projection.geometry, current);
    const double available_distance =
        static_cast<double>(distance_edges[current_index]) *
        projection.geometry.resolution_m;
    if (projection.safe_stop_candidate_mask[current_index] != 0U &&
        available_distance + kComparisonTolerance >=
            stopping_distance) {
      selected = current;
      break;
    }

    for (std::size_t neighbor = 0U; neighbor < 4U; ++neighbor) {
      const Cell next{
          .x = current.x + kDx[neighbor],
          .y = current.y + kDy[neighbor],
      };
      if (!projection.HardFeasible(next)) {
        continue;
      }
      const std::size_t next_index =
          CellIndex(projection.geometry, next);
      if (distance_edges[next_index] !=
          std::numeric_limits<std::size_t>::max()) {
        continue;
      }
      distance_edges[next_index] =
          distance_edges[current_index] + 1U;
      queue.push(next);
    }
  }

  if (!selected.has_value()) {
    return Invalid(
        "safe_stop.path_distance",
        "no certified stop candidate has sufficient reachable distance");
  }
  const std::size_t selected_index =
      CellIndex(projection.geometry, *selected);
  const Vec3 position{
      .x = projection.geometry.origin_m.x +
           (static_cast<double>(selected->x) + 0.5) *
               projection.geometry.resolution_m,
      .y = projection.geometry.origin_m.y +
           (static_cast<double>(selected->y) + 0.5) *
               projection.geometry.resolution_m,
      .z = static_cast<double>(
          projection.ElevationMeters()[selected_index]),
  };
  const auto& map_ref = projection.source_map()->snapshot_ref();
  return SafeStopAnchor{
      .anchor_id =
          "safe-stop/" + map_ref.id + "/" +
          std::to_string(map_ref.revision) + "/" +
          std::to_string(selected->y) + "/" +
          std::to_string(selected->x),
      .pose =
          {
              .position_m = position,
              .yaw_rad = state.yaw_rad,
          },
      .target_linear_velocity_mps = 0.0,
      .target_yaw_rate_radps = 0.0,
      .terrain_certification_ref = map_ref,
  };
}

}  // namespace lunar::planning::v3
