#include "legged/anytime_legged_planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numbers>
#include <optional>
#include <ranges>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "shared/anytime_ara_star.hpp"
#include "shared/controlled_work.hpp"
#include "shared/edge_validation_cache.hpp"

namespace lunar::pure_planning::legged {
namespace {

constexpr double kTolerance = 1.0e-9;
constexpr double kPhysicalMatchTolerance = 1.0e-6;
constexpr double kMinimumEdgeCost = 1.0e-6;
constexpr std::size_t kMaximumSearchStates = 131072U;
constexpr std::array<double, 5U> kCostWeights{1.0, 1.0, 1.0, 1.0, 1.0};

enum class LeggedMotionMode : std::uint8_t {
  kStart,
  kForward,
  kBackward,
  kLateralLeft,
  kLateralRight,
  kSpin,
  kCoupled,
};

[[nodiscard]] LeggedMotionMode ModeFor(
    const LeggedPrimitiveKind kind) noexcept {
  switch (kind) {
    case LeggedPrimitiveKind::kForward:
      return LeggedMotionMode::kForward;
    case LeggedPrimitiveKind::kBackward:
      return LeggedMotionMode::kBackward;
    case LeggedPrimitiveKind::kLateralLeft:
      return LeggedMotionMode::kLateralLeft;
    case LeggedPrimitiveKind::kLateralRight:
      return LeggedMotionMode::kLateralRight;
    case LeggedPrimitiveKind::kSpin:
      return LeggedMotionMode::kSpin;
    case LeggedPrimitiveKind::kCoupled:
      return LeggedMotionMode::kCoupled;
  }
  return LeggedMotionMode::kStart;
}

struct StateKey final {
  std::int32_t x{};
  std::int32_t y{};
  std::int32_t yaw{};
  LeggedMotionMode mode{LeggedMotionMode::kStart};
  bool narrow{};

  bool operator==(const StateKey&) const = default;
};

struct StateKeyHash final {
  [[nodiscard]] std::size_t operator()(const StateKey& key) const noexcept {
    std::size_t value = static_cast<std::uint32_t>(key.x);
    value ^= static_cast<std::size_t>(static_cast<std::uint32_t>(key.y)) +
        0x9e3779b9U + (value << 6U) + (value >> 2U);
    value ^= static_cast<std::size_t>(static_cast<std::uint32_t>(key.yaw)) +
        0x9e3779b9U + (value << 6U) + (value >> 2U);
    value ^= static_cast<std::size_t>(key.mode) +
        0x9e3779b9U + (value << 6U) + (value >> 2U);
    value ^= static_cast<std::size_t>(key.narrow) +
        0x9e3779b9U + (value << 6U) + (value >> 2U);
    return value;
  }
};

struct EdgeKey final {
  std::size_t source{};
  std::size_t primitive{};

  bool operator==(const EdgeKey&) const = default;
};

struct EdgeKeyHash final {
  [[nodiscard]] std::size_t operator()(const EdgeKey& key) const noexcept {
    std::size_t value = key.source;
    value ^= key.primitive + 0x9e3779b9U + (value << 6U) + (value >> 2U);
    return value;
  }
};

struct SearchState final {
  LeggedPose pose;
  LeggedMotionMode mode{LeggedMotionMode::kStart};
  bool narrow{};
};

struct EvaluatedEdge final {
  bool valid{};
  bool canceled{};
  double cost{};
  double maximum_translation_step_m{};
  bool mode_changed{};
  std::size_t sweep_cell_checks{};
  std::size_t fast_path_accepts{};
  std::size_t exact_sweep_fallbacks{};
  std::size_t exact_sweep_cell_checks{};
  std::array<double, 5U> cost_components{};
  LeggedTransition transition;
};

struct TransitionRecord final {
  EdgeKey key;
  LeggedTransition transition;
  bool mode_changed{};
  std::array<double, 5U> cost_components{};
};

[[nodiscard]] LeggedPlanResult Failure(
    const LocalPlanStatus status, std::string reason_code) {
  return LeggedPlanResult{
      .status = status,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

[[nodiscard]] bool Finite(const Pose3& pose) noexcept {
  return Finite(pose.position_m) && std::isfinite(pose.orientation.w) &&
      std::isfinite(pose.orientation.x) &&
      std::isfinite(pose.orientation.y) &&
      std::isfinite(pose.orientation.z);
}

[[nodiscard]] bool ValidVelocityInterval(const Interval interval) noexcept {
  return ValidInterval(interval) && interval.lower <= 0.0 &&
      interval.upper >= 0.0 && interval.lower < interval.upper;
}

[[nodiscard]] bool ValidCapability(
    const LeggedCapability& capability) noexcept {
  if (!Finite(capability.body_extent_m) ||
      capability.body_extent_m.x <= 0.0 ||
      capability.body_extent_m.y <= 0.0 ||
      capability.body_extent_m.z <= 0.0 ||
      !std::isfinite(capability.maximum_slope_rad) ||
      capability.maximum_slope_rad < 0.0 ||
      !std::isfinite(capability.maximum_step_height_m) ||
      capability.maximum_step_height_m < 0.0 ||
      !std::isfinite(capability.minimum_body_clearance_m) ||
      capability.minimum_body_clearance_m < 0.0 ||
      !std::isfinite(capability.step_vertical_rate_mps) ||
      capability.step_vertical_rate_mps <= 0.0 ||
      !ValidInterval(capability.body_height_m) ||
      !ValidVelocityInterval(capability.forward_speed_mps) ||
      !ValidVelocityInterval(capability.lateral_speed_mps) ||
      !ValidVelocityInterval(capability.yaw_rate_radps) ||
      !std::isfinite(capability.maximum_linear_acceleration_mps2) ||
      capability.maximum_linear_acceleration_mps2 <= 0.0 ||
      !std::isfinite(capability.maximum_yaw_acceleration_radps2) ||
      capability.maximum_yaw_acceleration_radps2 <= 0.0 ||
      capability.motion_primitives.empty()) {
    return false;
  }
  return std::ranges::all_of(
      capability.motion_primitives,
      [](const LeggedBodyPrimitive& primitive) {
        return !primitive.primitive_id.empty() &&
            Finite(primitive.body_frame_displacement_m) &&
            std::isfinite(primitive.yaw_change_rad) &&
            (std::hypot(primitive.body_frame_displacement_m.x,
                        primitive.body_frame_displacement_m.y) > kTolerance ||
             std::abs(primitive.body_frame_displacement_m.z) > kTolerance ||
             std::abs(primitive.yaw_change_rad) > kTolerance);
      });
}

[[nodiscard]] bool ValidTerrain(
    const shared::LocalTerrainProjection& terrain) noexcept {
  return terrain.map != nullptr && terrain.map->cell_count() > 0U &&
      terrain.free_with_height.size() == terrain.map->cell_count() &&
      terrain.clearance_m.size() == terrain.map->cell_count() &&
      terrain.slope_rad.size() == terrain.map->cell_count() &&
      terrain.roughness_m.size() == terrain.map->cell_count() &&
      terrain.map->FloatLayer("occupancy").size() == terrain.map->cell_count() &&
      terrain.map->FloatLayer("elevation").size() == terrain.map->cell_count();
}

[[nodiscard]] double CircumscribedRadius(
    const LeggedCapability& capability) noexcept {
  return std::hypot(
      capability.body_extent_m.x / 2.0 +
          capability.minimum_body_clearance_m,
      capability.body_extent_m.y / 2.0 +
          capability.minimum_body_clearance_m);
}

[[nodiscard]] std::array<double, 5U> CostScales(
    const LeggedCapability& capability) noexcept {
  const double length = std::max(
      {capability.body_extent_m.x, capability.body_extent_m.y,
       capability.body_extent_m.z, 2.0 * CircumscribedRadius(capability)});
  const double forward_speed = std::max(
      std::abs(capability.forward_speed_mps.lower),
      std::abs(capability.forward_speed_mps.upper));
  const double lateral_speed = std::max(
      std::abs(capability.lateral_speed_mps.lower),
      std::abs(capability.lateral_speed_mps.upper));
  const double planar_speed = std::hypot(forward_speed, lateral_speed);
  return {
      length,
      length / planar_speed,
      std::max(1.0e-3, capability.maximum_slope_rad +
              capability.maximum_step_height_m / length),
      1.0,
      3.0,
  };
}

[[nodiscard]] bool NarrowAt(
    const Vec2 position, const shared::LocalTerrainProjection& terrain,
    const LeggedCapability& capability) noexcept {
  const auto cell = terrain.map->PositionToCell(position);
  if (!cell.has_value()) {
    return false;
  }
  const double clearance = terrain.clearance_m[terrain.map->Index(*cell)];
  return clearance < CircumscribedRadius(capability) +
      2.0 * terrain.map->resolution_m();
}

[[nodiscard]] std::int32_t Quantize(
    const double value, const double origin, const double resolution) noexcept {
  const double quantized = std::round((value - origin) / resolution);
  if (!std::isfinite(quantized) ||
      quantized < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
      quantized > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
    return std::numeric_limits<std::int32_t>::min();
  }
  return static_cast<std::int32_t>(quantized);
}

[[nodiscard]] std::int32_t YawBin(
    const double yaw, const std::size_t bins) noexcept {
  const double positive = NormalizeYaw(yaw) < 0.0
      ? NormalizeYaw(yaw) + 2.0 * std::numbers::pi
      : NormalizeYaw(yaw);
  const auto raw = static_cast<std::int64_t>(std::llround(
      positive * static_cast<double>(bins) / (2.0 * std::numbers::pi)));
  return static_cast<std::int32_t>(raw % static_cast<std::int64_t>(bins));
}

[[nodiscard]] StateKey KeyFor(
    const LeggedPose& pose, const LeggedMotionMode mode, const bool narrow,
    const shared::LocalTerrainProjection& terrain) noexcept {
  const double resolution = terrain.map->resolution_m() *
      (narrow ? 0.5 : 1.0);
  return StateKey{
      .x = Quantize(pose.position_m.x, terrain.map->origin_m().x, resolution),
      .y = Quantize(pose.position_m.y, terrain.map->origin_m().y, resolution),
      .yaw = YawBin(pose.yaw_rad, narrow ? 128U : 64U),
      .mode = mode,
      .narrow = narrow,
  };
}

[[nodiscard]] bool RectangleIntersectsCell(
    const Vec2 center, const double yaw, const double half_length,
    const double half_width, const shared::MapSnapshot& map,
    const shared::GridCell cell) noexcept {
  const Vec3 cell_center = map.CellCenter(cell);
  const Vec2 delta{
      .x = cell_center.x - center.x,
      .y = cell_center.y - center.y,
  };
  const Vec2 longitudinal{.x = std::cos(yaw), .y = std::sin(yaw)};
  const Vec2 lateral{.x = -longitudinal.y, .y = longitudinal.x};
  const double cell_half = map.resolution_m() / 2.0;
  const auto cell_radius = [cell_half](const Vec2 axis) {
    return cell_half * (std::abs(axis.x) + std::abs(axis.y));
  };
  if (std::abs(delta.x * longitudinal.x + delta.y * longitudinal.y) >
      half_length + cell_radius(longitudinal) + kTolerance) {
    return false;
  }
  if (std::abs(delta.x * lateral.x + delta.y * lateral.y) >
      half_width + cell_radius(lateral) + kTolerance) {
    return false;
  }
  if (std::abs(delta.x) >
      cell_half + half_length * std::abs(longitudinal.x) +
          half_width * std::abs(lateral.x) + kTolerance) {
    return false;
  }
  return std::abs(delta.y) <=
      cell_half + half_length * std::abs(longitudinal.y) +
          half_width * std::abs(lateral.y) + kTolerance;
}

[[nodiscard]] bool PoseHeightFeasible(
    const LeggedPose& pose, const shared::GridCell center_cell,
    const shared::LocalTerrainProjection& terrain,
    const LeggedCapability& capability) noexcept {
  const double elevation = terrain.map->FloatLayer("elevation")
      [terrain.map->Index(center_cell)];
  return std::isfinite(elevation) &&
      pose.position_m.z >=
          elevation + capability.body_height_m.lower - kTolerance &&
      pose.position_m.z <=
          elevation + capability.body_height_m.upper + kTolerance;
}

void AppendUnique(std::vector<shared::GridCell>& cells,
                  const shared::GridCell cell) {
  if (cells.empty() || cells.back() != cell) {
    cells.push_back(cell);
  }
}

[[nodiscard]] std::vector<shared::GridCell> Supercover(
    const shared::GridCell start, const shared::GridCell end) {
  std::vector<shared::GridCell> cells;
  std::int32_t x = start.x;
  std::int32_t y = start.y;
  const std::int32_t dx = end.x - start.x;
  const std::int32_t dy = end.y - start.y;
  const std::int32_t sign_x = dx < 0 ? -1 : 1;
  const std::int32_t sign_y = dy < 0 ? -1 : 1;
  const std::int64_t nx = std::abs(static_cast<std::int64_t>(dx));
  const std::int64_t ny = std::abs(static_cast<std::int64_t>(dy));
  std::int64_t ix{};
  std::int64_t iy{};
  AppendUnique(cells, {.x = x, .y = y});
  while (ix < nx || iy < ny) {
    const std::int64_t decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx;
    if (decision == 0) {
      AppendUnique(cells, {.x = x + sign_x, .y = y});
      AppendUnique(cells, {.x = x, .y = y + sign_y});
      x += sign_x;
      y += sign_y;
      ++ix;
      ++iy;
    } else if (decision < 0) {
      x += sign_x;
      ++ix;
    } else {
      y += sign_y;
      ++iy;
    }
    AppendUnique(cells, {.x = x, .y = y});
  }
  return cells;
}

[[nodiscard]] EvaluatedEdge SweepBody(
    const std::size_t primitive_index, const LeggedPrimitiveKind kind,
    const LeggedPose& source, const LeggedPose& target,
    const LeggedMotionMode source_mode, const LeggedMotionMode target_mode,
    const shared::LocalTerrainProjection& terrain,
    const LeggedTraversalProjection& traversal,
    const LeggedCapability& capability,
    const std::array<double, 5U>& cost_scales,
    const SearchControl& control,
    const bool pose_only = false) {
  EvaluatedEdge result;
  if (control.canceled()) {
    result.canceled = true;
    return result;
  }
  const double translation = std::hypot(
      target.position_m.x - source.position_m.x,
      target.position_m.y - source.position_m.y);
  const double yaw_delta = ShortestYawDelta(source.yaw_rad, target.yaw_rad);
  const double swept_distance = translation +
      std::abs(yaw_delta) * CircumscribedRadius(capability);
  const double maximum_sample_distance = terrain.map->resolution_m() / 4.0;
  const std::size_t subdivisions = std::max<std::size_t>(
      1U, static_cast<std::size_t>(
              std::ceil(swept_distance / maximum_sample_distance)));
  result.maximum_translation_step_m = translation /
      static_cast<double>(subdivisions);

  const double half_length = capability.body_extent_m.x / 2.0 +
      capability.minimum_body_clearance_m;
  const double half_width = capability.body_extent_m.y / 2.0 +
      capability.minimum_body_clearance_m;
  double maximum_slope = 0.0;
  double maximum_roughness = 0.0;
  double minimum_clearance = std::numeric_limits<double>::infinity();
  const double radius = std::hypot(half_length, half_width);
  const auto swept_minimum = terrain.map->PositionToCell({
      .x = std::min(source.position_m.x, target.position_m.x) - radius,
      .y = std::min(source.position_m.y, target.position_m.y) - radius,
  });
  const auto swept_maximum = terrain.map->PositionToCell({
      .x = std::max(source.position_m.x, target.position_m.x) + radius,
      .y = std::max(source.position_m.y, target.position_m.y) + radius,
  });
  const auto source_cell = terrain.map->PositionToCell(
      {.x = source.position_m.x, .y = source.position_m.y});
  const auto target_cell = terrain.map->PositionToCell(
      {.x = target.position_m.x, .y = target.position_m.y});
  if (!swept_minimum.has_value() || !swept_maximum.has_value() ||
      !source_cell.has_value() || !target_cell.has_value()) {
    return result;
  }
  const bool fast_path = traversal.AllCellsTraversable(
      *swept_minimum, *swept_maximum);
  if (fast_path) {
    result.fast_path_accepts = 1U;
    const std::vector<shared::GridCell> centerline =
        Supercover(*source_cell, *target_cell);
    for (std::size_t index = 0U; index < centerline.size(); ++index) {
      if ((index & 63U) == 0U &&
          (control.canceled() || control.expired())) {
        result.canceled = control.canceled();
        return result;
      }
      const double ratio = centerline.size() <= 1U
          ? 0.0
          : static_cast<double>(index) /
                static_cast<double>(centerline.size() - 1U);
      const LeggedPose sample_pose{
          .position_m = {
              source.position_m.x +
                  ratio * (target.position_m.x - source.position_m.x),
              source.position_m.y +
                  ratio * (target.position_m.y - source.position_m.y),
              source.position_m.z +
                  ratio * (target.position_m.z - source.position_m.z)},
          .yaw_rad = source.yaw_rad + ratio * yaw_delta,
      };
      if (!PoseHeightFeasible(
              sample_pose, centerline[index], terrain, capability)) {
        return result;
      }
      const std::size_t cell_index = terrain.map->Index(centerline[index]);
      maximum_slope = std::max(
          maximum_slope, static_cast<double>(traversal.slope_rad[cell_index]));
      const double roughness = traversal.roughness_m[cell_index];
      if (std::isfinite(roughness)) {
        maximum_roughness = std::max(maximum_roughness, roughness);
      }
      minimum_clearance = std::min(
          minimum_clearance,
          static_cast<double>(traversal.clearance_m[cell_index]));
    }
  } else {
    result.exact_sweep_fallbacks = 1U;
    for (std::size_t sample = 0U; sample <= subdivisions; ++sample) {
    if (control.canceled()) {
      result.canceled = true;
      return result;
    }
    if (control.expired()) {
      return result;
    }
    const double ratio = static_cast<double>(sample) /
        static_cast<double>(subdivisions);
    const Vec2 center{
        .x = source.position_m.x +
            ratio * (target.position_m.x - source.position_m.x),
        .y = source.position_m.y +
            ratio * (target.position_m.y - source.position_m.y),
    };
    const double yaw = source.yaw_rad + ratio * yaw_delta;
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    const double aabb_x = std::abs(cosine) * half_length +
        std::abs(sine) * half_width;
    const double aabb_y = std::abs(sine) * half_length +
        std::abs(cosine) * half_width;
    const auto minimum_cell = terrain.map->PositionToCell(
        {.x = center.x - aabb_x, .y = center.y - aabb_y});
    const auto maximum_cell = terrain.map->PositionToCell(
        {.x = center.x + aabb_x, .y = center.y + aabb_y});
    const auto center_cell = terrain.map->PositionToCell(center);
    if (!minimum_cell.has_value() || !maximum_cell.has_value() ||
        !center_cell.has_value()) {
      return result;
    }
    LeggedPose sample_pose{
        .position_m = {
            center.x,
            center.y,
            source.position_m.z +
                ratio * (target.position_m.z - source.position_m.z)},
        .yaw_rad = yaw,
    };
    if (!PoseHeightFeasible(
            sample_pose, *center_cell, terrain, capability)) {
      return result;
    }
    for (std::int32_t y = minimum_cell->y; y <= maximum_cell->y; ++y) {
      for (std::int32_t x = minimum_cell->x; x <= maximum_cell->x; ++x) {
        ++result.sweep_cell_checks;
        ++result.exact_sweep_cell_checks;
        if ((result.sweep_cell_checks & 63U) == 0U) {
          if (control.canceled()) {
            result.canceled = true;
            return result;
          }
          if (control.expired() || control.canceled()) {
            return result;
          }
        }
        const shared::GridCell cell{.x = x, .y = y};
        if (!RectangleIntersectsCell(
                center, yaw, half_length, half_width, *terrain.map, cell)) {
          continue;
        }
        const std::size_t index = terrain.map->Index(cell);
        const double slope = traversal.slope_rad[index];
        const double roughness = traversal.roughness_m[index];
        if (traversal.hard_feasible[index] == 0U ||
            traversal.step_feasible[index] == 0U) {
          return result;
        }
        maximum_slope = std::max(maximum_slope, slope);
        if (std::isfinite(roughness)) {
          maximum_roughness = std::max(maximum_roughness, roughness);
        }
        minimum_clearance = std::min(
            minimum_clearance,
            static_cast<double>(traversal.clearance_m[index]));
      }
    }
  }
  }

  const double cosine = std::cos(source.yaw_rad);
  const double sine = std::sin(source.yaw_rad);
  const double dx = target.position_m.x - source.position_m.x;
  const double dy = target.position_m.y - source.position_m.y;
  const double forward = cosine * dx + sine * dy;
  const double lateral = -sine * dx + cosine * dy;
  const auto duration = [](const double displacement, const Interval limits) {
    if (std::abs(displacement) <= kTolerance) {
      return 0.0;
    }
    const double speed = displacement > 0.0 ? limits.upper : -limits.lower;
    return speed > 0.0
        ? std::abs(displacement) / speed
        : std::numeric_limits<double>::infinity();
  };
  const double distance = std::hypot(
      translation, target.position_m.z - source.position_m.z);
  const double execution_time = std::max({
      duration(forward, capability.forward_speed_mps),
      duration(lateral, capability.lateral_speed_mps),
      duration(yaw_delta, capability.yaw_rate_radps),
      std::abs(target.position_m.z - source.position_m.z) /
          capability.step_vertical_rate_mps,
      std::sqrt(6.0 * distance /
                capability.maximum_linear_acceleration_mps2),
      std::sqrt(6.0 * std::abs(yaw_delta) /
                capability.maximum_yaw_acceleration_radps2),
  });
  const double clearance_threshold = CircumscribedRadius(capability) +
      2.0 * terrain.map->resolution_m();
  const double low_clearance = std::isfinite(minimum_clearance)
      ? std::max(0.0, clearance_threshold - minimum_clearance) /
            clearance_threshold
      : 0.0;
  const double terrain_cost = maximum_slope +
      maximum_roughness / cost_scales[0U];
  double mode_cost = std::abs(yaw_delta) / std::numbers::pi;
  if (kind == LeggedPrimitiveKind::kBackward ||
      kind == LeggedPrimitiveKind::kLateralLeft ||
      kind == LeggedPrimitiveKind::kLateralRight) {
    mode_cost += 0.1;
  }
  result.mode_changed = source_mode != target_mode;
  if (result.mode_changed) {
    mode_cost += 0.1;
  }
  result.cost_components = {
      distance,
      execution_time,
      terrain_cost,
      low_clearance,
      mode_cost,
  };
  double cost = 0.0;
  for (std::size_t component = 0U; component < cost_scales.size();
       ++component) {
    cost += kCostWeights[component] * result.cost_components[component] /
        cost_scales[component];
  }
  cost = std::max(kMinimumEdgeCost, cost);
  if (!pose_only && (!std::isfinite(cost) || cost <= 0.0)) {
    return result;
  }
  result.valid = true;
  result.cost = pose_only ? 0.0 : cost;
  result.transition = LeggedTransition{
      .source_pose = source,
      .target_pose = target,
      .target_body_z_m = {
          .lower = target.position_m.z,
          .upper = target.position_m.z,
      },
      .primitive_index = primitive_index,
      .primitive_kind = kind,
      .path_length_m = distance,
  };
  return result;
}

[[nodiscard]] std::optional<double> ConnectorScale(
    const LeggedPose& source, const LeggedBodyPrimitive& primitive,
    const PointGoal& goal, const std::optional<double> goal_yaw,
    const double yaw_tolerance) noexcept {
  const double cosine = std::cos(source.yaw_rad);
  const double sine = std::sin(source.yaw_rad);
  const double world_dx =
      cosine * primitive.body_frame_displacement_m.x -
      sine * primitive.body_frame_displacement_m.y;
  const double world_dy =
      sine * primitive.body_frame_displacement_m.x +
      cosine * primitive.body_frame_displacement_m.y;
  const double offset_x = source.position_m.x - goal.position_m.x;
  const double offset_y = source.position_m.y - goal.position_m.y;
  const double quadratic = world_dx * world_dx + world_dy * world_dy;
  const double linear = world_dx * offset_x + world_dy * offset_y;
  const double constant = offset_x * offset_x + offset_y * offset_y -
      goal.tolerance_m * goal.tolerance_m;
  double xy_lower = 0.0;
  double xy_upper = 1.0;
  double preferred_scale = 1.0;
  if (quadratic <= kTolerance) {
    if (constant > kTolerance) {
      return std::nullopt;
    }
  } else {
    const double discriminant = linear * linear - quadratic * constant;
    if (discriminant < -kTolerance) {
      return std::nullopt;
    }
    const double root = std::sqrt(std::max(0.0, discriminant));
    xy_lower = std::max(0.0, (-linear - root) / quadratic);
    xy_upper = std::min(1.0, (-linear + root) / quadratic);
    preferred_scale = std::clamp(-linear / quadratic, xy_lower, xy_upper);
  }
  xy_lower = std::max(xy_lower, kTolerance);
  if (xy_lower > xy_upper + kTolerance) {
    return std::nullopt;
  }
  if (!goal_yaw.has_value()) {
    return std::clamp(preferred_scale, xy_lower, xy_upper);
  }
  const double primitive_yaw = primitive.yaw_change_rad;
  if (std::abs(primitive_yaw) <= kTolerance) {
    return std::abs(ShortestYawDelta(source.yaw_rad, *goal_yaw)) <=
            yaw_tolerance + kTolerance
        ? std::optional<double>{
              std::clamp(preferred_scale, xy_lower, xy_upper)}
        : std::nullopt;
  }
  std::optional<double> best;
  double best_error = std::numeric_limits<double>::infinity();
  for (int turn = -2; turn <= 2; ++turn) {
    const double desired = *goal_yaw - source.yaw_rad +
        static_cast<double>(turn) * 2.0 * std::numbers::pi;
    double yaw_lower = (desired - yaw_tolerance) / primitive_yaw;
    double yaw_upper = (desired + yaw_tolerance) / primitive_yaw;
    if (yaw_lower > yaw_upper) {
      std::swap(yaw_lower, yaw_upper);
    }
    const double lower = std::max(xy_lower, yaw_lower);
    const double upper = std::min(xy_upper, yaw_upper);
    if (lower > upper + kTolerance) {
      continue;
    }
    const double candidate = std::clamp(preferred_scale, lower, upper);
    const double error = std::abs(candidate - preferred_scale);
    if (!best.has_value() || error < best_error) {
      best = candidate;
      best_error = error;
    }
  }
  return best;
}

[[nodiscard]] double BodyHeightAt(
    const Vec2 position, const shared::LocalTerrainProjection& terrain,
    const LeggedCapability& capability) noexcept {
  const auto cell = terrain.map->PositionToCell(position);
  if (!cell.has_value()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double elevation = terrain.map->FloatLayer("elevation")
      [terrain.map->Index(*cell)];
  return elevation + 0.5 *
      (capability.body_height_m.lower + capability.body_height_m.upper);
}

[[nodiscard]] bool PhysicalPoseMatches(
    const LeggedPose& lhs, const LeggedPose& rhs) noexcept {
  return std::hypot(lhs.position_m.x - rhs.position_m.x,
                    lhs.position_m.y - rhs.position_m.y) <=
          kPhysicalMatchTolerance &&
      std::abs(lhs.position_m.z - rhs.position_m.z) <=
          kPhysicalMatchTolerance &&
      std::abs(ShortestYawDelta(lhs.yaw_rad, rhs.yaw_rad)) <=
          kPhysicalMatchTolerance;
}

class LeggedSearchGraph final {
 public:
  LeggedSearchGraph(const LeggedPlanRequest& request, const LeggedPose start,
                    const PointGoal goal_region,
                    const std::optional<double> goal_yaw)
      : request_(request), terrain_(*request.terrain),
        traversal_(*request.traversal),
        capability_(*request.capability), goal_region_(goal_region),
        goal_yaw_(goal_yaw), cost_scales_(CostScales(capability_)) {
    const std::size_t cell_count = terrain_.map->cell_count();
    assignable_state_limit_ = kMaximumSearchStates;
    goal_state_ = assignable_state_limit_;
    state_count_ = assignable_state_limit_ + 1U;
    const std::size_t reserve_hint =
        cell_count > assignable_state_limit_ / 2U
            ? assignable_state_limit_
            : cell_count * 2U + 1U;
    states_.reserve(std::min(assignable_state_limit_, reserve_hint));
    state_ids_.reserve(states_.capacity());
    ordered_primitives_.resize(capability_.motion_primitives.size());
    for (std::size_t index = 0U; index < ordered_primitives_.size(); ++index) {
      ordered_primitives_[index] = index;
    }
    std::stable_sort(
        ordered_primitives_.begin(), ordered_primitives_.end(),
        [&](const std::size_t lhs, const std::size_t rhs) {
          return std::tie(capability_.motion_primitives[lhs].primitive_id,
                          lhs) <
              std::tie(capability_.motion_primitives[rhs].primitive_id, rhs);
        });
    const StateKey start_key = QuantizeState(start, LeggedMotionMode::kStart);
    states_.push_back(SearchState{
        .pose = start,
        .mode = LeggedMotionMode::kStart,
        .narrow = start_key.narrow,
    });
    state_ids_.emplace(start_key, 0U);
  }

  [[nodiscard]] std::size_t state_count() const noexcept {
    return state_count_;
  }

  [[nodiscard]] bool resource_exhausted() const noexcept {
    return resource_exhausted_;
  }

  [[nodiscard]] bool used_narrow_resolution() const noexcept {
    return used_narrow_resolution_;
  }

  [[nodiscard]] std::size_t validation_count() const noexcept {
    return validation_cache_.evaluation_count();
  }

  [[nodiscard]] std::size_t validation_cache_hits() const noexcept {
    return validation_requests_ - validation_cache_.evaluation_count();
  }

  [[nodiscard]] std::size_t maximum_edge_sweep_evaluations() const noexcept {
    return maximum_edge_sweep_evaluations_;
  }

  [[nodiscard]] std::size_t sweep_cell_checks() const noexcept {
    return sweep_cell_checks_;
  }

  [[nodiscard]] std::size_t fast_path_accepts() const noexcept {
    return fast_path_accepts_;
  }

  [[nodiscard]] std::size_t exact_sweep_fallbacks() const noexcept {
    return exact_sweep_fallbacks_;
  }

  [[nodiscard]] std::size_t exact_sweep_cell_checks() const noexcept {
    return exact_sweep_cell_checks_;
  }

  [[nodiscard]] double maximum_sweep_step() const noexcept {
    return maximum_sweep_step_;
  }

  [[nodiscard]] std::size_t quantized_endpoint_aliases() const noexcept {
    return quantized_endpoint_aliases_;
  }

  [[nodiscard]] std::size_t quantized_state_count() const noexcept {
    return state_ids_.size();
  }

  [[nodiscard]] const std::array<double, 5U>& cost_scales() const noexcept {
    return cost_scales_;
  }

  [[nodiscard]] const TransitionRecord* TransitionFor(
      const std::size_t stable_index) {
    const auto found = transitions_.find(stable_index);
    if (found == transitions_.end()) {
      return nullptr;
    }
    static_cast<void>(CachedEvaluation(
        found->second.key, [] { return EvaluatedEdge{}; }));
    return &found->second;
  }

  void Expand(const std::size_t state,
              std::vector<shared::GraphEdge>& edges) {
    if (state >= states_.size() || state == goal_state_ ||
        request_.control.canceled() || request_.control.expired()) {
      return;
    }
    const SearchState source = states_[state];
    for (const std::size_t primitive_index : ordered_primitives_) {
      if (request_.control.canceled() || request_.control.expired()) {
        return;
      }
      const LeggedBodyPrimitive& primitive =
          capability_.motion_primitives[primitive_index];
      const LeggedMotionMode target_mode = ModeFor(primitive.kind);
      const auto target_pose = ApplyPrimitive(source.pose, primitive);
      if (!target_pose.has_value()) {
        continue;
      }
      if (PoseSatisfiesGoal(*target_pose)) {
        const EdgeKey key{.source = state, .primitive = primitive_index};
        const EvaluatedEdge& evaluated = CachedEvaluation(
            key, [&] {
              EvaluatedEdge value = SweepBody(
                  primitive_index, primitive.kind, source.pose, *target_pose,
                  source.mode, target_mode, terrain_, traversal_, capability_,
                  cost_scales_, request_.control);
              maximum_sweep_step_ = std::max(
                  maximum_sweep_step_, value.maximum_translation_step_m);
              return value;
            });
        if (!evaluated.valid) {
          continue;
        }
        const std::size_t stable_index =
            StableEdgeIndex(state, primitive_index);
        transitions_[stable_index] = TransitionRecord{
            .key = key,
            .transition = evaluated.transition,
            .mode_changed = evaluated.mode_changed,
            .cost_components = evaluated.cost_components,
        };
        edges.push_back(shared::GraphEdge{
            .target_state = goal_state_,
            .cost = evaluated.cost,
            .stable_index = stable_index,
        });
        continue;
      }
      const StateKey target_key = QuantizeState(*target_pose, target_mode);
      const auto target_state = Intern(target_key, *target_pose, target_mode);
      if (!target_state.has_value() || *target_state == state) {
        continue;
      }
      const LeggedPose canonical_target = states_[*target_state].pose;
      const EdgeKey key{.source = state, .primitive = primitive_index};
      const EvaluatedEdge& evaluated = CachedEvaluation(
          key, [&] {
            EvaluatedEdge value = SweepBody(
                primitive_index, primitive.kind, source.pose,
                canonical_target, source.mode, target_mode, terrain_,
                traversal_, capability_, cost_scales_, request_.control);
            maximum_sweep_step_ = std::max(
                maximum_sweep_step_, value.maximum_translation_step_m);
            return value;
          });
      if (!evaluated.valid) {
        continue;
      }
      const std::size_t stable_index =
          StableEdgeIndex(state, primitive_index);
      const std::size_t edge_target = PoseSatisfiesGoal(canonical_target)
          ? goal_state_
          : *target_state;
      transitions_[stable_index] = TransitionRecord{
          .key = key,
          .transition = evaluated.transition,
          .mode_changed = evaluated.mode_changed,
          .cost_components = evaluated.cost_components,
      };
      edges.push_back(shared::GraphEdge{
          .target_state = edge_target,
          .cost = evaluated.cost,
          .stable_index = stable_index,
      });
    }
    AppendExactGoalEdge(state, source, edges);
  }

  [[nodiscard]] double Heuristic(const std::size_t state) const noexcept {
    if (state == goal_state_) {
      return 0.0;
    }
    if (state >= states_.size()) {
      return 0.0;
    }
    return std::max(
        0.0,
        std::hypot(
            goal_region_.position_m.x - states_[state].pose.position_m.x,
            goal_region_.position_m.y - states_[state].pose.position_m.y) -
            goal_region_.tolerance_m);
  }

  [[nodiscard]] bool IsGoal(const std::size_t state) const noexcept {
    return state == goal_state_;
  }

 private:
  template <class EvaluateFn>
  const EvaluatedEdge& CachedEvaluation(
      const EdgeKey& key, EvaluateFn&& evaluate) {
    ++validation_requests_;
    return validation_cache_.GetOrEvaluate(
        key, [&] {
          std::size_t& evaluations = edge_sweep_evaluations_[key];
          ++evaluations;
          maximum_edge_sweep_evaluations_ = std::max(
              maximum_edge_sweep_evaluations_, evaluations);
          EvaluatedEdge result = evaluate();
          sweep_cell_checks_ += result.sweep_cell_checks;
          fast_path_accepts_ += result.fast_path_accepts;
          exact_sweep_fallbacks_ += result.exact_sweep_fallbacks;
          exact_sweep_cell_checks_ += result.exact_sweep_cell_checks;
          return result;
        });
  }

  [[nodiscard]] StateKey QuantizeState(
      const LeggedPose& pose, const LeggedMotionMode mode) {
    const bool narrow = NarrowAt(
        {.x = pose.position_m.x, .y = pose.position_m.y},
        terrain_, capability_);
    used_narrow_resolution_ = used_narrow_resolution_ || narrow;
    return KeyFor(pose, mode, narrow, terrain_);
  }

  [[nodiscard]] std::optional<std::size_t> Intern(
      const StateKey& key, const LeggedPose pose,
      const LeggedMotionMode mode) {
    if (key.x == std::numeric_limits<std::int32_t>::min() ||
        key.y == std::numeric_limits<std::int32_t>::min()) {
      return std::nullopt;
    }
    if (const auto found = state_ids_.find(key); found != state_ids_.end()) {
      if (!PhysicalPoseMatches(states_[found->second].pose, pose) ||
          states_[found->second].mode != mode) {
        ++quantized_endpoint_aliases_;
        return std::nullopt;
      }
      return found->second;
    }
    if (states_.size() >= assignable_state_limit_) {
      resource_exhausted_ = true;
      return std::nullopt;
    }
    const std::size_t state = states_.size();
    states_.push_back(SearchState{
        .pose = pose,
        .mode = mode,
        .narrow = key.narrow,
    });
    state_ids_.emplace(key, state);
    return state;
  }

  [[nodiscard]] std::optional<LeggedPose> ApplyPrimitive(
      const LeggedPose source,
      const LeggedBodyPrimitive& primitive) const noexcept {
    const double cosine = std::cos(source.yaw_rad);
    const double sine = std::sin(source.yaw_rad);
    LeggedPose target{
        .position_m = {
            .x = source.position_m.x +
                cosine * primitive.body_frame_displacement_m.x -
                sine * primitive.body_frame_displacement_m.y,
            .y = source.position_m.y +
                sine * primitive.body_frame_displacement_m.x +
                cosine * primitive.body_frame_displacement_m.y,
            .z = source.position_m.z,
        },
        .yaw_rad = NormalizeYaw(source.yaw_rad + primitive.yaw_change_rad),
    };
    if (!terrain_.map->PositionToCell(
            {.x = target.position_m.x, .y = target.position_m.y})
             .has_value()) {
      return std::nullopt;
    }
    target.position_m.z = BodyHeightAt(
        {.x = target.position_m.x, .y = target.position_m.y},
        terrain_, capability_);
    return target;
  }

  [[nodiscard]] bool PoseSatisfiesGoal(
      const LeggedPose& pose) const noexcept {
    return std::hypot(
               pose.position_m.x - goal_region_.position_m.x,
               pose.position_m.y - goal_region_.position_m.y) <=
            goal_region_.tolerance_m + kTolerance &&
        (!goal_yaw_.has_value() ||
         std::abs(ShortestYawDelta(pose.yaw_rad, *goal_yaw_)) <=
             request_.goal_odom.yaw_tolerance_rad + kTolerance);
  }

  void AppendExactGoalEdge(const std::size_t state,
                           const SearchState& source,
                           std::vector<shared::GraphEdge>& edges) {
    for (const std::size_t primitive_index : ordered_primitives_) {
      const LeggedBodyPrimitive& primitive =
          capability_.motion_primitives[primitive_index];
      const auto scale = ConnectorScale(
          source.pose, primitive, goal_region_, goal_yaw_,
          request_.goal_odom.yaw_tolerance_rad);
      if (!scale.has_value()) {
        continue;
      }
      const double cosine = std::cos(source.pose.yaw_rad);
      const double sine = std::sin(source.pose.yaw_rad);
      LeggedPose target{
          .position_m = {
              .x = source.pose.position_m.x + *scale *
                  (cosine * primitive.body_frame_displacement_m.x -
                   sine * primitive.body_frame_displacement_m.y),
              .y = source.pose.position_m.y + *scale *
                  (sine * primitive.body_frame_displacement_m.x +
                   cosine * primitive.body_frame_displacement_m.y),
              .z = source.pose.position_m.z,
          },
          .yaw_rad = NormalizeYaw(
              source.pose.yaw_rad + *scale * primitive.yaw_change_rad),
      };
      target.position_m.z = BodyHeightAt(
          {.x = target.position_m.x, .y = target.position_m.y},
          terrain_, capability_);
      if (!PoseSatisfiesGoal(target)) {
        continue;
      }
      const std::size_t connector_slot =
          capability_.motion_primitives.size() + primitive_index;
      const LeggedMotionMode target_mode = ModeFor(primitive.kind);
      const EdgeKey key{.source = state, .primitive = connector_slot};
      const EvaluatedEdge& evaluated = CachedEvaluation(
          key, [&] {
            EvaluatedEdge value = SweepBody(
                primitive_index, primitive.kind, source.pose, target,
                source.mode, target_mode, terrain_, traversal_, capability_,
                cost_scales_, request_.control);
            maximum_sweep_step_ = std::max(
                maximum_sweep_step_, value.maximum_translation_step_m);
            return value;
          });
      if (!evaluated.valid) {
        continue;
      }
      const std::size_t stable_index =
          StableEdgeIndex(state, connector_slot);
      transitions_[stable_index] = TransitionRecord{
          .key = key,
          .transition = evaluated.transition,
          .mode_changed = evaluated.mode_changed,
          .cost_components = evaluated.cost_components,
      };
      edges.push_back(shared::GraphEdge{
          .target_state = goal_state_,
          .cost = evaluated.cost,
          .stable_index = stable_index,
      });
    }
  }

  [[nodiscard]] static std::size_t StableEdgeIndex(
      const std::size_t state, const std::size_t primitive) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::uint64_t value :
         {static_cast<std::uint64_t>(state),
          static_cast<std::uint64_t>(primitive)}) {
      for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
        hash ^= value & 0xffU;
        hash *= 1099511628211ULL;
        value >>= 8U;
      }
    }
    return static_cast<std::size_t>(hash);
  }

  const LeggedPlanRequest& request_;
  const shared::LocalTerrainProjection& terrain_;
  const LeggedTraversalProjection& traversal_;
  const LeggedCapability& capability_;
  PointGoal goal_region_;
  std::optional<double> goal_yaw_;
  std::array<double, 5U> cost_scales_;
  std::size_t assignable_state_limit_{};
  std::size_t goal_state_{};
  std::size_t state_count_{};
  bool resource_exhausted_{};
  bool used_narrow_resolution_{};
  std::size_t validation_requests_{};
  std::size_t maximum_edge_sweep_evaluations_{};
  std::size_t sweep_cell_checks_{};
  std::size_t fast_path_accepts_{};
  std::size_t exact_sweep_fallbacks_{};
  std::size_t exact_sweep_cell_checks_{};
  std::size_t quantized_endpoint_aliases_{};
  double maximum_sweep_step_{};
  std::vector<std::size_t> ordered_primitives_;
  std::vector<SearchState> states_;
  std::unordered_map<StateKey, std::size_t, StateKeyHash> state_ids_;
  shared::EdgeValidationCache<EdgeKey, EvaluatedEdge, EdgeKeyHash>
      validation_cache_;
  std::unordered_map<EdgeKey, std::size_t, EdgeKeyHash>
      edge_sweep_evaluations_;
  std::unordered_map<std::size_t, TransitionRecord> transitions_;
};

}  // namespace

LeggedPlanResult PlanLegged(const LeggedPlanRequest& request) try {
  if (request.control.canceled()) {
    return Failure(LocalPlanStatus::kCanceled, "REQUEST_CANCELED");
  }
  if (request.control.expired()) {
    return Failure(LocalPlanStatus::kTimedOut, "TIMEOUT");
  }
  const auto* point = std::get_if<PointGoal>(&request.goal_odom.target);
  const auto start_yaw = YawFromQuaternion(request.start.body_pose.orientation);
  const std::optional<double> goal_yaw =
      request.goal_odom.yaw_rad.has_value() &&
              std::isfinite(*request.goal_odom.yaw_rad)
          ? std::optional<double>{NormalizeYaw(*request.goal_odom.yaw_rad)}
          : std::nullopt;
  if (request.terrain == nullptr || request.traversal == nullptr ||
      request.capability == nullptr ||
      request.traversal->terrain.get() != request.terrain ||
      !ValidTerrain(*request.terrain) ||
      !ValidCapability(*request.capability) || point == nullptr ||
      !Finite(request.start.body_pose) ||
      !std::isfinite(point->position_m.x) ||
      !std::isfinite(point->position_m.y) ||
      !std::isfinite(point->tolerance_m) || point->tolerance_m < 0.0 ||
      (request.goal_odom.yaw_rad.has_value() &&
       !std::isfinite(*request.goal_odom.yaw_rad)) ||
      !std::isfinite(request.goal_odom.yaw_tolerance_rad) ||
      request.goal_odom.yaw_tolerance_rad < 0.0 ||
      !start_yaw.has_value()) {
    return Failure(LocalPlanStatus::kInvalidInput, "LEGGED_REQUEST_INVALID");
  }
  const shared::LocalTerrainProjection& terrain = *request.terrain;
  const LeggedCapability& capability = *request.capability;
  const auto goal_cell = terrain.map->PositionToCell(
      {.x = point->position_m.x, .y = point->position_m.y});
  const auto start_cell = terrain.map->PositionToCell(
      {.x = request.start.body_pose.position_m.x,
       .y = request.start.body_pose.position_m.y});
  if (!goal_cell.has_value() || !start_cell.has_value()) {
    return Failure(LocalPlanStatus::kNoPath, "LEGGED_GOAL_OUTSIDE_LOCAL_MAP");
  }

  LeggedPose start{
      .position_m = request.start.body_pose.position_m,
      .yaw_rad = *start_yaw,
  };

  const bool position_satisfied = std::hypot(
      start.position_m.x - point->position_m.x,
      start.position_m.y - point->position_m.y) <=
      point->tolerance_m + kTolerance;
  const bool yaw_satisfied = !request.goal_odom.yaw_rad.has_value() ||
      std::abs(ShortestYawDelta(start.yaw_rad, *goal_yaw)) <=
          request.goal_odom.yaw_tolerance_rad + kTolerance;
  if (position_satisfied && yaw_satisfied) {
    const EvaluatedEdge pose = SweepBody(
        capability.motion_primitives.size() + 1U,
        LeggedPrimitiveKind::kForward, start, start,
        LeggedMotionMode::kStart, LeggedMotionMode::kStart, terrain,
        *request.traversal, capability, CostScales(capability),
        request.control, true);
    if (!pose.valid) {
      if (request.control.canceled()) {
        return Failure(LocalPlanStatus::kCanceled, "REQUEST_CANCELED");
      }
      if (request.control.expired()) {
        return Failure(LocalPlanStatus::kTimedOut, "TIMEOUT");
      }
      return Failure(LocalPlanStatus::kNoPath, "LEGGED_START_INFEASIBLE");
    }
    const bool narrow = NarrowAt(
        {.x = start.position_m.x, .y = start.position_m.y},
        terrain, capability);
    return LeggedPlanResult{
        .status = LocalPlanStatus::kSolved,
        .reason_code = "LEGGED_PLAN_SOLVED",
        .trajectory = {},
        .cost = 0.0,
        .metrics = LocalPlanMetrics{
            .expanded_states = 0U,
            .edge_validation_evaluations = 1U,
            .used_narrow_resolution = narrow,
        },
        .fast_path_accepts = pose.fast_path_accepts,
        .exact_sweep_fallbacks = pose.exact_sweep_fallbacks,
        .exact_sweep_cell_checks = pose.exact_sweep_cell_checks,
        .finest_xy_key_resolution_m = terrain.map->resolution_m() *
            (narrow ? 0.5 : 1.0),
        .maximum_yaw_bin_count = narrow ? 128U : 64U,
        .maximum_sweep_translation_step_m = 0.0,
        .cost_scales = CostScales(capability),
    };
  }

  LeggedSearchGraph graph{request, start, *point, goal_yaw};
  shared::anytime::AraStarProblem problem{
      .state_count = graph.state_count(),
      .start_state = 0U,
      .expand = [&](const std::size_t state,
                    const double,
                    std::vector<shared::GraphEdge>& edges) {
        graph.Expand(state, edges);
      },
      .heuristic = [&](const std::size_t state) {
        return graph.Heuristic(state);
      },
      .is_goal = [&](const std::size_t state) {
        return graph.IsGoal(state);
      },
      .config = request.search,
      .control = request.control,
  };
  const shared::anytime::AraStarResult search =
      shared::anytime::SearchAnytimeAraStar(problem);
  const auto make_result = [&](const LocalPlanStatus status,
                               std::string reason_code) {
    LeggedPlanResult result = Failure(status, std::move(reason_code));
    result.metrics.expanded_states = search.expanded_states;
    result.metrics.edge_validation_evaluations =
        graph.validation_count();
    result.metrics.used_narrow_resolution =
        graph.used_narrow_resolution();
    result.edge_validation_cache_hits = graph.validation_cache_hits();
    result.maximum_edge_sweep_evaluations =
        graph.maximum_edge_sweep_evaluations();
    result.sweep_cell_checks = graph.sweep_cell_checks();
    result.fast_path_accepts = graph.fast_path_accepts();
    result.exact_sweep_fallbacks = graph.exact_sweep_fallbacks();
    result.exact_sweep_cell_checks = graph.exact_sweep_cell_checks();
    result.quantized_endpoint_aliases =
        graph.quantized_endpoint_aliases();
    result.quantized_state_count = graph.quantized_state_count();
    result.finest_xy_key_resolution_m = terrain.map->resolution_m() *
        (graph.used_narrow_resolution() ? 0.5 : 1.0);
    result.maximum_yaw_bin_count =
        graph.used_narrow_resolution() ? 128U : 64U;
    result.maximum_sweep_translation_step_m = graph.maximum_sweep_step();
    result.cost_scales = graph.cost_scales();
    return result;
  };
  switch (search.status) {
    case shared::anytime::AraStarStatus::kCanceled:
      return make_result(LocalPlanStatus::kCanceled, "REQUEST_CANCELED");
    case shared::anytime::AraStarStatus::kTimedOut:
      return make_result(LocalPlanStatus::kTimedOut, "TIMEOUT");
    case shared::anytime::AraStarStatus::kNoPath:
      return make_result(
          graph.resource_exhausted() ? LocalPlanStatus::kPlannerError
                                     : LocalPlanStatus::kNoPath,
          graph.resource_exhausted() ? "LEGGED_SEARCH_CAPACITY_EXHAUSTED"
                                     : "LEGGED_NO_PATH");
    case shared::anytime::AraStarStatus::kInvalidProblem:
      return make_result(LocalPlanStatus::kInvalidInput, search.reason_code);
    case shared::anytime::AraStarStatus::kResourceExhausted:
      return make_result(LocalPlanStatus::kPlannerError,
                         "LEGGED_RESOURCE_EXHAUSTED");
    case shared::anytime::AraStarStatus::kSolved:
      break;
  }
  if (search.candidates.empty()) {
    return make_result(LocalPlanStatus::kPlannerError,
                       "LEGGED_SEARCH_RESULT_INVALID");
  }
  const shared::SearchCandidate& best = *std::ranges::min_element(
      search.candidates, {}, &shared::SearchCandidate::cost);
  const auto control_failure = [&]() -> std::optional<LeggedPlanResult> {
    const auto stopped = shared::StopReason(request.control);
    if (!stopped.has_value()) {
      return std::nullopt;
    }
    return make_result(
        *stopped == "REQUEST_CANCELED" ? LocalPlanStatus::kCanceled
                                        : LocalPlanStatus::kTimedOut,
        std::string{*stopped});
  };
  if (const auto stopped = control_failure(); stopped.has_value()) {
    return *stopped;
  }
  LeggedPlanResult result = make_result(
      LocalPlanStatus::kSolved, "LEGGED_PLAN_SOLVED");
  result.cost = best.cost;
  result.trajectory.reserve(best.stable_edge_indices.size());
  if (const auto stopped = control_failure(); stopped.has_value()) {
    return *stopped;
  }
  for (const std::size_t stable_index : best.stable_edge_indices) {
    if (const auto stopped = control_failure(); stopped.has_value()) {
      return *stopped;
    }
    const TransitionRecord* const transition =
        graph.TransitionFor(stable_index);
    if (transition == nullptr) {
      return make_result(LocalPlanStatus::kPlannerError,
                         "LEGGED_SEARCH_RESULT_INVALID");
    }
    result.trajectory.push_back(transition->transition);
    result.mode_change_edge_count +=
        static_cast<std::size_t>(transition->mode_changed);
    for (std::size_t component = 0U;
         component < result.cost_components.size(); ++component) {
      result.cost_components[component] +=
          transition->cost_components[component];
    }
  }
  result.edge_validation_cache_hits = graph.validation_cache_hits();
  result.maximum_edge_sweep_evaluations =
      graph.maximum_edge_sweep_evaluations();
  result.sweep_cell_checks = graph.sweep_cell_checks();
  result.fast_path_accepts = graph.fast_path_accepts();
  result.exact_sweep_fallbacks = graph.exact_sweep_fallbacks();
  result.exact_sweep_cell_checks = graph.exact_sweep_cell_checks();
  if (const auto stopped = control_failure(); stopped.has_value()) {
    return *stopped;
  }
  return result;
} catch (const std::bad_alloc&) {
  return Failure(LocalPlanStatus::kPlannerError, "LEGGED_RESOURCE_EXHAUSTED");
} catch (...) {
  return Failure(LocalPlanStatus::kPlannerError, "LEGGED_PLANNER_ERROR");
}

}  // namespace lunar::pure_planning::legged
