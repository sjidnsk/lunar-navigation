#include "wheel/anytime_wheel_planner.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shared/anytime_ara_star.hpp"
#include "shared/active_planner_cache.hpp"
#include "shared/controlled_work.hpp"
#include "shared/edge_validation_cache.hpp"
#include "shared/goal_distance_field.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::pure_planning::wheel {
namespace {

constexpr double kTolerance = 1.0e-9;
constexpr double kPhysicalMatchTolerance = 1.0e-6;
constexpr double kMinimumEdgeCost = 1.0e-6;
constexpr std::size_t kModeCount = 3U;
constexpr std::size_t kMaximumActiveLabelsPerKey = 4U;
constexpr std::size_t kMaximumPreferredTemplateEdges = 8192U;
constexpr std::size_t kMaximumPreferredTemplatesEvaluated = 8U;
constexpr std::array<double, 5U> kCostWeights{1.0, 1.0, 1.0, 1.0, 1.0};

struct CellWindow final {
  std::int64_t minimum_x{};
  std::int64_t minimum_y{};
  std::int64_t maximum_x{};
  std::int64_t maximum_y{};
};

[[nodiscard]] bool FootprintWithinClosedMapExtent(
    const shared::MapSnapshot& map, const double minimum_x,
    const double minimum_y, const double maximum_x,
    const double maximum_y) noexcept {
  const double map_minimum_x = map.origin_m().x;
  const double map_minimum_y = map.origin_m().y;
  const double map_maximum_x =
      map_minimum_x + static_cast<double>(map.width()) * map.resolution_m();
  const double map_maximum_y =
      map_minimum_y + static_cast<double>(map.height()) * map.resolution_m();
  return std::isfinite(minimum_x) && std::isfinite(minimum_y) &&
         std::isfinite(maximum_x) && std::isfinite(maximum_y) &&
         minimum_x <= maximum_x && minimum_y <= maximum_y &&
         minimum_x >= map_minimum_x - kTolerance &&
         minimum_y >= map_minimum_y - kTolerance &&
         maximum_x <= map_maximum_x + kTolerance &&
         maximum_y <= map_maximum_y + kTolerance;
}

[[nodiscard]] CellWindow ClipOccupiedWindow(
    const shared::MapSnapshot& map, const double minimum_x,
    const double minimum_y, const double maximum_x,
    const double maximum_y) noexcept {
  const auto clip = [&](const double coordinate, const double origin,
                        const std::size_t extent) {
    const double cell = std::floor((coordinate - origin) / map.resolution_m());
    return static_cast<std::int64_t>(std::clamp(
        cell, 0.0, static_cast<double>(extent - 1U)));
  };
  return CellWindow{
      .minimum_x = clip(minimum_x, map.origin_m().x, map.width()),
      .minimum_y = clip(minimum_y, map.origin_m().y, map.height()),
      .maximum_x = clip(maximum_x, map.origin_m().x, map.width()),
      .maximum_y = clip(maximum_y, map.origin_m().y, map.height()),
  };
}

[[nodiscard]] CellWindow ClosedFootprintContactWindow(
    const shared::MapSnapshot& map, const double minimum_x,
    const double minimum_y, const double maximum_x,
    const double maximum_y) noexcept {
  const auto clip = [](const double cell, const std::size_t extent) {
    return static_cast<std::int64_t>(std::clamp(
        cell, 0.0, static_cast<double>(extent - 1U)));
  };
  const auto minimum_cell = [&](const double coordinate,
                                const double origin,
                                const std::size_t extent) {
    const double relative = (coordinate - origin) / map.resolution_m();
    return clip(std::ceil(relative) - 1.0, extent);
  };
  const auto maximum_cell = [&](const double coordinate,
                                const double origin,
                                const std::size_t extent) {
    const double relative = (coordinate - origin) / map.resolution_m();
    return clip(std::floor(relative), extent);
  };
  return CellWindow{
      .minimum_x = minimum_cell(minimum_x, map.origin_m().x, map.width()),
      .minimum_y = minimum_cell(minimum_y, map.origin_m().y, map.height()),
      .maximum_x = maximum_cell(maximum_x, map.origin_m().x, map.width()),
      .maximum_y = maximum_cell(maximum_y, map.origin_m().y, map.height()),
  };
}

struct BilinearSupportNeighborhood final {
  std::array<std::size_t, 4U> indices{};
  std::array<double, 4U> weights{};
  std::size_t count{};
  double total_weight{};
};

[[nodiscard]] std::optional<BilinearSupportNeighborhood>
KnownFreeBilinearSupportNeighborhood(
    const shared::MapSnapshot& map,
    const shared::LocalTerrainProjection& terrain,
    const std::span<const float> elevations,
    const Vec2 position_m) noexcept {
  if (!std::isfinite(position_m.x) || !std::isfinite(position_m.y)) {
    return std::nullopt;
  }
  const double sample_x =
      (position_m.x - map.origin_m().x) / map.resolution_m() - 0.5;
  const double sample_y =
      (position_m.y - map.origin_m().y) / map.resolution_m() - 0.5;
  if (!std::isfinite(sample_x) || !std::isfinite(sample_y) ||
      sample_x < static_cast<double>(
                     std::numeric_limits<std::int32_t>::min()) ||
      sample_x > static_cast<double>(
                     std::numeric_limits<std::int32_t>::max()) ||
      sample_y < static_cast<double>(
                     std::numeric_limits<std::int32_t>::min()) ||
      sample_y > static_cast<double>(
                     std::numeric_limits<std::int32_t>::max())) {
    return std::nullopt;
  }
  const auto x0 = static_cast<std::int64_t>(std::floor(sample_x));
  const auto y0 = static_cast<std::int64_t>(std::floor(sample_y));
  const double fraction_x = sample_x - static_cast<double>(x0);
  const double fraction_y = sample_y - static_cast<double>(y0);
  if (elevations.size() != map.cell_count()) {
    return std::nullopt;
  }
  BilinearSupportNeighborhood neighborhood;
  for (std::int64_t dy = 0; dy <= 1; ++dy) {
    const double weight_y = dy == 0 ? 1.0 - fraction_y : fraction_y;
    for (std::int64_t dx = 0; dx <= 1; ++dx) {
      const double weight_x = dx == 0 ? 1.0 - fraction_x : fraction_x;
      if (weight_x * weight_y <= 1.0e-15) {
        continue;
      }
      const std::int64_t x = x0 + dx;
      const std::int64_t y = y0 + dy;
      if (x < 0 || y < 0 ||
          x > static_cast<std::int64_t>(
                  std::numeric_limits<std::int32_t>::max()) ||
          y > static_cast<std::int64_t>(
                  std::numeric_limits<std::int32_t>::max())) {
        return std::nullopt;
      }
      const shared::GridCell cell{
          .x = static_cast<std::int32_t>(x),
          .y = static_cast<std::int32_t>(y),
      };
      if (!map.InBounds(cell)) {
        return std::nullopt;
      }
      const std::size_t index = map.Index(cell);
      if (index >= terrain.free_with_height.size() ||
          terrain.free_with_height[index] != 1U ||
          !std::isfinite(elevations[index])) {
        return std::nullopt;
      }
      const double weight = weight_x * weight_y;
      neighborhood.indices[neighborhood.count] = index;
      neighborhood.weights[neighborhood.count] = weight;
      ++neighborhood.count;
      neighborhood.total_weight += weight;
    }
  }
  if (neighborhood.count == 0U || neighborhood.total_weight <= 0.0 ||
      !std::isfinite(neighborhood.total_weight)) {
    return std::nullopt;
  }
  return neighborhood;
}

[[nodiscard]] bool Finite(const Vec2& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y);
}

[[nodiscard]] bool Finite(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool Finite(const Pose3& pose) noexcept {
  return Finite(pose.position_m) && std::isfinite(pose.orientation.w) &&
         std::isfinite(pose.orientation.x) &&
         std::isfinite(pose.orientation.y) &&
         std::isfinite(pose.orientation.z);
}

[[nodiscard]] bool Finite(const Twist3& twist) noexcept {
  return Finite(twist.linear_mps) && Finite(twist.angular_radps);
}

[[nodiscard]] std::array<std::uint64_t, 7U> PoseBits(
    const Pose3& pose) noexcept {
  return {
      std::bit_cast<std::uint64_t>(pose.position_m.x),
      std::bit_cast<std::uint64_t>(pose.position_m.y),
      std::bit_cast<std::uint64_t>(pose.position_m.z),
      std::bit_cast<std::uint64_t>(pose.orientation.w),
      std::bit_cast<std::uint64_t>(pose.orientation.x),
      std::bit_cast<std::uint64_t>(pose.orientation.y),
      std::bit_cast<std::uint64_t>(pose.orientation.z),
  };
}

[[nodiscard]] std::array<std::uint64_t, 6U> TwistBits(
    const Twist3& twist) noexcept {
  return {
      std::bit_cast<std::uint64_t>(twist.linear_mps.x),
      std::bit_cast<std::uint64_t>(twist.linear_mps.y),
      std::bit_cast<std::uint64_t>(twist.linear_mps.z),
      std::bit_cast<std::uint64_t>(twist.angular_radps.x),
      std::bit_cast<std::uint64_t>(twist.angular_radps.y),
      std::bit_cast<std::uint64_t>(twist.angular_radps.z),
  };
}

[[nodiscard]] bool IsForward(const WheelPrimitiveKind kind) noexcept {
  return kind == WheelPrimitiveKind::kForward ||
         kind == WheelPrimitiveKind::kForwardArc;
}

[[nodiscard]] bool IsReverse(const WheelPrimitiveKind kind) noexcept {
  return kind == WheelPrimitiveKind::kReverse ||
         kind == WheelPrimitiveKind::kReverseArc;
}

[[nodiscard]] bool IsSpin(const WheelPrimitiveKind kind) noexcept {
  return kind == WheelPrimitiveKind::kSpinClockwise ||
         kind == WheelPrimitiveKind::kSpinCounterclockwise;
}

[[nodiscard]] bool ModeAllows(const WheelMotionMode mode,
                              const WheelPrimitiveKind kind) noexcept {
  if (IsForward(kind)) {
    return mode == WheelMotionMode::kStart ||
           mode == WheelMotionMode::kForward;
  }
  if (IsReverse(kind)) {
    return mode == WheelMotionMode::kStart ||
           mode == WheelMotionMode::kReverse;
  }
  if (kind == WheelPrimitiveKind::kStopAndSwitch) {
    return mode != WheelMotionMode::kStart;
  }
  return IsSpin(kind);
}

[[nodiscard]] WheelMotionMode TargetMode(
    const WheelPrimitiveKind kind) noexcept {
  if (IsForward(kind)) {
    return WheelMotionMode::kForward;
  }
  if (IsReverse(kind)) {
    return WheelMotionMode::kReverse;
  }
  return WheelMotionMode::kStart;
}

[[nodiscard]] double CircumscribedRadius(
    const WheeledCapability& capability) noexcept {
  double radius = 0.0;
  for (const Vec2& vertex : capability.footprint_xy_m) {
    radius = std::max(radius, std::hypot(vertex.x, vertex.y));
  }
  return radius;
}

struct RelativeTwist final {
  double velocity_x{};
  double velocity_y{};
  double yaw_rate{};
};

[[nodiscard]] std::optional<RelativeTwist> LogRelativePose(
    const Pose3& relative_pose) noexcept {
  const auto yaw = YawFromQuaternion(relative_pose.orientation);
  if (!yaw.has_value()) {
    return std::nullopt;
  }
  if (std::abs(*yaw) <= kTolerance) {
    return RelativeTwist{
        .velocity_x = relative_pose.position_m.x,
        .velocity_y = relative_pose.position_m.y,
        .yaw_rate = 0.0,
    };
  }
  const double a = std::sin(*yaw) / *yaw;
  const double b = (1.0 - std::cos(*yaw)) / *yaw;
  const double determinant = a * a + b * b;
  if (!std::isfinite(determinant) || determinant <= kTolerance) {
    return std::nullopt;
  }
  return RelativeTwist{
      .velocity_x =
          (a * relative_pose.position_m.x + b * relative_pose.position_m.y) /
          determinant,
      .velocity_y =
          (-b * relative_pose.position_m.x + a * relative_pose.position_m.y) /
          determinant,
      .yaw_rate = *yaw,
  };
}

[[nodiscard]] Pose3 ExpRelativePose(const RelativeTwist& twist,
                                    const double ratio) noexcept {
  const double yaw = ratio * twist.yaw_rate;
  double x = ratio * twist.velocity_x;
  double y = ratio * twist.velocity_y;
  if (std::abs(twist.yaw_rate) > kTolerance) {
    const double sine = std::sin(yaw);
    const double one_minus_cosine = 1.0 - std::cos(yaw);
    x = (sine * twist.velocity_x -
         one_minus_cosine * twist.velocity_y) /
        twist.yaw_rate;
    y = (one_minus_cosine * twist.velocity_x +
         sine * twist.velocity_y) /
        twist.yaw_rate;
  }
  return Pose3{
      .position_m = Vec3{.x = x, .y = y, .z = 0.0},
      .orientation = QuaternionFromYaw(yaw),
  };
}

[[nodiscard]] bool ValidPrimitiveMotion(
    const WheelMotionPrimitive& primitive) noexcept {
  const auto twist = LogRelativePose(primitive.relative_end_pose);
  if (!twist.has_value()) {
    return false;
  }
  const double travel = std::hypot(twist->velocity_x, twist->velocity_y);
  const double lateral_tolerance =
      std::max(kPhysicalMatchTolerance, 0.01 * std::abs(twist->velocity_x));
  if (primitive.kind == WheelPrimitiveKind::kStopAndSwitch) {
    return travel <= kPhysicalMatchTolerance &&
           std::abs(twist->yaw_rate) <= kPhysicalMatchTolerance;
  }
  if (IsSpin(primitive.kind)) {
    const bool direction_matches =
        primitive.kind == WheelPrimitiveKind::kSpinCounterclockwise
            ? twist->yaw_rate > kTolerance
            : twist->yaw_rate < -kTolerance;
    return travel <= kPhysicalMatchTolerance && direction_matches;
  }
  if (std::abs(twist->velocity_y) > lateral_tolerance ||
      std::abs(twist->velocity_x) <= kTolerance) {
    return false;
  }
  if (IsForward(primitive.kind) && twist->velocity_x <= kTolerance) {
    return false;
  }
  if (IsReverse(primitive.kind) && twist->velocity_x >= -kTolerance) {
    return false;
  }
  if ((primitive.kind == WheelPrimitiveKind::kForward ||
       primitive.kind == WheelPrimitiveKind::kReverse) &&
      std::abs(twist->yaw_rate) > kPhysicalMatchTolerance) {
    return false;
  }
  return true;
}

[[nodiscard]] bool ValidCapability(
    const WheeledCapability& capability) noexcept {
  if (capability.footprint_xy_m.size() < 3U ||
      capability.motion_primitives.empty() ||
      !std::ranges::all_of(capability.footprint_xy_m,
                           [](const Vec2& value) { return Finite(value); }) ||
      !std::isfinite(capability.wheel_diameter_m) ||
      capability.wheel_diameter_m <= 0.0 ||
      !std::isfinite(capability.wheelbase_m) ||
      capability.wheelbase_m <= 0.0 ||
      !std::isfinite(capability.track_width_m) ||
      capability.track_width_m <= 0.0 ||
      !std::isfinite(capability.minimum_underbody_clearance_m) ||
      capability.minimum_underbody_clearance_m <= 0.0 ||
      !std::isfinite(capability.maximum_forward_speed_mps) ||
      capability.maximum_forward_speed_mps <= 0.0 ||
      !std::isfinite(capability.maximum_reverse_speed_mps) ||
      capability.maximum_reverse_speed_mps <= 0.0 ||
      !std::isfinite(capability.maximum_spin_rate_radps) ||
      capability.maximum_spin_rate_radps <= 0.0 ||
      !std::isfinite(capability.maximum_acceleration_mps2) ||
      capability.maximum_acceleration_mps2 <= 0.0 ||
      !std::isfinite(capability.maximum_braking_deceleration_mps2) ||
      capability.maximum_braking_deceleration_mps2 <= 0.0 ||
      !std::isfinite(capability.maximum_yaw_acceleration_radps2) ||
      capability.maximum_yaw_acceleration_radps2 <= 0.0 ||
      !std::isfinite(capability.maximum_lateral_acceleration_mps2) ||
      capability.maximum_lateral_acceleration_mps2 <= 0.0 ||
      !std::isfinite(capability.maximum_curvature_per_m) ||
      capability.maximum_curvature_per_m <= 0.0 ||
      !std::isfinite(capability.maximum_slope_rad) ||
      capability.maximum_slope_rad < 0.0 ||
      !std::isfinite(capability.maximum_local_obstacle_relief_m) ||
      capability.maximum_local_obstacle_relief_m < 0.0 ||
      !std::isfinite(capability.minimum_clearance_m) ||
      capability.minimum_clearance_m < 0.0) {
    return false;
  }
  return std::ranges::all_of(
      capability.motion_primitives,
      [](const WheelMotionPrimitive& primitive) {
        return !primitive.primitive_id.empty() &&
               Finite(primitive.relative_end_pose) &&
               YawFromQuaternion(primitive.relative_end_pose.orientation)
                   .has_value() &&
               ValidPrimitiveMotion(primitive);
      });
}

[[nodiscard]] double Cross(const Vec2& first, const Vec2& second,
                           const Vec2& third) noexcept {
  return (second.x - first.x) * (third.y - first.y) -
         (second.y - first.y) * (third.x - first.x);
}

[[nodiscard]] bool PointOnSegment(const Vec2& point, const Vec2& start,
                                  const Vec2& finish) noexcept {
  return std::abs(Cross(start, finish, point)) <= kTolerance &&
         point.x + kTolerance >= std::min(start.x, finish.x) &&
         point.x <= std::max(start.x, finish.x) + kTolerance &&
         point.y + kTolerance >= std::min(start.y, finish.y) &&
         point.y <= std::max(start.y, finish.y) + kTolerance;
}

[[nodiscard]] bool SegmentsIntersect(const Vec2& first_start,
                                     const Vec2& first_finish,
                                     const Vec2& second_start,
                                     const Vec2& second_finish) noexcept {
  const double first_side_start =
      Cross(first_start, first_finish, second_start);
  const double first_side_finish =
      Cross(first_start, first_finish, second_finish);
  const double second_side_start =
      Cross(second_start, second_finish, first_start);
  const double second_side_finish =
      Cross(second_start, second_finish, first_finish);
  const bool proper =
      ((first_side_start > kTolerance && first_side_finish < -kTolerance) ||
       (first_side_start < -kTolerance && first_side_finish > kTolerance)) &&
      ((second_side_start > kTolerance && second_side_finish < -kTolerance) ||
       (second_side_start < -kTolerance && second_side_finish > kTolerance));
  return proper || PointOnSegment(second_start, first_start, first_finish) ||
         PointOnSegment(second_finish, first_start, first_finish) ||
         PointOnSegment(first_start, second_start, second_finish) ||
         PointOnSegment(first_finish, second_start, second_finish);
}

[[nodiscard]] bool PointInPolygon(const Vec2& point,
                                  const std::vector<Vec2>& polygon) noexcept {
  bool inside = false;
  for (std::size_t current = 0U, previous = polygon.size() - 1U;
       current < polygon.size(); previous = current++) {
    const Vec2& start = polygon[previous];
    const Vec2& finish = polygon[current];
    if (PointOnSegment(point, start, finish)) {
      return true;
    }
    if ((start.y > point.y) == (finish.y > point.y)) {
      continue;
    }
    const double crossing_x =
        start.x + (point.y - start.y) * (finish.x - start.x) /
                      (finish.y - start.y);
    if (point.x < crossing_x) {
      inside = !inside;
    }
  }
  return inside;
}

[[nodiscard]] bool PolygonIntersectsCell(const std::vector<Vec2>& polygon,
                                         const double minimum_x,
                                         const double minimum_y,
                                         const double maximum_x,
                                         const double maximum_y) noexcept {
  const auto inside_cell = [&](const Vec2& point) {
    return point.x + kTolerance >= minimum_x &&
           point.x <= maximum_x + kTolerance &&
           point.y + kTolerance >= minimum_y &&
           point.y <= maximum_y + kTolerance;
  };
  if (std::ranges::any_of(polygon, inside_cell)) {
    return true;
  }
  const std::array<Vec2, 4U> corners{
      Vec2{.x = minimum_x, .y = minimum_y},
      Vec2{.x = maximum_x, .y = minimum_y},
      Vec2{.x = maximum_x, .y = maximum_y},
      Vec2{.x = minimum_x, .y = maximum_y},
  };
  if (std::ranges::any_of(corners, [&](const Vec2& corner) {
        return PointInPolygon(corner, polygon);
      })) {
    return true;
  }
  for (std::size_t polygon_index = 0U; polygon_index < polygon.size();
       ++polygon_index) {
    for (std::size_t cell_index = 0U; cell_index < corners.size();
         ++cell_index) {
      if (SegmentsIntersect(
              polygon[polygon_index],
              polygon[(polygon_index + 1U) % polygon.size()],
              corners[cell_index],
              corners[(cell_index + 1U) % corners.size()])) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] double SquaredDistanceToSegment(
    const Vec2& point, const Vec2& start, const Vec2& finish) noexcept {
  const double dx = finish.x - start.x;
  const double dy = finish.y - start.y;
  const double squared_length = dx * dx + dy * dy;
  if (squared_length <= kTolerance) {
    const double px = point.x - start.x;
    const double py = point.y - start.y;
    return px * px + py * py;
  }
  const double ratio = std::clamp(
      ((point.x - start.x) * dx + (point.y - start.y) * dy) /
          squared_length,
      0.0, 1.0);
  const double px = point.x - (start.x + ratio * dx);
  const double py = point.y - (start.y + ratio * dy);
  return px * px + py * py;
}

[[nodiscard]] double PolygonDistanceToCell(
    const std::vector<Vec2>& polygon, const double minimum_x,
    const double minimum_y, const double maximum_x,
    const double maximum_y) noexcept {
  if (PolygonIntersectsCell(polygon, minimum_x, minimum_y, maximum_x,
                            maximum_y)) {
    return 0.0;
  }
  const std::array<Vec2, 4U> corners{
      Vec2{.x = minimum_x, .y = minimum_y},
      Vec2{.x = maximum_x, .y = minimum_y},
      Vec2{.x = maximum_x, .y = maximum_y},
      Vec2{.x = minimum_x, .y = maximum_y},
  };
  double squared_distance = std::numeric_limits<double>::infinity();
  for (std::size_t polygon_index = 0U; polygon_index < polygon.size();
       ++polygon_index) {
    const Vec2& polygon_start = polygon[polygon_index];
    const Vec2& polygon_finish =
        polygon[(polygon_index + 1U) % polygon.size()];
    for (std::size_t cell_index = 0U; cell_index < corners.size();
         ++cell_index) {
      const Vec2& cell_start = corners[cell_index];
      const Vec2& cell_finish = corners[(cell_index + 1U) % corners.size()];
      squared_distance = std::min(
          {squared_distance,
           SquaredDistanceToSegment(polygon_start, cell_start, cell_finish),
           SquaredDistanceToSegment(polygon_finish, cell_start, cell_finish),
           SquaredDistanceToSegment(cell_start, polygon_start,
                                    polygon_finish),
           SquaredDistanceToSegment(cell_finish, polygon_start,
                                    polygon_finish)});
    }
  }
  return std::sqrt(squared_distance);
}

struct WheelStateKey final {
  std::int64_t x{};
  std::int64_t y{};
  std::uint16_t yaw{};
  WheelMotionMode mode{WheelMotionMode::kStart};
  bool narrow{};

  bool operator==(const WheelStateKey&) const = default;
};

struct WheelStateKeyHash final {
  [[nodiscard]] std::size_t operator()(
      const WheelStateKey& key) const noexcept {
    std::size_t value = static_cast<std::size_t>(key.x);
    value ^= static_cast<std::size_t>(key.y) + 0x9e3779b9U +
             (value << 6U) + (value >> 2U);
    value ^= static_cast<std::size_t>(key.yaw) + 0x9e3779b9U +
             (value << 6U) + (value >> 2U);
    value ^= static_cast<std::size_t>(key.mode) << 1U;
    value ^= static_cast<std::size_t>(key.narrow);
    return value;
  }
};

struct RejectedFingerprint final {
  WheelStateKey key;
  std::int64_t x{};
  std::int64_t y{};
  std::int64_t yaw{};
  std::uint32_t terminal_goal_mask{};

  bool operator==(const RejectedFingerprint&) const = default;
};

struct RejectedFingerprintHash final {
  [[nodiscard]] std::size_t operator()(
      const RejectedFingerprint& fingerprint) const noexcept {
    std::size_t value = WheelStateKeyHash{}(fingerprint.key);
    for (const std::int64_t coordinate :
         {fingerprint.x, fingerprint.y, fingerprint.yaw}) {
      value ^= static_cast<std::size_t>(coordinate) + 0x9e3779b9U +
               (value << 6U) + (value >> 2U);
    }
    value ^= static_cast<std::size_t>(fingerprint.terminal_goal_mask) +
             0x9e3779b9U + (value << 6U) + (value >> 2U);
    return value;
  }
};

struct BarrierRectangle final {
  double minimum_x{};
  double minimum_y{};
  double maximum_x{};
  double maximum_y{};
  std::array<Vec2, 4U> corners{};
  std::array<double, 4U> corner_to_goal{};
};

[[nodiscard]] bool SegmentCrossesRectangleInterior(
    const Vec2& start, const Vec2& finish,
    const BarrierRectangle& rectangle) noexcept {
  double lower = 0.0;
  double upper = 1.0;
  const auto intersect_open_axis = [&](const double source,
                                       const double target,
                                       const double minimum,
                                       const double maximum) {
    const double delta = target - source;
    if (std::abs(delta) <= kTolerance) {
      if (source <= minimum || source >= maximum) {
        lower = 1.0;
        upper = 0.0;
      }
      return;
    }
    double first = (minimum - source) / delta;
    double second = (maximum - source) / delta;
    if (first > second) {
      std::swap(first, second);
    }
    lower = std::max(lower, first);
    upper = std::min(upper, second);
  };
  intersect_open_axis(start.x, finish.x, rectangle.minimum_x,
                      rectangle.maximum_x);
  intersect_open_axis(start.y, finish.y, rectangle.minimum_y,
                      rectangle.maximum_y);
  return lower + kTolerance < upper && upper > kTolerance &&
         lower < 1.0 - kTolerance;
}

[[nodiscard]] double PointObstacleDistance(
    const Vec2& start, const Vec2& goal,
    const BarrierRectangle& rectangle) noexcept {
  const auto distance = [](const Vec2& first, const Vec2& second) {
    return std::hypot(first.x - second.x, first.y - second.y);
  };
  if (!SegmentCrossesRectangleInterior(start, goal, rectangle)) {
    return distance(start, goal);
  }
  double result = std::numeric_limits<double>::infinity();
  for (std::size_t corner = 0U; corner < rectangle.corners.size(); ++corner) {
    if (!SegmentCrossesRectangleInterior(
            start, rectangle.corners[corner], rectangle)) {
      result = std::min(result,
                        distance(start, rectangle.corners[corner]) +
                            rectangle.corner_to_goal[corner]);
    }
  }
  return result;
}

[[nodiscard]] bool SupportsInsetDisk(
    const std::vector<Vec2>& polygon) noexcept {
  if (polygon.size() < 3U || !PointInPolygon(Vec2{}, polygon)) {
    return false;
  }
  double turn_sign = 0.0;
  for (std::size_t index = 0U; index < polygon.size(); ++index) {
    const Vec2& first = polygon[index];
    const Vec2& second = polygon[(index + 1U) % polygon.size()];
    const Vec2& third = polygon[(index + 2U) % polygon.size()];
    const double turn = Cross(first, second, third);
    if (std::abs(turn) > kTolerance) {
      if (turn_sign != 0.0 && std::signbit(turn) != std::signbit(turn_sign)) {
        return false;
      }
      turn_sign = turn;
    }
    for (std::size_t other = index + 1U; other < polygon.size(); ++other) {
      if (other == index || other == (index + 1U) % polygon.size() ||
          (other + 1U) % polygon.size() == index) {
        continue;
      }
      if (SegmentsIntersect(first, second, polygon[other],
                            polygon[(other + 1U) % polygon.size()])) {
        return false;
      }
    }
  }
  return turn_sign != 0.0;
}

[[nodiscard]] double PointDistanceToRectangle(
    const Vec2& point, const BarrierRectangle& rectangle) noexcept {
  const double dx = std::max(
      {rectangle.minimum_x - point.x, 0.0,
       point.x - rectangle.maximum_x});
  const double dy = std::max(
      {rectangle.minimum_y - point.y, 0.0,
       point.y - rectangle.maximum_y});
  return std::hypot(dx, dy);
}

struct EdgeKey final {
  std::size_t source_state{};
  std::size_t primitive_index{};

  bool operator==(const EdgeKey&) const = default;
};

struct EdgeKeyHash final {
  [[nodiscard]] std::size_t operator()(const EdgeKey& key) const noexcept {
    return key.source_state ^
           (key.primitive_index + 0x9e3779b97f4a7c15ULL +
            (key.source_state << 6U) + (key.source_state >> 2U));
  }
};

struct StatePair final {
  std::size_t source{};
  std::size_t target{};

  bool operator==(const StatePair&) const = default;
};

struct StatePairHash final {
  [[nodiscard]] std::size_t operator()(const StatePair& pair) const noexcept {
    return pair.source ^
           (pair.target + 0x9e3779b97f4a7c15ULL + (pair.source << 6U) +
            (pair.source >> 2U));
  }
};

struct Transition final {
  Pose3 source;
  Pose3 target;
  WheelMotionMode source_mode{WheelMotionMode::kStart};
  WheelMotionMode target_mode{WheelMotionMode::kStart};
  WheelPrimitiveKind kind{WheelPrimitiveKind::kForward};
  double path_length_m{};
  double yaw_delta_rad{};
  double curvature_per_m{};
  bool reverse{};
  bool initial_edge{};
};

enum class EdgeCertificationKind : std::uint8_t {
  kStartPose,
  kFullPrimitive,
  kScaledPrimitive,
};

enum class RejectionBucket : std::uint8_t {
  kNone,
  kDirectUnknownOrUnsupportedFootprint,
  kMeasuredObstacleClearance,
  kSlopeOrRoughness,
  kReliefOrUnderbody,
  kDynamicsOrPrimitiveShape,
  kDeadlineOrCancellation,
};

struct EdgeCertificateIdentity final {
  std::uint64_t local_source_sequence{};
  std::uint64_t local_terrain_semantics_id{};
  std::uintptr_t zero_sequence_request_identity{};
  std::array<std::uint64_t, 6U> map_metadata_bits{};
  std::uint64_t capability_fingerprint{};
  std::array<std::uint64_t, 7U> source_pose_bits{};
  std::array<std::uint64_t, 7U> target_pose_bits{};
  std::array<std::uint64_t, 6U> initial_velocity_bits{};
  WheelMotionMode source_mode{WheelMotionMode::kStart};
  WheelMotionMode target_mode{WheelMotionMode::kStart};
  WheelPrimitiveKind primitive_kind{WheelPrimitiveKind::kForward};
  EdgeCertificationKind certification_kind{
      EdgeCertificationKind::kStartPose};
  std::size_t primitive_stable_rank{};
  bool initial_edge{};
  bool pose_only{};

  bool operator==(const EdgeCertificateIdentity&) const = default;
};

struct EdgeEvaluation final {
  bool valid{};
  Transition transition;
  double cost{};
  double maximum_slope_rad{};
  double maximum_roughness_m{};
  double minimum_clearance_m{std::numeric_limits<double>::infinity()};
  double initial_speed{};
  double execution_time_s{};
  std::array<double, 5U> cost_components{};
  EdgeCertificateIdentity certificate_identity;
  RejectionBucket rejection_bucket{RejectionBucket::kNone};
};

struct BroadPhaseAssessment final {
  bool rejected{};
  bool interrupted{};
  RejectionBucket rejection_bucket{RejectionBucket::kNone};
};

struct PreferredEdgeRecord final {
  Pose3 endpoint;
  EdgeEvaluation evaluation;
  double cumulative_cost{};
  std::size_t stable_primitive_rank{};
  bool terminal_connector{};
};

struct CertifiedPreferredCandidate final {
  std::vector<PreferredEdgeRecord> edges;
  double cost{};
  std::size_t full_primitive_edges{};
  std::size_t terminal_connector_edges{};
  std::size_t goal_index{};
};

struct SpeedProfile final {
  double distance{};
  double initial_speed{};
  double peak_speed{};
  double acceleration{};
  double deceleration{};
  double acceleration_time{};
  double cruise_time{};
  double deceleration_time{};

  [[nodiscard]] double duration() const noexcept {
    return acceleration_time + cruise_time + deceleration_time;
  }
};

struct ProfileSample final {
  double distance{};
  double speed{};
};

struct ScaleInterval final {
  double lower{};
  double upper{};
};

struct Node final {
  WheelStateKey key;
  Pose3 pose;
  double best_g{std::numeric_limits<double>::infinity()};
  double certified_clearance_m{};
  std::size_t creation_sequence{};
  bool expandable{true};
  bool exact_goal{};
  std::optional<std::size_t> goal_index;
  std::uint32_t certified_terminal_goal_mask{};
};

struct WheelGoal final {
  PointGoal point;
  std::optional<double> yaw_rad;
  double yaw_tolerance_rad{};
};

struct PlanningLatticeFrame final {
  Vec2 origin_world_m;
  double yaw_world_rad{};
  double cosine{1.0};
  double sine{};

  explicit PlanningLatticeFrame(const Pose3& start)
      : origin_world_m{.x = start.position_m.x, .y = start.position_m.y},
        yaw_world_rad(
            YawFromQuaternion(start.orientation).value_or(0.0)),
        cosine(std::cos(yaw_world_rad)),
        sine(std::sin(yaw_world_rad)) {}

  [[nodiscard]] Vec2 ToLocalPosition(const Vec2 world) const noexcept {
    const double dx = world.x - origin_world_m.x;
    const double dy = world.y - origin_world_m.y;
    return {
        .x = cosine * dx + sine * dy,
        .y = -sine * dx + cosine * dy,
    };
  }

  [[nodiscard]] Vec2 ToWorldPosition(const Vec2 local) const noexcept {
    return {
        .x = origin_world_m.x + cosine * local.x - sine * local.y,
        .y = origin_world_m.y + sine * local.x + cosine * local.y,
    };
  }

  [[nodiscard]] double ToLocalYaw(const double world_yaw) const noexcept {
    return NormalizeYaw(world_yaw - yaw_world_rad);
  }

  [[nodiscard]] double ToWorldYaw(const double local_yaw) const noexcept {
    return NormalizeYaw(yaw_world_rad + local_yaw);
  }
};

class WheelSearchGraph final {
 public:
  WheelSearchGraph(const WheelPlanRequest& request,
                   std::vector<WheelGoal> goals)
      : request_(request),
        terrain_(*request.terrain),
        capability_(*request.capability),
        map_(*request.terrain->map),
        lattice_frame_(request.start.pose),
        goals_(std::move(goals)),
        goal_(goals_.front().point),
        goal_yaw_(goals_.front().yaw_rad),
        goal_yaw_tolerance_rad_(goals_.front().yaw_tolerance_rad),
        footprint_radius_m_(CircumscribedRadius(capability_)),
        footprint_contains_origin_(
            PointInPolygon(Vec2{}, capability_.footprint_xy_m)),
        capability_fingerprint_(
            request.capability_fingerprint != 0U
                ? request.capability_fingerprint
                : shared::StableCapabilityFingerprint(
                      PlatformCapability{capability_})),
        map_metadata_bits_({
            static_cast<std::uint64_t>(map_.width()),
            static_cast<std::uint64_t>(map_.height()),
            std::bit_cast<std::uint64_t>(map_.resolution_m()),
            std::bit_cast<std::uint64_t>(map_.origin_m().x),
            std::bit_cast<std::uint64_t>(map_.origin_m().y),
            std::bit_cast<std::uint64_t>(map_.origin_m().z),
        }) {
    const std::size_t cell_count = map_.cell_count();
    if (SupportsInsetDisk(capability_.footprint_xy_m)) {
      broad_inset_radius_m_ = std::numeric_limits<double>::infinity();
      for (std::size_t index = 0U;
           index < capability_.footprint_xy_m.size(); ++index) {
        broad_inset_radius_m_ = std::min(
            broad_inset_radius_m_,
            std::sqrt(SquaredDistanceToSegment(
                Vec2{}, capability_.footprint_xy_m[index],
                capability_.footprint_xy_m[
                    (index + 1U) % capability_.footprint_xy_m.size()])));
      }
    }
    nodes_.reserve(std::min<std::size_t>(cell_count, 65536U));
    state_ids_.reserve(nodes_.capacity());
    const std::size_t prefix_width = map_.width() + 1U;
    occupied_integral_.assign(prefix_width * (map_.height() + 1U), 0U);
    complex_terrain_integral_.assign(
        prefix_width * (map_.height() + 1U), 0U);
    occupied_by_row_.resize(map_.height());
    const auto elevations = map_.FloatLayer("elevation");
    for (std::size_t y = 0U; y < map_.height(); ++y) {
      std::uint32_t row_occupied = 0U;
      std::uint32_t row_complex_terrain = 0U;
      for (std::size_t x = 0U; x < map_.width(); ++x) {
        const std::size_t index = y * map_.width() + x;
        const bool occupied = terrain_.occupied[index] != 0U;
        row_occupied += static_cast<std::uint32_t>(occupied);
        if (occupied) {
          occupied_by_row_[y].push_back(static_cast<std::int32_t>(x));
        }
        const bool unsupported = terrain_.free_with_height[index] == 0U;
        const bool complex_terrain =
            unsupported || index >= elevations.size() ||
            !std::isfinite(elevations[index]) ||
            !std::isfinite(terrain_.slope_rad[index]) ||
            terrain_.slope_rad[index] != 0.0F ||
            !std::isfinite(terrain_.roughness_m[index]) ||
            terrain_.roughness_m[index] != 0.0F;
        row_complex_terrain +=
            static_cast<std::uint32_t>(complex_terrain);
        occupied_integral_[(y + 1U) * prefix_width + x + 1U] =
            occupied_integral_[y * prefix_width + x + 1U] + row_occupied;
        complex_terrain_integral_[(y + 1U) * prefix_width + x + 1U] =
            complex_terrain_integral_[y * prefix_width + x + 1U] +
            row_complex_terrain;
      }
    }
    ordered_primitives_.resize(capability_.motion_primitives.size());
    primitive_stable_rank_.resize(capability_.motion_primitives.size());
    for (std::size_t index = 0U; index < ordered_primitives_.size(); ++index) {
      ordered_primitives_[index] = index;
      const auto& primitive = capability_.motion_primitives[index];
      const double length = std::hypot(
          primitive.relative_end_pose.position_m.x,
          primitive.relative_end_pose.position_m.y);
      maximum_primitive_reach_m_ =
          std::max(maximum_primitive_reach_m_, length);
      maximum_primitive_yaw_rad_ = std::max(
          maximum_primitive_yaw_rad_,
          std::abs(YawFromQuaternion(primitive.relative_end_pose.orientation)
                       .value_or(0.0)));
      if (IsForward(primitive.kind) || IsReverse(primitive.kind)) {
        maximum_translation_reach_m_ =
            std::max(maximum_translation_reach_m_, length);
        const auto twist = LogRelativePose(primitive.relative_end_pose);
        if (twist.has_value()) {
          const double primitive_path_length =
              std::hypot(twist->velocity_x, twist->velocity_y);
          maximum_translation_path_length_m_ = std::max(
              maximum_translation_path_length_m_, primitive_path_length);
          Transition transition{
              .path_length_m = primitive_path_length,
              .yaw_delta_rad = twist->yaw_rate,
              .curvature_per_m =
                  std::hypot(twist->velocity_x, twist->velocity_y) > kTolerance
                      ? twist->yaw_rate /
                            std::hypot(twist->velocity_x, twist->velocity_y)
                      : 0.0,
              .reverse = IsReverse(primitive.kind),
          };
          const auto profile = MakeProfile(
              transition.path_length_m, 0.0,
              TranslationSpeedLimit(transition),
              TranslationAccelerationLimit(transition),
              TranslationBrakingLimit(transition));
          if (profile.has_value()) {
            minimum_zero_speed_translation_duration_s_ = std::min(
                minimum_zero_speed_translation_duration_s_,
                profile->duration());
          }
        }
      }
    }
    yaw_bins_ = SelectYawBins();
    std::stable_sort(
        ordered_primitives_.begin(), ordered_primitives_.end(),
        [&](const std::size_t lhs, const std::size_t rhs) {
          const auto primitive_key = [&](const std::size_t index) {
            const auto& primitive = capability_.motion_primitives[index];
            return std::tuple{
                primitive.primitive_id, primitive.kind,
                primitive.relative_end_pose.position_m.x,
                primitive.relative_end_pose.position_m.y,
                primitive.relative_end_pose.position_m.z,
                YawFromQuaternion(primitive.relative_end_pose.orientation)
                    .value_or(0.0)};
          };
          return primitive_key(lhs) < primitive_key(rhs);
        });
    for (std::size_t rank = 0U; rank < ordered_primitives_.size(); ++rank) {
      primitive_stable_rank_[ordered_primitives_[rank]] = rank;
    }
    platform_length_scale_m_ = std::max(
        {2.0 * footprint_radius_m_, capability_.wheelbase_m,
         capability_.track_width_m, capability_.wheel_diameter_m});
    cost_scales_ = {
        platform_length_scale_m_,
        platform_length_scale_m_ /
            std::max(capability_.maximum_forward_speed_mps,
                     capability_.maximum_reverse_speed_mps),
        std::max(1.0e-3,
                 capability_.maximum_slope_rad +
                     capability_.maximum_local_obstacle_relief_m /
                         platform_length_scale_m_),
        1.0,
        3.0,
    };
    const WheelStateKey key = Quantize(request_.start.pose,
                                       WheelMotionMode::kStart);
    nodes_.push_back(Node{
        .key = key,
        .pose = request_.start.pose,
        .best_g = 0.0,
        .certified_clearance_m = std::numeric_limits<double>::infinity(),
        .creation_sequence = next_creation_sequence_++,
        .expandable = true,
        .exact_goal = false,
    });
    state_ids_[key].push_back(0U);
    maximum_active_labels_per_key_ = 1U;
    std::vector<shared::GridCell> goal_cells;
    goal_cells.reserve(goals_.size());
    for (const WheelGoal& goal : goals_) {
      if (const auto goal_cell = map_.PositionToCell(
              Vec2{.x = goal.point.position_m.x,
                   .y = goal.point.position_m.y});
          goal_cell.has_value()) {
        goal_cells.push_back(*goal_cell);
      }
    }
    goal_distance_field_ = request_.goal_distance_field;
    if (goal_distance_field_ == nullptr && !goal_cells.empty()) {
      auto built = shared::BuildGoalDistanceField(terrain_, goal_cells,
                                                  request_.control);
      if (built.has_value()) {
        goal_distance_field_ =
            std::make_shared<const shared::GoalDistanceField>(
                std::move(*built));
      }
    }
    barriers_by_goal_.resize(goals_.size());
    std::size_t preferred_goal_index = 0U;
    double preferred_goal_lower_bound =
        std::numeric_limits<double>::infinity();
    for (std::size_t goal_index = 0U; goal_index < goals_.size();
         ++goal_index) {
      ActivateGoal(goal_index);
      const double lower_bound = HeuristicForActiveGoal(request_.start.pose);
      if (std::tie(lower_bound, goal_index) <
          std::tie(preferred_goal_lower_bound, preferred_goal_index)) {
        preferred_goal_lower_bound = lower_bound;
        preferred_goal_index = goal_index;
      }
    }
    ActivateGoal(preferred_goal_index);
    barriers_.clear();
    BuildBarrierLowerBounds();
    barriers_by_goal_[preferred_goal_index] = barriers_;
    ++preferred_builder_invocations_;
    BuildCertifiedPreferredCandidate();
    if (certified_preferred_candidate_.has_value()) {
      certified_preferred_candidate_->goal_index = preferred_goal_index;
    }
    BuildCertifiedInitialCandidate();
  }

  [[nodiscard]] std::size_t state_count() const noexcept {
    return std::numeric_limits<std::size_t>::max();
  }

  [[nodiscard]] bool used_narrow_resolution() const noexcept {
    return used_narrow_resolution_;
  }

  [[nodiscard]] std::size_t validation_count() const noexcept {
    return validation_cache_.evaluation_count() +
           preferred_validation_count_;
  }

  [[nodiscard]] std::size_t validation_cache_hits() const noexcept {
    return validation_requests_ - validation_cache_.evaluation_count();
  }

  [[nodiscard]] std::size_t broad_phase_rejects() const noexcept {
    return broad_phase_rejects_;
  }

  [[nodiscard]] std::size_t full_certifications() const noexcept {
    return full_certifications_;
  }

  [[nodiscard]] std::size_t full_invalidations() const noexcept {
    return full_invalidations_;
  }

  [[nodiscard]] std::size_t
  returned_edge_certificate_confirmations() const noexcept {
    return returned_edge_certificate_confirmations_;
  }

  [[nodiscard]] std::size_t quantization_alias_states() const noexcept {
    return quantized_state_reuses_;
  }

  [[nodiscard]] std::size_t quantized_state_reuses() const noexcept {
    return quantized_state_reuses_;
  }

  [[nodiscard]] std::size_t quantized_endpoint_aliases() const noexcept {
    return quantized_endpoint_aliases_;
  }

  [[nodiscard]] std::size_t quantized_state_count() const noexcept {
    return nodes_.size();
  }

  [[nodiscard]] std::size_t maximum_active_labels_per_key() const noexcept {
    return maximum_active_labels_per_key_;
  }

  [[nodiscard]] std::size_t preferred_builder_invocations() const noexcept {
    return preferred_builder_invocations_;
  }

  [[nodiscard]] std::size_t sweep_cell_checks() const noexcept {
    return sweep_cell_checks_;
  }

  [[nodiscard]] std::size_t
  direct_unknown_or_unsupported_footprint_rejects() const noexcept {
    return direct_unknown_or_unsupported_footprint_rejects_;
  }

  [[nodiscard]] std::size_t
  measured_obstacle_clearance_rejects() const noexcept {
    return measured_obstacle_clearance_rejects_;
  }

  [[nodiscard]] std::size_t slope_or_roughness_rejects() const noexcept {
    return slope_or_roughness_rejects_;
  }

  [[nodiscard]] std::size_t relief_or_underbody_rejects() const noexcept {
    return relief_or_underbody_rejects_;
  }

  [[nodiscard]] std::size_t
  dynamics_or_primitive_shape_rejects() const noexcept {
    return dynamics_or_primitive_shape_rejects_;
  }

  [[nodiscard]] std::size_t
  deadline_or_cancellation_interruptions() const noexcept {
    return deadline_or_cancellation_interruptions_;
  }

  [[nodiscard]] std::size_t far_clearance_scan_skips() const noexcept {
    return 0U;
  }

  [[nodiscard]] std::size_t occupied_clearance_cell_checks() const noexcept {
    return occupied_clearance_cell_checks_;
  }

  [[nodiscard]] const std::array<double, 5U>& cost_scales() const noexcept {
    return cost_scales_;
  }

  [[nodiscard]] double finest_xy_key_resolution_m() const noexcept {
    return finest_xy_key_resolution_m_;
  }

  [[nodiscard]] std::size_t maximum_yaw_bins() const noexcept {
    return maximum_yaw_bins_;
  }

  [[nodiscard]] const std::optional<CertifiedPreferredCandidate>&
  certified_preferred_candidate() const noexcept {
    return certified_preferred_candidate_;
  }

  [[nodiscard]] const std::optional<shared::SearchCandidate>&
  certified_initial_candidate() const noexcept {
    return certified_initial_candidate_;
  }

  [[nodiscard]] bool StateExpandable(const std::size_t state) const noexcept {
    return state < nodes_.size() && nodes_[state].expandable;
  }

  void OnRelaxed(const std::size_t state, const double best_g) {
    if (state >= nodes_.size() || !std::isfinite(best_g)) {
      return;
    }
    nodes_[state].best_g = best_g;
    if (!nodes_[state].exact_goal) {
      RerankLabels(nodes_[state].key, state);
    }
  }

  [[nodiscard]] double RelaxedDistance(
      const std::size_t state) const noexcept {
    if (goal_distance_field_ == nullptr || state >= nodes_.size()) {
      return std::numeric_limits<double>::infinity();
    }
    const auto cell = map_.PositionToCell(
        Vec2{.x = nodes_[state].pose.position_m.x,
             .y = nodes_[state].pose.position_m.y});
    if (!cell.has_value()) {
      return std::numeric_limits<double>::infinity();
    }
    return goal_distance_field_->distance_m[map_.Index(*cell)];
  }

  void BuildBarrierLowerBounds() {
    const Vec2 origin{};
    if (!SupportsInsetDisk(capability_.footprint_xy_m)) {
      return;
    }
    double inset_radius = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0U;
         index < capability_.footprint_xy_m.size(); ++index) {
      inset_radius = std::min(
          inset_radius,
          std::sqrt(SquaredDistanceToSegment(
              origin, capability_.footprint_xy_m[index],
              capability_.footprint_xy_m[
                  (index + 1U) % capability_.footprint_xy_m.size()])));
    }
    if (!std::isfinite(inset_radius) || inset_radius <= kTolerance) {
      return;
    }
    barrier_inset_radius_m_ = inset_radius;
    struct Run final {
      std::size_t length{};
      bool vertical{};
      std::size_t fixed{};
      std::size_t first{};
      std::size_t last{};
    };
    std::vector<Run> runs;
    for (std::size_t x = 0U; x < map_.width(); ++x) {
      if ((x & 63U) == 0U && ControlInterrupted()) {
        return;
      }
      std::size_t first = 0U;
      bool active = false;
      for (std::size_t y = 0U; y <= map_.height(); ++y) {
        const bool hazard =
            y < map_.height() &&
            terrain_.free_with_height[y * map_.width() + x] == 0U;
        if (hazard && !active) {
          first = y;
          active = true;
        }
        if (!hazard && active) {
          const std::size_t length = y - first;
          if (length >= 2U) {
            runs.push_back(Run{.length = length,
                               .vertical = true,
                               .fixed = x,
                               .first = first,
                               .last = y - 1U});
          }
          active = false;
        }
      }
    }
    for (std::size_t y = 0U; y < map_.height(); ++y) {
      if ((y & 63U) == 0U && ControlInterrupted()) {
        return;
      }
      std::size_t first = 0U;
      bool active = false;
      for (std::size_t x = 0U; x <= map_.width(); ++x) {
        const bool hazard =
            x < map_.width() &&
            terrain_.free_with_height[y * map_.width() + x] == 0U;
        if (hazard && !active) {
          first = x;
          active = true;
        }
        if (!hazard && active) {
          const std::size_t length = x - first;
          if (length >= 2U) {
            runs.push_back(Run{.length = length,
                               .vertical = false,
                               .fixed = y,
                               .first = first,
                               .last = x - 1U});
          }
          active = false;
        }
      }
    }
    const double radius =
        inset_radius +
        std::max(0.0, capability_.minimum_clearance_m - kTolerance);
    const double resolution = map_.resolution_m();
    const Vec3 map_origin = map_.origin_m();
    const Vec2 start{.x = request_.start.pose.position_m.x,
                     .y = request_.start.pose.position_m.y};
    const Vec2 goal{.x = goal_.position_m.x, .y = goal_.position_m.y};
    struct Candidate final {
      BarrierRectangle rectangle;
      Run run;
      double impact{};
      double corridor_distance{};
      std::size_t creation{};
    };
    std::vector<Candidate> candidates;
    for (const Run& run : runs) {
      BarrierRectangle rectangle;
      if (run.vertical) {
        rectangle.minimum_x =
            map_origin.x + static_cast<double>(run.fixed) * resolution;
        rectangle.maximum_x = rectangle.minimum_x + resolution;
        rectangle.minimum_y =
            map_origin.y + static_cast<double>(run.first) * resolution -
            radius;
        rectangle.maximum_y =
            map_origin.y + static_cast<double>(run.last + 1U) * resolution +
            radius;
      } else {
        rectangle.minimum_y =
            map_origin.y + static_cast<double>(run.fixed) * resolution;
        rectangle.maximum_y = rectangle.minimum_y + resolution;
        rectangle.minimum_x =
            map_origin.x + static_cast<double>(run.first) * resolution -
            radius;
        rectangle.maximum_x =
            map_origin.x + static_cast<double>(run.last + 1U) * resolution +
            radius;
      }
      rectangle.corners = {
          Vec2{.x = rectangle.minimum_x, .y = rectangle.minimum_y},
          Vec2{.x = rectangle.maximum_x, .y = rectangle.minimum_y},
          Vec2{.x = rectangle.maximum_x, .y = rectangle.maximum_y},
          Vec2{.x = rectangle.minimum_x, .y = rectangle.maximum_y},
      };
      if (PointDistanceToRectangle(goal, rectangle) <=
          goal_.tolerance_m + kTolerance) {
        continue;
      }
      constexpr std::size_t kNodeCount = 5U;
      std::array<Vec2, kNodeCount> points{
          rectangle.corners[0], rectangle.corners[1], rectangle.corners[2],
          rectangle.corners[3], goal};
      std::array<double, kNodeCount> distances;
      distances.fill(std::numeric_limits<double>::infinity());
      distances[4U] = 0.0;
      std::array<bool, kNodeCount> visited{};
      for (std::size_t iteration = 0U; iteration < kNodeCount; ++iteration) {
        std::size_t current = kNodeCount;
        for (std::size_t node = 0U; node < kNodeCount; ++node) {
          if (!visited[node] &&
              (current == kNodeCount || distances[node] < distances[current])) {
            current = node;
          }
        }
        if (current == kNodeCount || !std::isfinite(distances[current])) {
          break;
        }
        visited[current] = true;
        for (std::size_t next = 0U; next < kNodeCount; ++next) {
          if (visited[next] || SegmentCrossesRectangleInterior(
                                   points[current], points[next], rectangle)) {
            continue;
          }
          distances[next] = std::min(
              distances[next],
              distances[current] +
                  std::hypot(points[current].x - points[next].x,
                             points[current].y - points[next].y));
        }
      }
      std::copy_n(distances.begin(), 4U,
                  rectangle.corner_to_goal.begin());
      const double direct_distance =
          std::hypot(start.x - goal.x, start.y - goal.y);
      const double impact = std::max(
          0.0, PointObstacleDistance(start, goal, rectangle) -
                   direct_distance);
      const Vec2 center{
          .x = 0.5 * (rectangle.minimum_x + rectangle.maximum_x),
          .y = 0.5 * (rectangle.minimum_y + rectangle.maximum_y)};
      candidates.push_back(Candidate{
          .rectangle = rectangle,
          .run = run,
          .impact = impact,
          .corridor_distance =
              std::sqrt(SquaredDistanceToSegment(center, start, goal)),
          .creation = candidates.size(),
      });
    }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& left, const Candidate& right) {
      return std::tuple{-left.impact,
                        std::numeric_limits<std::size_t>::max() -
                            left.run.length,
                        left.corridor_distance, left.run.vertical,
                        left.run.fixed, left.run.first, left.run.last,
                        left.creation} <
             std::tuple{-right.impact,
                        std::numeric_limits<std::size_t>::max() -
                            right.run.length,
                        right.corridor_distance, right.run.vertical,
                        right.run.fixed, right.run.first, right.run.last,
                        right.creation};
    });
    for (const Candidate& candidate : candidates) {
      const auto contains = [](const BarrierRectangle& outer,
                               const BarrierRectangle& inner) {
        return outer.minimum_x <= inner.minimum_x + kTolerance &&
               outer.minimum_y <= inner.minimum_y + kTolerance &&
               outer.maximum_x + kTolerance >= inner.maximum_x &&
               outer.maximum_y + kTolerance >= inner.maximum_y;
      };
      if (std::ranges::any_of(
              barriers_, [&](const BarrierRectangle& retained) {
                return contains(retained, candidate.rectangle);
              })) {
        continue;
      }
      barriers_.push_back(candidate.rectangle);
      if (barriers_.size() == 32U) {
        break;
      }
    }
  }

  [[nodiscard]] double Guidance(const std::size_t state) const noexcept {
    const double relaxed_distance = RelaxedDistance(state);
    if (!std::isfinite(relaxed_distance) || state >= nodes_.size()) {
      return 0.0;
    }
    const double clearance_deficit = std::max(
        0.0, footprint_radius_m_ - nodes_[state].certified_clearance_m);
    return relaxed_distance + clearance_deficit;
  }

  [[nodiscard]] const EdgeEvaluation* EvaluationForEdge(
      const std::size_t source, const std::size_t target) {
    const EdgeEvaluation* evaluation = nullptr;
    if (const auto edge = PreferredEdgeIndexForState(target);
        edge.has_value()) {
      const std::size_t expected_source =
          *edge == 0U ? 0U : PreferredStateId(*edge - 1U);
      if (source != expected_source) {
        return nullptr;
      }
      evaluation = &certified_preferred_candidate_->edges[*edge].evaluation;
    } else {
      const auto found = emitted_edges_.find(
          StatePair{.source = source, .target = target});
      if (found == emitted_edges_.end()) {
        return nullptr;
      }
      evaluation =
          &CachedEvaluation(found->second, [] { return EdgeEvaluation{}; });
    }
    if (!ConfirmReturnedCertificate(source, target, *evaluation)) {
      return nullptr;
    }
    ++returned_edge_certificate_confirmations_;
    return evaluation;
  }

  [[nodiscard]] bool AppendTimedEdge(
      const EdgeEvaluation& evaluation, std::chrono::nanoseconds* elapsed,
      std::vector<TrajectoryPoint>* trajectory,
      const SearchControl& control) const {
    if (elapsed == nullptr || trajectory == nullptr || !evaluation.valid ||
        !std::isfinite(evaluation.execution_time_s) ||
        evaluation.execution_time_s <= 0.0) {
      return false;
    }
    constexpr std::size_t kSamples = 8U;
    const Transition& transition = evaluation.transition;
    const bool translation = transition.path_length_m > kTolerance;
    const double motion_distance =
        translation ? transition.path_length_m
                    : std::abs(transition.yaw_delta_rad);
    const double speed_limit =
        translation ? TranslationSpeedLimit(transition)
                    : capability_.maximum_spin_rate_radps;
    const double acceleration =
        translation ? TranslationAccelerationLimit(transition)
                    : capability_.maximum_yaw_acceleration_radps2;
    const double deceleration =
        translation ? TranslationBrakingLimit(transition)
                    : capability_.maximum_yaw_acceleration_radps2;
    const auto duration = std::chrono::nanoseconds{std::max<std::int64_t>(
        1, static_cast<std::int64_t>(
               std::ceil(evaluation.execution_time_s * 1.0e9)))};
    std::optional<SpeedProfile> profile;
    if (motion_distance > kTolerance) {
      profile = MakeProfile(motion_distance, evaluation.initial_speed,
                            speed_limit, acceleration, deceleration);
      if (!profile.has_value()) {
        return false;
      }
    }
    for (std::size_t sample = 1U; sample <= kSamples; ++sample) {
      if (shared::StopReason(control).has_value()) {
        return false;
      }
      const double time_ratio =
          static_cast<double>(sample) / static_cast<double>(kSamples);
      const double sample_time = time_ratio * evaluation.execution_time_s;
      double traveled = motion_distance * time_ratio;
      double speed = 0.0;
      if (profile.has_value()) {
        const ProfileSample profile_sample =
            SampleProfile(*profile, sample_time);
        speed = profile_sample.speed;
        traveled = profile_sample.distance;
      }
      const double progress =
          motion_distance > kTolerance
              ? std::clamp(traveled / motion_distance, 0.0, 1.0)
              : time_ratio;
      const auto pose = Interpolate(transition, progress);
      if (!pose.has_value()) {
        return false;
      }
      Twist3 velocity{};
      if (translation && speed > 0.0) {
        const double yaw =
            YawFromQuaternion(pose->orientation).value_or(0.0);
        const double direction = transition.reverse ? -1.0 : 1.0;
        velocity.linear_mps.x = direction * speed * std::cos(yaw);
        velocity.linear_mps.y = direction * speed * std::sin(yaw);
        velocity.angular_radps.z =
            speed * transition.yaw_delta_rad / transition.path_length_m;
      } else if (!translation && motion_distance > kTolerance) {
        velocity.angular_radps.z =
            std::copysign(speed, transition.yaw_delta_rad);
      }
      const auto offset = std::chrono::nanoseconds{
          (duration.count() * static_cast<std::int64_t>(sample)) /
          static_cast<std::int64_t>(kSamples)};
      trajectory->push_back(TrajectoryPoint{
          .time_from_start = *elapsed + offset,
          .pose = sample == kSamples ? transition.target : *pose,
          .velocity = sample == kSamples ? Twist3{} : velocity,
      });
    }
    *elapsed += duration;
    return true;
  }

  [[nodiscard]] const Pose3& PoseForState(const std::size_t state) const {
    if (const auto edge = PreferredEdgeIndexForState(state);
        edge.has_value()) {
      return certified_preferred_candidate_->edges[*edge].endpoint;
    }
    return nodes_.at(state).pose;
  }

  [[nodiscard]] std::optional<WheelMotionMode> ModeForAnyState(
      const std::size_t state) const noexcept {
    if (state < nodes_.size()) {
      return nodes_[state].key.mode;
    }
    if (const auto edge = PreferredEdgeIndexForState(state);
        edge.has_value()) {
      return certified_preferred_candidate_->edges[*edge]
          .evaluation.transition.target_mode;
    }
    return std::nullopt;
  }

  [[nodiscard]] bool ConfirmReturnedCertificate(
      const std::size_t source, const std::size_t target,
      const EdgeEvaluation& evaluation) const noexcept {
    if (!evaluation.valid) {
      return false;
    }
    const auto source_mode = ModeForAnyState(source);
    const auto target_mode = ModeForAnyState(target);
    if (!source_mode.has_value() || !target_mode.has_value() ||
        *source_mode != evaluation.transition.source_mode ||
        *target_mode != evaluation.transition.target_mode ||
        PoseBits(PoseForState(source)) !=
            evaluation.certificate_identity.source_pose_bits ||
        PoseBits(PoseForState(target)) !=
            evaluation.certificate_identity.target_pose_bits) {
      return false;
    }
    return evaluation.certificate_identity == MakeCertificateIdentity(
               evaluation.transition,
               evaluation.certificate_identity.pose_only,
               evaluation.certificate_identity.certification_kind,
               evaluation.certificate_identity.primitive_stable_rank);
  }

  [[nodiscard]] std::optional<std::size_t> GoalIndexForState(
      const std::size_t state) const noexcept {
    if (const auto edge = PreferredEdgeIndexForState(state);
        edge.has_value()) {
      if (*edge + 1U == certified_preferred_candidate_->edges.size()) {
        return certified_preferred_candidate_->goal_index;
      }
      return std::nullopt;
    }
    if (state >= nodes_.size()) {
      return std::nullopt;
    }
    if (nodes_[state].goal_index.has_value()) {
      return nodes_[state].goal_index;
    }
    return MatchingGoalIndex(nodes_[state].pose);
  }

  [[nodiscard]] WheelMotionMode ModeForState(
      const std::size_t state) const noexcept {
    return nodes_[state].key.mode;
  }

  [[nodiscard]] bool ValidateStart() {
    const EdgeKey key{.source_state = 0U,
                      .primitive_index =
                          (2U + 2U * goals_.size()) *
                          capability_.motion_primitives.size()};
    return CachedEvaluation(key, [&] {
          Transition start{
              .source = request_.start.pose,
              .target = request_.start.pose,
          };
          return Evaluate(start, true, EdgeCertificationKind::kStartPose,
                          std::numeric_limits<std::size_t>::max());
        })
        .valid;
  }

  void Expand(const std::size_t state, const double source_g,
              std::vector<shared::GraphEdge>& edges) {
    if (state >= nodes_.size() || nodes_[state].exact_goal ||
        !nodes_[state].expandable || ControlInterrupted()) {
      return;
    }
    const Node source = nodes_[state];
    bool generated_goal_terminal = false;
    std::unordered_map<WheelStateKey, std::size_t, WheelStateKeyHash>
        generated_target_keys;
    for (const std::size_t primitive_index : ordered_primitives_) {
      if (ControlInterrupted()) {
        return;
      }
      auto transition = ApplyPrimitive(
          source, capability_.motion_primitives[primitive_index]);
      if (!transition.has_value()) {
        continue;
      }
      transition->initial_edge = state == 0U;
      const Pose3 actual_endpoint = transition->target;
      bool full_edge_reaches_goal = false;
      for (std::size_t goal_index = 0U; goal_index < goals_.size();
           ++goal_index) {
        ActivateGoal(goal_index);
        if (!PoseSatisfiesGoal(actual_endpoint)) {
          continue;
        }
        const EdgeKey goal_edge_key{
            .source_state = state,
            .primitive_index =
                (2U + goal_index) * capability_.motion_primitives.size() +
                primitive_index,
        };
        full_edge_reaches_goal =
            AppendGoalTerminal(state, *transition, goal_edge_key, goal_index,
                               0U, primitive_stable_rank_[primitive_index],
                               edges) ||
            full_edge_reaches_goal;
      }
      if (full_edge_reaches_goal) {
        generated_goal_terminal = true;
        continue;
      }
      const WheelStateKey target_key =
          Quantize(actual_endpoint, transition->target_mode);
      if (!generated_target_keys.emplace(target_key, primitive_index).second) {
        ++quantized_endpoint_aliases_;
      }
      const EdgeKey edge_key{.source_state = state,
                             .primitive_index = primitive_index};
      const EdgeEvaluation& evaluation = CachedEvaluation(
          edge_key, [&] {
            return Evaluate(*transition, false,
                            EdgeCertificationKind::kFullPrimitive,
                            primitive_stable_rank_[primitive_index]);
          });
      if (!evaluation.valid) {
        continue;
      }
      const auto target_state =
          Intern(target_key, actual_endpoint, source_g + evaluation.cost,
                 evaluation.minimum_clearance_m);
      if (!target_state.has_value()) {
        continue;
      }
      if (*target_state == state) {
        continue;
      }
      edges.push_back(shared::GraphEdge{
          .target_state = *target_state,
          .cost = evaluation.cost,
          .stable_index = StableEdgeIndex(
              state, primitive_stable_rank_[primitive_index]),
      });
      const StatePair state_pair{.source = state, .target = *target_state};
      const auto emitted = emitted_edges_.find(state_pair);
      if (emitted == emitted_edges_.end()) {
        emitted_edges_.emplace(state_pair, edge_key);
      } else {
        const EdgeEvaluation& previous = CachedEvaluation(
            emitted->second, [] { return EdgeEvaluation{}; });
        if (evaluation.cost < previous.cost) {
          emitted->second = edge_key;
        }
      }
      generated_goal_terminal =
          generated_goal_terminal || IsGoal(*target_state);
    }
    if (!generated_goal_terminal) {
      AppendGoalConnectorEdges(state, source, edges);
    }
  }

  [[nodiscard]] double Heuristic(const std::size_t state) const noexcept {
    if (state >= nodes_.size()) {
      return 0.0;
    }
    if (nodes_[state].exact_goal) {
      return 0.0;
    }
    return HeuristicForPose(nodes_[state].pose);
  }

  [[nodiscard]] double HeuristicForPose(const Pose3& pose) const noexcept {
    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t goal_index = 0U; goal_index < goals_.size();
         ++goal_index) {
      ActivateGoal(goal_index);
      minimum = std::min(minimum, HeuristicForActiveGoal(pose));
    }
    return std::isfinite(minimum) ? minimum : 0.0;
  }

  [[nodiscard]] double HeuristicForActiveGoal(
      const Pose3& pose) const noexcept {
    const double distance_to_region = std::max(
        0.0,
        std::hypot(pose.position_m.x - goal_.position_m.x,
                   pose.position_m.y - goal_.position_m.y) -
            goal_.tolerance_m);
    double yaw_error = 0.0;
    if (goal_yaw_.has_value()) {
      const auto yaw = YawFromQuaternion(pose.orientation);
      if (yaw.has_value()) {
        yaw_error = std::max(
            0.0, std::abs(ShortestYawDelta(*yaw, *goal_yaw_)) -
                     goal_yaw_tolerance_rad_);
      }
    }
    double barrier_distance_lower_bound = 0.0;
    const Vec2 planar_pose{.x = pose.position_m.x, .y = pose.position_m.y};
    const Vec2 planar_goal{.x = goal_.position_m.x,
                           .y = goal_.position_m.y};
    const auto& active_barriers =
        active_goal_index_ < barriers_by_goal_.size()
            ? barriers_by_goal_[active_goal_index_]
            : barriers_;
    for (const BarrierRectangle& barrier : active_barriers) {
      const double obstacle_distance =
          PointObstacleDistance(planar_pose, planar_goal, barrier);
      if (std::isfinite(obstacle_distance)) {
        barrier_distance_lower_bound =
            std::max(barrier_distance_lower_bound,
                     std::max(0.0, obstacle_distance - goal_.tolerance_m));
      }
    }
    const double distance_lower_bound =
        std::max(distance_to_region, barrier_distance_lower_bound);
    std::size_t displacement_edges = 0U;
    if (maximum_translation_reach_m_ > kTolerance) {
      displacement_edges = static_cast<std::size_t>(std::ceil(
          distance_to_region / maximum_translation_reach_m_ - kTolerance));
    }
    std::size_t obstacle_path_edges = 0U;
    if (maximum_translation_path_length_m_ > kTolerance) {
      obstacle_path_edges = static_cast<std::size_t>(std::ceil(
          barrier_distance_lower_bound /
              maximum_translation_path_length_m_ -
          kTolerance));
    }
    const std::size_t required_translation_edges =
        std::max(displacement_edges, obstacle_path_edges);
    const std::size_t zero_speed_full_edges =
        required_translation_edges > 2U ? required_translation_edges - 2U
                                        : 0U;
    const double translation_time_lower_bound =
        std::isfinite(minimum_zero_speed_translation_duration_s_)
            ? static_cast<double>(zero_speed_full_edges) *
                  minimum_zero_speed_translation_duration_s_
            : 0.0;
    const double yaw_time_lower_bound =
        yaw_error / capability_.maximum_spin_rate_radps;
    const double path_cost =
        kCostWeights[0] * distance_lower_bound / cost_scales_[0];
    const double execution_cost =
        kCostWeights[1] *
        std::max(translation_time_lower_bound, yaw_time_lower_bound) /
        cost_scales_[1];
    const double mode_cost =
        kCostWeights[4] * yaw_error / std::numbers::pi / cost_scales_[4];
    return path_cost + execution_cost + mode_cost;
  }

  [[nodiscard]] bool IsGoal(const std::size_t state) const noexcept {
    if (const auto edge = PreferredEdgeIndexForState(state);
        edge.has_value()) {
      return *edge + 1U == certified_preferred_candidate_->edges.size();
    }
    if (state >= nodes_.size()) {
      return false;
    }
    if (nodes_[state].exact_goal) {
      return true;
    }
    return MatchingGoalIndex(nodes_[state].pose).has_value();
  }

 private:
  static constexpr std::size_t kPreferredStateNamespace =
      std::size_t{1U}
      << (std::numeric_limits<std::size_t>::digits - 1U);
  static constexpr std::uint64_t kPreferredEdgeNamespace =
      0x5052454645525245ULL;

  void ActivateGoal(const std::size_t goal_index) const noexcept {
    active_goal_index_ = goal_index;
    goal_ = goals_[goal_index].point;
    goal_yaw_ = goals_[goal_index].yaw_rad;
    goal_yaw_tolerance_rad_ = goals_[goal_index].yaw_tolerance_rad;
  }

  [[nodiscard]] static std::size_t PreferredStateId(
      const std::size_t edge_index) noexcept {
    return kPreferredStateNamespace + edge_index;
  }

  [[nodiscard]] std::optional<std::size_t> PreferredEdgeIndexForState(
      const std::size_t state) const noexcept {
    if (state < kPreferredStateNamespace ||
        !certified_preferred_candidate_.has_value()) {
      return std::nullopt;
    }
    const std::size_t edge_index = state - kPreferredStateNamespace;
    if (edge_index >= certified_preferred_candidate_->edges.size()) {
      return std::nullopt;
    }
    return edge_index;
  }

  [[nodiscard]] std::size_t StablePreferredEdgeIndex(
      const std::size_t sequence,
      const PreferredEdgeRecord& edge) const noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::uint64_t value :
         {kPreferredEdgeNamespace,
          static_cast<std::uint64_t>(
              certified_preferred_candidate_->goal_index),
          static_cast<std::uint64_t>(edge.terminal_connector),
          static_cast<std::uint64_t>(edge.stable_primitive_rank),
          static_cast<std::uint64_t>(sequence)}) {
      for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
        hash ^= value & 0xffU;
        hash *= 1099511628211ULL;
        value >>= 8U;
      }
    }
    return static_cast<std::size_t>(hash);
  }

  [[nodiscard]] static std::size_t StablePreferredPathIndex(
      const std::vector<std::size_t>& stable_edges) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::size_t edge : stable_edges) {
      std::uint64_t value = static_cast<std::uint64_t>(edge);
      for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
        hash ^= value & 0xffU;
        hash *= 1099511628211ULL;
        value >>= 8U;
      }
    }
    return static_cast<std::size_t>(hash);
  }

  void BuildCertifiedInitialCandidate() {
    if (!certified_preferred_candidate_.has_value() ||
        certified_preferred_candidate_->edges.empty() ||
        certified_preferred_candidate_->edges.size() >=
            kPreferredStateNamespace) {
      return;
    }
    shared::SearchCandidate candidate;
    candidate.states.reserve(certified_preferred_candidate_->edges.size() +
                             1U);
    candidate.stable_edge_indices.reserve(
        certified_preferred_candidate_->edges.size());
    candidate.states.push_back(0U);
    for (std::size_t edge = 0U;
         edge < certified_preferred_candidate_->edges.size(); ++edge) {
      candidate.states.push_back(PreferredStateId(edge));
      candidate.stable_edge_indices.push_back(StablePreferredEdgeIndex(
          edge, certified_preferred_candidate_->edges[edge]));
    }
    candidate.cost = certified_preferred_candidate_->cost;
    candidate.stable_index =
        StablePreferredPathIndex(candidate.stable_edge_indices);
    certified_initial_candidate_ = std::move(candidate);
  }

  struct PreferredMacroGeometry final {
    Pose3 endpoint;
    double minimum_x{};
    double maximum_x{};
  };

  struct PreferredOption final {
    std::size_t straight_index{};
    std::size_t outward_first_index{};
    std::size_t outward_second_index{};
    std::size_t arc_repetitions{};
    std::size_t macro_repetitions{};
    std::size_t straight_repetitions{};
    std::size_t estimated_edges{};
    bool positive_side{};
  };

  [[nodiscard]] static Pose3 ComposePreferredEndpoint(
      const Pose3& source, const Pose3& relative) noexcept {
    const double yaw = YawFromQuaternion(source.orientation).value_or(0.0);
    const double relative_yaw =
        YawFromQuaternion(relative.orientation).value_or(0.0);
    return Pose3{
        .position_m =
            Vec3{.x = source.position_m.x +
                      std::cos(yaw) * relative.position_m.x -
                      std::sin(yaw) * relative.position_m.y,
                 .y = source.position_m.y +
                      std::sin(yaw) * relative.position_m.x +
                      std::cos(yaw) * relative.position_m.y,
                 .z = source.position_m.z + relative.position_m.z},
        .orientation =
            QuaternionFromYaw(NormalizeYaw(yaw + relative_yaw)),
    };
  }

  [[nodiscard]] PreferredMacroGeometry PreferredMacro(
      const std::size_t first_index, const std::size_t second_index,
      const std::size_t repetitions) const noexcept {
    PreferredMacroGeometry geometry{
        .endpoint = Pose3{.orientation = QuaternionFromYaw(0.0)}};
    const auto append = [&](const std::size_t primitive_index) {
      geometry.endpoint = ComposePreferredEndpoint(
          geometry.endpoint,
          capability_.motion_primitives[primitive_index].relative_end_pose);
      geometry.minimum_x =
          std::min(geometry.minimum_x, geometry.endpoint.position_m.x);
      geometry.maximum_x =
          std::max(geometry.maximum_x, geometry.endpoint.position_m.x);
    };
    for (std::size_t edge = 0U; edge < repetitions; ++edge) {
      append(first_index);
    }
    for (std::size_t edge = 0U; edge < repetitions; ++edge) {
      append(second_index);
    }
    return geometry;
  }

  [[nodiscard]] std::optional<CertifiedPreferredCandidate>
  EvaluatePreferredOption(const PreferredOption& option) const {
    if (option.estimated_edges == 0U ||
        option.estimated_edges > kMaximumPreferredTemplateEdges ||
        option.estimated_edges ==
            std::numeric_limits<std::size_t>::max()) {
      return std::nullopt;
    }
    CertifiedPreferredCandidate candidate;
    candidate.edges.reserve(option.estimated_edges + 1U);
    Pose3 endpoint = request_.start.pose;
    WheelMotionMode mode = WheelMotionMode::kStart;
    auto append_full = [&](const std::size_t primitive_index) {
      if (ControlInterrupted()) {
        return false;
      }
      const Node source{
          .key = WheelStateKey{.mode = mode},
          .pose = endpoint,
          .best_g = candidate.cost,
          .certified_clearance_m = std::numeric_limits<double>::infinity(),
          .creation_sequence = 0U,
          .expandable = false,
          .exact_goal = false,
      };
      auto transition = ApplyPrimitive(
          source, capability_.motion_primitives[primitive_index]);
      if (!transition.has_value()) {
        return false;
      }
      transition->initial_edge = candidate.edges.empty();
      ++preferred_validation_count_;
      EdgeEvaluation evaluation = Evaluate(
          *transition, false, EdgeCertificationKind::kFullPrimitive,
          primitive_stable_rank_[primitive_index]);
      if (!evaluation.valid) {
        return false;
      }
      candidate.cost += evaluation.cost;
      endpoint = evaluation.transition.target;
      mode = evaluation.transition.target_mode;
      candidate.edges.push_back(PreferredEdgeRecord{
          .endpoint = endpoint,
          .evaluation = std::move(evaluation),
          .cumulative_cost = candidate.cost,
          .stable_primitive_rank = primitive_stable_rank_[primitive_index],
          .terminal_connector = false,
      });
      ++candidate.full_primitive_edges;
      return true;
    };
    auto append_macro = [&](const std::size_t first,
                            const std::size_t second) {
      for (std::size_t macro = 0U; macro < option.macro_repetitions;
           ++macro) {
        for (std::size_t edge = 0U; edge < option.arc_repetitions; ++edge) {
          if (!append_full(first)) {
            return false;
          }
        }
        for (std::size_t edge = 0U; edge < option.arc_repetitions; ++edge) {
          if (!append_full(second)) {
            return false;
          }
        }
      }
      return true;
    };
    if (!append_macro(option.outward_first_index,
                      option.outward_second_index)) {
      return std::nullopt;
    }
    for (std::size_t edge = 0U; edge < option.straight_repetitions; ++edge) {
      if (!append_full(option.straight_index)) {
        return std::nullopt;
      }
    }
    if (!append_macro(option.outward_second_index,
                      option.outward_first_index)) {
      return std::nullopt;
    }
    if (PoseSatisfiesGoal(endpoint)) {
      return candidate;
    }
    const Node source{
        .key = WheelStateKey{.mode = mode},
        .pose = endpoint,
        .best_g = candidate.cost,
        .certified_clearance_m = std::numeric_limits<double>::infinity(),
        .creation_sequence = 0U,
        .expandable = false,
        .exact_goal = false,
    };
    for (const std::size_t primitive_index : ordered_primitives_) {
      if (ControlInterrupted()) {
        return std::nullopt;
      }
      const WheelMotionPrimitive& primitive =
          capability_.motion_primitives[primitive_index];
      const auto ratio = MatchingPrimitiveScale(source, primitive);
      if (!ratio.has_value()) {
        continue;
      }
      const auto transition = ScaledPrimitive(source, primitive, *ratio);
      if (!transition.has_value()) {
        continue;
      }
      ++preferred_validation_count_;
      EdgeEvaluation evaluation = Evaluate(
          *transition, false, EdgeCertificationKind::kScaledPrimitive,
          primitive_stable_rank_[primitive_index]);
      if (!evaluation.valid ||
          !PoseSatisfiesGoal(evaluation.transition.target)) {
        continue;
      }
      candidate.cost += evaluation.cost;
      candidate.edges.push_back(PreferredEdgeRecord{
          .endpoint = evaluation.transition.target,
          .evaluation = std::move(evaluation),
          .cumulative_cost = candidate.cost,
          .stable_primitive_rank =
              capability_.motion_primitives.size() +
              primitive_stable_rank_[primitive_index],
          .terminal_connector = true,
      });
      ++candidate.terminal_connector_edges;
      return candidate;
    }
    return std::nullopt;
  }

  void BuildCertifiedPreferredCandidate() {
    if (ControlInterrupted() || barriers_.empty()) {
      return;
    }
    constexpr double kGeometryTolerance = 1.0e-7;
    const Vec2 local_goal = lattice_frame_.ToLocalPosition(
        Vec2{.x = goal_.position_m.x, .y = goal_.position_m.y});
    const auto start_yaw = YawFromQuaternion(request_.start.pose.orientation);
    if (!start_yaw.has_value() || local_goal.x <= kGeometryTolerance ||
        std::abs(local_goal.y) > goal_.tolerance_m + kGeometryTolerance ||
        (goal_yaw_.has_value() &&
         std::abs(ShortestYawDelta(*start_yaw, *goal_yaw_)) >
             goal_yaw_tolerance_rad_ + kGeometryTolerance)) {
      return;
    }
    const Vec2 world_start{.x = request_.start.pose.position_m.x,
                           .y = request_.start.pose.position_m.y};
    const Vec2 world_goal{.x = goal_.position_m.x,
                          .y = goal_.position_m.y};
    double first_blocking_x = std::numeric_limits<double>::infinity();
    double last_blocking_x = -std::numeric_limits<double>::infinity();
    double minimum_blocking_y = std::numeric_limits<double>::infinity();
    double maximum_blocking_y = -std::numeric_limits<double>::infinity();
    for (const BarrierRectangle& barrier : barriers_) {
      if (ControlInterrupted()) {
        return;
      }
      if (!SegmentCrossesRectangleInterior(world_start, world_goal,
                                           barrier)) {
        continue;
      }
      for (const Vec2 corner : barrier.corners) {
        if (ControlInterrupted()) {
          return;
        }
        const Vec2 local = lattice_frame_.ToLocalPosition(corner);
        first_blocking_x = std::min(first_blocking_x, local.x);
        last_blocking_x = std::max(last_blocking_x, local.x);
        minimum_blocking_y = std::min(minimum_blocking_y, local.y);
        maximum_blocking_y = std::max(maximum_blocking_y, local.y);
      }
    }
    if (!std::isfinite(first_blocking_x) || first_blocking_x <= 0.0 ||
        last_blocking_x >= local_goal.x) {
      return;
    }
    std::vector<std::size_t> straights;
    std::vector<std::pair<std::size_t, std::size_t>> arc_pairs;
    for (std::size_t index = 0U;
         index < capability_.motion_primitives.size(); ++index) {
      if (ControlInterrupted()) {
        return;
      }
      const WheelMotionPrimitive& primitive =
          capability_.motion_primitives[index];
      const auto yaw = YawFromQuaternion(primitive.relative_end_pose.orientation);
      if (!yaw.has_value()) {
        continue;
      }
      const Vec3& position = primitive.relative_end_pose.position_m;
      if (primitive.kind == WheelPrimitiveKind::kForward &&
          position.x > kGeometryTolerance &&
          std::abs(position.y) <= kGeometryTolerance &&
          std::abs(position.z) <= kGeometryTolerance &&
          std::abs(*yaw) <= kGeometryTolerance) {
        straights.push_back(index);
      }
      if (primitive.kind != WheelPrimitiveKind::kForwardArc ||
          std::abs(*yaw) <= kGeometryTolerance) {
        continue;
      }
      for (std::size_t other = index + 1U;
           other < capability_.motion_primitives.size(); ++other) {
        if (ControlInterrupted()) {
          return;
        }
        const WheelMotionPrimitive& mirror =
            capability_.motion_primitives[other];
        const auto mirror_yaw =
            YawFromQuaternion(mirror.relative_end_pose.orientation);
        if (mirror.kind != WheelPrimitiveKind::kForwardArc ||
            !mirror_yaw.has_value()) {
          continue;
        }
        const Vec3& mirrored = mirror.relative_end_pose.position_m;
        if (std::abs(position.x - mirrored.x) <= kGeometryTolerance &&
            std::abs(position.y + mirrored.y) <= kGeometryTolerance &&
            std::abs(position.z - mirrored.z) <= kGeometryTolerance &&
            std::abs(*yaw + *mirror_yaw) <= kGeometryTolerance) {
          arc_pairs.emplace_back(index, other);
        }
      }
    }
    if (straights.empty() || arc_pairs.empty()) {
      return;
    }
    const double longitudinal_margin =
        footprint_radius_m_ + capability_.minimum_clearance_m;
    const double lateral_rotation_margin = std::max(
        0.0, footprint_radius_m_ - barrier_inset_radius_m_);
    const double shift_before_x = first_blocking_x - longitudinal_margin;
    const double return_after_x = last_blocking_x + longitudinal_margin;
    std::vector<PreferredOption> options;
    for (const std::size_t straight_index : straights) {
      if (ControlInterrupted()) {
        return;
      }
      const double straight_reach =
          capability_.motion_primitives[straight_index]
              .relative_end_pose.position_m.x;
      for (const auto [first, second] : arc_pairs) {
        if (ControlInterrupted()) {
          return;
        }
        const double first_yaw = YawFromQuaternion(
            capability_.motion_primitives[first].relative_end_pose.orientation)
                                     .value_or(0.0);
        const std::size_t positive = first_yaw > 0.0 ? first : second;
        const std::size_t negative = first_yaw > 0.0 ? second : first;
        const double yaw_step = std::abs(first_yaw);
        if (yaw_step <= kGeometryTolerance) {
          continue;
        }
        const std::size_t half_cycle = std::min(
            yaw_bins_ / 2U,
            static_cast<std::size_t>(std::floor(
                std::numbers::pi / yaw_step + kGeometryTolerance)));
        for (std::size_t arc_repetitions = 1U;
             arc_repetitions <= half_cycle; ++arc_repetitions) {
          if (ControlInterrupted()) {
            return;
          }
          for (const bool positive_side : {true, false}) {
            if (ControlInterrupted()) {
              return;
            }
            const std::size_t outward_first =
                positive_side ? positive : negative;
            const std::size_t outward_second =
                positive_side ? negative : positive;
            const PreferredMacroGeometry outward = PreferredMacro(
                outward_first, outward_second, arc_repetitions);
            const PreferredMacroGeometry inward = PreferredMacro(
                outward_second, outward_first, arc_repetitions);
            const double outward_yaw = YawFromQuaternion(
                outward.endpoint.orientation).value_or(0.0);
            if (std::abs(outward_yaw) > kGeometryTolerance ||
                outward.endpoint.position_m.x < -kGeometryTolerance ||
                (positive_side &&
                 outward.endpoint.position_m.y <= kGeometryTolerance) ||
                (!positive_side &&
                 outward.endpoint.position_m.y >= -kGeometryTolerance)) {
              continue;
            }
            const double required_lateral =
                positive_side
                    ? maximum_blocking_y + lateral_rotation_margin
                    : -minimum_blocking_y + lateral_rotation_margin;
            const double lateral_progress =
                std::abs(outward.endpoint.position_m.y);
            if (required_lateral <= 0.0 ||
                lateral_progress <= kGeometryTolerance) {
              continue;
            }
            const double repeat_value = std::ceil(
                required_lateral / lateral_progress - kGeometryTolerance);
            if (!std::isfinite(repeat_value) || repeat_value < 1.0 ||
                repeat_value >
                    static_cast<double>(
                        std::numeric_limits<std::size_t>::max())) {
              continue;
            }
            const std::size_t macro_repetitions =
                static_cast<std::size_t>(repeat_value);
            const double shift_maximum_x =
                static_cast<double>(macro_repetitions - 1U) *
                    outward.endpoint.position_m.x +
                outward.maximum_x;
            if (shift_maximum_x > shift_before_x + kGeometryTolerance) {
              continue;
            }
            const double arc_longitudinal =
                static_cast<double>(macro_repetitions) *
                (outward.endpoint.position_m.x +
                 inward.endpoint.position_m.x);
            const double straight_distance = local_goal.x - arc_longitudinal;
            if (straight_distance < -kGeometryTolerance) {
              continue;
            }
            const double straight_repeat_value = std::floor(
                std::max(0.0, straight_distance) / straight_reach +
                kGeometryTolerance);
            if (!std::isfinite(straight_repeat_value) ||
                straight_repeat_value < 0.0 ||
                straight_repeat_value >
                    static_cast<double>(kMaximumPreferredTemplateEdges)) {
              continue;
            }
            const std::size_t straight_repetitions =
                static_cast<std::size_t>(straight_repeat_value);
            const double inward_start_x =
                static_cast<double>(macro_repetitions) *
                    outward.endpoint.position_m.x +
                static_cast<double>(straight_repetitions) * straight_reach;
            const double inward_minimum_x =
                inward_start_x + inward.minimum_x;
            if (inward_minimum_x + kGeometryTolerance < return_after_x) {
              continue;
            }
            if (arc_repetitions >
                    std::numeric_limits<std::size_t>::max() / 4U ||
                macro_repetitions >
                    std::numeric_limits<std::size_t>::max() /
                        (4U * arc_repetitions)) {
              continue;
            }
            const std::size_t arc_edges =
                4U * arc_repetitions * macro_repetitions;
            if (arc_edges > kMaximumPreferredTemplateEdges ||
                straight_repetitions >
                    kMaximumPreferredTemplateEdges - arc_edges) {
              continue;
            }
            options.push_back(PreferredOption{
                .straight_index = straight_index,
                .outward_first_index = outward_first,
                .outward_second_index = outward_second,
                .arc_repetitions = arc_repetitions,
                .macro_repetitions = macro_repetitions,
                .straight_repetitions = straight_repetitions,
                .estimated_edges = arc_edges + straight_repetitions,
                .positive_side = positive_side,
            });
          }
        }
      }
    }
    std::stable_sort(options.begin(), options.end(), [&](const auto& left,
                                                         const auto& right) {
      return std::tuple{
                 left.estimated_edges,
                 primitive_stable_rank_[left.straight_index],
                 primitive_stable_rank_[left.outward_first_index],
                 primitive_stable_rank_[left.outward_second_index],
                 left.arc_repetitions, left.macro_repetitions,
                 !left.positive_side} <
             std::tuple{
                 right.estimated_edges,
                 primitive_stable_rank_[right.straight_index],
                 primitive_stable_rank_[right.outward_first_index],
                 primitive_stable_rank_[right.outward_second_index],
                 right.arc_repetitions, right.macro_repetitions,
                 !right.positive_side};
    });
    std::size_t evaluated_templates = 0U;
    for (const PreferredOption& option : options) {
      if (ControlInterrupted()) {
        return;
      }
      if (evaluated_templates == kMaximumPreferredTemplatesEvaluated) {
        return;
      }
      ++evaluated_templates;
      auto candidate = EvaluatePreferredOption(option);
      if (candidate.has_value()) {
        certified_preferred_candidate_ = std::move(candidate);
        return;
      }
    }
  }

  [[nodiscard]] bool PoseSatisfiesGoal(const Pose3& pose) const noexcept {
    const auto yaw = YawFromQuaternion(pose.orientation);
    const double position_error = std::hypot(
        pose.position_m.x - goal_.position_m.x,
        pose.position_m.y - goal_.position_m.y);
    if (!yaw.has_value() || position_error > goal_.tolerance_m + kTolerance ||
        (goal_yaw_.has_value() &&
         std::abs(ShortestYawDelta(*yaw, *goal_yaw_)) >
             goal_yaw_tolerance_rad_ + kTolerance)) {
      return false;
    }
    return true;
  }

  [[nodiscard]] std::optional<std::size_t> MatchingGoalIndex(
      const Pose3& pose) const noexcept {
    for (std::size_t goal_index = 0U; goal_index < goals_.size();
         ++goal_index) {
      ActivateGoal(goal_index);
      if (PoseSatisfiesGoal(pose)) {
        return goal_index;
      }
    }
    return std::nullopt;
  }
  [[nodiscard]] bool ControlInterrupted() const {
    if (request_.control.canceled()) {
      return true;
    }
    if (request_.control.expired()) {
      return true;
    }
    return request_.control.canceled();
  }

  [[nodiscard]] bool HasOccupiedInCells(
      const std::int64_t minimum_x, const std::int64_t minimum_y,
      const std::int64_t maximum_x,
      const std::int64_t maximum_y) const noexcept {
    if (minimum_x < 0 || minimum_y < 0 || maximum_x < minimum_x ||
        maximum_y < minimum_y ||
        maximum_x >= static_cast<std::int64_t>(map_.width()) ||
        maximum_y >= static_cast<std::int64_t>(map_.height())) {
      return true;
    }
    const std::size_t stride = map_.width() + 1U;
    const std::size_t left = static_cast<std::size_t>(minimum_x);
    const std::size_t top = static_cast<std::size_t>(minimum_y);
    const std::size_t right = static_cast<std::size_t>(maximum_x) + 1U;
    const std::size_t bottom = static_cast<std::size_t>(maximum_y) + 1U;
    const std::uint32_t count =
        occupied_integral_[bottom * stride + right] -
        occupied_integral_[top * stride + right] -
        occupied_integral_[bottom * stride + left] +
        occupied_integral_[top * stride + left];
    return count != 0U;
  }

  [[nodiscard]] bool HasComplexTerrainInCells(
      const std::int64_t minimum_x, const std::int64_t minimum_y,
      const std::int64_t maximum_x,
      const std::int64_t maximum_y) const noexcept {
    if (minimum_x < 0 || minimum_y < 0 || maximum_x < minimum_x ||
        maximum_y < minimum_y ||
        maximum_x >= static_cast<std::int64_t>(map_.width()) ||
        maximum_y >= static_cast<std::int64_t>(map_.height())) {
      return true;
    }
    const std::size_t stride = map_.width() + 1U;
    const std::size_t left = static_cast<std::size_t>(minimum_x);
    const std::size_t top = static_cast<std::size_t>(minimum_y);
    const std::size_t right = static_cast<std::size_t>(maximum_x) + 1U;
    const std::size_t bottom = static_cast<std::size_t>(maximum_y) + 1U;
    const std::uint32_t count =
        complex_terrain_integral_[bottom * stride + right] -
        complex_terrain_integral_[top * stride + right] -
        complex_terrain_integral_[bottom * stride + left] +
        complex_terrain_integral_[top * stride + left];
    return count != 0U;
  }

  template <class EvaluateFn>
  const EdgeEvaluation& CachedEvaluation(const EdgeKey& key,
                                         EvaluateFn&& evaluate) {
    ++validation_requests_;
    return validation_cache_.GetOrEvaluate(
        key, std::forward<EvaluateFn>(evaluate));
  }

  [[nodiscard]] bool IsNarrow(const Pose3& pose) const noexcept {
    const auto cell = map_.PositionToCell(
        Vec2{.x = pose.position_m.x, .y = pose.position_m.y});
    if (!cell.has_value()) {
      return false;
    }
    const float clearance =
        terrain_.narrow_band_distance_m[map_.Index(*cell)];
    return std::isfinite(clearance) &&
           static_cast<double>(clearance) <
               footprint_radius_m_ + 2.0 * map_.resolution_m();
  }

  [[nodiscard]] std::size_t SelectYawBins() const noexcept {
    constexpr std::array<std::size_t, 5U> candidates{
        16U, 32U, 64U, 128U, 256U};
    for (const std::size_t bins : candidates) {
      const double bin_width =
          2.0 * std::numbers::pi / static_cast<double>(bins);
      const bool compatible = std::ranges::all_of(
          capability_.motion_primitives,
          [&](const WheelMotionPrimitive& primitive) {
            const double delta = std::abs(YawFromQuaternion(
                primitive.relative_end_pose.orientation).value_or(0.0));
            if (delta <= kTolerance) {
              return true;
            }
            const double steps = std::round(delta / bin_width);
            return std::abs(delta - steps * bin_width) <= 1.0e-9;
          });
      if (compatible) {
        return bins;
      }
    }
    return candidates.back();
  }

  [[nodiscard]] WheelStateKey Quantize(const Pose3& pose,
                                       const WheelMotionMode mode) {
    const bool narrow = IsNarrow(pose);
    used_narrow_resolution_ = used_narrow_resolution_ || narrow;
    const double xy_resolution =
        narrow ? map_.resolution_m() / 2.0 : map_.resolution_m();
    finest_xy_key_resolution_m_ =
        std::min(finest_xy_key_resolution_m_, xy_resolution);
    maximum_yaw_bins_ = std::max(maximum_yaw_bins_, yaw_bins_);
    const auto yaw = YawFromQuaternion(pose.orientation).value_or(0.0);
    const Vec2 local_position = lattice_frame_.ToLocalPosition(
        Vec2{.x = pose.position_m.x, .y = pose.position_m.y});
    const double local_yaw = lattice_frame_.ToLocalYaw(yaw);
    const double positive_yaw = local_yaw < 0.0
                                    ? local_yaw + 2.0 * std::numbers::pi
                                    : local_yaw;
    const auto yaw_bin = static_cast<std::size_t>(std::llround(
        positive_yaw * static_cast<double>(yaw_bins_) /
        (2.0 * std::numbers::pi))) % yaw_bins_;
    return WheelStateKey{
        .x = static_cast<std::int64_t>(std::llround(
            local_position.x / xy_resolution)),
        .y = static_cast<std::int64_t>(std::llround(
            local_position.y / xy_resolution)),
        .yaw = static_cast<std::uint16_t>(yaw_bin),
        .mode = mode,
        .narrow = narrow,
    };
  }

  [[nodiscard]] double XYResidual(const Node& node) const noexcept {
    const double xy_resolution =
        node.key.narrow ? map_.resolution_m() / 2.0 : map_.resolution_m();
    const Vec2 local = lattice_frame_.ToLocalPosition(
        Vec2{.x = node.pose.position_m.x, .y = node.pose.position_m.y});
    return std::hypot(
        local.x - static_cast<double>(node.key.x) * xy_resolution,
        local.y - static_cast<double>(node.key.y) * xy_resolution);
  }

  [[nodiscard]] double YawResidual(const Node& node) const noexcept {
    const double yaw =
        YawFromQuaternion(node.pose.orientation).value_or(0.0);
    const double local_yaw = lattice_frame_.ToLocalYaw(yaw);
    const double bin_yaw =
        static_cast<double>(node.key.yaw) * 2.0 * std::numbers::pi /
        static_cast<double>(yaw_bins_);
    return std::abs(ShortestYawDelta(bin_yaw, local_yaw));
  }

  [[nodiscard]] static RejectedFingerprint Fingerprint(
      const WheelStateKey& key, const Pose3& pose,
      const std::uint32_t terminal_goal_mask) noexcept {
    constexpr double resolution = kPhysicalMatchTolerance;
    const double yaw = YawFromQuaternion(pose.orientation).value_or(0.0);
    return RejectedFingerprint{
        .key = key,
        .x = static_cast<std::int64_t>(
            std::llround(pose.position_m.x / resolution)),
        .y = static_cast<std::int64_t>(
            std::llround(pose.position_m.y / resolution)),
        .yaw = static_cast<std::int64_t>(std::llround(yaw / resolution)),
        .terminal_goal_mask = terminal_goal_mask,
    };
  }

  void RecordRejected(const Node& node) {
    const RejectedFingerprint fingerprint = Fingerprint(
        node.key, node.pose, node.certified_terminal_goal_mask);
    const auto [record, inserted] =
        rejected_best_g_.try_emplace(fingerprint, node.best_g);
    if (!inserted) {
      record->second = std::min(record->second, node.best_g);
    }
  }

  [[nodiscard]] bool LabelRanksBefore(const Node& left,
                                      const Node& right) const noexcept {
    const double left_anchor = left.best_g + HeuristicForPose(left.pose);
    const double right_anchor = right.best_g + HeuristicForPose(right.pose);
    return std::tuple{
               left_anchor, left.best_g, -left.certified_clearance_m,
               XYResidual(left), YawResidual(left), left.creation_sequence} <
           std::tuple{
               right_anchor, right.best_g, -right.certified_clearance_m,
               XYResidual(right), YawResidual(right),
               right.creation_sequence};
  }

  void RerankLabels(const WheelStateKey& key,
                    const std::size_t candidate) {
    auto& active = state_ids_[key];
    std::vector<std::size_t> ranked = active;
    if (std::ranges::find(ranked, candidate) == ranked.end()) {
      ranked.push_back(candidate);
    }
    for (const std::size_t state : ranked) {
      nodes_[state].expandable = false;
    }
    std::stable_sort(ranked.begin(), ranked.end(), [&](const std::size_t lhs,
                                                       const std::size_t rhs) {
      return LabelRanksBefore(nodes_[lhs], nodes_[rhs]);
    });
    if (ranked.size() > kMaximumActiveLabelsPerKey) {
      std::vector<std::size_t> retained(
          ranked.begin(),
          ranked.begin() +
              static_cast<std::ptrdiff_t>(kMaximumActiveLabelsPerKey));
      std::vector<std::uint32_t> retained_terminal_signatures;
      for (const std::size_t state : retained) {
        const std::uint32_t signature =
            nodes_[state].certified_terminal_goal_mask;
        if (signature != 0U &&
            std::ranges::find(retained_terminal_signatures, signature) ==
                retained_terminal_signatures.end()) {
          retained_terminal_signatures.push_back(signature);
        }
      }
      for (const std::size_t state : ranked) {
        const std::uint32_t signature =
            nodes_[state].certified_terminal_goal_mask;
        if (signature == 0U ||
            std::ranges::find(retained_terminal_signatures, signature) !=
                retained_terminal_signatures.end()) {
          continue;
        }
        const auto replace = std::ranges::find_if(
            retained.rbegin(), retained.rend(), [&](const std::size_t kept) {
              const std::uint32_t kept_signature =
                  nodes_[kept].certified_terminal_goal_mask;
              return kept_signature == 0U ||
                     std::ranges::count_if(
                         retained, [&](const std::size_t other) {
                           return nodes_[other]
                                      .certified_terminal_goal_mask ==
                                  kept_signature;
                         }) > 1;
            });
        if (replace == retained.rend()) {
          break;
        }
        *replace = state;
        retained_terminal_signatures.push_back(signature);
      }
      std::stable_sort(retained.begin(), retained.end(),
                       [&](const std::size_t lhs, const std::size_t rhs) {
                         return LabelRanksBefore(nodes_[lhs], nodes_[rhs]);
                       });
      for (const std::size_t state : ranked) {
        if (std::ranges::find(retained, state) == retained.end()) {
          RecordRejected(nodes_[state]);
        }
      }
      ranked = std::move(retained);
    }
    for (const std::size_t state : ranked) {
      nodes_[state].expandable = true;
    }
    active = std::move(ranked);
    maximum_active_labels_per_key_ =
        std::max(maximum_active_labels_per_key_, active.size());
  }

  [[nodiscard]] std::optional<std::size_t> Intern(
      const WheelStateKey& key, const Pose3& pose, const double best_g,
      const double certified_clearance_m) {
    auto& active = state_ids_[key];
    const double xy_resolution =
        key.narrow ? map_.resolution_m() / 2.0 : map_.resolution_m();
    const double yaw_threshold =
        0.25 * 2.0 * std::numbers::pi / static_cast<double>(yaw_bins_);
    const auto pose_yaw = YawFromQuaternion(pose.orientation).value_or(0.0);
    const auto physically_matches = [&](const Node& retained,
                                        const double xy_threshold,
                                        const double allowed_yaw) {
      const auto retained_yaw =
          YawFromQuaternion(retained.pose.orientation).value_or(0.0);
      return std::hypot(retained.pose.position_m.x - pose.position_m.x,
                        retained.pose.position_m.y - pose.position_m.y) <=
                 xy_threshold + kTolerance &&
             std::abs(ShortestYawDelta(retained_yaw, pose_yaw)) <=
                 allowed_yaw + kTolerance;
    };
    Node candidate{
        .key = key,
        .pose = pose,
        .best_g = best_g,
        .certified_clearance_m = certified_clearance_m,
        .creation_sequence = next_creation_sequence_,
        .expandable = true,
        .exact_goal = false,
    };
    candidate.certified_terminal_goal_mask =
        CertifiedTerminalGoalMask(candidate);
    const bool active_has_signature = std::ranges::any_of(
        active, [&](const std::size_t state) {
          return nodes_[state].certified_terminal_goal_mask ==
                 candidate.certified_terminal_goal_mask;
        });
    for (const std::size_t retained_state : active) {
      const Node& retained = nodes_[retained_state];
      if (physically_matches(retained, 0.25 * xy_resolution, yaw_threshold) &&
          retained.certified_terminal_goal_mask ==
              candidate.certified_terminal_goal_mask &&
          retained.best_g <= best_g) {
        ++quantized_state_reuses_;
        return std::nullopt;
      }
    }
    const RejectedFingerprint fingerprint = Fingerprint(
        key, pose, candidate.certified_terminal_goal_mask);
    if (const auto rejected = rejected_best_g_.find(fingerprint);
        rejected != rejected_best_g_.end() && rejected->second <= best_g &&
        !(candidate.certified_terminal_goal_mask != 0U &&
          !active_has_signature)) {
      ++quantized_state_reuses_;
      return std::nullopt;
    }
    if (active.size() >= kMaximumActiveLabelsPerKey) {
      const std::size_t labels_before_candidate =
          static_cast<std::size_t>(std::ranges::count_if(
              active, [&](const std::size_t state) {
                return LabelRanksBefore(nodes_[state], candidate);
              }));
      if (labels_before_candidate >= kMaximumActiveLabelsPerKey &&
          !(candidate.certified_terminal_goal_mask != 0U &&
            !active_has_signature)) {
        candidate.expandable = false;
        RecordRejected(candidate);
        ++next_creation_sequence_;
        return std::nullopt;
      }
    }
    const std::size_t state = nodes_.size();
    ++next_creation_sequence_;
    nodes_.push_back(candidate);
    RerankLabels(key, state);
    return state;
  }

  [[nodiscard]] std::optional<Transition> ApplyPrimitive(
      const Node& source,
      const WheelMotionPrimitive& primitive) const noexcept {
    if (!ModeAllows(source.key.mode, primitive.kind)) {
      return std::nullopt;
    }
    const auto source_yaw = YawFromQuaternion(source.pose.orientation);
    const auto relative_yaw =
        YawFromQuaternion(primitive.relative_end_pose.orientation);
    if (!source_yaw.has_value() || !relative_yaw.has_value()) {
      return std::nullopt;
    }
    const double cosine = std::cos(*source_yaw);
    const double sine = std::sin(*source_yaw);
    const Vec3& relative = primitive.relative_end_pose.position_m;
    Pose3 target{
        .position_m = Vec3{
            .x = source.pose.position_m.x + cosine * relative.x -
                 sine * relative.y,
            .y = source.pose.position_m.y + sine * relative.x +
                 cosine * relative.y,
            .z = source.pose.position_m.z + relative.z,
        },
        .orientation = QuaternionFromYaw(
            NormalizeYaw(*source_yaw + *relative_yaw)),
    };
    if (primitive.kind == WheelPrimitiveKind::kStopAndSwitch) {
      target = source.pose;
    }
    if (!map_.PositionToCell(Vec2{.x = target.position_m.x,
                                  .y = target.position_m.y})
             .has_value()) {
      return std::nullopt;
    }
    const auto ground = map_.SampleElevationBilinear(
        Vec2{.x = target.position_m.x, .y = target.position_m.y});
    if (!ground.has_value()) {
      return std::nullopt;
    }
    target.position_m.z = *ground;
    Transition transition{
        .source = source.pose,
        .target = target,
        .source_mode = source.key.mode,
        .target_mode = TargetMode(primitive.kind),
        .kind = primitive.kind,
        .reverse = IsReverse(primitive.kind),
    };
    RecomputeGeometry(transition);
    if (transition.path_length_m <= kTolerance &&
        std::abs(transition.yaw_delta_rad) <= kTolerance &&
        transition.source_mode == transition.target_mode) {
      return std::nullopt;
    }
    return transition;
  }

  [[nodiscard]] static bool PrimitiveReachesTarget(
      const Node& source, const WheelMotionPrimitive& primitive,
      const Pose3& target) noexcept {
    const auto source_yaw = YawFromQuaternion(source.pose.orientation);
    const auto target_yaw = YawFromQuaternion(target.orientation);
    const auto primitive_twist =
        LogRelativePose(primitive.relative_end_pose);
    if (!source_yaw.has_value() || !target_yaw.has_value() ||
        !primitive_twist.has_value()) {
      return false;
    }
    const double cosine = std::cos(*source_yaw);
    const double sine = std::sin(*source_yaw);
    const double world_x = target.position_m.x - source.pose.position_m.x;
    const double world_y = target.position_m.y - source.pose.position_m.y;
    const Pose3 target_relative{
        .position_m = Vec3{
            .x = cosine * world_x + sine * world_y,
            .y = -sine * world_x + cosine * world_y,
            .z = 0.0,
        },
        .orientation = QuaternionFromYaw(
            ShortestYawDelta(*source_yaw, *target_yaw)),
    };
    if (primitive.kind == WheelPrimitiveKind::kStopAndSwitch) {
      return std::hypot(target_relative.position_m.x,
                        target_relative.position_m.y) <=
                 kPhysicalMatchTolerance &&
             std::abs(ShortestYawDelta(*source_yaw, *target_yaw)) <=
                 kPhysicalMatchTolerance;
    }
    double scale = 0.0;
    if (std::abs(primitive_twist->yaw_rate) > kTolerance) {
      scale = ShortestYawDelta(*source_yaw, *target_yaw) /
              primitive_twist->yaw_rate;
    } else {
      const double squared_motion =
          primitive_twist->velocity_x * primitive_twist->velocity_x +
          primitive_twist->velocity_y * primitive_twist->velocity_y;
      if (squared_motion <= kTolerance) {
        return false;
      }
      scale =
          (target_relative.position_m.x * primitive_twist->velocity_x +
           target_relative.position_m.y * primitive_twist->velocity_y) /
          squared_motion;
    }
    if (!std::isfinite(scale) || scale <= kTolerance ||
        scale > 1.0 + kTolerance) {
      return false;
    }
    const Pose3 reached =
        ExpRelativePose(*primitive_twist, std::clamp(scale, 0.0, 1.0));
    const auto reached_yaw = YawFromQuaternion(reached.orientation);
    return reached_yaw.has_value() &&
           std::hypot(reached.position_m.x - target_relative.position_m.x,
                      reached.position_m.y - target_relative.position_m.y) <=
               kPhysicalMatchTolerance &&
           std::abs(ShortestYawDelta(*reached_yaw,
                                     ShortestYawDelta(*source_yaw,
                                                      *target_yaw))) <=
               kPhysicalMatchTolerance;
  }

  static void RecomputeGeometry(Transition& transition) noexcept {
    const auto source_yaw =
        YawFromQuaternion(transition.source.orientation).value_or(0.0);
    const auto target_yaw =
        YawFromQuaternion(transition.target.orientation).value_or(0.0);
    transition.yaw_delta_rad = ShortestYawDelta(source_yaw, target_yaw);
    const double cosine = std::cos(source_yaw);
    const double sine = std::sin(source_yaw);
    const double world_x =
        transition.target.position_m.x - transition.source.position_m.x;
    const double world_y =
        transition.target.position_m.y - transition.source.position_m.y;
    const Pose3 relative{
        .position_m = Vec3{
            .x = cosine * world_x + sine * world_y,
            .y = -sine * world_x + cosine * world_y,
            .z = transition.target.position_m.z -
                 transition.source.position_m.z,
        },
        .orientation = QuaternionFromYaw(transition.yaw_delta_rad),
    };
    const auto twist = LogRelativePose(relative);
    transition.path_length_m =
        twist.has_value()
            ? std::hypot(twist->velocity_x, twist->velocity_y)
            : std::numeric_limits<double>::infinity();
    transition.curvature_per_m =
        transition.path_length_m > kTolerance && !IsSpin(transition.kind)
            ? transition.yaw_delta_rad / transition.path_length_m
            : 0.0;
  }

  [[nodiscard]] static bool MotionKindMatches(
      const Transition& transition) noexcept {
    const auto source_yaw = YawFromQuaternion(transition.source.orientation);
    if (!source_yaw.has_value()) {
      return false;
    }
    const double cosine = std::cos(*source_yaw);
    const double sine = std::sin(*source_yaw);
    const double world_x =
        transition.target.position_m.x - transition.source.position_m.x;
    const double world_y =
        transition.target.position_m.y - transition.source.position_m.y;
    const Pose3 relative{
        .position_m = Vec3{
            .x = cosine * world_x + sine * world_y,
            .y = -sine * world_x + cosine * world_y,
            .z = transition.target.position_m.z -
                 transition.source.position_m.z,
        },
        .orientation = QuaternionFromYaw(transition.yaw_delta_rad),
    };
    const auto twist = LogRelativePose(relative);
    if (!twist.has_value()) {
      return false;
    }
    const double travel = std::hypot(twist->velocity_x, twist->velocity_y);
    if (transition.kind == WheelPrimitiveKind::kStopAndSwitch) {
      return travel <= kPhysicalMatchTolerance &&
             std::abs(twist->yaw_rate) <= kPhysicalMatchTolerance;
    }
    if (IsSpin(transition.kind)) {
      const bool direction_matches =
          transition.kind == WheelPrimitiveKind::kSpinCounterclockwise
              ? twist->yaw_rate > kTolerance
              : twist->yaw_rate < -kTolerance;
      return travel <= kPhysicalMatchTolerance && direction_matches;
    }
    if (IsForward(transition.kind) && twist->velocity_x <= kTolerance) {
      return false;
    }
    if (IsReverse(transition.kind) && twist->velocity_x >= -kTolerance) {
      return false;
    }
    if ((transition.kind == WheelPrimitiveKind::kForward ||
         transition.kind == WheelPrimitiveKind::kReverse) &&
        (std::abs(twist->velocity_y) > kPhysicalMatchTolerance ||
         std::abs(twist->yaw_rate) > kPhysicalMatchTolerance)) {
      return false;
    }
    return true;
  }

  [[nodiscard]] static std::optional<Pose3> Interpolate(
      const Transition& transition, const double ratio) noexcept {
    const auto source_yaw = YawFromQuaternion(transition.source.orientation);
    const auto target_yaw = YawFromQuaternion(transition.target.orientation);
    if (!source_yaw.has_value() || !target_yaw.has_value()) {
      return std::nullopt;
    }
    const double cosine = std::cos(*source_yaw);
    const double sine = std::sin(*source_yaw);
    const double world_x =
        transition.target.position_m.x - transition.source.position_m.x;
    const double world_y =
        transition.target.position_m.y - transition.source.position_m.y;
    const Pose3 relative{
        .position_m = Vec3{
            .x = cosine * world_x + sine * world_y,
            .y = -sine * world_x + cosine * world_y,
            .z = transition.target.position_m.z -
                 transition.source.position_m.z,
        },
        .orientation = QuaternionFromYaw(
            ShortestYawDelta(*source_yaw, *target_yaw)),
    };
    const auto twist = LogRelativePose(relative);
    if (!twist.has_value()) {
      return std::nullopt;
    }
    const Pose3 scaled = ExpRelativePose(*twist, ratio);
    return Pose3{
        .position_m = Vec3{
            .x = transition.source.position_m.x +
                 cosine * scaled.position_m.x - sine * scaled.position_m.y,
            .y = transition.source.position_m.y +
                 sine * scaled.position_m.x + cosine * scaled.position_m.y,
            .z = transition.source.position_m.z +
                 ratio * (transition.target.position_m.z -
                          transition.source.position_m.z),
        },
        .orientation = QuaternionFromYaw(
            NormalizeYaw(*source_yaw + ratio * twist->yaw_rate)),
    };
  }

  [[nodiscard]] EdgeCertificateIdentity MakeCertificateIdentity(
      const Transition& transition, const bool pose_only,
      const EdgeCertificationKind certification_kind,
      const std::size_t primitive_stable_rank) const noexcept {
    const std::uint64_t capability_fingerprint =
        request_.capability_fingerprint != 0U
            ? request_.capability_fingerprint
            : capability_fingerprint_;
    return EdgeCertificateIdentity{
        .local_source_sequence = request_.local_source_sequence,
        .local_terrain_semantics_id =
            request_.local_terrain_semantics_id,
        .zero_sequence_request_identity =
            request_.local_source_sequence == 0U
                ? reinterpret_cast<std::uintptr_t>(&request_)
                : 0U,
        .map_metadata_bits = map_metadata_bits_,
        .capability_fingerprint = capability_fingerprint,
        .source_pose_bits = PoseBits(transition.source),
        .target_pose_bits = PoseBits(transition.target),
        .initial_velocity_bits = TwistBits(request_.start.velocity),
        .source_mode = transition.source_mode,
        .target_mode = transition.target_mode,
        .primitive_kind = transition.kind,
        .certification_kind = certification_kind,
        .primitive_stable_rank = primitive_stable_rank,
        .initial_edge = transition.initial_edge,
        .pose_only = pose_only,
    };
  }

  [[nodiscard]] BroadPhaseAssessment AssessBroadPhase(
      const Transition& transition, const bool pose_only) const {
    BroadPhaseAssessment assessment;
    if (!Finite(transition.source) || !Finite(transition.target) ||
        !std::isfinite(transition.path_length_m) ||
        transition.path_length_m < 0.0 ||
        !std::isfinite(transition.yaw_delta_rad) ||
        !std::isfinite(transition.curvature_per_m) ||
        std::abs(transition.curvature_per_m) >
            capability_.maximum_curvature_per_m + kTolerance ||
        (!pose_only &&
         (!ModeAllows(transition.source_mode, transition.kind) ||
          transition.reverse != IsReverse(transition.kind)))) {
      assessment.rejected = true;
      assessment.rejection_bucket =
          RejectionBucket::kDynamicsOrPrimitiveShape;
      return assessment;
    }
    const double swept_distance =
        transition.path_length_m +
        std::abs(transition.yaw_delta_rad) * footprint_radius_m_;
    const double maximum_step = map_.resolution_m() / 4.0;
    if (!std::isfinite(swept_distance) || swept_distance < 0.0 ||
        !std::isfinite(maximum_step) || maximum_step <= 0.0) {
      assessment.rejected = true;
      assessment.rejection_bucket =
          RejectionBucket::kDynamicsOrPrimitiveShape;
      return assessment;
    }
    const std::size_t subdivisions =
        pose_only
            ? 0U
            : std::max<std::size_t>(
                  1U, static_cast<std::size_t>(
                          std::ceil(swept_distance / maximum_step)));
    for (std::size_t sample = 0U; sample <= subdivisions; ++sample) {
      if (ControlInterrupted()) {
        assessment.interrupted = true;
        assessment.rejection_bucket =
            RejectionBucket::kDeadlineOrCancellation;
        return assessment;
      }
      const double ratio =
          subdivisions == 0U
              ? 0.0
              : static_cast<double>(sample) /
                    static_cast<double>(subdivisions);
      const auto pose = Interpolate(transition, ratio);
      const auto yaw = pose.has_value()
                           ? YawFromQuaternion(pose->orientation)
                           : std::nullopt;
      if (!pose.has_value() || !yaw.has_value()) {
        assessment.rejected = true;
        assessment.rejection_bucket =
            RejectionBucket::kDynamicsOrPrimitiveShape;
        return assessment;
      }
      const double cosine = std::cos(*yaw);
      const double sine = std::sin(*yaw);
      double minimum_x = std::numeric_limits<double>::infinity();
      double minimum_y = std::numeric_limits<double>::infinity();
      double maximum_x = -std::numeric_limits<double>::infinity();
      double maximum_y = -std::numeric_limits<double>::infinity();
      for (const Vec2& vertex : capability_.footprint_xy_m) {
        const double x = pose->position_m.x + cosine * vertex.x -
                         sine * vertex.y;
        const double y = pose->position_m.y + sine * vertex.x +
                         cosine * vertex.y;
        minimum_x = std::min(minimum_x, x);
        minimum_y = std::min(minimum_y, y);
        maximum_x = std::max(maximum_x, x);
        maximum_y = std::max(maximum_y, y);
      }
      const double margin = capability_.minimum_clearance_m;
      if (!FootprintWithinClosedMapExtent(
              map_, minimum_x, minimum_y, maximum_x, maximum_y)) {
        assessment.rejected = true;
        assessment.rejection_bucket =
            RejectionBucket::kDirectUnknownOrUnsupportedFootprint;
        return assessment;
      }
      const CellWindow occupied_window = ClipOccupiedWindow(
          map_, minimum_x - margin, minimum_y - margin,
          maximum_x + margin, maximum_y + margin);
      if (broad_inset_radius_m_ > kTolerance &&
          HasOccupiedInCells(
              occupied_window.minimum_x, occupied_window.minimum_y,
              occupied_window.maximum_x, occupied_window.maximum_y)) {
        const double rejection_radius =
            broad_inset_radius_m_ + capability_.minimum_clearance_m;
        for (std::int64_t y = occupied_window.minimum_y;
             y <= occupied_window.maximum_y; ++y) {
          const auto& row = occupied_by_row_[static_cast<std::size_t>(y)];
          const auto begin = std::ranges::lower_bound(
              row, static_cast<std::int32_t>(occupied_window.minimum_x));
          const auto finish = std::ranges::upper_bound(
              row, static_cast<std::int32_t>(occupied_window.maximum_x));
          for (auto hazard = begin; hazard != finish; ++hazard) {
            const double cell_minimum_x =
                map_.origin_m().x +
                static_cast<double>(*hazard) * map_.resolution_m();
            const double cell_minimum_y =
                map_.origin_m().y +
                static_cast<double>(y) * map_.resolution_m();
            const double dx = std::max(
                {cell_minimum_x - pose->position_m.x, 0.0,
                 pose->position_m.x -
                     (cell_minimum_x + map_.resolution_m())});
            const double dy = std::max(
                {cell_minimum_y - pose->position_m.y, 0.0,
                 pose->position_m.y -
                     (cell_minimum_y + map_.resolution_m())});
            const double distance = std::hypot(dx, dy);
            const bool intersects_inset =
                distance <= broad_inset_radius_m_ + kTolerance;
            const bool violates_clearance =
                capability_.minimum_clearance_m > kTolerance &&
                distance < rejection_radius - kTolerance;
            if (intersects_inset || violates_clearance) {
              assessment.rejected = true;
              assessment.rejection_bucket =
                  RejectionBucket::kMeasuredObstacleClearance;
              return assessment;
            }
          }
        }
      }
      const auto center_cell = map_.PositionToCell(
          Vec2{.x = pose->position_m.x, .y = pose->position_m.y});
      if (!center_cell.has_value()) {
        assessment.rejected = true;
        assessment.rejection_bucket =
            RejectionBucket::kDirectUnknownOrUnsupportedFootprint;
        return assessment;
      }
      const std::size_t center_index = map_.Index(*center_cell);
      if (footprint_contains_origin_ &&
          terrain_.free_with_height[center_index] == 0U) {
        assessment.rejected = true;
        assessment.rejection_bucket =
            RejectionBucket::kDirectUnknownOrUnsupportedFootprint;
        return assessment;
      }
    }
    return assessment;
  }

  [[nodiscard]] EdgeEvaluation EvaluateFullExact(
      const Transition& transition, const bool pose_only) const {
    EdgeEvaluation result{.transition = transition};
    if (std::abs(transition.curvature_per_m) >
        capability_.maximum_curvature_per_m + kTolerance) {
      result.rejection_bucket =
          RejectionBucket::kDynamicsOrPrimitiveShape;
      return result;
    }
    const double swept_distance =
        transition.path_length_m +
        std::abs(transition.yaw_delta_rad) * footprint_radius_m_;
    const double maximum_step = map_.resolution_m() / 4.0;
    const std::size_t subdivisions = pose_only
                                         ? 0U
                                         : std::max<std::size_t>(
                                               1U, static_cast<std::size_t>(
                                                       std::ceil(swept_distance /
                                                                 maximum_step)));
    const std::size_t samples = subdivisions + 1U;
    const auto elevations = map_.FloatLayer("elevation");
    std::vector<Vec2> polygon;
    polygon.reserve(capability_.footprint_xy_m.size());
    std::vector<std::pair<shared::GridCell, double>> terrain_samples;
    terrain_samples.reserve(capability_.footprint_xy_m.size() * 4U);
    for (std::size_t sample = 0U; sample < samples; ++sample) {
      if (ControlInterrupted()) {
        result.rejection_bucket =
            RejectionBucket::kDeadlineOrCancellation;
        return result;
      }
      const double ratio = subdivisions == 0U
                               ? 0.0
                               : static_cast<double>(sample) /
                                     static_cast<double>(subdivisions);
      const auto pose = Interpolate(transition, ratio);
      if (!pose.has_value()) {
        result.rejection_bucket =
            RejectionBucket::kDynamicsOrPrimitiveShape;
        return result;
      }
      const double center_x = pose->position_m.x;
      const double center_y = pose->position_m.y;
      const double yaw = YawFromQuaternion(pose->orientation).value_or(0.0);
      const double cosine = std::cos(yaw);
      const double sine = std::sin(yaw);
      polygon.clear();
      double minimum_x = std::numeric_limits<double>::infinity();
      double minimum_y = std::numeric_limits<double>::infinity();
      double maximum_x = -std::numeric_limits<double>::infinity();
      double maximum_y = -std::numeric_limits<double>::infinity();
      for (const Vec2& vertex : capability_.footprint_xy_m) {
        const Vec2 transformed{
            .x = center_x + cosine * vertex.x - sine * vertex.y,
            .y = center_y + sine * vertex.x + cosine * vertex.y,
        };
        polygon.push_back(transformed);
        minimum_x = std::min(minimum_x, transformed.x);
        minimum_y = std::min(minimum_y, transformed.y);
        maximum_x = std::max(maximum_x, transformed.x);
        maximum_y = std::max(maximum_y, transformed.y);
      }
      const double clearance_margin = capability_.minimum_clearance_m;
      const double narrow_threshold =
          footprint_radius_m_ + 2.0 * map_.resolution_m();
      const double clearance_scan_margin =
          std::max(clearance_margin, narrow_threshold);
      if (!FootprintWithinClosedMapExtent(
              map_, minimum_x, minimum_y, maximum_x, maximum_y)) {
        result.rejection_bucket =
            RejectionBucket::kDirectUnknownOrUnsupportedFootprint;
        return result;
      }
      const CellWindow clearance_window = ClipOccupiedWindow(
          map_, minimum_x - clearance_scan_margin,
          minimum_y - clearance_scan_margin,
          maximum_x + clearance_scan_margin,
          maximum_y + clearance_scan_margin);
      if (HasOccupiedInCells(
              clearance_window.minimum_x, clearance_window.minimum_y,
              clearance_window.maximum_x, clearance_window.maximum_y)) {
        for (std::int64_t y = clearance_window.minimum_y;
             y <= clearance_window.maximum_y; ++y) {
          const auto& row = occupied_by_row_[static_cast<std::size_t>(y)];
          const auto begin = std::ranges::lower_bound(
              row, static_cast<std::int32_t>(clearance_window.minimum_x));
          const auto finish = std::ranges::upper_bound(
              row, static_cast<std::int32_t>(clearance_window.maximum_x));
          for (auto hazard = begin; hazard != finish; ++hazard) {
            const std::int64_t x = *hazard;
            ++sweep_cell_checks_;
            if ((sweep_cell_checks_ & 63U) == 0U && ControlInterrupted()) {
              result.rejection_bucket =
                  RejectionBucket::kDeadlineOrCancellation;
              return result;
            }
            const double cell_minimum_x =
                map_.origin_m().x +
                static_cast<double>(x) * map_.resolution_m();
            const double cell_minimum_y =
                map_.origin_m().y +
                static_cast<double>(y) * map_.resolution_m();
            ++occupied_clearance_cell_checks_;
            const double distance = PolygonDistanceToCell(
                polygon, cell_minimum_x, cell_minimum_y,
                cell_minimum_x + map_.resolution_m(),
                cell_minimum_y + map_.resolution_m());
            result.minimum_clearance_m =
                std::min(result.minimum_clearance_m, distance);
            if (distance + kTolerance < clearance_margin ||
                distance <= kTolerance) {
              result.rejection_bucket =
                  RejectionBucket::kMeasuredObstacleClearance;
              return result;
            }
          }
        }
      }

      const CellWindow footprint_window = ClosedFootprintContactWindow(
          map_, minimum_x, minimum_y, maximum_x, maximum_y);
      const std::int64_t minimum_cell_x = footprint_window.minimum_x;
      const std::int64_t minimum_cell_y = footprint_window.minimum_y;
      const std::int64_t maximum_cell_x = footprint_window.maximum_x;
      const std::int64_t maximum_cell_y = footprint_window.maximum_y;

      std::array<BilinearSupportNeighborhood, 4U> wheel_supports{};
      std::size_t wheel_index = 0U;
      for (const double body_x : {-0.5 * capability_.wheelbase_m,
                                  0.5 * capability_.wheelbase_m}) {
        for (const double body_y : {-0.5 * capability_.track_width_m,
                                    0.5 * capability_.track_width_m}) {
          const Vec2 wheel_position{
              .x = center_x + cosine * body_x - sine * body_y,
              .y = center_y + sine * body_x + cosine * body_y,
          };
          const auto support = KnownFreeBilinearSupportNeighborhood(
              map_, terrain_, elevations, wheel_position);
          if (!support.has_value()) {
            result.rejection_bucket =
                RejectionBucket::kDirectUnknownOrUnsupportedFootprint;
            return result;
          }
          wheel_supports[wheel_index++] = *support;
        }
      }

      const bool footprint_proof_has_complex_terrain =
          HasComplexTerrainInCells(
              std::max<std::int64_t>(0, minimum_cell_x - 1),
              std::max<std::int64_t>(0, minimum_cell_y - 1),
              std::min<std::int64_t>(
                  static_cast<std::int64_t>(map_.width()) - 1,
                  maximum_cell_x + 1),
              std::min<std::int64_t>(
                  static_cast<std::int64_t>(map_.height()) - 1,
                  maximum_cell_y + 1));
      bool wheel_support_proof_has_complex_terrain = false;
      for (const BilinearSupportNeighborhood& support : wheel_supports) {
        for (std::size_t index = 0U; index < support.count; ++index) {
          const std::size_t support_index = support.indices[index];
          const std::int64_t support_x = static_cast<std::int64_t>(
              support_index % map_.width());
          const std::int64_t support_y = static_cast<std::int64_t>(
              support_index / map_.width());
          if (HasComplexTerrainInCells(support_x, support_y, support_x,
                                       support_y)) {
            wheel_support_proof_has_complex_terrain = true;
            break;
          }
        }
        if (wheel_support_proof_has_complex_terrain) {
          break;
        }
      }

      // The exact bilinear support cells, including cells outside the
      // footprint, must be flat before taking the fast path. Their support
      // neighborhoods were already sampled and validated above.
      if (!footprint_proof_has_complex_terrain &&
          !wheel_support_proof_has_complex_terrain) {
        if (!std::isfinite(result.minimum_clearance_m)) {
          result.minimum_clearance_m = narrow_threshold;
        }
        continue;
      }

      std::array<double, 4U> wheel_heights{};
      for (std::size_t wheel = 0U; wheel < wheel_supports.size(); ++wheel) {
        const BilinearSupportNeighborhood& support = wheel_supports[wheel];
        double height = 0.0;
        for (std::size_t index = 0U; index < support.count; ++index) {
          height += support.weights[index] * elevations[support.indices[index]];
        }
        wheel_heights[wheel] = height / support.total_weight;
        if (!std::isfinite(wheel_heights[wheel])) {
          result.rejection_bucket =
              RejectionBucket::kDirectUnknownOrUnsupportedFootprint;
          return result;
        }
      }
      const double rear_height = 0.5 * (wheel_heights[0] + wheel_heights[1]);
      const double front_height = 0.5 * (wheel_heights[2] + wheel_heights[3]);
      const double negative_y_height =
          0.5 * (wheel_heights[0] + wheel_heights[2]);
      const double positive_y_height =
          0.5 * (wheel_heights[1] + wheel_heights[3]);
      const double support_slope_x =
          (front_height - rear_height) / capability_.wheelbase_m;
      const double support_slope_y =
          (positive_y_height - negative_y_height) /
          capability_.track_width_m;
      const double support_height =
          0.25 * (wheel_heights[0] + wheel_heights[1] + wheel_heights[2] +
                  wheel_heights[3]);
      const double support_slope =
          std::atan(std::hypot(support_slope_x, support_slope_y));
      if (!std::isfinite(support_slope) ||
          support_slope > capability_.maximum_slope_rad +
                              kPhysicalMatchTolerance) {
        result.rejection_bucket = RejectionBucket::kSlopeOrRoughness;
        return result;
      }
      result.maximum_slope_rad =
          std::max(result.maximum_slope_rad, support_slope);

      terrain_samples.clear();
      bool touched_cell = false;
      double maximum_positive_relief = 0.0;
      for (std::int64_t y = minimum_cell_y; y <= maximum_cell_y; ++y) {
        for (std::int64_t x = minimum_cell_x; x <= maximum_cell_x; ++x) {
          ++sweep_cell_checks_;
          if ((sweep_cell_checks_ & 63U) == 0U && ControlInterrupted()) {
            result.rejection_bucket =
                RejectionBucket::kDeadlineOrCancellation;
            return result;
          }
          const double cell_minimum_x =
              map_.origin_m().x + static_cast<double>(x) * map_.resolution_m();
          const double cell_minimum_y =
              map_.origin_m().y + static_cast<double>(y) * map_.resolution_m();
          if (!PolygonIntersectsCell(
                  polygon, cell_minimum_x, cell_minimum_y,
                  cell_minimum_x + map_.resolution_m(),
                  cell_minimum_y + map_.resolution_m())) {
            continue;
          }
          touched_cell = true;
          const shared::GridCell cell{
              .x = static_cast<std::int32_t>(x),
              .y = static_cast<std::int32_t>(y),
          };
          const std::size_t index = map_.Index(cell);
          if (index >= terrain_.free_with_height.size() ||
              terrain_.free_with_height[index] == 0U ||
              index >= elevations.size() || !std::isfinite(elevations[index])) {
            result.rejection_bucket =
                RejectionBucket::kDirectUnknownOrUnsupportedFootprint;
            return result;
          }
          if (!std::isfinite(terrain_.slope_rad[index]) ||
              terrain_.slope_rad[index] >
                  capability_.maximum_slope_rad + kTolerance ||
              !std::isfinite(terrain_.roughness_m[index])) {
            result.rejection_bucket = RejectionBucket::kSlopeOrRoughness;
            return result;
          }
          const double cell_center_x =
              cell_minimum_x + 0.5 * map_.resolution_m();
          const double cell_center_y =
              cell_minimum_y + 0.5 * map_.resolution_m();
          const double body_x = cosine * (cell_center_x - center_x) +
                                sine * (cell_center_y - center_y);
          const double body_y = -sine * (cell_center_x - center_x) +
                                cosine * (cell_center_y - center_y);
          const double plane_height = support_height +
                                      support_slope_x * body_x +
                                      support_slope_y * body_y;
          const double height = static_cast<double>(elevations[index]);
          maximum_positive_relief =
              std::max(maximum_positive_relief, height - plane_height);
          terrain_samples.emplace_back(cell, height);
          result.maximum_slope_rad = std::max(
              result.maximum_slope_rad,
              static_cast<double>(terrain_.slope_rad[index]));
          result.maximum_roughness_m = std::max(
              result.maximum_roughness_m,
              static_cast<double>(terrain_.roughness_m[index]));
        }
      }
      if (!touched_cell) {
        result.rejection_bucket =
            RejectionBucket::kDirectUnknownOrUnsupportedFootprint;
        return result;
      }
      if (maximum_positive_relief >
              capability_.maximum_local_obstacle_relief_m +
                  kPhysicalMatchTolerance ||
          maximum_positive_relief >
              capability_.minimum_underbody_clearance_m +
                  kPhysicalMatchTolerance) {
        result.rejection_bucket = RejectionBucket::kReliefOrUnderbody;
        return result;
      }
      const double maximum_continuous_step =
          std::tan(capability_.maximum_slope_rad) * map_.resolution_m();
      std::size_t terrain_pair_checks = 0U;
      for (std::size_t first = 0U; first < terrain_samples.size(); ++first) {
        for (std::size_t second = first + 1U;
             second < terrain_samples.size(); ++second) {
          ++terrain_pair_checks;
          if ((terrain_pair_checks & 63U) == 0U && ControlInterrupted()) {
            result.rejection_bucket =
                RejectionBucket::kDeadlineOrCancellation;
            return result;
          }
          const auto& [first_cell, first_height] = terrain_samples[first];
          const auto& [second_cell, second_height] = terrain_samples[second];
          const std::int64_t separation =
              std::abs(static_cast<std::int64_t>(first_cell.x) -
                       static_cast<std::int64_t>(second_cell.x)) +
              std::abs(static_cast<std::int64_t>(first_cell.y) -
                       static_cast<std::int64_t>(second_cell.y));
          if (separation == 1 &&
              std::abs(first_height - second_height) >
                  maximum_continuous_step + kTolerance) {
            result.rejection_bucket = RejectionBucket::kReliefOrUnderbody;
            return result;
          }
        }
      }
      if (!std::isfinite(result.minimum_clearance_m)) {
        result.minimum_clearance_m = narrow_threshold;
      }
    }
    result.valid = true;
    result.execution_time_s =
        pose_only ? 0.0 : ExecutionDuration(transition, &result.initial_speed);
    if (!std::isfinite(result.execution_time_s) ||
        result.execution_time_s < 0.0) {
      result.valid = false;
      result.rejection_bucket =
          RejectionBucket::kDynamicsOrPrimitiveShape;
      return result;
    }
    result.cost = pose_only ? 0.0 : EdgeCost(result);
    if (!std::isfinite(result.cost) || (!pose_only && result.cost <= 0.0)) {
      result.valid = false;
      result.rejection_bucket =
          RejectionBucket::kDynamicsOrPrimitiveShape;
    }
    return result;
  }

  [[nodiscard]] EdgeEvaluation Evaluate(
      const Transition& transition, const bool pose_only,
      const EdgeCertificationKind certification_kind,
      const std::size_t primitive_stable_rank) const {
    const EdgeCertificateIdentity identity = MakeCertificateIdentity(
        transition, pose_only, certification_kind, primitive_stable_rank);
    const BroadPhaseAssessment broad =
        AssessBroadPhase(transition, pose_only);
    if (broad.rejected || broad.interrupted) {
      broad_phase_rejects_ += static_cast<std::size_t>(broad.rejected);
      RecordRejection(broad.rejection_bucket);
      return EdgeEvaluation{.transition = transition,
                            .certificate_identity = identity};
    }
    ++full_certifications_;
    EdgeEvaluation result = EvaluateFullExact(transition, pose_only);
    result.certificate_identity = identity;
    if (!result.valid && !ControlInterrupted()) {
      ++full_invalidations_;
    }
    if (!result.valid) {
      RecordRejection(result.rejection_bucket);
    }
    return result;
  }

  void RecordRejection(const RejectionBucket bucket) const noexcept {
    switch (bucket) {
      case RejectionBucket::kDirectUnknownOrUnsupportedFootprint:
        ++direct_unknown_or_unsupported_footprint_rejects_;
        break;
      case RejectionBucket::kMeasuredObstacleClearance:
        ++measured_obstacle_clearance_rejects_;
        break;
      case RejectionBucket::kSlopeOrRoughness:
        ++slope_or_roughness_rejects_;
        break;
      case RejectionBucket::kReliefOrUnderbody:
        ++relief_or_underbody_rejects_;
        break;
      case RejectionBucket::kDynamicsOrPrimitiveShape:
        ++dynamics_or_primitive_shape_rejects_;
        break;
      case RejectionBucket::kDeadlineOrCancellation:
        ++deadline_or_cancellation_interruptions_;
        break;
      case RejectionBucket::kNone:
        break;
    }
  }

  [[nodiscard]] static std::optional<SpeedProfile> MakeProfile(
      const double distance, const double initial_speed,
      const double speed_limit, const double acceleration,
      const double deceleration) noexcept {
    if (!std::isfinite(distance) || distance <= kTolerance ||
        !std::isfinite(initial_speed) || initial_speed < 0.0 ||
        !std::isfinite(speed_limit) || speed_limit <= 0.0 ||
        !std::isfinite(acceleration) || acceleration <= 0.0 ||
        !std::isfinite(deceleration) || deceleration <= 0.0 ||
        initial_speed > speed_limit + kTolerance) {
      return std::nullopt;
    }
    const double peak_squared =
        (2.0 * acceleration * deceleration * distance +
         deceleration * initial_speed * initial_speed) /
        (acceleration + deceleration);
    const double peak = std::min(speed_limit, std::sqrt(peak_squared));
    if (!std::isfinite(peak) || peak + kTolerance < initial_speed) {
      return std::nullopt;
    }
    const double acceleration_distance = std::max(
        0.0, (peak * peak - initial_speed * initial_speed) /
                 (2.0 * acceleration));
    const double deceleration_distance =
        peak * peak / (2.0 * deceleration);
    const double cruise_distance = std::max(
        0.0, distance - acceleration_distance - deceleration_distance);
    SpeedProfile profile{
        .distance = distance,
        .initial_speed = initial_speed,
        .peak_speed = peak,
        .acceleration = acceleration,
        .deceleration = deceleration,
        .acceleration_time = (peak - initial_speed) / acceleration,
        .cruise_time = cruise_distance / std::max(peak, kTolerance),
        .deceleration_time = peak / deceleration,
    };
    return std::isfinite(profile.duration()) &&
                   profile.duration() > kTolerance
               ? std::optional<SpeedProfile>{profile}
               : std::nullopt;
  }

  [[nodiscard]] static ProfileSample SampleProfile(
      const SpeedProfile& profile, const double time) noexcept {
    const double clamped_time = std::clamp(time, 0.0, profile.duration());
    if (clamped_time <= profile.acceleration_time) {
      return ProfileSample{
          .distance = profile.initial_speed * clamped_time +
                      0.5 * profile.acceleration * clamped_time * clamped_time,
          .speed = profile.initial_speed +
                   profile.acceleration * clamped_time,
      };
    }
    const double acceleration_distance =
        profile.initial_speed * profile.acceleration_time +
        0.5 * profile.acceleration * profile.acceleration_time *
            profile.acceleration_time;
    if (clamped_time <=
        profile.acceleration_time + profile.cruise_time) {
      const double cruise_elapsed =
          clamped_time - profile.acceleration_time;
      return ProfileSample{
          .distance = acceleration_distance +
                      profile.peak_speed * cruise_elapsed,
          .speed = profile.peak_speed,
      };
    }
    const double deceleration_elapsed =
        clamped_time - profile.acceleration_time - profile.cruise_time;
    return ProfileSample{
        .distance = std::min(
            profile.distance,
            acceleration_distance +
                profile.peak_speed * profile.cruise_time +
                profile.peak_speed * deceleration_elapsed -
                0.5 * profile.deceleration * deceleration_elapsed *
                    deceleration_elapsed),
        .speed = std::max(
            0.0, profile.peak_speed -
                     profile.deceleration * deceleration_elapsed),
    };
  }

  [[nodiscard]] double TranslationSpeedLimit(
      const Transition& transition) const noexcept {
    double speed = transition.reverse
                       ? capability_.maximum_reverse_speed_mps
                       : capability_.maximum_forward_speed_mps;
    const double curvature = std::abs(transition.curvature_per_m);
    if (curvature > kTolerance) {
      speed = std::min(
          {speed, capability_.maximum_spin_rate_radps / curvature,
           std::sqrt(capability_.maximum_lateral_acceleration_mps2 /
                     curvature)});
    }
    return speed;
  }

  [[nodiscard]] double TranslationAccelerationLimit(
      const Transition& transition) const noexcept {
    const double curvature = std::abs(transition.curvature_per_m);
    return curvature > kTolerance
               ? std::min(capability_.maximum_acceleration_mps2,
                          capability_.maximum_yaw_acceleration_radps2 /
                              curvature)
               : capability_.maximum_acceleration_mps2;
  }

  [[nodiscard]] double TranslationBrakingLimit(
      const Transition& transition) const noexcept {
    const double curvature = std::abs(transition.curvature_per_m);
    return curvature > kTolerance
               ? std::min(capability_.maximum_braking_deceleration_mps2,
                          capability_.maximum_yaw_acceleration_radps2 /
                              curvature)
               : capability_.maximum_braking_deceleration_mps2;
  }

  [[nodiscard]] std::optional<double> InitialTranslationSpeed(
      const Transition& transition) const noexcept {
    if (!transition.initial_edge ||
        transition.path_length_m <= kTolerance) {
      return 0.0;
    }
    const auto yaw = YawFromQuaternion(transition.source.orientation);
    if (!yaw.has_value()) {
      return std::nullopt;
    }
    const double direction = transition.reverse ? -1.0 : 1.0;
    const Vec3 tangent{
        .x = direction * std::cos(*yaw),
        .y = direction * std::sin(*yaw),
        .z = 0.0,
    };
    const Vec3& linear = request_.start.velocity.linear_mps;
    const Vec3& angular = request_.start.velocity.angular_radps;
    const double speed = linear.x * tangent.x + linear.y * tangent.y;
    const double lateral_x = linear.x - speed * tangent.x;
    const double lateral_y = linear.y - speed * tangent.y;
    const double expected_yaw_rate =
        speed * transition.curvature_per_m;
    if (speed < -kPhysicalMatchTolerance ||
        speed > TranslationSpeedLimit(transition) +
                    kPhysicalMatchTolerance ||
        std::hypot(lateral_x, lateral_y) > kPhysicalMatchTolerance ||
        std::abs(linear.z) > kPhysicalMatchTolerance ||
        std::abs(angular.x) > kPhysicalMatchTolerance ||
        std::abs(angular.y) > kPhysicalMatchTolerance ||
        std::abs(angular.z - expected_yaw_rate) >
            kPhysicalMatchTolerance) {
      return std::nullopt;
    }
    return std::max(0.0, speed);
  }

  [[nodiscard]] std::optional<double> InitialAngularSpeed(
      const Transition& transition) const noexcept {
    if (!transition.initial_edge) {
      return 0.0;
    }
    const Vec3& linear = request_.start.velocity.linear_mps;
    const Vec3& angular = request_.start.velocity.angular_radps;
    const double direction = std::copysign(1.0, transition.yaw_delta_rad);
    const double speed = direction * angular.z;
    if (std::hypot(linear.x, linear.y) > kPhysicalMatchTolerance ||
        std::abs(linear.z) > kPhysicalMatchTolerance ||
        std::abs(angular.x) > kPhysicalMatchTolerance ||
        std::abs(angular.y) > kPhysicalMatchTolerance ||
        speed < -kPhysicalMatchTolerance ||
        speed > capability_.maximum_spin_rate_radps +
                    kPhysicalMatchTolerance) {
      return std::nullopt;
    }
    return std::max(0.0, speed);
  }

  [[nodiscard]] double ExecutionDuration(
      const Transition& transition, double* initial_speed) const noexcept {
    if (initial_speed == nullptr) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    *initial_speed = 0.0;
    if (transition.path_length_m > kTolerance) {
      const auto start_speed = InitialTranslationSpeed(transition);
      if (!start_speed.has_value()) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      *initial_speed = *start_speed;
      const auto profile = MakeProfile(
          transition.path_length_m, *initial_speed,
          TranslationSpeedLimit(transition),
          TranslationAccelerationLimit(transition),
          TranslationBrakingLimit(transition));
      return profile.has_value()
                 ? profile->duration()
                 : std::numeric_limits<double>::quiet_NaN();
    }
    if (std::abs(transition.yaw_delta_rad) > kTolerance) {
      const auto start_speed = InitialAngularSpeed(transition);
      if (!start_speed.has_value()) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      *initial_speed = *start_speed;
      const auto profile = MakeProfile(
          std::abs(transition.yaw_delta_rad), *initial_speed,
          capability_.maximum_spin_rate_radps,
          capability_.maximum_yaw_acceleration_radps2,
          capability_.maximum_yaw_acceleration_radps2);
      return profile.has_value()
                 ? profile->duration()
                 : std::numeric_limits<double>::quiet_NaN();
    }
    return 1.0e-3;
  }

  [[nodiscard]] double EdgeCost(EdgeEvaluation& edge) const noexcept {
    const Transition& transition = edge.transition;
    const double terrain_cost =
        edge.maximum_slope_rad + edge.maximum_roughness_m /
                                     platform_length_scale_m_;
    const double clearance_reference = std::max(
        {footprint_radius_m_, capability_.minimum_clearance_m,
         kPhysicalMatchTolerance});
    const double low_clearance_cost =
        std::isfinite(edge.minimum_clearance_m)
            ? std::max(0.0,
                       clearance_reference - edge.minimum_clearance_m) /
                  clearance_reference
            : 0.0;
    const double mode_cost =
        std::abs(transition.yaw_delta_rad) / std::numbers::pi +
        (transition.reverse ? 0.25 : 0.0) +
        (transition.source_mode != transition.target_mode ? 0.1 : 0.0);
    edge.cost_components = {
        transition.path_length_m,
        edge.execution_time_s,
        terrain_cost,
        low_clearance_cost,
        mode_cost,
    };
    double normalized = 0.0;
    for (std::size_t component = 0U; component < cost_scales_.size();
         ++component) {
      normalized += kCostWeights[component] * edge.cost_components[component] /
                    cost_scales_[component];
    }
    return std::max(kMinimumEdgeCost, normalized);
  }

  static void AddScaleInterval(std::vector<ScaleInterval>& intervals,
                               double lower, double upper) {
    lower = std::max(0.0, lower);
    upper = std::min(1.0, upper);
    if (lower > upper + kTolerance) {
      return;
    }
    if (!intervals.empty() &&
        lower <= intervals.back().upper + kTolerance) {
      intervals.back().upper = std::max(intervals.back().upper, upper);
      return;
    }
    intervals.push_back(ScaleInterval{.lower = lower, .upper = upper});
  }

  [[nodiscard]] std::vector<ScaleInterval> PositionScaleIntervals(
      const RelativeTwist& twist, const double target_x,
      const double target_y) const {
    // A scaled primitive is a line or circular arc.  Split the arc at its
    // distance extrema so every subinterval has at most one region boundary.
    const double radius = goal_.tolerance_m + kTolerance;
    const auto error = [&](const double ratio) {
      const Pose3 relative = ExpRelativePose(twist, ratio);
      return std::hypot(relative.position_m.x - target_x,
                        relative.position_m.y - target_y);
    };
    std::vector<ScaleInterval> intervals;
    if (std::abs(twist.yaw_rate) <= kTolerance) {
      const double squared_speed = twist.velocity_x * twist.velocity_x +
                                   twist.velocity_y * twist.velocity_y;
      if (squared_speed <= kTolerance) {
        return intervals;
      }
      const double projection =
          (target_x * twist.velocity_x + target_y * twist.velocity_y) /
          squared_speed;
      const double closest_x = projection * twist.velocity_x;
      const double closest_y = projection * twist.velocity_y;
      const double residual_squared =
          (closest_x - target_x) * (closest_x - target_x) +
          (closest_y - target_y) * (closest_y - target_y);
      const double radius_squared = radius * radius;
      if (residual_squared > radius_squared + kTolerance) {
        return intervals;
      }
      const double half_width = std::sqrt(
          std::max(0.0, radius_squared - residual_squared) / squared_speed);
      AddScaleInterval(intervals, projection - half_width,
                       projection + half_width);
      return intervals;
    }

    const double yaw_rate = twist.yaw_rate;
    const double center_x = -twist.velocity_y / yaw_rate;
    const double center_y = twist.velocity_x / yaw_rate;
    const double radial_x = twist.velocity_y / yaw_rate;
    const double radial_y = -twist.velocity_x / yaw_rate;
    const double offset_x = center_x - target_x;
    const double offset_y = center_y - target_y;
    const double cosine_coefficient =
        offset_x * radial_x + offset_y * radial_y;
    const double sine_coefficient =
        -offset_x * radial_y + offset_y * radial_x;
    std::vector<double> breakpoints{0.0, 1.0};
    if (std::hypot(cosine_coefficient, sine_coefficient) > kTolerance) {
      const double base =
          std::atan2(sine_coefficient, cosine_coefficient);
      const double minimum_theta = std::min(0.0, yaw_rate);
      const double maximum_theta = std::max(0.0, yaw_rate);
      const auto first_turn = static_cast<std::int64_t>(
          std::floor((minimum_theta - base) / std::numbers::pi)) - 1;
      const auto last_turn = static_cast<std::int64_t>(
          std::ceil((maximum_theta - base) / std::numbers::pi)) + 1;
      for (std::int64_t turn = first_turn; turn <= last_turn; ++turn) {
        const double ratio =
            (base + static_cast<double>(turn) * std::numbers::pi) /
            yaw_rate;
        if (ratio > 0.0 && ratio < 1.0) {
          breakpoints.push_back(ratio);
        }
      }
    }
    std::ranges::sort(breakpoints);
    breakpoints.erase(
        std::unique(breakpoints.begin(), breakpoints.end(),
                    [](const double lhs, const double rhs) {
                      return std::abs(lhs - rhs) <= kTolerance;
                    }),
        breakpoints.end());
    const auto inside = [&](const double ratio) {
      return error(ratio) <= radius;
    };
    for (std::size_t index = 1U; index < breakpoints.size(); ++index) {
      const double lower = breakpoints[index - 1U];
      const double upper = breakpoints[index];
      const bool lower_inside = inside(lower);
      const bool upper_inside = inside(upper);
      if (lower_inside && upper_inside) {
        AddScaleInterval(intervals, lower, upper);
        continue;
      }
      if (lower_inside == upper_inside) {
        continue;
      }
      double outside = lower_inside ? upper : lower;
      double within = lower_inside ? lower : upper;
      for (std::size_t iteration = 0U; iteration < 64U; ++iteration) {
        const double middle = 0.5 * (outside + within);
        if (inside(middle)) {
          within = middle;
        } else {
          outside = middle;
        }
      }
      const double boundary = within;
      AddScaleInterval(intervals, lower_inside ? lower : boundary,
                       lower_inside ? boundary : upper);
    }
    return intervals;
  }

  [[nodiscard]] std::vector<ScaleInterval> YawScaleIntervals(
      const double source_yaw, const RelativeTwist& twist) const {
    if (!goal_yaw_.has_value() ||
        goal_yaw_tolerance_rad_ + kTolerance >=
            std::numbers::pi) {
      return {ScaleInterval{.lower = 0.0, .upper = 1.0}};
    }
    const double tolerance =
        goal_yaw_tolerance_rad_ + kTolerance;
    if (std::abs(twist.yaw_rate) <= kTolerance) {
      return std::abs(ShortestYawDelta(source_yaw, *goal_yaw_)) <= tolerance
                 ? std::vector<ScaleInterval>{
                       ScaleInterval{.lower = 0.0, .upper = 1.0}}
                 : std::vector<ScaleInterval>{};
    }
    const double start_delta = source_yaw - *goal_yaw_;
    const double finish_delta = start_delta + twist.yaw_rate;
    const double minimum_delta = std::min(start_delta, finish_delta);
    const double maximum_delta = std::max(start_delta, finish_delta);
    const double full_turn = 2.0 * std::numbers::pi;
    const auto first_turn = static_cast<std::int64_t>(
        std::floor((minimum_delta - tolerance) / full_turn)) - 1;
    const auto last_turn = static_cast<std::int64_t>(
        std::ceil((maximum_delta + tolerance) / full_turn)) + 1;
    std::vector<ScaleInterval> intervals;
    for (std::int64_t turn = first_turn; turn <= last_turn; ++turn) {
      const double center = static_cast<double>(turn) * full_turn;
      double lower = (center - tolerance - start_delta) / twist.yaw_rate;
      double upper = (center + tolerance - start_delta) / twist.yaw_rate;
      if (lower > upper) {
        std::swap(lower, upper);
      }
      lower = std::max(0.0, lower);
      upper = std::min(1.0, upper);
      if (lower <= upper + kTolerance) {
        intervals.push_back(
            ScaleInterval{.lower = lower, .upper = upper});
      }
    }
    std::ranges::sort(intervals, {}, &ScaleInterval::lower);
    std::vector<ScaleInterval> merged;
    for (const ScaleInterval& interval : intervals) {
      AddScaleInterval(merged, interval.lower, interval.upper);
    }
    return merged;
  }

  [[nodiscard]] bool ScaleSatisfiesGoal(
      const double source_yaw, const RelativeTwist& twist,
      const double target_x, const double target_y,
      const double ratio) const noexcept {
    if (!std::isfinite(ratio) || ratio <= kTolerance ||
        ratio > 1.0 + kTolerance) {
      return false;
    }
    const Pose3 relative = ExpRelativePose(twist, ratio);
    if (std::hypot(relative.position_m.x - target_x,
                   relative.position_m.y - target_y) >
        goal_.tolerance_m + kTolerance) {
      return false;
    }
    return !goal_yaw_.has_value() ||
           std::abs(ShortestYawDelta(
               NormalizeYaw(source_yaw + ratio * twist.yaw_rate),
               *goal_yaw_)) <=
               goal_yaw_tolerance_rad_ + kTolerance;
  }

  [[nodiscard]] std::optional<double> MatchingPrimitiveScale(
      const Node& source, const WheelMotionPrimitive& primitive) const {
    if (!ModeAllows(source.key.mode, primitive.kind) ||
        primitive.kind == WheelPrimitiveKind::kStopAndSwitch) {
      return std::nullopt;
    }
    const auto source_yaw = YawFromQuaternion(source.pose.orientation);
    const auto twist = LogRelativePose(primitive.relative_end_pose);
    if (!source_yaw.has_value() || !twist.has_value()) {
      return std::nullopt;
    }
    const double cosine = std::cos(*source_yaw);
    const double sine = std::sin(*source_yaw);
    const double world_x = goal_.position_m.x - source.pose.position_m.x;
    const double world_y = goal_.position_m.y - source.pose.position_m.y;
    const double target_x = cosine * world_x + sine * world_y;
    const double target_y = -sine * world_x + cosine * world_y;
    const auto position_error = [&](const double ratio) {
      const Pose3 relative = ExpRelativePose(*twist, ratio);
      return std::hypot(relative.position_m.x - target_x,
                        relative.position_m.y - target_y);
    };

    double closest_ratio = 0.0;
    if (std::abs(twist->yaw_rate) <= kTolerance) {
      const double denominator = twist->velocity_x * twist->velocity_x +
                                 twist->velocity_y * twist->velocity_y;
      if (denominator > kTolerance) {
        closest_ratio = (target_x * twist->velocity_x +
                         target_y * twist->velocity_y) /
                        denominator;
      }
    } else {
      constexpr std::size_t kSamples = 256U;
      std::size_t best_sample = 0U;
      double best_error = std::numeric_limits<double>::infinity();
      for (std::size_t sample = 0U; sample <= kSamples; ++sample) {
        const double candidate =
            static_cast<double>(sample) / static_cast<double>(kSamples);
        const double error = position_error(candidate);
        if (error < best_error) {
          best_error = error;
          best_sample = sample;
        }
      }
      double lower = static_cast<double>(best_sample == 0U ? 0U
                                                           : best_sample - 1U) /
                     static_cast<double>(kSamples);
      double upper = static_cast<double>(
                         std::min(kSamples, best_sample + 1U)) /
                     static_cast<double>(kSamples);
      for (std::size_t iteration = 0U; iteration < 64U; ++iteration) {
        const double first = (2.0 * lower + upper) / 3.0;
        const double second = (lower + 2.0 * upper) / 3.0;
        if (position_error(first) <= position_error(second)) {
          upper = second;
        } else {
          lower = first;
        }
      }
      closest_ratio = 0.5 * (lower + upper);
    }
    closest_ratio = std::clamp(closest_ratio, 0.0, 1.0);
    if (position_error(closest_ratio) <= kPhysicalMatchTolerance &&
        ScaleSatisfiesGoal(*source_yaw, *twist, target_x, target_y,
                           closest_ratio)) {
      return closest_ratio;
    }
    const std::vector<ScaleInterval> position_intervals =
        PositionScaleIntervals(*twist, target_x, target_y);
    const std::vector<ScaleInterval> yaw_intervals =
        YawScaleIntervals(*source_yaw, *twist);
    for (const ScaleInterval& position : position_intervals) {
      for (const ScaleInterval& yaw : yaw_intervals) {
        const double lower = std::max(position.lower, yaw.lower);
        const double upper = std::min(position.upper, yaw.upper);
        if (lower > upper + kTolerance) {
          continue;
        }
        const double positive_lower = std::max(lower, 2.0 * kTolerance);
        for (const double candidate :
             {0.5 * (positive_lower + upper), positive_lower, upper}) {
          if (ScaleSatisfiesGoal(*source_yaw, *twist, target_x, target_y,
                                 candidate)) {
            return candidate;
          }
        }
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<Transition> ScaledPrimitive(
      const Node& source, const WheelMotionPrimitive& primitive,
      const double ratio) const noexcept {
    const auto source_yaw = YawFromQuaternion(source.pose.orientation);
    const auto twist = LogRelativePose(primitive.relative_end_pose);
    if (!source_yaw.has_value() || !twist.has_value()) {
      return std::nullopt;
    }
    const Pose3 relative = ExpRelativePose(*twist, ratio);
    const double cosine = std::cos(*source_yaw);
    const double sine = std::sin(*source_yaw);
    const double reached_x = source.pose.position_m.x +
                             cosine * relative.position_m.x -
                             sine * relative.position_m.y;
    const double reached_y = source.pose.position_m.y +
                             sine * relative.position_m.x +
                             cosine * relative.position_m.y;
    const auto ground = map_.SampleElevationBilinear(
        Vec2{.x = reached_x, .y = reached_y});
    if (!ground.has_value()) {
      return std::nullopt;
    }
    Transition transition{
        .source = source.pose,
        .target = Pose3{
            .position_m = Vec3{.x = reached_x, .y = reached_y, .z = *ground},
            .orientation = QuaternionFromYaw(NormalizeYaw(
                *source_yaw +
                YawFromQuaternion(relative.orientation).value_or(0.0))),
        },
        .source_mode = source.key.mode,
        .target_mode = TargetMode(primitive.kind),
        .kind = primitive.kind,
        .reverse = IsReverse(primitive.kind),
    };
    if (!PoseSatisfiesGoal(transition.target)) {
      return std::nullopt;
    }
    RecomputeGeometry(transition);
    return transition;
  }

  [[nodiscard]] bool HasCertifiedTerminalSuccessorForActiveGoal(
      const Node& source) const {
    const double distance = std::hypot(
        goal_.position_m.x - source.pose.position_m.x,
        goal_.position_m.y - source.pose.position_m.y);
    if (distance > maximum_primitive_reach_m_ + goal_.tolerance_m +
                       2.0 * kTolerance) {
      return false;
    }
    const auto source_yaw = YawFromQuaternion(source.pose.orientation);
    if (!source_yaw.has_value() ||
        (goal_yaw_.has_value() &&
         std::abs(ShortestYawDelta(*source_yaw, *goal_yaw_)) >
             maximum_primitive_yaw_rad_ +
                 goal_yaw_tolerance_rad_ + kTolerance)) {
      return false;
    }
    for (const std::size_t primitive_index : ordered_primitives_) {
      const WheelMotionPrimitive& primitive =
          capability_.motion_primitives[primitive_index];
      const auto ratio = MatchingPrimitiveScale(source, primitive);
      if (!ratio.has_value()) {
        continue;
      }
      const auto connector = ScaledPrimitive(source, primitive, *ratio);
      if (!connector.has_value()) {
        continue;
      }
      Transition transition = *connector;
      transition.initial_edge = false;
      const EdgeEvaluation evaluation = Evaluate(
          transition, false, EdgeCertificationKind::kScaledPrimitive,
          primitive_stable_rank_[primitive_index]);
      if (evaluation.valid && PoseSatisfiesGoal(evaluation.transition.target)) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] std::uint32_t CertifiedTerminalGoalMask(
      const Node& source) const {
    std::uint32_t mask = 0U;
    for (std::size_t goal_index = 0U; goal_index < goals_.size();
         ++goal_index) {
      ActivateGoal(goal_index);
      if (HasCertifiedTerminalSuccessorForActiveGoal(source)) {
        mask |= std::uint32_t{1U} << goal_index;
      }
    }
    return mask;
  }

  [[nodiscard]] bool AppendGoalTerminal(
      const std::size_t state, Transition transition, const EdgeKey edge_key,
      const std::size_t goal_index, const std::size_t connector_kind,
      const std::size_t stable_primitive_index,
      std::vector<shared::GraphEdge>& edges) {
    const EdgeEvaluation& evaluation = CachedEvaluation(edge_key, [&] {
      return Evaluate(
          transition, false,
          connector_kind == 0U ? EdgeCertificationKind::kFullPrimitive
                               : EdgeCertificationKind::kScaledPrimitive,
          stable_primitive_index);
    });
    if (!evaluation.valid || !PoseSatisfiesGoal(evaluation.transition.target)) {
      return false;
    }
    const WheelStateKey goal_key =
        Quantize(evaluation.transition.target,
                 evaluation.transition.target_mode);
    const std::size_t goal_state = nodes_.size();
    nodes_.push_back(Node{
        .key = goal_key,
        .pose = evaluation.transition.target,
        .certified_clearance_m = evaluation.minimum_clearance_m,
        .creation_sequence = next_creation_sequence_++,
        .expandable = false,
        .exact_goal = true,
        .goal_index = goal_index,
    });
    emitted_edges_.emplace(
        StatePair{.source = state, .target = goal_state}, edge_key);
    edges.push_back(shared::GraphEdge{
        .target_state = goal_state,
        .cost = evaluation.cost,
        .stable_index = StableGoalEdgeIndex(
            state, goal_index, connector_kind, stable_primitive_index),
    });
    return true;
  }

  void AppendGoalConnectorEdges(const std::size_t state, const Node& source,
                                std::vector<shared::GraphEdge>& edges) {
    if (IsGoal(state)) {
      return;
    }
    const auto source_yaw = YawFromQuaternion(source.pose.orientation);
    if (!source_yaw.has_value()) {
      return;
    }
    for (std::size_t goal_index = 0U; goal_index < goals_.size();
         ++goal_index) {
      ActivateGoal(goal_index);
      const double distance = std::hypot(
          goal_.position_m.x - source.pose.position_m.x,
          goal_.position_m.y - source.pose.position_m.y);
      const double connector_limit = maximum_primitive_reach_m_ +
                                     goal_.tolerance_m + kTolerance;
      if (distance > connector_limit + kTolerance ||
          (goal_yaw_.has_value() &&
           std::abs(ShortestYawDelta(*source_yaw, *goal_yaw_)) >
               maximum_primitive_yaw_rad_ + goal_yaw_tolerance_rad_ +
                   kTolerance)) {
        continue;
      }
      for (const std::size_t primitive_index : ordered_primitives_) {
        const WheelMotionPrimitive& primitive =
            capability_.motion_primitives[primitive_index];
        const auto ratio = MatchingPrimitiveScale(source, primitive);
        if (!ratio.has_value()) {
          continue;
        }
        const auto connector = ScaledPrimitive(source, primitive, *ratio);
        if (!connector.has_value()) {
          continue;
        }
        Transition initial_connector = *connector;
        initial_connector.initial_edge = state == 0U;
        const std::size_t connector_index =
            (2U + goals_.size() + goal_index) *
                capability_.motion_primitives.size() +
            primitive_index;
        const EdgeKey key{.source_state = state,
                          .primitive_index = connector_index};
        static_cast<void>(AppendGoalTerminal(
            state, initial_connector, key, goal_index, 1U,
            primitive_stable_rank_[primitive_index], edges));
      }
    }
  }

  [[nodiscard]] static std::size_t StableGoalEdgeIndex(
      const std::size_t state, const std::size_t goal_index,
      const std::size_t connector_kind,
      const std::size_t primitive_rank) noexcept {
    constexpr std::size_t kDigits =
        std::numeric_limits<std::size_t>::digits;
    constexpr std::size_t kGoalShift = kDigits - 5U;
    constexpr std::size_t kConnectorShift = kGoalShift - 1U;
    constexpr std::size_t kPrimitiveBits = 16U;
    constexpr std::size_t kStateBits = kConnectorShift - kPrimitiveBits;
    constexpr std::size_t kStateMask =
        (std::size_t{1U} << kStateBits) - 1U;
    constexpr std::size_t kPrimitiveMask =
        (std::size_t{1U} << kPrimitiveBits) - 1U;
    return (goal_index << kGoalShift) |
           (connector_kind << kConnectorShift) |
           ((primitive_rank & kPrimitiveMask) << kStateBits) |
           (StableEdgeIndex(state, primitive_rank) & kStateMask);
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

  const WheelPlanRequest& request_;
  const shared::LocalTerrainProjection& terrain_;
  const WheeledCapability& capability_;
  const shared::MapSnapshot& map_;
  const PlanningLatticeFrame lattice_frame_;
  std::vector<WheelGoal> goals_;
  mutable std::size_t active_goal_index_{};
  mutable PointGoal goal_;
  mutable std::optional<double> goal_yaw_;
  mutable double goal_yaw_tolerance_rad_{};
  double footprint_radius_m_{};
  bool footprint_contains_origin_{};
  double broad_inset_radius_m_{};
  std::uint64_t capability_fingerprint_{};
  std::array<std::uint64_t, 6U> map_metadata_bits_{};
  double barrier_inset_radius_m_{};
  double platform_length_scale_m_{};
  double maximum_primitive_reach_m_{};
  double maximum_primitive_yaw_rad_{};
  double maximum_translation_reach_m_{};
  double maximum_translation_path_length_m_{};
  double minimum_zero_speed_translation_duration_s_{
      std::numeric_limits<double>::infinity()};
  std::size_t yaw_bins_{256U};
  std::size_t next_creation_sequence_{};
  std::size_t maximum_active_labels_per_key_{};
  bool used_narrow_resolution_{};
  std::size_t validation_requests_{};
  mutable std::size_t preferred_validation_count_{};
  mutable std::size_t broad_phase_rejects_{};
  mutable std::size_t full_certifications_{};
  mutable std::size_t full_invalidations_{};
  mutable std::size_t returned_edge_certificate_confirmations_{};
  std::size_t preferred_builder_invocations_{};
  std::size_t quantized_state_reuses_{};
  std::size_t quantized_endpoint_aliases_{};
  mutable std::size_t sweep_cell_checks_{};
  mutable std::size_t direct_unknown_or_unsupported_footprint_rejects_{};
  mutable std::size_t measured_obstacle_clearance_rejects_{};
  mutable std::size_t slope_or_roughness_rejects_{};
  mutable std::size_t relief_or_underbody_rejects_{};
  mutable std::size_t dynamics_or_primitive_shape_rejects_{};
  mutable std::size_t deadline_or_cancellation_interruptions_{};
  mutable std::size_t occupied_clearance_cell_checks_{};
  std::array<double, 5U> cost_scales_{};
  std::shared_ptr<const shared::GoalDistanceField> goal_distance_field_;
  double finest_xy_key_resolution_m_{
      std::numeric_limits<double>::infinity()};
  std::size_t maximum_yaw_bins_{};
  std::vector<std::size_t> ordered_primitives_;
  std::vector<std::size_t> primitive_stable_rank_;
  std::vector<std::uint32_t> occupied_integral_;
  std::vector<std::uint32_t> complex_terrain_integral_;
  std::vector<std::vector<std::int32_t>> occupied_by_row_;
  std::vector<BarrierRectangle> barriers_;
  std::vector<std::vector<BarrierRectangle>> barriers_by_goal_;
  std::optional<CertifiedPreferredCandidate> certified_preferred_candidate_;
  std::optional<shared::SearchCandidate> certified_initial_candidate_;
  std::vector<Node> nodes_;
  std::unordered_map<WheelStateKey, std::vector<std::size_t>,
                     WheelStateKeyHash>
      state_ids_;
  std::unordered_map<RejectedFingerprint, double, RejectedFingerprintHash>
      rejected_best_g_;
  std::unordered_map<StatePair, EdgeKey, StatePairHash> emitted_edges_;
  shared::EdgeValidationCache<EdgeKey, EdgeEvaluation, EdgeKeyHash>
      validation_cache_;
};

[[nodiscard]] WheelPlanResult Failure(const LocalPlanStatus status,
                                      std::string reason,
                                      const LocalPlanMetrics metrics = {}) {
  return WheelPlanResult{
      .status = status,
      .reason_code = std::move(reason),
      .metrics = metrics,
  };
}

[[nodiscard]] bool ValidTerrain(
    const shared::LocalTerrainProjection& terrain) noexcept {
  return terrain.map != nullptr && terrain.map->cell_count() != 0U &&
         terrain.free_with_height.size() == terrain.map->cell_count() &&
         terrain.occupied.size() == terrain.map->cell_count() &&
         terrain.clearance_m.size() == terrain.map->cell_count() &&
         terrain.narrow_band_distance_m.size() == terrain.map->cell_count() &&
         terrain.slope_rad.size() == terrain.map->cell_count() &&
         terrain.roughness_m.size() == terrain.map->cell_count();
}

}  // namespace

WheelPlanResult PlanWheel(const WheelPlanRequest& request) try {
  if (request.control.canceled()) {
    return Failure(LocalPlanStatus::kCanceled, "REQUEST_CANCELED");
  }
  const auto start_yaw = YawFromQuaternion(request.start.pose.orientation);
  const std::vector<GoalRegion>& requested_goals =
      request.goals_odom.goals_odom;
  const bool exact_final_goal = request.goals_odom.exact_final_goal;
  if (request.terrain == nullptr || request.capability == nullptr ||
      !ValidTerrain(*request.terrain) ||
      !ValidCapability(*request.capability) || !Finite(request.start.pose) ||
      !Finite(request.start.velocity) ||
      !start_yaw.has_value() || requested_goals.empty() ||
      requested_goals.size() > 32U ||
      (exact_final_goal && requested_goals.size() != 1U) ||
      request.search.epsilon_schedule !=
          std::array<double, 4>{2.5, 2.0, 1.5, 1.0}) {
    return Failure(LocalPlanStatus::kInvalidInput, "WHEEL_INPUT_INVALID");
  }
  std::vector<WheelGoal> goals;
  goals.reserve(requested_goals.size());
  for (const GoalRegion& goal : requested_goals) {
    const auto* point = std::get_if<PointGoal>(&goal.target);
    if (point == nullptr || !std::isfinite(point->position_m.x) ||
        !std::isfinite(point->position_m.y) ||
        !std::isfinite(point->tolerance_m) || point->tolerance_m < 0.0 ||
        (goal.yaw_rad.has_value() && !std::isfinite(*goal.yaw_rad)) ||
        !std::isfinite(goal.yaw_tolerance_rad) ||
        goal.yaw_tolerance_rad < 0.0 ||
        (!exact_final_goal && goal.yaw_rad.has_value()) ||
        !request.terrain->map
             ->PositionToCell(Vec2{.x = point->position_m.x,
                                   .y = point->position_m.y})
             .has_value()) {
      return Failure(LocalPlanStatus::kInvalidInput, "WHEEL_INPUT_INVALID");
    }
    goals.push_back(WheelGoal{.point = *point,
                              .yaw_rad = goal.yaw_rad,
                              .yaw_tolerance_rad = goal.yaw_tolerance_rad});
  }
  const auto start_cell = request.terrain->map->PositionToCell(
      Vec2{.x = request.start.pose.position_m.x,
           .y = request.start.pose.position_m.y});
  if (!start_cell.has_value()) {
    return Failure(LocalPlanStatus::kInvalidInput,
                   "WHEEL_POSE_OUTSIDE_LOCAL_MAP");
  }
  WheelSearchGraph graph{request, std::move(goals)};
  std::size_t ara_search_invocations = 0U;
  const auto decorate = [&](WheelPlanResult result) noexcept {
    result.ara_search_invocations = ara_search_invocations;
    result.edge_validation_cache_hits = graph.validation_cache_hits();
    result.quantization_alias_states = graph.quantization_alias_states();
    result.quantized_state_reuses = graph.quantized_state_reuses();
    result.quantized_endpoint_aliases = graph.quantized_endpoint_aliases();
    result.quantized_state_count = graph.quantized_state_count();
    result.maximum_active_labels_per_key =
        graph.maximum_active_labels_per_key();
    result.preferred_builder_invocations =
        graph.preferred_builder_invocations();
    result.broad_phase_rejects = graph.broad_phase_rejects();
    result.full_certifications = graph.full_certifications();
    result.full_invalidations = graph.full_invalidations();
    result.returned_edge_certificate_confirmations =
        graph.returned_edge_certificate_confirmations();
    result.sweep_cell_checks = graph.sweep_cell_checks();
    result.finest_xy_key_resolution_m = graph.finest_xy_key_resolution_m();
    result.maximum_yaw_bins = graph.maximum_yaw_bins();
    result.start_heuristic_lower_bound = graph.Heuristic(0U);
    result.cost_scales = graph.cost_scales();
    if (const auto& preferred = graph.certified_preferred_candidate();
        preferred.has_value()) {
      result.has_certified_preferred_candidate = true;
      result.preferred_candidate_full_primitive_edge_count =
          preferred->full_primitive_edges;
      result.preferred_candidate_terminal_connector_edge_count =
          preferred->terminal_connector_edges;
      result.preferred_candidate_certified_edge_count =
          preferred->edges.size();
      result.preferred_candidate_cost = preferred->cost;
    }
    result.wheel_metrics = WheelPlanningMetrics{
        .expanded_states = result.metrics.expanded_states,
        .edge_validation_evaluations =
            result.metrics.edge_validation_evaluations,
        .edge_validation_cache_hits = result.edge_validation_cache_hits,
        .broad_phase_rejects = result.broad_phase_rejects,
        .full_certifications = result.full_certifications,
        .full_invalidations = result.full_invalidations,
        .sweep_cell_checks = result.sweep_cell_checks,
        .quantization_alias_states = result.quantization_alias_states,
        .quantized_state_reuses = result.quantized_state_reuses,
        .quantized_endpoint_aliases = result.quantized_endpoint_aliases,
        .quantized_state_count = result.quantized_state_count,
        .maximum_active_labels_per_key =
            result.maximum_active_labels_per_key,
        .used_narrow_resolution = result.metrics.used_narrow_resolution,
        .finest_xy_key_resolution_m = result.finest_xy_key_resolution_m,
        .maximum_yaw_bins = result.maximum_yaw_bins,
        .ara_search_invocations = result.ara_search_invocations,
        .returned_edge_certificate_confirmations =
            result.returned_edge_certificate_confirmations,
        .mode_switch_edge_count = result.mode_switch_edge_count,
        .reverse_edge_count = result.reverse_edge_count,
        .start_heuristic_lower_bound = result.start_heuristic_lower_bound,
        .has_certified_preferred_candidate =
            result.has_certified_preferred_candidate,
        .preferred_candidate_full_primitive_edge_count =
            result.preferred_candidate_full_primitive_edge_count,
        .preferred_candidate_terminal_connector_edge_count =
            result.preferred_candidate_terminal_connector_edge_count,
        .preferred_candidate_certified_edge_count =
            result.preferred_candidate_certified_edge_count,
        .preferred_candidate_cost = result.preferred_candidate_cost,
        .preferred_builder_invocations = result.preferred_builder_invocations,
        .cost_components = result.cost_components,
        .cost_scales = result.cost_scales,
        .direct_unknown_or_unsupported_footprint_rejects =
            graph.direct_unknown_or_unsupported_footprint_rejects(),
        .measured_obstacle_clearance_rejects =
            graph.measured_obstacle_clearance_rejects(),
        .slope_or_roughness_rejects = graph.slope_or_roughness_rejects(),
        .relief_or_underbody_rejects =
            graph.relief_or_underbody_rejects(),
        .dynamics_or_primitive_shape_rejects =
            graph.dynamics_or_primitive_shape_rejects(),
        .deadline_or_cancellation_interruptions =
            graph.deadline_or_cancellation_interruptions(),
        .far_clearance_scan_skips = graph.far_clearance_scan_skips(),
        .occupied_clearance_cell_checks =
            graph.occupied_clearance_cell_checks(),
    };
    return result;
  };
  try {
  if (!graph.ValidateStart()) {
    const LocalPlanMetrics metrics{
        .edge_validation_evaluations = graph.validation_count(),
        .used_narrow_resolution = graph.used_narrow_resolution(),
    };
    if (request.control.canceled()) {
      return decorate(
          Failure(LocalPlanStatus::kCanceled, "REQUEST_CANCELED", metrics));
    }
    if (request.control.expired()) {
      return decorate(
          Failure(LocalPlanStatus::kTimedOut, "TIMEOUT", metrics));
    }
    return decorate(Failure(LocalPlanStatus::kNoPath,
                            "WHEEL_START_INFEASIBLE", metrics));
  }

  ++ara_search_invocations;
  const shared::anytime::AraStarResult search =
      shared::anytime::SearchAnytimeAraStar(
          shared::anytime::AraStarProblem{
              .state_count = graph.state_count(),
              .start_state = 0U,
              .expand = [&](const std::size_t state,
                            const double source_g,
                            std::vector<shared::GraphEdge>& edges) {
                graph.Expand(state, source_g, edges);
              },
              .heuristic = [&](const std::size_t state) {
                return graph.Heuristic(state);
              },
              .guidance = [&](const std::size_t state) {
                return graph.Guidance(state);
              },
              .is_goal = [&](const std::size_t state) {
                return graph.IsGoal(state);
              },
              .state_expandable = [&](const std::size_t state) {
                return graph.StateExpandable(state);
              },
              .on_relaxed = [&](const std::size_t state,
                                const double best_g) {
                graph.OnRelaxed(state, best_g);
              },
              .certified_initial_candidate =
                  graph.certified_initial_candidate(),
              .config = request.search,
              .control = request.control,
          });
  const LocalPlanMetrics metrics{
      .expanded_states = search.expanded_states,
      .edge_validation_evaluations = graph.validation_count(),
      .used_narrow_resolution = graph.used_narrow_resolution(),
  };
  switch (search.status) {
    case shared::anytime::AraStarStatus::kCanceled:
      return decorate(
          Failure(LocalPlanStatus::kCanceled, "REQUEST_CANCELED", metrics));
    case shared::anytime::AraStarStatus::kTimedOut:
      return decorate(
          Failure(LocalPlanStatus::kTimedOut, "TIMEOUT", metrics));
    case shared::anytime::AraStarStatus::kNoPath:
      return decorate(Failure(LocalPlanStatus::kNoPath, "NO_PATH", metrics));
    case shared::anytime::AraStarStatus::kResourceExhausted:
      return decorate(Failure(LocalPlanStatus::kPlannerError,
                              "WHEEL_SEARCH_RESOURCE_EXHAUSTED", metrics));
    case shared::anytime::AraStarStatus::kInvalidProblem:
      return decorate(Failure(
          LocalPlanStatus::kPlannerError,
          search.reason_code.empty() ? "WHEEL_SEARCH_INVALID"
                                     : search.reason_code,
          metrics));
    case shared::anytime::AraStarStatus::kSolved:
      break;
  }
  if (search.candidates.empty()) {
    return decorate(Failure(LocalPlanStatus::kPlannerError,
                            "WHEEL_INCUMBENT_MISSING", metrics));
  }
  const shared::SearchCandidate& candidate = search.candidates.back();
  const std::optional<std::size_t> selected_goal_index =
      graph.GoalIndexForState(candidate.states.back());
  if (!selected_goal_index.has_value()) {
    return decorate(Failure(LocalPlanStatus::kPlannerError,
                            "WHEEL_TERMINAL_GOAL_INDEX_MISSING", metrics));
  }
  const auto control_failure = [&]() -> std::optional<WheelPlanResult> {
    const auto stopped = shared::StopReason(request.control);
    if (!stopped.has_value()) {
      return std::nullopt;
    }
    return decorate(Failure(
        *stopped == "REQUEST_CANCELED" ? LocalPlanStatus::kCanceled
                                        : LocalPlanStatus::kTimedOut,
        std::string{*stopped}, metrics));
  };
  if (const auto stopped = control_failure(); stopped.has_value()) {
    return *stopped;
  }
  std::vector<TrajectoryPoint> trajectory;
  trajectory.reserve(1U + candidate.states.size() * 8U);
  if (const auto stopped = control_failure(); stopped.has_value()) {
    return *stopped;
  }
  std::chrono::nanoseconds time{};
  trajectory.push_back(TrajectoryPoint{
      .time_from_start = time,
      .pose = graph.PoseForState(candidate.states.front()),
      .velocity = request.start.velocity,
  });
  std::array<double, 5U> cost_components{};
  std::size_t mode_switch_edge_count = 0U;
  std::size_t reverse_edge_count = 0U;
  for (std::size_t index = 1U; index < candidate.states.size(); ++index) {
    if (const auto stopped = control_failure(); stopped.has_value()) {
      return *stopped;
    }
    const EdgeEvaluation* evaluation = graph.EvaluationForEdge(
        candidate.states[index - 1U], candidate.states[index]);
    if (evaluation == nullptr || !evaluation->valid) {
      return decorate(Failure(LocalPlanStatus::kPlannerError,
                              "WHEEL_CERTIFIED_EDGE_MISSING", metrics));
    }
    const Transition& transition = evaluation->transition;
    mode_switch_edge_count += static_cast<std::size_t>(
        transition.kind == WheelPrimitiveKind::kStopAndSwitch);
    reverse_edge_count += static_cast<std::size_t>(transition.reverse);
    for (std::size_t component = 0U; component < cost_components.size();
         ++component) {
      cost_components[component] += evaluation->cost_components[component];
    }
    if (!graph.AppendTimedEdge(*evaluation, &time, &trajectory,
                               request.control)) {
      if (const auto stopped = control_failure(); stopped.has_value()) {
        return *stopped;
      }
      return decorate(Failure(LocalPlanStatus::kPlannerError,
                              "WHEEL_TIMING_FAILED", metrics));
    }
  }
  if (const auto stopped = control_failure(); stopped.has_value()) {
    return *stopped;
  }
  return decorate(WheelPlanResult{
      .status = LocalPlanStatus::kSolved,
      .reason_code = "WHEEL_PLAN_AVAILABLE",
      .trajectory = std::move(trajectory),
      .metrics = metrics,
      .mode_switch_edge_count = mode_switch_edge_count,
      .reverse_edge_count = reverse_edge_count,
      .selected_goal_index = selected_goal_index,
      .cost_components = cost_components,
      .cost = candidate.cost,
  });
  } catch (...) {
    return decorate(Failure(
        LocalPlanStatus::kPlannerError, "PLANNER_ERROR",
        LocalPlanMetrics{
            .edge_validation_evaluations = graph.validation_count(),
            .used_narrow_resolution = graph.used_narrow_resolution(),
        }));
  }
} catch (...) {
  return Failure(LocalPlanStatus::kPlannerError, "PLANNER_ERROR");
}

}  // namespace lunar::pure_planning::wheel
