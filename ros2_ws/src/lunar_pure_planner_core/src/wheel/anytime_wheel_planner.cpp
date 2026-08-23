#include "wheel/anytime_wheel_planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <ranges>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shared/anytime_ara_star.hpp"
#include "shared/controlled_work.hpp"
#include "shared/edge_validation_cache.hpp"
#include "shared/goal_distance_field.hpp"
#include "wheel/wheel_types.hpp"

namespace lunar::pure_planning::wheel {
namespace {

constexpr double kTolerance = 1.0e-9;
constexpr double kPhysicalMatchTolerance = 1.0e-6;
constexpr double kMinimumEdgeCost = 1.0e-6;
constexpr std::size_t kMaximumSearchStates = 131072U;
constexpr std::size_t kModeCount = 3U;
constexpr std::array<double, 5U> kCostWeights{1.0, 1.0, 1.0, 1.0, 1.0};

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
  bool exact_goal{};
};

class WheelSearchGraph final {
 public:
  WheelSearchGraph(const WheelPlanRequest& request, const PointGoal& goal,
                   const std::optional<double> goal_yaw)
      : request_(request),
        terrain_(*request.terrain),
        capability_(*request.capability),
        map_(*request.terrain->map),
        lattice_origin_x_m_(request.start.pose.position_m.x),
        lattice_origin_y_m_(request.start.pose.position_m.y),
        goal_(goal),
        goal_yaw_(goal_yaw),
        footprint_radius_m_(CircumscribedRadius(capability_)) {
    const std::size_t cell_count = map_.cell_count();
    const std::size_t scaled =
        cell_count > kMaximumSearchStates / 16U
            ? kMaximumSearchStates
            : std::max<std::size_t>(4096U, cell_count * 16U);
    assignable_state_limit_ = std::min(
        {kMaximumSearchStates, scaled, request_.maximum_search_states});
    state_count_ = assignable_state_limit_;
    const std::size_t reserve_hint =
        cell_count > (assignable_state_limit_ - 1U) / 2U
            ? assignable_state_limit_
            : cell_count * 2U + 1U;
    nodes_.reserve(std::min(assignable_state_limit_, reserve_hint));
    state_ids_.reserve(nodes_.capacity());
    const std::size_t prefix_width = map_.width() + 1U;
    hazard_integral_.assign(prefix_width * (map_.height() + 1U), 0U);
    for (std::size_t y = 0U; y < map_.height(); ++y) {
      std::uint32_t row_hazards = 0U;
      for (std::size_t x = 0U; x < map_.width(); ++x) {
        const std::size_t index = y * map_.width() + x;
        row_hazards += static_cast<std::uint32_t>(
            terrain_.free_with_height[index] == 0U);
        hazard_integral_[(y + 1U) * prefix_width + x + 1U] =
            hazard_integral_[y * prefix_width + x + 1U] + row_hazards;
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
    }
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
    nodes_.push_back(
        Node{.key = key, .pose = request_.start.pose, .exact_goal = false});
    state_ids_.emplace(key, 0U);
    if (const auto goal_cell = map_.PositionToCell(
            Vec2{.x = goal_.position_m.x, .y = goal_.position_m.y});
        goal_cell.has_value()) {
      goal_distance_field_ =
          shared::BuildGoalDistanceField(terrain_, *goal_cell, request_.control);
    }
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
    return state_ids_.size();
  }

  [[nodiscard]] std::size_t sweep_cell_checks() const noexcept {
    return sweep_cell_checks_;
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

  [[nodiscard]] double Guidance(const std::size_t state) const noexcept {
    if (!goal_distance_field_.has_value() || state >= nodes_.size()) {
      return 0.0;
    }
    const auto cell = map_.PositionToCell(
        Vec2{.x = nodes_[state].pose.position_m.x,
             .y = nodes_[state].pose.position_m.y});
    if (!cell.has_value()) {
      return 0.0;
    }
    const double distance = goal_distance_field_->distance_m[map_.Index(*cell)];
    return std::isfinite(distance) ? distance : 0.0;
  }

  [[nodiscard]] const EdgeEvaluation* EvaluationForEdge(
      const std::size_t source, const std::size_t target) {
    const auto found = emitted_edges_.find(
        StatePair{.source = source, .target = target});
    if (found == emitted_edges_.end()) {
      return nullptr;
    }
    return &CachedEvaluation(found->second, [] { return EdgeEvaluation{}; });
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
    return nodes_.at(state).pose;
  }

  [[nodiscard]] WheelMotionMode ModeForState(
      const std::size_t state) const noexcept {
    return nodes_[state].key.mode;
  }

  [[nodiscard]] bool ValidateStart() {
    const EdgeKey key{.source_state = 0U,
                      .primitive_index =
                          3U * capability_.motion_primitives.size()};
    return CachedEvaluation(key, [&] {
          Transition start{
              .source = request_.start.pose,
              .target = request_.start.pose,
          };
          return Evaluate(start, true);
        })
        .valid;
  }

  void Expand(const std::size_t state,
              std::vector<shared::GraphEdge>& edges) {
    if (state >= nodes_.size() || nodes_[state].exact_goal ||
        ControlInterrupted()) {
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
      if (PoseSatisfiesGoal(actual_endpoint)) {
        const EdgeKey goal_edge_key{
            .source_state = state,
            .primitive_index =
                2U * capability_.motion_primitives.size() + primitive_index,
        };
        if (AppendGoalTerminal(
                state, *transition, goal_edge_key,
                2U * capability_.motion_primitives.size() +
                    primitive_stable_rank_[primitive_index],
                edges)) {
          generated_goal_terminal = true;
          continue;
        }
      }
      const WheelStateKey target_key =
          Quantize(actual_endpoint, transition->target_mode);
      if (target_key == source.key) {
        continue;
      }
      if (!generated_target_keys.emplace(target_key, primitive_index).second) {
        ++quantized_endpoint_aliases_;
      }
      const auto canonical_pose = RepresentativePose(target_key);
      if (!canonical_pose.has_value()) {
        continue;
      }
      const double xy_resolution = target_key.narrow
                                       ? map_.resolution_m() / 2.0
                                       : map_.resolution_m();
      const std::size_t yaw_bins = target_key.narrow ? 128U : 64U;
      const auto actual_yaw = YawFromQuaternion(actual_endpoint.orientation);
      const auto canonical_yaw =
          YawFromQuaternion(canonical_pose->orientation);
      if (!actual_yaw.has_value() || !canonical_yaw.has_value() ||
          std::hypot(actual_endpoint.position_m.x -
                         canonical_pose->position_m.x,
                     actual_endpoint.position_m.y -
                         canonical_pose->position_m.y) >
              std::numbers::sqrt2 * 0.5 * xy_resolution +
                  kPhysicalMatchTolerance ||
          std::abs(ShortestYawDelta(*actual_yaw, *canonical_yaw)) >
              std::numbers::pi / static_cast<double>(yaw_bins) +
                  kPhysicalMatchTolerance) {
        continue;
      }
      transition->target = *canonical_pose;
      RecomputeGeometry(*transition);
      if (!PrimitiveReachesTarget(source, capability_.motion_primitives[
                                              primitive_index],
                                  *canonical_pose) ||
          !MotionKindMatches(*transition)) {
        continue;
      }
      const EdgeKey edge_key{.source_state = state,
                             .primitive_index = primitive_index};
      const EdgeEvaluation& evaluation = CachedEvaluation(
          edge_key, [&] { return Evaluate(*transition, false); });
      if (!evaluation.valid) {
        continue;
      }
      const auto target_state = Intern(target_key);
      if (!target_state.has_value()) {
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
    const Pose3& pose = nodes_[state].pose;
    const double distance_to_region = std::max(
        0.0,
        std::hypot(pose.position_m.x - goal_.position_m.x,
                   pose.position_m.y - goal_.position_m.y) -
            goal_.tolerance_m);
    return kCostWeights[0] *
           distance_to_region / cost_scales_[0];
  }

  [[nodiscard]] bool IsGoal(const std::size_t state) const noexcept {
    if (state >= nodes_.size()) {
      return false;
    }
    if (nodes_[state].exact_goal) {
      return true;
    }
    return PoseSatisfiesGoal(nodes_[state].pose);
  }

 private:
  [[nodiscard]] bool PoseSatisfiesGoal(const Pose3& pose) const noexcept {
    const auto yaw = YawFromQuaternion(pose.orientation);
    const double position_error = std::hypot(
        pose.position_m.x - goal_.position_m.x,
        pose.position_m.y - goal_.position_m.y);
    if (!yaw.has_value() || position_error > goal_.tolerance_m + kTolerance ||
        (goal_yaw_.has_value() &&
         std::abs(ShortestYawDelta(*yaw, *goal_yaw_)) >
             request_.goal_odom.yaw_tolerance_rad + kTolerance)) {
      return false;
    }
    return true;
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

  [[nodiscard]] bool HasHazardInCells(const std::int64_t minimum_x,
                                      const std::int64_t minimum_y,
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
        hazard_integral_[bottom * stride + right] -
        hazard_integral_[top * stride + right] -
        hazard_integral_[bottom * stride + left] +
        hazard_integral_[top * stride + left];
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
    const float clearance = terrain_.clearance_m[map_.Index(*cell)];
    return std::isfinite(clearance) &&
           static_cast<double>(clearance) <
               footprint_radius_m_ + 2.0 * map_.resolution_m();
  }

  [[nodiscard]] WheelStateKey Quantize(const Pose3& pose,
                                       const WheelMotionMode mode) {
    const bool narrow = IsNarrow(pose);
    used_narrow_resolution_ = used_narrow_resolution_ || narrow;
    const double xy_resolution =
        narrow ? map_.resolution_m() / 2.0 : map_.resolution_m();
    const std::size_t yaw_bins = narrow ? 128U : 64U;
    finest_xy_key_resolution_m_ =
        std::min(finest_xy_key_resolution_m_, xy_resolution);
    maximum_yaw_bins_ = std::max(maximum_yaw_bins_, yaw_bins);
    const auto yaw = YawFromQuaternion(pose.orientation).value_or(0.0);
    const double positive_yaw = NormalizeYaw(yaw) < 0.0
                                    ? NormalizeYaw(yaw) + 2.0 * std::numbers::pi
                                    : NormalizeYaw(yaw);
    const auto yaw_bin = static_cast<std::size_t>(std::llround(
        positive_yaw * static_cast<double>(yaw_bins) /
        (2.0 * std::numbers::pi))) % yaw_bins;
    return WheelStateKey{
        .x = static_cast<std::int64_t>(std::llround(
            (pose.position_m.x - lattice_origin_x_m_) / xy_resolution)),
        .y = static_cast<std::int64_t>(std::llround(
            (pose.position_m.y - lattice_origin_y_m_) / xy_resolution)),
        .yaw = static_cast<std::uint16_t>(yaw_bin),
        .mode = mode,
        .narrow = narrow,
    };
  }

  [[nodiscard]] std::optional<Pose3> CanonicalPose(
      const WheelStateKey& key) const noexcept {
    const double xy_resolution =
        key.narrow ? map_.resolution_m() / 2.0 : map_.resolution_m();
    const std::size_t yaw_bins = key.narrow ? 128U : 64U;
    const double x = lattice_origin_x_m_ +
                     static_cast<double>(key.x) * xy_resolution;
    const double y = lattice_origin_y_m_ +
                     static_cast<double>(key.y) * xy_resolution;
    const auto elevation =
        map_.SampleElevationBilinear(Vec2{.x = x, .y = y});
    if (!elevation.has_value()) {
      return std::nullopt;
    }
    const Pose3 pose{
        .position_m = Vec3{.x = x, .y = y, .z = *elevation},
        .orientation = QuaternionFromYaw(NormalizeYaw(
            static_cast<double>(key.yaw) * 2.0 * std::numbers::pi /
            static_cast<double>(yaw_bins))),
    };
    return IsNarrow(pose) == key.narrow ? std::optional<Pose3>{pose}
                                        : std::nullopt;
  }

  [[nodiscard]] std::optional<Pose3> RepresentativePose(
      const WheelStateKey& key) const noexcept {
    if (const auto found = state_ids_.find(key); found != state_ids_.end()) {
      return nodes_[found->second].pose;
    }
    return CanonicalPose(key);
  }

  [[nodiscard]] std::optional<std::size_t> Intern(
      const WheelStateKey& key) {
    if (const auto found = state_ids_.find(key); found != state_ids_.end()) {
      ++quantized_state_reuses_;
      return found->second;
    }
    if (nodes_.size() >= assignable_state_limit_) {
      resource_exhausted_ = true;
      return std::nullopt;
    }
    const auto pose = CanonicalPose(key);
    if (!pose.has_value()) {
      return std::nullopt;
    }
    const std::size_t state = nodes_.size();
    nodes_.push_back(Node{.key = key, .pose = *pose, .exact_goal = false});
    state_ids_.emplace(key, state);
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

  [[nodiscard]] EdgeEvaluation Evaluate(const Transition& transition,
                                        const bool pose_only) const {
    EdgeEvaluation result{.transition = transition};
    if (std::abs(transition.curvature_per_m) >
        capability_.maximum_curvature_per_m + kTolerance) {
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
        return result;
      }
      const double ratio = subdivisions == 0U
                               ? 0.0
                               : static_cast<double>(sample) /
                                     static_cast<double>(subdivisions);
      const auto pose = Interpolate(transition, ratio);
      if (!pose.has_value()) {
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
      const auto hard_minimum_cell_x = static_cast<std::int64_t>(
          std::floor((minimum_x - clearance_margin - map_.origin_m().x) /
                     map_.resolution_m()));
      const auto hard_minimum_cell_y = static_cast<std::int64_t>(
          std::floor((minimum_y - clearance_margin - map_.origin_m().y) /
                     map_.resolution_m()));
      const auto hard_maximum_cell_x = static_cast<std::int64_t>(
          std::floor((maximum_x + clearance_margin - map_.origin_m().x) /
                     map_.resolution_m()));
      const auto hard_maximum_cell_y = static_cast<std::int64_t>(
          std::floor((maximum_y + clearance_margin - map_.origin_m().y) /
                     map_.resolution_m()));
      if (hard_minimum_cell_x < 0 || hard_minimum_cell_y < 0 ||
          hard_maximum_cell_x >= static_cast<std::int64_t>(map_.width()) ||
          hard_maximum_cell_y >= static_cast<std::int64_t>(map_.height())) {
        return result;
      }
      const auto clearance_minimum_cell_x = std::max<std::int64_t>(
          0, static_cast<std::int64_t>(
          std::floor((minimum_x - clearance_scan_margin - map_.origin_m().x) /
                     map_.resolution_m())));
      const auto clearance_minimum_cell_y = std::max<std::int64_t>(
          0, static_cast<std::int64_t>(
          std::floor((minimum_y - clearance_scan_margin - map_.origin_m().y) /
                     map_.resolution_m())));
      const auto clearance_maximum_cell_x = std::min<std::int64_t>(
          static_cast<std::int64_t>(map_.width()) - 1,
          static_cast<std::int64_t>(
          std::floor((maximum_x + clearance_scan_margin - map_.origin_m().x) /
                     map_.resolution_m())));
      const auto clearance_maximum_cell_y = std::min<std::int64_t>(
          static_cast<std::int64_t>(map_.height()) - 1,
          static_cast<std::int64_t>(
          std::floor((maximum_y + clearance_scan_margin - map_.origin_m().y) /
                     map_.resolution_m())));
      if (HasHazardInCells(clearance_minimum_cell_x, clearance_minimum_cell_y,
                           clearance_maximum_cell_x,
                           clearance_maximum_cell_y)) {
        for (std::int64_t y = clearance_minimum_cell_y;
             y <= clearance_maximum_cell_y; ++y) {
          for (std::int64_t x = clearance_minimum_cell_x;
               x <= clearance_maximum_cell_x; ++x) {
            ++sweep_cell_checks_;
            if ((sweep_cell_checks_ & 63U) == 0U && ControlInterrupted()) {
              return result;
            }
            const shared::GridCell cell{
                .x = static_cast<std::int32_t>(x),
                .y = static_cast<std::int32_t>(y),
            };
            const std::size_t index = map_.Index(cell);
            if (index < terrain_.free_with_height.size() &&
                terrain_.free_with_height[index] != 0U) {
              continue;
            }
            const double cell_minimum_x = map_.origin_m().x +
                                          static_cast<double>(x) * map_.resolution_m();
            const double cell_minimum_y = map_.origin_m().y +
                                          static_cast<double>(y) * map_.resolution_m();
            const double distance = PolygonDistanceToCell(
                polygon, cell_minimum_x, cell_minimum_y,
                cell_minimum_x + map_.resolution_m(),
                cell_minimum_y + map_.resolution_m());
            result.minimum_clearance_m =
                std::min(result.minimum_clearance_m, distance);
            if (distance + kTolerance < clearance_margin ||
                distance <= kTolerance) {
              return result;
            }
          }
        }
      }

      const auto minimum_cell_x = static_cast<std::int64_t>(std::floor(
          (minimum_x - map_.origin_m().x) / map_.resolution_m()));
      const auto minimum_cell_y = static_cast<std::int64_t>(std::floor(
          (minimum_y - map_.origin_m().y) / map_.resolution_m()));
      const auto maximum_cell_x = static_cast<std::int64_t>(std::floor(
          (maximum_x - map_.origin_m().x) / map_.resolution_m()));
      const auto maximum_cell_y = static_cast<std::int64_t>(std::floor(
          (maximum_y - map_.origin_m().y) / map_.resolution_m()));
      if (minimum_cell_x < 0 || minimum_cell_y < 0 ||
          maximum_cell_x >= static_cast<std::int64_t>(map_.width()) ||
          maximum_cell_y >= static_cast<std::int64_t>(map_.height())) {
        return result;
      }

      std::array<double, 4U> wheel_heights{};
      std::size_t wheel_index = 0U;
      for (const double body_x : {-0.5 * capability_.wheelbase_m,
                                  0.5 * capability_.wheelbase_m}) {
        for (const double body_y : {-0.5 * capability_.track_width_m,
                                    0.5 * capability_.track_width_m}) {
          const Vec2 wheel_position{
              .x = center_x + cosine * body_x - sine * body_y,
              .y = center_y + sine * body_x + cosine * body_y,
          };
          const auto elevation = map_.SampleElevationBilinear(wheel_position);
          if (!elevation.has_value()) {
            return result;
          }
          wheel_heights[wheel_index++] = *elevation;
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
              !std::isfinite(terrain_.slope_rad[index]) ||
              terrain_.slope_rad[index] >
                  capability_.maximum_slope_rad + kTolerance ||
              !std::isfinite(terrain_.roughness_m[index]) ||
              index >= elevations.size() || !std::isfinite(elevations[index])) {
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
      if (!touched_cell ||
          maximum_positive_relief >
              capability_.maximum_local_obstacle_relief_m +
                  kPhysicalMatchTolerance ||
          maximum_positive_relief >
              capability_.minimum_underbody_clearance_m +
                  kPhysicalMatchTolerance) {
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
      return result;
    }
    result.cost = pose_only ? 0.0 : EdgeCost(result);
    if (!std::isfinite(result.cost) || (!pose_only && result.cost <= 0.0)) {
      result.valid = false;
    }
    return result;
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
        request_.goal_odom.yaw_tolerance_rad + kTolerance >=
            std::numbers::pi) {
      return {ScaleInterval{.lower = 0.0, .upper = 1.0}};
    }
    const double tolerance =
        request_.goal_odom.yaw_tolerance_rad + kTolerance;
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
               request_.goal_odom.yaw_tolerance_rad + kTolerance;
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

  [[nodiscard]] bool AppendGoalTerminal(
      const std::size_t state, Transition transition, const EdgeKey edge_key,
      const std::size_t stable_primitive_index,
      std::vector<shared::GraphEdge>& edges) {
    const EdgeEvaluation& evaluation = CachedEvaluation(
        edge_key, [&] { return Evaluate(transition, false); });
    if (!evaluation.valid || !PoseSatisfiesGoal(evaluation.transition.target)) {
      return false;
    }
    if (nodes_.size() >= assignable_state_limit_) {
      resource_exhausted_ = true;
      return false;
    }
    const WheelStateKey goal_key =
        Quantize(evaluation.transition.target,
                 evaluation.transition.target_mode);
    const std::size_t goal_state = nodes_.size();
    nodes_.push_back(Node{
        .key = goal_key,
        .pose = evaluation.transition.target,
        .exact_goal = true,
    });
    emitted_edges_.emplace(
        StatePair{.source = state, .target = goal_state}, edge_key);
    edges.push_back(shared::GraphEdge{
        .target_state = goal_state,
        .cost = evaluation.cost,
        .stable_index = StableEdgeIndex(state, stable_primitive_index),
    });
    return true;
  }

  void AppendGoalConnectorEdges(const std::size_t state, const Node& source,
                                std::vector<shared::GraphEdge>& edges) {
    if (IsGoal(state)) {
      return;
    }
    const double distance = std::hypot(
        goal_.position_m.x - source.pose.position_m.x,
        goal_.position_m.y - source.pose.position_m.y);
    const double connector_limit = maximum_primitive_reach_m_ +
                                   goal_.tolerance_m + kTolerance;
    if (distance > connector_limit + kTolerance) {
      return;
    }
    const auto source_yaw = YawFromQuaternion(source.pose.orientation);
    if (!source_yaw.has_value()) {
      return;
    }
    if (goal_yaw_.has_value() &&
        std::abs(ShortestYawDelta(*source_yaw, *goal_yaw_)) >
            maximum_primitive_yaw_rad_ +
                request_.goal_odom.yaw_tolerance_rad + kTolerance) {
      return;
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
          capability_.motion_primitives.size() + primitive_index;
      const EdgeKey key{.source_state = state,
                        .primitive_index = connector_index};
      static_cast<void>(AppendGoalTerminal(
          state, initial_connector, key,
          capability_.motion_primitives.size() +
              primitive_stable_rank_[primitive_index],
          edges));
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

  const WheelPlanRequest& request_;
  const shared::LocalTerrainProjection& terrain_;
  const WheeledCapability& capability_;
  const shared::MapSnapshot& map_;
  const double lattice_origin_x_m_;
  const double lattice_origin_y_m_;
  PointGoal goal_;
  std::optional<double> goal_yaw_;
  double footprint_radius_m_{};
  double platform_length_scale_m_{};
  double maximum_primitive_reach_m_{};
  double maximum_primitive_yaw_rad_{};
  std::size_t assignable_state_limit_{};
  std::size_t state_count_{};
  bool resource_exhausted_{};
  bool used_narrow_resolution_{};
  std::size_t validation_requests_{};
  std::size_t quantized_state_reuses_{};
  std::size_t quantized_endpoint_aliases_{};
  mutable std::size_t sweep_cell_checks_{};
  std::array<double, 5U> cost_scales_{};
  std::optional<shared::GoalDistanceField> goal_distance_field_;
  double finest_xy_key_resolution_m_{
      std::numeric_limits<double>::infinity()};
  std::size_t maximum_yaw_bins_{};
  std::vector<std::size_t> ordered_primitives_;
  std::vector<std::size_t> primitive_stable_rank_;
  std::vector<std::uint32_t> hazard_integral_;
  std::vector<Node> nodes_;
  std::unordered_map<WheelStateKey, std::size_t, WheelStateKeyHash> state_ids_;
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
         terrain.clearance_m.size() == terrain.map->cell_count() &&
         terrain.slope_rad.size() == terrain.map->cell_count() &&
         terrain.roughness_m.size() == terrain.map->cell_count();
}

}  // namespace

WheelPlanResult PlanWheel(const WheelPlanRequest& request) try {
  if (request.control.canceled()) {
    return Failure(LocalPlanStatus::kCanceled, "REQUEST_CANCELED");
  }
  const auto* point = std::get_if<PointGoal>(&request.goal_odom.target);
  const auto start_yaw = YawFromQuaternion(request.start.pose.orientation);
  if (request.terrain == nullptr || request.capability == nullptr ||
      !ValidTerrain(*request.terrain) ||
      !ValidCapability(*request.capability) || !Finite(request.start.pose) ||
      !Finite(request.start.velocity) ||
      request.maximum_search_states < 2U ||
      !start_yaw.has_value() || point == nullptr ||
      !std::isfinite(point->position_m.x) ||
      !std::isfinite(point->position_m.y) ||
      !std::isfinite(point->tolerance_m) ||
      point->tolerance_m < 0.0 ||
      (request.goal_odom.yaw_rad.has_value() &&
       !std::isfinite(*request.goal_odom.yaw_rad)) ||
      !std::isfinite(request.goal_odom.yaw_tolerance_rad) ||
      request.goal_odom.yaw_tolerance_rad < 0.0 ||
      request.search.epsilon_schedule !=
          std::array<double, 4>{2.5, 2.0, 1.5, 1.0}) {
    return Failure(LocalPlanStatus::kInvalidInput, "WHEEL_INPUT_INVALID");
  }
  const auto goal_cell = request.terrain->map->PositionToCell(
      Vec2{.x = point->position_m.x, .y = point->position_m.y});
  const auto start_cell = request.terrain->map->PositionToCell(
      Vec2{.x = request.start.pose.position_m.x,
           .y = request.start.pose.position_m.y});
  if (!goal_cell.has_value() || !start_cell.has_value()) {
    return Failure(LocalPlanStatus::kInvalidInput,
                   "WHEEL_POSE_OUTSIDE_LOCAL_MAP");
  }
  WheelSearchGraph graph{request, *point, request.goal_odom.yaw_rad};
  const auto decorate = [&](WheelPlanResult result) {
    result.edge_validation_cache_hits = graph.validation_cache_hits();
    result.quantization_alias_states = graph.quantization_alias_states();
    result.quantized_state_reuses = graph.quantized_state_reuses();
    result.quantized_endpoint_aliases = graph.quantized_endpoint_aliases();
    result.quantized_state_count = graph.quantized_state_count();
    result.sweep_cell_checks = graph.sweep_cell_checks();
    result.finest_xy_key_resolution_m = graph.finest_xy_key_resolution_m();
    result.maximum_yaw_bins = graph.maximum_yaw_bins();
    result.cost_scales = graph.cost_scales();
    return result;
  };
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

  const shared::anytime::AraStarResult search =
      shared::anytime::SearchAnytimeAraStar(
          shared::anytime::AraStarProblem{
              .state_count = graph.state_count(),
              .start_state = 0U,
              .expand = [&](const std::size_t state,
                            std::vector<shared::GraphEdge>& edges) {
                graph.Expand(state, edges);
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
      return decorate(Failure(
          graph.resource_exhausted() ? LocalPlanStatus::kPlannerError
                                     : LocalPlanStatus::kNoPath,
          graph.resource_exhausted() ? "WHEEL_SEARCH_CAPACITY_EXHAUSTED"
                                     : "NO_PATH",
          metrics));
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
      .cost_components = cost_components,
      .cost = candidate.cost,
  });
} catch (...) {
  return Failure(LocalPlanStatus::kPlannerError, "PLANNER_ERROR");
}

}  // namespace lunar::pure_planning::wheel
