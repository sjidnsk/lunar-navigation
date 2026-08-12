#include "wheel/wheel_sweep_validator.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>

#include "shared/terrain_checks.hpp"

namespace lunar::planning::wheel {
namespace {

constexpr double kComparisonTolerance = 1.0e-9;

[[nodiscard]] WheelSweepValidation
Failure(std::string reason_code, const std::size_t sample_count = 0U) {
  return WheelSweepValidation{
      .valid = false,
      .canceled = false,
      .sample_count = sample_count,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool IsFinite(const WheelPose &pose) noexcept {
  return std::isfinite(pose.position_m.x) && std::isfinite(pose.position_m.y) &&
         std::isfinite(pose.position_m.z) && std::isfinite(pose.yaw_rad);
}

[[nodiscard]] double Cross(const Vec2 &first, const Vec2 &second,
                           const Vec2 &third) noexcept {
  return (second.x - first.x) * (third.y - first.y) -
         (second.y - first.y) * (third.x - first.x);
}

[[nodiscard]] bool PointOnSegment(const Vec2 &point, const Vec2 &start,
                                  const Vec2 &finish) noexcept {
  return std::abs(Cross(start, finish, point)) <= kComparisonTolerance &&
         point.x + kComparisonTolerance >= std::min(start.x, finish.x) &&
         point.x <= std::max(start.x, finish.x) + kComparisonTolerance &&
         point.y + kComparisonTolerance >= std::min(start.y, finish.y) &&
         point.y <= std::max(start.y, finish.y) + kComparisonTolerance;
}

[[nodiscard]] bool SegmentsIntersect(const Vec2 &first_start,
                                     const Vec2 &first_finish,
                                     const Vec2 &second_start,
                                     const Vec2 &second_finish) noexcept {
  const double first_side_start =
      Cross(first_start, first_finish, second_start);
  const double first_side_finish =
      Cross(first_start, first_finish, second_finish);
  const double second_side_start =
      Cross(second_start, second_finish, first_start);
  const double second_side_finish =
      Cross(second_start, second_finish, first_finish);
  const bool proper_crossing = ((first_side_start > kComparisonTolerance &&
                                 first_side_finish < -kComparisonTolerance) ||
                                (first_side_start < -kComparisonTolerance &&
                                 first_side_finish > kComparisonTolerance)) &&
                               ((second_side_start > kComparisonTolerance &&
                                 second_side_finish < -kComparisonTolerance) ||
                                (second_side_start < -kComparisonTolerance &&
                                 second_side_finish > kComparisonTolerance));
  return proper_crossing ||
         PointOnSegment(second_start, first_start, first_finish) ||
         PointOnSegment(second_finish, first_start, first_finish) ||
         PointOnSegment(first_start, second_start, second_finish) ||
         PointOnSegment(first_finish, second_start, second_finish);
}

[[nodiscard]] bool PointInPolygon(const Vec2 &point,
                                  const std::vector<Vec2> &polygon) noexcept {
  bool inside = false;
  for (std::size_t current = 0U, previous = polygon.size() - 1U;
       current < polygon.size(); previous = current++) {
    const Vec2 &start = polygon[previous];
    const Vec2 &finish = polygon[current];
    if (PointOnSegment(point, start, finish)) {
      return true;
    }
    if ((start.y > point.y) == (finish.y > point.y)) {
      continue;
    }
    const double crossing_x = start.x + (point.y - start.y) *
                                            (finish.x - start.x) /
                                            (finish.y - start.y);
    if (point.x < crossing_x) {
      inside = !inside;
    }
  }
  return inside;
}

[[nodiscard]] bool PolygonIntersectsCell(const std::vector<Vec2> &polygon,
                                         const double minimum_x,
                                         const double minimum_y,
                                         const double maximum_x,
                                         const double maximum_y) noexcept {
  const auto inside_cell = [&](const Vec2 &point) {
    return point.x + kComparisonTolerance >= minimum_x &&
           point.x <= maximum_x + kComparisonTolerance &&
           point.y + kComparisonTolerance >= minimum_y &&
           point.y <= maximum_y + kComparisonTolerance;
  };
  if (std::ranges::any_of(polygon, inside_cell)) {
    return true;
  }
  const std::array<Vec2, 4U> corners{
      Vec2{minimum_x, minimum_y},
      Vec2{maximum_x, minimum_y},
      Vec2{maximum_x, maximum_y},
      Vec2{minimum_x, maximum_y},
  };
  if (std::ranges::any_of(corners, [&](const Vec2 &corner) {
        return PointInPolygon(corner, polygon);
      })) {
    return true;
  }
  for (std::size_t polygon_index = 0U; polygon_index < polygon.size();
       ++polygon_index) {
    const Vec2 &polygon_start = polygon[polygon_index];
    const Vec2 &polygon_finish = polygon[(polygon_index + 1U) % polygon.size()];
    for (std::size_t cell_index = 0U; cell_index < corners.size();
         ++cell_index) {
      if (SegmentsIntersect(polygon_start, polygon_finish, corners[cell_index],
                            corners[(cell_index + 1U) % corners.size()])) {
        return true;
      }
    }
  }
  return false;
}

} // namespace

std::size_t WheelTerrainPoseKeyHash::operator()(
    const WheelTerrainPoseKey& key) const noexcept {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto append = [&](std::uint64_t value) {
    for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
      hash ^= value & 0xffU;
      hash *= 1099511628211ULL;
      value >>= 8U;
    }
  };
  append(key.x_bits);
  append(key.y_bits);
  append(key.yaw_bits);
  return static_cast<std::size_t>(hash);
}

WheelSweepValidator::WheelSweepValidator(
    const shared::SafeProjection &projection,
    const WheeledCapability &capability) noexcept
    : projection_(&projection), capability_(&capability) {
  for (const Vec2 &vertex : capability.footprint_xy_m) {
    footprint_support_radius_m_ =
        std::max(footprint_support_radius_m_, std::hypot(vertex.x, vertex.y));
  }
  if (projection.source_map() != nullptr) {
    terrain_cache_.reserve(projection.source_map()->cell_count() * 4U);
  }
}

WheelSweepValidator::WheelSweepValidator(
    const shared::SafeProjection &physical_projection,
    const WheeledCapability &capability,
    const hierarchical::LocalSearchDomain &search_domain) noexcept
    : WheelSweepValidator(physical_projection, capability) {
  search_domain_ = &search_domain;
}

WheelSweepValidation
WheelSweepValidator::Validate(const WheelTransition &transition,
                              const std::stop_token stop_token) const {
  if (stop_token.stop_requested()) {
    return WheelSweepValidation{
        .valid = false,
        .canceled = true,
        .sample_count = 0U,
        .reason_code = "REQUEST_CANCELED",
    };
  }
  if (projection_ == nullptr || projection_->source_map() == nullptr ||
      capability_ == nullptr || capability_->footprint_xy_m.size() < 3U ||
      !IsFinite(transition.source_pose) || !IsFinite(transition.target_pose) ||
      !std::isfinite(transition.curvature_per_m) ||
      !std::isfinite(capability_->maximum_curvature_per_m) ||
      capability_->maximum_curvature_per_m <= 0.0) {
    return Failure("WHEEL_SWEEP_REQUEST_INVALID");
  }
  if (std::abs(transition.curvature_per_m) >
      capability_->maximum_curvature_per_m + kComparisonTolerance) {
    return Failure("WHEEL_CURVATURE_LIMIT");
  }

  const auto normalized_bits = [](double value) {
    if (value == 0.0) {
      value = 0.0;
    }
    return std::bit_cast<std::uint64_t>(value);
  };
  const double translation_m = std::hypot(
      transition.target_pose.position_m.x - transition.source_pose.position_m.x,
      transition.target_pose.position_m.y -
          transition.source_pose.position_m.y);
  const double yaw_delta = std::abs(ShortestYawDelta(
      transition.source_pose.yaw_rad, transition.target_pose.yaw_rad));
  const double swept_distance_m =
      translation_m + yaw_delta * footprint_support_radius_m_;
  const double maximum_step_m =
      projection_->source_map()->resolution_m() * 0.25;
  const std::size_t required_subdivisions = std::max<std::size_t>(
      1U,
      static_cast<std::size_t>(std::ceil(swept_distance_m / maximum_step_m)));
  const auto &map = *projection_->source_map();
  if (search_domain_ != nullptr &&
      (search_domain_->width() != map.width() ||
       search_domain_->height() != map.height())) {
    return Failure("WHEEL_SWEEP_REQUEST_INVALID");
  }
  const double resolution = map.resolution_m();
  const double origin_x = map.origin_m().x;
  const double origin_y = map.origin_m().y;
  const double signed_yaw_delta = ShortestYawDelta(
      transition.source_pose.yaw_rad, transition.target_pose.yaw_rad);
  std::size_t sample_count = 0U;
  double maximum_surface_slope_rad = 0.0;
  double maximum_roughness_m = 0.0;
  double maximum_positive_relief_m = 0.0;
  double minimum_underbody_clearance_m =
      std::numeric_limits<double>::infinity();
  for (std::size_t sample = 0U; sample <= required_subdivisions; ++sample) {
    if (stop_token.stop_requested()) {
      return WheelSweepValidation{
          .valid = false,
          .canceled = true,
          .sample_count = sample_count,
          .reason_code = "REQUEST_CANCELED",
      };
    }
    ++sample_count;
    const double ratio = static_cast<double>(sample) /
                         static_cast<double>(required_subdivisions);
    const double center_x = transition.source_pose.position_m.x +
                            ratio * (transition.target_pose.position_m.x -
                                     transition.source_pose.position_m.x);
    const double center_y = transition.source_pose.position_m.y +
                            ratio * (transition.target_pose.position_m.y -
                                     transition.source_pose.position_m.y);
    const auto center_cell = map.PositionToCell(Vec2{
        .x = center_x,
        .y = center_y,
    });
    if (search_domain_ != nullptr &&
        (!center_cell.has_value() ||
         !search_domain_->Contains(*center_cell))) {
      return Failure("WHEEL_SWEEP_OUTSIDE_SEARCH_DOMAIN", sample_count);
    }
    const double yaw =
        transition.source_pose.yaw_rad + ratio * signed_yaw_delta;
    const WheelTerrainPoseKey key{
        .x_bits = normalized_bits(center_x),
        .y_bits = normalized_bits(center_y),
        .yaw_bits = normalized_bits(yaw),
    };
    auto found = terrain_cache_.find(key);
    if (found == terrain_cache_.end()) {
      found = terrain_cache_.emplace(
          key, shared::EvaluateWheelTerrainPose(
                   map, Vec2{.x = center_x, .y = center_y}, yaw,
                   *capability_)).first;
      ++terrain_evaluation_count_;
    }
    const shared::WheelTerrainPoseEvaluation& terrain = found->second;
    if (!terrain.feasible) {
      return Failure(
          terrain.rejection_codes.empty()
              ? "WHEEL_TERRAIN_POSE_INVALID"
              : terrain.rejection_codes.front(),
          sample_count);
    }
    maximum_surface_slope_rad = std::max(
        maximum_surface_slope_rad, terrain.surface_slope_rad);
    maximum_roughness_m = std::max(
        maximum_roughness_m, terrain.roughness_m);
    maximum_positive_relief_m = std::max(
        maximum_positive_relief_m, terrain.maximum_positive_relief_m);
    minimum_underbody_clearance_m = std::min(
        minimum_underbody_clearance_m,
        terrain.minimum_underbody_clearance_m);
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    double minimum_x = std::numeric_limits<double>::infinity();
    double maximum_x = -std::numeric_limits<double>::infinity();
    double minimum_y = std::numeric_limits<double>::infinity();
    double maximum_y = -std::numeric_limits<double>::infinity();
    std::vector<Vec2> footprint;
    footprint.reserve(capability_->footprint_xy_m.size());
    for (const Vec2 &vertex : capability_->footprint_xy_m) {
      if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y)) {
        return Failure("WHEEL_FOOTPRINT_NONFINITE", sample_count);
      }
      const double x = center_x + cosine * vertex.x - sine * vertex.y;
      const double y = center_y + sine * vertex.x + cosine * vertex.y;
      footprint.push_back(Vec2{.x = x, .y = y});
      minimum_x = std::min(minimum_x, x);
      maximum_x = std::max(maximum_x, x);
      minimum_y = std::min(minimum_y, y);
      maximum_y = std::max(maximum_y, y);
    }
    const auto minimum_cell_x = static_cast<std::int64_t>(
        std::floor((minimum_x - origin_x) / resolution));
    const auto maximum_cell_x = static_cast<std::int64_t>(
        std::floor((maximum_x - origin_x) / resolution));
    const auto minimum_cell_y = static_cast<std::int64_t>(
        std::floor((minimum_y - origin_y) / resolution));
    const auto maximum_cell_y = static_cast<std::int64_t>(
        std::floor((maximum_y - origin_y) / resolution));
    if (minimum_cell_x < 0 || minimum_cell_y < 0 ||
        maximum_cell_x >= static_cast<std::int64_t>(map.width()) ||
        maximum_cell_y >= static_cast<std::int64_t>(map.height())) {
      return Failure("WHEEL_SWEEP_OUTSIDE_MAP", sample_count);
    }
    for (std::int64_t y = minimum_cell_y; y <= maximum_cell_y; ++y) {
      for (std::int64_t x = minimum_cell_x; x <= maximum_cell_x; ++x) {
        const shared::GridCell cell{
            .x = static_cast<std::int32_t>(x),
            .y = static_cast<std::int32_t>(y),
        };
        if (!projection_->IntrinsicFeasible(cell) &&
            PolygonIntersectsCell(
                footprint, origin_x + static_cast<double>(x) * resolution,
                origin_y + static_cast<double>(y) * resolution,
                origin_x + static_cast<double>(x + 1) * resolution,
                origin_y + static_cast<double>(y + 1) * resolution)) {
          return Failure("WHEEL_SWEEP_COLLISION", sample_count);
        }
      }
    }
  }
  return WheelSweepValidation{
      .valid = true,
      .canceled = false,
      .sample_count = sample_count,
      .maximum_surface_slope_rad = maximum_surface_slope_rad,
      .maximum_roughness_m = maximum_roughness_m,
      .maximum_positive_relief_m = maximum_positive_relief_m,
      .minimum_underbody_clearance_m = minimum_underbody_clearance_m,
      .reason_code = "WHEEL_SWEEP_VALID",
  };
}

std::size_t WheelSweepValidator::terrain_evaluation_count() const noexcept {
  return terrain_evaluation_count_;
}

} // namespace lunar::planning::wheel
