#include "wheel/wheel_sweep_validator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace lunar::planning::wheel {
namespace {

constexpr double kComparisonTolerance = 1.0e-9;

[[nodiscard]] WheelSweepValidation Failure(
    std::string reason_code, const std::size_t sample_count = 0U) {
  return WheelSweepValidation{
      .valid = false,
      .canceled = false,
      .sample_count = sample_count,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool IsFinite(const WheelPose& pose) noexcept {
  return std::isfinite(pose.position_m.x) &&
         std::isfinite(pose.position_m.y) &&
         std::isfinite(pose.position_m.z) && std::isfinite(pose.yaw_rad);
}

}  // namespace

WheelSweepValidator::WheelSweepValidator(
    const shared::SafeProjection& projection,
    const WheeledCapability& capability,
    const std::size_t maximum_subdivisions) noexcept
    : projection_(&projection),
      capability_(&capability),
      maximum_subdivisions_(maximum_subdivisions) {
  for (const Vec2& vertex : capability.footprint_xy_m) {
    footprint_support_radius_m_ = std::max(
        footprint_support_radius_m_, std::hypot(vertex.x, vertex.y));
  }
}

WheelSweepValidation WheelSweepValidator::Validate(
    const WheelTransition& transition,
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
      maximum_subdivisions_ == 0U ||
      !IsFinite(transition.source_pose) ||
      !IsFinite(transition.target_pose) ||
      !std::isfinite(transition.curvature_per_m) ||
      !std::isfinite(capability_->maximum_curvature_per_m) ||
      capability_->maximum_curvature_per_m <= 0.0) {
    return Failure("WHEEL_SWEEP_REQUEST_INVALID");
  }
  if (std::abs(transition.curvature_per_m) >
      capability_->maximum_curvature_per_m + kComparisonTolerance) {
    return Failure("WHEEL_CURVATURE_LIMIT");
  }

  const double translation_m = std::hypot(
      transition.target_pose.position_m.x -
          transition.source_pose.position_m.x,
      transition.target_pose.position_m.y -
          transition.source_pose.position_m.y);
  const double yaw_delta = std::abs(ShortestYawDelta(
      transition.source_pose.yaw_rad, transition.target_pose.yaw_rad));
  const double swept_distance_m =
      translation_m + yaw_delta * footprint_support_radius_m_;
  const double maximum_step_m =
      projection_->source_map()->resolution_m() * 0.25;
  const std::size_t required_subdivisions = std::max<std::size_t>(
      1U, static_cast<std::size_t>(
              std::ceil(swept_distance_m / maximum_step_m)));
  if (required_subdivisions > maximum_subdivisions_) {
    return Failure("WHEEL_SWEEP_RESOLUTION_LIMIT");
  }

  const auto& map = *projection_->source_map();
  const double resolution = map.resolution_m();
  const double origin_x = map.origin_m().x;
  const double origin_y = map.origin_m().y;
  const double signed_yaw_delta = ShortestYawDelta(
      transition.source_pose.yaw_rad, transition.target_pose.yaw_rad);
  std::size_t sample_count = 0U;
  for (std::size_t sample = 0U;
       sample <= required_subdivisions; ++sample) {
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
    const double yaw =
        transition.source_pose.yaw_rad + ratio * signed_yaw_delta;
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    double minimum_x = std::numeric_limits<double>::infinity();
    double maximum_x = -std::numeric_limits<double>::infinity();
    double minimum_y = std::numeric_limits<double>::infinity();
    double maximum_y = -std::numeric_limits<double>::infinity();
    for (const Vec2& vertex : capability_->footprint_xy_m) {
      if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y)) {
        return Failure("WHEEL_FOOTPRINT_NONFINITE", sample_count);
      }
      const double x = center_x + cosine * vertex.x - sine * vertex.y;
      const double y = center_y + sine * vertex.x + cosine * vertex.y;
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
        if (!projection_->HardFeasible(shared::GridCell{
                .x = static_cast<std::int32_t>(x),
                .y = static_cast<std::int32_t>(y),
            })) {
          return Failure("WHEEL_SWEEP_COLLISION", sample_count);
        }
      }
    }
  }
  return WheelSweepValidation{
      .valid = true,
      .canceled = false,
      .sample_count = sample_count,
      .reason_code = "WHEEL_SWEEP_VALID",
  };
}

}  // namespace lunar::planning::wheel
