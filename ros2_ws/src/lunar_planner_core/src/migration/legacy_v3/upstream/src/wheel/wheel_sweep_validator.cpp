#include "lunar_path_planner/v3/wheel/wheel_sweep_validator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace lunar::planning::v3 {
namespace {

constexpr double kGeometryTolerance = 1.0e-9;
constexpr std::size_t kMaximumSupportedSubdivisionDepth = 16U;

[[nodiscard]] ValidationReport Issue(
    std::string reason_code,
    std::string message) {
  return {
      .issues =
          {
              {
                  .field_path = "/segments",
                  .reason_code = std::move(reason_code),
                  .message = std::move(message),
              },
          },
  };
}

[[nodiscard]] bool Finite(const Vec2& point) noexcept {
  return std::isfinite(point.x) && std::isfinite(point.y);
}

[[nodiscard]] bool Finite(const Vec3& point) noexcept {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}

[[nodiscard]] bool Finite(const PoseXyzYaw& pose) noexcept {
  return Finite(pose.position_m) && std::isfinite(pose.yaw_rad);
}

[[nodiscard]] bool PointInConvexPolygon(
    std::span<const Vec2> polygon,
    const Vec2& point) noexcept {
  if (polygon.size() < 3U) {
    return false;
  }
  double sign = 0.0;
  for (std::size_t index = 0U; index < polygon.size();
       ++index) {
    const Vec2& first = polygon[index];
    const Vec2& second =
        polygon[(index + 1U) % polygon.size()];
    const double cross =
        (second.x - first.x) * (point.y - first.y) -
        (second.y - first.y) * (point.x - first.x);
    if (std::abs(cross) <= kGeometryTolerance) {
      continue;
    }
    if (sign == 0.0) {
      sign = cross;
    } else if ((sign > 0.0) != (cross > 0.0)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::optional<Cell> PositionCell(
    const GridGeometry& geometry,
    const Vec2& point) noexcept {
  if (!Finite(point) || geometry.resolution_m <= 0.0) {
    return std::nullopt;
  }
  const double grid_x =
      (point.x - geometry.origin_m.x) /
      geometry.resolution_m;
  const double grid_y =
      (point.y - geometry.origin_m.y) /
      geometry.resolution_m;
  if (!std::isfinite(grid_x) || !std::isfinite(grid_y) ||
      grid_x < 0.0 || grid_y < 0.0 ||
      grid_x >= static_cast<double>(geometry.width) ||
      grid_y >= static_cast<double>(geometry.height)) {
    return std::nullopt;
  }
  return Cell{
      static_cast<std::int32_t>(std::floor(grid_x)),
      static_cast<std::int32_t>(std::floor(grid_y)),
  };
}

[[nodiscard]] std::vector<Vec2> TransformFootprint(
    const ExtrudedConvexFootprint& footprint,
    const PoseXyzYaw& pose) {
  std::vector<Vec2> transformed;
  transformed.reserve(footprint.vertices_xy_m.size());
  const double cosine = std::cos(pose.yaw_rad);
  const double sine = std::sin(pose.yaw_rad);
  for (const Vec2& vertex : footprint.vertices_xy_m) {
    transformed.push_back(
        {
            pose.position_m.x + cosine * vertex.x -
                sine * vertex.y,
            pose.position_m.y + sine * vertex.x +
                cosine * vertex.y,
        });
  }
  return transformed;
}

[[nodiscard]] ValidationReport ValidateFootprintAtPose(
    const SafeProjection& projection,
    const WheelCollisionEnvelope& envelope,
    const WheelSweepValidationConfig& config,
    const PoseXyzYaw& pose) {
  if (!Finite(pose) ||
      envelope.footprint.vertices_xy_m.size() < 3U ||
      config.maximum_footprint_cells_per_sample == 0U ||
      !std::isfinite(config.additional_margin_m) ||
      config.additional_margin_m < 0.0) {
    return Issue("INVALID_SWEEP_REQUEST",
                 "wheel sweep request is invalid");
  }
  const auto polygon =
      TransformFootprint(envelope.footprint, pose);
  if (!std::ranges::all_of(
          polygon,
          [](const Vec2& point) { return Finite(point); })) {
    return Issue("INVALID_SWEEP_REQUEST",
                 "wheel footprint is not finite");
  }

  double minimum_x = std::numeric_limits<double>::infinity();
  double minimum_y = std::numeric_limits<double>::infinity();
  double maximum_x = -std::numeric_limits<double>::infinity();
  double maximum_y = -std::numeric_limits<double>::infinity();
  for (const Vec2& point : polygon) {
    minimum_x = std::min(minimum_x, point.x);
    minimum_y = std::min(minimum_y, point.y);
    maximum_x = std::max(maximum_x, point.x);
    maximum_y = std::max(maximum_y, point.y);
  }
  const auto& geometry = projection.geometry;
  if (geometry.width == 0U || geometry.height == 0U ||
      geometry.resolution_m <= 0.0) {
    return Issue("INVALID_SWEEP_REQUEST",
                 "projection geometry is invalid");
  }
  const auto lower_x = static_cast<std::int64_t>(std::floor(
      (minimum_x - geometry.origin_m.x) /
      geometry.resolution_m));
  const auto lower_y = static_cast<std::int64_t>(std::floor(
      (minimum_y - geometry.origin_m.y) /
      geometry.resolution_m));
  const auto upper_x = static_cast<std::int64_t>(std::floor(
      (maximum_x - geometry.origin_m.x) /
      geometry.resolution_m));
  const auto upper_y = static_cast<std::int64_t>(std::floor(
      (maximum_y - geometry.origin_m.y) /
      geometry.resolution_m));
  if (lower_x < 0 || lower_y < 0 ||
      upper_x >= static_cast<std::int64_t>(geometry.width) ||
      upper_y >= static_cast<std::int64_t>(geometry.height)) {
    return Issue("SWEPT_COLLISION",
                 "wheel footprint leaves the map");
  }
  const auto span_x = static_cast<std::uint64_t>(
      upper_x - lower_x + 1);
  const auto span_y = static_cast<std::uint64_t>(
      upper_y - lower_y + 1);
  if (span_y != 0U &&
      span_x >
          std::numeric_limits<std::uint64_t>::max() / span_y) {
    return Issue("CONTINUOUS_VALIDATION_INCONCLUSIVE",
                 "wheel footprint cell count overflowed");
  }
  const std::uint64_t cell_count = span_x * span_y;
  if (cell_count >
      config.maximum_footprint_cells_per_sample) {
    return Issue("CONTINUOUS_VALIDATION_INCONCLUSIVE",
                 "wheel footprint cell bound was exceeded");
  }

  const auto validate_cell =
      [&](Cell cell) -> ValidationReport {
    if (!projection.Known(cell)) {
      return Issue("UNKNOWN_OR_UNSAFE_CELL",
                   "wheel footprint touches an unknown cell");
    }
    if (!projection.HardFeasible(cell)) {
      return Issue("SWEPT_COLLISION",
                   "wheel footprint touches a hard obstacle");
    }
    const double required_clearance =
        envelope.horizontal_tracking_error_m +
        envelope.minimum_clearance_m +
        config.additional_margin_m;
    if (static_cast<double>(
            projection.ClearanceMeters(cell)) +
            kGeometryTolerance <
        required_clearance) {
      return Issue("INSUFFICIENT_CLEARANCE",
                   "wheel footprint clearance is insufficient");
    }
    return {};
  };

  for (const Vec2& vertex : polygon) {
    const auto cell = PositionCell(geometry, vertex);
    if (!cell.has_value()) {
      return Issue("SWEPT_COLLISION",
                   "wheel footprint leaves the map");
    }
    auto report = validate_cell(*cell);
    if (!report.ok()) {
      return report;
    }
  }
  for (std::int64_t y = lower_y; y <= upper_y; ++y) {
    for (std::int64_t x = lower_x; x <= upper_x; ++x) {
      const Vec2 center{
          geometry.origin_m.x +
              (static_cast<double>(x) + 0.5) *
                  geometry.resolution_m,
          geometry.origin_m.y +
              (static_cast<double>(y) + 0.5) *
                  geometry.resolution_m,
      };
      if (!PointInConvexPolygon(polygon, center)) {
        continue;
      }
      auto report = validate_cell(
          {static_cast<std::int32_t>(x),
           static_cast<std::int32_t>(y)});
      if (!report.ok()) {
        return report;
      }
    }
  }
  return {};
}

[[nodiscard]] bool ValidSubdivisionConfig(
    const WheelSweepValidationConfig& config) noexcept {
  return config.maximum_subdivisions > 0U &&
         config.maximum_subdivisions <=
             kMaximumSupportedSubdivisionDepth;
}

[[nodiscard]] std::size_t AvailableSamples(
    std::size_t subdivision_depth) noexcept {
  return std::size_t{1U} << subdivision_depth;
}

[[nodiscard]] double EvaluatePolynomial(
    const CubicPolynomialSegment& segment,
    DurationNanoseconds absolute_offset) noexcept {
  const auto relative =
      absolute_offset.value - segment.start_offset.value;
  const double seconds =
      std::chrono::duration<double>(relative).count();
  const auto& c = segment.coefficients;
  return ((c[3] * seconds + c[2]) * seconds + c[1]) *
             seconds +
         c[0];
}

[[nodiscard]] std::optional<double> EvaluateTrajectory(
    const PiecewiseCubicScalarTrajectory& trajectory,
    DurationNanoseconds offset) noexcept {
  for (const auto& segment : trajectory.segments) {
    if (offset.value >= segment.start_offset.value &&
        offset.value <= segment.end_offset.value) {
      return EvaluatePolynomial(segment, offset);
    }
  }
  return std::nullopt;
}

[[nodiscard]] bool ValidateTimeScaling(
    const MonotoneTimeScaling& scaling) noexcept {
  if (scaling.segments.empty()) {
    return false;
  }
  double previous_value = -kGeometryTolerance;
  for (const auto& segment : scaling.segments) {
    if (segment.end_offset.value <=
        segment.start_offset.value) {
      return false;
    }
    for (std::size_t sample = 0U; sample <= 8U; ++sample) {
      const auto duration =
          segment.end_offset.value - segment.start_offset.value;
      const auto offset = DurationNanoseconds{
          segment.start_offset.value +
          duration * static_cast<std::int64_t>(sample) / 8};
      const double value = EvaluatePolynomial(segment, offset);
      if (!std::isfinite(value) ||
          value + kGeometryTolerance < previous_value) {
        return false;
      }
      previous_value = value;
    }
  }
  return std::abs(previous_value - 1.0) <= 1.0e-7;
}

[[nodiscard]] std::optional<PoseXyzYaw> EvaluateSpline(
    const ClampedCubicBSplinePath& spline,
    double parameter) {
  constexpr std::size_t kDegree = 3U;
  if (spline.control_points.size() < 4U ||
      spline.knots.size() !=
          spline.control_points.size() + kDegree + 1U ||
      parameter < 0.0 || parameter > 1.0) {
    return std::nullopt;
  }
  for (std::size_t index = 1U; index < spline.knots.size();
       ++index) {
    if (!std::isfinite(spline.knots[index]) ||
        spline.knots[index] < spline.knots[index - 1U]) {
      return std::nullopt;
    }
  }
  std::size_t span = spline.control_points.size() - 1U;
  if (parameter < 1.0) {
    for (std::size_t index = kDegree;
         index < spline.control_points.size(); ++index) {
      if (parameter >= spline.knots[index] &&
          parameter < spline.knots[index + 1U]) {
        span = index;
        break;
      }
    }
  }
  std::array<PoseXyzYaw, 4U> work{};
  for (std::size_t index = 0U; index <= kDegree; ++index) {
    work[index] =
        spline.control_points[span - kDegree + index];
  }
  for (std::size_t level = 1U; level <= kDegree; ++level) {
    for (std::size_t index = kDegree; index >= level; --index) {
      const std::size_t knot_index =
          span - kDegree + index;
      const double denominator =
          spline.knots[knot_index + kDegree - level + 1U] -
          spline.knots[knot_index];
      const double alpha =
          denominator > 0.0
              ? (parameter - spline.knots[knot_index]) /
                    denominator
              : 0.0;
      const auto blend = [alpha](double lhs, double rhs) {
        return (1.0 - alpha) * lhs + alpha * rhs;
      };
      work[index] = {
          .position_m =
              {
                  blend(work[index - 1U].position_m.x,
                        work[index].position_m.x),
                  blend(work[index - 1U].position_m.y,
                        work[index].position_m.y),
                  blend(work[index - 1U].position_m.z,
                        work[index].position_m.z),
              },
          .yaw_rad =
              blend(work[index - 1U].yaw_rad,
                    work[index].yaw_rad),
      };
      if (index == level) {
        break;
      }
    }
  }
  return work[kDegree];
}

}  // namespace

WheelSweepValidator::WheelSweepValidator(
    const SafeProjection& projection,
    WheelCollisionEnvelope envelope,
    WheelSweepValidationConfig config)
    : projection_(&projection),
      envelope_(std::move(envelope)),
      config_(config) {}

ValidationReport WheelSweepValidator::ValidatePrimitiveChain(
    const ValidatedPrimitiveChain& chain) const {
  if (chain.primitives.empty() ||
      !ValidSubdivisionConfig(config_)) {
    return Issue("CONTINUOUS_VALIDATION_INCONCLUSIVE",
                 "primitive sweep subdivision bound is invalid");
  }
  const std::size_t available =
      AvailableSamples(config_.maximum_subdivisions);
  for (const ValidatedPrimitive& primitive : chain.primitives) {
    if (!Finite(primitive.start_pose) ||
        !Finite(primitive.end_pose) ||
        primitive.nominal_duration.value.count() <= 0 ||
        primitive.validation_ref.id.empty()) {
      return Issue("INVALID_PRIMITIVE_CHAIN",
                   "primitive chain metadata is invalid");
    }
    const double distance_m =
        std::hypot(primitive.end_pose.position_m.x -
                       primitive.start_pose.position_m.x,
                   primitive.end_pose.position_m.y -
                       primitive.start_pose.position_m.y);
    const double yaw_change =
        std::abs(primitive.end_pose.yaw_rad -
                 primitive.start_pose.yaw_rad);
    const double linear_step =
        std::max(projection_->geometry.resolution_m * 0.25,
                 kGeometryTolerance);
    const std::size_t required =
        std::max<std::size_t>(
            1U,
            static_cast<std::size_t>(std::ceil(
                std::max(distance_m / linear_step,
                         yaw_change / 0.05))));
    if (required > available) {
      return Issue("CONTINUOUS_VALIDATION_INCONCLUSIVE",
                   "primitive sweep sample bound is insufficient");
    }
    for (std::size_t sample = 0U; sample <= required; ++sample) {
      const double fraction =
          static_cast<double>(sample) /
          static_cast<double>(required);
      const PoseXyzYaw pose{
          .position_m =
              {
                  primitive.start_pose.position_m.x +
                      fraction *
                          (primitive.end_pose.position_m.x -
                           primitive.start_pose.position_m.x),
                  primitive.start_pose.position_m.y +
                      fraction *
                          (primitive.end_pose.position_m.y -
                           primitive.start_pose.position_m.y),
                  primitive.start_pose.position_m.z +
                      fraction *
                          (primitive.end_pose.position_m.z -
                           primitive.start_pose.position_m.z),
              },
          .yaw_rad =
              primitive.start_pose.yaw_rad +
              fraction * (primitive.end_pose.yaw_rad -
                          primitive.start_pose.yaw_rad),
      };
      auto report = ValidateFootprintAtPose(
          *projection_, envelope_, config_, pose);
      if (!report.ok()) {
        return report;
      }
    }
  }
  return {};
}

ValidationReport WheelSweepValidator::ValidateSpline(
    const ClampedCubicBSplinePath& spline,
    const MonotoneTimeScaling& time_scaling) const {
  if (!ValidSubdivisionConfig(config_) ||
      !ValidateTimeScaling(time_scaling)) {
    return Issue("CONTINUOUS_VALIDATION_INCONCLUSIVE",
                 "spline validation bounds or time scaling are invalid");
  }
  const std::size_t samples =
      AvailableSamples(config_.maximum_subdivisions);
  for (std::size_t sample = 0U; sample <= samples; ++sample) {
    const double parameter =
        static_cast<double>(sample) /
        static_cast<double>(samples);
    const auto pose = EvaluateSpline(spline, parameter);
    if (!pose.has_value()) {
      return Issue("INVALID_SPLINE",
                   "wheel spline representation is invalid");
    }
    auto report = ValidateFootprintAtPose(
        *projection_, envelope_, config_, *pose);
    if (!report.ok()) {
      return report;
    }
  }
  return {};
}

ValidationReport WheelSweepValidator::ValidateSpin(
    const Vec3& fixed_position,
    const PiecewiseCubicScalarTrajectory& yaw) const {
  if (!Finite(fixed_position) || yaw.segments.empty() ||
      !ValidSubdivisionConfig(config_)) {
    return Issue("CONTINUOUS_VALIDATION_INCONCLUSIVE",
                 "spin validation bounds are invalid");
  }
  for (std::size_t index = 0U; index < yaw.segments.size();
       ++index) {
    const auto& segment = yaw.segments[index];
    if (segment.end_offset.value <=
            segment.start_offset.value ||
        (index > 0U &&
         yaw.segments[index - 1U].end_offset.value !=
             segment.start_offset.value)) {
      return Issue("INVALID_SPIN_TRAJECTORY",
                   "spin yaw segments are not contiguous");
    }
  }
  const auto start = yaw.segments.front().start_offset;
  const auto finish = yaw.segments.back().end_offset;
  const std::size_t samples =
      AvailableSamples(config_.maximum_subdivisions);
  const auto total = finish.value - start.value;
  for (std::size_t sample = 0U; sample <= samples; ++sample) {
    const auto offset = DurationNanoseconds{
        start.value +
        total * static_cast<std::int64_t>(sample) /
            static_cast<std::int64_t>(samples)};
    const auto yaw_value = EvaluateTrajectory(yaw, offset);
    if (!yaw_value.has_value() ||
        !std::isfinite(*yaw_value)) {
      return Issue("INVALID_SPIN_TRAJECTORY",
                   "spin yaw trajectory is not finite");
    }
    auto report = ValidateFootprintAtPose(
        *projection_, envelope_, config_,
        {.position_m = fixed_position, .yaw_rad = *yaw_value});
    if (!report.ok()) {
      return report;
    }
  }
  return {};
}

}  // namespace lunar::planning::v3
