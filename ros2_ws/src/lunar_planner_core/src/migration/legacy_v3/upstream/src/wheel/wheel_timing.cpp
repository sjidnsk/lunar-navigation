#include "lunar_path_planner/v3/wheel/wheel_timing.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace lunar::planning::v3 {
namespace {

constexpr double kGeometryTolerance = 1.0e-9;

[[nodiscard]] Error Invalid(std::string field_path,
                            std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] Error Resource(std::string message) {
  return {
      .code = ErrorCode::kResourceLimit,
      .field_path =
          "/algorithm_config/wheeled/time_scaling/"
          "maximum_adaptive_samples",
      .message = std::move(message),
  };
}

[[nodiscard]] bool FinitePositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool FinitePose(const PoseXyzYaw& pose) noexcept {
  return std::isfinite(pose.position_m.x) &&
         std::isfinite(pose.position_m.y) &&
         std::isfinite(pose.position_m.z) &&
         std::isfinite(pose.yaw_rad);
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

[[nodiscard]] std::optional<PoseXyzYaw>
EvaluatePrimitiveChain(
    const ValidatedPrimitiveChain& chain,
    double parameter) {
  if (chain.primitives.empty() || parameter < 0.0 ||
      parameter > 1.0) {
    return std::nullopt;
  }
  std::vector<double> lengths;
  lengths.reserve(chain.primitives.size());
  double total_length = 0.0;
  for (const auto& primitive : chain.primitives) {
    if (!FinitePose(primitive.start_pose) ||
        !FinitePose(primitive.end_pose)) {
      return std::nullopt;
    }
    const double length =
        std::hypot(primitive.end_pose.position_m.x -
                       primitive.start_pose.position_m.x,
                   primitive.end_pose.position_m.y -
                       primitive.start_pose.position_m.y);
    lengths.push_back(length);
    total_length += length;
  }
  if (!FinitePositive(total_length)) {
    return std::nullopt;
  }
  const double target = parameter * total_length;
  double traversed = 0.0;
  for (std::size_t index = 0U; index < chain.primitives.size();
       ++index) {
    if (lengths[index] <= kGeometryTolerance) {
      continue;
    }
    if (target <= traversed + lengths[index] +
                      kGeometryTolerance ||
        index + 1U == chain.primitives.size()) {
      const auto& primitive = chain.primitives[index];
      const double fraction = std::clamp(
          (target - traversed) / lengths[index], 0.0, 1.0);
      return PoseXyzYaw{
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
    }
    traversed += lengths[index];
  }
  return chain.primitives.back().end_pose;
}

[[nodiscard]] std::optional<PoseXyzYaw> EvaluatePath(
    const GeometricPath& path,
    double parameter) {
  return std::visit(
      [parameter](const auto& concrete)
          -> std::optional<PoseXyzYaw> {
        using Path = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<
                          Path,
                          ClampedCubicBSplinePath>) {
          return EvaluateSpline(concrete, parameter);
        } else {
          return EvaluatePrimitiveChain(concrete, parameter);
        }
      },
      path);
}

[[nodiscard]] double Curvature(
    const PoseXyzYaw& first,
    const PoseXyzYaw& middle,
    const PoseXyzYaw& last) noexcept {
  const double ab =
      std::hypot(middle.position_m.x - first.position_m.x,
                 middle.position_m.y - first.position_m.y);
  const double bc =
      std::hypot(last.position_m.x - middle.position_m.x,
                 last.position_m.y - middle.position_m.y);
  const double ac =
      std::hypot(last.position_m.x - first.position_m.x,
                 last.position_m.y - first.position_m.y);
  const double denominator = ab * bc * ac;
  if (denominator <= kGeometryTolerance) {
    return 0.0;
  }
  const double twice_area =
      std::abs(
          (middle.position_m.x - first.position_m.x) *
              (last.position_m.y - first.position_m.y) -
          (middle.position_m.y - first.position_m.y) *
              (last.position_m.x - first.position_m.x));
  return 2.0 * twice_area / denominator;
}

struct SampledPath final {
  std::vector<PoseXyzYaw> poses;
  std::vector<double> segment_lengths_m;
  std::vector<double> curvature_per_m;
  double total_length_m{};
};

[[nodiscard]] Result<SampledPath> SamplePath(
    const GeometricPath& path,
    const WheelTimingConfig& config) {
  if (!FinitePositive(config.minimum_parameter_step) ||
      config.minimum_parameter_step > 0.5 ||
      !FinitePositive(
          config.curvature_refinement_threshold_per_m)) {
    return Invalid("/wheel_timing/config",
                   "INVALID_WHEEL_TIMING_CONFIG");
  }
  std::size_t sample_count =
      static_cast<std::size_t>(
          std::ceil(1.0 / config.minimum_parameter_step)) +
      1U;
  sample_count = std::max<std::size_t>(3U, sample_count);
  if (sample_count > config.maximum_adaptive_samples) {
    return Resource("TIMING_SAMPLE_LIMIT");
  }

  while (true) {
    SampledPath sampled;
    sampled.poses.reserve(sample_count);
    for (std::size_t index = 0U; index < sample_count;
         ++index) {
      const double parameter =
          static_cast<double>(index) /
          static_cast<double>(sample_count - 1U);
      const auto pose = EvaluatePath(path, parameter);
      if (!pose.has_value() || !FinitePose(*pose)) {
        return Invalid("/geometric_path",
                       "INVALID_WHEEL_GEOMETRIC_PATH");
      }
      sampled.poses.push_back(*pose);
    }
    sampled.segment_lengths_m.reserve(sample_count - 1U);
    for (std::size_t index = 1U; index < sample_count;
         ++index) {
      const double distance =
          std::hypot(
              sampled.poses[index].position_m.x -
                  sampled.poses[index - 1U].position_m.x,
              sampled.poses[index].position_m.y -
                  sampled.poses[index - 1U].position_m.y);
      if (!std::isfinite(distance) ||
          distance <= kGeometryTolerance) {
        return Invalid("/geometric_path",
                       "DEGENERATE_WHEEL_GEOMETRIC_PATH");
      }
      sampled.segment_lengths_m.push_back(distance);
      sampled.total_length_m += distance;
    }
    sampled.curvature_per_m.assign(sample_count, 0.0);
    for (std::size_t index = 1U;
         index + 1U < sample_count; ++index) {
      sampled.curvature_per_m[index] =
          Curvature(sampled.poses[index - 1U],
                    sampled.poses[index],
                    sampled.poses[index + 1U]);
    }
    sampled.curvature_per_m.front() =
        sampled.curvature_per_m[1U];
    sampled.curvature_per_m.back() =
        sampled.curvature_per_m[sample_count - 2U];

    bool refine = false;
    for (std::size_t index = 1U;
         index < sampled.curvature_per_m.size(); ++index) {
      if (std::abs(sampled.curvature_per_m[index] -
                   sampled.curvature_per_m[index - 1U]) >
          config.curvature_refinement_threshold_per_m) {
        refine = true;
        break;
      }
    }
    if (!refine) {
      return sampled;
    }
    if (sample_count >
        (config.maximum_adaptive_samples + 1U) / 2U) {
      return Resource("TIMING_SAMPLE_LIMIT");
    }
    sample_count = 2U * sample_count - 1U;
  }
}

[[nodiscard]] double LocalSpeedLimit(
    double curvature,
    DriveDirection direction,
    const WheelTimingLimits& limits) noexcept {
  double value =
      direction == DriveDirection::kForward
          ? limits.maximum_forward_speed_mps
          : limits.maximum_reverse_speed_mps;
  value = std::min(value, limits.terrain_speed_limit_mps);
  value = std::min(value, limits.clearance_speed_limit_mps);
  value = std::min(value, limits.traction_speed_limit_mps);
  if (curvature > kGeometryTolerance) {
    value = std::min(
        value, limits.maximum_yaw_rate_radps / curvature);
    value = std::min(
        value,
        std::sqrt(limits.maximum_lateral_acceleration_mps2 /
                  curvature));
  }
  return value;
}

[[nodiscard]] std::optional<double> EvaluateSegment(
    const CubicPolynomialSegment& segment,
    DurationNanoseconds offset,
    bool derivative) noexcept {
  if (offset.value < segment.start_offset.value ||
      offset.value > segment.end_offset.value) {
    return std::nullopt;
  }
  const double seconds =
      std::chrono::duration<double>(
          offset.value - segment.start_offset.value)
          .count();
  const auto& c = segment.coefficients;
  if (derivative) {
    return c[1] + 2.0 * c[2] * seconds +
           3.0 * c[3] * seconds * seconds;
  }
  return ((c[3] * seconds + c[2]) * seconds + c[1]) *
             seconds +
         c[0];
}

template <class Trajectory>
[[nodiscard]] std::optional<double> Evaluate(
    const Trajectory& trajectory,
    DurationNanoseconds offset,
    bool derivative) noexcept {
  for (std::size_t index = 0U;
       index < trajectory.segments.size(); ++index) {
    const auto& segment = trajectory.segments[index];
    if (offset.value >= segment.start_offset.value &&
        offset.value <= segment.end_offset.value) {
      if (offset.value == segment.end_offset.value &&
          index + 1U < trajectory.segments.size()) {
        continue;
      }
      return EvaluateSegment(segment, offset, derivative);
    }
  }
  return std::nullopt;
}

}  // namespace

Result<WheelTimingResult> ParameterizeWheelDrive(
    const GeometricPath& path,
    DriveDirection direction,
    const WheelTimingLimits& limits,
    const WheelTimingConfig& config) {
  if (!FinitePositive(limits.maximum_forward_speed_mps) ||
      !FinitePositive(limits.maximum_reverse_speed_mps) ||
      !FinitePositive(
          limits.maximum_linear_acceleration_mps2) ||
      !FinitePositive(
          limits.maximum_braking_deceleration_mps2) ||
      !FinitePositive(limits.maximum_yaw_rate_radps) ||
      !FinitePositive(
          limits.maximum_lateral_acceleration_mps2) ||
      !FinitePositive(limits.terrain_speed_limit_mps) ||
      !FinitePositive(limits.clearance_speed_limit_mps) ||
      !FinitePositive(limits.traction_speed_limit_mps) ||
      config.maximum_forward_passes == 0U ||
      config.maximum_backward_passes == 0U) {
    return Invalid("/wheel_timing/limits",
                   "INVALID_WHEEL_TIMING_LIMIT");
  }
  const auto sampled_result = SamplePath(path, config);
  if (!IsOk(sampled_result)) {
    return std::get<Error>(sampled_result);
  }
  const auto& sampled = std::get<SampledPath>(sampled_result);
  const std::size_t count = sampled.poses.size();
  std::vector<double> speed(count, 0.0);
  for (std::size_t index = 0U; index < count; ++index) {
    speed[index] = LocalSpeedLimit(
        sampled.curvature_per_m[index], direction, limits);
    if (!FinitePositive(speed[index])) {
      return Invalid("/wheel_timing/local_speed_limit",
                     "INVALID_LOCAL_SPEED_LIMIT");
    }
  }
  speed.front() = 0.0;
  for (std::size_t pass = 0U;
       pass < config.maximum_forward_passes; ++pass) {
    for (std::size_t index = 1U; index < count; ++index) {
      speed[index] = std::min(
          speed[index],
          std::sqrt(speed[index - 1U] * speed[index - 1U] +
                    2.0 *
                        limits.maximum_linear_acceleration_mps2 *
                        sampled.segment_lengths_m[index - 1U]));
    }
  }
  speed.back() = 0.0;
  for (std::size_t pass = 0U;
       pass < config.maximum_backward_passes; ++pass) {
    for (std::size_t index = count - 1U; index > 0U;
         --index) {
      speed[index - 1U] = std::min(
          speed[index - 1U],
          std::sqrt(speed[index] * speed[index] +
                    2.0 *
                        limits.maximum_braking_deceleration_mps2 *
                        sampled.segment_lengths_m[index - 1U]));
    }
  }

  MonotoneTimeScaling scaling;
  scaling.segments.reserve(count - 1U);
  std::chrono::nanoseconds offset{};
  const double parameter_step =
      1.0 / static_cast<double>(count - 1U);
  for (std::size_t index = 1U; index < count; ++index) {
    const double speed_sum = speed[index - 1U] + speed[index];
    if (!FinitePositive(speed_sum)) {
      return Error{
          .code = ErrorCode::kNumericalFailure,
          .field_path = "/wheel_timing",
          .message = "UNREACHABLE_ZERO_SPEED_INTERVAL",
      };
    }
    const double duration_seconds =
        2.0 * sampled.segment_lengths_m[index - 1U] /
        speed_sum;
    const long double raw_nanoseconds =
        static_cast<long double>(duration_seconds) * 1.0e9L;
    if (!std::isfinite(duration_seconds) ||
        duration_seconds <= 0.0 ||
        raw_nanoseconds >
            static_cast<long double>(
                std::numeric_limits<std::int64_t>::max() -
                offset.count())) {
      return Error{
          .code = ErrorCode::kNumericalFailure,
          .field_path = "/wheel_timing",
          .message = "TIMING_DURATION_OVERFLOW",
      };
    }
    const auto duration = std::chrono::nanoseconds{
        std::max<std::int64_t>(
            1LL,
            static_cast<std::int64_t>(
                std::ceil(raw_nanoseconds)))};
    const double actual_seconds =
        std::chrono::duration<double>(duration).count();
    const double metric =
        sampled.segment_lengths_m[index - 1U] /
        parameter_step;
    const double maximum_monotone_slope =
        3.0 * parameter_step / actual_seconds;
    const double start_rate = std::clamp(
        speed[index - 1U] / metric, 0.0,
        maximum_monotone_slope);
    const double finish_rate = std::clamp(
        speed[index] / metric, 0.0,
        maximum_monotone_slope);
    const double start_parameter =
        static_cast<double>(index - 1U) * parameter_step;
    const double c2 =
        3.0 * parameter_step /
            (actual_seconds * actual_seconds) -
        (2.0 * start_rate + finish_rate) / actual_seconds;
    const double c3 =
        -2.0 * parameter_step /
            (actual_seconds * actual_seconds * actual_seconds) +
        (start_rate + finish_rate) /
            (actual_seconds * actual_seconds);
    scaling.segments.push_back(
        {
            .start_offset = DurationNanoseconds{offset},
            .end_offset =
                DurationNanoseconds{offset + duration},
            .coefficients =
                {start_parameter, start_rate, c2, c3},
        });
    offset += duration;
  }
  return WheelTimingResult{
      .time_scaling = std::move(scaling),
      .duration = DurationNanoseconds{offset},
      .path_length_m = sampled.total_length_m,
      .diagnostics =
          {
              .sample_count = count,
              .forward_passes = config.maximum_forward_passes,
              .backward_passes = config.maximum_backward_passes,
              .termination_reason = "BOUNDED_REACHABILITY_COMPLETE",
          },
  };
}

Result<PiecewiseCubicScalarTrajectory> ParameterizeWheelSpin(
    double yaw_start_rad,
    double yaw_end_unwrapped_rad,
    const SpinTimingLimits& limits) {
  if (!std::isfinite(yaw_start_rad) ||
      !std::isfinite(yaw_end_unwrapped_rad) ||
      !FinitePositive(limits.maximum_yaw_rate_radps) ||
      !FinitePositive(
          limits.maximum_yaw_acceleration_radps2)) {
    return Invalid("/wheel_spin", "INVALID_SPIN_TIMING_INPUT");
  }
  const double delta = yaw_end_unwrapped_rad - yaw_start_rad;
  const double absolute_delta = std::abs(delta);
  if (absolute_delta <= kGeometryTolerance) {
    return Invalid("/wheel_spin",
                   "ZERO_ANGLE_SPIN_IS_NOT_A_SEGMENT");
  }
  const double duration_seconds = std::max(
      1.5 * absolute_delta / limits.maximum_yaw_rate_radps,
      std::sqrt(
          6.0 * absolute_delta /
          limits.maximum_yaw_acceleration_radps2));
  const long double raw_nanoseconds =
      static_cast<long double>(duration_seconds) * 1.0e9L;
  if (!std::isfinite(duration_seconds) ||
      raw_nanoseconds >=
          static_cast<long double>(
              std::numeric_limits<std::int64_t>::max())) {
    return Error{
        .code = ErrorCode::kNumericalFailure,
        .field_path = "/wheel_spin",
        .message = "SPIN_DURATION_OVERFLOW",
    };
  }
  const auto duration = std::chrono::nanoseconds{
      std::max<std::int64_t>(
          1LL,
          static_cast<std::int64_t>(std::ceil(raw_nanoseconds)))};
  const double seconds =
      std::chrono::duration<double>(duration).count();
  return PiecewiseCubicScalarTrajectory{
      .value_semantics = "unwrapped_yaw_rad",
      .segments =
          {
              {
                  .start_offset = DurationNanoseconds{},
                  .end_offset = DurationNanoseconds{duration},
                  .coefficients =
                      {
                          yaw_start_rad,
                          0.0,
                          3.0 * delta / (seconds * seconds),
                          -2.0 * delta /
                              (seconds * seconds * seconds),
                      },
              },
          },
  };
}

std::optional<double> EvaluateMonotoneTimeScaling(
    const MonotoneTimeScaling& scaling,
    DurationNanoseconds offset) noexcept {
  return Evaluate(scaling, offset, false);
}

std::optional<double> EvaluateMonotoneTimeScalingDerivative(
    const MonotoneTimeScaling& scaling,
    DurationNanoseconds offset) noexcept {
  return Evaluate(scaling, offset, true);
}

std::optional<double> EvaluateScalarTrajectory(
    const PiecewiseCubicScalarTrajectory& trajectory,
    DurationNanoseconds offset) noexcept {
  return Evaluate(trajectory, offset, false);
}

std::optional<double> EvaluateScalarTrajectoryDerivative(
    const PiecewiseCubicScalarTrajectory& trajectory,
    DurationNanoseconds offset) noexcept {
  return Evaluate(trajectory, offset, true);
}

double SignedBodyForwardSpeed(
    double path_parameter_rate_per_s,
    double path_length_m,
    DriveDirection direction) noexcept {
  const double magnitude =
      path_parameter_rate_per_s * path_length_m;
  return direction == DriveDirection::kReverse ? -magnitude
                                               : magnitude;
}

}  // namespace lunar::planning::v3
