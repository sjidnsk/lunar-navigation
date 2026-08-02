#include "lunar_path_planner/v3/legged/legged_timing.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "lunar_path_planner/v3/legged/legged_spline_optimizer.hpp"

namespace lunar::planning::v3 {
namespace {

constexpr double kGeometryTolerance = 1.0e-10;

[[nodiscard]] Error Invalid(std::string field_path,
                            std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(field_path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool FinitePose(const PoseXyzYaw& pose) noexcept {
  return std::isfinite(pose.position_m.x) &&
         std::isfinite(pose.position_m.y) &&
         std::isfinite(pose.position_m.z) &&
         std::isfinite(pose.yaw_rad);
}

[[nodiscard]] bool ValidVelocityInterval(
    const Interval& interval) noexcept {
  return std::isfinite(interval.lower) &&
         std::isfinite(interval.upper) &&
         interval.lower <= 0.0 && interval.upper >= 0.0 &&
         interval.lower < interval.upper;
}

[[nodiscard]] bool ValidSpline(
    const ClampedCubicBSplinePath& spline) noexcept {
  if (spline.control_points.size() < 4U ||
      spline.knots.size() != spline.control_points.size() + 4U) {
    return false;
  }
  for (const auto& pose : spline.control_points) {
    if (!FinitePose(pose)) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < spline.knots.size();
       ++index) {
    if (!std::isfinite(spline.knots[index]) ||
        (index > 0U &&
         spline.knots[index] < spline.knots[index - 1U])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool ValidPrimitiveChain(
    const ValidatedPrimitiveChain& chain) noexcept {
  if (chain.primitives.empty()) {
    return false;
  }
  for (std::size_t index = 0U; index < chain.primitives.size();
       ++index) {
    const auto& primitive = chain.primitives[index];
    if (!FinitePose(primitive.start_pose) ||
        !FinitePose(primitive.end_pose) ||
        primitive.nominal_duration.value.count() <= 0) {
      return false;
    }
    if (index > 0U) {
      const auto& previous = chain.primitives[index - 1U];
      if (std::abs(previous.end_pose.position_m.x -
                   primitive.start_pose.position_m.x) >
              kGeometryTolerance ||
          std::abs(previous.end_pose.position_m.y -
                   primitive.start_pose.position_m.y) >
              kGeometryTolerance ||
          std::abs(previous.end_pose.position_m.z -
                   primitive.start_pose.position_m.z) >
              kGeometryTolerance ||
          std::abs(previous.end_pose.yaw_rad -
                   primitive.start_pose.yaw_rad) >
              kGeometryTolerance) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] PoseXyzYaw Interpolate(
    const PoseXyzYaw& start, const PoseXyzYaw& finish,
    double alpha) noexcept {
  const auto blend = [alpha](double lhs, double rhs) {
    return lhs + alpha * (rhs - lhs);
  };
  return {
      .position_m =
          {
              blend(start.position_m.x, finish.position_m.x),
              blend(start.position_m.y, finish.position_m.y),
              blend(start.position_m.z, finish.position_m.z),
          },
      .yaw_rad = blend(start.yaw_rad, finish.yaw_rad),
  };
}

[[nodiscard]] std::optional<PoseXyzYaw> EvaluatePath(
    const GeometricPath& path, double parameter) noexcept {
  if (!std::isfinite(parameter) || parameter < 0.0 ||
      parameter > 1.0) {
    return std::nullopt;
  }
  return std::visit(
      [parameter](const auto& concrete)
          -> std::optional<PoseXyzYaw> {
        using Path = std::decay_t<decltype(concrete)>;
        if constexpr (
            std::is_same_v<Path, ClampedCubicBSplinePath>) {
          if (!ValidSpline(concrete)) {
            return std::nullopt;
          }
          return EvaluateLeggedSpline(concrete, parameter);
        } else {
          if (!ValidPrimitiveChain(concrete)) {
            return std::nullopt;
          }
          const double scaled =
              parameter *
              static_cast<double>(concrete.primitives.size());
          const std::size_t index =
              parameter >= 1.0
                  ? concrete.primitives.size() - 1U
                  : std::min(
                        static_cast<std::size_t>(
                            std::floor(scaled)),
                        concrete.primitives.size() - 1U);
          const double alpha =
              parameter >= 1.0
                  ? 1.0
                  : scaled - static_cast<double>(index);
          return Interpolate(
              concrete.primitives[index].start_pose,
              concrete.primitives[index].end_pose, alpha);
        }
      },
      path);
}

struct PathDerivative final {
  double world_x_per_parameter{};
  double world_y_per_parameter{};
  double vertical_per_parameter{};
  double yaw_per_parameter{};
};

[[nodiscard]] std::optional<PathDerivative> DerivativeAt(
    const GeometricPath& path, double parameter,
    double parameter_step) noexcept {
  const double before_parameter =
      std::max(0.0, parameter - parameter_step);
  const double after_parameter =
      std::min(1.0, parameter + parameter_step);
  const double span = after_parameter - before_parameter;
  if (span <= kGeometryTolerance) {
    return std::nullopt;
  }
  const auto before = EvaluatePath(path, before_parameter);
  const auto after = EvaluatePath(path, after_parameter);
  if (!before.has_value() || !after.has_value()) {
    return std::nullopt;
  }
  return PathDerivative{
      .world_x_per_parameter =
          (after->position_m.x - before->position_m.x) / span,
      .world_y_per_parameter =
          (after->position_m.y - before->position_m.y) / span,
      .vertical_per_parameter =
          (after->position_m.z - before->position_m.z) / span,
      .yaw_per_parameter =
          (after->yaw_rad - before->yaw_rad) / span,
  };
}

struct LocalDerivative final {
  double forward_per_parameter{};
  double lateral_per_parameter{};
  double vertical_per_parameter{};
  double yaw_per_parameter{};
  double linear_norm_per_parameter{};
};

[[nodiscard]] LocalDerivative ToBodyFrame(
    const PathDerivative& derivative,
    double yaw_rad) noexcept {
  const double cosine = std::cos(yaw_rad);
  const double sine = std::sin(yaw_rad);
  return {
      .forward_per_parameter =
          cosine * derivative.world_x_per_parameter +
          sine * derivative.world_y_per_parameter,
      .lateral_per_parameter =
          -sine * derivative.world_x_per_parameter +
          cosine * derivative.world_y_per_parameter,
      .vertical_per_parameter =
          derivative.vertical_per_parameter,
      .yaw_per_parameter = derivative.yaw_per_parameter,
      .linear_norm_per_parameter =
          std::hypot(
              std::hypot(derivative.world_x_per_parameter,
                         derivative.world_y_per_parameter),
              derivative.vertical_per_parameter),
  };
}

[[nodiscard]] double DirectionalRateLimit(
    double derivative, const Interval& bounds) noexcept {
  if (derivative > kGeometryTolerance) {
    return bounds.upper / derivative;
  }
  if (derivative < -kGeometryTolerance) {
    return bounds.lower / derivative;
  }
  return std::numeric_limits<double>::infinity();
}

[[nodiscard]] double LocalParameterRateLimit(
    const LocalDerivative& derivative,
    const BodyFrameVelocityEnvelope& envelope) noexcept {
  return std::min(
      {
          DirectionalRateLimit(
              derivative.forward_per_parameter,
              envelope.forward_mps),
          DirectionalRateLimit(
              derivative.lateral_per_parameter,
              envelope.lateral_mps),
          DirectionalRateLimit(
              derivative.vertical_per_parameter,
              envelope.vertical_mps),
          DirectionalRateLimit(
              derivative.yaw_per_parameter,
              envelope.yaw_rate_radps),
      });
}

[[nodiscard]] double LocalParameterAccelerationLimit(
    const LocalDerivative& derivative,
    const LeggedTimingConfig& config) noexcept {
  double limit = std::numeric_limits<double>::infinity();
  if (derivative.linear_norm_per_parameter >
      kGeometryTolerance) {
    limit = std::min(
        limit,
        config.maximum_linear_acceleration_mps2 /
            derivative.linear_norm_per_parameter);
  }
  if (std::abs(derivative.yaw_per_parameter) >
      kGeometryTolerance) {
    limit = std::min(
        limit,
        config.maximum_yaw_acceleration_radps2 /
            std::abs(derivative.yaw_per_parameter));
  }
  return limit;
}

[[nodiscard]] double ComponentRatio(
    double value, const Interval& bounds) noexcept {
  if (value > 0.0) {
    return value / bounds.upper;
  }
  if (value < 0.0) {
    return value / bounds.lower;
  }
  return 0.0;
}

[[nodiscard]] std::optional<std::chrono::nanoseconds>
ToDuration(double seconds,
           std::chrono::nanoseconds offset) noexcept {
  const long double raw =
      static_cast<long double>(seconds) * 1.0e9L;
  const long double remaining =
      static_cast<long double>(
          std::numeric_limits<std::int64_t>::max() -
          offset.count());
  if (!std::isfinite(seconds) || seconds <= 0.0 ||
      raw > remaining) {
    return std::nullopt;
  }
  return std::chrono::nanoseconds{
      std::max<std::int64_t>(
          1LL,
          static_cast<std::int64_t>(std::ceil(raw)))};
}

}  // namespace

Result<LeggedTimingResult> ParameterizeLeggedBodyPath(
    const GeometricPath& path,
    const BodyFrameVelocityEnvelope& velocity_envelope,
    const LeggedTimingConfig& config) {
  if (config.sampling.maximum_adaptive_samples < 3U) {
    return Error{
        .code = ErrorCode::kResourceLimit,
        .field_path =
            "/algorithm_config/legged/time_scaling/"
            "maximum_adaptive_samples",
        .message = "LEGGED_TIMING_SAMPLE_LIMIT",
    };
  }
  if (!ValidVelocityInterval(velocity_envelope.forward_mps) ||
      !ValidVelocityInterval(velocity_envelope.lateral_mps) ||
      !ValidVelocityInterval(velocity_envelope.vertical_mps) ||
      !ValidVelocityInterval(
          velocity_envelope.yaw_rate_radps) ||
      !std::isfinite(config.sampling.minimum_parameter_step) ||
      config.sampling.minimum_parameter_step <= 0.0 ||
      config.sampling.minimum_parameter_step > 1.0 ||
      config.sampling.maximum_forward_passes == 0U ||
      config.sampling.maximum_backward_passes == 0U ||
      !std::isfinite(
          config.maximum_linear_acceleration_mps2) ||
      config.maximum_linear_acceleration_mps2 <= 0.0 ||
      !std::isfinite(
          config.maximum_yaw_acceleration_radps2) ||
      config.maximum_yaw_acceleration_radps2 <= 0.0 ||
      (config.sampling.enable_jerk_smoothing &&
       config.sampling.maximum_jerk_smoothing_iterations ==
           0U)) {
    return Invalid("/legged_timing",
                   "INVALID_LEGGED_TIMING_CONFIGURATION");
  }

  const std::size_t sample_count =
      config.sampling.maximum_adaptive_samples;
  const double parameter_step =
      1.0 / static_cast<double>(sample_count - 1U);
  const double derivative_step =
      std::max(
          config.sampling.minimum_parameter_step,
          std::min(0.25 * parameter_step, 1.0e-3));

  std::vector<LocalDerivative> derivatives;
  std::vector<double> rate_limits;
  std::vector<double> acceleration_limits;
  derivatives.reserve(sample_count);
  rate_limits.reserve(sample_count);
  acceleration_limits.reserve(sample_count);
  bool has_motion = false;
  for (std::size_t index = 0U; index < sample_count;
       ++index) {
    const double parameter =
        static_cast<double>(index) * parameter_step;
    const auto pose = EvaluatePath(path, parameter);
    const auto derivative =
        DerivativeAt(path, parameter, derivative_step);
    if (!pose.has_value() || !FinitePose(*pose) ||
        !derivative.has_value()) {
      return Invalid("/geometric_path",
                     "INVALID_LEGGED_GEOMETRIC_PATH");
    }
    const auto local = ToBodyFrame(*derivative, pose->yaw_rad);
    const double rate_limit =
        LocalParameterRateLimit(local, velocity_envelope);
    const double acceleration_limit =
        LocalParameterAccelerationLimit(local, config);
    if (std::isnan(rate_limit) || rate_limit <= 0.0 ||
        std::isnan(acceleration_limit) ||
        acceleration_limit <= 0.0) {
      return Invalid("/legged_timing/local_limit",
                     "INVALID_LOCAL_BODY_RATE_LIMIT");
    }
    has_motion =
        has_motion ||
        local.linear_norm_per_parameter >
            kGeometryTolerance ||
        std::abs(local.yaw_per_parameter) >
            kGeometryTolerance;
    derivatives.push_back(local);
    rate_limits.push_back(rate_limit);
    acceleration_limits.push_back(acceleration_limit);
  }
  if (!has_motion) {
    return Invalid("/geometric_path",
                   "DEGENERATE_LEGGED_GEOMETRIC_PATH");
  }

  std::vector<double> parameter_rate = rate_limits;
  parameter_rate.front() = 0.0;
  for (std::size_t pass = 0U;
       pass < config.sampling.maximum_forward_passes; ++pass) {
    for (std::size_t index = 1U; index < sample_count;
         ++index) {
      const double acceleration =
          std::min(acceleration_limits[index - 1U],
                   acceleration_limits[index]);
      parameter_rate[index] =
          std::min(
              parameter_rate[index],
              std::sqrt(
                  parameter_rate[index - 1U] *
                          parameter_rate[index - 1U] +
                      2.0 * acceleration * parameter_step));
    }
  }
  parameter_rate.back() = 0.0;
  for (std::size_t pass = 0U;
       pass < config.sampling.maximum_backward_passes;
       ++pass) {
    for (std::size_t index = sample_count - 1U;
         index > 0U; --index) {
      const double acceleration =
          std::min(acceleration_limits[index - 1U],
                   acceleration_limits[index]);
      parameter_rate[index - 1U] =
          std::min(
              parameter_rate[index - 1U],
              std::sqrt(
                  parameter_rate[index] *
                          parameter_rate[index] +
                      2.0 * acceleration * parameter_step));
    }
  }

  MonotoneTimeScaling scaling;
  scaling.segments.reserve(sample_count - 1U);
  std::chrono::nanoseconds offset{};
  for (std::size_t index = 1U; index < sample_count;
       ++index) {
    const double rate_sum =
        parameter_rate[index - 1U] + parameter_rate[index];
    if (!std::isfinite(rate_sum) ||
        rate_sum <= kGeometryTolerance) {
      return Error{
          .code = ErrorCode::kNumericalFailure,
          .field_path = "/legged_timing",
          .message =
              "UNREACHABLE_ZERO_PARAMETER_RATE_INTERVAL",
      };
    }
    const auto duration =
        ToDuration(2.0 * parameter_step / rate_sum, offset);
    if (!duration.has_value()) {
      return Error{
          .code = ErrorCode::kNumericalFailure,
          .field_path = "/legged_timing",
          .message = "LEGGED_TIMING_DURATION_OVERFLOW",
      };
    }
    const double seconds =
        std::chrono::duration<double>(*duration).count();
    const double maximum_monotone_rate =
        3.0 * parameter_step / seconds;
    const double start_rate =
        std::clamp(parameter_rate[index - 1U], 0.0,
                   maximum_monotone_rate);
    const double end_rate =
        std::clamp(parameter_rate[index], 0.0,
                   maximum_monotone_rate);
    const double c2 =
        3.0 * parameter_step / (seconds * seconds) -
        (2.0 * start_rate + end_rate) / seconds;
    const double c3 =
        -2.0 * parameter_step /
            (seconds * seconds * seconds) +
        (start_rate + end_rate) / (seconds * seconds);
    scaling.segments.push_back(
        {
            .start_offset = DurationNanoseconds{offset},
            .end_offset =
                DurationNanoseconds{offset + *duration},
            .coefficients =
                {
                    static_cast<double>(index - 1U) *
                        parameter_step,
                    start_rate,
                    c2,
                    c3,
                },
        });
    offset += *duration;
  }

  double maximum_body_ratio = 0.0;
  double maximum_yaw_ratio = 0.0;
  for (std::size_t index = 0U; index < sample_count;
       ++index) {
    const auto& derivative = derivatives[index];
    const double rate = parameter_rate[index];
    maximum_body_ratio =
        std::max(
            maximum_body_ratio,
            std::max(
                {
                    ComponentRatio(
                        derivative.forward_per_parameter * rate,
                        velocity_envelope.forward_mps),
                    ComponentRatio(
                        derivative.lateral_per_parameter * rate,
                        velocity_envelope.lateral_mps),
                    ComponentRatio(
                        derivative.vertical_per_parameter * rate,
                        velocity_envelope.vertical_mps),
                }));
    maximum_yaw_ratio =
        std::max(
            maximum_yaw_ratio,
            ComponentRatio(
                derivative.yaw_per_parameter * rate,
                velocity_envelope.yaw_rate_radps));
  }

  return LeggedTimingResult{
      .time_scaling = std::move(scaling),
      .duration = DurationNanoseconds{offset},
      .diagnostics =
          {
              .sample_count = sample_count,
              .forward_passes =
                  config.sampling.maximum_forward_passes,
              .backward_passes =
                  config.sampling.maximum_backward_passes,
              .maximum_body_rate_ratio =
                  maximum_body_ratio,
              .maximum_yaw_rate_ratio =
                  maximum_yaw_ratio,
              .ends_stopped =
                  parameter_rate.front() == 0.0 &&
                  parameter_rate.back() == 0.0,
              .termination_reason =
                  "BOUNDED_BODY_RATE_REACHABILITY_COMPLETE",
          },
  };
}

}  // namespace lunar::planning::v3
