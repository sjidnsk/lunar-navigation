#include "hopper/ballistic_envelope.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stop_token>
#include <utility>
#include <vector>

#include "hopper/ballistic_kinematics.hpp"

namespace lunar::planning::hopper {
namespace {

constexpr long double kCoefficientTolerance = 1.0e-15L;
constexpr std::size_t kMaximumRootIterations = 192U;

struct Interval final {
  long double lower{};
  long double upper{};
};

struct RootIsolationResult final {
  std::vector<long double> roots;
  bool canceled{};
  bool indeterminate{};
};

struct ConstraintResult final {
  std::vector<Interval> intervals;
  std::size_t examined_intervals{};
  bool canceled{};
  bool indeterminate{};
};

[[nodiscard]] bool IsFinite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] long double Dot(const Vec3 left, const Vec3 right) noexcept {
  return static_cast<long double>(left.x) * right.x +
         static_cast<long double>(left.y) * right.y +
         static_cast<long double>(left.z) * right.z;
}

[[nodiscard]] long double SquaredNorm(const Vec3 value) noexcept {
  return Dot(value, value);
}

[[nodiscard]] double Norm(const Vec3 value) noexcept {
  return std::hypot(std::hypot(value.x, value.y), value.z);
}

[[nodiscard]] BallisticEnvelopeResult
Failure(const BallisticEnvelopeStatus status, std::string reason,
        const std::size_t examined_intervals = 0U) {
  return BallisticEnvelopeResult{
      .status = status,
      .arc = std::nullopt,
      .examined_intervals = examined_intervals,
      .reason_code = std::move(reason),
  };
}

[[nodiscard]] long double TimeTolerance(const long double left,
                                        const long double right) noexcept {
  return 64.0L * std::numeric_limits<long double>::epsilon() *
         (1.0L + std::max(std::abs(left), std::abs(right)));
}

void SortUnique(std::vector<long double> &values) {
  std::ranges::sort(values);
  std::vector<long double> unique;
  unique.reserve(values.size());
  for (const long double value : values) {
    if (unique.empty() ||
        std::abs(value - unique.back()) > TimeTolerance(value, unique.back())) {
      unique.push_back(value);
    }
  }
  values = std::move(unique);
}

void TrimPolynomial(std::vector<long double> &coefficients) {
  while (coefficients.size() > 1U && coefficients.back() == 0.0L) {
    coefficients.pop_back();
  }
}

[[nodiscard]] long double
EvaluatePolynomial(const std::vector<long double> &coefficients,
                   const long double value) noexcept {
  long double result = 0.0L;
  for (auto iterator = coefficients.rbegin(); iterator != coefficients.rend();
       ++iterator) {
    result = result * value + *iterator;
  }
  return result;
}

[[nodiscard]] long double
PolynomialScale(const std::vector<long double> &coefficients,
                const long double value) noexcept {
  long double scale = 0.0L;
  long double power = 1.0L;
  for (const long double coefficient : coefficients) {
    scale += std::abs(coefficient) * power;
    power *= std::abs(value);
  }
  return std::max(1.0L, scale);
}

[[nodiscard]] int PolynomialSign(const std::vector<long double> &coefficients,
                                 const long double value) noexcept {
  const long double evaluated = EvaluatePolynomial(coefficients, value);
  if (!std::isfinite(evaluated)) {
    return 2;
  }
  const long double tolerance =
      kCoefficientTolerance * PolynomialScale(coefficients, value);
  if (std::abs(evaluated) <= tolerance) {
    return 0;
  }
  return evaluated < 0.0L ? -1 : 1;
}

[[nodiscard]] std::vector<long double>
Derivative(const std::vector<long double> &coefficients) {
  std::vector<long double> derivative;
  derivative.reserve(coefficients.size() - 1U);
  for (std::size_t degree = 1U; degree < coefficients.size(); ++degree) {
    derivative.push_back(coefficients[degree] *
                         static_cast<long double>(degree));
  }
  TrimPolynomial(derivative);
  return derivative;
}

[[nodiscard]] std::optional<long double>
BisectRoot(const std::vector<long double> &coefficients, long double lower,
           long double upper, const std::stop_token stop_token,
           bool &canceled) noexcept {
  int lower_sign = PolynomialSign(coefficients, lower);
  int upper_sign = PolynomialSign(coefficients, upper);
  if (lower_sign == 2 || upper_sign == 2 || lower_sign == upper_sign) {
    return std::nullopt;
  }
  if (lower_sign == 0) {
    return lower;
  }
  if (upper_sign == 0) {
    return upper;
  }
  for (std::size_t iteration = 0U; iteration < kMaximumRootIterations;
       ++iteration) {
    if (stop_token.stop_requested()) {
      canceled = true;
      return std::nullopt;
    }
    const long double midpoint = lower + (upper - lower) / 2.0L;
    const int midpoint_sign = PolynomialSign(coefficients, midpoint);
    if (midpoint_sign == 2) {
      return std::nullopt;
    }
    if (midpoint_sign == 0 || upper - lower <= TimeTolerance(lower, upper)) {
      return midpoint;
    }
    if (midpoint_sign == lower_sign) {
      lower = midpoint;
      lower_sign = midpoint_sign;
    } else {
      upper = midpoint;
      upper_sign = midpoint_sign;
    }
  }
  return std::nullopt;
}

[[nodiscard]] RootIsolationResult
IsolateRoots(std::vector<long double> coefficients, const long double lower,
             const long double upper, const std::stop_token stop_token) {
  RootIsolationResult result;
  if (stop_token.stop_requested()) {
    result.canceled = true;
    return result;
  }
  TrimPolynomial(coefficients);
  const std::size_t degree = coefficients.size() - 1U;
  if (degree == 0U) {
    return result;
  }
  if (degree == 1U) {
    const long double root = -coefficients[0] / coefficients[1];
    if (!std::isfinite(root)) {
      result.indeterminate = true;
      return result;
    }
    if (root + TimeTolerance(root, lower) >= lower &&
        root <= upper + TimeTolerance(root, upper)) {
      result.roots.push_back(std::clamp(root, lower, upper));
    }
    return result;
  }

  RootIsolationResult critical =
      IsolateRoots(Derivative(coefficients), lower, upper, stop_token);
  if (critical.canceled || critical.indeterminate) {
    return critical;
  }
  std::vector<long double> partitions{lower};
  partitions.insert(partitions.end(), critical.roots.begin(),
                    critical.roots.end());
  partitions.push_back(upper);
  SortUnique(partitions);

  for (const long double point : partitions) {
    const int sign = PolynomialSign(coefficients, point);
    if (sign == 2) {
      result.indeterminate = true;
      return result;
    }
    if (sign == 0) {
      result.roots.push_back(point);
    }
  }
  for (std::size_t index = 1U; index < partitions.size(); ++index) {
    const long double left = partitions[index - 1U];
    const long double right = partitions[index];
    const int left_sign = PolynomialSign(coefficients, left);
    const int right_sign = PolynomialSign(coefficients, right);
    if (left_sign == 2 || right_sign == 2) {
      result.indeterminate = true;
      return result;
    }
    if (left_sign == 0 || right_sign == 0 || left_sign == right_sign) {
      continue;
    }
    bool canceled = false;
    const std::optional<long double> root =
        BisectRoot(coefficients, left, right, stop_token, canceled);
    if (canceled) {
      result.canceled = true;
      return result;
    }
    if (!root.has_value()) {
      result.indeterminate = true;
      return result;
    }
    result.roots.push_back(*root);
  }
  SortUnique(result.roots);
  return result;
}

[[nodiscard]] std::vector<Interval>
MergeIntervals(std::vector<Interval> intervals) {
  std::ranges::sort(intervals, {}, &Interval::lower);
  std::vector<Interval> merged;
  for (const Interval interval : intervals) {
    if (interval.upper + TimeTolerance(interval.lower, interval.upper) <
        interval.lower) {
      continue;
    }
    if (merged.empty() ||
        interval.lower >
            merged.back().upper +
                TimeTolerance(interval.lower, merged.back().upper)) {
      merged.push_back(interval);
      continue;
    }
    merged.back().upper = std::max(merged.back().upper, interval.upper);
  }
  return merged;
}

[[nodiscard]] std::vector<Interval> Intersect(const std::vector<Interval> &left,
                                              const Interval right) {
  std::vector<Interval> result;
  result.reserve(left.size());
  for (const Interval interval : left) {
    const long double lower = std::max(interval.lower, right.lower);
    const long double upper = std::min(interval.upper, right.upper);
    if (lower <= upper + TimeTolerance(lower, upper)) {
      result.push_back(Interval{.lower = lower, .upper = upper});
    }
  }
  return MergeIntervals(std::move(result));
}

[[nodiscard]] std::optional<Interval> SpeedInterval(
    const long double displacement_squared,
    const long double displacement_dot_g, const long double gravity_squared,
    const long double speed_limit_squared, const bool landing) noexcept {
  const long double quadratic = 0.25L * gravity_squared;
  const long double linear =
      (landing ? displacement_dot_g : -displacement_dot_g) -
      speed_limit_squared;
  const long double constant = displacement_squared;
  const long double discriminant =
      linear * linear - 4.0L * quadratic * constant;
  const long double scale =
      std::max(1.0L, linear * linear + 4.0L * std::abs(quadratic * constant));
  if (!std::isfinite(discriminant) ||
      discriminant < -kCoefficientTolerance * scale) {
    return std::nullopt;
  }
  const long double root_term = std::sqrt(std::max(0.0L, discriminant));
  long double lower_u = (-linear - root_term) / (2.0L * quadratic);
  long double upper_u = (-linear + root_term) / (2.0L * quadratic);
  if (!std::isfinite(lower_u) || !std::isfinite(upper_u)) {
    return std::nullopt;
  }
  if (lower_u > upper_u) {
    std::swap(lower_u, upper_u);
  }
  if (upper_u < 0.0L) {
    return std::nullopt;
  }
  lower_u = std::max(0.0L, lower_u);
  return Interval{
      .lower = std::sqrt(lower_u),
      .upper = std::sqrt(std::max(lower_u, upper_u)),
  };
}

[[nodiscard]] ConstraintResult
ApplyPolynomialConstraint(const std::vector<Interval> &input,
                          const std::vector<long double> &coefficients,
                          const std::stop_token stop_token) {
  ConstraintResult result;
  for (const Interval interval : input) {
    RootIsolationResult roots =
        IsolateRoots(coefficients, interval.lower, interval.upper, stop_token);
    if (roots.canceled) {
      result.canceled = true;
      return result;
    }
    if (roots.indeterminate) {
      result.indeterminate = true;
      return result;
    }
    std::vector<long double> partitions{interval.lower};
    partitions.insert(partitions.end(), roots.roots.begin(), roots.roots.end());
    partitions.push_back(interval.upper);
    SortUnique(partitions);

    for (const long double root : roots.roots) {
      result.intervals.push_back(Interval{.lower = root, .upper = root});
    }
    for (const long double endpoint :
         std::array<long double, 2U>{interval.lower, interval.upper}) {
      ++result.examined_intervals;
      const int sign = PolynomialSign(coefficients, endpoint);
      if (sign == 2) {
        result.indeterminate = true;
        return result;
      }
      if (sign <= 0) {
        result.intervals.push_back(
            Interval{.lower = endpoint, .upper = endpoint});
      }
    }
    for (std::size_t index = 1U; index < partitions.size(); ++index) {
      const long double left = partitions[index - 1U];
      const long double right = partitions[index];
      if (right - left <= TimeTolerance(left, right)) {
        continue;
      }
      const long double midpoint = left + (right - left) / 2.0L;
      ++result.examined_intervals;
      const int sign = PolynomialSign(coefficients, midpoint);
      if (sign == 2) {
        result.indeterminate = true;
        return result;
      }
      if (sign <= 0) {
        result.intervals.push_back(Interval{.lower = left, .upper = right});
      }
    }
  }
  result.intervals = MergeIntervals(std::move(result.intervals));
  return result;
}

[[nodiscard]] bool
CandidateWithinLimits(const BallisticArc &arc, const Vec3 initial_velocity,
                      const HopperCapability &capability) noexcept {
  const double launch_speed = Norm(arc.launch_velocity_mps);
  const double landing_speed = Norm(arc.landing_velocity_mps);
  const Vec3 velocity_change{
      .x = arc.launch_velocity_mps.x - initial_velocity.x,
      .y = arc.launch_velocity_mps.y - initial_velocity.y,
      .z = arc.launch_velocity_mps.z - initial_velocity.z,
  };
  const double impulse = capability.platform_mass_kg * Norm(velocity_change);
  const double gravity_norm = Norm(capability.gravity_mps2);
  const double downward_speed =
      Dot(arc.landing_velocity_mps, capability.gravity_mps2) / gravity_norm;
  constexpr double kValidationTolerance = 1.0e-8;
  return std::isfinite(launch_speed) && std::isfinite(landing_speed) &&
         std::isfinite(impulse) && std::isfinite(downward_speed) &&
         launch_speed <=
             capability.maximum_launch_speed_mps + kValidationTolerance &&
         landing_speed <=
             capability.maximum_landing_speed_mps + kValidationTolerance &&
         impulse <= capability.maximum_launch_impulse_newton_seconds +
                        kValidationTolerance &&
         downward_speed + kValidationTolerance >=
             capability.minimum_downward_impact_speed_mps;
}

} // namespace

BallisticEnvelopeResult SolveBallisticEnvelope(
    const Vec3 launch_position_m, const Vec3 landing_position_m,
    const Vec3 initial_velocity_mps, const HopperCapability &capability,
    const double minimum_attitude_time_s, const std::stop_token stop_token) {
  if (stop_token.stop_requested()) {
    return Failure(BallisticEnvelopeStatus::kCanceled, "REQUEST_CANCELED");
  }
  const double minimum_flight_s =
      std::chrono::duration<double>(capability.minimum_flight_time).count();
  const double maximum_flight_s =
      std::chrono::duration<double>(capability.maximum_flight_time).count();
  const double gravity_norm = Norm(capability.gravity_mps2);
  if (!IsFinite(launch_position_m) || !IsFinite(landing_position_m) ||
      !IsFinite(initial_velocity_mps) || !IsFinite(capability.gravity_mps2) ||
      !std::isfinite(minimum_attitude_time_s) ||
      minimum_attitude_time_s < 0.0 || !std::isfinite(minimum_flight_s) ||
      minimum_flight_s <= 0.0 || !std::isfinite(maximum_flight_s) ||
      maximum_flight_s < minimum_flight_s || !std::isfinite(gravity_norm) ||
      gravity_norm <= 1.0e-9 || !std::isfinite(capability.platform_mass_kg) ||
      capability.platform_mass_kg <= 0.0 ||
      !std::isfinite(capability.maximum_launch_speed_mps) ||
      capability.maximum_launch_speed_mps <= 0.0 ||
      !std::isfinite(capability.maximum_launch_impulse_newton_seconds) ||
      capability.maximum_launch_impulse_newton_seconds <= 0.0 ||
      !std::isfinite(capability.maximum_landing_speed_mps) ||
      capability.maximum_landing_speed_mps <= 0.0 ||
      !std::isfinite(capability.minimum_downward_impact_speed_mps) ||
      capability.minimum_downward_impact_speed_mps < 0.0) {
    return Failure(BallisticEnvelopeStatus::kInvalid,
                   "HOPPER_BALLISTIC_INPUT_INVALID");
  }

  const long double minimum_time =
      std::max<long double>(minimum_flight_s, minimum_attitude_time_s);
  const long double maximum_time = maximum_flight_s;
  if (minimum_time > maximum_time + TimeTolerance(minimum_time, maximum_time)) {
    return Failure(BallisticEnvelopeStatus::kInfeasible,
                   "HOPPER_BALLISTIC_INFEASIBLE");
  }
  const Vec3 displacement{
      .x = landing_position_m.x - launch_position_m.x,
      .y = landing_position_m.y - launch_position_m.y,
      .z = landing_position_m.z - launch_position_m.z,
  };
  const long double displacement_squared = SquaredNorm(displacement);
  const long double displacement_dot_g =
      Dot(displacement, capability.gravity_mps2);
  const long double gravity_squared = SquaredNorm(capability.gravity_mps2);

  std::vector<Interval> feasible{
      Interval{.lower = minimum_time, .upper = maximum_time}};
  const auto launch_interval = SpeedInterval(
      displacement_squared, displacement_dot_g, gravity_squared,
      static_cast<long double>(capability.maximum_launch_speed_mps) *
          capability.maximum_launch_speed_mps,
      false);
  const auto landing_interval = SpeedInterval(
      displacement_squared, displacement_dot_g, gravity_squared,
      static_cast<long double>(capability.maximum_landing_speed_mps) *
          capability.maximum_landing_speed_mps,
      true);
  if (!launch_interval.has_value() || !landing_interval.has_value()) {
    return Failure(BallisticEnvelopeStatus::kInfeasible,
                   "HOPPER_BALLISTIC_INFEASIBLE");
  }
  feasible = Intersect(feasible, *launch_interval);
  feasible = Intersect(feasible, *landing_interval);
  if (feasible.empty()) {
    return Failure(BallisticEnvelopeStatus::kInfeasible,
                   "HOPPER_BALLISTIC_INFEASIBLE");
  }

  const long double impulse_speed =
      capability.maximum_launch_impulse_newton_seconds /
      capability.platform_mass_kg;
  const std::vector<long double> impulse_polynomial{
      displacement_squared,
      -2.0L * Dot(displacement, initial_velocity_mps),
      SquaredNorm(initial_velocity_mps) - displacement_dot_g -
          impulse_speed * impulse_speed,
      Dot(capability.gravity_mps2, initial_velocity_mps),
      0.25L * gravity_squared,
  };
  ConstraintResult impulse =
      ApplyPolynomialConstraint(feasible, impulse_polynomial, stop_token);
  std::size_t examined = impulse.examined_intervals;
  if (impulse.canceled) {
    return Failure(BallisticEnvelopeStatus::kCanceled, "REQUEST_CANCELED",
                   examined);
  }
  if (impulse.indeterminate) {
    return Failure(BallisticEnvelopeStatus::kNumericalIndeterminate,
                   "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE", examined);
  }
  feasible = std::move(impulse.intervals);
  if (feasible.empty()) {
    return Failure(BallisticEnvelopeStatus::kInfeasible,
                   "HOPPER_BALLISTIC_INFEASIBLE", examined);
  }

  const std::vector<long double> downward_polynomial{
      -displacement_dot_g / gravity_norm,
      capability.minimum_downward_impact_speed_mps,
      -0.5L * gravity_norm,
  };
  ConstraintResult downward =
      ApplyPolynomialConstraint(feasible, downward_polynomial, stop_token);
  examined += downward.examined_intervals;
  if (downward.canceled) {
    return Failure(BallisticEnvelopeStatus::kCanceled, "REQUEST_CANCELED",
                   examined);
  }
  if (downward.indeterminate) {
    return Failure(BallisticEnvelopeStatus::kNumericalIndeterminate,
                   "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE", examined);
  }
  feasible = std::move(downward.intervals);
  if (feasible.empty()) {
    return Failure(BallisticEnvelopeStatus::kInfeasible,
                   "HOPPER_BALLISTIC_INFEASIBLE", examined);
  }

  const long double stationary_time =
      displacement_squared > 0.0L
          ? std::pow(4.0L * displacement_squared / gravity_squared, 0.25L)
          : minimum_time;
  std::optional<BallisticArc> selected;
  long double selected_score = std::numeric_limits<long double>::infinity();
  for (const Interval interval : feasible) {
    if (stop_token.stop_requested()) {
      return Failure(BallisticEnvelopeStatus::kCanceled, "REQUEST_CANCELED",
                     examined);
    }
    std::vector<long double> candidates{
        interval.lower,
        interval.upper,
        interval.lower + (interval.upper - interval.lower) / 2.0L,
        std::clamp(stationary_time, interval.lower, interval.upper),
    };
    SortUnique(candidates);
    for (const long double candidate : candidates) {
      const BallisticSolveResult solved = SolveBallisticArc(
          launch_position_m, landing_position_m, capability.gravity_mps2,
          static_cast<double>(candidate));
      if (!solved.ok() || !CandidateWithinLimits(
                              *solved.arc, initial_velocity_mps, capability)) {
        continue;
      }
      const long double score = SquaredNorm(solved.arc->launch_velocity_mps) +
                                SquaredNorm(solved.arc->landing_velocity_mps);
      if (!selected.has_value() || score < selected_score ||
          (score == selected_score &&
           solved.arc->flight_time_s < selected->flight_time_s)) {
        selected = solved.arc;
        selected_score = score;
      }
    }
  }
  if (!selected.has_value()) {
    return Failure(BallisticEnvelopeStatus::kNumericalIndeterminate,
                   "HOPPER_BALLISTIC_NUMERICAL_INDETERMINATE", examined);
  }
  return BallisticEnvelopeResult{
      .status = BallisticEnvelopeStatus::kSolved,
      .arc = std::move(selected),
      .examined_intervals = examined,
      .reason_code = {},
  };
}

} // namespace lunar::planning::hopper
