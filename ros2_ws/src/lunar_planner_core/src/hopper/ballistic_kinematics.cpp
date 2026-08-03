#include "hopper/ballistic_kinematics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace lunar::planning::hopper {
namespace {

[[nodiscard]] bool IsFinite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

[[nodiscard]] Vec3 Add(const Vec3 lhs, const Vec3 rhs) noexcept {
  return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

[[nodiscard]] Vec3 Scale(const Vec3 value, const double scale) noexcept {
  return {value.x * scale, value.y * scale, value.z * scale};
}

[[nodiscard]] double Dot(const Vec3 lhs, const Vec3 rhs) noexcept {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

}  // namespace

BallisticSolveResult SolveBallisticArc(
    const Vec3 launch_position_m, const Vec3 landing_position_m,
    const Vec3 gravity_mps2, const double flight_time_s) noexcept {
  if (!IsFinite(launch_position_m) || !IsFinite(landing_position_m) ||
      !IsFinite(gravity_mps2) || !std::isfinite(flight_time_s) ||
      flight_time_s <= 0.0) {
    return BallisticSolveResult{
        .arc = std::nullopt,
        .reason_code = "HOPPER_BALLISTIC_INPUT_INVALID",
    };
  }
  const Vec3 displacement{
      .x = landing_position_m.x - launch_position_m.x,
      .y = landing_position_m.y - launch_position_m.y,
      .z = landing_position_m.z - launch_position_m.z,
  };
  const Vec3 launch_velocity{
      .x = displacement.x / flight_time_s -
          0.5 * gravity_mps2.x * flight_time_s,
      .y = displacement.y / flight_time_s -
          0.5 * gravity_mps2.y * flight_time_s,
      .z = displacement.z / flight_time_s -
          0.5 * gravity_mps2.z * flight_time_s,
  };
  const Vec3 landing_velocity =
      Add(launch_velocity, Scale(gravity_mps2, flight_time_s));
  if (!IsFinite(launch_velocity) || !IsFinite(landing_velocity)) {
    return BallisticSolveResult{
        .arc = std::nullopt,
        .reason_code = "HOPPER_BALLISTIC_NUMERICAL_FAILURE",
    };
  }
  return BallisticSolveResult{
      .arc = BallisticArc{
          .launch_position_m = launch_position_m,
          .landing_position_m = landing_position_m,
          .gravity_mps2 = gravity_mps2,
          .launch_velocity_mps = launch_velocity,
          .landing_velocity_mps = landing_velocity,
          .flight_time_s = flight_time_s,
      },
      .reason_code = {},
  };
}

BallisticState EvaluateBallisticState(
    const BallisticArc& arc, const double time_s) noexcept {
  const double time_squared = time_s * time_s;
  return BallisticState{
      .position_m =
          Add(
              Add(
                  arc.launch_position_m,
                  Scale(arc.launch_velocity_mps, time_s)),
              Scale(arc.gravity_mps2, 0.5 * time_squared)),
      .velocity_mps =
          Add(arc.launch_velocity_mps, Scale(arc.gravity_mps2, time_s)),
  };
}

double BallisticApexTime(const BallisticArc& arc) noexcept {
  const double gravity_squared = Dot(arc.gravity_mps2, arc.gravity_mps2);
  if (!std::isfinite(gravity_squared) ||
      gravity_squared <= std::numeric_limits<double>::epsilon()) {
    return 0.0;
  }
  return std::clamp(
      -Dot(arc.launch_velocity_mps, arc.gravity_mps2) / gravity_squared,
      0.0, arc.flight_time_s);
}

}  // namespace lunar::planning::hopper
