#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stop_token>
#include <string>

#include "hopper/hopper_types.hpp"
#include "lunar_planner_core/types/platform_capability.hpp"

namespace lunar::planning::hopper {

enum class BallisticEnvelopeStatus : std::uint8_t {
  kSolved,
  kInfeasible,
  kCanceled,
  kInvalid,
  kNumericalIndeterminate,
};

struct BallisticEnvelopeResult final {
  BallisticEnvelopeStatus status{BallisticEnvelopeStatus::kInvalid};
  std::optional<BallisticArc> arc;
  std::size_t examined_intervals{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return status == BallisticEnvelopeStatus::kSolved && arc.has_value() &&
           reason_code.empty();
  }
};

[[nodiscard]] BallisticEnvelopeResult SolveBallisticEnvelope(
    Vec3 launch_position_m, Vec3 landing_position_m, Vec3 initial_velocity_mps,
    const HopperCapability &capability, double minimum_attitude_time_s,
    std::stop_token stop_token);

} // namespace lunar::planning::hopper
