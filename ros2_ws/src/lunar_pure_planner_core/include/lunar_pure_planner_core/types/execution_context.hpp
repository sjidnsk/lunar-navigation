#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace lunar::pure_planning {

enum class GroundExecutionState : std::uint8_t {
  kIdle,
  kExecuting,
  kHolding,
  kFault,
};

struct GroundExecutionContext final {
  GroundExecutionState state{GroundExecutionState::kIdle};
  std::optional<std::string> active_plan_id;
  std::optional<std::string> active_segment_id;
};

enum class HopperExecutionState : std::uint8_t {
  kGroundHold,
  kJumpReady,
  kJumpCommitted,
  kInFlight,
  kLandedHold,
  kEmergencyDelegated,
};

struct HopperExecutionContext final {
  HopperExecutionState state{HopperExecutionState::kGroundHold};
  std::optional<std::string> active_plan_id;
  std::optional<std::string> active_segment_id;
};

using ExecutionContext =
    std::variant<GroundExecutionContext, HopperExecutionContext>;

}  // namespace lunar::pure_planning
