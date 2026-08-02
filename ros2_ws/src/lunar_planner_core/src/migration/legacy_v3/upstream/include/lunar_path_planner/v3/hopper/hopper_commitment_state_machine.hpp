#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "lunar_path_planner/v3/contracts/base_types.hpp"

namespace lunar::planning::v3 {

enum class HopperCommitmentState : std::uint8_t {
  kGroundHold,
  kJumpReady,
  kJumpCommitted,
  kInFlight,
  kLandedHold,
  kEmergencyDelegated,
};

enum class HopperCommitmentEventType : std::uint8_t {
  kPublishCertifiedCandidate,
  kWithdrawCandidate,
  kLockJumpBoundary,
  kDetectLaunch,
  kDetectStableLanding,
  kInvalidateCommittedAction,
};

enum class HopperCommitmentScope : std::uint8_t {
  kGroundHold,
  kJumpBoundary,
  kEmergencyDelegated,
};

struct HopperCommitmentSnapshot final {
  HopperCommitmentState state{HopperCommitmentState::kGroundHold};
  std::optional<std::string> active_bundle_id;
  std::optional<std::string> active_boundary_id;
  HopperCommitmentScope commitment_scope{
      HopperCommitmentScope::kGroundHold};
};

struct HopperCommitmentEvent final {
  HopperCommitmentEventType type{
      HopperCommitmentEventType::kPublishCertifiedCandidate};
  std::optional<std::string> bundle_id;
  std::optional<std::string> boundary_id;
};

struct HopperCommitmentTransition final {
  HopperCommitmentSnapshot next;
  bool accepted{};
  ReasonCode reason_code;
};

class HopperCommitmentStateMachine final {
 public:
  [[nodiscard]] HopperCommitmentTransition transition(
      const HopperCommitmentSnapshot& current,
      const HopperCommitmentEvent& event) const;

  [[nodiscard]] bool may_publish_new_jump(
      const HopperCommitmentSnapshot& current) const noexcept;
};

}  // namespace lunar::planning::v3
