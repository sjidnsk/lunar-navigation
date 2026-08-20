#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include <lunar_planning_msgs/msg/motion_reference.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/time.hpp>

#include "lunar_planner_core/types/geometry.hpp"

namespace lunar::planning::ros {

enum class ReferenceGuardState : std::uint8_t {
  kGroundHold,
  kJumpCommitted,
  kInFlight,
  kLandedHold,
  kUnresolved,
};

struct ReferenceGuardLimits final {
  std::chrono::nanoseconds minimum_settle_guard{};
  double maximum_landing_speed_mps{};
  double maximum_angular_speed_radps{};
};

struct ReferenceGuardDecision final {
  bool may_replace{};
  ReferenceGuardState state{ReferenceGuardState::kGroundHold};
  std::string reason_code;
};

class ReferenceGuard final {
 public:
  explicit ReferenceGuard(ReferenceGuardLimits limits);

  [[nodiscard]] bool Commit(
      const lunar_planning_msgs::msg::MotionReference& reference,
      lunar::planning::Vec3 execution_gravity_mps2);
  [[nodiscard]] ReferenceGuardDecision MayReplace(
      rclcpp::Time now,
      const std::optional<nav_msgs::msg::Odometry>& odometry);
  void Reset() noexcept;

  [[nodiscard]] ReferenceGuardState state() const noexcept;
 [[nodiscard]] bool unresolved() const noexcept;

 private:
  ReferenceGuardLimits limits_;
  ReferenceGuardState state_{ReferenceGuardState::kGroundHold};
  std::optional<lunar_planning_msgs::msg::HopSegment> committed_hop_;
  std::optional<rclcpp::Time> landed_since_;
};

}  // namespace lunar::planning::ros
