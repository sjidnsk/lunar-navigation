#include "lunar_planner_ros/reference_guard.hpp"

#include <cmath>
#include <stdexcept>

#include <rclcpp/duration.hpp>

namespace lunar::planning::ros {
namespace {

[[nodiscard]] bool Finite(const double value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] bool ValidPoint(const geometry_msgs::msg::Point32& point) noexcept {
  return Finite(point.x) && Finite(point.y) && Finite(point.z);
}

[[nodiscard]] bool OnSegment(
    const double x,
    const double y,
    const geometry_msgs::msg::Point32& first,
    const geometry_msgs::msg::Point32& second) noexcept {
  constexpr double kTolerance = 1.0e-9;
  const double cross =
      (x - first.x) * (second.y - first.y) -
      (y - first.y) * (second.x - first.x);
  if (std::abs(cross) > kTolerance) {
    return false;
  }
  const double dot =
      (x - first.x) * (x - second.x) +
      (y - first.y) * (y - second.y);
  return dot <= kTolerance;
}

[[nodiscard]] bool Contains(
    const geometry_msgs::msg::Polygon& polygon,
    const double x,
    const double y) noexcept {
  if (polygon.points.size() < 3U || !Finite(x) || !Finite(y)) {
    return false;
  }
  bool inside = false;
  for (std::size_t index = 0U, previous = polygon.points.size() - 1U;
       index < polygon.points.size(); previous = index++) {
    const auto& first = polygon.points[previous];
    const auto& second = polygon.points[index];
    if (!ValidPoint(first) || !ValidPoint(second)) {
      return false;
    }
    if (OnSegment(x, y, first, second)) {
      return true;
    }
    const bool crosses = (first.y > y) != (second.y > y);
    if (crosses) {
      const double edge_x =
          (second.x - first.x) * (y - first.y) /
              (second.y - first.y) + first.x;
      if (x < edge_x) {
        inside = !inside;
      }
    }
  }
  return inside;
}

[[nodiscard]] double Norm(
    const geometry_msgs::msg::Vector3& vector) noexcept {
  return std::hypot(vector.x, vector.y, vector.z);
}

[[nodiscard]] ReferenceGuardDecision Locked(
    const ReferenceGuardState state,
    std::string reason_code) {
  return ReferenceGuardDecision{
      .may_replace = false,
      .state = state,
      .reason_code = std::move(reason_code),
  };
}

[[nodiscard]] bool ValidHop(
    const lunar_planning_msgs::msg::MotionReference& reference,
    const lunar_planning_msgs::msg::HopSegment& hop) noexcept {
  if (reference.plan_id.empty() || reference.header.frame_id != "odom" ||
      hop.segment_id.empty() || hop.header.frame_id != "odom" ||
      hop.landing_region.points.size() < 3U) {
    return false;
  }
  try {
    if (rclcpp::Time{hop.header.stamp, RCL_ROS_TIME}.nanoseconds() <= 0 ||
        rclcpp::Duration{hop.flight_time}.nanoseconds() <= 0) {
      return false;
    }
  } catch (const std::exception&) {
    return false;
  }
  for (const auto& point : hop.landing_region.points) {
    if (!ValidPoint(point)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool StableLanding(
    const nav_msgs::msg::Odometry& odometry,
    const lunar_planning_msgs::msg::HopSegment& hop,
    const ReferenceGuardLimits& limits,
    const rclcpp::Time& flight_end,
    const rclcpp::Time& now) noexcept {
  if (odometry.header.frame_id != "odom" ||
      odometry.child_frame_id != "base_link") {
    return false;
  }
  try {
    const rclcpp::Time odometry_time{odometry.header.stamp, RCL_ROS_TIME};
    if (odometry_time < flight_end || odometry_time > now) {
      return false;
    }
  } catch (const std::exception&) {
    return false;
  }
  const auto& position = odometry.pose.pose.position;
  if (!Finite(position.x) || !Finite(position.y) || !Finite(position.z) ||
      !Contains(hop.landing_region, position.x, position.y)) {
    return false;
  }
  const double linear_speed = Norm(odometry.twist.twist.linear);
  const double angular_speed = Norm(odometry.twist.twist.angular);
  return Finite(linear_speed) && Finite(angular_speed) &&
      linear_speed <= limits.maximum_landing_speed_mps &&
      angular_speed <= limits.maximum_angular_speed_radps;
}

}  // namespace

ReferenceGuard::ReferenceGuard(ReferenceGuardLimits limits)
    : limits_(limits) {
  if (limits_.minimum_settle_guard.count() <= 0 ||
      !Finite(limits_.maximum_landing_speed_mps) ||
      limits_.maximum_landing_speed_mps <= 0.0 ||
      !Finite(limits_.maximum_angular_speed_radps) ||
      limits_.maximum_angular_speed_radps <= 0.0) {
    throw std::invalid_argument{"invalid reference guard limits"};
  }
}

bool ReferenceGuard::Commit(
    const lunar_planning_msgs::msg::MotionReference& reference) {
  if (state_ != ReferenceGuardState::kGroundHold ||
      reference.platform_type != reference.HOPPER || reference.hops.empty() ||
      !ValidHop(reference, reference.hops.front())) {
    return false;
  }
  committed_hop_ = reference.hops.front();
  landed_since_.reset();
  state_ = ReferenceGuardState::kJumpCommitted;
  return true;
}

ReferenceGuardDecision ReferenceGuard::MayReplace(
    const rclcpp::Time now,
    const std::optional<nav_msgs::msg::Odometry>& odometry) {
  if (state_ == ReferenceGuardState::kGroundHold) {
    return ReferenceGuardDecision{
        .may_replace = true,
        .state = state_,
        .reason_code = {},
    };
  }
  if (state_ == ReferenceGuardState::kUnresolved || !committed_hop_) {
    state_ = ReferenceGuardState::kUnresolved;
    return Locked(state_, "HOP_EXECUTION_STATE_UNRESOLVED");
  }

  const rclcpp::Time launch_time{committed_hop_->header.stamp, RCL_ROS_TIME};
  const rclcpp::Time flight_end =
      launch_time + rclcpp::Duration{committed_hop_->flight_time};
  if (now < launch_time) {
    state_ = ReferenceGuardState::kJumpCommitted;
    return Locked(state_, "HOP_JUMP_COMMITTED");
  }
  if (now < flight_end) {
    state_ = ReferenceGuardState::kInFlight;
    return Locked(state_, "HOP_IN_FLIGHT");
  }
  if (!odometry ||
      !StableLanding(*odometry, *committed_hop_, limits_, flight_end, now)) {
    state_ = ReferenceGuardState::kUnresolved;
    landed_since_.reset();
    return Locked(state_, "HOP_EXECUTION_STATE_UNRESOLVED");
  }
  if (state_ != ReferenceGuardState::kLandedHold || !landed_since_) {
    state_ = ReferenceGuardState::kLandedHold;
    landed_since_ = now;
    return Locked(state_, "HOP_LANDED_SETTLE_GUARD");
  }
  if (now - *landed_since_ < rclcpp::Duration{limits_.minimum_settle_guard}) {
    return Locked(state_, "HOP_LANDED_SETTLE_GUARD");
  }

  Reset();
  return ReferenceGuardDecision{
      .may_replace = true,
      .state = state_,
      .reason_code = {},
  };
}

void ReferenceGuard::Reset() noexcept {
  state_ = ReferenceGuardState::kGroundHold;
  committed_hop_.reset();
  landed_since_.reset();
}

ReferenceGuardState ReferenceGuard::state() const noexcept {
  return state_;
}

bool ReferenceGuard::unresolved() const noexcept {
  return state_ == ReferenceGuardState::kUnresolved;
}

}  // namespace lunar::planning::ros
