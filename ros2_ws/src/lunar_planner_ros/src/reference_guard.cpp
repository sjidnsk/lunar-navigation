#include "lunar_planner_ros/reference_guard.hpp"

#include <algorithm>
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

[[nodiscard]] bool ValidPoint(const geometry_msgs::msg::Point& point) noexcept {
  return Finite(point.x) && Finite(point.y) && Finite(point.z);
}

[[nodiscard]] bool ValidPose(const geometry_msgs::msg::Pose& pose) noexcept {
  const auto& q = pose.orientation;
  const double norm = std::hypot(
      std::hypot(q.w, q.x), std::hypot(q.y, q.z));
  return ValidPoint(pose.position) && Finite(norm) &&
      std::abs(norm - 1.0) <= 1.0e-6;
}

[[nodiscard]] bool NearlyEqual(
    const double lhs, const double rhs,
    const double relative_tolerance = 1.0e-6) noexcept {
  return Finite(lhs) && Finite(rhs) &&
      std::abs(lhs - rhs) <= relative_tolerance *
          std::max({1.0, std::abs(lhs), std::abs(rhs)});
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
  if (reference.plan_id.empty() || reference.header.frame_id != "map" ||
      reference.hops.size() != 1U || hop.segment_id.empty() ||
      hop.header.frame_id != "odom" ||
      hop.landing_region.points.size() < 3U || !ValidPose(hop.launch_pose) ||
      !ValidPoint(hop.nominal_landing_point) ||
      !Finite(hop.launch_velocity.x) || !Finite(hop.launch_velocity.y) ||
      !Finite(hop.launch_velocity.z) ||
      !Finite(hop.flight_tube_radius_m) ||
      hop.flight_tube_radius_m <= 0.0 ||
      !Finite(hop.required_delta_v_mps) ||
      hop.required_delta_v_mps < 0.0 ||
      !Finite(hop.available_delta_v_mps) ||
      hop.available_delta_v_mps < hop.required_delta_v_mps ||
      hop.capability_version.empty() || hop.global_map_generation == 0U ||
      hop.local_map_generation == 0U ||
      !Contains(
          hop.landing_region, hop.nominal_landing_point.x,
          hop.nominal_landing_point.y)) {
    return false;
  }
  double flight_seconds = 0.0;
  try {
    if (rclcpp::Time{hop.header.stamp, RCL_ROS_TIME}.nanoseconds() <= 0 ||
        rclcpp::Duration{hop.flight_time}.nanoseconds() <= 0) {
      return false;
    }
    flight_seconds = rclcpp::Duration{hop.flight_time}.seconds();
  } catch (const std::exception&) {
    return false;
  }
  constexpr double kLunarGravityMps2 = -1.62;
  const double landing_x = hop.launch_pose.position.x +
      hop.launch_velocity.x * flight_seconds;
  const double landing_y = hop.launch_pose.position.y +
      hop.launch_velocity.y * flight_seconds;
  const double landing_z = hop.launch_pose.position.z +
      hop.launch_velocity.z * flight_seconds +
      0.5 * kLunarGravityMps2 * flight_seconds * flight_seconds;
  if (!NearlyEqual(landing_x, hop.nominal_landing_point.x) ||
      !NearlyEqual(landing_y, hop.nominal_landing_point.y) ||
      !NearlyEqual(landing_z, hop.nominal_landing_point.z)) {
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
