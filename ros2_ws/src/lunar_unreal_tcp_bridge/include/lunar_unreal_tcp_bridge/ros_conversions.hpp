#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>

#include <grid_map_msgs/msg/grid_map.hpp>
#include <lunar_navigation_msgs/msg/motion_execution_feedback.hpp>
#include <lunar_planning_msgs/msg/motion_reference.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lunar_unreal_tcp_bridge/coordinate_transform.hpp"
#include "lunar_unreal_tcp_bridge/protocol.hpp"
#include "lunar_unreal_tcp_bridge/session_protocol.hpp"

namespace lunar::unreal_tcp {

template<typename Value>
struct ConversionResult final {
  std::optional<Value> value;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept { return value.has_value(); }
};

struct SimulationCovarianceProfile final {
  std::array<double, 36> pose{};
  std::array<double, 36> twist{};
};

struct RobotStateMessages final {
  nav_msgs::msg::Odometry external_odometry;
  nav_msgs::msg::Odometry wheeled_odometry;
  tf2_msgs::msg::TFMessage dynamic_tf;
  tf2_msgs::msg::TFMessage static_tf;
};

[[nodiscard]] ConversionResult<RobotStateMessages> ConvertRobotState(
    const Frame& frame,
    const CoordinateTransform& transform,
    const SimulationCovarianceProfile& covariance_profile);

[[nodiscard]] ConversionResult<grid_map_msgs::msg::GridMap>
ConvertObservedElevation(
    const Frame& frame, const CoordinateTransform& transform);

[[nodiscard]] ConversionResult<Frame> ConvertMotionReference(
    const lunar_planning_msgs::msg::MotionReference& message,
    const CoordinateTransform& transform,
    const FrozenSession& session,
    std::uint64_t outgoing_sequence);

[[nodiscard]] ConversionResult<
    lunar_navigation_msgs::msg::MotionExecutionFeedback>
ConvertExecutionFeedback(
    const Frame& frame, std::string_view expected_session_id);

}  // namespace lunar::unreal_tcp
