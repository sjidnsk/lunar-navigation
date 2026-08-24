#pragma once

#include <optional>
#include <string_view>

#include <rclcpp/qos.hpp>

namespace lunar::pure_planner_ros {

[[nodiscard]] std::optional<rclcpp::QoS> MakeTraversabilityInputQos(
    std::string_view reliability, std::string_view durability);

}  // namespace lunar::pure_planner_ros
