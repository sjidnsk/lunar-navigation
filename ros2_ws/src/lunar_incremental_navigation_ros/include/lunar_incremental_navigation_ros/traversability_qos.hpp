#pragma once

#include <optional>
#include <string_view>

#include <rclcpp/qos.hpp>

namespace lunar::incremental_navigation_ros {

[[nodiscard]] rclcpp::QoS MakeTask3LocalMapInputQos();

[[nodiscard]] std::optional<rclcpp::QoS> MakeTraversabilityInputQos(
    std::string_view reliability, std::string_view durability);

}  // namespace lunar::incremental_navigation_ros
