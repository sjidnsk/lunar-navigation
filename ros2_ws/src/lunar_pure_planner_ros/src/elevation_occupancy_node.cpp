#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include <grid_map_msgs/msg/grid_map.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include "lunar_pure_planner_ros/elevation_occupancy.hpp"
#include "lunar_pure_planner_ros/traversability_qos.hpp"

namespace lunar::pure_planner_ros {
namespace {

[[nodiscard]] bool AbsoluteTopic(const std::string& value) noexcept {
  return value.size() > 1U && value.front() == '/';
}

class ElevationOccupancyNode final : public rclcpp::Node {
 public:
  ElevationOccupancyNode() : Node("lunar_elevation_occupancy") {
    const std::string input_topic = declare_parameter<std::string>(
        "input_topic", "/Car/T3/mapping/elevation_grid_map");
    const std::string output_topic = declare_parameter<std::string>(
        "output_topic", "/Car/T3/mapping/grid_map");
    elevation_layer_ = declare_parameter<std::string>("elevation_layer", "elevation");
    occupancy_layer_ = declare_parameter<std::string>("occupancy_layer", "occupancy");
    overwrite_existing_occupancy_ = declare_parameter<bool>(
        "overwrite_existing_occupancy", false);
    const std::string input_reliability = declare_parameter<std::string>(
        "input_qos_reliability", "reliable");
    const std::string input_durability = declare_parameter<std::string>(
        "input_qos_durability", "transient_local");
    if (!AbsoluteTopic(input_topic) || !AbsoluteTopic(output_topic) ||
        elevation_layer_.empty() || occupancy_layer_.empty()) {
      throw std::runtime_error{"invalid elevation occupancy node parameters"};
    }
    const auto input_qos = MakeTraversabilityInputQos(
        input_reliability, input_durability);
    if (!input_qos.has_value()) {
      throw std::runtime_error{"invalid elevation occupancy input QoS parameters"};
    }
    publisher_ = create_publisher<grid_map_msgs::msg::GridMap>(
        output_topic, rclcpp::QoS{1}.reliable().transient_local());
    subscription_ = create_subscription<grid_map_msgs::msg::GridMap>(
        input_topic, *input_qos,
        [this](grid_map_msgs::msg::GridMap::ConstSharedPtr message) {
          const auto output = AddOccupancyFromElevation(
              *message, elevation_layer_, occupancy_layer_,
              overwrite_existing_occupancy_);
          if (!output.has_value()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                 "dropped map without a valid elevation layer");
            return;
          }
          publisher_->publish(*output);
        });
  }

 private:
  std::string elevation_layer_;
  std::string occupancy_layer_;
  bool overwrite_existing_occupancy_{};
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr publisher_;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr subscription_;
};

}  // namespace

std::shared_ptr<rclcpp::Node> MakeElevationOccupancyNode() {
  return std::make_shared<ElevationOccupancyNode>();
}

}  // namespace lunar::pure_planner_ros
