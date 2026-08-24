#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include "lunar_pure_planner_core/types/platform_capability.hpp"
#include "lunar_pure_planner_ros/incremental_traversability.hpp"
#include "lunar_pure_planner_ros/platform_config.hpp"
#include "lunar_pure_planner_ros/traversability_qos.hpp"

namespace lunar::pure_planner_ros {
namespace {

constexpr std::string_view kPackageName{"lunar_pure_planner_ros"};

[[nodiscard]] bool AbsoluteTopic(const std::string& value) noexcept {
  return value.size() > 1U && value.front() == '/';
}

[[nodiscard]] std::filesystem::path ResolvePlatformConfig(
    const std::string_view platform, const std::string_view configured_path) {
  const std::filesystem::path share =
      ament_index_cpp::get_package_share_directory(std::string{kPackageName});
  if (configured_path.empty()) return share / "config" / (std::string{platform} + ".yaml");
  const std::filesystem::path explicit_path{configured_path};
  return explicit_path.is_absolute() ? explicit_path : share / "config" / explicit_path;
}

class LocalTraversabilityNode final : public rclcpp::Node {
 public:
  LocalTraversabilityNode() : Node("lunar_local_traversability") {
    const std::string platform = declare_parameter<std::string>("platform_type", "wheel");
    const std::string configured_path =
        declare_parameter<std::string>("platform_config", "");
    const double occupancy_threshold =
        declare_parameter<double>("local_occupancy_threshold", 0.5);
    const std::string local_topic =
        declare_parameter<std::string>("local_map_topic", "/Car/T3/mapping/grid_map");
    const std::string output_topic = declare_parameter<std::string>(
        "traversability_topic", "/Car/T4/planning/local_traversability");
    const std::string input_reliability = declare_parameter<std::string>(
        "input_qos_reliability", "reliable");
    const std::string input_durability = declare_parameter<std::string>(
        "input_qos_durability", "transient_local");
    if (!AbsoluteTopic(local_topic) || !AbsoluteTopic(output_topic) ||
        !std::isfinite(occupancy_threshold) || occupancy_threshold < 0.0 ||
        occupancy_threshold > 1.0) {
      throw std::runtime_error{"invalid traversability node parameters"};
    }
    const auto input_qos = MakeTraversabilityInputQos(
        input_reliability, input_durability);
    if (!input_qos.has_value()) {
      throw std::runtime_error{"invalid traversability input QoS parameters"};
    }
    const auto config_path = ResolvePlatformConfig(platform, configured_path);
    auto loaded = LoadPlatformConfig(config_path, platform);
    if (!loaded.capability.has_value()) {
      throw std::runtime_error{"platform_config rejected: " + config_path.string()};
    }
    builder_ = std::make_unique<IncrementalTraversability>(
        MakeTraversabilityProfile(*loaded.capability, occupancy_threshold));
    publisher_ = create_publisher<grid_map_msgs::msg::GridMap>(
        output_topic, rclcpp::QoS{1}.reliable().transient_local());
    subscription_ = create_subscription<grid_map_msgs::msg::GridMap>(
        local_topic, *input_qos,
        [this](grid_map_msgs::msg::GridMap::ConstSharedPtr message) {
          const auto update = builder_->Update(*message);
          if (update.has_value()) publisher_->publish(update->map);
        });
  }

 private:
  std::unique_ptr<IncrementalTraversability> builder_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr publisher_;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr subscription_;
};

}  // namespace

std::shared_ptr<rclcpp::Node> MakeLocalTraversabilityNode() {
  return std::make_shared<LocalTraversabilityNode>();
}

}  // namespace lunar::pure_planner_ros
