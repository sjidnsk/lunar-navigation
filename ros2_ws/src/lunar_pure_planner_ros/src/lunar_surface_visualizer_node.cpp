#include <cstdint>
#include <memory>

#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lunar_pure_planner_ros/lunar_surface_scenario.hpp"

namespace lunar::pure_planner_ros {
namespace {

class LunarSurfaceVisualizerNode final : public rclcpp::Node {
 public:
  LunarSurfaceVisualizerNode()
      : Node("lunar_surface_visualizer"),
        scenario_(BuildLunarSurfaceScenario(
            static_cast<std::uint32_t>(declare_parameter<std::int64_t>("seed", 20260823)))) {
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/lunar_demo/path", rclcpp::QoS{10}.reliable());
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/lunar_demo/terrain_markers", rclcpp::QoS{1}.transient_local());
    path_sub_ = create_subscription<nav_msgs::msg::Path>(
        "/lunar_demo/wheeled_path", rclcpp::QoS{10}.reliable(),
        [this](nav_msgs::msg::Path::ConstSharedPtr message) {
          // A rejected request contains an empty path. Publishing it makes
          // RViz emit a frame-id warning and does not represent a valid route.
          if (!message->header.frame_id.empty() && !message->poses.empty()) {
            path_pub_->publish(*message);
          }
        });
    PublishMarkers();
  }

 private:
  void PublishMarkers() {
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker terrain;
    terrain.header.frame_id = "map";
    terrain.ns = "lunar_surface";
    terrain.id = 0;
    terrain.type = visualization_msgs::msg::Marker::CUBE_LIST;
    terrain.action = visualization_msgs::msg::Marker::ADD;
    terrain.pose.orientation.w = 1.0;
    terrain.scale.x = 0.95;
    terrain.scale.y = 0.95;
    terrain.scale.z = 0.06;
    for (std::size_t y = 0U; y < scenario_.height; y += 5U) {
      for (std::size_t x = 0U; x < scenario_.width; x += 5U) {
        const auto index = scenario_.Index({x, y});
        geometry_msgs::msg::Point point;
        point.x = scenario_.origin_x_m + (static_cast<double>(x) + 0.5) * scenario_.resolution_m;
        point.y = scenario_.origin_y_m + (static_cast<double>(y) + 0.5) * scenario_.resolution_m;
        point.z = scenario_.elevation_m[index];
        terrain.points.push_back(point);
        std_msgs::msg::ColorRGBA color;
        const float height = scenario_.elevation_m[index];
        color.r = 0.35F + 0.25F * height;
        color.g = 0.30F + 0.20F * height;
        color.b = 0.24F + 0.15F * height;
        color.a = 0.75F;
        terrain.colors.push_back(color);
      }
    }
    markers.markers.push_back(std::move(terrain));
    marker_pub_->publish(markers);
  }

  LunarSurfaceScenario scenario_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
};

}  // namespace
}  // namespace lunar::pure_planner_ros

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lunar::pure_planner_ros::LunarSurfaceVisualizerNode>());
  rclcpp::shutdown();
  return 0;
}
