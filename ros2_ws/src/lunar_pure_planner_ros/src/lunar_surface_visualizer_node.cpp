#include <cstdint>
#include <memory>

#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
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
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/lunar_demo/odometry", rclcpp::QoS{10}.reliable(),
        [this](nav_msgs::msg::Odometry::ConstSharedPtr odometry) {
          PublishRoverMarkers(odometry->pose.pose.position.x,
                              odometry->pose.pose.position.y,
                              odometry->pose.pose.position.z);
        });
    PublishMarkers();
  }

  void PublishRoverMarkers(const double x, const double y, const double z) {
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker rover;
    rover.header.frame_id = "map";
    rover.ns = "lunar_surface_rover";
    rover.id = 1;
    rover.type = visualization_msgs::msg::Marker::CYLINDER;
    rover.action = visualization_msgs::msg::Marker::ADD;
    rover.pose.position.x = x; rover.pose.position.y = y; rover.pose.position.z = z + 0.25;
    rover.pose.orientation.w = 1.0;
    rover.scale.x = 12.0; rover.scale.y = 12.0; rover.scale.z = 0.8;
    rover.color.r = 0.80F; rover.color.g = 0.47F; rover.color.b = 0.65F; rover.color.a = 1.0F;
    markers.markers.push_back(std::move(rover));
    visualization_msgs::msg::Marker label;
    label.header.frame_id = "map";
    label.ns = "lunar_surface_rover";
    label.id = 2;
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    label.pose.position.x = x; label.pose.position.y = y; label.pose.position.z = z + 7.25;
    label.pose.orientation.w = 1.0;
    label.scale.z = 8.0;
    label.color.r = 1.0F; label.color.g = 1.0F; label.color.b = 1.0F; label.color.a = 1.0F;
    label.text = "Current rover position";
    markers.markers.push_back(std::move(label));
    marker_pub_->publish(markers);
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
    // One marker per metre is needlessly dense at this scale. The sampled
    // terrain complements the full-resolution map and traversal overlays.
    terrain.scale.x = 8.0;
    terrain.scale.y = 8.0;
    terrain.scale.z = 0.06;
    for (std::size_t y = 0U; y < scenario_.height; y += 10U) {
      for (std::size_t x = 0U; x < scenario_.width; x += 10U) {
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

    const auto start_index = scenario_.Index(scenario_.start_cell);
    const double start_x = scenario_.origin_x_m +
                           (static_cast<double>(scenario_.start_cell.x) + 0.5) *
                               scenario_.resolution_m;
    const double start_y = scenario_.origin_y_m +
                           (static_cast<double>(scenario_.start_cell.y) + 0.5) *
                               scenario_.resolution_m;
    const double start_z = scenario_.elevation_m[start_index] + 0.25;

    // The standard Odometry arrow is only about one metre long, which is
    // imperceptible in a 1 km top-down view.  Add a map-scale, labelled marker
    // at the demo's current static rover position.
    visualization_msgs::msg::Marker rover;
    rover.header.frame_id = "map";
    rover.ns = "lunar_surface_rover";
    rover.id = 1;
    rover.type = visualization_msgs::msg::Marker::CYLINDER;
    rover.action = visualization_msgs::msg::Marker::ADD;
    rover.pose.position.x = start_x;
    rover.pose.position.y = start_y;
    rover.pose.position.z = start_z;
    rover.pose.orientation.w = 1.0;
    rover.scale.x = 12.0;
    rover.scale.y = 12.0;
    rover.scale.z = 0.8;
    rover.color.r = 0.80F;
    rover.color.g = 0.47F;
    rover.color.b = 0.65F;
    rover.color.a = 1.0F;
    markers.markers.push_back(std::move(rover));

    visualization_msgs::msg::Marker rover_label;
    rover_label.header.frame_id = "map";
    rover_label.ns = "lunar_surface_rover";
    rover_label.id = 2;
    rover_label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    rover_label.action = visualization_msgs::msg::Marker::ADD;
    rover_label.pose.position.x = start_x;
    rover_label.pose.position.y = start_y;
    rover_label.pose.position.z = start_z + 7.0;
    rover_label.pose.orientation.w = 1.0;
    rover_label.scale.z = 8.0;
    rover_label.color.r = 1.0F;
    rover_label.color.g = 1.0F;
    rover_label.color.b = 1.0F;
    rover_label.color.a = 1.0F;
    rover_label.text = "Current rover position";
    markers.markers.push_back(std::move(rover_label));

    marker_pub_->publish(markers);
  }

  LunarSurfaceScenario scenario_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
};

}  // namespace
}  // namespace lunar::pure_planner_ros

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lunar::pure_planner_ros::LunarSurfaceVisualizerNode>());
  rclcpp::shutdown();
  return 0;
}
