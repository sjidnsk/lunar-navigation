#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lunar_pure_planner_ros/lunar_surface_scenario.hpp"

namespace lunar::pure_planner_ros {
namespace {

std_msgs::msg::Float32MultiArray MakeLayer(const std::vector<float>& values,
                                           const LunarSurfaceScenario& scenario) {
  std_msgs::msg::Float32MultiArray layer;
  layer.layout.dim.resize(2U);
  layer.layout.dim[0].label = "column_index";
  layer.layout.dim[0].size = scenario.height;
  layer.layout.dim[0].stride = scenario.width * scenario.height;
  layer.layout.dim[1].label = "row_index";
  layer.layout.dim[1].size = scenario.width;
  layer.layout.dim[1].stride = scenario.width;
  layer.data.resize(values.size());
  for (std::size_t y = 0U; y < scenario.height; ++y) {
    for (std::size_t x = 0U; x < scenario.width; ++x) {
      const std::size_t logical = y * scenario.width + x;
      const std::size_t physical =
          (scenario.height - 1U - y) * scenario.width +
          (scenario.width - 1U - x);
      layer.data[physical] = values[logical];
    }
  }
  return layer;
}

class LunarSurfaceDemoNode final : public rclcpp::Node {
 public:
  LunarSurfaceDemoNode()
      : Node("lunar_surface_demo"),
        scenario_(BuildLunarSurfaceScenario(
            static_cast<std::uint32_t>(declare_parameter<std::int64_t>("seed", 20260823)))) {
    // RViz and tf2 listeners request reliable delivery by default.  A reliable
    // writer also remains compatible with the planner's best-effort readers.
    const auto input_qos = rclcpp::QoS{10}.reliable();
    global_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
        "/lunar_demo/global_overview", input_qos);
    local_pub_ = create_publisher<grid_map_msgs::msg::GridMap>(
        "/lunar_demo/grid_map", input_qos);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/lunar_demo/odometry", input_qos);
    tf_pub_ = create_publisher<tf2_msgs::msg::TFMessage>("/tf", input_qos);
    goal_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        "/lunar_demo/rviz_goal", rclcpp::QoS{1}.reliable());
    default_goal_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        "/lunar_demo/default_goal", rclcpp::QoS{1}.transient_local());
    timer_ = create_wall_timer(std::chrono::milliseconds{500}, [this] { Publish(); });
  }

 private:
  [[nodiscard]] geometry_msgs::msg::PoseStamped Goal(const rclcpp::Time& stamp) const {
    geometry_msgs::msg::PoseStamped message;
    message.header.stamp = stamp;
    message.header.frame_id = "map";
    message.pose.position.x = scenario_.origin_x_m +
                              (static_cast<double>(scenario_.default_goal_cell.x) + 0.5) * scenario_.resolution_m;
    message.pose.position.y = scenario_.origin_y_m +
                              (static_cast<double>(scenario_.default_goal_cell.y) + 0.5) * scenario_.resolution_m;
    message.pose.orientation.w = 1.0;
    return message;
  }

  void Publish() {
    const rclcpp::Time stamp = now();
    nav_msgs::msg::OccupancyGrid global;
    global.header.stamp = stamp;
    global.header.frame_id = "map";
    global.info.resolution = scenario_.resolution_m;
    global.info.width = scenario_.width;
    global.info.height = scenario_.height;
    global.info.origin.position.x = scenario_.origin_x_m;
    global.info.origin.position.y = scenario_.origin_y_m;
    global.info.origin.orientation.w = 1.0;
    global.data = scenario_.occupancy;
    global_pub_->publish(global);

    grid_map_msgs::msg::GridMap local;
    local.header.stamp = stamp;
    local.header.frame_id = "odom";
    local.info.resolution = scenario_.resolution_m;
    local.info.length_x = static_cast<double>(scenario_.width) * scenario_.resolution_m;
    local.info.length_y = static_cast<double>(scenario_.height) * scenario_.resolution_m;
    local.info.pose.position.x = 0.0;
    local.info.pose.position.y = 0.0;
    local.info.pose.orientation.w = 1.0;
    local.layers = {"occupancy", "elevation"};
    local.basic_layers = local.layers;
    std::vector<float> occupancy(scenario_.occupancy.size());
    for (std::size_t index = 0U; index < occupancy.size(); ++index) {
      occupancy[index] = scenario_.occupancy[index] >= 50 ? 1.0F : 0.0F;
    }
    local.data = {MakeLayer(occupancy, scenario_), MakeLayer(scenario_.elevation_m, scenario_)};
    local_pub_->publish(local);

    const double start_x = scenario_.origin_x_m +
                           (static_cast<double>(scenario_.start_cell.x) + 0.5) * scenario_.resolution_m;
    const double start_y = scenario_.origin_y_m +
                           (static_cast<double>(scenario_.start_cell.y) + 0.5) * scenario_.resolution_m;
    nav_msgs::msg::Odometry odometry;
    odometry.header.stamp = stamp;
    odometry.header.frame_id = "odom";
    odometry.child_frame_id = "base_link";
    odometry.pose.pose.position.x = start_x;
    odometry.pose.pose.position.y = start_y;
    odometry.pose.pose.position.z =
        scenario_.elevation_m[scenario_.Index(scenario_.start_cell)];
    odometry.pose.pose.orientation.w = 1.0;
    odom_pub_->publish(odometry);

    tf2_msgs::msg::TFMessage transforms;
    transforms.transforms.resize(1U);
    auto& transform = transforms.transforms.front();
    transform.header.stamp = stamp;
    transform.header.frame_id = "map";
    transform.child_frame_id = "odom";
    transform.transform.rotation.w = 1.0;
    tf_pub_->publish(transforms);

    const auto goal = Goal(stamp);
    default_goal_pub_->publish(goal);
    if (++publish_count_ == 3U) {
      goal_pub_->publish(goal);
    }
  }

  LunarSurfaceScenario scenario_;
  std::size_t publish_count_{};
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_pub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr local_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr default_goal_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace
}  // namespace lunar::pure_planner_ros

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lunar::pure_planner_ros::LunarSurfaceDemoNode>());
  rclcpp::shutdown();
  return 0;
}
