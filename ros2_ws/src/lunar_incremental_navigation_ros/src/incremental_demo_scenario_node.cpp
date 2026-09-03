#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <lunar_planning_msgs/msg/path_reference.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lunar_incremental_navigation_ros/incremental_demo_scenario.hpp"

namespace lunar::incremental_navigation_ros {
namespace {

using namespace std::chrono_literals;

rclcpp::QoS LatchedQos() {
  return rclcpp::QoS{rclcpp::KeepLast{1}}.reliable().transient_local();
}

class IncrementalDemoScenarioNode final : public rclcpp::Node {
 public:
  IncrementalDemoScenarioNode()
      : Node("incremental_demo_scenario"),
        platform_type_(declare_parameter<std::string>("platform_type", "wheel")),
        show_ground_truth_(
            declare_parameter<bool>("show_ground_truth", false)),
        scenario_(IncrementalDemoScenarioConfig{
            .fine_resolution_m =
                declare_parameter<double>("fine_resolution_m", 0.2),
            .local_window_size_m = 16.0,
            .task_size_m = declare_parameter<double>("task_size_m", 300.0)}) {
    if (platform_type_ != "wheel" && platform_type_ != "legged") {
      throw std::invalid_argument{"platform_type must be wheel or legged"};
    }
    grid_map_publisher_ = create_publisher<grid_map_msgs::msg::GridMap>(
        "/planning_demo/grid_map", LatchedQos());
    active_path_publisher_ = create_publisher<nav_msgs::msg::Path>(
        "/planning_demo/visualization/active_path", LatchedQos());
    trace_publisher_ = create_publisher<nav_msgs::msg::Path>(
        "/planning_demo/robot_trace", LatchedQos());
    robot_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/planning_demo/robot", rclcpp::QoS{1}.reliable());
    tf_publisher_ = create_publisher<tf2_msgs::msg::TFMessage>(
        "/tf", rclcpp::QoS{10}.reliable());
    local_window_publisher_ =
        create_publisher<visualization_msgs::msg::MarkerArray>(
            "/planning_demo/local_window", LatchedQos());
    hud_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/planning_demo/hud", LatchedQos());
    if (show_ground_truth_) {
      ground_truth_publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
          "/planning_demo/ground_truth", LatchedQos());
    }
    odometry_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
        "/planning_demo/odometry", rclcpp::QoS{10}.reliable(),
        [this](const nav_msgs::msg::Odometry::SharedPtr message) {
          std::scoped_lock lock{mutex_};
          pose_ = IncrementalDemoPose{
              .x_m = message->pose.pose.position.x,
              .y_m = message->pose.pose.position.y,
              .yaw_rad = std::atan2(
                  2.0 * (message->pose.pose.orientation.w *
                             message->pose.pose.orientation.z +
                         message->pose.pose.orientation.x *
                             message->pose.pose.orientation.y),
                  1.0 - 2.0 *
                            (message->pose.pose.orientation.y *
                                 message->pose.pose.orientation.y +
                             message->pose.pose.orientation.z *
                             message->pose.pose.orientation.z))};
        });
    path_subscription_ =
        create_subscription<lunar_planning_msgs::msg::PathReference>(
            "/planning_demo/planning/path_reference", LatchedQos(),
            [this](
                const lunar_planning_msgs::msg::PathReference::SharedPtr path) {
              nav_msgs::msg::Path visualization;
              visualization.header.frame_id = "map";
              visualization.header.stamp = now();
              if (path->state ==
                      lunar_planning_msgs::msg::PathReference::ACTIVE &&
                  !path->path.poses.empty()) {
                visualization = path->path;
                visualization.header.frame_id = "map";
              }
              active_path_publisher_->publish(visualization);
            });
    trace_.header.frame_id = "map";
    timer_ = create_wall_timer(200ms, [this] { Publish(); });
  }

 private:
  void Publish() {
    const auto stamp = now();
    IncrementalDemoPose pose;
    {
      std::scoped_lock lock{mutex_};
      if (!pose_) {
        return;
      }
      pose = *pose_;
    }

    auto observation = scenario_.MakeLocalObservation(pose);
    observation.header.stamp = stamp;
    grid_map_publisher_->publish(observation);

    // Publish the synthetic direct map -> odom transform; no other TF chain is
    // needed by either production node in this isolated demo.
    tf2_msgs::msg::TFMessage transforms;
    transforms.transforms.resize(1U);
    auto& transform = transforms.transforms.front();
    transform.header.stamp = stamp;
    transform.header.frame_id = "map";
    transform.child_frame_id = "odom";
    transform.transform.rotation.w = 1.0;
    tf_publisher_->publish(transforms);

    auto local_window = scenario_.MakeLocalWindow(pose);
    for (auto& marker : local_window.markers) {
      marker.header.stamp = stamp;
    }
    local_window_publisher_->publish(local_window);

    visualization_msgs::msg::Marker hud;
    hud.header.stamp = stamp;
    hud.header.frame_id = "map";
    hud.ns = "incremental_demo_hud";
    hud.id = 0;
    hud.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    hud.action = visualization_msgs::msg::Marker::ADD;
    hud.pose.position.x = pose.x_m - scenario_.local_window_size_m() * 0.47;
    hud.pose.position.y = pose.y_m + scenario_.local_window_size_m() * 0.43;
    hud.pose.position.z = 1.0;
    hud.pose.orientation.w = 1.0;
    hud.scale.z = 0.65;
    hud.color.r = 1.0F;
    hud.color.g = 1.0F;
    hud.color.b = 1.0F;
    hud.color.a = 1.0F;
    hud.text = "incremental_v2 | " + platform_type_ + " | r_fine=" +
               std::to_string(scenario_.fine_resolution_m()) +
               " m | local elevation only";
    visualization_msgs::msg::MarkerArray hud_array;
    hud_array.markers.push_back(std::move(hud));
    hud_publisher_->publish(hud_array);

    geometry_msgs::msg::PoseStamped sample;
    sample.header.stamp = stamp;
    sample.header.frame_id = "map";
    sample.pose.position.x = pose.x_m;
    sample.pose.position.y = pose.y_m;
    sample.pose.position.z = platform_type_ == "legged" ? 0.33 : 0.0;
    sample.pose.orientation.z = std::sin(0.5 * pose.yaw_rad);
    sample.pose.orientation.w = std::cos(0.5 * pose.yaw_rad);
    if (trace_.poses.empty() ||
        std::hypot(trace_.poses.back().pose.position.x - pose.x_m,
                   trace_.poses.back().pose.position.y - pose.y_m) >= 0.10) {
      trace_.poses.push_back(sample);
    }
    trace_.header.stamp = stamp;
    trace_publisher_->publish(trace_);

    visualization_msgs::msg::Marker body;
    body.header = sample.header;
    body.ns = "demo_robot";
    body.id = 0;
    body.type = visualization_msgs::msg::Marker::CUBE;
    body.action = visualization_msgs::msg::Marker::ADD;
    body.pose = sample.pose;
    body.pose.position.z = platform_type_ == "legged" ? 0.33 : 0.18;
    body.scale.x = platform_type_ == "legged" ? 0.68 : 1.182;
    body.scale.y = platform_type_ == "legged" ? 0.33 : 0.818;
    body.scale.z = platform_type_ == "legged" ? 0.35 : 0.36;
    body.color.r = 1.0F;
    body.color.g = 1.0F;
    body.color.b = 1.0F;
    body.color.a = 1.0F;
    visualization_msgs::msg::MarkerArray robot;
    robot.markers.push_back(std::move(body));
    robot_publisher_->publish(robot);

    if (ground_truth_publisher_) {
      auto ground_truth = scenario_.MakeGroundTruth();
      ground_truth.header.stamp = stamp;
      ground_truth_publisher_->publish(ground_truth);
    }
  }

  std::string platform_type_;
  bool show_ground_truth_{};
  IncrementalDemoScenario scenario_;
  std::mutex mutex_;
  std::optional<IncrementalDemoPose> pose_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr
      grid_map_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr active_path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trace_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      robot_publisher_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      local_window_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      hud_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr
      ground_truth_publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      odometry_subscription_;
  rclcpp::Subscription<lunar_planning_msgs::msg::PathReference>::SharedPtr
      path_subscription_;
  nav_msgs::msg::Path trace_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace
}  // namespace lunar::incremental_navigation_ros

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<lunar::incremental_navigation_ros::
                                     IncrementalDemoScenarioNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("incremental_demo_scenario"), "%s",
                 error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
