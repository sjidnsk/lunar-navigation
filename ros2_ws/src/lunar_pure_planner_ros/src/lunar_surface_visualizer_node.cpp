#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lunar_pure_planner_ros/lunar_surface_scenario.hpp"
#include "lunar_pure_planner_ros/lunar_surface_traversability_viz.hpp"

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
        "/lunar_demo/rover_markers", rclcpp::QoS{1}.transient_local());
    classic_global_map_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>(
            "/lunar_demo/classic_global_map",
            rclcpp::QoS{1}.reliable().transient_local());
    classic_local_map_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>(
            "/lunar_demo/classic_local_map",
            rclcpp::QoS{1}.reliable().transient_local());
    traversability_viz_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
        "/lunar_demo/traversability_viz",
        rclcpp::QoS{1}.reliable().transient_local());
    pose_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/lunar_demo/pose_markers", rclcpp::QoS{1}.transient_local());
    local_window_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/lunar_demo/local_window", rclcpp::QoS{1}.reliable());
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
          PublishLocalWindow(odometry->pose.pose.position.x,
                             odometry->pose.pose.position.y);
        });
    start_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/lunar_demo/accepted_start", rclcpp::QoS{10}.reliable(),
        [this](geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr start) {
          start_pose_ = start->pose.pose;
          PublishPoseMarkers();
          PublishClassicGlobalMap();
          PublishClassicLocalMap();
        });
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/lunar_demo/rviz_goal", rclcpp::QoS{10}.reliable(),
        [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr goal) {
          goal_pose_ = goal->pose;
          PublishPoseMarkers();
        });
    traversability_sub_ = create_subscription<grid_map_msgs::msg::GridMap>(
        "/lunar_demo/traversability",
        rclcpp::QoS{1}.reliable().transient_local(),
        [this](grid_map_msgs::msg::GridMap::ConstSharedPtr traversability) {
          const auto costmap = MakeTraversabilityCostmap(*traversability);
          if (costmap.has_value()) {
            latest_traversability_costmap_ = *costmap;
            traversability_viz_pub_->publish(*costmap);
            PublishClassicLocalMap();
          }
        });
    global_traversability_sub_ =
        create_subscription<grid_map_msgs::msg::GridMap>(
            "/lunar_demo/global_traversability",
            rclcpp::QoS{1}.reliable().transient_local(),
            [this](grid_map_msgs::msg::GridMap::ConstSharedPtr traversability) {
              const auto costmap = MakeTraversabilityCostmap(*traversability);
              if (!costmap.has_value()) return;
              latest_global_traversability_costmap_ = *costmap;
              PublishClassicGlobalMap();
            });
    local_obstacles_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/lunar_demo/local_map_viz", rclcpp::QoS{1}.reliable(),
        [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr obstacles) {
          latest_local_obstacles_ = *obstacles;
          PublishClassicLocalMap();
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
    rover.scale.x = 5.0; rover.scale.y = 5.0; rover.scale.z = 0.8;
    rover.color.r = 0.80F; rover.color.g = 0.47F; rover.color.b = 0.65F; rover.color.a = 1.0F;
    markers.markers.push_back(std::move(rover));
    visualization_msgs::msg::Marker label;
    label.header.frame_id = "map";
    label.ns = "lunar_surface_rover";
    label.id = 2;
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    label.pose.position.x = x; label.pose.position.y = y + 4.0;
    label.pose.position.z = z + 4.0;
    label.pose.orientation.w = 1.0;
    label.scale.z = 2.0;
    label.color.r = 1.0F; label.color.g = 1.0F; label.color.b = 1.0F; label.color.a = 1.0F;
    label.text = "Current rover position";
    markers.markers.push_back(std::move(label));
    marker_pub_->publish(markers);
  }

  void PublishLocalWindow(const double center_x, const double center_y) {
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker window;
    window.header.frame_id = "map";
    window.ns = "lunar_surface_local_window";
    window.id = 0;
    window.type = visualization_msgs::msg::Marker::LINE_STRIP;
    window.action = visualization_msgs::msg::Marker::ADD;
    window.pose.orientation.w = 1.0;
    window.scale.x = 0.8;
    window.color.r = 0.34F;
    window.color.g = 0.71F;
    window.color.b = 0.91F;
    window.color.a = 1.0F;
    constexpr double kHalfLengthM = 32.0;
    for (const auto [x, y] : {
             std::pair{center_x - kHalfLengthM, center_y - kHalfLengthM},
             std::pair{center_x + kHalfLengthM, center_y - kHalfLengthM},
             std::pair{center_x + kHalfLengthM, center_y + kHalfLengthM},
             std::pair{center_x - kHalfLengthM, center_y + kHalfLengthM},
             std::pair{center_x - kHalfLengthM, center_y - kHalfLengthM}}) {
      geometry_msgs::msg::Point point;
      point.x = x;
      point.y = y;
      point.z = 0.15;
      window.points.push_back(point);
    }
    markers.markers.push_back(std::move(window));
    local_window_pub_->publish(markers);
  }

 private:
  void PublishMarkers() {
    visualization_msgs::msg::MarkerArray markers;
    const auto start_index = scenario_.Index(scenario_.start_cell);
    const double start_x = scenario_.origin_x_m +
                           (static_cast<double>(scenario_.start_cell.x) + 0.5) *
                               scenario_.resolution_m;
    const double start_y = scenario_.origin_y_m +
                           (static_cast<double>(scenario_.start_cell.y) + 0.5) *
                               scenario_.resolution_m;
    const double start_z = scenario_.elevation_m[start_index] + 0.25;
    geometry_msgs::msg::Pose initial_start;
    initial_start.position.x = start_x;
    initial_start.position.y = start_y;
    initial_start.position.z = start_z;
    initial_start.orientation.w = 1.0;
    start_pose_ = initial_start;
    geometry_msgs::msg::Pose initial_goal;
    initial_goal.position.x = scenario_.origin_x_m +
        (static_cast<double>(scenario_.default_goal_cell.x) + 0.5) *
            scenario_.resolution_m;
    initial_goal.position.y = scenario_.origin_y_m +
        (static_cast<double>(scenario_.default_goal_cell.y) + 0.5) *
            scenario_.resolution_m;
    initial_goal.orientation.w = 1.0;
    goal_pose_ = initial_goal;

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
    rover.scale.x = 5.0;
    rover.scale.y = 5.0;
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
    rover_label.pose.position.y = start_y + 4.0;
    rover_label.pose.position.z = start_z + 4.0;
    rover_label.pose.orientation.w = 1.0;
    rover_label.scale.z = 2.0;
    rover_label.color.r = 1.0F;
    rover_label.color.g = 1.0F;
    rover_label.color.b = 1.0F;
    rover_label.color.a = 1.0F;
    rover_label.text = "Current rover position";
    markers.markers.push_back(std::move(rover_label));

    marker_pub_->publish(markers);
    classic_global_map_pub_->publish(
        MakeClassicGlobalObstacleMarkers(scenario_));
    PublishPoseMarkers();
    PublishLocalWindow(start_x, start_y);
  }

  void PublishClassicLocalMap() {
    if (!latest_local_obstacles_.has_value() ||
        !latest_traversability_costmap_.has_value() ||
        !start_pose_.has_value()) {
      return;
    }
    const auto markers = MakeClassicLocalOverlayMarkers(
        *latest_local_obstacles_, *latest_traversability_costmap_,
        start_pose_->position.x, start_pose_->position.y,
        ClassicMapBaseZ(scenario_));
    if (markers.has_value()) classic_local_map_pub_->publish(*markers);
  }

  void PublishClassicGlobalMap() {
    if (!latest_global_traversability_costmap_.has_value() ||
        !start_pose_.has_value()) {
      return;
    }
    const auto markers = MakeClassicGlobalTraversabilityMarkers(
        scenario_, *latest_global_traversability_costmap_,
        start_pose_->position.x, start_pose_->position.y);
    if (markers.has_value()) classic_global_map_pub_->publish(*markers);
  }

  void PublishPoseMarkers() {
    visualization_msgs::msg::MarkerArray markers;
    const auto add_pose = [&markers](const geometry_msgs::msg::Pose& pose,
                                     const int id, const std::string& text,
                                     const float red, const float green,
                                     const float blue) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "map";
      marker.ns = "lunar_surface_start_goal";
      marker.id = id;
      marker.type = visualization_msgs::msg::Marker::ARROW;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose = pose;
      marker.pose.position.z += 1.0;
      marker.scale.x = 6.0;
      marker.scale.y = 2.0;
      marker.scale.z = 2.0;
      marker.color.r = red;
      marker.color.g = green;
      marker.color.b = blue;
      marker.color.a = 1.0F;
      markers.markers.push_back(marker);

      marker.id = id + 1;
      marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      marker.pose.position.y -= 4.0;
      marker.pose.position.z += 4.0;
      marker.scale.x = 0.0;
      marker.scale.y = 0.0;
      marker.scale.z = 3.0;
      marker.text = text;
      markers.markers.push_back(std::move(marker));
    };
    if (start_pose_.has_value()) {
      add_pose(*start_pose_, 10, "START", 0.0F, 0.45F, 0.70F);
    }
    if (goal_pose_.has_value()) {
      add_pose(*goal_pose_, 20, "GOAL", 0.90F, 0.62F, 0.0F);
    }
    pose_marker_pub_->publish(markers);
  }

  LunarSurfaceScenario scenario_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      classic_global_map_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      classic_local_map_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr traversability_viz_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      pose_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      local_window_pub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      start_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr traversability_sub_;
  rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr
      global_traversability_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
      local_obstacles_sub_;
  std::optional<nav_msgs::msg::OccupancyGrid> latest_local_obstacles_;
  std::optional<nav_msgs::msg::OccupancyGrid> latest_traversability_costmap_;
  std::optional<nav_msgs::msg::OccupancyGrid>
      latest_global_traversability_costmap_;
  std::optional<geometry_msgs::msg::Pose> start_pose_;
  std::optional<geometry_msgs::msg::Pose> goal_pose_;
};

}  // namespace
}  // namespace lunar::pure_planner_ros

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lunar::pure_planner_ros::LunarSurfaceVisualizerNode>());
  rclcpp::shutdown();
  return 0;
}
