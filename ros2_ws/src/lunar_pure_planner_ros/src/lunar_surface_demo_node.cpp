#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lunar_pure_planner_ros/lunar_surface_scenario.hpp"

namespace lunar::pure_planner_ros {
namespace {

constexpr std::size_t kLocalWidth = 500U;
constexpr std::size_t kLocalHeight = 500U;
constexpr double kLocalResolutionM = 0.2;

std_msgs::msg::Float32MultiArray MakeLayer(const std::vector<float>& values,
                                           const std::size_t width,
                                           const std::size_t height) {
  std_msgs::msg::Float32MultiArray layer;
  layer.layout.dim.resize(2U);
  layer.layout.dim[0].label = "column_index";
  layer.layout.dim[0].size = height;
  layer.layout.dim[0].stride = width * height;
  layer.layout.dim[1].label = "row_index";
  layer.layout.dim[1].size = width;
  layer.layout.dim[1].stride = width;
  layer.data.resize(values.size());
  for (std::size_t y = 0U; y < height; ++y) {
    for (std::size_t x = 0U; x < width; ++x) {
      const std::size_t logical = y * width + x;
      const std::size_t physical = (height - 1U - y) * width + (width - 1U - x);
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
    rover_x_m_ = scenario_.origin_x_m +
                 (static_cast<double>(scenario_.start_cell.x) + 0.5) *
                     scenario_.resolution_m;
    rover_y_m_ = scenario_.origin_y_m +
                 (static_cast<double>(scenario_.start_cell.y) + 0.5) *
                     scenario_.resolution_m;
    path_sub_ = create_subscription<nav_msgs::msg::Path>(
        "/lunar_demo/wheeled_path", rclcpp::QoS{10}.reliable(),
        [this](nav_msgs::msg::Path::ConstSharedPtr path) {
          if (!path->poses.empty()) {
            active_path_ = *path;
            next_path_pose_ = active_path_.poses.size() > 1U ? 1U : 0U;
          }
        });
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
    AdvanceRover();
    const rclcpp::Time stamp = now();
    const double start_x = rover_x_m_;
    const double start_y = rover_y_m_;
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
    // Global terrain and map->odom are static in this test-only scene.
    // Republishing them after rolling has started increments their input
    // sequences and forces an unnecessary global-route rebuild every cycle.
    if (publish_count_ < 12U) {
      global_pub_->publish(global);
    }

    grid_map_msgs::msg::GridMap local;
    local.header.stamp = stamp;
    local.header.frame_id = "odom";
    local.info.resolution = kLocalResolutionM;
    local.info.length_x = static_cast<double>(kLocalWidth) * kLocalResolutionM;
    local.info.length_y = static_cast<double>(kLocalHeight) * kLocalResolutionM;
    local.info.pose.position.x = start_x;
    local.info.pose.position.y = start_y;
    local.info.pose.orientation.w = 1.0;
    local.layers = {"occupancy", "elevation"};
    local.basic_layers = local.layers;
    std::vector<float> occupancy(kLocalWidth * kLocalHeight);
    std::vector<float> elevation(kLocalWidth * kLocalHeight);
    const double local_origin_x = start_x - local.info.length_x / 2.0;
    const double local_origin_y = start_y - local.info.length_y / 2.0;
    for (std::size_t y = 0U; y < kLocalHeight; ++y) {
      for (std::size_t x = 0U; x < kLocalWidth; ++x) {
        const double world_x = local_origin_x +
                               (static_cast<double>(x) + 0.5) * kLocalResolutionM;
        const double world_y = local_origin_y +
                               (static_cast<double>(y) + 0.5) * kLocalResolutionM;
        const auto source_x = static_cast<std::ptrdiff_t>(
            std::floor((world_x - scenario_.origin_x_m) / scenario_.resolution_m));
        const auto source_y = static_cast<std::ptrdiff_t>(
            std::floor((world_y - scenario_.origin_y_m) / scenario_.resolution_m));
        const std::size_t index = y * kLocalWidth + x;
        if (source_x < 0 || source_y < 0 ||
            source_x >= static_cast<std::ptrdiff_t>(scenario_.width) ||
            source_y >= static_cast<std::ptrdiff_t>(scenario_.height)) {
          occupancy[index] = std::numeric_limits<float>::quiet_NaN();
          elevation[index] = std::numeric_limits<float>::quiet_NaN();
          continue;
        }
        const auto source = scenario_.Index(
            {static_cast<std::size_t>(source_x), static_cast<std::size_t>(source_y)});
        occupancy[index] = scenario_.occupancy[source] >= 50 ? 1.0F : 0.0F;
        elevation[index] = scenario_.elevation_m[source];
      }
    }
    local.data = {MakeLayer(occupancy, kLocalWidth, kLocalHeight),
                  MakeLayer(elevation, kLocalWidth, kLocalHeight)};
    local_pub_->publish(local);

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

    if (publish_count_ < 12U) {
      tf2_msgs::msg::TFMessage transforms;
      transforms.transforms.resize(1U);
      auto& transform = transforms.transforms.front();
      transform.header.stamp = stamp;
      transform.header.frame_id = "map";
      transform.child_frame_id = "odom";
      transform.transform.rotation.w = 1.0;
      tf_pub_->publish(transforms);
    }

    const auto goal = Goal(stamp);
    default_goal_pub_->publish(goal);
    // The 1 km map contains one million cells.  Wait for twelve 2 Hz
    // publications before issuing the automatic goal so every planner input
    // has reached its callback cache; the former 1.5 s delay raced startup
    // and produced an INVALID_INPUT result.
    if (++publish_count_ == 12U) {
      goal_pub_->publish(goal);
    }
  }

  void AdvanceRover() {
    constexpr double kStepM = 0.5;
    while (next_path_pose_ < active_path_.poses.size()) {
      const auto& target = active_path_.poses[next_path_pose_].pose.position;
      const double dx = target.x - rover_x_m_;
      const double dy = target.y - rover_y_m_;
      const double distance = std::hypot(dx, dy);
      if (distance <= kStepM) {
        rover_x_m_ = target.x;
        rover_y_m_ = target.y;
        ++next_path_pose_;
        continue;
      }
      rover_x_m_ += kStepM * dx / distance;
      rover_y_m_ += kStepM * dy / distance;
      break;
    }
  }

  LunarSurfaceScenario scenario_;
  double rover_x_m_{};
  double rover_y_m_{};
  nav_msgs::msg::Path active_path_;
  std::size_t next_path_pose_{};
  std::size_t publish_count_{};
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_pub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr local_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr default_goal_pub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
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
