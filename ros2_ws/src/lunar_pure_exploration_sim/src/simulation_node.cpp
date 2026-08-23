#include "lunar_pure_exploration_sim/simulation_node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "lunar_pure_exploration_sim/map_messages.hpp"

namespace lunar::pure_exploration_sim {
namespace {

constexpr double kMaximumWallStepS = 0.1;
constexpr double kFrozenSensorRangeM = 10.0;
constexpr double kFrozenSensorFovDeg = 90.0;
constexpr double kContractTolerance = 1.0e-12;

geometry_msgs::msg::Quaternion YawQuaternion(double yaw_rad) {
  geometry_msgs::msg::Quaternion quaternion;
  quaternion.z = std::sin(yaw_rad / 2.0);
  quaternion.w = std::cos(yaw_rad / 2.0);
  return quaternion;
}

geometry_msgs::msg::Point Point(double x, double y, double z = 0.0) {
  geometry_msgs::msg::Point point;
  point.x = x;
  point.y = y;
  point.z = z;
  return point;
}

rclcpp::QoS ReliableStateQos() {
  return rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
}

rclcpp::QoS ReliableStreamQos() {
  return rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
}

bool FiniteCommand(const geometry_msgs::msg::Twist& command) {
  return std::isfinite(command.linear.x) && std::isfinite(command.angular.z);
}

}  // namespace

SimulationNode::SimulationNode() : SimulationNode(rclcpp::NodeOptions{}, true) {}

SimulationNode::SimulationNode(const rclcpp::NodeOptions& options,
                               bool start_timer)
    : Node("lunar_pure_exploration_sim", options),
      scene_(BuildLunarScene(static_cast<std::uint32_t>(
          declare_parameter<std::int64_t>("seed", 20260824)))),
      last_wall_tick_(std::chrono::steady_clock::now()) {
  plant_parameters_.speed_multiplier =
      declare_parameter<double>("speed_multiplier", 20.0);
  sensor_model_.range_m = declare_parameter<double>("sensor_range_m", 10.0);
  const double sensor_fov_deg =
      declare_parameter<double>("sensor_fov_deg", 90.0);
  update_rate_hz_ = declare_parameter<double>("update_rate_hz", 20.0);
  if (!std::isfinite(plant_parameters_.speed_multiplier) ||
      plant_parameters_.speed_multiplier <= 0.0) {
    throw std::invalid_argument(
        "speed_multiplier must be finite and positive");
  }
  if (!std::isfinite(sensor_model_.range_m) ||
      std::abs(sensor_model_.range_m - kFrozenSensorRangeM) >
          kContractTolerance) {
    throw std::invalid_argument("sensor_range_m must equal 10.0");
  }
  if (!std::isfinite(sensor_fov_deg) ||
      std::abs(sensor_fov_deg - kFrozenSensorFovDeg) > kContractTolerance) {
    throw std::invalid_argument("sensor_fov_deg must equal 90.0");
  }
  if (!std::isfinite(update_rate_hz_) || update_rate_hz_ <= 0.0) {
    throw std::invalid_argument("update_rate_hz must be finite and positive");
  }
  sensor_model_.horizontal_fov_rad =
      sensor_fov_deg * std::numbers::pi / 180.0;

  const std::string command_topic = declare_parameter<std::string>(
      "command_topic", "/Car/T5/Car_Cmd_Vel");
  const std::string global_topic = declare_parameter<std::string>(
      "global_overview_topic", "/Car/T3/mapping/global_overview");
  const std::string local_topic = declare_parameter<std::string>(
      "local_grid_map_topic", "/Car/T3/mapping/grid_map");
  const std::string odometry_topic = declare_parameter<std::string>(
      "odometry_topic", "/Car/T3/localization/odometry");
  const std::string tf_topic = declare_parameter<std::string>("tf_topic", "/tf");
  const std::string fov_topic = declare_parameter<std::string>(
      "sensor_fov_topic", "/Car/T4/simulation/sensor_fov");
  const std::string vehicle_topic = declare_parameter<std::string>(
      "vehicle_markers_topic", "/Car/T4/simulation/vehicle_markers");
  const std::string local_markers_topic = declare_parameter<std::string>(
      "local_map_markers_topic", "/Car/T4/simulation/local_map_markers");
  const std::string path_topic = declare_parameter<std::string>(
      "actual_path_topic", "/Car/T4/simulation/actual_path");
  const std::string elapsed_topic = declare_parameter<std::string>(
      "sim_elapsed_topic", "/Car/T4/simulation/sim_elapsed");

  command_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      command_topic, ReliableStreamQos(),
      std::bind(&SimulationNode::SetCommand, this, std::placeholders::_1));
  global_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      global_topic, ReliableStateQos());
  local_pub_ = create_publisher<grid_map_msgs::msg::GridMap>(
      local_topic, ReliableStreamQos());
  odometry_pub_ = create_publisher<nav_msgs::msg::Odometry>(
      odometry_topic, ReliableStreamQos());
  tf_pub_ = create_publisher<tf2_msgs::msg::TFMessage>(
      tf_topic, ReliableStreamQos());
  fov_pub_ = create_publisher<visualization_msgs::msg::Marker>(
      fov_topic, ReliableStateQos());
  vehicle_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      vehicle_topic, ReliableStateQos());
  local_markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      local_markers_topic, ReliableStateQos());
  path_pub_ = create_publisher<nav_msgs::msg::Path>(
      path_topic, ReliableStateQos());
  elapsed_pub_ = create_publisher<std_msgs::msg::Float64>(
      elapsed_topic, ReliableStateQos());

  observations_.Observe(scene_, Pose2{}, sensor_model_);
  BuildMessages(now());
  PublishLatest();

  if (start_timer) {
    const auto period = std::chrono::duration<double>(1.0 / update_rate_hz_);
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&SimulationNode::OnTimer, this));
  }
}

void SimulationNode::Tick(double wall_dt_s) {
  const auto computation_start = std::chrono::steady_clock::now();
  const double safe_wall_dt_s =
      std::isfinite(wall_dt_s) && wall_dt_s > 0.0
          ? std::min(wall_dt_s, kMaximumWallStepS)
          : 0.0;
  plant_state_ =
      StepPlant(plant_state_, command_, safe_wall_dt_s, plant_parameters_);
  const Pose2 pose{plant_state_.x_m, plant_state_.y_m, plant_state_.yaw_rad};
  observations_.Observe(scene_, pose, sensor_model_);
  BuildMessages(now());
  PublishLatest();

  last_tick_duration_s_ = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() -
                              computation_start)
                              .count();
  max_tick_duration_s_ =
      std::max(max_tick_duration_s_, last_tick_duration_s_);
  ++tick_count_;
  if (last_tick_duration_s_ > deadline_period_s()) {
    ++deadline_miss_count_;
  }
}

void SimulationNode::SetCommand(const geometry_msgs::msg::Twist& command) {
  command_ = FiniteCommand(command)
                 ? BodyCommand{command.linear.x, command.angular.z}
                 : BodyCommand{};
}

const SimulationMessages& SimulationNode::latest_messages() const noexcept {
  return messages_;
}

std::size_t SimulationNode::known_global_count() const noexcept {
  return observations_.KnownGlobalCount();
}

std::uint64_t SimulationNode::tick_count() const noexcept {
  return tick_count_;
}

std::uint64_t SimulationNode::deadline_miss_count() const noexcept {
  return deadline_miss_count_;
}

double SimulationNode::last_tick_duration_s() const noexcept {
  return last_tick_duration_s_;
}

double SimulationNode::max_tick_duration_s() const noexcept {
  return max_tick_duration_s_;
}

double SimulationNode::deadline_period_s() const noexcept {
  return 1.0 / update_rate_hz_;
}

void SimulationNode::OnTimer() {
  const auto current_wall_tick = std::chrono::steady_clock::now();
  const double wall_dt_s =
      std::chrono::duration<double>(current_wall_tick - last_wall_tick_).count();
  last_wall_tick_ = current_wall_tick;
  Tick(wall_dt_s);
}

void SimulationNode::BuildMessages(const rclcpp::Time& stamp) {
  const Pose2 pose{plant_state_.x_m, plant_state_.y_m, plant_state_.yaw_rad};
  messages_.global_overview = MakeGlobalOverview(scene_, observations_, stamp);
  messages_.local_grid_map =
      MakeLocalGridMap(scene_, observations_, pose, stamp);

  messages_.odometry.header.stamp = stamp;
  messages_.odometry.header.frame_id = "odom";
  messages_.odometry.child_frame_id = "base_link";
  messages_.odometry.pose.pose.position.x = plant_state_.x_m;
  messages_.odometry.pose.pose.position.y = plant_state_.y_m;
  messages_.odometry.pose.pose.orientation = YawQuaternion(plant_state_.yaw_rad);
  messages_.odometry.twist.twist.linear.x =
      plant_state_.longitudinal_velocity_mps;
  messages_.odometry.twist.twist.linear.y = 0.0;
  messages_.odometry.twist.twist.angular.z = plant_state_.yaw_rate_rps;

  geometry_msgs::msg::TransformStamped map_to_odom;
  map_to_odom.header.stamp = stamp;
  map_to_odom.header.frame_id = "map";
  map_to_odom.child_frame_id = "odom";
  map_to_odom.transform.rotation.w = 1.0;
  geometry_msgs::msg::TransformStamped odom_to_base;
  odom_to_base.header.stamp = stamp;
  odom_to_base.header.frame_id = "odom";
  odom_to_base.child_frame_id = "base_link";
  odom_to_base.transform.translation.x = plant_state_.x_m;
  odom_to_base.transform.translation.y = plant_state_.y_m;
  odom_to_base.transform.rotation = YawQuaternion(plant_state_.yaw_rad);
  messages_.transforms.transforms = {map_to_odom, odom_to_base};

  messages_.sensor_fov.header.stamp = stamp;
  messages_.sensor_fov.header.frame_id = "odom";
  messages_.sensor_fov.ns = "sensor_fov";
  messages_.sensor_fov.id = 0;
  messages_.sensor_fov.type = visualization_msgs::msg::Marker::LINE_STRIP;
  messages_.sensor_fov.action = visualization_msgs::msg::Marker::ADD;
  messages_.sensor_fov.pose.orientation.w = 1.0;
  messages_.sensor_fov.scale.x = 0.08;
  messages_.sensor_fov.color.r = 0.95F;
  messages_.sensor_fov.color.g = 0.75F;
  messages_.sensor_fov.color.b = 0.10F;
  messages_.sensor_fov.color.a = 1.0F;
  messages_.sensor_fov.points.clear();
  messages_.sensor_fov.points.push_back(Point(plant_state_.x_m, plant_state_.y_m));
  constexpr std::size_t kFovArcSegments = 24U;
  for (std::size_t index = 0U; index <= kFovArcSegments; ++index) {
    const double fraction = static_cast<double>(index) /
                            static_cast<double>(kFovArcSegments);
    const double angle = plant_state_.yaw_rad -
                         sensor_model_.horizontal_fov_rad / 2.0 +
                         fraction * sensor_model_.horizontal_fov_rad;
    messages_.sensor_fov.points.push_back(
        Point(plant_state_.x_m + sensor_model_.range_m * std::cos(angle),
              plant_state_.y_m + sensor_model_.range_m * std::sin(angle)));
  }
  messages_.sensor_fov.points.push_back(Point(plant_state_.x_m, plant_state_.y_m));

  messages_.vehicle_markers.markers.clear();
  visualization_msgs::msg::Marker body;
  body.header.stamp = stamp;
  body.header.frame_id = "odom";
  body.ns = "vehicle";
  body.id = 0;
  body.type = visualization_msgs::msg::Marker::CUBE;
  body.action = visualization_msgs::msg::Marker::ADD;
  body.pose.position.x = plant_state_.x_m;
  body.pose.position.y = plant_state_.y_m;
  body.pose.position.z = 0.25;
  body.pose.orientation = YawQuaternion(plant_state_.yaw_rad);
  body.scale.x = plant_parameters_.wheelbase_m;
  body.scale.y = plant_parameters_.track_width_m;
  body.scale.z = 0.25;
  body.color.r = 0.25F;
  body.color.g = 0.55F;
  body.color.b = 0.95F;
  body.color.a = 1.0F;
  messages_.vehicle_markers.markers.push_back(body);

  const std::array<double, 4U> body_x{
      plant_parameters_.wheelbase_m / 2.0,
      plant_parameters_.wheelbase_m / 2.0,
      -plant_parameters_.wheelbase_m / 2.0,
      -plant_parameters_.wheelbase_m / 2.0};
  const std::array<double, 4U> body_y{
      plant_parameters_.track_width_m / 2.0,
      -plant_parameters_.track_width_m / 2.0,
      plant_parameters_.track_width_m / 2.0,
      -plant_parameters_.track_width_m / 2.0};
  const std::array<double, 4U> wheel_angle{
      plant_state_.wheel_angles.front_left_rad,
      plant_state_.wheel_angles.front_right_rad,
      plant_state_.wheel_angles.rear_left_rad,
      plant_state_.wheel_angles.rear_right_rad};
  for (std::size_t index = 0U; index < body_x.size(); ++index) {
    visualization_msgs::msg::Marker wheel = body;
    wheel.id = static_cast<int>(index + 1U);
    wheel.pose.position.x = plant_state_.x_m +
                            body_x[index] * std::cos(plant_state_.yaw_rad) -
                            body_y[index] * std::sin(plant_state_.yaw_rad);
    wheel.pose.position.y = plant_state_.y_m +
                            body_x[index] * std::sin(plant_state_.yaw_rad) +
                            body_y[index] * std::cos(plant_state_.yaw_rad);
    wheel.pose.position.z = 0.14;
    wheel.pose.orientation =
        YawQuaternion(plant_state_.yaw_rad + wheel_angle[index]);
    wheel.scale.x = 0.28;
    wheel.scale.y = 0.10;
    wheel.scale.z = 0.14;
    wheel.color.r = 0.10F;
    wheel.color.g = 0.10F;
    wheel.color.b = 0.10F;
    messages_.vehicle_markers.markers.push_back(wheel);
  }

  messages_.local_map_markers.markers.clear();
  visualization_msgs::msg::Marker window;
  window.header.stamp = stamp;
  window.header.frame_id = "odom";
  window.ns = "local_map_window";
  window.id = 0;
  window.type = visualization_msgs::msg::Marker::LINE_STRIP;
  window.action = visualization_msgs::msg::Marker::ADD;
  window.pose.orientation.w = 1.0;
  window.scale.x = 0.15;
  window.color.r = 0.15F;
  window.color.g = 0.90F;
  window.color.b = 0.95F;
  window.color.a = 1.0F;
  constexpr double kLocalHalfLengthM = 32.0;
  const double minimum_x = plant_state_.x_m - kLocalHalfLengthM;
  const double maximum_x = plant_state_.x_m + kLocalHalfLengthM;
  const double minimum_y = plant_state_.y_m - kLocalHalfLengthM;
  const double maximum_y = plant_state_.y_m + kLocalHalfLengthM;
  window.points = {Point(minimum_x, minimum_y, 0.05),
                   Point(maximum_x, minimum_y, 0.05),
                   Point(maximum_x, maximum_y, 0.05),
                   Point(minimum_x, maximum_y, 0.05),
                   Point(minimum_x, minimum_y, 0.05)};
  messages_.local_map_markers.markers.push_back(std::move(window));

  visualization_msgs::msg::Marker elevation;
  elevation.header.stamp = stamp;
  elevation.header.frame_id = "odom";
  elevation.ns = "local_elevation";
  elevation.id = 1;
  elevation.type = visualization_msgs::msg::Marker::POINTS;
  elevation.action = visualization_msgs::msg::Marker::ADD;
  elevation.pose.orientation.w = 1.0;
  elevation.scale.x = 0.35;
  elevation.scale.y = 0.35;
  elevation.color.r = 0.72F;
  elevation.color.g = 0.70F;
  elevation.color.b = 0.62F;
  elevation.color.a = 0.90F;
  constexpr std::size_t kLocalCells = 320U;
  constexpr std::size_t kVisualizationStride = 2U;
  constexpr double kLocalResolutionM = 0.2;
  for (std::size_t logical_y = 0U; logical_y < kLocalCells;
       logical_y += kVisualizationStride) {
    for (std::size_t logical_x = 0U; logical_x < kLocalCells;
         logical_x += kVisualizationStride) {
      const TruthSample* sample =
          observations_.CurrentLocalSample(pose, logical_x, logical_y);
      if (sample == nullptr) {
        continue;
      }
      elevation.points.push_back(Point(
          minimum_x + (static_cast<double>(logical_x) + 0.5) *
                          kLocalResolutionM,
          minimum_y + (static_cast<double>(logical_y) + 0.5) *
                          kLocalResolutionM,
          sample->elevation_m + 0.08));
    }
  }
  messages_.local_map_markers.markers.push_back(std::move(elevation));

  messages_.actual_path.header.stamp = stamp;
  messages_.actual_path.header.frame_id = "map";
  geometry_msgs::msg::PoseStamped path_pose;
  path_pose.header = messages_.actual_path.header;
  path_pose.pose = messages_.odometry.pose.pose;
  messages_.actual_path.poses.push_back(path_pose);

  messages_.sim_elapsed.data = plant_state_.simulated_elapsed_s;
}

void SimulationNode::PublishLatest() {
  global_pub_->publish(messages_.global_overview);
  local_pub_->publish(messages_.local_grid_map);
  odometry_pub_->publish(messages_.odometry);
  tf_pub_->publish(messages_.transforms);
  fov_pub_->publish(messages_.sensor_fov);
  vehicle_pub_->publish(messages_.vehicle_markers);
  local_markers_pub_->publish(messages_.local_map_markers);
  path_pub_->publish(messages_.actual_path);
  elapsed_pub_->publish(messages_.sim_elapsed);
}

}  // namespace lunar::pure_exploration_sim
