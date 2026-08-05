#include "lunar_nav2_adapter/lunar_global_planner.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stop_token>
#include <stdexcept>
#include <thread>
#include <utility>

#include <nav2_core/exceptions.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp_action/create_client.hpp>
#include <tf2/time.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>

namespace lunar::planning::nav2 {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using ActionClient = rclcpp_action::Client<Action>;

template<typename Value>
Value DeclareOrGetParameter(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
    const std::string& name,
    const Value& default_value) {
  if (!node->has_parameter(name)) {
    return node->declare_parameter<Value>(name, default_value);
  }
  return node->get_parameter(name).get_value<Value>();
}

void RequirePositiveFinite(const double value, const std::string& name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw nav2_core::PlannerException{name + " must be finite and positive"};
  }
}

std::chrono::nanoseconds ToTimeout(const double seconds) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>{seconds});
}

std::string ResultCodeName(const rclcpp_action::ResultCode code) {
  switch (code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      return "SUCCEEDED";
    case rclcpp_action::ResultCode::ABORTED:
      return "ABORTED";
    case rclcpp_action::ResultCode::CANCELED:
      return "CANCELED";
    case rclcpp_action::ResultCode::UNKNOWN:
      return "UNKNOWN";
  }
  return "INVALID";
}

}  // namespace

struct LunarGlobalPlanner::Impl final {
  mutable std::mutex mutex;
  rclcpp_lifecycle::LifecycleNode::SharedPtr node;
  std::shared_ptr<tf2_ros::Buffer> tf;
  rclcpp::CallbackGroup::SharedPtr callback_group;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> io_executor;
  std::jthread io_thread;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      odometry_subscription;
  ActionClient::SharedPtr action_client;
  std::optional<nav_msgs::msg::Odometry> latest_odometry;
  std::atomic<std::uint64_t> request_sequence{0U};
  std::string plugin_name;
  std::string action_name;
  std::string mission_id;
  std::string global_frame{"map"};
  std::uint64_t mission_revision{0U};
  double start_tolerance_m{0.0};
  double goal_tolerance_m{0.0};
  double yaw_tolerance_rad{0.0};
  double transform_timeout_s{0.0};
  std::chrono::nanoseconds action_server_timeout{0};
  std::chrono::nanoseconds action_result_timeout{0};
  bool replace_active_request{false};
  bool configured{false};
  bool active{false};
};

LunarGlobalPlanner::LunarGlobalPlanner() : impl_(std::make_unique<Impl>()) {}

LunarGlobalPlanner::~LunarGlobalPlanner() {
  try {
    cleanup();
  } catch (...) {
    // Destructors must not propagate executor teardown failures.
  }
}

void LunarGlobalPlanner::configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) {
  auto node = parent.lock();
  if (!node) {
    throw nav2_core::PlannerException{"Nav2 lifecycle parent expired"};
  }
  if (name.empty()) {
    throw nav2_core::PlannerException{"Nav2 planner plugin name is empty"};
  }

  std::scoped_lock lock{impl_->mutex};
  if (impl_->configured) {
    throw nav2_core::PlannerException{"LunarGlobalPlanner is already configured"};
  }

  const std::string prefix = name + ".";
  const std::string global_frame = costmap_ros
      ? costmap_ros->getGlobalFrameID() : std::string{"map"};
  const auto action_name = DeclareOrGetParameter<std::string>(
      node, prefix + "action_name", "/plan_motion");
  const auto odometry_topic = DeclareOrGetParameter<std::string>(
      node, prefix + "odometry_topic", "/localization/odometry");
  const auto mission_id = DeclareOrGetParameter<std::string>(
      node, prefix + "mission_id", "");
  const auto mission_revision_parameter = DeclareOrGetParameter<std::int64_t>(
      node, prefix + "mission_revision", 0);
  const auto start_tolerance_m = DeclareOrGetParameter<double>(
      node, prefix + "start_tolerance_m", 0.25);
  const auto goal_tolerance_m = DeclareOrGetParameter<double>(
      node, prefix + "goal_tolerance_m", 0.25);
  const auto yaw_tolerance_rad = DeclareOrGetParameter<double>(
      node, prefix + "yaw_tolerance_rad", 0.20);
  const auto transform_timeout_s = DeclareOrGetParameter<double>(
      node, prefix + "transform_timeout_s", 0.20);
  const auto action_server_timeout_s = DeclareOrGetParameter<double>(
      node, prefix + "action_server_timeout_s", 2.0);
  const auto action_result_timeout_s = DeclareOrGetParameter<double>(
      node, prefix + "action_result_timeout_s", 10.0);
  const auto replace_active_request = DeclareOrGetParameter<bool>(
      node, prefix + "replace_active_request", false);

  if (action_name.empty() || odometry_topic.empty() || global_frame.empty()) {
    throw nav2_core::PlannerException{
        "action_name, odometry_topic, and global frame must be non-empty"};
  }
  if (mission_id.empty() || mission_revision_parameter <= 0) {
    throw nav2_core::PlannerException{
        "mission_id must be non-empty and mission_revision must be positive"};
  }
  RequirePositiveFinite(start_tolerance_m, prefix + "start_tolerance_m");
  RequirePositiveFinite(goal_tolerance_m, prefix + "goal_tolerance_m");
  RequirePositiveFinite(yaw_tolerance_rad, prefix + "yaw_tolerance_rad");
  RequirePositiveFinite(transform_timeout_s, prefix + "transform_timeout_s");
  RequirePositiveFinite(
      action_server_timeout_s, prefix + "action_server_timeout_s");
  RequirePositiveFinite(
      action_result_timeout_s, prefix + "action_result_timeout_s");

  auto callback_group = node->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive,
      false);
  rclcpp::SubscriptionOptions subscription_options;
  subscription_options.callback_group = callback_group;
  auto odometry_subscription = node->create_subscription<nav_msgs::msg::Odometry>(
      odometry_topic,
      rclcpp::SensorDataQoS{},
      [this](const nav_msgs::msg::Odometry::SharedPtr message) {
        std::scoped_lock callback_lock{impl_->mutex};
        impl_->latest_odometry = *message;
      },
      subscription_options);
  auto action_client = rclcpp_action::create_client<Action>(
      node, action_name, callback_group);
  rclcpp::ExecutorOptions executor_options;
  executor_options.context = node->get_node_base_interface()->get_context();
  auto io_executor =
      std::make_unique<rclcpp::executors::SingleThreadedExecutor>(
          executor_options);
  io_executor->add_callback_group(
      callback_group, node->get_node_base_interface());

  impl_->node = std::move(node);
  impl_->tf = std::move(tf);
  impl_->callback_group = std::move(callback_group);
  impl_->io_executor = std::move(io_executor);
  impl_->odometry_subscription = std::move(odometry_subscription);
  impl_->action_client = std::move(action_client);
  impl_->plugin_name = std::move(name);
  impl_->action_name = action_name;
  impl_->mission_id = mission_id;
  impl_->global_frame = global_frame;
  impl_->mission_revision =
      static_cast<std::uint64_t>(mission_revision_parameter);
  impl_->start_tolerance_m = start_tolerance_m;
  impl_->goal_tolerance_m = goal_tolerance_m;
  impl_->yaw_tolerance_rad = yaw_tolerance_rad;
  impl_->transform_timeout_s = transform_timeout_s;
  impl_->action_server_timeout = ToTimeout(action_server_timeout_s);
  impl_->action_result_timeout = ToTimeout(action_result_timeout_s);
  impl_->replace_active_request = replace_active_request;
  impl_->configured = true;
  impl_->active = false;
  auto* const io_executor_pointer = impl_->io_executor.get();
  const auto io_context = impl_->node->get_node_base_interface()->get_context();
  impl_->io_thread = std::jthread(
      [io_executor_pointer, io_context](const std::stop_token stop_token) {
        while (!stop_token.stop_requested() && rclcpp::ok(io_context)) {
          io_executor_pointer->spin_once(std::chrono::milliseconds{100});
        }
      });
}

void LunarGlobalPlanner::cleanup() {
  if (!impl_) {
    return;
  }
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> io_executor;
  std::jthread io_thread;
  rclcpp::CallbackGroup::SharedPtr callback_group;
  ActionClient::SharedPtr action_client;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      odometry_subscription;
  {
    std::scoped_lock lock{impl_->mutex};
    impl_->active = false;
    impl_->configured = false;
    io_executor = std::move(impl_->io_executor);
    io_thread = std::move(impl_->io_thread);
    callback_group = std::move(impl_->callback_group);
    action_client = std::move(impl_->action_client);
    odometry_subscription = std::move(impl_->odometry_subscription);
    impl_->tf.reset();
    impl_->node.reset();
    impl_->latest_odometry.reset();
  }
  io_thread.request_stop();
  if (io_executor) {
    io_executor->cancel();
  }
  if (io_thread.joinable()) {
    io_thread.join();
  }
  if (io_executor && callback_group) {
    io_executor->remove_callback_group(callback_group);
  }
}

void LunarGlobalPlanner::activate() {
  std::scoped_lock lock{impl_->mutex};
  if (!impl_->configured) {
    throw nav2_core::PlannerException{
        "LunarGlobalPlanner must be configured before activation"};
  }
  impl_->active = true;
}

void LunarGlobalPlanner::deactivate() {
  std::scoped_lock lock{impl_->mutex};
  impl_->active = false;
}

nav_msgs::msg::Path LunarGlobalPlanner::createPlan(
    const geometry_msgs::msg::PoseStamped& start,
    const geometry_msgs::msg::PoseStamped& goal) {
  ActionClient::SharedPtr action_client;
  std::shared_ptr<tf2_ros::Buffer> tf;
  std::optional<nav_msgs::msg::Odometry> odometry;
  std::string plugin_name;
  std::string action_name;
  std::string mission_id;
  std::uint64_t mission_revision = 0U;
  double start_tolerance_m = 0.0;
  double goal_tolerance_m = 0.0;
  double yaw_tolerance_rad = 0.0;
  double transform_timeout_s = 0.0;
  std::chrono::nanoseconds action_server_timeout{0};
  std::chrono::nanoseconds action_result_timeout{0};
  bool replace_active_request = false;
  {
    std::scoped_lock lock{impl_->mutex};
    if (!impl_->configured || !impl_->active) {
      throw nav2_core::PlannerException{
          "LunarGlobalPlanner is not active"};
    }
    action_client = impl_->action_client;
    tf = impl_->tf;
    odometry = impl_->latest_odometry;
    plugin_name = impl_->plugin_name;
    action_name = impl_->action_name;
    mission_id = impl_->mission_id;
    mission_revision = impl_->mission_revision;
    start_tolerance_m = impl_->start_tolerance_m;
    goal_tolerance_m = impl_->goal_tolerance_m;
    yaw_tolerance_rad = impl_->yaw_tolerance_rad;
    transform_timeout_s = impl_->transform_timeout_s;
    action_server_timeout = impl_->action_server_timeout;
    action_result_timeout = impl_->action_result_timeout;
    replace_active_request = impl_->replace_active_request;
  }

  if (!odometry.has_value()) {
    throw nav2_core::PlannerException{
        "latest odometry is unavailable"};
  }
  if (start.header.frame_id.empty() || odometry->header.frame_id.empty()) {
    throw nav2_core::PlannerException{
        "start and odometry frames must be non-empty"};
  }

  geometry_msgs::msg::PoseStamped odometry_pose;
  odometry_pose.header = odometry->header;
  odometry_pose.pose = odometry->pose.pose;
  if (odometry_pose.header.frame_id != start.header.frame_id) {
    if (!tf) {
      throw nav2_core::PlannerException{
          "TF buffer is unavailable for start pose validation"};
    }
    try {
      odometry_pose = tf->transform(
          odometry_pose,
          start.header.frame_id,
          tf2::durationFromSec(transform_timeout_s));
    } catch (const tf2::TransformException& error) {
      throw nav2_core::PlannerException{
          std::string{"cannot transform odometry into start frame: "} +
          error.what()};
    }
  }

  const double dx = start.pose.position.x - odometry_pose.pose.position.x;
  const double dy = start.pose.position.y - odometry_pose.pose.position.y;
  const double distance = std::hypot(dx, dy);
  if (!std::isfinite(distance) || distance > start_tolerance_m) {
    std::ostringstream detail;
    detail << "Nav2 start differs from latest odometry by " << distance
           << " m; tolerance is " << start_tolerance_m << " m";
    throw nav2_core::PlannerException{detail.str()};
  }
  if (goal.header.frame_id.empty() ||
      !std::isfinite(goal.pose.position.x) ||
      !std::isfinite(goal.pose.position.y) ||
      !std::isfinite(goal.pose.position.z)) {
    throw nav2_core::PlannerException{"Nav2 goal is not finite or has no frame"};
  }

  const auto& orientation = goal.pose.orientation;
  const double orientation_norm = std::sqrt(
      orientation.x * orientation.x + orientation.y * orientation.y +
      orientation.z * orientation.z + orientation.w * orientation.w);
  if (!std::isfinite(orientation_norm) ||
      orientation_norm <= std::numeric_limits<double>::epsilon()) {
    throw nav2_core::PlannerException{"Nav2 goal orientation is invalid"};
  }

  const auto sequence = impl_->request_sequence.fetch_add(1U);
  const std::string request_id =
      plugin_name + "-" + std::to_string(sequence);
  Action::Goal action_goal;
  action_goal.request_id = request_id;
  action_goal.mission_id = mission_id;
  action_goal.mission_revision = mission_revision;
  action_goal.replace_active_request = replace_active_request;
  action_goal.goal.header = goal.header;
  action_goal.goal.goal_id = request_id + "-goal";
  action_goal.goal.goal_type = action_goal.goal.POINT;
  action_goal.goal.point = goal.pose.position;
  action_goal.goal.position_tolerance_m = goal_tolerance_m;
  action_goal.goal.has_yaw_constraint = true;
  action_goal.goal.yaw_rad = tf2::getYaw(orientation);
  action_goal.goal.yaw_tolerance_rad = yaw_tolerance_rad;

  if (!action_client ||
      !action_client->wait_for_action_server(action_server_timeout)) {
    throw nav2_core::PlannerException{
        "PlanMotion action server is unavailable: " + action_name};
  }

  try {
    auto goal_future = action_client->async_send_goal(action_goal);
    if (goal_future.wait_for(action_server_timeout) !=
        std::future_status::ready) {
      throw nav2_core::PlannerException{
          "timed out waiting for PlanMotion goal acceptance"};
    }
    const auto goal_handle = goal_future.get();
    if (!goal_handle) {
      throw nav2_core::PlannerException{"PlanMotion goal was rejected"};
    }

    auto result_future = action_client->async_get_result(goal_handle);
    if (result_future.wait_for(action_result_timeout) !=
        std::future_status::ready) {
      static_cast<void>(action_client->async_cancel_goal(goal_handle));
      throw nav2_core::PlannerException{
          "timed out waiting for PlanMotion result"};
    }
    const auto wrapped_result = result_future.get();
    if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED) {
      throw nav2_core::PlannerException{
          "PlanMotion finished with action state " +
          ResultCodeName(wrapped_result.code)};
    }
    if (!wrapped_result.result) {
      throw nav2_core::PlannerException{"PlanMotion returned no result message"};
    }
    return ConvertResult(*wrapped_result.result);
  } catch (const nav2_core::PlannerException&) {
    throw;
  } catch (const std::exception& error) {
    throw nav2_core::PlannerException{
        std::string{"PlanMotion request failed: "} + error.what()};
  }
}

nav_msgs::msg::Path LunarGlobalPlanner::ConvertResult(
    const Action::Result& result) const {
  std::string global_frame;
  {
    std::scoped_lock lock{impl_->mutex};
    global_frame = impl_->global_frame;
  }
  if (!result.has_reference) {
    throw nav2_core::PlannerException{
        "PlanMotion did not return an executable reference: " +
        result.reason_code};
  }
  if (result.reference.platform_type !=
      lunar_planning_msgs::msg::MotionReference::WHEELED) {
    throw nav2_core::PlannerException{
        "Nav2 adapter accepts only WHEELED MotionReference results"};
  }
  const auto& preview = result.reference.path_preview;
  if (preview.header.frame_id != global_frame) {
    throw nav2_core::PlannerException{
        "PlanMotion path preview is not in the configured global frame"};
  }
  if (preview.poses.size() < 2U) {
    throw nav2_core::PlannerException{
        "PlanMotion path preview must contain at least two poses"};
  }
  for (const auto& pose : preview.poses) {
    if (pose.header != preview.header) {
      throw nav2_core::PlannerException{
          "PlanMotion path preview pose header is inconsistent"};
    }
  }
  return preview;
}

void LunarGlobalPlanner::SetLatestOdometryForTesting(
    const nav_msgs::msg::Odometry& odometry) {
  std::scoped_lock lock{impl_->mutex};
  impl_->latest_odometry = odometry;
}

}  // namespace lunar::planning::nav2

PLUGINLIB_EXPORT_CLASS(
    lunar::planning::nav2::LunarGlobalPlanner,
    nav2_core::GlobalPlanner)
