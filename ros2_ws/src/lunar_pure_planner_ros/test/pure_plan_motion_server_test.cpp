#include "lunar_pure_planner_ros/pure_plan_motion_server.hpp"
#include "accepted_goal_finalizer.hpp"
#include "action_execution_state.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <action_msgs/msg/goal_status.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <gtest/gtest.h>
#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <lunar_planning_msgs/msg/motion_reference.hpp>
#include <lunar_planning_msgs/msg/timed_path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lunar_pure_planner_ros/request_diagnostics.hpp"

namespace lunar::pure_planner_ros {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using ClientGoalHandle = rclcpp_action::ClientGoalHandle<Action>;
using ServerGoalHandle = rclcpp_action::ServerGoalHandle<Action>;
using namespace std::chrono_literals;

constexpr auto kRequestBudget = 3s;
constexpr auto kSchedulingTolerance = 150ms;

class RosEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    if (!rclcpp::ok()) {
      int argc = 0;
      rclcpp::init(argc, nullptr);
    }
  }

  void TearDown() override {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

const auto* const kRosEnvironment =
    ::testing::AddGlobalTestEnvironment(new RosEnvironment{});

template <typename Predicate>
bool WaitFor(Predicate&& predicate, const std::chrono::milliseconds timeout = 3s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(2ms);
  }
  return predicate();
}

std::filesystem::path ConfigPath(const std::string& filename) {
  return std::filesystem::path{LUNAR_PURE_PLANNER_CONFIG_DIR} / filename;
}

rclcpp::NodeOptions ServerOptions(
    const std::string& platform = "wheel",
    const std::string& platform_config = ConfigPath("wheel.yaml").string(),
    const std::vector<rclcpp::Parameter>& additional_parameters = {}) {
  static std::atomic<std::uint64_t> sequence{0U};
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-r",
                     "__node:=pure_planner_test_" +
                         std::to_string(sequence.fetch_add(1U))});
  std::vector<rclcpp::Parameter> parameters{
      rclcpp::Parameter{"platform_type", platform},
      rclcpp::Parameter{"platform_config", platform_config},
  };
  parameters.insert(parameters.end(), additional_parameters.begin(),
                    additional_parameters.end());
  options.parameter_overrides(parameters);
  return options;
}

std_msgs::msg::Float32MultiArray Layer(const std::size_t width,
                                       const std::size_t height,
                                       const float value) {
  std_msgs::msg::Float32MultiArray layer;
  std_msgs::msg::MultiArrayDimension outer;
  outer.label = "column_index";
  outer.size = height;
  outer.stride = width * height;
  std_msgs::msg::MultiArrayDimension inner;
  inner.label = "row_index";
  inner.size = width;
  inner.stride = width;
  layer.layout.dim = {outer, inner};
  layer.data.assign(width * height, value);
  return layer;
}

grid_map_msgs::msg::GridMap LocalMap() {
  constexpr std::size_t kWidth = 8U;
  constexpr std::size_t kHeight = 8U;
  grid_map_msgs::msg::GridMap map;
  map.header.frame_id = "odom";
  map.info.resolution = 0.2;
  map.info.length_x = kWidth * map.info.resolution;
  map.info.length_y = kHeight * map.info.resolution;
  map.info.pose.orientation.w = 1.0;
  map.layers = {"occupancy", "semantic_id", "elevation", "roughness"};
  map.data = {Layer(kWidth, kHeight, 0.0F), Layer(kWidth, kHeight, 17.0F),
              Layer(kWidth, kHeight, 0.0F), Layer(kWidth, kHeight, 99.0F)};
  return map;
}

grid_map_msgs::msg::GridMap BlockedLocalMap() {
  auto map = LocalMap();
  map.data[0] = Layer(8U, 8U, 1.0F);
  return map;
}

nav_msgs::msg::OccupancyGrid GlobalMap(const std::uint32_t width = 8U) {
  nav_msgs::msg::OccupancyGrid map;
  map.header.frame_id = "map";
  map.info.width = width;
  map.info.height = width;
  map.info.resolution = 1.0F;
  map.info.origin.position.x = -static_cast<double>(width) / 2.0;
  map.info.origin.position.y = -static_cast<double>(width) / 2.0;
  map.info.origin.orientation.w = 1.0;
  map.data.assign(map.info.width * map.info.height, 0);
  return map;
}

nav_msgs::msg::Odometry Odometry(const double position_x = 0.0) {
  nav_msgs::msg::Odometry state;
  state.header.frame_id = "odom";
  state.child_frame_id = "base_link";
  state.pose.pose.position.x = position_x;
  state.pose.pose.orientation.w = 1.0;
  state.pose.covariance[0] = 1'000'000.0;
  state.twist.covariance[0] = 1'000'000.0;
  return state;
}

tf2_msgs::msg::TFMessage Transforms() {
  tf2_msgs::msg::TFMessage message;
  geometry_msgs::msg::TransformStamped map_from_odom;
  map_from_odom.header.frame_id = "map";
  map_from_odom.child_frame_id = "odom";
  map_from_odom.transform.rotation.w = 1.0;
  geometry_msgs::msg::TransformStamped ignored_odom_from_base;
  ignored_odom_from_base.header.frame_id = "odom";
  ignored_odom_from_base.child_frame_id = "base_link";
  ignored_odom_from_base.transform.translation.x = 1000.0;
  ignored_odom_from_base.transform.rotation.w = 1.0;
  message.transforms = {map_from_odom, ignored_odom_from_base};
  return message;
}

lunar::pure_planning::PlanningResult Failure(
    const lunar::pure_planning::PlanningStatus status,
    std::string reason) {
  return {.status = status, .reason_code = std::move(reason)};
}

lunar::pure_planning::PlanningResult Success(
    const lunar::pure_planning::PlanningRequest& request) {
  const auto& point = std::get<lunar::pure_planning::PointGoal>(
      request.goal_map.target);
  lunar::pure_planning::TrajectoryReference trajectory{
      .semantics = lunar::pure_planning::TrajectorySemantics::kWheeledBase,
      .points = {{.pose = {.position_m = point.position_m,
                            .orientation = {.w = 1.0}}}},
  };
  lunar::pure_planning::MotionReference reference{
      .plan_id = request.request_id,
      .platform_type = lunar::pure_planning::PlatformType::kWheeled,
      .input_time = request.world.local_map.stamp,
      .preview = {.poses_map = {{.position_m = point.position_m,
                                 .orientation = {.w = 1.0}}}},
      .data = std::move(trajectory),
  };
  return {.status = lunar::pure_planning::PlanningStatus::kSuccess,
          .reason_code = "PLAN_FOUND",
          .reference = std::move(reference)};
}

lunar::pure_planning::LocalStageResult LocalSuccess(
    const lunar::pure_planning::PlanningRequest& request,
    const lunar::pure_planning::LocalGoalSet& goals,
    const std::size_t selected_goal_index = 0U) {
  if (selected_goal_index >= goals.goals_odom.size()) {
    return {.status = lunar::pure_planning::LocalPlanStatus::kInvalidInput,
            .reason_code = "INVALID_INPUT"};
  }
  const auto* point = std::get_if<lunar::pure_planning::PointGoal>(
      &goals.goals_odom[selected_goal_index].target);
  const auto* state = std::get_if<lunar::pure_planning::WheeledState>(
      &request.current_state);
  if (point == nullptr || state == nullptr) {
    return {.status = lunar::pure_planning::LocalPlanStatus::kInvalidInput,
            .reason_code = "INVALID_INPUT"};
  }
  lunar::pure_planning::TrajectoryReference trajectory{
      .semantics = lunar::pure_planning::TrajectorySemantics::kWheeledBase,
      .points = {{.pose = state->pose},
                 {.pose = {.position_m = point->position_m,
                           .orientation = {.w = 1.0}}}},
  };
  return {
      .status = lunar::pure_planning::LocalPlanStatus::kSolved,
      .data = std::move(trajectory),
      .reason_code = "WHEEL_PLAN_AVAILABLE",
      .selected_goal_index = selected_goal_index,
      .expanded_states = 7U,
      .best_cost = 1.25,
  };
}

class RunningSystem final {
 public:
  explicit RunningSystem(PlannerFn planner,
                         const std::string& platform = "wheel",
                         const std::string& platform_config =
                             ConfigPath("wheel.yaml").string(),
                         const std::vector<rclcpp::Parameter>&
                             additional_parameters = {},
                         LocalPlannerFn local_planner = RealLocalPlannerFn())
      : server(std::make_shared<PurePlanMotionServer>(
            ServerOptions(platform, platform_config, additional_parameters),
            std::move(planner), std::move(local_planner))),
        client(std::make_shared<rclcpp::Node>(UniqueName())),
        executor(MakePurePlannerExecutor()) {
    global_publisher = client->create_publisher<nav_msgs::msg::OccupancyGrid>(
        "/Car/T3/mapping/global_overview", rclcpp::QoS{10}.reliable());
    local_publisher = client->create_publisher<grid_map_msgs::msg::GridMap>(
        "/Car/T3/mapping/grid_map", rclcpp::QoS{10}.reliable());
    odometry_publisher = client->create_publisher<nav_msgs::msg::Odometry>(
        "/Car/T3/localization/odometry", rclcpp::QoS{10}.reliable());
    tf_publisher = client->create_publisher<tf2_msgs::msg::TFMessage>(
        "/tf", rclcpp::QoS{10}.reliable());
    diagnostics_subscription =
        client->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
            "/Car/T4/planning/diagnostics", rclcpp::QoS{10}.reliable(),
            [this](diagnostic_msgs::msg::DiagnosticArray::ConstSharedPtr value) {
              std::scoped_lock lock{diagnostics_mutex};
              diagnostics.push_back(*value);
            });
    wheeled_reference_subscription = client->create_subscription<
        lunar_planning_msgs::msg::MotionReference>(
        "/Car/T4/planning/wheeled_reference", rclcpp::QoS{10}.reliable(),
        [this](lunar_planning_msgs::msg::MotionReference::ConstSharedPtr value) {
          std::scoped_lock lock{wheeled_references_mutex};
          wheeled_references.push_back(*value);
        });
    wheeled_path_subscription = client->create_subscription<nav_msgs::msg::Path>(
        "/Car/T4/planning/wheeled_path", rclcpp::QoS{10}.reliable(),
        [this](nav_msgs::msg::Path::ConstSharedPtr value) {
          std::scoped_lock lock{wheeled_paths_mutex};
          wheeled_paths.push_back(*value);
        });
    wheeled_global_path_subscription =
        client->create_subscription<nav_msgs::msg::Path>(
            "/Car/T4/planning/wheeled_global_path", rclcpp::QoS{10}.reliable(),
            [this](nav_msgs::msg::Path::ConstSharedPtr value) {
              std::scoped_lock lock{wheeled_global_paths_mutex};
              wheeled_global_paths.push_back(*value);
            });
    timed_path_subscription = client->create_subscription<
        lunar_planning_msgs::msg::TimedPath>(
        "/Car/T4/planning/wheeled_path_timing", rclcpp::QoS{10}.reliable(),
        [this](lunar_planning_msgs::msg::TimedPath::ConstSharedPtr value) {
          std::scoped_lock lock{timed_paths_mutex};
          timed_paths.push_back(*value);
        });
    action_client = rclcpp_action::create_client<Action>(
        client, "/Car/T4/plan_motion");
    executor->add_node(server);
    executor->add_node(client);
    spin_thread = std::jthread([this] { executor->spin(); });
    EXPECT_TRUE(action_client->wait_for_action_server(3s));
  }

  ~RunningSystem() {
    executor->cancel();
    if (spin_thread.joinable()) {
      spin_thread.join();
    }
    executor->remove_node(client);
    executor->remove_node(server);
    action_client.reset();
    diagnostics_subscription.reset();
    wheeled_reference_subscription.reset();
    wheeled_path_subscription.reset();
    wheeled_global_path_subscription.reset();
    timed_path_subscription.reset();
    client.reset();
    server.reset();
  }

  void PublishInputs(const bool include_global = true,
                     const double odometry_x = 0.0,
                     const std::uint32_t global_width = 8U) {
    // Destroyed endpoints may briefly remain in the same-process DDS graph
    // cache, so require a live subscriber without requiring an exact count.
    ASSERT_TRUE(WaitFor([this] {
      return local_publisher->get_subscription_count() >= 1U &&
             odometry_publisher->get_subscription_count() >= 1U &&
             tf_publisher->get_subscription_count() >= 1U &&
             global_publisher->get_subscription_count() >= 1U;
    }));
    for (std::size_t attempt = 0U; attempt < 3U; ++attempt) {
      if (include_global) {
        global_publisher->publish(GlobalMap(global_width));
      }
      local_publisher->publish(LocalMap());
      odometry_publisher->publish(Odometry(odometry_x));
      tf_publisher->publish(Transforms());
      std::this_thread::sleep_for(20ms);
    }
  }

  Action::Goal Goal(const std::string& id,
                    const std::uint8_t mode = Action::Goal::LUNAR_SURFACE,
                    const bool replace = false) const {
    Action::Goal goal;
    goal.environment_mode = mode;
    goal.request_id = id;
    goal.mission_id = "compatibility-only";
    goal.mission_revision = 42U;
    goal.replace_active_request = replace;
    goal.goal.header.frame_id = "map";
    goal.goal.goal_id = "point";
    goal.goal.goal_type = goal.goal.POINT;
    goal.goal.point.x = 0.2;
    goal.goal.point.y = 0.0;
    goal.goal.point.z = std::numeric_limits<double>::quiet_NaN();
    goal.goal.position_tolerance_m = 0.1;
    goal.goal.yaw_tolerance_rad = 0.1;
    return goal;
  }

  ClientGoalHandle::SharedPtr SendGoal(const Action::Goal& goal) {
    rclcpp_action::Client<Action>::SendGoalOptions options;
    options.feedback_callback =
        [this](ClientGoalHandle::SharedPtr,
               std::shared_ptr<const Action::Feedback> feedback) {
          if (!feedback) {
            return;
          }
          std::scoped_lock lock{feedback_mutex};
          feedback_samples.push_back(*feedback);
        };
    auto future = action_client->async_send_goal(goal, options);
    EXPECT_EQ(future.wait_for(3s), std::future_status::ready);
    return future.get();
  }

  ClientGoalHandle::WrappedResult Result(
      const ClientGoalHandle::SharedPtr& handle) {
    auto future = action_client->async_get_result(handle);
    EXPECT_EQ(future.wait_for(5s), std::future_status::ready);
    return future.get();
  }

  std::size_t DiagnosticCount() const {
    std::scoped_lock lock{diagnostics_mutex};
    return diagnostics.size();
  }

  std::vector<diagnostic_msgs::msg::DiagnosticArray> Diagnostics() const {
    std::scoped_lock lock{diagnostics_mutex};
    return diagnostics;
  }

  std::vector<lunar_planning_msgs::msg::MotionReference>
  WheeledReferences() const {
    std::scoped_lock lock{wheeled_references_mutex};
    return wheeled_references;
  }

  std::vector<nav_msgs::msg::Path> WheeledPaths() const {
    std::scoped_lock lock{wheeled_paths_mutex};
    return wheeled_paths;
  }

  std::vector<nav_msgs::msg::Path> WheeledGlobalPaths() const {
    std::scoped_lock lock{wheeled_global_paths_mutex};
    return wheeled_global_paths;
  }

  std::vector<lunar_planning_msgs::msg::TimedPath> TimedPaths() const {
    std::scoped_lock lock{timed_paths_mutex};
    return timed_paths;
  }

  std::vector<Action::Feedback> Feedback() const {
    std::scoped_lock lock{feedback_mutex};
    return feedback_samples;
  }

  void PublishLocalOnly() { local_publisher->publish(LocalMap()); }

  void PublishLocal(grid_map_msgs::msg::GridMap map) {
    local_publisher->publish(std::move(map));
  }

  void PublishOdometryOnly(const double position_x) {
    odometry_publisher->publish(Odometry(position_x));
  }

  std::shared_ptr<PurePlanMotionServer> server;
  std::shared_ptr<rclcpp::Node> client;
  std::unique_ptr<rclcpp::Executor> executor;
  std::jthread spin_thread;
  rclcpp_action::Client<Action>::SharedPtr action_client;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_publisher;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr local_publisher;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_publisher;

 private:
  static std::string UniqueName() {
    static std::atomic<std::uint64_t> sequence{0U};
    return "pure_plan_motion_client_" +
           std::to_string(sequence.fetch_add(1U));
  }

  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_subscription;
  rclcpp::Subscription<lunar_planning_msgs::msg::MotionReference>::SharedPtr
      wheeled_reference_subscription;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr wheeled_path_subscription;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr
      wheeled_global_path_subscription;
  rclcpp::Subscription<lunar_planning_msgs::msg::TimedPath>::SharedPtr
      timed_path_subscription;
  mutable std::mutex diagnostics_mutex;
  std::vector<diagnostic_msgs::msg::DiagnosticArray> diagnostics;
  mutable std::mutex wheeled_references_mutex;
  std::vector<lunar_planning_msgs::msg::MotionReference> wheeled_references;
  mutable std::mutex wheeled_paths_mutex;
  std::vector<nav_msgs::msg::Path> wheeled_paths;
  mutable std::mutex wheeled_global_paths_mutex;
  std::vector<nav_msgs::msg::Path> wheeled_global_paths;
  mutable std::mutex timed_paths_mutex;
  std::vector<lunar_planning_msgs::msg::TimedPath> timed_paths;
  mutable std::mutex feedback_mutex;
  std::vector<Action::Feedback> feedback_samples;
};

class DelayedCancelActionSystem final {
 public:
  struct WaitObservation final {
    detail::CancelTransitionWaiter::Result result;
    std::chrono::steady_clock::time_point finished;
  };

  DelayedCancelActionSystem()
      : server_node(std::make_shared<rclcpp::Node>(UniqueName("server"))),
        client_node(std::make_shared<rclcpp::Node>(UniqueName("client"))),
        action_name("/" + UniqueName("action")),
        accepted(accepted_promise.get_future()),
        cancel_entered(cancel_entered_promise.get_future()),
        wait_finished(wait_finished_promise.get_future()),
        executor(MakePurePlannerExecutor()) {
    action_server = rclcpp_action::create_server<Action>(
        server_node, action_name,
        [](const rclcpp_action::GoalUUID&,
           const std::shared_ptr<const Action::Goal>) {
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [this](const std::shared_ptr<ServerGoalHandle>) {
          cancel_entered_promise.set_value(
              std::chrono::steady_clock::now());
          std::this_thread::sleep_for(350ms);
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](const std::shared_ptr<ServerGoalHandle> goal_handle) {
          wait_worker = std::jthread(
              [this, goal_handle](const std::stop_token stop_token) {
                const auto result = cancel_waiter.Wait(
                    [&] { return goal_handle->is_canceling(); },
                    [this] {
                      return rclcpp::ok(
                          server_node->get_node_base_interface()->get_context());
                    },
                    [&] { return stop_token.stop_requested(); });
                if (result ==
                    detail::CancelTransitionWaiter::Result::kCanceling) {
                  auto action_result = std::make_shared<Action::Result>();
                  action_result->planning_outcome = Action::Result::CANCELED;
                  action_result->execution_directive =
                      Action::Result::NO_SAFE_REFERENCE;
                  action_result->reason_code = "REQUEST_CANCELED";
                  ++terminal_attempts;
                  try {
                    goal_handle->canceled(action_result);
                  } catch (...) {
                    ++terminal_failures;
                  }
                }
                wait_finished_promise.set_value(
                    {.result = result,
                     .finished = std::chrono::steady_clock::now()});
              });
          accepted_promise.set_value();
        });
    action_client =
        rclcpp_action::create_client<Action>(client_node, action_name);
    executor->add_node(server_node);
    executor->add_node(client_node);
    spin_thread = std::jthread([this] { executor->spin(); });
  }

  ~DelayedCancelActionSystem() {
    if (wait_worker.joinable()) {
      wait_worker.request_stop();
      cancel_waiter.Notify();
    }
    executor->cancel();
    if (spin_thread.joinable()) {
      spin_thread.join();
    }
    if (wait_worker.joinable()) {
      wait_worker.join();
    }
    executor->remove_node(client_node);
    executor->remove_node(server_node);
  }

  std::shared_ptr<rclcpp::Node> server_node;
  std::shared_ptr<rclcpp::Node> client_node;
  std::string action_name;
  rclcpp_action::Server<Action>::SharedPtr action_server;
  rclcpp_action::Client<Action>::SharedPtr action_client;
  std::promise<void> accepted_promise;
  std::future<void> accepted;
  std::promise<std::chrono::steady_clock::time_point> cancel_entered_promise;
  std::future<std::chrono::steady_clock::time_point> cancel_entered;
  std::promise<WaitObservation> wait_finished_promise;
  std::future<WaitObservation> wait_finished;
  std::atomic<std::uint64_t> terminal_attempts{0U};
  std::atomic<std::uint64_t> terminal_failures{0U};

 private:
  static std::string UniqueName(const std::string_view role) {
    static std::atomic<std::uint64_t> sequence{0U};
    return "delayed_cancel_" + std::string{role} + "_" +
           std::to_string(sequence.fetch_add(1U));
  }

  std::unique_ptr<rclcpp::Executor> executor;
  std::jthread spin_thread;
  detail::CancelTransitionWaiter cancel_waiter;
  std::jthread wait_worker;
};

std::set<std::string> SubscriptionTopics(
    const std::shared_ptr<PurePlanMotionServer>& node) {
  std::set<std::string> topics;
  for (const auto& [topic, unused_types] : node->get_topic_names_and_types()) {
    (void)unused_types;
    for (const auto& endpoint :
         node->get_subscriptions_info_by_topic(topic)) {
      if (endpoint.node_name() == node->get_name() &&
          endpoint.node_namespace() == node->get_namespace()) {
        topics.insert(topic);
      }
    }
  }
  return topics;
}

void ExpectBaseDiagnostic(
    const diagnostic_msgs::msg::DiagnosticArray& diagnostics) {
  ASSERT_EQ(diagnostics.status.size(), 1U);
  EXPECT_EQ(diagnostics.status.front().values.size(), 20U);
}

void ExpectBounded(const std::chrono::steady_clock::duration elapsed) {
  EXPECT_LT(elapsed, kRequestBudget + kSchedulingTolerance);
}

TEST(PurePlanMotionServer, ExposesExactlyTheFrozenOrdinaryNodeGraph) {
  RunningSystem system{[](const auto&) {
    return Failure(lunar::pure_planning::PlanningStatus::kNoPath, "NO_PATH");
  }};

  const std::set<std::string> expected{
      "/Car/T3/localization/odometry", "/Car/T3/mapping/global_overview",
      "/Car/T3/mapping/grid_map", "/tf"};
  const auto raw = SubscriptionTopics(system.server);
  auto business = raw;
  // Humble's ordinary Node time source owns this framework endpoint.
  business.erase("/parameter_events");
  EXPECT_EQ(raw, (std::set<std::string>{
                     "/Car/T3/localization/odometry",
                     "/Car/T3/mapping/global_overview",
                     "/Car/T3/mapping/grid_map", "/parameter_events", "/tf"}));
  EXPECT_EQ(business.size(), 4U);
  EXPECT_EQ(business, expected);
  EXPECT_EQ(system.server->count_publishers(
                "/Car/T4/planning/diagnostics"),
            1U);
  EXPECT_EQ(system.server->count_publishers(
                "/Car/T4/planning/wheeled_reference"),
            1U);
  EXPECT_TRUE(system.action_client->action_server_is_ready());
}

TEST(PurePlanMotionServer, DefaultConfigSelectsEveryPlatformAndMismatchFailsClosed) {
  for (const std::string platform : {"wheel", "legged", "hopper"}) {
    EXPECT_NO_THROW({
      auto node = std::make_shared<PurePlanMotionServer>(
          ServerOptions(platform, ""),
          [](const auto&) {
            return Failure(lunar::pure_planning::PlanningStatus::kNoPath,
                           "NO_PATH");
          });
    }) << platform;
  }
  EXPECT_THROW(
      PurePlanMotionServer(
          ServerOptions("legged", "wheel.yaml"),
          [](const auto&) {
            return Failure(lunar::pure_planning::PlanningStatus::kNoPath,
                           "NO_PATH");
          }),
      std::runtime_error);
  EXPECT_THROW(
      PurePlanMotionServer(
          ServerOptions("hopper", ConfigPath("wheel.yaml").string()),
          [](const auto&) {
            return Failure(lunar::pure_planning::PlanningStatus::kNoPath,
                           "NO_PATH");
          }),
      std::runtime_error);
  EXPECT_THROW(
      PurePlanMotionServer(
          ServerOptions("unknown", ""),
          [](const auto&) {
            return Failure(lunar::pure_planning::PlanningStatus::kNoPath,
                           "NO_PATH");
          }),
      std::runtime_error);
}

TEST(PurePlanMotionServer, RejectsInvalidRollingParameter) {
  try {
    [[maybe_unused]] PurePlanMotionServer server{
        ServerOptions("wheel", ConfigPath("wheel.yaml").string(),
                      {rclcpp::Parameter{"rolling_horizon_m", -1.0}}),
        [](const auto&) {
          return Failure(lunar::pure_planning::PlanningStatus::kNoPath,
                         "NO_PATH");
        }};
    FAIL() << "invalid rolling parameter unexpectedly constructed the server";
  } catch (const std::exception& error) {
    SCOPED_TRACE(error.what());
    EXPECT_NE(std::string{error.what()}.find("PLANNER_ERROR: rolling parameter invalid"),
              std::string::npos);
  }
}

TEST(PurePlanMotionServer, RejectsUnknownWheelPlannerMode) {
  EXPECT_THROW(
      PurePlanMotionServer(
          ServerOptions(
              "wheel", ConfigPath("wheel.yaml").string(),
              {rclcpp::Parameter{"wheel_planner_mode", "unknown_mode"}}),
          [](const auto&) {
            return Failure(lunar::pure_planning::PlanningStatus::kNoPath,
                           "NO_PATH");
          }),
      std::runtime_error);
}

TEST(PurePlanMotionServer,
     ExplicitGridTraversabilityV1AttachesOriginalResolutionSnapshot) {
  std::atomic<bool> saw_v1_snapshot{false};
  RunningSystem system{
      [&](const lunar::pure_planning::PlanningRequest& request) {
        saw_v1_snapshot =
            request.config.wheel_planner_mode ==
                lunar::pure_planning::WheelPlannerMode::kGridTraversabilityV1 &&
            request.world.traversability_snapshot &&
            request.world.traversability_snapshot->valid() &&
            request.world.traversability_snapshot->resolution_m() == 0.2 &&
            request.world.traversability_snapshot->revision() > 0U;
        return Failure(lunar::pure_planning::PlanningStatus::kNoPath,
                       "NO_PATH");
      },
      "wheel", ConfigPath("wheel.yaml").string(),
      {rclcpp::Parameter{"wheel_planner_mode", "grid_traversability_v1"}}};
  system.PublishInputs();

  const auto handle = system.SendGoal(system.Goal("grid-v1-snapshot"));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(system.Result(handle).code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_TRUE(saw_v1_snapshot.load());
}

TEST(PurePlanMotionServer,
     GridTraversabilityV1PlansAndPublishesWithoutGlobalInput) {
  RunningSystem system{
      RealPlannerFn(), "wheel", ConfigPath("wheel.yaml").string(),
      {rclcpp::Parameter{"wheel_planner_mode", "grid_traversability_v1"}}};
  system.PublishInputs(false);

  const auto handle = system.SendGoal(system.Goal("grid-v1-real"));
  ASSERT_NE(handle, nullptr);
  const auto wrapped = system.Result(handle);
  ASSERT_EQ(wrapped.code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_TRUE(wrapped.result->has_reference);
  EXPECT_EQ(wrapped.result->planning_outcome,
            Action::Result::NEW_REFERENCE_AVAILABLE);
  EXPECT_FALSE(wrapped.result->reference.trajectory.points.empty());
  EXPECT_FALSE(wrapped.result->reference.path_preview.poses.empty());
  ASSERT_TRUE(WaitFor([&] {
    return system.WheeledReferences().size() == 1U &&
           system.WheeledPaths().size() == 1U &&
           system.WheeledGlobalPaths().size() == 1U &&
           system.TimedPaths().size() == 1U &&
           system.DiagnosticCount() == 1U;
  }));
  const auto reference = system.WheeledReferences().front();
  const auto path = system.WheeledPaths().front();
  const auto global_path = system.WheeledGlobalPaths().front();
  const auto timed = system.TimedPaths().front();
  ASSERT_FALSE(path.poses.empty());
  ASSERT_FALSE(global_path.poses.empty());
  EXPECT_EQ(global_path.header.frame_id, "map");
  EXPECT_EQ(global_path.header.stamp, path.header.stamp);
  EXPECT_NEAR(global_path.poses.back().pose.position.x, 0.2, 0.3);
  EXPECT_NEAR(global_path.poses.back().pose.position.y, 0.0, 0.3);
  ASSERT_EQ(reference.path_preview.poses.size(), path.poses.size());
  ASSERT_EQ(timed.path.poses.size(), path.poses.size());
  EXPECT_EQ(reference.path_preview.header.frame_id, path.header.frame_id);
  EXPECT_EQ(timed.path.header.frame_id, path.header.frame_id);
  EXPECT_EQ(reference.path_preview.header.stamp, path.header.stamp);
  EXPECT_EQ(timed.path.header.stamp, path.header.stamp);
  for (std::size_t index = 0U; index < path.poses.size(); ++index) {
    EXPECT_EQ(reference.path_preview.poses[index].pose, path.poses[index].pose);
    EXPECT_EQ(timed.path.poses[index].pose, path.poses[index].pose);
  }
  const auto diagnostics = system.Diagnostics().front();
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "planning_outcome"), "0");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "has_reference"), "true");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "grid_v1_active"), "true");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "global_input_sequence"), "0");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "global_call_count"), "1");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "local_call_count"), "1");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "canonical_resolution_m"),
            "0.2");
  EXPECT_TRUE(timed.planning_time.sec > 0 ||
              timed.planning_time.nanosec > 0U);
  const double timed_path_ms =
      static_cast<double>(timed.planning_time.sec) * 1000.0 +
      static_cast<double>(timed.planning_time.nanosec) / 1.0e6;
  EXPECT_NEAR(timed_path_ms,
              std::stod(FindDiagnosticValue(diagnostics,
                                            "total_elapsed_ms")),
              1.0e-6);
}

TEST(PurePlanMotionServer,
     GridTraversabilityV1RejectsPathBlockedAfterPlanningSnapshot) {
  std::promise<void> planner_entered_promise;
  auto planner_entered = planner_entered_promise.get_future();
  std::promise<void> release_planner_promise;
  auto release_planner = release_planner_promise.get_future().share();
  RunningSystem system{
      [&](const lunar::pure_planning::PlanningRequest& request) {
        planner_entered_promise.set_value();
        release_planner.wait();
        return Success(request);
      },
      "wheel", ConfigPath("wheel.yaml").string(),
      {rclcpp::Parameter{"wheel_planner_mode", "grid_traversability_v1"}}};
  system.PublishInputs();

  const auto handle = system.SendGoal(system.Goal("grid-v1-stale"));
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(planner_entered.wait_for(2s), std::future_status::ready);
  system.PublishLocal(BlockedLocalMap());
  std::this_thread::sleep_for(100ms);
  release_planner_promise.set_value();

  const auto wrapped = system.Result(handle);
  EXPECT_EQ(wrapped.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_FALSE(wrapped.result->has_reference);
  EXPECT_EQ(wrapped.result->planning_outcome,
            Action::Result::ACTIVE_REFERENCE_INVALIDATED);
  EXPECT_EQ(wrapped.result->reason_code, "STALE_PATH_INVALIDATED");
  EXPECT_EQ(wrapped.result->execution_directive,
            Action::Result::NO_SAFE_REFERENCE);
  EXPECT_TRUE(wrapped.result->reference.plan_id.empty());
  EXPECT_TRUE(wrapped.result->reference.path_preview.poses.empty());
  EXPECT_TRUE(wrapped.result->reference.trajectory.points.empty());
  ASSERT_TRUE(WaitFor([&] { return system.DiagnosticCount() == 1U; }));
  EXPECT_EQ(FindDiagnosticValue(system.Diagnostics().front(), "reason_code"),
            "STALE_PATH_INVALIDATED");
  ASSERT_TRUE(WaitFor([&] {
    return system.WheeledReferences().size() == 1U &&
           system.WheeledPaths().size() == 1U &&
           system.TimedPaths().size() == 1U;
  }));
  EXPECT_TRUE(system.WheeledReferences().front().plan_id.empty());
  EXPECT_TRUE(system.WheeledReferences().front().path_preview.poses.empty());
  EXPECT_TRUE(system.WheeledReferences().front().trajectory.points.empty());
  EXPECT_TRUE(system.WheeledPaths().front().poses.empty());
  EXPECT_TRUE(system.TimedPaths().front().path.poses.empty());
}

TEST(PurePlanMotionServer, SurfaceNeedsGlobalButLavaDoesNotTouchIt) {
  std::atomic<std::uint64_t> calls{0U};
  RunningSystem system{[&](const auto& request) {
    ++calls;
    return Success(request);
  }};
  system.PublishInputs(false);

  const auto surface = system.SendGoal(system.Goal("surface"));
  ASSERT_NE(surface, nullptr);
  const auto surface_result = system.Result(surface);
  EXPECT_EQ(surface_result.result->reason_code, "INVALID_INPUT");
  EXPECT_EQ(calls.load(), 0U);

  const auto lava = system.SendGoal(
      system.Goal("lava", Action::Goal::LAVA_TUBE));
  ASSERT_NE(lava, nullptr);
  const auto lava_result = system.Result(lava);
  EXPECT_EQ(lava_result.code, rclcpp_action::ResultCode::SUCCEEDED);
  EXPECT_TRUE(lava_result.result->has_reference);
  EXPECT_EQ(calls.load(), 1U);
  ASSERT_TRUE(WaitFor([&] { return system.DiagnosticCount() == 2U; }));
}

TEST(PurePlanMotionServer,
     PublishesExecutableReferenceAndObservablePathsAndClearsAllOnFailure) {
  {
    RunningSystem successful{
        [](const auto& request) { return Success(request); }};
    successful.PublishInputs(false);
    const auto success_handle = successful.SendGoal(
        successful.Goal("publish-wheel-reference", Action::Goal::LAVA_TUBE));
    ASSERT_NE(success_handle, nullptr);
    EXPECT_EQ(successful.Result(success_handle).code,
              rclcpp_action::ResultCode::SUCCEEDED);
    ASSERT_TRUE(
        WaitFor([&] { return successful.WheeledReferences().size() == 1U; }));
    const auto reference = successful.WheeledReferences().front();
    EXPECT_EQ(reference.plan_id, "publish-wheel-reference");
    EXPECT_EQ(reference.platform_type, reference.WHEELED);
    EXPECT_FALSE(reference.trajectory.points.empty());
    ASSERT_TRUE(WaitFor([&] { return successful.WheeledPaths().size() == 1U; }));
    const auto published = successful.WheeledPaths().front();
    EXPECT_EQ(published.header.frame_id, "map");
    EXPECT_FALSE(published.poses.empty());
    ASSERT_TRUE(WaitFor([&] { return successful.TimedPaths().size() == 1U; }));
    const auto timed = successful.TimedPaths().front();
    EXPECT_EQ(timed.path.poses.size(), published.poses.size());
    EXPECT_TRUE(timed.planning_time.sec > 0 || timed.planning_time.nanosec > 0U);
    ASSERT_TRUE(WaitFor(
        [&] { return successful.WheeledGlobalPaths().size() == 1U; }));
  }

  RunningSystem failed{[](const auto&) {
    return Failure(lunar::pure_planning::PlanningStatus::kNoPath, "NO_PATH");
  }};
  failed.PublishInputs(false);
  const auto failed_handle = failed.SendGoal(
      failed.Goal("clear-wheel-reference", Action::Goal::LAVA_TUBE));
  ASSERT_NE(failed_handle, nullptr);
  EXPECT_EQ(failed.Result(failed_handle).code,
            rclcpp_action::ResultCode::ABORTED);
  ASSERT_TRUE(WaitFor([&] { return failed.WheeledReferences().size() == 1U; }));
  const auto cleared_reference = failed.WheeledReferences().front();
  EXPECT_TRUE(cleared_reference.plan_id.empty());
  EXPECT_TRUE(cleared_reference.trajectory.points.empty());
  ASSERT_TRUE(WaitFor([&] { return failed.WheeledPaths().size() == 1U; }));
  const auto cleared = failed.WheeledPaths().front();
  EXPECT_TRUE(cleared.header.frame_id.empty());
  EXPECT_TRUE(cleared.poses.empty());
  ASSERT_TRUE(
      WaitFor([&] { return failed.WheeledGlobalPaths().size() == 1U; }));
  EXPECT_TRUE(failed.WheeledGlobalPaths().front().poses.empty());
  ASSERT_TRUE(WaitFor([&] { return failed.TimedPaths().size() == 1U; }));
  EXPECT_TRUE(failed.TimedPaths().front().path.poses.empty());
}

TEST(PurePlanMotionServer,
     PublishesImmediatePhaseChangesAndThrottlesRepeatedPhaseFeedback) {
  RunningSystem system{[](const lunar::pure_planning::PlanningRequest& request) {
    EXPECT_TRUE(static_cast<bool>(request.progress));
    request.progress({.phase =
                          lunar::pure_planning::PlannerPhase::kSnapshotProjection});
    for (std::size_t repeat = 0U; repeat < 8U; ++repeat) {
      request.progress({.phase =
                            lunar::pure_planning::PlannerPhase::kSnapshotProjection});
    }
    request.progress(
        {.phase = lunar::pure_planning::PlannerPhase::kGlobal});
    auto result = Success(request);
    result.expanded_states = 23U;
    result.best_cost = 4.5;
    return result;
  }};
  system.PublishInputs();

  const auto handle = system.SendGoal(system.Goal("feedback"));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(system.Result(handle).code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_TRUE(WaitFor([&] { return system.Feedback().size() >= 4U; }));
  const auto feedback = system.Feedback();
  ASSERT_GE(feedback.size(), 4U);
  EXPECT_EQ(feedback[0].phase, Action::Feedback::VALIDATING_INPUT);
  EXPECT_EQ(feedback[1].phase, Action::Feedback::BUILDING_SNAPSHOT);
  EXPECT_EQ(feedback[2].phase, Action::Feedback::SEARCHING);
  EXPECT_EQ(feedback[3].phase, Action::Feedback::CERTIFYING);
  EXPECT_EQ(std::ranges::count_if(feedback, [](const auto& sample) {
              return sample.phase == Action::Feedback::BUILDING_SNAPSHOT;
            }),
            1);
  for (std::size_t index = 1U; index < feedback.size(); ++index) {
    EXPECT_GE(feedback[index].elapsed_s, feedback[index - 1U].elapsed_s);
    EXPECT_GE(feedback[index].expanded_states,
              feedback[index - 1U].expanded_states);
  }
  EXPECT_EQ(feedback[3].expanded_states, 23U);
  EXPECT_TRUE(feedback[3].has_best_cost);
  EXPECT_DOUBLE_EQ(feedback[3].best_cost, 4.5);
}

TEST(PurePlanMotionServer, ConsumesTrustedBridgeParameterAfterOneAcceptedGoal) {
  RunningSystem system{[](const auto& request) { return Success(request); }};
  system.PublishInputs(false);

  ASSERT_TRUE(system.server->set_parameter(
      rclcpp::Parameter{"trusted_bridge_once", true}).successful);
  const auto handle = system.SendGoal(
      system.Goal("trusted_bridge_once", Action::Goal::LAVA_TUBE));
  ASSERT_NE(handle, nullptr);
  const auto result = system.Result(handle);

  EXPECT_EQ(result.code, rclcpp_action::ResultCode::SUCCEEDED);
  bool enabled = true;
  ASSERT_TRUE(system.server->get_parameter("trusted_bridge_once", enabled));
  EXPECT_FALSE(enabled);
}

struct SuccessModeCase final {
  const char* name;
  std::uint8_t mode;
  bool publish_global;
};

class SuccessfulRequestLifecycleTest
    : public ::testing::TestWithParam<SuccessModeCase> {};

TEST_P(SuccessfulRequestLifecycleTest,
       UsesTheModeSnapshotAndTerminatesOnceWithinTheRequestBudget) {
  const auto parameters = GetParam();
  std::atomic<bool> saw_expected_snapshot{false};
  RunningSystem system{[&](const lunar::pure_planning::PlanningRequest& request) {
    if (parameters.publish_global && request.world.global_map.has_value()) {
      const auto& global = *request.world.global_map;
      saw_expected_snapshot =
          global.frame_id == "map" && global.width == 8U &&
          global.height == 8U && global.resolution_m == 1.0 &&
          global.origin_m.x == -4.0 && global.origin_m.y == -4.0 &&
          global.HasLayer("occupancy") &&
          global.layers.at("occupancy").size() == 64U;
    } else {
      saw_expected_snapshot =
          !parameters.publish_global && !request.world.global_map.has_value();
    }
    return Success(request);
  }};
  system.PublishInputs(parameters.publish_global);

  const auto started = std::chrono::steady_clock::now();
  const auto handle = system.SendGoal(system.Goal(parameters.name,
                                                  parameters.mode));
  ASSERT_NE(handle, nullptr);
  const auto result = system.Result(handle);
  ExpectBounded(std::chrono::steady_clock::now() - started);

  EXPECT_EQ(result.code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_NE(result.result, nullptr);
  EXPECT_EQ(result.result->planning_outcome,
            Action::Result::NEW_REFERENCE_AVAILABLE);
  EXPECT_TRUE(result.result->has_reference);
  EXPECT_TRUE(saw_expected_snapshot.load());
  ASSERT_TRUE(WaitFor([&] { return system.DiagnosticCount() == 1U; }));
  // The server publishes diagnostics before committing the Action terminal;
  // with the result future complete, this is a closed publication boundary.
  ASSERT_EQ(system.DiagnosticCount(), 1U);
  const auto diagnostics = system.Diagnostics().front();
  ExpectBaseDiagnostic(diagnostics);
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "reason_code"), "PLAN_FOUND");
}

INSTANTIATE_TEST_SUITE_P(
    SurfaceAndLava, SuccessfulRequestLifecycleTest,
    ::testing::Values(
        SuccessModeCase{"surface_success", Action::Goal::LUNAR_SURFACE, true},
        SuccessModeCase{"lava_success", Action::Goal::LAVA_TUBE, false}),
    [](const ::testing::TestParamInfo<SuccessModeCase>& info) {
      return std::string{info.param.name};
    });

struct PlatformCase final {
  const char* name;
  lunar::pure_planning::PlatformType type;
};

class SelectedPlatformVariantTest
    : public ::testing::TestWithParam<PlatformCase> {};

TEST_P(SelectedPlatformVariantTest,
       UsesMatchingStateAndCapabilityVariants) {
  const auto parameters = GetParam();
  std::atomic<bool> saw_matching_variants{false};
  RunningSystem system{
      [&](const lunar::pure_planning::PlanningRequest& request) {
        using lunar::pure_planning::HopperCapability;
        using lunar::pure_planning::HopperState;
        using lunar::pure_planning::LeggedCapability;
        using lunar::pure_planning::LeggedState;
        using lunar::pure_planning::PlatformType;
        using lunar::pure_planning::WheeledCapability;
        using lunar::pure_planning::WheeledState;
        switch (parameters.type) {
          case PlatformType::kWheeled:
            saw_matching_variants =
                std::holds_alternative<WheeledState>(request.current_state) &&
                std::holds_alternative<WheeledCapability>(request.capability);
            break;
          case PlatformType::kLegged:
            saw_matching_variants =
                std::holds_alternative<LeggedState>(request.current_state) &&
                std::holds_alternative<LeggedCapability>(request.capability);
            break;
          case PlatformType::kHopper:
            saw_matching_variants =
                std::holds_alternative<HopperState>(request.current_state) &&
                std::holds_alternative<HopperCapability>(request.capability);
            break;
        }
        return Failure(lunar::pure_planning::PlanningStatus::kNoPath,
                       "NO_PATH");
      },
      parameters.name,
      ConfigPath(std::string{parameters.name} + ".yaml").string()};
  system.PublishInputs(false);
  const auto handle = system.SendGoal(
      system.Goal(parameters.name, Action::Goal::LAVA_TUBE));
  ASSERT_NE(handle, nullptr);
  (void)system.Result(handle);
  EXPECT_TRUE(saw_matching_variants.load());
}

INSTANTIATE_TEST_SUITE_P(
    WheelLeggedHopper, SelectedPlatformVariantTest,
    ::testing::Values(
        PlatformCase{"wheel", lunar::pure_planning::PlatformType::kWheeled},
        PlatformCase{"legged", lunar::pure_planning::PlatformType::kLegged},
        PlatformCase{"hopper", lunar::pure_planning::PlatformType::kHopper}),
    [](const ::testing::TestParamInfo<PlatformCase>& info) {
      return std::string{info.param.name};
    });

TEST(PurePlanMotionServer, RejectsConflictAndSerializesReplacement) {
  std::atomic<std::uint64_t> calls{0U};
  std::atomic<std::uint64_t> active{0U};
  std::atomic<std::uint64_t> maximum{0U};
  std::atomic<bool> release_second{false};
  RunningSystem system{[&](const lunar::pure_planning::PlanningRequest& request) {
    const std::uint64_t call = ++calls;
    const std::uint64_t now_active = ++active;
    auto observed = maximum.load();
    while (observed < now_active &&
           !maximum.compare_exchange_weak(observed, now_active)) {
    }
    while (!request.control.stop_token.stop_requested() &&
           !(call == 2U && release_second.load())) {
      std::this_thread::sleep_for(1ms);
    }
    --active;
    if (request.control.stop_token.stop_requested()) {
      return Failure(lunar::pure_planning::PlanningStatus::kCanceled,
                     "REQUEST_CANCELED");
    }
    return Failure(lunar::pure_planning::PlanningStatus::kNoPath, "NO_PATH");
  }};
  system.PublishInputs();

  const auto first = system.SendGoal(system.Goal("first"));
  ASSERT_NE(first, nullptr);
  ASSERT_TRUE(WaitFor([&] { return calls.load() == 1U; }));
  EXPECT_EQ(system.SendGoal(system.Goal("rejected")), nullptr);
  const auto replacement_started = std::chrono::steady_clock::now();
  const auto replacement = system.SendGoal(system.Goal("replacement",
                                                        Action::Goal::LUNAR_SURFACE,
                                                        true));
  ASSERT_NE(replacement, nullptr);
  ASSERT_TRUE(WaitFor([&] { return calls.load() == 2U; }));
  release_second = true;

  const auto first_result = system.Result(first);
  const auto replacement_result = system.Result(replacement);
  ExpectBounded(std::chrono::steady_clock::now() - replacement_started);
  // Replacement is a server-side preemption, not a client CancelGoal request,
  // so ROS represents the terminal state as ABORTED while the frozen Action
  // payload still carries CANCELED/REQUEST_CANCELED.
  EXPECT_EQ(first_result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(first_result.result->reason_code, "REQUEST_CANCELED");
  EXPECT_EQ(replacement_result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(replacement_result.result->reason_code, "NO_PATH");
  EXPECT_EQ(maximum.load(), 1U);
  ASSERT_TRUE(WaitFor([&] { return system.DiagnosticCount() == 2U; }));
  ASSERT_EQ(system.DiagnosticCount(), 2U);
  std::set<std::string> diagnostic_request_ids;
  std::set<std::string> diagnostic_reasons;
  for (const auto& diagnostics : system.Diagnostics()) {
    ExpectBaseDiagnostic(diagnostics);
    diagnostic_request_ids.insert(
        FindDiagnosticValue(diagnostics, "request_id"));
    diagnostic_reasons.insert(FindDiagnosticValue(diagnostics, "reason_code"));
  }
  EXPECT_EQ(diagnostic_request_ids,
            (std::set<std::string>{"first", "replacement"}));
  EXPECT_EQ(diagnostic_reasons,
            (std::set<std::string>{"NO_PATH", "REQUEST_CANCELED"}));
}

TEST(ActionExecutionState,
     GoalDecisionWithoutAcceptedCallbackDoesNotReserveServer) {
  detail::ActionExecutionState state;
  int first_goal{};

  EXPECT_FALSE(state.MayAccept(false, false));
  EXPECT_FALSE(state.TryStart(nullptr).has_value());
  EXPECT_TRUE(state.MayAccept(true, false));
  EXPECT_EQ(state.phase(), detail::ActionExecutionState::Phase::kIdle);
  EXPECT_TRUE(state.TryStart(&first_goal).has_value());
  EXPECT_EQ(state.phase(), detail::ActionExecutionState::Phase::kRunning);
}

TEST(ActionExecutionState,
     DroppedReplacementDecisionDoesNotMutateActiveExecution) {
  detail::ActionExecutionState state;
  int active_goal{};
  const auto generation = state.TryStart(&active_goal);
  ASSERT_TRUE(generation.has_value());

  EXPECT_TRUE(state.MayAccept(true, true));
  EXPECT_EQ(state.phase(), detail::ActionExecutionState::Phase::kRunning);
  const auto claim = state.ClaimTerminal(*generation, &active_goal);
  ASSERT_TRUE(claim.claimed);
  EXPECT_FALSE(claim.replacement_accepted);
}

TEST(PurePlanMotionServer,
     TimeoutAbortsBoundedlyWithOneTenKeyDiagnostic) {
  RunningSystem system{[](const lunar::pure_planning::PlanningRequest& request) {
    while (!request.control.stop_token.stop_requested() &&
           request.control.now() < request.control.deadline) {
      std::this_thread::sleep_for(1ms);
    }
    return Failure(lunar::pure_planning::PlanningStatus::kTimedOut,
                   "TIMEOUT");
  }};
  system.PublishInputs();

  const auto started = std::chrono::steady_clock::now();
  const auto handle = system.SendGoal(system.Goal("timeout"));
  ASSERT_NE(handle, nullptr);
  const auto result = system.Result(handle);
  const auto wall_elapsed = std::chrono::steady_clock::now() - started;
  ExpectBounded(wall_elapsed);

  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  ASSERT_NE(result.result, nullptr);
  EXPECT_EQ(result.result->planning_outcome,
            Action::Result::RESOURCE_EXHAUSTED);
  EXPECT_EQ(result.result->reason_code, "TIMEOUT");
  EXPECT_FALSE(result.result->has_reference);
  ASSERT_TRUE(WaitFor([&] { return system.DiagnosticCount() == 1U; }));
  ASSERT_EQ(system.DiagnosticCount(), 1U);
  const auto diagnostics = system.Diagnostics().front();
  ExpectBaseDiagnostic(diagnostics);
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "reason_code"), "TIMEOUT");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "planning_outcome"),
            std::to_string(Action::Result::RESOURCE_EXHAUSTED));
  const double total_ms =
      std::stod(FindDiagnosticValue(diagnostics, "total_elapsed_ms"));
  const double maximum_total_ms = std::chrono::duration<double, std::milli>(
                                      kRequestBudget + kSchedulingTolerance)
                                      .count();
  EXPECT_GE(total_ms, 3000.0);
  EXPECT_LT(total_ms, maximum_total_ms);
  EXPECT_NEAR(result.result->diagnostics.elapsed_s * 1000.0, total_ms, 1.0);
}

TEST(PurePlanMotionServer, ClientCancelTerminatesOnceAndPublishesOneDiagnostic) {
  RunningSystem system{[](const lunar::pure_planning::PlanningRequest& request) {
    while (!request.control.stop_token.stop_requested()) {
      std::this_thread::sleep_for(1ms);
    }
    return Failure(lunar::pure_planning::PlanningStatus::kCanceled,
                   "REQUEST_CANCELED");
  }};
  system.PublishInputs();
  const auto started = std::chrono::steady_clock::now();
  const auto handle = system.SendGoal(system.Goal("cancel"));
  ASSERT_NE(handle, nullptr);
  auto cancel = system.action_client->async_cancel_goal(handle);
  ASSERT_EQ(cancel.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(cancel.get()->return_code,
            action_msgs::srv::CancelGoal::Response::ERROR_NONE);
  const auto result = system.Result(handle);
  ExpectBounded(std::chrono::steady_clock::now() - started);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::CANCELED);
  EXPECT_EQ(result.result->reason_code, "REQUEST_CANCELED");
  ASSERT_TRUE(WaitFor([&] { return system.DiagnosticCount() == 1U; }));
  ASSERT_EQ(system.DiagnosticCount(), 1U);
  const auto diagnostics = system.Diagnostics().front();
  ExpectBaseDiagnostic(diagnostics);
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "request_id"), "cancel");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "reason_code"),
            "REQUEST_CANCELED");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "planning_outcome"),
            std::to_string(Action::Result::CANCELED));
}

TEST(CancelTransitionWaiter,
     AcceptedCancelWaitsPastFormerGraceUntilCancelingTransition) {
  detail::CancelTransitionWaiter waiter;
  std::atomic<bool> canceling{false};
  std::atomic<bool> context_valid{true};
  std::atomic<bool> teardown{false};
  std::atomic<std::uint64_t> canceled_terminal_count{0U};
  std::atomic<std::uint64_t> result_count{0U};
  std::atomic<std::uint64_t> diagnostic_count{0U};
  auto waiting = std::async(std::launch::async, [&] {
    ++diagnostic_count;
    const auto transition = waiter.Wait(
        [&] { return canceling.load(); },
        [&] { return context_valid.load(); },
        [&] { return teardown.load(); });
    if (transition == detail::CancelTransitionWaiter::Result::kCanceling) {
      ++canceled_terminal_count;
      ++result_count;
    }
    return transition;
  });

  std::this_thread::sleep_for(350ms);
  EXPECT_EQ(waiting.wait_for(0ms), std::future_status::timeout);
  EXPECT_EQ(diagnostic_count.load(), 1U);
  EXPECT_EQ(canceled_terminal_count.load(), 0U);
  EXPECT_EQ(result_count.load(), 0U);
  canceling = true;
  waiter.Notify();
  ASSERT_EQ(waiting.wait_for(100ms), std::future_status::ready);
  EXPECT_EQ(waiting.get(),
            detail::CancelTransitionWaiter::Result::kCanceling);
  EXPECT_EQ(diagnostic_count.load(), 1U);
  EXPECT_EQ(canceled_terminal_count.load(), 1U);
  EXPECT_EQ(result_count.load(), 1U);
}

TEST(CancelTransitionWaiter,
     RealRclCancelCallbackDelayedFor350msStillTerminatesCanceledOnce) {
  DelayedCancelActionSystem system;
  ASSERT_TRUE(system.action_client->wait_for_action_server(3s));
  auto goal_future = system.action_client->async_send_goal(Action::Goal{});
  ASSERT_EQ(goal_future.wait_for(3s), std::future_status::ready);
  const auto goal_handle = goal_future.get();
  ASSERT_NE(goal_handle, nullptr);
  ASSERT_EQ(system.accepted.wait_for(3s), std::future_status::ready);
  auto result_future = system.action_client->async_get_result(goal_handle);

  auto cancel_future = system.action_client->async_cancel_goal(goal_handle);
  ASSERT_EQ(system.cancel_entered.wait_for(3s), std::future_status::ready);
  const auto cancel_entered = system.cancel_entered.get();
  EXPECT_EQ(cancel_future.wait_for(250ms), std::future_status::timeout);
  ASSERT_EQ(cancel_future.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(cancel_future.get()->return_code,
            action_msgs::srv::CancelGoal::Response::ERROR_NONE);

  ASSERT_EQ(system.wait_finished.wait_for(1s), std::future_status::ready);
  const auto observation = system.wait_finished.get();
  EXPECT_EQ(observation.result,
            detail::CancelTransitionWaiter::Result::kCanceling);
  EXPECT_GE(observation.finished - cancel_entered, 330ms);
  ASSERT_EQ(result_future.wait_for(1s), std::future_status::ready);
  const auto result = result_future.get();
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::CANCELED);
  EXPECT_EQ(system.terminal_attempts.load(), 1U);
  EXPECT_EQ(system.terminal_failures.load(), 0U);
}

TEST(CancelTransitionWaiter, ExplicitTeardownReleasesWaitBoundedly) {
  detail::CancelTransitionWaiter waiter;
  std::atomic<bool> teardown{false};
  auto waiting = std::async(std::launch::async, [&] {
    return waiter.Wait([] { return false; }, [] { return true; },
                       [&] { return teardown.load(); });
  });

  std::this_thread::sleep_for(10ms);
  teardown = true;
  waiter.Notify();
  ASSERT_EQ(waiting.wait_for(100ms), std::future_status::ready);
  EXPECT_EQ(waiting.get(), detail::CancelTransitionWaiter::Result::kStopped);
}

TEST(CancelTransitionWaiter,
     ExistingTeardownStopsWithoutCallingCancelPredicate) {
  detail::CancelTransitionWaiter waiter;
  std::atomic<std::uint64_t> cancel_predicate_calls{0U};

  const auto result = waiter.Wait(
      [&] {
        ++cancel_predicate_calls;
        return false;
      },
      [] { return true; }, [] { return true; });

  EXPECT_EQ(result, detail::CancelTransitionWaiter::Result::kStopped);
  EXPECT_EQ(cancel_predicate_calls.load(), 0U);
}

TEST(CancelTransitionWaiter,
     InvalidContextReleasesWithoutCallingCancelPredicate) {
  detail::CancelTransitionWaiter waiter;
  std::atomic<std::uint64_t> cancel_predicate_calls{0U};

  const auto result = waiter.Wait(
      [&] {
        ++cancel_predicate_calls;
        return false;
      },
      [] { return false; }, [] { return false; });

  EXPECT_EQ(result, detail::CancelTransitionWaiter::Result::kStopped);
  EXPECT_EQ(cancel_predicate_calls.load(), 0U);
}

TEST(CancelTransitionWaiter,
     PersistentCancelPredicateExceptionFailsImmediatelyAndKeepsStateClosed) {
  detail::CancelTransitionWaiter waiter;
  detail::ActionExecutionState state;
  int active_goal{};
  int replacement_goal{};
  const auto generation = state.TryStart(&active_goal);
  ASSERT_TRUE(generation.has_value());
  ASSERT_TRUE(state.AcceptCancel(&active_goal, true));
  const auto claim = state.ClaimTerminal(*generation, &active_goal);
  ASSERT_TRUE(claim.claimed);
  std::atomic<std::uint64_t> predicate_calls{0U};

  const auto started = std::chrono::steady_clock::now();
  const auto result = waiter.Wait(
      [&]() -> bool {
        ++predicate_calls;
        throw std::runtime_error{"persistent is_canceling failure"};
      },
      [] { return true; }, [] { return false; });

  EXPECT_EQ(result, detail::CancelTransitionWaiter::Result::kPredicateError);
  EXPECT_EQ(predicate_calls.load(), 1U);
  EXPECT_LT(std::chrono::steady_clock::now() - started, 100ms);
  state.MarkTerminalFailed(*generation, &active_goal);
  state.Finish(*generation, &active_goal);
  EXPECT_EQ(state.phase(),
            detail::ActionExecutionState::Phase::kTerminalFailed);
  EXPECT_FALSE(state.TryStart(&replacement_goal).has_value());
}

TEST(PurePlanMotionServer,
     ProductionExecutorProcessesLatestInputWhileReplacementJoinsOldWorker) {
  std::atomic<std::uint64_t> calls{0U};
  std::atomic<bool> old_worker_waiting_for_release{false};
  std::atomic<bool> release_old_worker{false};
  std::atomic<double> replacement_start_x{-1.0};
  RunningSystem system{
      [&](const lunar::pure_planning::PlanningRequest& request) {
        const auto call = ++calls;
        if (call == 1U) {
          while (!request.control.stop_token.stop_requested()) {
            std::this_thread::sleep_for(1ms);
          }
          old_worker_waiting_for_release = true;
          while (!release_old_worker.load()) {
            std::this_thread::sleep_for(1ms);
          }
          return Failure(lunar::pure_planning::PlanningStatus::kCanceled,
                         "REQUEST_CANCELED");
        }
        replacement_start_x =
            std::get<lunar::pure_planning::WheeledState>(
                request.current_state)
                .pose.position_m.x;
        return Failure(lunar::pure_planning::PlanningStatus::kNoPath,
                       "NO_PATH");
      }};
  system.PublishInputs();

  const auto first = system.SendGoal(system.Goal("join_old"));
  ASSERT_NE(first, nullptr);
  ASSERT_TRUE(WaitFor([&] { return calls.load() == 1U; }));
  auto replacement_future = system.action_client->async_send_goal(
      system.Goal("capture_latest", Action::Goal::LUNAR_SURFACE, true));
  ASSERT_TRUE(WaitFor([&] { return old_worker_waiting_for_release.load(); }));

  system.PublishInputs(true, 42.0);
  std::this_thread::sleep_for(100ms);
  release_old_worker = true;

  ASSERT_EQ(replacement_future.wait_for(1s), std::future_status::ready);
  const auto replacement = replacement_future.get();
  ASSERT_NE(replacement, nullptr);
  EXPECT_EQ(system.Result(first).result->reason_code, "REQUEST_CANCELED");
  EXPECT_EQ(system.Result(replacement).result->reason_code, "NO_PATH");
  EXPECT_EQ(replacement_start_x.load(), 42.0);
}

TEST(ActionExecutionState,
     AcceptedReplacementCannotOverwriteOldTerminalFailure) {
  detail::ActionExecutionState state;
  int old_goal{};
  int replacement_goal{};
  int later_goal{};
  const auto old_generation = state.TryStart(&old_goal);
  ASSERT_TRUE(old_generation.has_value());

  // Middleware has accepted a replacement and the old worker owns the
  // terminal transition.  Simulate the old terminal primitive throwing.
  ASSERT_TRUE(state.RequestReplacement());
  const auto old_claim = state.ClaimTerminal(*old_generation, &old_goal);
  ASSERT_TRUE(old_claim.claimed);
  EXPECT_TRUE(old_claim.replacement_accepted);
  state.MarkTerminalFailed(*old_generation, &old_goal);
  state.Finish(*old_generation, &old_goal);

  EXPECT_EQ(state.phase(),
            detail::ActionExecutionState::Phase::kTerminalFailed);
  EXPECT_FALSE(state.TryStart(&replacement_goal).has_value());
  EXPECT_FALSE(state.MayAccept(true, false));
  EXPECT_FALSE(state.TryStart(&later_goal).has_value());
  EXPECT_EQ(state.phase(),
            detail::ActionExecutionState::Phase::kTerminalFailed);
}

TEST(AcceptedGoalFinalizer,
     DiagnosticConstructionFailureCannotBlockSingleMinimalAbortAttempt) {
  detail::ActionExecutionState state;
  int old_goal{};
  int accepted_replacement{};
  int later_goal{};
  const auto old_generation = state.TryStart(&old_goal);
  ASSERT_TRUE(old_generation.has_value());
  ASSERT_TRUE(state.RequestReplacement());
  ASSERT_TRUE(state.ClaimTerminal(*old_generation, &old_goal).claimed);
  state.MarkTerminalFailed(*old_generation, &old_goal);
  state.Finish(*old_generation, &old_goal);
  ASSERT_EQ(state.phase(),
            detail::ActionExecutionState::Phase::kTerminalFailed);

  std::vector<std::string> order;
  std::vector<detail::AcceptedGoalFinalizationFailure> failures;
  std::uint64_t abort_attempts = 0U;
  std::shared_ptr<Action::Result> delivered_result;
  detail::FinalizeAcceptedGoal(
      [&] {
        order.emplace_back("result");
        return detail::MakeMinimalPlannerErrorResult<Action>(17U, 3ms);
      },
      [&](const std::shared_ptr<Action::Result>& result) {
        order.emplace_back("abort");
        ++abort_attempts;
        delivered_result = result;
      },
      [&]() -> diagnostic_msgs::msg::DiagnosticArray {
        order.emplace_back("diagnostic_builder");
        throw std::runtime_error{"diagnostic formatting failed"};
      },
      [&](const diagnostic_msgs::msg::DiagnosticArray&) {
        ADD_FAILURE() << "publisher must not run without diagnostics";
      },
      [&](const diagnostic_msgs::msg::DiagnosticArray&) {
        ADD_FAILURE() << "formatter must not run without diagnostics";
      },
      [&](const detail::AcceptedGoalFinalizationFailure failure) {
        failures.push_back(failure);
      });

  EXPECT_EQ(order,
            (std::vector<std::string>{"result", "abort",
                                      "diagnostic_builder"}));
  EXPECT_EQ(abort_attempts, 1U);
  ASSERT_NE(delivered_result, nullptr);
  EXPECT_EQ(delivered_result->planning_outcome,
            Action::Result::NUMERICAL_FAILURE);
  EXPECT_EQ(delivered_result->execution_directive,
            Action::Result::NO_SAFE_REFERENCE);
  EXPECT_EQ(delivered_result->reason_code, "PLANNER_ERROR");
  EXPECT_EQ(delivered_result->mission_revision, 17U);
  EXPECT_FALSE(delivered_result->has_reference);
  EXPECT_EQ(failures,
            (std::vector<detail::AcceptedGoalFinalizationFailure>{
                detail::AcceptedGoalFinalizationFailure::
                    kDiagnosticConstruction}));
  EXPECT_EQ(state.generation(), *old_generation);
  EXPECT_EQ(state.phase(),
            detail::ActionExecutionState::Phase::kTerminalFailed);
  EXPECT_FALSE(state.TryStart(&accepted_replacement).has_value());
  EXPECT_FALSE(state.TryStart(&later_goal).has_value());
}

TEST(AcceptedGoalFinalizer,
     DiagnosticPublicationFailureOccursAfterExactlyOneAbortAttempt) {
  std::vector<std::string> order;
  std::vector<detail::AcceptedGoalFinalizationFailure> failures;
  std::uint64_t abort_attempts = 0U;
  detail::FinalizeAcceptedGoal(
      [&] {
        order.emplace_back("result");
        return detail::MakeMinimalPlannerErrorResult<Action>(0U, 5ms);
      },
      [&](const std::shared_ptr<Action::Result>&) {
        order.emplace_back("abort");
        ++abort_attempts;
      },
      [&] {
        order.emplace_back("diagnostic_builder");
        return diagnostic_msgs::msg::DiagnosticArray{};
      },
      [&](const diagnostic_msgs::msg::DiagnosticArray&) {
        order.emplace_back("diagnostic_publish");
        throw std::runtime_error{"diagnostic publish failed"};
      },
      [&](const diagnostic_msgs::msg::DiagnosticArray&) {
        order.emplace_back("diagnostic_format");
      },
      [&](const detail::AcceptedGoalFinalizationFailure failure) {
        failures.push_back(failure);
      });

  EXPECT_EQ(order,
            (std::vector<std::string>{"result", "abort",
                                      "diagnostic_builder",
                                      "diagnostic_publish",
                                      "diagnostic_format"}));
  EXPECT_EQ(abort_attempts, 1U);
  EXPECT_EQ(failures,
            (std::vector<detail::AcceptedGoalFinalizationFailure>{
                detail::AcceptedGoalFinalizationFailure::
                    kDiagnosticPublication}));
}

TEST(AcceptedGoalFinalizer,
     TerminalExceptionIsOneAttemptAndDiagnosticsRemainBestEffort) {
  std::uint64_t abort_attempts = 0U;
  std::uint64_t diagnostic_publications = 0U;
  std::vector<detail::AcceptedGoalFinalizationFailure> failures;
  detail::FinalizeAcceptedGoal(
      [] { return detail::MakeMinimalPlannerErrorResult<Action>(0U, 1ms); },
      [&](const std::shared_ptr<Action::Result>&) {
        ++abort_attempts;
        throw std::runtime_error{"abort failed"};
      },
      [] { return diagnostic_msgs::msg::DiagnosticArray{}; },
      [&](const diagnostic_msgs::msg::DiagnosticArray&) {
        ++diagnostic_publications;
      },
      [](const diagnostic_msgs::msg::DiagnosticArray&) {},
      [&](const detail::AcceptedGoalFinalizationFailure failure) {
        failures.push_back(failure);
      });

  EXPECT_EQ(abort_attempts, 1U);
  EXPECT_EQ(diagnostic_publications, 1U);
  EXPECT_EQ(failures,
            (std::vector<detail::AcceptedGoalFinalizationFailure>{
                detail::AcceptedGoalFinalizationFailure::kTerminalAttempt}));
}

TEST(AcceptedGoalFinalizer,
     DiagnosticFormattingFailureOccursAfterExactlyOneAbortAttempt) {
  std::uint64_t abort_attempts = 0U;
  std::uint64_t diagnostic_publications = 0U;
  std::vector<detail::AcceptedGoalFinalizationFailure> failures;
  detail::FinalizeAcceptedGoal(
      [] { return detail::MakeMinimalPlannerErrorResult<Action>(0U, 1ms); },
      [&](const std::shared_ptr<Action::Result>&) { ++abort_attempts; },
      [] { return diagnostic_msgs::msg::DiagnosticArray{}; },
      [&](const diagnostic_msgs::msg::DiagnosticArray&) {
        ++diagnostic_publications;
      },
      [](const diagnostic_msgs::msg::DiagnosticArray&) {
        throw std::runtime_error{"diagnostic formatting failed"};
      },
      [&](const detail::AcceptedGoalFinalizationFailure failure) {
        failures.push_back(failure);
      });

  EXPECT_EQ(abort_attempts, 1U);
  EXPECT_EQ(diagnostic_publications, 1U);
  EXPECT_EQ(failures,
            (std::vector<detail::AcceptedGoalFinalizationFailure>{
                detail::AcceptedGoalFinalizationFailure::
                    kDiagnosticFormatting}));
}

TEST(AcceptedGoalFinalizer,
     ResultConstructionExceptionIsReportedWithoutUnsafeLaterWork) {
  std::uint64_t abort_attempts = 0U;
  std::uint64_t diagnostic_builds = 0U;
  std::vector<detail::AcceptedGoalFinalizationFailure> failures;
  detail::FinalizeAcceptedGoal(
      []() -> std::shared_ptr<Action::Result> {
        throw std::runtime_error{"minimal result allocation failed"};
      },
      [&](const std::shared_ptr<Action::Result>&) { ++abort_attempts; },
      [&] {
        ++diagnostic_builds;
        return diagnostic_msgs::msg::DiagnosticArray{};
      },
      [](const diagnostic_msgs::msg::DiagnosticArray&) {},
      [](const diagnostic_msgs::msg::DiagnosticArray&) {},
      [&](const detail::AcceptedGoalFinalizationFailure failure) {
        failures.push_back(failure);
      });

  EXPECT_EQ(abort_attempts, 0U);
  EXPECT_EQ(diagnostic_builds, 0U);
  EXPECT_EQ(failures,
            (std::vector<detail::AcceptedGoalFinalizationFailure>{
                detail::AcceptedGoalFinalizationFailure::
                    kResultConstruction}));
}

TEST(PurePlanMotionServer, DestructorStopsAndJoinsTheActiveWorkerBoundedly) {
  std::atomic<bool> entered{false};
  auto system = std::make_unique<RunningSystem>(
      [&](const lunar::pure_planning::PlanningRequest& request) {
        entered = true;
        while (!request.control.stop_token.stop_requested()) {
          std::this_thread::sleep_for(1ms);
        }
        return Failure(lunar::pure_planning::PlanningStatus::kCanceled,
                       "REQUEST_CANCELED");
      });
  system->PublishInputs();
  const auto handle = system->SendGoal(system->Goal("destructor"));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor([&] { return entered.load(); }));

  const auto started = std::chrono::steady_clock::now();
  system.reset();
  EXPECT_LT(std::chrono::steady_clock::now() - started, 1s);
}

TEST(PurePlanMotionServer,
     ContextShutdownStopsAndJoinsActiveWorkerWithoutTerminating) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        std::atomic<bool> entered{false};
        auto system = std::make_unique<RunningSystem>(
            [&](const lunar::pure_planning::PlanningRequest& request) {
              entered = true;
              while (!request.control.stop_token.stop_requested()) {
                std::this_thread::sleep_for(1ms);
              }
              return Failure(
                  lunar::pure_planning::PlanningStatus::kCanceled,
                  "REQUEST_CANCELED");
            });
        system->PublishInputs();
        const auto handle = system->SendGoal(system->Goal("shutdown"));
        if (!handle || !WaitFor([&] { return entered.load(); })) {
          std::_Exit(2);
        }

        const auto started = std::chrono::steady_clock::now();
        rclcpp::shutdown();
        system.reset();
        if (std::chrono::steady_clock::now() - started >= 1s) {
          std::_Exit(3);
        }
        std::_Exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(PurePlanMotionServer, ExceptionMapsToPlannerErrorAndNodeContinues) {
  std::atomic<std::uint64_t> calls{0U};
  RunningSystem system{[&](const auto&) {
    if (++calls == 1U) {
      throw std::runtime_error{"planner failure"};
    }
    return Failure(lunar::pure_planning::PlanningStatus::kNoPath, "NO_PATH");
  }};
  system.PublishInputs();

  const auto first = system.SendGoal(system.Goal("throws"));
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(system.Result(first).result->reason_code, "PLANNER_ERROR");
  const auto second = system.SendGoal(system.Goal("continues"));
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(system.Result(second).result->reason_code, "NO_PATH");
  EXPECT_EQ(calls.load(), 2U);
  ASSERT_TRUE(WaitFor([&] { return system.DiagnosticCount() == 2U; }));
}

TEST(PurePlanMotionServer,
     SuccessWithoutReferenceNormalizesToPlannerError) {
  RunningSystem system{[](const auto&) {
    return lunar::pure_planning::PlanningResult{
        .status = lunar::pure_planning::PlanningStatus::kSuccess,
        .reason_code = "PLAN_FOUND",
        .reference = std::nullopt};
  }};
  system.PublishInputs();

  const auto handle = system.SendGoal(system.Goal("missing_reference"));
  ASSERT_NE(handle, nullptr);
  const auto result = system.Result(handle);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  ASSERT_NE(result.result, nullptr);
  EXPECT_EQ(result.result->planning_outcome,
            Action::Result::NUMERICAL_FAILURE);
  EXPECT_EQ(result.result->reason_code, "PLANNER_ERROR");
  EXPECT_FALSE(result.result->has_reference);
  ASSERT_TRUE(WaitFor([&] { return system.DiagnosticCount() == 1U; }));
  const auto diagnostics = system.Diagnostics().front();
  ExpectBaseDiagnostic(diagnostics);
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "reason_code"),
            "PLANNER_ERROR");
}

TEST(PurePlanMotionServer, OuterWallTimeOverridesOnlyTotalTiming) {
  RunningSystem system{[](const auto&) {
    std::this_thread::sleep_for(8ms);
    return lunar::pure_planning::PlanningResult{
        .status = lunar::pure_planning::PlanningStatus::kNoPath,
        .reason_code = "NO_PATH",
        .timing = {.global_elapsed = 2ms,
                   .global_call_count = 1U,
                   .local_elapsed = 3ms,
                   .local_call_count = 1U,
                   .total_elapsed = 1ns}};
  }};
  system.PublishInputs();
  const auto handle = system.SendGoal(system.Goal("timing"));
  ASSERT_NE(handle, nullptr);
  const auto result = system.Result(handle);
  ASSERT_TRUE(WaitFor([&] { return system.DiagnosticCount() == 1U; }));
  const auto diagnostics = system.Diagnostics().front();
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "global_elapsed_ms"), "2");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "local_elapsed_ms"), "3");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "global_call_count"), "1");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "local_call_count"), "1");
  const double total_ms =
      std::stod(FindDiagnosticValue(diagnostics, "total_elapsed_ms"));
  EXPECT_GE(total_ms, 8.0);
  EXPECT_NEAR(result.result->diagnostics.elapsed_s * 1000.0, total_ms, 1.0);
}

TEST(PurePlanMotionServer,
     CertifiedResultFinalizedAfterTwoPointFiveSecondsIsPlanFoundLate) {
  RunningSystem system{[](const auto& request) {
    std::this_thread::sleep_for(2500ms);
    return Success(request);
  }};
  system.PublishInputs();

  const auto handle = system.SendGoal(system.Goal("late_success"));
  ASSERT_NE(handle, nullptr);
  const auto result = system.Result(handle);

  EXPECT_EQ(result.code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_NE(result.result, nullptr);
  EXPECT_EQ(result.result->reason_code, "PLAN_FOUND_LATE");
  EXPECT_TRUE(result.result->has_reference);
  EXPECT_EQ(result.result->diagnostics.warning_codes,
            (std::vector<std::string>{"TARGET_MISSED",
                                      "PLANNING_SLA_MISSED"}));
  ASSERT_TRUE(WaitFor([&] { return system.DiagnosticCount() == 1U; }));
  const auto diagnostics = system.Diagnostics().front();
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "reason_code"),
            "PLAN_FOUND_LATE");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "latency_class"),
            "SLA_MISSED");
}

TEST(PurePlanMotionServer,
     RollingDiscardsLateLocalIdentityAndPublishesOnlyTheFreshCycle) {
  std::atomic<std::uint64_t> calls{0U};
  std::atomic<bool> first_entered{false};
  std::atomic<bool> release_first{false};
  RunningSystem system{
      [](const lunar::pure_planning::PlanningRequest&) {
        return Failure(lunar::pure_planning::PlanningStatus::kPlannerError,
                       "PLANNER_ERROR");
      },
      "wheel", ConfigPath("wheel.yaml").string(),
      {rclcpp::Parameter{"rolling_surface_enabled", true},
       rclcpp::Parameter{"rolling_poll_period_ms", 10},
       rclcpp::Parameter{"rolling_min_replan_interval_ms", 10}},
      [&](const lunar::pure_planning::PlanningRequest& request,
          const lunar::pure_planning::LocalGoalSet& goals,
          lunar::pure_planning::SearchControl control) {
        const std::uint64_t call = ++calls;
        if (call == 1U) {
          first_entered = true;
          while (!release_first.load() && !control.canceled() &&
                 !control.expired()) {
            std::this_thread::sleep_for(1ms);
          }
        }
        return LocalSuccess(request, goals);
      }};
  system.PublishInputs();

  const auto handle = system.SendGoal(system.Goal("rolling_identity"));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor([&] { return first_entered.load(); }));
  system.PublishLocalOnly();
  std::this_thread::sleep_for(20ms);
  release_first = true;
  ASSERT_TRUE(WaitFor([&] { return calls.load() >= 2U; }));
  ASSERT_TRUE(WaitFor([&] {
    return std::ranges::count_if(system.WheeledPaths(), [](const auto& path) {
             return !path.poses.empty();
           }) == 1;
  }));
  ASSERT_TRUE(WaitFor([&] { return system.DiagnosticCount() >= 2U; }));
  const auto diagnostics = system.Diagnostics();
  ASSERT_GE(diagnostics.size(), 2U);
  EXPECT_EQ(FindDiagnosticValue(diagnostics[0], "reason_code"), "STALE_INPUT");
  EXPECT_EQ(FindDiagnosticValue(diagnostics[1], "reason_code"), "PLAN_FOUND");

  system.PublishOdometryOnly(0.2);
  const auto result = system.Result(handle);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::SUCCEEDED);
  EXPECT_EQ(calls.load(), 2U);
}

TEST(PurePlanMotionServer,
     RollingFirstLocalRequestKeepsTheColdGlobalTimingWindow) {
  struct TimingObservation final {
    std::chrono::steady_clock::duration start_age;
    std::chrono::steady_clock::duration window;
  };
  std::promise<TimingObservation> observed_promise;
  auto observed = observed_promise.get_future();
  std::atomic<bool> first{true};
  RunningSystem system{
      [](const lunar::pure_planning::PlanningRequest&) {
        return Failure(lunar::pure_planning::PlanningStatus::kPlannerError,
                       "PLANNER_ERROR");
      },
      "wheel", ConfigPath("wheel.yaml").string(),
      {rclcpp::Parameter{"rolling_surface_enabled", true}},
      [&](const lunar::pure_planning::PlanningRequest& request,
          const lunar::pure_planning::LocalGoalSet& goals,
          lunar::pure_planning::SearchControl) {
        const auto invoked = std::chrono::steady_clock::now();
        if (first.exchange(false)) {
          EXPECT_TRUE(request.request_started_at.has_value());
          observed_promise.set_value({
              .start_age = invoked - *request.request_started_at,
              .window = request.control.deadline - *request.request_started_at,
          });
        }
        return LocalSuccess(request, goals);
      }};
  system.PublishInputs(true, 0.0, 2048U);

  const auto handle = system.SendGoal(system.Goal("rolling_cold_global"));
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(observed.wait_for(4s), std::future_status::ready);
  const auto timing = observed.get();

  EXPECT_EQ(timing.window, 3s);
  EXPECT_GE(timing.start_age, 2ms);
  auto cancel = system.action_client->async_cancel_goal(handle);
  ASSERT_EQ(cancel.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(cancel.get()->return_code,
            action_msgs::srv::CancelGoal::Response::ERROR_NONE);
  EXPECT_EQ(system.Result(handle).code, rclcpp_action::ResultCode::CANCELED);
}

TEST(PurePlanMotionServer,
     RollingFirstCycleFinalizedAtTwoPointFiveSecondsIsPlanFoundLate) {
  RunningSystem system{
      [](const lunar::pure_planning::PlanningRequest&) {
        return Failure(lunar::pure_planning::PlanningStatus::kPlannerError,
                       "PLANNER_ERROR");
      },
      "wheel", ConfigPath("wheel.yaml").string(),
      {rclcpp::Parameter{"rolling_surface_enabled", true},
       rclcpp::Parameter{"rolling_poll_period_ms", 10},
       rclcpp::Parameter{"rolling_min_replan_interval_ms", 10}},
      [](const lunar::pure_planning::PlanningRequest& request,
         const lunar::pure_planning::LocalGoalSet& goals,
         lunar::pure_planning::SearchControl) {
        EXPECT_TRUE(request.request_started_at.has_value());
        std::this_thread::sleep_until(*request.request_started_at + 2500ms);
        return LocalSuccess(request, goals);
      }};
  system.PublishInputs();

  const auto handle = system.SendGoal(system.Goal("rolling_late"));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor(
      [&] {
        const auto paths = system.WheeledPaths();
        return std::ranges::any_of(paths, [](const auto& path) {
          return !path.poses.empty();
        });
      },
      4s));
  system.PublishInputs(true, 0.2);
  const auto result = system.Result(handle);

  EXPECT_EQ(result.code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_NE(result.result, nullptr);
  EXPECT_EQ(result.result->reason_code, "PLAN_FOUND_LATE");
  EXPECT_TRUE(result.result->has_reference);
}

TEST(PurePlanMotionServer,
     RollingCycleFinalizedAtHardDeadlinePublishesAnEmptyPath) {
  RunningSystem system{
      [](const lunar::pure_planning::PlanningRequest&) {
        return Failure(lunar::pure_planning::PlanningStatus::kPlannerError,
                       "PLANNER_ERROR");
      },
      "wheel", ConfigPath("wheel.yaml").string(),
      {rclcpp::Parameter{"rolling_surface_enabled", true},
       rclcpp::Parameter{"rolling_poll_period_ms", 10},
       rclcpp::Parameter{"rolling_min_replan_interval_ms", 10}},
      [](const lunar::pure_planning::PlanningRequest& request,
         const lunar::pure_planning::LocalGoalSet& goals,
         lunar::pure_planning::SearchControl) {
        EXPECT_TRUE(request.request_started_at.has_value());
        std::this_thread::sleep_until(*request.request_started_at + 3s);
        return LocalSuccess(request, goals);
      }};
  system.PublishInputs();

  const auto handle = system.SendGoal(system.Goal("rolling_hard"));
  ASSERT_NE(handle, nullptr);
  auto result_future = system.action_client->async_get_result(handle);
  const auto result_status = result_future.wait_for(4s);
  EXPECT_EQ(result_status, std::future_status::ready);
  ASSERT_TRUE(WaitFor([&] { return system.WheeledPaths().size() == 1U; }));
  EXPECT_TRUE(system.WheeledPaths().front().poses.empty());
  if (result_status != std::future_status::ready) {
    auto cancel = system.action_client->async_cancel_goal(handle);
    ASSERT_EQ(cancel.wait_for(3s), std::future_status::ready);
    EXPECT_EQ(cancel.get()->return_code,
              action_msgs::srv::CancelGoal::Response::ERROR_NONE);
    ASSERT_EQ(result_future.wait_for(3s), std::future_status::ready);
    (void)result_future.get();
    return;
  }
  const auto result = result_future.get();
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  ASSERT_NE(result.result, nullptr);
  EXPECT_EQ(result.result->reason_code, "TIMEOUT");
  EXPECT_FALSE(result.result->has_reference);
}

TEST(PurePlanMotionServer,
     FinalizationCrossingAbsoluteDeadlineCommitsTimeoutWithoutReference) {
  std::atomic<std::int64_t> core_remaining_ms{-1};
  RunningSystem system{
      [&](const lunar::pure_planning::PlanningRequest& request) {
        auto result = Success(request);
        result.timing.global_elapsed = 2ms;
        result.timing.global_call_count = 1U;
        result.timing.local_elapsed = 3ms;
        result.timing.local_call_count = 1U;
        auto& trajectory =
            std::get<lunar::pure_planning::TrajectoryReference>(
                result.reference->data);
        trajectory.points.resize(200'000U, trajectory.points.front());
        result.reference->preview.poses_map.resize(
            200'000U, result.reference->preview.poses_map.front());
        while (request.control.now() + 8ms < request.control.deadline) {
          std::this_thread::sleep_for(1ms);
        }
        core_remaining_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                request.control.deadline - request.control.now())
                .count();
        return result;
      }};
  system.PublishInputs();

  const auto handle = system.SendGoal(system.Goal("finalization_deadline"));
  ASSERT_NE(handle, nullptr);

  const auto result = system.Result(handle);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  ASSERT_NE(result.result, nullptr);
  EXPECT_EQ(result.result->planning_outcome,
            Action::Result::RESOURCE_EXHAUSTED);
  EXPECT_EQ(result.result->reason_code, "TIMEOUT");
  EXPECT_FALSE(result.result->has_reference);
  EXPECT_GE(result.result->diagnostics.elapsed_s, 3.0);
  EXPECT_GT(core_remaining_ms.load(), 0);
  EXPECT_LE(core_remaining_ms.load(), 8);
  ASSERT_TRUE(WaitFor([&] { return system.DiagnosticCount() == 1U; }));
  const auto diagnostics = system.Diagnostics().front();
  ExpectBaseDiagnostic(diagnostics);
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "planning_outcome"),
            std::to_string(Action::Result::RESOURCE_EXHAUSTED));
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "reason_code"), "TIMEOUT");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "global_elapsed_ms"), "2");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "global_call_count"), "1");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "local_elapsed_ms"), "3");
  EXPECT_EQ(FindDiagnosticValue(diagnostics, "local_call_count"), "1");
  const double total_ms =
      std::stod(FindDiagnosticValue(diagnostics, "total_elapsed_ms"));
  EXPECT_GE(total_ms, 3000.0);
  EXPECT_NEAR(result.result->diagnostics.elapsed_s * 1000.0, total_ms,
              1.0e-9);
}

}  // namespace
}  // namespace lunar::pure_planner_ros
