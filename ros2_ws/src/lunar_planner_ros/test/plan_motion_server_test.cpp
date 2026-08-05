#include "lunar_planner_ros/plan_motion_server.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <action_msgs/msg/goal_status.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <gtest/gtest.h>
#include <lifecycle_msgs/msg/state.hpp>
#include <lunar_navigation_msgs/msg/exploration_task.hpp>
#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/create_client.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "test_fixtures.hpp"

namespace lunar::planning::ros {
namespace {

using namespace std::chrono_literals;
using Action = lunar_planning_msgs::action::PlanMotion;
using ClientGoalHandle = rclcpp_action::ClientGoalHandle<Action>;

template<typename Predicate>
bool WaitFor(Predicate predicate, const std::chrono::milliseconds timeout = 3s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(5ms);
  }
  return predicate();
}

lunar::planning::PlannerOutput NoRouteOutput(
    const std::string& reason = "NO_ROUTE") {
  return lunar::planning::PlannerOutput{
      .outcome = lunar::planning::PlanningOutcome::kNoKnownSafeRoute,
      .directive = lunar::planning::ExecutionDirective::kHoldPosition,
      .reason_code = reason,
      .reference = std::nullopt,
      .diagnostics = {
          .planner_name = "cpp_v3",
          .elapsed = 1ms,
          .expanded_states = 12U,
          .best_cost = std::nullopt,
          .warning_codes = {},
      },
  };
}

LoadedCapabilities WheelCapabilities() {
  return LoadedCapabilities{
      .platform_id = "test-rover",
      .capability_version = "test-v1",
      .base_frame_id = "base_link",
      .reference_point = {},
      .actuator_profile_id = {},
      .maximum_obstacle_height_m = std::nullopt,
      .source_motion_primitive_ids = {},
      .urdf_path = {},
      .mesh_paths = {},
      .observation = {},
      .platform = test::MakeWheeledCapability(),
  };
}

lunar::planning::HopperCapability MakeHopperCapability() {
  return lunar::planning::HopperCapability{
      .body_half_extent_m = {0.2, 0.2, 0.2},
      .platform_mass_kg = 10.0,
      .gravity_mps2 = {0.0, 0.0, -1.62},
      .maximum_landing_slope_rad = 0.3,
      .maximum_landing_roughness_m = 0.05,
      .maximum_plane_residual_m = 0.02,
      .minimum_overhead_clearance_m = 0.5,
      .minimum_lateral_clearance_m = 0.2,
      .minimum_landing_region_area_m2 = 0.5,
      .maximum_launch_speed_mps = 5.0,
      .maximum_launch_impulse_newton_seconds = 50.0,
      .minimum_flight_time = 100ms,
      .maximum_flight_time = 10s,
      .maximum_landing_speed_mps = 0.2,
      .minimum_downward_impact_speed_mps = 0.1,
      .minimum_landing_clearance_m = 0.1,
      .maximum_angular_speed_radps = 0.1,
      .maximum_angular_acceleration_radps2 = 1.0,
      .maximum_initial_angular_speed_radps = 0.1,
      .minimum_settle_guard = 500ms,
  };
}

PlanMotionServerDependencies DefaultDependencies() {
  return PlanMotionServerDependencies{
      .planner = [](const lunar::planning::PlannerInput&) {
        return NoRouteOutput();
      },
      .preloaded_capabilities = WheelCapabilities(),
  };
}

rclcpp::NodeOptions ValidOptions() {
  rclcpp::NodeOptions options;
  options.parameter_overrides({
      rclcpp::Parameter{"global_map_max_age", 5.0},
      rclcpp::Parameter{"local_map_max_age", 5.0},
      rclcpp::Parameter{"odometry_max_age", 5.0},
      rclcpp::Parameter{"localization_status_max_age", 5.0},
      rclcpp::Parameter{"tf_max_age", 5.0},
      rclcpp::Parameter{"max_pairwise_skew", 0.2},
      rclcpp::Parameter{"base_resolution_m", 1.0},
      rclcpp::Parameter{"maximum_global_level", 4},
      rclcpp::Parameter{"maximum_global_cells", 1'048'576},
      rclcpp::Parameter{"maximum_global_axis_cells", 4'096},
  });
  return options;
}

class RunningSystem final {
 public:
  explicit RunningSystem(
      PlanMotionServerDependencies dependencies = DefaultDependencies(),
      const bool activate = true)
      : server(std::make_shared<PlanMotionServer>(
            ValidOptions(), std::move(dependencies))),
        client_node(std::make_shared<rclcpp::Node>(UniqueName())),
        executor(rclcpp::ExecutorOptions{}, 6U) {
    EXPECT_EQ(
        server->configure().id(),
        lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
    if (activate) {
      EXPECT_EQ(
          server->activate().id(),
          lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
    }

    global_map_publisher =
        client_node->create_publisher<grid_map_msgs::msg::GridMap>(
            "/environment/map_global",
            rclcpp::QoS{1}.reliable().transient_local());
    local_map_publisher =
        client_node->create_publisher<grid_map_msgs::msg::GridMap>(
            "/environment/map_local",
            rclcpp::QoS{1}.reliable().transient_local());
    odometry_publisher = client_node->create_publisher<nav_msgs::msg::Odometry>(
        "/localization/odometry", rclcpp::SensorDataQoS{});
    localization_publisher = client_node->create_publisher<
        lunar_navigation_msgs::msg::LocalizationStatus>(
        "/localization/status", rclcpp::QoS{10}.reliable());
    tf_publisher = client_node->create_publisher<tf2_msgs::msg::TFMessage>(
        "/tf", rclcpp::QoS{100}.best_effort());
    mission_publisher = client_node->create_publisher<
        lunar_navigation_msgs::msg::ExplorationTask>(
        "/mission/exploration_task",
        rclcpp::QoS{1}.reliable().transient_local());
    diagnostics_subscription = client_node->create_subscription<
        diagnostic_msgs::msg::DiagnosticArray>(
        "/diagnostics", rclcpp::QoS{10}.reliable(),
        [this](const diagnostic_msgs::msg::DiagnosticArray::SharedPtr message) {
          std::scoped_lock lock{diagnostics_mutex};
          diagnostics.push_back(*message);
        });
    action_client = rclcpp_action::create_client<Action>(
        client_node, "/plan_motion");

    executor.add_node(server->get_node_base_interface());
    executor.add_node(client_node);
    spin_thread = std::jthread([this] { executor.spin(); });
    EXPECT_TRUE(action_client->wait_for_action_server(3s));
  }

  ~RunningSystem() {
    const auto state = server->get_current_state().id();
    if (state == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
      server->deactivate();
    }
    if (server->get_current_state().id() ==
        lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE) {
      server->cleanup();
    }
    executor.cancel();
    if (spin_thread.joinable()) {
      spin_thread.join();
    }
    executor.remove_node(client_node);
    executor.remove_node(server->get_node_base_interface());
    diagnostics_subscription.reset();
    action_client.reset();
    client_node.reset();
    server.reset();
  }

  void PublishInputs(
      const std::uint64_t revision = 7U,
      const std::uint8_t desired_state =
          lunar_navigation_msgs::msg::ExplorationTask::ACTIVE) {
    const rclcpp::Time now = server->now();
    last_stamp = now;
    const std::int64_t nanoseconds = now.nanoseconds();

    auto global_map = test::MakeGridMap("map");
    global_map.header.stamp = last_stamp;
    auto local_map = test::MakeGridMap("odom");
    local_map.header.stamp = last_stamp;
    auto odometry = test::MakeOdometry(nanoseconds);
    auto localization = test::MakeLocalizationStatus(
        lunar_navigation_msgs::msg::LocalizationStatus::VALID,
        nanoseconds);
    auto transforms = test::MakeTransforms(nanoseconds);
    lunar_navigation_msgs::msg::ExplorationTask mission;
    mission.header.frame_id = "map";
    mission.header.stamp = last_stamp;
    mission.mission_id = "mission-1";
    mission.revision = revision;
    mission.desired_state = desired_state;
    mission.roi_min_x_m = -10.0;
    mission.roi_min_y_m = -10.0;
    mission.roi_max_x_m = 10.0;
    mission.roi_max_y_m = 10.0;

    for (std::size_t attempt = 0U; attempt < 3U; ++attempt) {
      global_map_publisher->publish(global_map);
      local_map_publisher->publish(local_map);
      odometry_publisher->publish(odometry);
      localization_publisher->publish(localization);
      tf_publisher->publish(transforms);
      mission_publisher->publish(mission);
      std::this_thread::sleep_for(20ms);
    }
    ASSERT_TRUE(WaitFor([&] {
      return server->mission_revision_for_testing() == revision;
    }));
    std::this_thread::sleep_for(30ms);
  }

  Action::Goal Goal(
      std::string request_id,
      const std::uint64_t revision = 7U,
      const bool replace = false) const {
    Action::Goal goal;
    goal.request_id = std::move(request_id);
    goal.mission_id = "mission-1";
    goal.mission_revision = revision;
    goal.replace_active_request = replace;
    goal.goal.header.frame_id = "map";
    goal.goal.header.stamp = last_stamp;
    if (goal.goal.header.stamp.sec == 0) {
      goal.goal.header.stamp = server->now();
    }
    goal.goal.goal_id = "goal-1";
    goal.goal.goal_type = goal.goal.POINT;
    goal.goal.point.x = 12.0;
    goal.goal.point.y = 0.5;
    goal.goal.position_tolerance_m = 0.25;
    goal.goal.yaw_tolerance_rad = 0.1;
    return goal;
  }

  ClientGoalHandle::SharedPtr SendGoal(
      const Action::Goal& goal,
      const rclcpp_action::Client<Action>::SendGoalOptions& options = {}) {
    auto future = action_client->async_send_goal(goal, options);
    EXPECT_EQ(future.wait_for(3s), std::future_status::ready);
    return future.get();
  }

  ClientGoalHandle::WrappedResult Result(
      const ClientGoalHandle::SharedPtr& goal_handle) {
    auto future = action_client->async_get_result(goal_handle);
    EXPECT_EQ(future.wait_for(3s), std::future_status::ready);
    return future.get();
  }

  [[nodiscard]] bool SawDiagnostic(const std::string& reason) const {
    std::scoped_lock lock{diagnostics_mutex};
    return std::ranges::any_of(diagnostics, [&](const auto& array) {
      return std::ranges::any_of(array.status, [&](const auto& status) {
        return status.message == reason;
      });
    });
  }

  [[nodiscard]] std::optional<std::string> DiagnosticValue(
      const std::string& reason,
      const std::string& key) const {
    std::scoped_lock lock{diagnostics_mutex};
    for (auto array = diagnostics.rbegin(); array != diagnostics.rend();
         ++array) {
      for (const auto& status : array->status) {
        if (status.message != reason) {
          continue;
        }
        const auto value = std::ranges::find_if(
            status.values, [&](const auto& candidate) {
              return candidate.key == key;
            });
        if (value != status.values.end()) {
          return value->value;
        }
      }
    }
    return std::nullopt;
  }

  std::shared_ptr<PlanMotionServer> server;
  std::shared_ptr<rclcpp::Node> client_node;
  rclcpp::executors::MultiThreadedExecutor executor;
  std::jthread spin_thread;
  rclcpp_action::Client<Action>::SharedPtr action_client;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr
      global_map_publisher;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr
      local_map_publisher;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher;
  rclcpp::Publisher<
      lunar_navigation_msgs::msg::LocalizationStatus>::SharedPtr
      localization_publisher;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_publisher;
  rclcpp::Publisher<
      lunar_navigation_msgs::msg::ExplorationTask>::SharedPtr
      mission_publisher;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_subscription;
  builtin_interfaces::msg::Time last_stamp;
  mutable std::mutex diagnostics_mutex;
  std::vector<diagnostic_msgs::msg::DiagnosticArray> diagnostics;

 private:
  static std::string UniqueName() {
    static std::atomic<std::uint64_t> sequence{0U};
    return "plan_motion_test_client_" +
        std::to_string(sequence.fetch_add(1U));
  }
};

class PlanMotionServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int argc = 0;
    char** argv = nullptr;
    rclcpp::init(argc, argv);
  }

  void TearDown() override {
    rclcpp::shutdown();
  }
};

TEST_F(PlanMotionServerTest, UsesFiveDistinctMutuallyExclusiveCallbackGroups) {
  auto node = std::make_shared<PlanMotionServer>(
      ValidOptions(), DefaultDependencies());
  EXPECT_EQ(node->callback_group_count_for_testing(), 5U);
  EXPECT_TRUE(node->callback_groups_mutually_exclusive_for_testing());
  node.reset();
}

TEST_F(PlanMotionServerTest, ConfigureFailsWhenAnyRequiredTimeLimitIsMissing) {
  auto node = std::make_shared<PlanMotionServer>(
      rclcpp::NodeOptions{}, DefaultDependencies());
  const auto state = node->configure();
  EXPECT_EQ(state.id(), lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  EXPECT_EQ(
      node->last_diagnostic_reason_for_testing(),
      "PLANNER_CONFIGURE_FAILED");
  node.reset();
}

TEST_F(PlanMotionServerTest, ConfiguresAndActivatesWithExplicitSnapshotPolicy) {
  auto node = std::make_shared<PlanMotionServer>(
      ValidOptions(), DefaultDependencies());
  EXPECT_EQ(
      node->configure().id(),
      lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  EXPECT_EQ(
      node->activate().id(),
      lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  EXPECT_FALSE(node->worker_active_for_testing());
  EXPECT_EQ(
      node->deactivate().id(),
      lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  EXPECT_EQ(
      node->cleanup().id(),
      lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  node.reset();
}

TEST_F(PlanMotionServerTest, RejectsUnsupportedGlobalMapConfiguration) {
  auto options = ValidOptions();
  options.append_parameter_override("maximum_global_level", 3);
  auto node = std::make_shared<PlanMotionServer>(
      options, DefaultDependencies());
  EXPECT_EQ(
      node->configure().id(),
      lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  EXPECT_EQ(
      node->last_diagnostic_reason_for_testing(),
      "PLANNER_CONFIGURE_FAILED");
  node.reset();
}

TEST_F(PlanMotionServerTest, RejectsGoalsWhileInactive) {
  RunningSystem system{DefaultDependencies(), false};
  EXPECT_EQ(system.SendGoal(system.Goal("inactive")), nullptr);
  EXPECT_EQ(
      system.server->last_diagnostic_reason_for_testing(),
      "PLANNER_NOT_ACTIVE");
}

TEST_F(PlanMotionServerTest, ReturnsNoRouteAsSucceededActionWithEmptyReference) {
  RunningSystem system;
  system.PublishInputs();
  std::mutex feedback_mutex;
  std::vector<std::uint8_t> phases;
  rclcpp_action::Client<Action>::SendGoalOptions options;
  options.feedback_callback =
      [&](ClientGoalHandle::SharedPtr,
          const std::shared_ptr<const Action::Feedback> feedback) {
        std::scoped_lock lock{feedback_mutex};
        phases.push_back(feedback->phase);
      };
  const auto handle = system.SendGoal(system.Goal("single"), options);
  ASSERT_NE(handle, nullptr);
  const auto wrapped = system.Result(handle);
  ASSERT_EQ(wrapped.code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_NE(wrapped.result, nullptr);
  EXPECT_EQ(wrapped.result->planning_outcome, Action::Result::NO_KNOWN_SAFE_ROUTE);
  EXPECT_EQ(wrapped.result->reason_code, "NO_ROUTE");
  EXPECT_FALSE(wrapped.result->has_reference);
  EXPECT_EQ(
      wrapped.result->reference,
      lunar_planning_msgs::msg::MotionReference{});
  EXPECT_EQ(wrapped.result->mission_revision, 7U);
  EXPECT_EQ(wrapped.result->diagnostics.expanded_states, 12U);
  EXPECT_TRUE(WaitFor([&] { return system.SawDiagnostic("NO_ROUTE"); }));
  EXPECT_TRUE(WaitFor([&] {
    std::scoped_lock lock{feedback_mutex};
    return std::ranges::find(phases, Action::Feedback::SEARCHING) !=
        phases.end();
  }));
}

TEST_F(PlanMotionServerTest, PublishesStableHierarchicalDiagnosticMetrics) {
  auto output = NoRouteOutput("HIERARCHICAL_NO_ROUTE");
  output.diagnostics.planner_name = "cpp_v3_hierarchical";
  output.diagnostics.hierarchical = lunar::planning::HierarchicalPlannerMetrics{
      .global_level = 2U,
      .global_resolution_m = 0.8,
      .global_cells = 62'500U,
      .global_elapsed = 12ms,
      .local_elapsed = 3ms,
      .global_expanded_states = 401U,
      .local_expanded_states = 51U,
      .global_open_peak = 91U,
      .estimated_work_memory_bytes = 4'096U,
      .raw_route_points = 80U,
      .simplified_route_points = 12U,
      .local_frontier_distance_m = 4.0,
      .local_attempts = 2U,
      .corridor_width_m = 0.6,
      .hopper_graph_nodes = 11U,
      .hopper_graph_edges = 17U,
      .hopper_route_hops = 3U,
      .hopper_certification_attempts = 1U,
  };
  RunningSystem system{PlanMotionServerDependencies{
      .planner = [output](const lunar::planning::PlannerInput&) {
        return output;
      },
      .preloaded_capabilities = WheelCapabilities(),
  }};
  system.PublishInputs();
  const auto handle = system.SendGoal(system.Goal("hierarchical-diagnostics"));
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(system.Result(handle).code, rclcpp_action::ResultCode::SUCCEEDED);

  const std::vector<std::string> expected_keys{
      "hierarchical_global_level",
      "hierarchical_global_resolution_m",
      "hierarchical_global_cells",
      "hierarchical_global_elapsed_s",
      "hierarchical_local_elapsed_s",
      "hierarchical_global_expanded_states",
      "hierarchical_local_expanded_states",
      "hierarchical_global_open_peak",
      "hierarchical_estimated_work_memory_bytes",
      "hierarchical_raw_route_points",
      "hierarchical_simplified_route_points",
      "hierarchical_local_frontier_distance_m",
      "hierarchical_local_attempts",
      "hierarchical_corridor_width_m",
      "hierarchical_hopper_graph_nodes",
      "hierarchical_hopper_graph_edges",
      "hierarchical_hopper_route_hops",
      "hierarchical_hopper_certification_attempts",
  };
  ASSERT_TRUE(WaitFor([&] {
    return std::ranges::all_of(expected_keys, [&](const auto& key) {
      return system.DiagnosticValue("HIERARCHICAL_NO_ROUTE", key).has_value();
    });
  }));
  EXPECT_EQ(
      system.DiagnosticValue(
          "HIERARCHICAL_NO_ROUTE", "hierarchical_global_level"),
      "2");
  EXPECT_EQ(
      system.DiagnosticValue(
          "HIERARCHICAL_NO_ROUTE", "hierarchical_global_cells"),
      "62500");
  EXPECT_DOUBLE_EQ(
      std::stod(*system.DiagnosticValue(
          "HIERARCHICAL_NO_ROUTE", "hierarchical_global_resolution_m")),
      0.8);
  EXPECT_DOUBLE_EQ(
      std::stod(*system.DiagnosticValue(
          "HIERARCHICAL_NO_ROUTE", "hierarchical_global_elapsed_s")),
      0.012);
}

TEST_F(PlanMotionServerTest, PublishesStableLocalTrajectoryEvidence) {
  auto output = NoRouteOutput("DISCRETE_REFERENCE_AVAILABLE");
  output.diagnostics.warning_codes = {
      "WHEEL_OPTIMIZATION_CONFIG_INVALID",
      "WHEEL_OPTIMIZATION_DISCRETE_FALLBACK",
  };
  output.diagnostics.local_trajectory =
      lunar::planning::LocalTrajectoryDiagnostics{
          .trajectory_mode =
              lunar::planning::TrajectoryMode::kDiscreteFallback,
          .start_anchor_error_m = 0.0,
          .endpoint_error_m = 0.125,
          .maximum_curvature_per_m = 0.75,
          .collision_validation =
              lunar::planning::CollisionValidation::kCertified,
          .smoothing_elapsed_s = 0.004,
          .landing_field_elapsed_s = 0.0,
      };
  RunningSystem system{PlanMotionServerDependencies{
      .planner = [output](const lunar::planning::PlannerInput&) {
        return output;
      },
      .preloaded_capabilities = WheelCapabilities(),
  }};
  system.PublishInputs();
  const auto handle = system.SendGoal(system.Goal("local-diagnostics"));
  ASSERT_NE(handle, nullptr);
  const auto wrapped = system.Result(handle);
  ASSERT_EQ(wrapped.code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_NE(wrapped.result, nullptr);
  EXPECT_EQ(
      wrapped.result->diagnostics.warning_codes,
      output.diagnostics.warning_codes);

  const std::vector<std::string> expected_keys{
      "trajectory_mode",
      "start_anchor_error_m",
      "endpoint_error_m",
      "maximum_curvature_per_m",
      "collision_validation",
      "smoothing_elapsed_s",
      "landing_field_elapsed_s",
      "warning_codes",
  };
  ASSERT_TRUE(WaitFor([&] {
    return std::ranges::all_of(expected_keys, [&](const auto& key) {
      return system.DiagnosticValue(
          "DISCRETE_REFERENCE_AVAILABLE", key).has_value();
    });
  }));
  EXPECT_EQ(
      system.DiagnosticValue(
          "DISCRETE_REFERENCE_AVAILABLE", "trajectory_mode"),
      "DISCRETE_FALLBACK");
  EXPECT_EQ(
      system.DiagnosticValue(
          "DISCRETE_REFERENCE_AVAILABLE", "collision_validation"),
      "CERTIFIED");
  EXPECT_EQ(
      system.DiagnosticValue(
          "DISCRETE_REFERENCE_AVAILABLE", "warning_codes"),
      "WHEEL_OPTIMIZATION_CONFIG_INVALID,"
      "WHEEL_OPTIMIZATION_DISCRETE_FALLBACK");
  EXPECT_DOUBLE_EQ(
      std::stod(*system.DiagnosticValue(
          "DISCRETE_REFERENCE_AVAILABLE", "endpoint_error_m")),
      0.125);
}

TEST_F(PlanMotionServerTest, RejectsSecondGoalAndSerializesExplicitReplacement) {
  struct PlannerState final {
    std::atomic<int> calls{0};
    std::atomic<int> active{0};
    std::atomic<int> maximum_active{0};
  } state;
  auto planner = [&](const lunar::planning::PlannerInput& input) {
    const int call = state.calls.fetch_add(1) + 1;
    const int active = state.active.fetch_add(1) + 1;
    int maximum = state.maximum_active.load();
    while (active > maximum &&
           !state.maximum_active.compare_exchange_weak(maximum, active)) {}
    if (call == 1) {
      while (!input.stop_token.stop_requested()) {
        std::this_thread::sleep_for(1ms);
      }
    }
    state.active.fetch_sub(1);
    return call == 1
        ? lunar::planning::PlannerOutput{
              .outcome = lunar::planning::PlanningOutcome::kCanceled,
              .directive =
                  lunar::planning::ExecutionDirective::kHoldPosition,
              .reason_code = "REQUEST_CANCELED",
              .reference = std::nullopt,
              .diagnostics = {}}
        : NoRouteOutput("REPLACEMENT_COMPLETE");
  };
  RunningSystem system{PlanMotionServerDependencies{
      .planner = planner,
      .preloaded_capabilities = WheelCapabilities(),
  }};
  system.PublishInputs();

  const auto first = system.SendGoal(system.Goal("first"));
  ASSERT_NE(first, nullptr);
  ASSERT_TRUE(WaitFor([&] { return state.calls.load() == 1; }));
  EXPECT_EQ(system.SendGoal(system.Goal("second", 7U, false)), nullptr);

  const auto replacement =
      system.SendGoal(system.Goal("replacement", 7U, true));
  ASSERT_NE(replacement, nullptr);
  const auto first_result = system.Result(first);
  EXPECT_EQ(first_result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(first_result.result->planning_outcome, Action::Result::CANCELED);
  EXPECT_EQ(first_result.result->reason_code, "REQUEST_REPLACED");
  const auto replacement_result = system.Result(replacement);
  EXPECT_EQ(replacement_result.code, rclcpp_action::ResultCode::SUCCEEDED);
  EXPECT_EQ(replacement_result.result->reason_code, "REPLACEMENT_COMPLETE");
  EXPECT_EQ(state.maximum_active.load(), 1);
  EXPECT_EQ(state.calls.load(), 2);
}

TEST_F(PlanMotionServerTest, CooperativelyCancelsActiveGoal) {
  std::atomic<int> calls{0};
  RunningSystem system{PlanMotionServerDependencies{
      .planner = [&](const lunar::planning::PlannerInput& input) {
        calls.fetch_add(1);
        while (!input.stop_token.stop_requested()) {
          std::this_thread::sleep_for(1ms);
        }
        return lunar::planning::PlannerOutput{
            .outcome = lunar::planning::PlanningOutcome::kCanceled,
            .directive = lunar::planning::ExecutionDirective::kHoldPosition,
            .reason_code = "REQUEST_CANCELED",
            .reference = std::nullopt,
            .diagnostics = {},
        };
      },
      .preloaded_capabilities = WheelCapabilities(),
  }};
  system.PublishInputs();
  const auto handle = system.SendGoal(system.Goal("cancel"));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor([&] { return calls.load() == 1; }));

  auto cancel_future = system.action_client->async_cancel_goal(handle);
  ASSERT_EQ(cancel_future.wait_for(3s), std::future_status::ready);
  ASSERT_FALSE(cancel_future.get()->goals_canceling.empty());
  const auto result = system.Result(handle);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::CANCELED);
  EXPECT_EQ(result.result->planning_outcome, Action::Result::CANCELED);
  EXPECT_EQ(result.result->reason_code, "REQUEST_CANCELED");
}

TEST_F(PlanMotionServerTest, RejectsOldRevisionAndPausingCancelsUncommittedWork) {
  std::atomic<int> calls{0};
  RunningSystem system{PlanMotionServerDependencies{
      .planner = [&](const lunar::planning::PlannerInput& input) {
        calls.fetch_add(1);
        while (!input.stop_token.stop_requested()) {
          std::this_thread::sleep_for(1ms);
        }
        return lunar::planning::PlannerOutput{
            .outcome = lunar::planning::PlanningOutcome::kCanceled,
            .directive = lunar::planning::ExecutionDirective::kHoldPosition,
            .reason_code = "REQUEST_CANCELED",
            .reference = std::nullopt,
            .diagnostics = {},
        };
      },
      .preloaded_capabilities = WheelCapabilities(),
  }};
  system.PublishInputs(7U);
  EXPECT_EQ(system.SendGoal(system.Goal("old", 6U)), nullptr);

  const auto active = system.SendGoal(system.Goal("active", 7U));
  ASSERT_NE(active, nullptr);
  ASSERT_TRUE(WaitFor([&] { return calls.load() == 1; }));
  system.PublishInputs(
      7U, lunar_navigation_msgs::msg::ExplorationTask::PAUSED);
  const auto result = system.Result(active);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(result.result->reason_code, "MISSION_PAUSED");
  EXPECT_EQ(system.SendGoal(system.Goal("paused", 7U)), nullptr);
}

TEST_F(PlanMotionServerTest, LocksNewGoalsAfterActivatingHopperReference) {
  LoadedCapabilities hopper_capabilities{
      .platform_id = "test-hopper",
      .capability_version = "test-v1",
      .base_frame_id = "base_link",
      .reference_point = {},
      .actuator_profile_id = {},
      .maximum_obstacle_height_m = std::nullopt,
      .source_motion_primitive_ids = {},
      .urdf_path = {},
      .mesh_paths = {},
      .observation = {},
      .platform = MakeHopperCapability(),
  };
  RunningSystem system{PlanMotionServerDependencies{
      .planner = [](const lunar::planning::PlannerInput& input) {
        lunar::planning::HopReference hops{
            .segments = {lunar::planning::HopSegment{
                .segment_id = "hop-1",
                .launch_pose = {},
                .landing_region_boundary_m = {
                    {1.0, -1.0, 0.0}, {3.0, -1.0, 0.0},
                    {3.0, 1.0, 0.0}, {1.0, 1.0, 0.0}},
                .flight_time = 5s,
                .launch_velocity_mps = {1.0, 0.0, 2.0},
                .flight_tube_radius_m = 0.2,
            }},
        };
        return lunar::planning::PlannerOutput{
            .outcome =
                lunar::planning::PlanningOutcome::kNewReferenceAvailable,
            .directive =
                lunar::planning::ExecutionDirective::kActivateNewReference,
            .reason_code = "HOP_READY",
            .reference = lunar::planning::MotionReference{
                .plan_id = "hop-plan",
                .platform_type = lunar::planning::PlatformType::kHopper,
                .input_time = input.state_time,
                .preview = lunar::planning::GlobalRoutePreview{
                    .poses_map = {
                        lunar::planning::Pose3{
                            .position_m = {0.0, 0.0, 0.0},
                            .orientation = {},
                        },
                        lunar::planning::Pose3{
                            .position_m = {2.0, 0.0, 0.0},
                            .orientation = {},
                        },
                    },
                },
                .data = std::move(hops),
            },
            .diagnostics = {},
        };
      },
      .preloaded_capabilities = std::move(hopper_capabilities),
  }};
  system.PublishInputs();
  const auto first = system.SendGoal(system.Goal("hop"));
  ASSERT_NE(first, nullptr);
  const auto first_result = system.Result(first);
  ASSERT_EQ(first_result.code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_TRUE(first_result.result->has_reference);
  EXPECT_EQ(first_result.result->reference.header.frame_id, "map");
  EXPECT_EQ(first_result.result->reference.path_preview.header.frame_id, "map");
  ASSERT_EQ(first_result.result->reference.hops.size(), 1U);
  EXPECT_EQ(first_result.result->reference.hops.front().header.frame_id, "odom");

  EXPECT_EQ(system.SendGoal(system.Goal("replace-hop", 7U, true)), nullptr);
  const std::string reason =
      system.server->last_diagnostic_reason_for_testing();
  EXPECT_TRUE(reason == "HOP_IN_FLIGHT" || reason == "HOP_JUMP_COMMITTED");
}

TEST_F(PlanMotionServerTest, InternalResultInvariantUsesRecoverableErrorPath) {
  RunningSystem system{PlanMotionServerDependencies{
      .planner = [](const lunar::planning::PlannerInput&) {
        return lunar::planning::PlannerOutput{
            .outcome =
                lunar::planning::PlanningOutcome::kNewReferenceAvailable,
            .directive =
                lunar::planning::ExecutionDirective::kActivateNewReference,
            .reason_code = "BROKEN_OUTPUT",
            .reference = std::nullopt,
            .diagnostics = {},
        };
      },
      .preloaded_capabilities = WheelCapabilities(),
  }};
  system.PublishInputs();
  const auto handle = system.SendGoal(system.Goal("broken"));
  ASSERT_NE(handle, nullptr);
  const auto result = system.Result(handle);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(result.result->reason_code, "REFERENCE_REQUIRED_BY_OUTCOME");
  ASSERT_TRUE(WaitFor([&] {
    return system.server->get_current_state().id() ==
        lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED;
  }));
  EXPECT_EQ(
      system.server->configure().id(),
      lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  EXPECT_EQ(
      system.server->activate().id(),
      lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
}

}  // namespace
}  // namespace lunar::planning::ros
