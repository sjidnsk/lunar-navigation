#include "lunar_goal_coordinator/goal_coordinator_node.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <gtest/gtest.h>

namespace lunar::goal_coordinator {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;

builtin_interfaces::msg::Time Stamp(const std::int64_t nanoseconds) {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<std::int32_t>(nanoseconds / 1'000'000'000LL);
  stamp.nanosec = static_cast<std::uint32_t>(nanoseconds % 1'000'000'000LL);
  return stamp;
}

std_msgs::msg::Float32MultiArray Encode(
    const std::vector<float>& values, const std::size_t width,
    const std::size_t height) {
  std_msgs::msg::Float32MultiArray output;
  output.layout.dim.resize(2U);
  output.layout.dim[0].label = "column_index";
  output.layout.dim[0].size = height;
  output.layout.dim[0].stride = width * height;
  output.layout.dim[1].label = "row_index";
  output.layout.dim[1].size = width;
  output.layout.dim[1].stride = width;
  output.data.resize(values.size());
  for (std::size_t y = 0U; y < height; ++y) {
    for (std::size_t x = 0U; x < width; ++x) {
      output.data[(height - 1U - y) * width + (width - 1U - x)] =
          values[y * width + x];
    }
  }
  return output;
}

grid_map_msgs::msg::GridMap Map(
    const std::string& frame, const std::int64_t stamp_ns) {
  constexpr std::size_t width = 5U;
  constexpr std::size_t height = 5U;
  const std::size_t count = width * height;
  grid_map_msgs::msg::GridMap map;
  map.header.frame_id = frame;
  map.header.stamp = Stamp(stamp_ns);
  map.info.resolution = 1.0;
  map.info.length_x = 5.0;
  map.info.length_y = 5.0;
  map.info.pose.position.x = 2.5;
  map.info.pose.position.y = 2.5;
  map.info.pose.orientation.w = 1.0;
  map.layers = {
      "elevation", "valid_mask", "obstacle", "obstacle_height",
      "observation_age_s", "observation_quality", "elevation_variance",
      "obstacle_variance", "observation_count", "forbidden"};
  map.basic_layers = map.layers;
  map.data = {
      Encode(std::vector<float>(count, 3.0F), width, height),
      Encode(std::vector<float>(count, 1.0F), width, height),
      Encode(std::vector<float>(count, 0.0F), width, height),
      Encode(std::vector<float>(count, 0.0F), width, height),
      Encode(std::vector<float>(count, 0.0F), width, height),
      Encode(std::vector<float>(count, 1.0F), width, height),
      Encode(std::vector<float>(count, 0.0F), width, height),
      Encode(std::vector<float>(count, 0.0F), width, height),
      Encode(std::vector<float>(count, 1.0F), width, height),
      Encode(std::vector<float>(count, 0.0F), width, height),
  };
  return map;
}

nav_msgs::msg::Odometry Odometry(
    const std::int64_t stamp_ns, const double x = 0.5,
    const double y = 0.5, const double yaw = 0.0) {
  nav_msgs::msg::Odometry message;
  message.header.frame_id = "odom";
  message.header.stamp = Stamp(stamp_ns);
  message.child_frame_id = "base_footprint";
  message.pose.pose.position.x = x;
  message.pose.pose.position.y = y;
  message.pose.pose.orientation.z = std::sin(yaw * 0.5);
  message.pose.pose.orientation.w = std::cos(yaw * 0.5);
  return message;
}

tf2_msgs::msg::TFMessage Tf(const std::int64_t stamp_ns) {
  tf2_msgs::msg::TFMessage message;
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "map";
  transform.child_frame_id = "odom";
  transform.header.stamp = Stamp(stamp_ns);
  transform.transform.rotation.w = 1.0;
  message.transforms.push_back(transform);
  return message;
}

geometry_msgs::msg::PoseStamped Goal(const std::int64_t stamp_ns) {
  geometry_msgs::msg::PoseStamped message;
  message.header.frame_id = "map";
  message.header.stamp = Stamp(stamp_ns);
  message.pose.position.x = 4.5;
  message.pose.position.y = 4.5;
  message.pose.orientation.w = 1.0;
  return message;
}

diagnostic_msgs::msg::DiagnosticArray Session() {
  diagnostic_msgs::msg::DiagnosticArray message;
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "lunar_unreal_tcp_bridge";
  diagnostic_msgs::msg::KeyValue id;
  id.key = "session_id";
  id.value = "session-a";
  diagnostic_msgs::msg::KeyValue state;
  state.key = "session_state";
  state.value = "READY";
  status.values = {id, state};
  message.status.push_back(status);
  return message;
}

class FakePlanner final : public PlanMotionTransport {
 public:
  bool available{true};
  std::vector<Action::Goal> goals;
  std::vector<ResultCallback> callbacks;
  std::size_t cancel_count{};

  bool Available() const override { return available; }

  void Send(const Action::Goal& goal, ResultCallback callback) override {
    goals.push_back(goal);
    callbacks.push_back(std::move(callback));
  }

  void Cancel(CancelCallback callback) override {
    ++cancel_count;
    callback(true);
  }

  void Succeed(const std::size_t index, const std::string& plan_id) {
    PlanCompletion completion;
    completion.action_succeeded = true;
    completion.reason_code = "PLAN_MOTION_SUCCEEDED";
    completion.result.planning_outcome =
        completion.result.NEW_REFERENCE_AVAILABLE;
    completion.result.execution_directive =
        completion.result.ACTIVATE_NEW_REFERENCE;
    completion.result.reason_code = "REFERENCE_READY";
    completion.result.mission_revision = goals[index].mission_revision;
    completion.result.has_reference = true;
    completion.result.reference.header.frame_id = "odom";
    completion.result.reference.plan_id = plan_id;
    completion.result.reference.platform_type =
        completion.result.reference.WHEELED;
    callbacks[index](std::move(completion));
  }
};

class RosContext final : public ::testing::Environment {
 public:
  void SetUp() override {
    if (!rclcpp::ok()) {
      int argc = 1;
      char name[] = "goal_coordinator_node_test";
      char* argv[] = {name, nullptr};
      rclcpp::init(argc, argv);
    }
  }

  void TearDown() override {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

const auto* const kRosContext =
    ::testing::AddGlobalTestEnvironment(new RosContext);

struct RunningNode final {
  std::shared_ptr<FakePlanner> planner{std::make_shared<FakePlanner>()};
  std::size_t hold_count{};
  std::shared_ptr<GoalCoordinatorNode> node{
      std::make_shared<GoalCoordinatorNode>(
          rclcpp::NodeOptions{},
          GoalCoordinatorDependencies{
              .planner = planner,
              .target_uuid = [] { return "target-a"; },
              .request_hold = [this] { ++hold_count; },
          })};

  void Activate() {
    ASSERT_EQ(
        node->configure().id(),
        lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
    ASSERT_EQ(
        node->activate().id(),
        lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
    node->ReceiveBridgeStatusForTesting(Session());
  }

  void Snapshot(
      const std::int64_t stamp_ns, const double x = 0.5,
      const double y = 0.5, const double yaw = 0.0) {
    node->ReceiveGlobalMapForTesting(Map("map", stamp_ns));
    node->ReceiveLocalMapForTesting(Map("odom", stamp_ns));
    node->ReceiveOdometryForTesting(Odometry(stamp_ns, x, y, yaw));
    node->ReceiveTfForTesting(Tf(stamp_ns));
  }

  void GoalAndPlan(
      const std::int64_t map_stamp_ns = 2'000'000'000LL,
      const std::int64_t goal_stamp_ns = 1'000'000'000LL) {
    Activate();
    Snapshot(map_stamp_ns);
    node->ReceiveGoalForTesting(Goal(goal_stamp_ns));
    node->DispatchForTesting();
    ASSERT_EQ(planner->goals.size(), 1U);
  }

  lunar_navigation_msgs::msg::MotionExecutionFeedback Feedback(
      const std::uint8_t state, const std::int64_t stamp_ns,
      const std::string& plan_id = "plan-a") const {
    lunar_navigation_msgs::msg::MotionExecutionFeedback message;
    message.header.frame_id = "odom";
    message.header.stamp = Stamp(stamp_ns);
    message.platform_type = message.WHEELED;
    message.plan_id = plan_id;
    message.segment_id = plan_id;
    message.sequence = 1U;
    message.state = state;
    message.reason_code = state == message.CANCELED ? "USER_HOLD" : "OK";
    return message;
  }
};

TEST(GoalCoordinatorNode, GoalWaitsForContainingGlobalMapGeneration) {
  (void)kRosContext;
  RunningNode test;
  test.Activate();
  test.Snapshot(2'000'000'000LL);
  test.node->ReceiveGoalForTesting(Goal(3'000'000'000LL));
  EXPECT_TRUE(test.planner->goals.empty());
  EXPECT_EQ(test.node->state_for_testing(), CoordinatorState::kWaitingForMap);

  test.Snapshot(4'000'000'000LL);
  test.node->DispatchForTesting();
  EXPECT_EQ(test.planner->goals.size(), 1U);
}

TEST(GoalCoordinatorNode, SendsPlanMotionWithReplaceFalse) {
  RunningNode test;
  test.GoalAndPlan();
  EXPECT_FALSE(test.planner->goals.front().replace_active_request);
  EXPECT_EQ(test.planner->goals.front().goal.goal_type,
            test.planner->goals.front().goal.POINT);
  EXPECT_DOUBLE_EQ(
      test.planner->goals.front().goal.position_tolerance_m, 0.5);
  EXPECT_TRUE(test.planner->goals.front().goal.has_yaw_constraint);
}

TEST(GoalCoordinatorNode, PublishesMinimalMissionBeforeActionGoal) {
  RunningNode test;
  test.GoalAndPlan();
  const auto mission = test.node->last_mission_for_testing();
  ASSERT_TRUE(mission);
  EXPECT_EQ(mission->mission_id, test.planner->goals.front().mission_id);
  EXPECT_EQ(mission->revision, test.planner->goals.front().mission_revision);
  EXPECT_EQ(mission->desired_state, mission->ACTIVE);
  EXPECT_TRUE(mission->science_regions.empty());
  EXPECT_LT(mission->roi_min_x_m, mission->roi_max_x_m);
  EXPECT_LT(mission->roi_min_y_m, mission->roi_max_y_m);
}

TEST(GoalCoordinatorNode, PublishesReturnedReferenceOnce) {
  RunningNode test;
  test.GoalAndPlan();
  test.planner->Succeed(0U, "plan-a");
  const auto reference = test.node->last_reference_for_testing();
  ASSERT_TRUE(reference);
  EXPECT_EQ(reference->plan_id, "plan-a");
  EXPECT_EQ(test.node->state_for_testing(), CoordinatorState::kExecuting);
  test.node->DispatchForTesting();
  EXPECT_EQ(test.planner->goals.size(), 1U);
}

TEST(GoalCoordinatorNode, MatchingFeedbackAndNewSnapshotTriggerSecondPlan) {
  RunningNode test;
  test.GoalAndPlan();
  test.planner->Succeed(0U, "plan-a");
  test.node->ReceiveExecutionFeedbackForTesting(
      test.Feedback(
          lunar_navigation_msgs::msg::MotionExecutionFeedback::SEGMENT_COMPLETE,
          3'000'000'000LL));
  test.Snapshot(3'000'000'000LL, 2.0, 2.0);
  EXPECT_EQ(test.planner->goals.size(), 1U);
  test.Snapshot(4'000'000'000LL, 2.0, 2.0);
  test.node->DispatchForTesting();
  ASSERT_EQ(test.planner->goals.size(), 2U);
  EXPECT_EQ(
      test.planner->goals[1].mission_id, test.planner->goals[0].mission_id);
  EXPECT_NE(
      test.planner->goals[1].request_id, test.planner->goals[0].request_id);
}

TEST(GoalCoordinatorNode, CancelWaitsForCanceledThenAcceptsNewGoal) {
  RunningNode test;
  test.GoalAndPlan();
  test.planner->Succeed(0U, "plan-a");
  ASSERT_TRUE(test.node->CancelForTesting());
  EXPECT_EQ(test.node->state_for_testing(), CoordinatorState::kCanceling);
  test.node->ReceiveGoalForTesting(Goal(1'000'000'000LL));
  EXPECT_EQ(test.node->state_for_testing(), CoordinatorState::kCanceling);

  test.node->ReceiveExecutionFeedbackForTesting(
      test.Feedback(
          lunar_navigation_msgs::msg::MotionExecutionFeedback::CANCELED,
          3'000'000'000LL));
  EXPECT_EQ(test.node->state_for_testing(), CoordinatorState::kIdle);
  test.node->ReceiveGoalForTesting(Goal(1'000'000'000LL));
  EXPECT_EQ(test.node->state_for_testing(), CoordinatorState::kPlanning);
  EXPECT_EQ(test.planner->goals.size(), 1U);
  test.node->DispatchForTesting();
  EXPECT_EQ(test.planner->goals.size(), 2U);
}

TEST(GoalCoordinatorNode, LaunchGraphHasSingleMotionReferencePublisher) {
  RunningNode test;
  test.Activate();
  EXPECT_EQ(test.node->count_publishers("/lunar/motion_reference"), 1U);
  EXPECT_EQ(test.node->count_subscribers("/lunar/motion_reference"), 0U);
}

}  // namespace
}  // namespace lunar::goal_coordinator
