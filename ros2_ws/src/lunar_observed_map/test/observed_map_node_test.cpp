#include "lunar_observed_map/observed_map_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <lunar_planner_ros/grid_map_adapter.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include <gtest/gtest.h>

namespace lunar::observed_map {
namespace {

using namespace std::chrono_literals;
constexpr std::size_t kRawWidth = 320U;
constexpr std::size_t kRawHeight = 320U;

builtin_interfaces::msg::Time Stamp(const std::int64_t nanoseconds) {
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<std::int32_t>(nanoseconds / 1'000'000'000LL);
  stamp.nanosec =
      static_cast<std::uint32_t>(nanoseconds % 1'000'000'000LL);
  return stamp;
}

std_msgs::msg::Float32MultiArray Encode(const std::vector<float>& values) {
  std_msgs::msg::Float32MultiArray output;
  output.layout.dim.resize(2U);
  output.layout.dim[0].label = "column_index";
  output.layout.dim[0].size = kRawHeight;
  output.layout.dim[0].stride = kRawWidth * kRawHeight;
  output.layout.dim[1].label = "row_index";
  output.layout.dim[1].size = kRawWidth;
  output.layout.dim[1].stride = kRawWidth;
  output.data.resize(values.size());
  for (std::size_t y = 0U; y < kRawHeight; ++y) {
    for (std::size_t x = 0U; x < kRawWidth; ++x) {
      output.data[(kRawHeight - 1U - y) * kRawWidth +
                  (kRawWidth - 1U - x)] = values[y * kRawWidth + x];
    }
  }
  return output;
}

grid_map_msgs::msg::GridMap RawObservedMap(
    const std::int64_t nanoseconds = 2'000'000'000LL,
    const double center_x = 0.0, const double center_y = 0.0) {
  std::vector<float> elevation(kRawWidth * kRawHeight, 0.0F);
  std::vector<float> valid(kRawWidth * kRawHeight, 0.0F);
  const std::size_t center_row = 160U;
  for (std::size_t x = 160U; x <= 180U; ++x) {
    valid[center_row * kRawWidth + x] = 1.0F;
  }
  grid_map_msgs::msg::GridMap message;
  message.header.stamp = Stamp(nanoseconds);
  message.header.frame_id = "map";
  message.info.resolution = 0.2;
  message.info.length_x = 64.0;
  message.info.length_y = 64.0;
  message.info.pose.position.x = center_x;
  message.info.pose.position.y = center_y;
  message.info.pose.orientation.w = 1.0;
  message.layers = {"elevation", "valid_mask"};
  message.basic_layers = message.layers;
  message.data = {Encode(elevation), Encode(valid)};
  return message;
}

nav_msgs::msg::Odometry Odometry(
    const std::int64_t nanoseconds = 2'050'000'000LL,
    const double x = 0.1, const double y = 0.1) {
  nav_msgs::msg::Odometry message;
  message.header.stamp = Stamp(nanoseconds);
  message.header.frame_id = "odom";
  message.child_frame_id = "base_footprint";
  message.pose.pose.position.x = x;
  message.pose.pose.position.y = y;
  message.pose.pose.orientation.w = 1.0;
  return message;
}

geometry_msgs::msg::PoseStamped Goal(
    const std::int64_t nanoseconds = 2'050'000'000LL) {
  geometry_msgs::msg::PoseStamped message;
  message.header.stamp = Stamp(nanoseconds);
  message.header.frame_id = "map";
  message.pose.position.x = 2.1;
  message.pose.position.y = 0.1;
  message.pose.orientation.w = 1.0;
  return message;
}

rclcpp::NodeOptions Options() {
  return rclcpp::NodeOptions{}.parameter_overrides({
      rclcpp::Parameter{"initial_session_id", "session-a"},
  });
}

struct TestNode final {
  std::shared_ptr<ObservedMapNode> node{
      std::make_shared<ObservedMapNode>(Options())};

  void Activate() {
    ASSERT_EQ(
        node->configure().id(),
        lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
    ASSERT_EQ(
        node->activate().id(),
        lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  }

  void Ready() {
    Activate();
    node->ReceiveObservedMapForTesting(RawObservedMap());
    node->ReceiveOdometryForTesting(Odometry());
    ASSERT_EQ(node->state_for_testing(), ObservedMapNodeState::kReady);
  }
};

class RosContext final : public ::testing::Environment {
 public:
  void SetUp() override {
    if (!rclcpp::ok()) {
      int argc = 1;
      char name[] = "observed_map_node_test";
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

TEST(ObservedMapNode, ConfigureRequiresFrozenMapParameters) {
  (void)kRosContext;
  auto invalid = std::make_shared<ObservedMapNode>(
      rclcpp::NodeOptions{}.parameter_overrides({
          rclcpp::Parameter{"base_resolution_m", 0.25},
      }));
  EXPECT_EQ(
      invalid->configure().id(),
      lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);

  TestNode valid;
  ASSERT_EQ(
      valid.node->configure().id(),
      lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  const auto update = valid.node->set_parameter(
      rclcpp::Parameter{"maximum_tiles", 128});
  EXPECT_FALSE(update.successful);
  EXPECT_EQ(update.reason, "MAP_PARAMETERS_FROZEN_AFTER_CONFIGURE");
}

TEST(ObservedMapNode, ActivatePublishesInitialRobotCenteredGlobalMap) {
  TestNode test;
  test.Activate();
  auto observer = std::make_shared<rclcpp::Node>("map_product_observer");
  std::optional<grid_map_msgs::msg::GridMap> global;
  const auto subscription = observer->create_subscription<
      grid_map_msgs::msg::GridMap>(
      "/environment/map_global",
      rclcpp::QoS{1}.reliable().transient_local(),
      [&](const grid_map_msgs::msg::GridMap& message) { global = message; });
  (void)subscription;
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(observer);
  executor.add_node(test.node->get_node_base_interface());

  test.node->ReceiveObservedMapForTesting(RawObservedMap());
  test.node->ReceiveOdometryForTesting(Odometry());
  for (int attempt = 0; attempt < 20 && !global; ++attempt) {
    executor.spin_some();
    std::this_thread::sleep_for(5ms);
  }

  ASSERT_TRUE(global.has_value());
  EXPECT_EQ(global->header.frame_id, "map");
  EXPECT_TRUE(lunar::planning::ros::GridMapAdapter{}
                  .Adapt(*global, "map")
                  .ok());
  executor.remove_node(test.node->get_node_base_interface());
  executor.remove_node(observer);
}

TEST(ObservedMapNode, ObservedPatchAndOdometryProduceSameGenerationProducts) {
  TestNode test;
  test.Ready();
  const auto local = test.node->last_local_map_for_testing();
  const auto global = test.node->last_global_map_for_testing();
  ASSERT_TRUE(local.has_value());
  ASSERT_TRUE(global.has_value());
  EXPECT_EQ(local->header.stamp, global->header.stamp);
  EXPECT_EQ(test.node->generation_for_testing(), 1U);
  EXPECT_TRUE(lunar::planning::ros::GridMapAdapter{}
                  .Adapt(*local, "odom")
                  .ok());
  EXPECT_TRUE(lunar::planning::ros::GridMapAdapter{}
                  .Adapt(*global, "map")
                  .ok());
}

TEST(ObservedMapNode, OdometryWithoutNewObservationDoesNotAdvanceGeneration) {
  TestNode test;
  test.Ready();
  const std::uint64_t generation = test.node->generation_for_testing();
  const auto global_stamp =
      test.node->last_global_map_for_testing()->header.stamp;

  test.node->ReceiveOdometryForTesting(Odometry(2'100'000'000LL, 0.2, 0.1));

  EXPECT_EQ(test.node->generation_for_testing(), generation);
  EXPECT_EQ(
      test.node->last_global_map_for_testing()->header.stamp, global_stamp);
}

TEST(ObservedMapNode, FreezesArbitrarySessionLatticeOriginFromFirstPatch) {
  TestNode test;
  test.Activate();
  test.node->ReceiveObservedMapForTesting(
      RawObservedMap(2'000'000'000LL, 0.07, -0.03));
  test.node->ReceiveOdometryForTesting(
      Odometry(2'050'000'000LL, 0.17, 0.07));

  ASSERT_EQ(test.node->state_for_testing(), ObservedMapNodeState::kReady)
      << test.node->last_reason_for_testing();
  const auto global = test.node->last_global_map_for_testing();
  ASSERT_TRUE(global.has_value());
  const auto adapted =
      lunar::planning::ros::GridMapAdapter{}.Adapt(*global, "map");
  ASSERT_TRUE(adapted.ok());
  const std::size_t x = static_cast<std::size_t>(std::floor(
      (0.17 - adapted.map->origin_m.x) / adapted.map->resolution_m));
  const std::size_t y = static_cast<std::size_t>(std::floor(
      (0.07 - adapted.map->origin_m.y) / adapted.map->resolution_m));
  const auto& valid = std::get<std::vector<std::uint8_t>>(
      adapted.map->layers.at("valid_mask").values);
  ASSERT_LT(x, adapted.map->width);
  ASSERT_LT(y, adapted.map->height);
  EXPECT_EQ(valid[y * adapted.map->width + x], 1U);
}

TEST(ObservedMapNode, GoalStampForcesContainingGlobalGenerationBeforePlanning) {
  TestNode test;
  test.Ready();
  const std::uint64_t initial_generation =
      test.node->generation_for_testing();

  test.node->ReceiveGoalForTesting(Goal());

  const auto global = test.node->last_global_map_for_testing();
  ASSERT_TRUE(global.has_value());
  EXPECT_GT(test.node->generation_for_testing(), initial_generation);
  EXPECT_GE(global->header.stamp.sec, 2);
  const auto adapted =
      lunar::planning::ros::GridMapAdapter{}.Adapt(*global, "map");
  ASSERT_TRUE(adapted.ok());
  EXPECT_LE(adapted.map->origin_m.x, 2.1);
  EXPECT_GE(
      adapted.map->origin_m.x +
          adapted.map->resolution_m * adapted.map->width,
      2.1);
}

TEST(ObservedMapNode, TileBudgetErrorKeepsPreviousProducts) {
  TestNode test;
  test.Ready();
  const std::uint64_t generation = test.node->generation_for_testing();
  const auto previous = test.node->last_global_map_for_testing();
  ASSERT_TRUE(previous.has_value());

  ObservedPatch oversized;
  oversized.session_id = "session-a";
  oversized.simulation_time_ns = 3'000'000'000LL;
  oversized.cell_zero = {10'000, 0};
  oversized.width = 512U * kTileSideCells;
  oversized.height = 1U;
  oversized.elevation_m.assign(oversized.width, 0.0);
  oversized.valid.assign(oversized.width, 0U);
  for (std::size_t tile = 0U; tile < 512U; ++tile) {
    oversized.valid[tile * kTileSideCells] = 1U;
  }

  const FuseResult result = test.node->FusePatchForTesting(oversized);

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.reason_code, "MAP_TILE_BUDGET_EXCEEDED");
  EXPECT_EQ(test.node->generation_for_testing(), generation);
  EXPECT_EQ(
      test.node->last_global_map_for_testing()->header.stamp,
      previous->header.stamp);
}

TEST(ObservedMapNode, ResetSessionClearsProductsAndReturnsToSyncing) {
  TestNode test;
  test.Ready();

  diagnostic_msgs::msg::DiagnosticArray status;
  diagnostic_msgs::msg::DiagnosticStatus bridge;
  bridge.name = "lunar_unreal_tcp_bridge";
  diagnostic_msgs::msg::KeyValue session_state;
  session_state.key = "session_state";
  session_state.value = "SYNCING";
  diagnostic_msgs::msg::KeyValue session_id;
  session_id.key = "session_id";
  session_id.value = "session-b";
  bridge.values = {session_state, session_id};
  status.status.push_back(std::move(bridge));
  test.node->ReceiveBridgeStatusForTesting(status);

  EXPECT_EQ(test.node->state_for_testing(), ObservedMapNodeState::kSyncing);
  EXPECT_EQ(test.node->last_reason_for_testing(), "SESSION_RESET");
  EXPECT_FALSE(test.node->last_local_map_for_testing().has_value());
  EXPECT_FALSE(test.node->last_global_map_for_testing().has_value());
  EXPECT_EQ(test.node->generation_for_testing(), 0U);
}

}  // namespace
}  // namespace lunar::observed_map
