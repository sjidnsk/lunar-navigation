#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

#include "lunar_pure_exploration_sim/simulation_node.hpp"

namespace lunar::pure_exploration_sim {
namespace {

class RosContextTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }

  static void TearDownTestSuite() {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }

  static std::shared_ptr<SimulationNode> MakeNode() {
    rclcpp::NodeOptions options;
    options.parameter_overrides({
        rclcpp::Parameter("command_topic", "test/command"),
        rclcpp::Parameter("global_overview_topic", "test/global"),
        rclcpp::Parameter("local_grid_map_topic", "test/local"),
        rclcpp::Parameter("odometry_topic", "test/odometry"),
        rclcpp::Parameter("tf_topic", "test/tf"),
        rclcpp::Parameter("sensor_fov_topic", "test/fov"),
        rclcpp::Parameter("vehicle_markers_topic", "test/vehicle"),
        rclcpp::Parameter("local_map_markers_topic", "test/local_markers"),
        rclcpp::Parameter("actual_path_topic", "test/path"),
        rclcpp::Parameter("sim_elapsed_topic", "test/elapsed"),
    });
    return std::make_shared<SimulationNode>(options, false);
  }

  static rclcpp::NodeOptions OptionsWith(const rclcpp::Parameter& parameter) {
    rclcpp::NodeOptions options;
    options.parameter_overrides({parameter});
    return options;
  }
};

TEST_F(RosContextTest, RejectsInvalidSpeedMultiplierBeforeInitialPublish) {
  const std::vector<double> invalid_values{
      0.0, -1.0, std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()};
  for (const double value : invalid_values) {
    SCOPED_TRACE(value);
    EXPECT_THROW(
        SimulationNode(OptionsWith(rclcpp::Parameter("speed_multiplier", value)),
                       false),
        std::invalid_argument);
  }
}

TEST_F(RosContextTest, RejectsNonContractSensorRangeBeforeInitialPublish) {
  const std::vector<double> invalid_values{
      9.9, 10.1, std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()};
  for (const double value : invalid_values) {
    SCOPED_TRACE(value);
    EXPECT_THROW(
        SimulationNode(OptionsWith(rclcpp::Parameter("sensor_range_m", value)),
                       false),
        std::invalid_argument);
  }
}

TEST_F(RosContextTest, RejectsNonContractSensorFovBeforeInitialPublish) {
  const std::vector<double> invalid_values{
      89.0, 91.0, std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()};
  for (const double value : invalid_values) {
    SCOPED_TRACE(value);
    EXPECT_THROW(
        SimulationNode(OptionsWith(rclcpp::Parameter("sensor_fov_deg", value)),
                       false),
        std::invalid_argument);
  }
}

TEST_F(RosContextTest, InitialSnapshotContainsObservationAndFrozenInterfaces) {
  const auto node =
      std::make_shared<SimulationNode>(rclcpp::NodeOptions{}, false);

  EXPECT_EQ(node->get_parameter("seed").as_int(), 20260824);
  EXPECT_DOUBLE_EQ(node->get_parameter("speed_multiplier").as_double(), 20.0);
  EXPECT_DOUBLE_EQ(node->get_parameter("sensor_range_m").as_double(), 10.0);
  EXPECT_DOUBLE_EQ(node->get_parameter("sensor_fov_deg").as_double(), 90.0);
  EXPECT_DOUBLE_EQ(node->get_parameter("update_rate_hz").as_double(), 20.0);
  EXPECT_EQ(node->get_parameter("command_topic").as_string(),
            "/Car/T5/Car_Cmd_Vel");
  EXPECT_EQ(node->get_parameter("global_overview_topic").as_string(),
            "/Car/T3/mapping/global_overview");
  EXPECT_EQ(node->get_parameter("local_grid_map_topic").as_string(),
            "/Car/T3/mapping/grid_map");
  EXPECT_EQ(node->get_parameter("odometry_topic").as_string(),
            "/Car/T3/localization/odometry");
  EXPECT_EQ(node->get_parameter("tf_topic").as_string(), "/tf");
  EXPECT_EQ(node->get_parameter("sensor_fov_topic").as_string(),
            "/Car/T4/simulation/sensor_fov");
  EXPECT_EQ(node->get_parameter("vehicle_markers_topic").as_string(),
            "/Car/T4/simulation/vehicle_markers");
  EXPECT_EQ(node->get_parameter("local_map_markers_topic").as_string(),
            "/Car/T4/simulation/local_map_markers");
  EXPECT_EQ(node->get_parameter("actual_path_topic").as_string(),
            "/Car/T4/simulation/actual_path");
  EXPECT_EQ(node->get_parameter("sim_elapsed_topic").as_string(),
            "/Car/T4/simulation/sim_elapsed");

  const auto& messages = node->latest_messages();
  EXPECT_EQ(messages.global_overview.header.frame_id, "map");
  EXPECT_EQ(messages.global_overview.info.width, 300U);
  EXPECT_EQ(messages.global_overview.info.height, 300U);
  EXPECT_GT(node->known_global_count(), 0U);
  EXPECT_EQ(messages.local_grid_map.header.frame_id, "odom");
  EXPECT_EQ(messages.local_grid_map.layers,
            (std::vector<std::string>{"occupancy", "semantic_id", "elevation",
                                      "roughness"}));
  ASSERT_EQ(messages.local_grid_map.data.size(), 4U);
  EXPECT_EQ(messages.local_grid_map.data.front().data.size(), 320U * 320U);
  EXPECT_EQ(messages.odometry.header.frame_id, "odom");
  EXPECT_EQ(messages.odometry.child_frame_id, "base_link");
  EXPECT_DOUBLE_EQ(messages.odometry.twist.twist.linear.y, 0.0);
  ASSERT_EQ(messages.transforms.transforms.size(), 2U);
  EXPECT_EQ(messages.transforms.transforms[0].header.frame_id, "map");
  EXPECT_EQ(messages.transforms.transforms[0].child_frame_id, "odom");
  EXPECT_DOUBLE_EQ(messages.transforms.transforms[0].transform.rotation.w, 1.0);
  EXPECT_EQ(messages.transforms.transforms[1].header.frame_id, "odom");
  EXPECT_EQ(messages.transforms.transforms[1].child_frame_id, "base_link");
  EXPECT_FALSE(messages.sensor_fov.points.empty());
  EXPECT_GE(messages.vehicle_markers.markers.size(), 5U);
  ASSERT_EQ(messages.local_map_markers.markers.size(), 2U);
  EXPECT_EQ(messages.local_map_markers.markers[0].ns, "local_map_window");
  EXPECT_EQ(messages.local_map_markers.markers[0].type,
            visualization_msgs::msg::Marker::LINE_STRIP);
  EXPECT_EQ(messages.local_map_markers.markers[0].points.size(), 5U);
  EXPECT_EQ(messages.local_map_markers.markers[1].ns, "local_elevation");
  EXPECT_EQ(messages.local_map_markers.markers[1].type,
            visualization_msgs::msg::Marker::POINTS);
  EXPECT_FALSE(messages.local_map_markers.markers[1].points.empty());
  EXPECT_EQ(messages.actual_path.poses.size(), 1U);
  EXPECT_DOUBLE_EQ(messages.sim_elapsed.data, 0.0);
}

TEST_F(RosContextTest, ForwardCommandAdvancesCompleteSharedStateTick) {
  const auto node = MakeNode();
  const std::size_t initial_known_count = node->known_global_count();
  geometry_msgs::msg::Twist command;
  command.linear.x = 0.5;
  node->SetCommand(command);

  node->Tick(0.05);

  const auto& messages = node->latest_messages();
  EXPECT_GE(node->known_global_count(), initial_known_count);
  EXPECT_GT(node->known_global_count(), 0U);
  EXPECT_EQ(messages.global_overview.data.size(), 300U * 300U);
  EXPECT_NE(std::ranges::find_if(messages.global_overview.data,
                                 [](std::int8_t value) { return value >= 0; }),
            messages.global_overview.data.end());
  ASSERT_EQ(messages.local_grid_map.data.size(), 4U);
  for (const auto& layer : messages.local_grid_map.data) {
    EXPECT_EQ(layer.data.size(), 320U * 320U);
  }
  EXPECT_NEAR(messages.odometry.pose.pose.position.x, 0.5, 1.0e-12);
  EXPECT_NEAR(messages.odometry.pose.pose.position.y, 0.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(messages.odometry.twist.twist.linear.x, 0.5);
  EXPECT_DOUBLE_EQ(messages.odometry.twist.twist.linear.y, 0.0);
  EXPECT_DOUBLE_EQ(messages.sim_elapsed.data, 1.0);
  ASSERT_EQ(messages.transforms.transforms.size(), 2U);
  EXPECT_NEAR(messages.transforms.transforms[1].transform.translation.x, 0.5,
              1.0e-12);
  EXPECT_EQ(messages.actual_path.poses.size(), 2U);
  EXPECT_FALSE(messages.sensor_fov.points.empty());
  EXPECT_GE(messages.vehicle_markers.markers.size(), 5U);
  EXPECT_GT(node->last_tick_duration_s(), 0.0);
}

TEST_F(RosContextTest, NonFiniteCommandStopsAndWallStepIsCapped) {
  const auto node = MakeNode();
  geometry_msgs::msg::Twist command;
  command.linear.x = std::numeric_limits<double>::quiet_NaN();
  command.angular.z = 0.4;
  node->SetCommand(command);

  node->Tick(0.5);

  const auto& messages = node->latest_messages();
  EXPECT_DOUBLE_EQ(messages.odometry.pose.pose.position.x, 0.0);
  EXPECT_DOUBLE_EQ(messages.odometry.pose.pose.position.y, 0.0);
  EXPECT_DOUBLE_EQ(messages.odometry.pose.pose.orientation.z, 0.0);
  EXPECT_DOUBLE_EQ(messages.odometry.twist.twist.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(messages.odometry.twist.twist.angular.z, 0.0);
  EXPECT_DOUBLE_EQ(messages.sim_elapsed.data, 2.0);
}

TEST_F(RosContextTest, TickDiagnosticsExposeDeadlineBoundary) {
  const auto node = MakeNode();
  const auto before = node->tick_count();

  node->Tick(0.0);

  EXPECT_EQ(node->tick_count(), before + 1U);
  EXPECT_GE(node->max_tick_duration_s(), node->last_tick_duration_s());
  EXPECT_DOUBLE_EQ(node->deadline_period_s(), 0.05);
}

TEST_F(RosContextTest, RepeatedCompleteTicksRecordComputationCost) {
  const auto node = MakeNode();
  constexpr std::size_t kRepetitions = 20U;
  const auto start = std::chrono::steady_clock::now();

  for (std::size_t index = 0U; index < kRepetitions; ++index) {
    node->Tick(0.05);
  }

  const double measured_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  EXPECT_EQ(node->tick_count(), kRepetitions);
  EXPECT_EQ(node->latest_messages().actual_path.poses.size(),
            kRepetitions + 1U);
  EXPECT_DOUBLE_EQ(node->latest_messages().sim_elapsed.data, 20.0);
  EXPECT_GT(node->max_tick_duration_s(), 0.0);
  std::cout << "complete_tick_average_ms="
            << measured_s * 1000.0 / static_cast<double>(kRepetitions)
            << " complete_tick_max_ms=" << node->max_tick_duration_s() * 1000.0
            << " deadline_misses=" << node->deadline_miss_count() << '\n';
}

}  // namespace
}  // namespace lunar::pure_exploration_sim
