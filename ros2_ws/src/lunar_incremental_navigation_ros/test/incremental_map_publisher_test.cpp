#include <chrono>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"
#include "lunar_incremental_navigation_core/global_guidance_builder.hpp"
#include "lunar_incremental_navigation_ros/incremental_map_publisher.hpp"

namespace lunar::incremental_navigation_ros {
namespace {

namespace core = lunar::incremental_navigation;
using namespace std::chrono_literals;

[[nodiscard]] core::WheeledCapability Capability() {
  return core::WheeledCapability{
      .footprint_xy_m = {{-0.05, -0.05}, {0.05, -0.05},
                         {0.05, 0.05}, {-0.05, 0.05}},
      .body_extent_m = {.x = 0.1, .y = 0.1, .z = 0.1},
      .wheel_diameter_m = 0.2,
      .wheel_width_m = 0.05,
      .wheelbase_m = 0.1,
      .track_width_m = 0.1,
      .minimum_underbody_clearance_m = 0.1,
      .maximum_local_obstacle_relief_m = 0.2,
      .maximum_forward_speed_mps = 1.0,
      .maximum_reverse_speed_mps = 1.0,
      .maximum_spin_rate_radps = 1.0,
      .maximum_acceleration_mps2 = 1.0,
      .maximum_braking_deceleration_mps2 = 1.0,
      .maximum_yaw_acceleration_radps2 = 1.0,
      .maximum_lateral_acceleration_mps2 = 1.0,
      .maximum_curvature_per_m = 1.0,
      .maximum_slope_rad = std::numbers::pi / 4.0,
  };
}

[[nodiscard]] core::TraversabilityProfile Profile() {
  return core::TraversabilityProfile{
      .maximum_slope_rad = std::numbers::pi / 4.0,
      .planar_envelope_xy_m = {{-0.05, -0.05}, {0.05, -0.05},
                               {0.05, 0.05}, {-0.05, 0.05}},
      .slope_weight = 1.0,
      .relief_weight = 1.0,
  };
}

[[nodiscard]] core::SnapshotBundle Bundle() {
  core::PersistentElevationMap map;
  const core::GridGeometry geometry{.frame_id = "map",
                                    .width = 4U,
                                    .height = 4U,
                                    .resolution_m = 0.5,
                                    .origin_m = {.x = -1.0, .y = 2.0}};
  const std::vector<float> values(geometry.CellCount(), 0.0F);
  const auto update = map.Apply(core::ElevationEvidence{
      .geometry = geometry,
      .elevation_m = values,
      .map_from_source = {.parent_frame = "map",
                          .child_frame = "map",
                          .rotation = {.w = 1.0}},
  });
  if (update.status != core::ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("publisher elevation fixture rejected");
  }
  auto fine = core::FineTraversabilityBuilder{}.Derive(
      map.Snapshot(), Capability(), Profile());
  auto guidance = core::GlobalGuidanceBuilder{1.0}.Derive(
      fine, std::nullopt);
  return {.fine = std::move(fine), .guidance = std::move(guidance)};
}

[[nodiscard]] core::SnapshotBundle MismatchedBundle(
    const core::SnapshotBundle& consistent) {
  auto newer_fine = std::make_shared<const core::FineTraversabilitySnapshot>(
      consistent.fine->geometry(), consistent.fine->raw_elevation_revision(),
      consistent.fine->fine_traversability_revision() + 1U,
      consistent.fine->platform_profile_hash(),
      consistent.fine->hard_inflation_radius_m(),
      consistent.fine->preferred_clearance_m(),
      consistent.fine->cost_weights(), consistent.fine->elevation(),
      consistent.fine->tile_directory(), std::vector<core::TileIndex>{},
      std::vector<core::TileIndex>{}, consistent.fine->metrics());
  return {.fine = std::move(newer_fine),
          .guidance = consistent.guidance};
}

class IncrementalMapPublisherTest : public ::testing::Test {
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
};

TEST_F(IncrementalMapPublisherTest,
       RejectsInconsistentFineAndGuidanceSourceRevisions) {
  auto node = std::make_shared<rclcpp::Node>("projection_revision_test");
  IncrementalMapPublisher publisher(
      *node, "/test/incremental_map/revision");
  const auto consistent = Bundle();

  EXPECT_FALSE(publisher.Publish(MismatchedBundle(consistent)));
  EXPECT_TRUE(publisher.Publish(consistent));
}

TEST_F(IncrementalMapPublisherTest,
       PublishesCompleteLatchedMapWithRequiredQos) {
  const std::string topic = "/test/incremental_map/latched";
  auto producer = std::make_shared<rclcpp::Node>("projection_producer");
  IncrementalMapPublisher publisher(*producer, topic);
  const auto bundle = Bundle();
  ASSERT_TRUE(publisher.Publish(bundle));

  auto consumer = std::make_shared<rclcpp::Node>("projection_consumer");
  std::optional<nav_msgs::msg::OccupancyGrid> received;
  auto subscription = consumer->create_subscription<nav_msgs::msg::OccupancyGrid>(
      topic, ExplorationMapQos(),
      [&received](const nav_msgs::msg::OccupancyGrid& message) {
        received = message;
      });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(producer);
  executor.add_node(consumer);
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (!received && std::chrono::steady_clock::now() < deadline) {
    executor.spin_some();
  }

  ASSERT_TRUE(received);
  EXPECT_EQ(received->header.frame_id, "map");
  EXPECT_EQ(received->info.width, 2U);
  EXPECT_EQ(received->info.height, 2U);
  EXPECT_FLOAT_EQ(received->info.resolution, 1.0F);
  EXPECT_DOUBLE_EQ(received->info.origin.position.x, -1.0);
  EXPECT_DOUBLE_EQ(received->info.origin.position.y, 2.0);
  EXPECT_DOUBLE_EQ(received->info.origin.orientation.w, 1.0);
  EXPECT_EQ(received->data.size(), 4U);

  const auto qos = ExplorationMapQos().get_rmw_qos_profile();
  EXPECT_EQ(qos.depth, 1U);
  EXPECT_EQ(qos.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  EXPECT_EQ(qos.durability, RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  executor.remove_node(consumer);
  executor.remove_node(producer);
  static_cast<void>(subscription);
}

}  // namespace
}  // namespace lunar::incremental_navigation_ros
