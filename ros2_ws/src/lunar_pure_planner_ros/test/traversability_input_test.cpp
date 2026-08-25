#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_planner_ros/traversability_input.hpp"

namespace lunar::pure_planner_ros {
namespace {

std_msgs::msg::Float32MultiArray Layer(std::vector<float> values) {
  std_msgs::msg::Float32MultiArray layer;
  std_msgs::msg::MultiArrayDimension columns;
  columns.label = "column_index";
  columns.size = 2U;
  columns.stride = 4U;
  std_msgs::msg::MultiArrayDimension rows;
  rows.label = "row_index";
  rows.size = 2U;
  rows.stride = 2U;
  layer.layout.dim = {columns, rows};
  layer.data = std::move(values);
  return layer;
}

nav_msgs::msg::OccupancyGrid::ConstSharedPtr GlobalMap() {
  auto map = std::make_shared<nav_msgs::msg::OccupancyGrid>();
  map->header.frame_id = "map";
  map->info.width = 4U;
  map->info.height = 4U;
  map->info.resolution = 0.5F;
  map->info.origin.orientation.w = 1.0;
  map->data.assign(16U, 0);
  return map;
}

grid_map_msgs::msg::GridMap::ConstSharedPtr LocalMap() {
  auto map = std::make_shared<grid_map_msgs::msg::GridMap>();
  map->header.frame_id = "odom";
  map->info.resolution = 0.5;
  map->info.length_x = 1.0;
  map->info.length_y = 1.0;
  map->info.pose.orientation.w = 1.0;
  map->layers = {"occupancy", "elevation"};
  map->data = {Layer({0.F, 0.F, 0.F, 0.F}), Layer({0.F, 0.F, 0.F, 0.F})};
  return map;
}

tf2_msgs::msg::TFMessage MapFromOdom() {
  tf2_msgs::msg::TFMessage message;
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "map";
  transform.child_frame_id = "odom";
  transform.transform.rotation.w = 1.0;
  message.transforms.push_back(std::move(transform));
  return message;
}

tf2_msgs::msg::TFMessage InvalidMapFromOdom() {
  auto message = MapFromOdom();
  message.transforms.front().transform.rotation.w = 0.0;
  return message;
}

lunar::pure_planning::TraversabilityProfile Profile() {
  return {.global_occupancy_threshold = 50,
          .local_occupancy_threshold = 0.5,
          .maximum_slope_rad = 0.5,
          .inflation_radius_m = 0.0};
}

TEST(TraversabilityInput, RetriesPendingLocalAfterOutOfOrderTfOnlyOnce) {
  TraversabilityInput input{Profile()};

  input.UpdateGlobal(GlobalMap());
  input.UpdateLocal(LocalMap());
  const auto pending = input.Capture();
  EXPECT_FALSE(pending.snapshot);
  EXPECT_EQ(pending.reason_code, "TF_UNAVAILABLE");

  input.UpdateTf(MapFromOdom());
  const auto ready = input.Capture();
  ASSERT_TRUE(ready.snapshot);
  ASSERT_TRUE(ready.snapshot->valid());
  EXPECT_EQ(ready.snapshot->resolution_m(), 0.5);
  EXPECT_TRUE(ready.reason_code.empty());
  const std::uint64_t revision_after_retry = ready.snapshot->revision();

  input.UpdateTf(MapFromOdom());
  const auto repeated_tf = input.Capture();
  ASSERT_TRUE(repeated_tf.snapshot);
  EXPECT_EQ(repeated_tf.snapshot->revision(), revision_after_retry);
}

TEST(TraversabilityInput,
     LocalThenTfBuildsSnapshotBeforeOptionalGlobalAndReanchorsLater) {
  TraversabilityInput input{Profile()};
  auto global = std::make_shared<nav_msgs::msg::OccupancyGrid>(*GlobalMap());
  global->info.origin.position.x = 7.0;
  global->info.origin.position.y = -3.0;

  input.UpdateLocal(LocalMap());
  const auto before_tf = input.Capture();
  EXPECT_FALSE(before_tf.snapshot);
  EXPECT_EQ(before_tf.reason_code, "TF_UNAVAILABLE");

  input.UpdateTf(MapFromOdom());
  const auto before_global = input.Capture();
  ASSERT_TRUE(before_global.snapshot);
  ASSERT_TRUE(before_global.snapshot->valid());
  EXPECT_EQ(before_global.snapshot->origin_m(),
            (lunar::pure_planning::Vec3{.x = -0.5, .y = -0.5, .z = 0.0}));
  EXPECT_EQ(before_global.snapshot->StateAtWorld(0.25, 0.25),
            lunar::pure_planning::TraversabilityState::kFree);
  EXPECT_TRUE(before_global.reason_code.empty());
  const std::uint64_t local_only_revision = before_global.snapshot->revision();

  input.UpdateGlobal(std::move(global));
  const auto ready = input.Capture();
  ASSERT_TRUE(ready.snapshot);
  ASSERT_TRUE(ready.snapshot->valid());
  EXPECT_GT(ready.snapshot->revision(), local_only_revision);
  EXPECT_EQ(ready.snapshot->origin_m(),
            (lunar::pure_planning::Vec3{.x = 7.0, .y = -3.0, .z = 0.0}));
  EXPECT_EQ(ready.snapshot->StateAtWorld(0.25, 0.25),
            lunar::pure_planning::TraversabilityState::kFree);
  EXPECT_TRUE(ready.reason_code.empty());

  const std::uint64_t revision_after_global = ready.snapshot->revision();
  input.UpdateTf(MapFromOdom());
  const auto repeated_tf = input.Capture();
  ASSERT_TRUE(repeated_tf.snapshot);
  EXPECT_EQ(repeated_tf.snapshot->revision(), revision_after_global);
}

TEST(TraversabilityInput, RejectsInvalidInputWithoutReplacingLastValidSnapshot) {
  TraversabilityInput input{Profile()};
  input.UpdateGlobal(GlobalMap());
  input.UpdateTf(MapFromOdom());
  input.UpdateLocal(LocalMap());
  const auto valid = input.Capture();
  ASSERT_TRUE(valid.snapshot);
  const std::uint64_t revision = valid.snapshot->revision();

  auto malformed = std::make_shared<grid_map_msgs::msg::GridMap>(*LocalMap());
  malformed->header.frame_id = "map";
  input.UpdateLocal(std::move(malformed));
  const auto rejected = input.Capture();
  ASSERT_TRUE(rejected.snapshot);
  EXPECT_EQ(rejected.snapshot->revision(), revision);
  EXPECT_EQ(rejected.reason_code, "INVALID_INPUT");
}

TEST(TraversabilityInput,
     InvalidTfThenValidTfClearsReasonWithoutRevisionChange) {
  TraversabilityInput input{Profile()};
  input.UpdateTf(MapFromOdom());
  input.UpdateLocal(LocalMap());
  const auto valid = input.Capture();
  ASSERT_TRUE(valid.snapshot);
  const std::uint64_t revision = valid.snapshot->revision();

  input.UpdateTf(InvalidMapFromOdom());
  const auto rejected = input.Capture();
  ASSERT_TRUE(rejected.snapshot);
  EXPECT_EQ(rejected.snapshot->revision(), revision);
  EXPECT_EQ(rejected.reason_code, "INVALID_INPUT");

  input.UpdateTf(MapFromOdom());
  const auto recovered = input.Capture();
  ASSERT_TRUE(recovered.snapshot);
  EXPECT_EQ(recovered.snapshot->revision(), revision);
  EXPECT_TRUE(recovered.reason_code.empty());
}

}  // namespace
}  // namespace lunar::pure_planner_ros
