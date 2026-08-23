#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/frame_transform.hpp"
#include "hierarchical/local_planning_problem.hpp"
#include "hierarchical/map_level.hpp"
#include "lunar_pure_planner_core/types/motion_reference.hpp"
#include "lunar_pure_planner_core/types/planner_config.hpp"
#include "lunar_pure_planner_core/types/planner_io.hpp"

namespace lunar::pure_planning::hierarchical {
namespace {

template <typename T>
concept HasMaximumAttempts = requires(T value) { value.maximum_attempts; };

constexpr double kTolerance = 1.0e-9;

template <typename Config>
concept HasGlobalSearchResources = requires(Config config) {
  config.resources;
};

static_assert(!HasGlobalSearchResources<GlobalSearchConfig>);

[[nodiscard]] double Yaw(const Quaternion &quaternion) {
  return std::atan2(
      2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
      1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z));
}

[[nodiscard]] Quaternion YawQuaternion(const double yaw_rad) {
  return Quaternion{
      .w = std::cos(yaw_rad / 2.0),
      .z = std::sin(yaw_rad / 2.0),
  };
}

[[nodiscard]] GridMap Map(const char *frame, const std::size_t width,
                          const std::size_t height, const double resolution_m) {
  return GridMap{
      .frame_id = frame,
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .width = width,
      .height = height,
      .resolution_m = resolution_m,
  };
}

TEST(HierarchicalTypes, FreezesApprovedMapAndLocalRules) {
  const PlannerConfig config;

  EXPECT_DOUBLE_EQ(config.global_map.base_resolution_m, 0.2);
  EXPECT_EQ(config.global_map.maximum_level, 5U);
  EXPECT_EQ(config.global_map.maximum_cells, 1'048'576U);
  EXPECT_EQ(config.global_map.maximum_axis_cells, 4'096U);
  EXPECT_EQ(config.global_map.target_axis_cells, 256U);
  EXPECT_EQ(config.global_search.maximum_preview_points, 4'096U);
  EXPECT_DOUBLE_EQ(config.local_frontier.wheel_horizon_m, 4.0);
  EXPECT_DOUBLE_EQ(config.local_frontier.legged_horizon_m, 3.0);
  static_assert(!HasMaximumAttempts<LocalFrontierConfig>);
  EXPECT_DOUBLE_EQ(kFixedAdditionalCorridorMarginM, 2.0);
  EXPECT_DOUBLE_EQ(config.local_frontier.additional_corridor_margin_m,
                   kFixedAdditionalCorridorMarginM);
}

TEST(HierarchicalTypes, LocalSearchDomainCachesCanonicalShapeCountAndHash) {
  const LocalSearchDomain domain{2U, 2U, {1U, 0U, 1U, 1U}};

  EXPECT_EQ(domain.width(), 2U);
  EXPECT_EQ(domain.height(), 2U);
  ASSERT_EQ(domain.allowed().size(), 4U);
  EXPECT_EQ(domain.allowed()[0], 1U);
  EXPECT_EQ(domain.allowed()[1], 0U);
  EXPECT_EQ(domain.allowed()[2], 1U);
  EXPECT_EQ(domain.allowed()[3], 1U);
  EXPECT_EQ(domain.allowed_cell_count(), 3U);
  EXPECT_TRUE(domain.Contains(shared::GridCell{.x = 0, .y = 0}));
  EXPECT_FALSE(domain.Contains(shared::GridCell{.x = 1, .y = 0}));
  EXPECT_TRUE(domain.Contains(shared::GridCell{.x = 1, .y = 1}));
  EXPECT_FALSE(domain.Contains(shared::GridCell{.x = -1, .y = 0}));
  EXPECT_FALSE(domain.Contains(shared::GridCell{.x = 2, .y = 0}));
  EXPECT_EQ(domain.sha256(),
            "f248f5703378ff86250fb15dbd2540d4f54b5feba11b0c3dfe87c0951f099d10");
}

TEST(HierarchicalTypes, LocalSearchDomainRejectsWrongShapeAndNonBinaryBytes) {
  EXPECT_THROW((LocalSearchDomain{2U, 2U, {1U, 0U, 1U}}),
               std::invalid_argument);
  EXPECT_THROW((LocalSearchDomain{2U, 2U, {1U, 0U, 2U, 1U}}),
               std::invalid_argument);
}

TEST(HierarchicalTypes, SelectsSmallestAdmissibleConfiguredLevel) {
  const GlobalMapConfig config;

  const auto l0 = ExpectedGlobalMapLevel(51.2, 51.2, config);
  ASSERT_TRUE(l0.ok());
  EXPECT_EQ(l0.level, 0U);
  EXPECT_EQ(l0.width, 256U);
  EXPECT_EQ(l0.height, 256U);
  EXPECT_DOUBLE_EQ(l0.resolution_m, 0.2);

  const auto l1 = ExpectedGlobalMapLevel(51.3, 51.2, config);
  ASSERT_TRUE(l1.ok());
  EXPECT_EQ(l1.level, 1U);
  EXPECT_EQ(l1.width, 129U);
  EXPECT_EQ(l1.height, 128U);
  EXPECT_DOUBLE_EQ(l1.resolution_m, 0.4);

  const auto terminal = ExpectedGlobalMapLevel(1024.0, 1024.0, config);
  ASSERT_TRUE(terminal.ok());
  EXPECT_EQ(terminal.level, 5U);
  EXPECT_EQ(terminal.width, 256U);
  EXPECT_DOUBLE_EQ(terminal.resolution_m, 4.0);
}

TEST(HierarchicalTypes, RejectsPhysicalExtentThatExceedsTerminalLevel) {
  const auto result = ExpectedGlobalMapLevel(1'024.01, 0.2, GlobalMapConfig{});

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.reason_code, "GLOBAL_MAP_SCALE_UNSUPPORTED");
}

TEST(HierarchicalTypes, ValidatesGlobalAndLocalMapLevelsTogether) {
  const WorldSnapshot valid{
      .global_map = Map("map", 129U, 128U, 0.4),
      .local_map = Map("odom", 40U, 30U, 0.2),
  };
  const auto accepted = ValidateMapLevels(valid, GlobalMapConfig{});
  ASSERT_TRUE(accepted.ok());
  EXPECT_EQ(accepted.global_level, 1U);

  WorldSnapshot coarse = valid;
  coarse.global_map = Map("map", 65U, 64U, 0.8);
  const auto rejected_coarse = ValidateMapLevels(coarse, GlobalMapConfig{});
  EXPECT_FALSE(rejected_coarse.ok());
  EXPECT_EQ(rejected_coarse.reason_code, "GLOBAL_MAP_LEVEL_INVALID");

  WorldSnapshot non_l0_local = valid;
  non_l0_local.local_map.resolution_m = 0.4;
  const auto rejected_local =
      ValidateMapLevels(non_l0_local, GlobalMapConfig{});
  EXPECT_FALSE(rejected_local.ok());
  EXPECT_EQ(rejected_local.reason_code, "LOCAL_MAP_LEVEL_INVALID");
}

TEST(HierarchicalTypes, SelectsFourMetreTerminalLevelForConfiguredAxisBudget) {
  GlobalMapConfig config;

  const auto expected = ExpectedGlobalMapLevel(1024.0, 1024.0, config);
  ASSERT_TRUE(expected.ok());
  EXPECT_EQ(expected.level, 5U);
  EXPECT_EQ(expected.width, 256U);
  EXPECT_EQ(expected.height, 256U);
  EXPECT_DOUBLE_EQ(expected.resolution_m, 4.0);

  const WorldSnapshot world{
      .global_map = Map("map", 256U, 256U, 4.0),
      .local_map = Map("odom", 320U, 320U, 0.2),
  };
  const auto accepted = ValidateMapLevels(world, config);
  ASSERT_TRUE(accepted.ok());
  EXPECT_EQ(accepted.global_level, 5U);
}

TEST(HierarchicalTypes, AppliesAndInvertsNonUnitRigidTransform) {
  const RigidTransform map_from_odom{
      .parent_frame = "map",
      .child_frame = "odom",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .translation_m = {10.0, -2.0, 1.0},
      .rotation = YawQuaternion(std::numbers::pi / 2.0),
  };
  const Pose3 pose_odom{
      .position_m = {2.0, 1.0, 3.0},
      .orientation = YawQuaternion(std::numbers::pi / 4.0),
  };

  const auto pose_map = TransformPose(pose_odom, map_from_odom,
                                      TransformDirection::kChildToParent);
  ASSERT_TRUE(pose_map.has_value());
  EXPECT_NEAR(pose_map->position_m.x, 9.0, kTolerance);
  EXPECT_NEAR(pose_map->position_m.y, 0.0, kTolerance);
  EXPECT_NEAR(pose_map->position_m.z, 4.0, kTolerance);
  EXPECT_NEAR(Yaw(pose_map->orientation), 3.0 * std::numbers::pi / 4.0,
              kTolerance);

  const auto round_trip = TransformPose(*pose_map, map_from_odom,
                                        TransformDirection::kParentToChild);
  ASSERT_TRUE(round_trip.has_value());
  EXPECT_NEAR(round_trip->position_m.x, pose_odom.position_m.x, kTolerance);
  EXPECT_NEAR(round_trip->position_m.y, pose_odom.position_m.y, kTolerance);
  EXPECT_NEAR(round_trip->position_m.z, pose_odom.position_m.z, kTolerance);
  EXPECT_NEAR(Yaw(round_trip->orientation), std::numbers::pi / 4.0, kTolerance);
}

TEST(HierarchicalTypes, TransformsPointAndPlanarGoalsIncludingYaw) {
  const RigidTransform map_from_odom{
      .parent_frame = "map",
      .child_frame = "odom",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .translation_m = {5.0, 1.0, 0.0},
      .rotation = YawQuaternion(std::numbers::pi / 2.0),
  };
  const GoalRegion point{
      .goal_id = "point",
      .target = PointGoal{.position_m = {2.0, 0.0, 0.0}, .tolerance_m = 0.5},
      .yaw_rad = 0.25,
      .yaw_tolerance_rad = 0.1,
  };
  const auto point_map =
      TransformGoal(point, map_from_odom, TransformDirection::kChildToParent);
  ASSERT_TRUE(point_map.has_value());
  const auto *target = std::get_if<PointGoal>(&point_map->target);
  ASSERT_NE(target, nullptr);
  EXPECT_NEAR(target->position_m.x, 5.0, kTolerance);
  EXPECT_NEAR(target->position_m.y, 3.0, kTolerance);
  EXPECT_DOUBLE_EQ(target->tolerance_m, 0.5);
  ASSERT_TRUE(point_map->yaw_rad.has_value());
  EXPECT_NEAR(*point_map->yaw_rad, 0.25 + std::numbers::pi / 2.0, kTolerance);

  const GoalRegion polygon{
      .goal_id = "polygon",
      .target =
          PlanarRegionGoal{
              .boundary_m = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
              .normal_tolerance_m = 0.2,
          },
  };
  const auto polygon_map =
      TransformGoal(polygon, map_from_odom, TransformDirection::kChildToParent);
  ASSERT_TRUE(polygon_map.has_value());
  const auto *region = std::get_if<PlanarRegionGoal>(&polygon_map->target);
  ASSERT_NE(region, nullptr);
  ASSERT_EQ(region->boundary_m.size(), 3U);
  EXPECT_NEAR(region->boundary_m[1].x, 5.0, kTolerance);
  EXPECT_NEAR(region->boundary_m[1].y, 2.0, kTolerance);
  EXPECT_DOUBLE_EQ(region->normal_tolerance_m, 0.2);
}

TEST(HierarchicalTypes, RejectsInvalidTransformQuaternion) {
  const RigidTransform invalid{
      .parent_frame = "map",
      .child_frame = "odom",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .rotation = Quaternion{.w = 0.0, .x = 0.0, .y = 0.0, .z = 0.0},
  };

  EXPECT_FALSE(TransformPoint(Vec3{1.0, 2.0, 3.0}, invalid,
                              TransformDirection::kChildToParent)
                   .has_value());
}

TEST(HierarchicalTypes, StoresGlobalPreviewSeparatelyFromExecutionData) {
  MotionReference reference{
      .plan_id = "hierarchical/test",
      .platform_type = PlatformType::kWheeled,
      .input_time = TimePoint{.nanoseconds_since_epoch = 1},
      .preview =
          GlobalRoutePreview{
              .poses_map = {Pose3{.position_m = {1.0, 2.0, 0.0}}},
          },
      .data = TrajectoryReference{},
  };

  ASSERT_EQ(reference.preview.poses_map.size(), 1U);
  EXPECT_DOUBLE_EQ(reference.preview.poses_map.front().position_m.x, 1.0);
  EXPECT_TRUE(std::holds_alternative<TrajectoryReference>(reference.data));
}

} // namespace
} // namespace lunar::pure_planning::hierarchical
