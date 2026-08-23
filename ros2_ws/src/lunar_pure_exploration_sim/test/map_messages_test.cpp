#include "lunar_pure_exploration_sim/map_messages.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_pure_exploration_sim/lunar_scene.hpp"
#include "lunar_pure_exploration_sim/visibility.hpp"

namespace lunar::pure_exploration_sim {
namespace {

constexpr std::size_t kLocalWidth = 320U;
constexpr std::size_t kLocalHeight = 320U;

std::size_t PhysicalIndex(std::size_t logical_x, std::size_t logical_y) {
  const std::size_t physical_row = kLocalWidth - 1U - logical_x;
  const std::size_t physical_column = kLocalHeight - 1U - logical_y;
  return physical_column * kLocalWidth + physical_row;
}

std::size_t LogicalCell(double world_coordinate_m, double center_m) {
  return static_cast<std::size_t>(
      std::floor((world_coordinate_m - (center_m - 32.0)) / 0.2));
}

struct VisibleObstacle {
  Pose2 pose;
  std::size_t logical_x;
  std::size_t logical_y;
  std::size_t global_index;
};

VisibleObstacle FindVisibleObstacle(const LunarScene& scene) {
  for (double pose_y_m = -140.5; pose_y_m < 140.0; pose_y_m += 2.0) {
    for (double pose_x_m = -140.5; pose_x_m < 140.0; pose_x_m += 2.0) {
      const Pose2 pose{.x_m = pose_x_m, .y_m = pose_y_m, .yaw_rad = 0.0};
      if (scene.Sample(pose.x_m, pose.y_m).occupied) {
        continue;
      }
      for (double delta_x_m = 0.1; delta_x_m < 10.0; delta_x_m += 0.2) {
        const double cell_x_m = pose.x_m + delta_x_m;
        const double cell_y_m = pose.y_m + 0.1;
        if (!scene.Sample(cell_x_m, cell_y_m).occupied) {
          continue;
        }
        bool blocked_before_target = false;
        const double target_distance_m = std::hypot(delta_x_m, 0.1);
        for (double distance_m = 0.1;
             distance_m < target_distance_m - 1.0e-9; distance_m += 0.1) {
          const double ratio = distance_m / target_distance_m;
          if (scene.Sample(pose.x_m + ratio * delta_x_m,
                           pose.y_m + ratio * 0.1).occupied) {
            blocked_before_target = true;
            break;
          }
        }
        const auto global_x = static_cast<std::size_t>(
            std::floor(cell_x_m - scene.min_x_m()));
        const auto global_y = static_cast<std::size_t>(
            std::floor(cell_y_m - scene.min_x_m()));
        const std::size_t global_index =
            global_y * scene.global_width() + global_x;
        if (!blocked_before_target &&
            scene.GlobalOccupancy()[global_index] == 100) {
          return {.pose = pose,
                  .logical_x = LogicalCell(cell_x_m, pose.x_m),
                  .logical_y = LogicalCell(cell_y_m, pose.y_m),
                  .global_index = global_index};
        }
        break;
      }
    }
  }
  return {};
}

struct ScoredPose {
  Pose2 pose;
  std::size_t occupied_centers;
};

std::vector<Pose2> ObstacleRichPoses(const LunarScene& scene) {
  std::vector<ScoredPose> candidates;
  for (std::size_t y = 10U; y + 10U < scene.global_height(); y += 5U) {
    for (std::size_t x = 10U; x + 10U < scene.global_width(); x += 5U) {
      if (scene.GlobalOccupancy()[y * scene.global_width() + x] != 0) {
        continue;
      }
      const double pose_x_m = scene.min_x_m() + static_cast<double>(x) + 0.5;
      const double pose_y_m = scene.min_x_m() + static_cast<double>(y) + 0.5;
      for (const double yaw_rad : {0.0, std::numbers::pi / 2.0,
                                   std::numbers::pi, -std::numbers::pi / 2.0}) {
        std::size_t occupied_centers = 0U;
        for (long offset_y = -10L; offset_y <= 10L; ++offset_y) {
          for (long offset_x = -10L; offset_x <= 10L; ++offset_x) {
            const double distance_m = std::hypot(
                static_cast<double>(offset_x), static_cast<double>(offset_y));
            if (distance_m > 10.0 || distance_m < 1.0e-9) {
              continue;
            }
            const double angle = std::remainder(
                std::atan2(static_cast<double>(offset_y),
                           static_cast<double>(offset_x)) - yaw_rad,
                2.0 * std::numbers::pi);
            if (std::abs(angle) > std::numbers::pi / 4.0) {
              continue;
            }
            const auto cell_x = static_cast<std::size_t>(
                static_cast<long>(x) + offset_x);
            const auto cell_y = static_cast<std::size_t>(
                static_cast<long>(y) + offset_y);
            if (scene.GlobalOccupancy()[cell_y * scene.global_width() +
                                        cell_x] == 100) {
              ++occupied_centers;
            }
          }
        }
        candidates.push_back({.pose = {.x_m = pose_x_m,
                                       .y_m = pose_y_m,
                                       .yaw_rad = yaw_rad},
                              .occupied_centers = occupied_centers});
      }
    }
  }
  std::ranges::sort(candidates, [](const auto& left, const auto& right) {
    return left.occupied_centers > right.occupied_centers;
  });
  std::vector<Pose2> selected;
  for (const auto& candidate : candidates) {
    if (candidate.occupied_centers == 0U) {
      break;
    }
    const bool separated = std::ranges::all_of(selected, [&](const Pose2& pose) {
      return std::hypot(candidate.pose.x_m - pose.x_m,
                        candidate.pose.y_m - pose.y_m) > 20.0;
    });
    if (separated) {
      selected.push_back(candidate.pose);
      if (selected.size() == 3U) {
        break;
      }
    }
  }
  return selected;
}

enum class ReferenceVisibility { kOutsideSector, kOccluded, kVisible };

ReferenceVisibility ReferenceCellCenterVisibility(
    const LunarScene& scene, const Pose2& pose, double world_x_m,
    double world_y_m) {
  if (world_x_m < scene.min_x_m() || world_x_m >= scene.max_x_m() ||
      world_y_m < scene.min_x_m() || world_y_m >= scene.max_x_m()) {
    return ReferenceVisibility::kOutsideSector;
  }
  const double delta_x_m = world_x_m - pose.x_m;
  const double delta_y_m = world_y_m - pose.y_m;
  const double distance_m = std::hypot(delta_x_m, delta_y_m);
  if (distance_m > 10.0 + 1.0e-9) {
    return ReferenceVisibility::kOutsideSector;
  }
  if (distance_m > 1.0e-9) {
    const double yaw_offset = std::remainder(
        std::atan2(delta_y_m, delta_x_m) - pose.yaw_rad,
        2.0 * std::numbers::pi);
    if (std::abs(yaw_offset) > std::numbers::pi / 4.0 + 1.0e-9) {
      return ReferenceVisibility::kOutsideSector;
    }
    for (double distance = 0.1; distance < distance_m - 1.0e-9;
         distance += 0.1) {
      const double ratio = distance / distance_m;
      if (scene.Sample(pose.x_m + ratio * delta_x_m,
                       pose.y_m + ratio * delta_y_m).occupied) {
        return ReferenceVisibility::kOccluded;
      }
    }
  }
  return ReferenceVisibility::kVisible;
}

TEST(MapMessagesTest, PublishesPersistentNativeGlobalOccupancy) {
  const auto scene = BuildLunarScene(20260824U);
  ObservationState observations;
  observations.Observe(scene, Pose2{}, SensorModel{});
  const rclcpp::Time stamp{12, 345, RCL_ROS_TIME};

  const auto message = MakeGlobalOverview(scene, observations, stamp);

  EXPECT_EQ(message.header.frame_id, "map");
  EXPECT_EQ(message.header.stamp.sec, 12);
  EXPECT_EQ(message.header.stamp.nanosec, 345U);
  EXPECT_EQ(message.info.width, 300U);
  EXPECT_EQ(message.info.height, 300U);
  EXPECT_FLOAT_EQ(message.info.resolution, 1.0F);
  EXPECT_DOUBLE_EQ(message.info.origin.position.x, -150.0);
  EXPECT_DOUBLE_EQ(message.info.origin.position.y, -150.0);
  EXPECT_DOUBLE_EQ(message.info.origin.orientation.w, 1.0);
  ASSERT_EQ(message.data.size(), 90'000U);
  EXPECT_EQ(message.data[150U * 300U + 155U], 0);
  EXPECT_EQ(message.data[150U * 300U + 130U], -1);
  for (std::size_t index = 0U; index < message.data.size(); ++index) {
    if (!observations.KnownGlobalMask()[index]) {
      EXPECT_EQ(message.data[index], -1);
    } else {
      EXPECT_TRUE(message.data[index] == 0 || message.data[index] == 100);
    }
  }
}

TEST(MapMessagesTest, BuildsReversedFourLayerCurrentVisibilityWindow) {
  const auto scene = BuildLunarScene(20260824U);
  ObservationState observations;
  observations.Observe(scene, Pose2{}, SensorModel{});

  const auto message = MakeLocalGridMap(
      scene, observations, Pose2{}, rclcpp::Time{7, 8, RCL_ROS_TIME});

  EXPECT_EQ(message.header.frame_id, "odom");
  EXPECT_EQ(message.layers,
            (std::vector<std::string>{"occupancy", "semantic_id",
                                      "elevation", "roughness"}));
  EXPECT_DOUBLE_EQ(message.info.resolution, 0.2);
  EXPECT_DOUBLE_EQ(message.info.length_x, 64.0);
  EXPECT_DOUBLE_EQ(message.info.length_y, 64.0);
  EXPECT_DOUBLE_EQ(message.info.pose.position.x, 0.0);
  EXPECT_DOUBLE_EQ(message.info.pose.position.y, 0.0);
  EXPECT_DOUBLE_EQ(message.info.pose.orientation.w, 1.0);
  EXPECT_EQ(message.outer_start_index, 0U);
  EXPECT_EQ(message.inner_start_index, 0U);
  ASSERT_EQ(message.data.size(), 4U);
  for (const auto& layer : message.data) {
    ASSERT_EQ(layer.layout.dim.size(), 2U);
    EXPECT_EQ(layer.layout.dim[0].label, "column_index");
    EXPECT_EQ(layer.layout.dim[0].size, kLocalHeight);
    EXPECT_EQ(layer.layout.dim[0].stride, kLocalWidth * kLocalHeight);
    EXPECT_EQ(layer.layout.dim[1].label, "row_index");
    EXPECT_EQ(layer.layout.dim[1].size, kLocalWidth);
    EXPECT_EQ(layer.layout.dim[1].stride, kLocalWidth);
    EXPECT_EQ(layer.data.size(), kLocalWidth * kLocalHeight);
  }

  const std::size_t visible_x = LogicalCell(5.1, 0.0);
  const std::size_t visible_y = LogicalCell(0.1, 0.0);
  const std::size_t visible = PhysicalIndex(visible_x, visible_y);
  EXPECT_FLOAT_EQ(message.data[0].data[visible], 0.0F);
  EXPECT_TRUE(std::isfinite(message.data[1].data[visible]));
  EXPECT_TRUE(std::isfinite(message.data[2].data[visible]));
  EXPECT_TRUE(std::isfinite(message.data[3].data[visible]));

  const std::size_t behind_x = LogicalCell(-5.1, 0.0);
  const std::size_t behind_y = LogicalCell(0.1, 0.0);
  const std::size_t behind = PhysicalIndex(behind_x, behind_y);
  for (const auto& layer : message.data) {
    EXPECT_TRUE(std::isnan(layer.data[behind]));
  }
}

TEST(MapMessagesTest, EncodesVisibleObstacleAsExactOneAndOneHundred) {
  const auto scene = BuildLunarScene(20260824U);
  const auto fixture = FindVisibleObstacle(scene);
  ASSERT_NE(fixture.global_index, 0U);
  ObservationState observations;
  observations.Observe(scene, fixture.pose, SensorModel{});

  const auto local = MakeLocalGridMap(
      scene, observations, fixture.pose, rclcpp::Time{3, 0, RCL_ROS_TIME});
  const auto physical = PhysicalIndex(fixture.logical_x, fixture.logical_y);
  EXPECT_FLOAT_EQ(local.data[0].data[physical], 1.0F);
  EXPECT_TRUE(std::isfinite(local.data[1].data[physical]));

  const auto global = MakeGlobalOverview(
      scene, observations, rclcpp::Time{3, 0, RCL_ROS_TIME});
  ASSERT_TRUE(observations.KnownGlobalMask()[fixture.global_index]);
  EXPECT_EQ(global.data[fixture.global_index], 100);
}

TEST(MapMessagesTest, LocalMapDoesNotReusePersistentGlobalKnowledge) {
  const auto scene = BuildLunarScene(20260824U);
  ObservationState observations;
  observations.Observe(scene, Pose2{}, SensorModel{});
  observations.Observe(scene,
                       Pose2{.x_m = 0.0, .y_m = 0.0,
                             .yaw_rad = 3.14159265358979323846},
                       SensorModel{});

  const auto message = MakeLocalGridMap(
      scene, observations,
      Pose2{.x_m = 0.0, .y_m = 0.0,
            .yaw_rad = 3.14159265358979323846},
      rclcpp::Time{1, 0, RCL_ROS_TIME});
  const auto stale_index = PhysicalIndex(LogicalCell(5.1, 0.0),
                                         LogicalCell(0.1, 0.0));
  ASSERT_TRUE(observations.IsKnownGlobalCell(155U, 150U));
  for (const auto& layer : message.data) {
    EXPECT_TRUE(std::isnan(layer.data[stale_index]));
  }
}

TEST(MapMessagesTest, KeepsEveryLayerUnknownOutsideFiniteTruthDomain) {
  const auto scene = BuildLunarScene(20260824U);
  const Pose2 pose{.x_m = 149.0, .y_m = 149.0,
                   .yaw_rad = 0.7853981633974483};
  ObservationState observations;
  observations.Observe(scene, pose, SensorModel{});
  const auto message = MakeLocalGridMap(
      scene, observations, pose, rclcpp::Time{2, 0, RCL_ROS_TIME});
  const auto outside = PhysicalIndex(LogicalCell(150.1, pose.x_m),
                                     LogicalCell(150.1, pose.y_m));

  for (const auto& layer : message.data) {
    EXPECT_TRUE(std::isnan(layer.data[outside]));
  }
}

TEST(MapMessagesTest, CachedLocalMapConversionFitsTwentyHertzPeriod) {
  const auto scene = BuildLunarScene(20260824U);
  ObservationState observations;
  observations.Observe(scene, Pose2{}, SensorModel{});
  static_cast<void>(MakeLocalGridMap(
      scene, observations, Pose2{}, rclcpp::Time{4, 0, RCL_ROS_TIME}));

  std::vector<double> elapsed_ms;
  elapsed_ms.reserve(7U);
  std::size_t consumed_values = 0U;
  for (int iteration = 0; iteration < 7; ++iteration) {
    const auto begin = std::chrono::steady_clock::now();
    const auto message = MakeLocalGridMap(
        scene, observations, Pose2{},
        rclcpp::Time{4, static_cast<std::uint32_t>(iteration), RCL_ROS_TIME});
    const auto end = std::chrono::steady_clock::now();
    elapsed_ms.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
    consumed_values += message.data.front().data.size();
  }
  ASSERT_EQ(consumed_values, 7U * kLocalWidth * kLocalHeight);
  for (std::size_t index = 0U; index < elapsed_ms.size(); ++index) {
    std::cout << "local_map_conversion_ms[" << index
              << "]=" << elapsed_ms[index] << '\n';
  }
  std::ranges::sort(elapsed_ms);
  const double median_ms = elapsed_ms[elapsed_ms.size() / 2U];
  const double maximum_ms = elapsed_ms.back();
  RecordProperty("median_ms", median_ms);
  RecordProperty("maximum_ms", maximum_ms);

  EXPECT_LT(median_ms, 50.0);
  EXPECT_LT(maximum_ms, 200.0);
}

TEST(MapMessagesTest, CachedCellsMatchReferenceLosAcrossObstacleRichPoses) {
  const auto scene = BuildLunarScene(20260824U);
  const auto poses = ObstacleRichPoses(scene);
  ASSERT_EQ(poses.size(), 3U);
  std::size_t total_leaks = 0U;
  std::size_t total_holes = 0U;
  std::size_t total_occluded = 0U;

  for (std::size_t pose_index = 0U; pose_index < poses.size(); ++pose_index) {
    const Pose2 pose = poses[pose_index];
    ObservationState observations;
    observations.Observe(scene, pose, SensorModel{});
    const auto map = MakeLocalGridMap(
        scene, observations, pose,
        rclcpp::Time{5, static_cast<std::uint32_t>(pose_index), RCL_ROS_TIME});
    std::size_t pose_leaks = 0U;
    std::size_t pose_holes = 0U;
    std::optional<std::pair<std::size_t, std::size_t>> occluded_cell;
    const double origin_x_m = pose.x_m - 32.0;
    const double origin_y_m = pose.y_m - 32.0;
    for (std::size_t logical_y = 0U; logical_y < kLocalHeight; ++logical_y) {
      const double world_y_m =
          origin_y_m + (static_cast<double>(logical_y) + 0.5) * 0.2;
      for (std::size_t logical_x = 0U; logical_x < kLocalWidth; ++logical_x) {
        const double world_x_m =
            origin_x_m + (static_cast<double>(logical_x) + 0.5) * 0.2;
        const auto reference = ReferenceCellCenterVisibility(
            scene, pose, world_x_m, world_y_m);
        const bool expected_visible = reference == ReferenceVisibility::kVisible;
        const bool cached_visible =
            observations.CurrentLocalSample(pose, logical_x, logical_y) != nullptr;
        pose_leaks += cached_visible && !expected_visible ? 1U : 0U;
        pose_holes += !cached_visible && expected_visible ? 1U : 0U;
        if (reference == ReferenceVisibility::kOccluded &&
            !occluded_cell.has_value()) {
          occluded_cell = std::pair{logical_x, logical_y};
        }
      }
    }
    ASSERT_TRUE(occluded_cell.has_value());
    const std::size_t occluded_physical =
        PhysicalIndex(occluded_cell->first, occluded_cell->second);
    for (const auto& layer : map.data) {
      EXPECT_TRUE(std::isnan(layer.data[occluded_physical]));
    }
    std::cout << "los_compare_pose[" << pose_index << "]=" << pose.x_m << ','
              << pose.y_m << ',' << pose.yaw_rad << " leaks=" << pose_leaks
              << " holes=" << pose_holes << '\n';
    total_leaks += pose_leaks;
    total_holes += pose_holes;
    ++total_occluded;
  }
  EXPECT_EQ(total_occluded, poses.size());
  EXPECT_EQ(total_leaks, 0U);
  EXPECT_EQ(total_holes, 0U);
}

TEST(MapMessagesTest, CombinedObservationAndConversionFitsTwentyHertzPeriod) {
  const auto scene = BuildLunarScene(20260824U);
  const auto poses = ObstacleRichPoses(scene);
  ASSERT_EQ(poses.size(), 3U);
  ObservationState observations;
  std::vector<double> elapsed_ms;
  elapsed_ms.reserve(7U);
  std::size_t consumed_values = 0U;
  for (std::size_t iteration = 0U; iteration < 7U; ++iteration) {
    const Pose2 pose = poses[iteration % poses.size()];
    const auto begin = std::chrono::steady_clock::now();
    observations.Observe(scene, pose, SensorModel{});
    const auto map = MakeLocalGridMap(
        scene, observations, pose,
        rclcpp::Time{6, static_cast<std::uint32_t>(iteration), RCL_ROS_TIME});
    const auto end = std::chrono::steady_clock::now();
    elapsed_ms.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
    consumed_values += map.data.front().data.size();
  }
  ASSERT_EQ(consumed_values, 7U * kLocalWidth * kLocalHeight);
  for (std::size_t index = 0U; index < elapsed_ms.size(); ++index) {
    std::cout << "observe_plus_local_map_ms[" << index
              << "]=" << elapsed_ms[index] << '\n';
  }
  std::ranges::sort(elapsed_ms);
  const double median_ms = elapsed_ms[elapsed_ms.size() / 2U];
  const double maximum_ms = elapsed_ms.back();
  RecordProperty("combined_median_ms", median_ms);
  RecordProperty("combined_maximum_ms", maximum_ms);
  EXPECT_LT(median_ms, 35.0);
  EXPECT_LT(maximum_ms, 100.0);
}

}  // namespace
}  // namespace lunar::pure_exploration_sim
