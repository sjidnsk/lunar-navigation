#include "lunar_pure_exploration_sim/visibility.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>

#include <gtest/gtest.h>

#include "lunar_pure_exploration_sim/lunar_scene.hpp"

namespace lunar::pure_exploration_sim {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct OcclusionFixture {
  Pose2 pose;
  double blocker_x_m;
  double blocker_y_m;
  double behind_x_m;
  double behind_y_m;
};

std::optional<OcclusionFixture> FindOcclusionFixture(const LunarScene& scene) {
  for (double pose_y_m = -140.5; pose_y_m < 140.0; pose_y_m += 2.0) {
    for (double pose_x_m = -140.5; pose_x_m < 140.0; pose_x_m += 2.0) {
      if (scene.Sample(pose_x_m, pose_y_m).occupied) {
        continue;
      }
      for (double distance_m = 0.1; distance_m <= 8.0; distance_m += 0.1) {
        const double sample_x_m = pose_x_m + distance_m;
        if (!scene.Sample(sample_x_m, pose_y_m).occupied) {
          continue;
        }
        const double behind_x_m = sample_x_m + 1.0;
        if (behind_x_m < scene.max_x_m()) {
          return OcclusionFixture{
              .pose = {.x_m = pose_x_m, .y_m = pose_y_m, .yaw_rad = 0.0},
              .blocker_x_m = sample_x_m,
              .blocker_y_m = pose_y_m,
              .behind_x_m = behind_x_m,
              .behind_y_m = pose_y_m,
          };
        }
        break;
      }
    }
  }
  return std::nullopt;
}

TEST(VisibilityTest, EnforcesNinetyDegreeFovAndTenMeterRange) {
  const auto scene = BuildLunarScene(20260824U);
  ObservationState observations;
  observations.Observe(scene, Pose2{}, SensorModel{});

  const auto point = [](double range_m, double degrees) {
    const double angle_rad = degrees * kPi / 180.0;
    return std::pair{range_m * std::cos(angle_rad),
                     range_m * std::sin(angle_rad)};
  };
  const auto inside_angle = point(9.9, 44.9);
  const auto exact_angle = point(2.0, 45.0);
  const auto exact_range = point(10.0, 0.0);
  const auto outside_angle = point(9.9, 45.1);
  const auto outside_range = point(10.1, 0.0);

  EXPECT_TRUE(observations.IsCurrentlyVisible(
      scene, inside_angle.first, inside_angle.second));
  EXPECT_TRUE(observations.IsCurrentlyVisible(
      scene, exact_angle.first, exact_angle.second));
  EXPECT_TRUE(observations.IsCurrentlyVisible(
      scene, exact_range.first, exact_range.second));
  EXPECT_FALSE(observations.IsCurrentlyVisible(
      scene, outside_angle.first, outside_angle.second));
  EXPECT_FALSE(observations.IsCurrentlyVisible(
      scene, outside_range.first, outside_range.second));
}

TEST(VisibilityTest, KeepsFirstOccupiedSampleVisibleAndOccludesBehindIt) {
  const auto scene = BuildLunarScene(20260824U);
  const auto fixture = FindOcclusionFixture(scene);
  ASSERT_TRUE(fixture.has_value());
  ObservationState observations;
  observations.Observe(scene, fixture->pose, SensorModel{});

  EXPECT_TRUE(observations.IsCurrentlyVisible(
      scene, fixture->blocker_x_m, fixture->blocker_y_m));
  EXPECT_FALSE(observations.IsCurrentlyVisible(
      scene, fixture->behind_x_m, fixture->behind_y_m));
}

TEST(VisibilityTest, PersistsGlobalKnowledgeAcrossObservations) {
  const auto scene = BuildLunarScene(20260824U);
  ObservationState observations;
  observations.Observe(scene, Pose2{}, SensorModel{});
  const auto known_after_first = observations.KnownGlobalMask();
  ASSERT_EQ(known_after_first.size(), 90'000U);
  ASSERT_TRUE(observations.IsKnownGlobalCell(155U, 150U));

  observations.Observe(
      scene, Pose2{.x_m = 0.0, .y_m = 0.0, .yaw_rad = kPi},
      SensorModel{});

  EXPECT_TRUE(observations.IsKnownGlobalCell(155U, 150U));
  EXPECT_GE(observations.KnownGlobalCount(),
            static_cast<std::size_t>(std::count(known_after_first.begin(),
                                                known_after_first.end(), true)));
  EXPECT_FALSE(observations.IsCurrentlyVisible(scene, 5.5, 0.5));
}

}  // namespace
}  // namespace lunar::pure_exploration_sim
