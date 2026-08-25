#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stop_token>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "grid_v1/grid_v1_planner.hpp"
#include "lunar_pure_planner_core/traversability_map.hpp"
#include "lunar_pure_planner_core/types/planning_request.hpp"

namespace lunar::pure_planning::grid_v1 {
namespace {

constexpr double kEpsilon = 1.0e-9;

[[nodiscard]] Quaternion YawQuaternion(const double yaw_rad) {
  return Quaternion{.w = std::cos(yaw_rad / 2.0),
                    .z = std::sin(yaw_rad / 2.0)};
}

[[nodiscard]] GridMap MakeLocalMap(const std::size_t width,
                                   const std::size_t height,
                                   const std::vector<float>& occupancy) {
  const std::size_t count = width * height;
  EXPECT_EQ(occupancy.size(), count);
  return GridMap{
      .frame_id = "odom",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .width = width,
      .height = height,
      .resolution_m = 1.0,
      .origin_m = Vec3{},
      .layers = {
          {"occupancy", GridLayer{.values = occupancy}},
          {"elevation", GridLayer{.values = std::vector<float>(count, 0.0F)}},
      },
  };
}

[[nodiscard]] RigidTransform MapFromOdom(const Vec3 translation_m = {},
                                          const double yaw_rad = 0.0) {
  return RigidTransform{
      .parent_frame = "map",
      .child_frame = "odom",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .translation_m = translation_m,
      .rotation = YawQuaternion(yaw_rad),
  };
}

[[nodiscard]] std::shared_ptr<const TraversabilitySnapshot> MakeSnapshot(
    const std::size_t width, const std::size_t height,
    const std::vector<float>& occupancy,
    const RigidTransform& map_from_odom = MapFromOdom()) {
  PersistentTraversabilityMap map(TraversabilityProfile{
      .local_occupancy_threshold = 0.5,
      .maximum_slope_rad = std::numbers::pi / 2.0,
      .inflation_radius_m = 0.0,
  });
  const auto update = map.UpdateLocal(
      MakeLocalMap(width, height, occupancy),
      map_from_odom,
      1U);
  EXPECT_TRUE(update.accepted) << update.reason_code;
  auto snapshot = map.Capture();
  EXPECT_NE(snapshot, nullptr);
  return snapshot;
}

[[nodiscard]] PlanningRequest MakeRequest(
    std::shared_ptr<const TraversabilitySnapshot> snapshot, const Vec3 start,
    const Vec3 goal, const double start_yaw = 0.0) {
  return PlanningRequest{
      .request_id = "grid-v1-test",
      .environment_mode = EnvironmentMode::kLunarSurface,
      .current_state = WheeledState{
          .pose = Pose3{.position_m = start,
                        .orientation = YawQuaternion(start_yaw)},
      },
      .goal_map = GoalRegion{
          .goal_id = "goal",
          .target = PointGoal{.position_m = goal, .tolerance_m = 0.0},
      },
      .world = MinimalWorldSnapshot{
          .map_from_odom = RigidTransform{
              .parent_frame = "map", .child_frame = "odom",
              .stamp = TimePoint{.nanoseconds_since_epoch = 1},
              .translation_m = Vec3{}, .rotation = Quaternion{}},
          .traversability_snapshot = std::move(snapshot),
      },
      .capability = WheeledCapability{
          .maximum_forward_speed_mps = 1.0,
          .maximum_reverse_speed_mps = 0.5,
          .maximum_spin_rate_radps = 1.0,
          .maximum_acceleration_mps2 = 1.0,
          .maximum_braking_deceleration_mps2 = 1.0,
          .maximum_yaw_acceleration_radps2 = 1.0,
      },
      .config = AnytimePlannerConfig{.grid_v1_local_horizon_m = 8.0},
  };
}

[[nodiscard]] const TrajectoryReference& Trajectory(
    const PlanningResult& result) {
  if (result.status != PlanningStatus::kSuccess || !result.reference.has_value()) {
    ADD_FAILURE() << "expected successful wheeled trajectory: "
                  << result.reason_code;
    static const TrajectoryReference empty;
    return empty;
  }
  EXPECT_EQ(result.reference->platform_type, PlatformType::kWheeled);
  const auto* trajectory = std::get_if<TrajectoryReference>(&result.reference->data);
  if (trajectory == nullptr) {
    ADD_FAILURE() << "missing trajectory data";
    static const TrajectoryReference empty;
    return empty;
  }
  return *trajectory;
}

[[nodiscard]] std::vector<Pose3> Poses(const TrajectoryReference& trajectory) {
  std::vector<Pose3> poses;
  poses.reserve(trajectory.points.size());
  for (const auto& point : trajectory.points) {
    poses.push_back(point.pose);
  }
  return poses;
}

[[nodiscard]] double LocalPreviewLength(const MotionReference& reference) {
  double total{};
  for (std::size_t index = 1U; index < reference.preview.poses_map.size(); ++index) {
    const Vec3& previous = reference.preview.poses_map[index - 1U].position_m;
    const Vec3& current = reference.preview.poses_map[index].position_m;
    total += std::hypot(current.x - previous.x, current.y - previous.y);
  }
  return total;
}

[[nodiscard]] double GlobalPreviewLength(const GlobalRoutePreview& preview) {
  double total{};
  for (std::size_t index = 1U; index < preview.poses_map.size(); ++index) {
    const Vec3& previous = preview.poses_map[index - 1U].position_m;
    const Vec3& current = preview.poses_map[index].position_m;
    total += std::hypot(current.x - previous.x, current.y - previous.y);
  }
  return total;
}

void ExpectTimedBoundedTrajectory(const TrajectoryReference& trajectory,
                                  const double maximum_speed_mps) {
  ASSERT_GE(trajectory.points.size(), 2U);
  EXPECT_EQ(trajectory.points.front().time_from_start,
            std::chrono::nanoseconds::zero());
  EXPECT_NEAR(trajectory.points.front().velocity.linear_mps.x, 0.0, kEpsilon);
  EXPECT_NEAR(trajectory.points.front().velocity.linear_mps.y, 0.0, kEpsilon);
  EXPECT_NEAR(trajectory.points.back().velocity.linear_mps.x, 0.0, kEpsilon);
  EXPECT_NEAR(trajectory.points.back().velocity.linear_mps.y, 0.0, kEpsilon);
  for (std::size_t index = 1U; index < trajectory.points.size(); ++index) {
    const auto& previous = trajectory.points[index - 1U];
    const auto& current = trajectory.points[index];
    EXPECT_GT(current.time_from_start, previous.time_from_start);
    EXPECT_TRUE(std::isfinite(current.velocity.linear_mps.x));
    EXPECT_TRUE(std::isfinite(current.velocity.linear_mps.y));
    EXPECT_LE(std::hypot(current.velocity.linear_mps.x,
                         current.velocity.linear_mps.y),
              maximum_speed_mps + kEpsilon);
    EXPECT_LE(std::hypot(current.pose.position_m.x - previous.pose.position_m.x,
                         current.pose.position_m.y - previous.pose.position_m.y),
              1.0 + kEpsilon);
  }
}

TEST(GridV1Planner, ConnectedFreeGridReturnsTimedSafeTrajectory) {
  const auto snapshot = MakeSnapshot(12U, 3U, std::vector<float>(36U, 0.0F));
  PlanningRequest request = MakeRequest(
      snapshot, Vec3{.x = 0.5, .y = 1.5}, Vec3{.x = 6.5, .y = 1.5});
  request.world.local_map.stamp = TimePoint{.nanoseconds_since_epoch = 42};
  const PlanningResult result = Plan(request);

  const TrajectoryReference& trajectory = Trajectory(result);
  EXPECT_EQ(result.timing.global_call_count, 1U);
  EXPECT_EQ(result.timing.local_call_count, 1U);
  EXPECT_EQ(result.grid_v1.postprocess_mode, "SHORTCUT");
  std::size_t checked_cells{};
  EXPECT_TRUE(PathIsFree(*snapshot, Poses(trajectory), &checked_cells));
  EXPECT_GT(checked_cells, 0U);
  ASSERT_TRUE(result.reference.has_value());
  EXPECT_EQ(result.reference->input_time, request.world.local_map.stamp);
  ASSERT_EQ(result.reference->preview.poses_map.size(), trajectory.points.size());
  for (std::size_t index = 0U; index < trajectory.points.size(); ++index) {
    EXPECT_EQ(result.reference->preview.poses_map[index], trajectory.points[index].pose);
  }
  ExpectTimedBoundedTrajectory(trajectory, 1.0);
}

TEST(GridV1Planner, UnknownAndBlockedGoalsHaveStageSpecificReasons) {
  std::vector<float> unknown(25U, 0.0F);
  for (std::size_t y = 1U; y <= 3U; ++y) {
    for (std::size_t x = 1U; x <= 3U; ++x) {
      unknown[y * 5U + x] = std::numeric_limits<float>::quiet_NaN();
    }
  }
  const PlanningResult unknown_result = Plan(MakeRequest(
      MakeSnapshot(5U, 5U, unknown), Vec3{.x = 0.5, .y = 0.5},
      Vec3{.x = 2.5, .y = 2.5}));
  EXPECT_EQ(unknown_result.status, PlanningStatus::kNoPath);
  EXPECT_EQ(unknown_result.reason_code, "GOAL_NOT_FREE");

  std::vector<float> blocked(9U, 0.0F);
  blocked[4U] = 1.0F;
  const PlanningResult blocked_result = Plan(MakeRequest(
      MakeSnapshot(3U, 3U, blocked), Vec3{.x = 0.5, .y = 0.5},
      Vec3{.x = 1.5, .y = 1.5}));
  EXPECT_EQ(blocked_result.status, PlanningStatus::kNoPath);
  EXPECT_EQ(blocked_result.reason_code, "GOAL_NOT_FREE");
}

TEST(GridV1Planner, DoesNotCutDiagonalBetweenBlockedCorners) {
  std::vector<float> occupancy(9U, 0.0F);
  occupancy[1U] = 1.0F;
  occupancy[3U] = 1.0F;
  const PlanningResult result = Plan(MakeRequest(
      MakeSnapshot(3U, 3U, occupancy), Vec3{.x = 0.5, .y = 0.5},
      Vec3{.x = 1.5, .y = 1.5}));

  EXPECT_EQ(result.status, PlanningStatus::kNoPath);
  EXPECT_EQ(result.reason_code, "GLOBAL_NO_PATH");
  EXPECT_FALSE(result.reference.has_value());
}

TEST(GridV1Planner, LimitsLocalCandidateToEightMeters) {
  const auto snapshot = MakeSnapshot(16U, 3U, std::vector<float>(48U, 0.0F));
  const PlanningResult result = Plan(MakeRequest(
      snapshot, Vec3{.x = 0.5, .y = 1.5}, Vec3{.x = 12.5, .y = 1.5}));

  ASSERT_EQ(result.global_route_preview.poses_map.size(), 13U);
  EXPECT_DOUBLE_EQ(result.global_route_preview.poses_map.front().position_m.x,
                   0.5);
  EXPECT_DOUBLE_EQ(result.global_route_preview.poses_map.back().position_m.x,
                   12.5);

  const TrajectoryReference& trajectory = Trajectory(result);
  const Vec3 end = trajectory.points.back().pose.position_m;
  EXPECT_LE(std::hypot(end.x - 0.5, end.y - 1.5), 8.0 + kEpsilon);
  EXPECT_NEAR(std::hypot(end.x - 0.5, end.y - 1.5), 8.0, kEpsilon);
}

TEST(GridV1Planner, ReportsCompleteGlobalRouteCostBeyondLocalHorizon) {
  const auto snapshot = MakeSnapshot(28U, 3U, std::vector<float>(84U, 0.0F));
  const PlanningResult result = Plan(MakeRequest(
      snapshot, Vec3{.x = 0.5, .y = 1.5}, Vec3{.x = 24.5, .y = 1.5}));

  ASSERT_EQ(result.status, PlanningStatus::kSuccess);
  ASSERT_TRUE(result.best_cost.has_value());
  EXPECT_GT(*result.best_cost, 20.0);
  ASSERT_TRUE(result.reference.has_value());
  EXPECT_LT(LocalPreviewLength(*result.reference), 9.0);
  EXPECT_GT(GlobalPreviewLength(result.global_route_preview), 20.0);
  EXPECT_NEAR(*result.best_cost,
              GlobalPreviewLength(result.global_route_preview), 1.0e-9);
}

TEST(GridV1Planner, ObstacleDetourStaysFreeAfterPostprocessOrRawFallback) {
  std::vector<float> occupancy(10U * 7U, 0.0F);
  for (std::size_t y = 0U; y < 6U; ++y) {
    occupancy[y * 10U + 4U] = 1.0F;
  }
  const auto snapshot = MakeSnapshot(10U, 7U, occupancy);
  const PlanningResult result = Plan(MakeRequest(
      snapshot, Vec3{.x = 0.5, .y = 0.5}, Vec3{.x = 8.5, .y = 0.5}));

  const TrajectoryReference& trajectory = Trajectory(result);
  EXPECT_TRUE(result.grid_v1.postprocess_mode == "SHORTCUT" ||
              result.grid_v1.postprocess_mode == "RAW_GRID_FALLBACK");
  std::size_t checked_cells{};
  EXPECT_TRUE(PathIsFree(*snapshot, Poses(trajectory), &checked_cells));
  EXPECT_GT(checked_cells, 0U);
  bool has_signed_spin{};
  for (const TrajectoryPoint& point : trajectory.points) {
    EXPECT_TRUE(std::isfinite(point.velocity.angular_radps.z));
    EXPECT_LE(std::abs(point.velocity.angular_radps.z), 1.0 + kEpsilon);
    has_signed_spin = has_signed_spin ||
                      std::abs(point.velocity.angular_radps.z) > kEpsilon;
  }
  EXPECT_TRUE(has_signed_spin);
}

TEST(GridV1Planner, RepeatsDeterministically) {
  const auto snapshot = MakeSnapshot(10U, 4U, std::vector<float>(40U, 0.0F));
  const PlanningResult first = Plan(MakeRequest(
      snapshot, Vec3{.x = 0.5, .y = 1.5}, Vec3{.x = 7.5, .y = 2.5}));
  const PlanningResult second = Plan(MakeRequest(
      snapshot, Vec3{.x = 0.5, .y = 1.5}, Vec3{.x = 7.5, .y = 2.5}));

  const TrajectoryReference& first_trajectory = Trajectory(first);
  const TrajectoryReference& second_trajectory = Trajectory(second);
  ASSERT_EQ(first_trajectory.points.size(), second_trajectory.points.size());
  EXPECT_EQ(first.grid_v1.direction, second.grid_v1.direction);
  for (std::size_t index = 0U; index < first_trajectory.points.size(); ++index) {
    EXPECT_EQ(first_trajectory.points[index].pose,
              second_trajectory.points[index].pose);
    EXPECT_EQ(first_trajectory.points[index].time_from_start,
              second_trajectory.points[index].time_from_start);
    EXPECT_EQ(first_trajectory.points[index].velocity,
              second_trajectory.points[index].velocity);
  }
}

TEST(GridV1Planner, SelectsForwardOrReverseForWholeTrajectory) {
  const auto snapshot = MakeSnapshot(8U, 3U, std::vector<float>(24U, 0.0F));
  const PlanningResult forward = Plan(MakeRequest(
      snapshot, Vec3{.x = 1.5, .y = 1.5}, Vec3{.x = 5.5, .y = 1.5}));
  PlanningRequest reverse_request = MakeRequest(
      snapshot, Vec3{.x = 5.5, .y = 1.5}, Vec3{.x = 1.5, .y = 1.5});
  std::get<WheeledCapability>(reverse_request.capability).maximum_reverse_speed_mps =
      1.0;
  const PlanningResult reverse = Plan(reverse_request);

  const TrajectoryReference& forward_trajectory = Trajectory(forward);
  const TrajectoryReference& reverse_trajectory = Trajectory(reverse);
  EXPECT_EQ(forward.grid_v1.direction, "FORWARD");
  EXPECT_EQ(reverse.grid_v1.direction, "REVERSE");
  ASSERT_GT(forward_trajectory.points.size(), 2U);
  ASSERT_GT(reverse_trajectory.points.size(), 2U);
  EXPECT_GT(forward_trajectory.points[1U].velocity.linear_mps.x, 0.0);
  EXPECT_LT(reverse_trajectory.points[1U].velocity.linear_mps.x, 0.0);
  ExpectTimedBoundedTrajectory(forward_trajectory, 1.0);
  ExpectTimedBoundedTrajectory(reverse_trajectory, 1.0);
}

TEST(GridV1Planner, TransformsOdomStateIntoMapBeforeSearching) {
  const RigidTransform map_from_odom =
      MapFromOdom(Vec3{.x = 10.0, .y = -5.0}, std::numbers::pi / 2.0);
  const auto snapshot = MakeSnapshot(12U, 3U, std::vector<float>(36U, 0.0F),
                                     map_from_odom);
  PlanningRequest request = MakeRequest(
      snapshot, Vec3{.x = 0.5, .y = 1.5}, Vec3{.x = 8.5, .y = -4.5}, 0.0);
  request.world.map_from_odom = map_from_odom;
  std::get<WheeledState>(request.current_state).velocity.linear_mps =
      Vec3{.x = 1.0};

  const PlanningResult result = Plan(request);
  const TrajectoryReference& trajectory = Trajectory(result);
  EXPECT_NEAR(trajectory.points.front().pose.position_m.x, 8.5, kEpsilon);
  EXPECT_NEAR(trajectory.points.front().pose.position_m.y, -4.5, kEpsilon);
  EXPECT_NEAR(trajectory.points.back().pose.position_m.x, 8.5, kEpsilon);
  EXPECT_NEAR(trajectory.points.back().pose.position_m.y, -4.5, kEpsilon);
}

TEST(GridV1Planner, UsesGlobalSearchBeforeLocalCandidatesAndFailsClosed) {
  std::vector<float> disconnected(15U * 3U, 0.0F);
  for (std::size_t y = 0U; y < 3U; ++y) {
    disconnected[y * 15U + 7U] = 1.0F;
  }
  const PlanningResult no_global_path = Plan(MakeRequest(
      MakeSnapshot(15U, 3U, disconnected), Vec3{.x = 0.5, .y = 1.5},
      Vec3{.x = 12.5, .y = 1.5}));
  EXPECT_EQ(no_global_path.status, PlanningStatus::kNoPath);
  EXPECT_EQ(no_global_path.reason_code, "GLOBAL_NO_PATH");
  EXPECT_GT(no_global_path.grid_v1.global_expanded_states, 0U);

  std::vector<float> blocked_goal(15U * 3U, 0.0F);
  blocked_goal[1U * 15U + 12U] = 1.0F;
  const PlanningResult nonfree_goal = Plan(MakeRequest(
      MakeSnapshot(15U, 3U, blocked_goal), Vec3{.x = 0.5, .y = 1.5},
      Vec3{.x = 12.5, .y = 1.5}));
  EXPECT_EQ(nonfree_goal.status, PlanningStatus::kNoPath);
  EXPECT_EQ(nonfree_goal.reason_code, "GOAL_NOT_FREE");
}

TEST(GridV1Planner, HonorsImmediateCancellationAndDeadline) {
  const auto snapshot = MakeSnapshot(12U, 3U, std::vector<float>(36U, 0.0F));
  PlanningRequest canceled = MakeRequest(
      snapshot, Vec3{.x = 0.5, .y = 1.5}, Vec3{.x = 6.5, .y = 1.5});
  std::stop_source stop_source;
  stop_source.request_stop();
  canceled.control.stop_token = stop_source.get_token();
  const PlanningResult canceled_result = Plan(canceled);
  EXPECT_EQ(canceled_result.status, PlanningStatus::kCanceled);
  EXPECT_EQ(canceled_result.reason_code, "REQUEST_CANCELED");
  EXPECT_FALSE(canceled_result.reference.has_value());

  PlanningRequest timed_out = MakeRequest(
      snapshot, Vec3{.x = 0.5, .y = 1.5}, Vec3{.x = 6.5, .y = 1.5});
  timed_out.control.deadline = SteadyClock::time_point::min();
  const PlanningResult timeout_result = Plan(timed_out);
  EXPECT_EQ(timeout_result.status, PlanningStatus::kTimedOut);
  EXPECT_EQ(timeout_result.reason_code, "TIMEOUT");
  EXPECT_FALSE(timeout_result.reference.has_value());
}

TEST(GridV1Planner, RejectsInvalidQuaternionAndGoalYaw) {
  const auto snapshot = MakeSnapshot(8U, 3U, std::vector<float>(24U, 0.0F));
  PlanningRequest invalid_orientation = MakeRequest(
      snapshot, Vec3{.x = 0.5, .y = 1.5}, Vec3{.x = 5.5, .y = 1.5});
  std::get<WheeledState>(invalid_orientation.current_state).pose.orientation =
      Quaternion{.w = 0.0};
  EXPECT_EQ(Plan(invalid_orientation).status, PlanningStatus::kInvalidInput);

  PlanningRequest invalid_yaw = MakeRequest(
      snapshot, Vec3{.x = 0.5, .y = 1.5}, Vec3{.x = 5.5, .y = 1.5});
  invalid_yaw.goal_map.yaw_rad = std::numeric_limits<double>::quiet_NaN();
  invalid_yaw.goal_map.yaw_tolerance_rad = -0.1;
  EXPECT_EQ(Plan(invalid_yaw).status, PlanningStatus::kInvalidInput);
}

}  // namespace
}  // namespace lunar::pure_planning::grid_v1
