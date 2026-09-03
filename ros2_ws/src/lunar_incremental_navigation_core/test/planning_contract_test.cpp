#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_core/types/path_reference.hpp"
#include "lunar_incremental_navigation_core/types/planning_cycle.hpp"

namespace lunar::incremental_navigation {
namespace {

static_assert(std::is_same_v<decltype(NavigateToPoseGoal::target_x_m), double>);
static_assert(std::is_same_v<decltype(NavigateToPoseGoal::target_y_m), double>);
static_assert(std::is_same_v<decltype(NavigateToPoseGoal::has_target_yaw), bool>);
static_assert(std::is_same_v<decltype(NavigateToPoseGoal::target_yaw_rad), double>);
static_assert(std::is_same_v<FinalGoal, NavigateToPoseGoal>);

static_assert(
    std::is_same_v<decltype(NavigateToPoseFeedback::session_state), SessionState>);
static_assert(
    std::is_same_v<decltype(NavigateToPoseFeedback::planning_cycle), std::uint64_t>);
static_assert(std::is_same_v<
              decltype(NavigateToPoseFeedback::active_segment_revision),
              std::uint64_t>);
static_assert(
    std::is_same_v<decltype(NavigateToPoseFeedback::reason_code), std::string>);

static_assert(
    std::is_same_v<decltype(NavigateToPoseResult::outcome), SessionOutcome>);
static_assert(
    std::is_same_v<decltype(NavigateToPoseResult::reason_code), std::string>);
static_assert(std::is_same_v<
              decltype(NavigateToPoseResult::last_segment_revision),
              std::uint64_t>);

SessionId MakeSessionId(const std::uint8_t seed) {
  SessionId id;
  for (std::size_t index = 0; index < id.bytes.size(); ++index) {
    id.bytes[index] = static_cast<std::uint8_t>(seed + index);
  }
  return id;
}

Pose3 MakePose(const double x_m) {
  Pose3 pose;
  pose.position_m.x = x_m;
  return pose;
}

TEST(PlanningContractTest, FreezesWireEnumValues) {
  EXPECT_EQ(static_cast<std::uint8_t>(PathState::kActive), 0U);
  EXPECT_EQ(static_cast<std::uint8_t>(PathState::kInvalidated), 1U);
  EXPECT_EQ(static_cast<std::uint8_t>(SessionState::kPlanning), 0U);
  EXPECT_EQ(static_cast<std::uint8_t>(SessionState::kExecuting), 1U);
  EXPECT_EQ(static_cast<std::uint8_t>(SessionState::kReplanning), 2U);
  EXPECT_EQ(static_cast<std::uint8_t>(SessionOutcome::kGoalReached), 0U);
  EXPECT_EQ(static_cast<std::uint8_t>(SessionOutcome::kNoPath), 1U);
  EXPECT_EQ(static_cast<std::uint8_t>(SessionOutcome::kInvalidGoal), 2U);
  EXPECT_EQ(static_cast<std::uint8_t>(SessionOutcome::kMapUnavailable), 3U);
  EXPECT_EQ(static_cast<std::uint8_t>(SessionOutcome::kTimeout), 4U);
  EXPECT_EQ(static_cast<std::uint8_t>(SessionOutcome::kCanceled), 5U);
  EXPECT_EQ(static_cast<std::uint8_t>(SessionOutcome::kInternalError), 6U);
}

TEST(PlanningContractTest, ActiveReferenceRequiresANonEmptyGeometricPath) {
  PathReference reference;
  reference.session_id = MakeSessionId(1U);
  reference.segment_revision = 1U;
  reference.traversability_revision = 7U;
  reference.state = PathState::kActive;

  EXPECT_FALSE(IsValidPathReference(reference));
  reference.path.poses.push_back(MakePose(1.25));
  EXPECT_TRUE(IsValidPathReference(reference));
}

TEST(PlanningContractTest, InvalidatedReferenceRequiresAnEmptyPath) {
  PathReference reference;
  reference.session_id = MakeSessionId(2U);
  reference.segment_revision = 3U;
  reference.traversability_revision = 9U;
  reference.state = PathState::kInvalidated;

  EXPECT_TRUE(IsValidPathReference(reference));
  reference.path.poses.push_back(MakePose(2.0));
  EXPECT_FALSE(IsValidPathReference(reference));
}

TEST(PlanningContractTest, SegmentRevisionMustIncreaseWithinTheSameSession) {
  PathReference current;
  current.session_id = MakeSessionId(3U);
  current.segment_revision = 8U;

  PathReference candidate = current;
  candidate.segment_revision = 9U;
  EXPECT_TRUE(IsStrictlyNewerSegment(candidate, current));

  candidate.segment_revision = 8U;
  EXPECT_FALSE(IsStrictlyNewerSegment(candidate, current));
  candidate.segment_revision = 7U;
  EXPECT_FALSE(IsStrictlyNewerSegment(candidate, current));

  candidate.segment_revision = 9U;
  candidate.session_id = MakeSessionId(4U);
  EXPECT_FALSE(IsStrictlyNewerSegment(candidate, current));
}

TEST(PlanningContractTest, GoalRequiresFinitePositionAndSelectedYaw) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  EXPECT_TRUE(IsValidFinalGoal(
      FinalGoal{.target_x_m = 1.0,
                .target_y_m = -2.0,
                .has_target_yaw = false,
                .target_yaw_rad = nan}));
  EXPECT_TRUE(IsValidFinalGoal(
      FinalGoal{.target_x_m = 1.0,
                .target_y_m = -2.0,
                .has_target_yaw = true,
                .target_yaw_rad = 0.5}));
  EXPECT_FALSE(IsValidFinalGoal(
      FinalGoal{.target_x_m = nan, .target_y_m = -2.0}));
  EXPECT_FALSE(IsValidFinalGoal(
      FinalGoal{.target_x_m = 1.0, .target_y_m = infinity}));
  EXPECT_FALSE(IsValidFinalGoal(
      FinalGoal{.target_x_m = 1.0,
                .target_y_m = -2.0,
                .has_target_yaw = true,
                .target_yaw_rad = infinity}));
}

}  // namespace
}  // namespace lunar::incremental_navigation
