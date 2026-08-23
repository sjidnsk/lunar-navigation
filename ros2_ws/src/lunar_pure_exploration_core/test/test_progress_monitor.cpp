#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

#include "lunar_pure_exploration_core/progress_monitor.hpp"

namespace lunar::pure_exploration {
namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

TimePoint AtSeconds(std::int64_t seconds) {
  return TimePoint{std::chrono::seconds{seconds}};
}

ProgressParameters Parameters(std::size_t maximum_points = 16U) {
  return ProgressParameters{maximum_points, std::chrono::seconds{30}, 0.2};
}

std::vector<Vec2> UnitLine() { return {{0.0, 0.0}, {1.0, 0.0}}; }

void ResetPath(ProgressMonitor& monitor, TimePoint now,
               std::initializer_list<Vec2> points, Vec2 position) {
  const std::vector<Vec2> path(points);
  monitor.Reset(now, path, position);
}

void ExpectTimeoutFrom(ProgressMonitor& monitor, std::int64_t start_seconds,
                       Vec2 position = {0.0, 0.0}) {
  EXPECT_FALSE(
      monitor.Update(AtSeconds(start_seconds + 29), position, false));
  EXPECT_TRUE(monitor.Update(AtSeconds(start_seconds + 30), position, false));
}

TEST(ProgressMonitor, ValidatesParametersAndRequiresSuccessfulReset) {
  EXPECT_THROW(ProgressMonitor(ProgressParameters{
                   0U, std::chrono::seconds{30}, 0.2}),
               std::invalid_argument);
  EXPECT_THROW(ProgressMonitor(ProgressParameters{
                   1U, Clock::duration::zero(), 0.2}),
               std::invalid_argument);
  EXPECT_THROW(ProgressMonitor(ProgressParameters{
                   1U, -std::chrono::seconds{1}, 0.2}),
               std::invalid_argument);
  for (const double invalid :
       {0.0, -0.2, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()}) {
    EXPECT_THROW(ProgressMonitor(ProgressParameters{
                     1U, std::chrono::seconds{30}, invalid}),
                 std::invalid_argument);
  }

  ProgressMonitor monitor(Parameters());
  EXPECT_THROW(monitor.Update(AtSeconds(0), {0.0, 0.0}, false),
               std::logic_error);
  EXPECT_THROW(monitor.historical_max_arc_length_m(), std::logic_error);
}

TEST(ProgressMonitor, EnforcesPathBoundAndValidatesFiniteInputs) {
  ProgressMonitor monitor(Parameters(2U));
  const std::vector<Vec2> over_limit{
      {std::numeric_limits<double>::quiet_NaN(), 0.0}, {1.0, 0.0},
      {2.0, 0.0}};
  EXPECT_THROW(monitor.Reset(AtSeconds(0), over_limit, {0.0, 0.0}),
               std::length_error);
  EXPECT_THROW(monitor.Reset(AtSeconds(0), {}, {0.0, 0.0}),
               std::invalid_argument);

  const auto exact_limit = UnitLine();
  EXPECT_NO_THROW(monitor.Reset(AtSeconds(0), exact_limit, {0.0, 0.0}));
  EXPECT_DOUBLE_EQ(monitor.historical_max_arc_length_m(), 0.0);

  for (const Vec2 invalid_point :
       {Vec2{std::numeric_limits<double>::quiet_NaN(), 0.0},
        Vec2{0.0, std::numeric_limits<double>::quiet_NaN()},
        Vec2{std::numeric_limits<double>::infinity(), 0.0},
        Vec2{0.0, -std::numeric_limits<double>::infinity()}}) {
    const std::vector<Vec2> invalid_path{{0.0, 0.0}, invalid_point};
    EXPECT_THROW(monitor.Reset(AtSeconds(0), invalid_path, {0.0, 0.0}),
                 std::invalid_argument);
    EXPECT_DOUBLE_EQ(monitor.historical_max_arc_length_m(), 0.0);
  }
  for (const Vec2 invalid_position :
       {Vec2{std::numeric_limits<double>::quiet_NaN(), 0.0},
        Vec2{0.0, std::numeric_limits<double>::infinity()}}) {
    EXPECT_THROW(monitor.Reset(AtSeconds(0), exact_limit, invalid_position),
                 std::invalid_argument);
    EXPECT_DOUBLE_EQ(monitor.historical_max_arc_length_m(), 0.0);
  }
}

TEST(ProgressMonitor, OwnsADeepCopyOfTheBoundedPath) {
  ProgressMonitor monitor(Parameters(2U));
  std::vector<Vec2> path{{0.0, 0.0}, {2.0, 0.0}};
  monitor.Reset(AtSeconds(0), path, {0.0, 0.0});
  path[0] = {100.0, 100.0};
  path[1] = {100.0, 101.0};
  path.clear();

  EXPECT_FALSE(monitor.Update(AtSeconds(1), {1.0, 0.0}, false));
  EXPECT_DOUBLE_EQ(monitor.historical_max_arc_length_m(), 1.0);
}

TEST(ProgressMonitor, ProjectsLiteralSingleRepeatedAndNonconsecutivePoints) {
  ProgressMonitor single(Parameters());
  ResetPath(single, AtSeconds(0), {{1.0, 1.0}}, {-9.0, 7.0});
  EXPECT_DOUBLE_EQ(single.historical_max_arc_length_m(), 0.0);

  ProgressMonitor repeated(Parameters());
  ResetPath(repeated, AtSeconds(0),
            {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}},
            {1.0, 0.5});
  EXPECT_NEAR(repeated.historical_max_arc_length_m(), 1.5, 1.0e-12);

  ProgressMonitor nonconsecutive(Parameters());
  ResetPath(nonconsecutive, AtSeconds(0),
            {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}},
            {0.0, 0.0});
  EXPECT_NEAR(nonconsecutive.historical_max_arc_length_m(), 2.0, 1.0e-12);
}

TEST(ProgressMonitor, ChoosesGreaterArcAtSelfCrossing) {
  ProgressMonitor monitor(Parameters());
  ResetPath(monitor, AtSeconds(0),
            {{0.0, 0.0}, {2.0, 2.0}, {0.0, 2.0}, {2.0, 0.0}},
            {1.0, 1.0});
  const double expected = 2.0 + 3.0 * std::sqrt(2.0);
  EXPECT_NEAR(monitor.historical_max_arc_length_m(), expected,
              1.0e-12 * expected);
}

TEST(ProgressMonitor, ProjectsOntoBentPathInteriorNotEndpoint) {
  ProgressMonitor monitor(Parameters());
  ResetPath(monitor, AtSeconds(0),
            {{0.0, 0.0}, {2.0, 0.0}, {2.0, 2.0}}, {1.0, 0.5});
  EXPECT_NEAR(monitor.historical_max_arc_length_m(), 1.0, 1.0e-12);
}

TEST(ProgressMonitor, RejectsSegmentAndCumulativeDoubleArcOverflow) {
  const double maximum = std::numeric_limits<double>::max();
  const std::vector<Vec2> overflowing_segment{{maximum, 0.0},
                                               {-maximum, 0.0}};
  const std::vector<Vec2> overflowing_sum{{0.0, 0.0},
                                           {maximum * 0.75, 0.0},
                                           {0.0, 0.0}};
  ProgressMonitor fresh(Parameters());
  EXPECT_THROW(fresh.Reset(AtSeconds(0), overflowing_segment, {0.0, 0.0}),
               std::overflow_error);
  EXPECT_THROW(fresh.historical_max_arc_length_m(), std::logic_error);

  ProgressMonitor active(Parameters());
  active.Reset(AtSeconds(0), UnitLine(), {0.1, 0.0});
  EXPECT_THROW(active.Reset(AtSeconds(10), overflowing_sum, {0.0, 0.0}),
               std::overflow_error);
  EXPECT_DOUBLE_EQ(active.historical_max_arc_length_m(), 0.1);
  EXPECT_FALSE(active.Update(AtSeconds(29), {0.1, 0.0}, false));
  EXPECT_TRUE(active.Update(AtSeconds(30), {0.1, 0.0}, false));

  ProgressMonitor latched(Parameters());
  latched.Reset(AtSeconds(0), UnitLine(), {0.0, 0.0});
  EXPECT_TRUE(latched.Update(AtSeconds(30), {0.0, 0.0}, false));
  EXPECT_THROW(
      latched.Reset(AtSeconds(31), overflowing_segment, {0.0, 0.0}),
      std::overflow_error);
  EXPECT_DOUBLE_EQ(latched.historical_max_arc_length_m(), 0.0);
  EXPECT_TRUE(latched.Update(AtSeconds(31), {1.0, 0.0}, false));
  EXPECT_DOUBLE_EQ(latched.historical_max_arc_length_m(), 1.0);
}

TEST(ProgressMonitor, UsesLongDoubleForExtremeFinitePositionProjection) {
  const double maximum = std::numeric_limits<double>::max();
  ProgressMonitor monitor(Parameters());
  ResetPath(monitor, AtSeconds(0), {{-1.0, 0.0}, {1.0, 0.0}},
            {maximum, maximum});
  EXPECT_DOUBLE_EQ(monitor.historical_max_arc_length_m(), 2.0);

  ResetPath(monitor, AtSeconds(1), {{-1.0, 0.0}, {1.0, 0.0}},
            {-maximum, -maximum});
  EXPECT_DOUBLE_EQ(monitor.historical_max_arc_length_m(), 0.0);
}

TEST(ProgressMonitor, FreezesTwentyNineThirtyAndProgressBoundaries) {
  const double just_below = std::nextafter(0.2, 0.0);
  const double just_above =
      std::nextafter(0.2, std::numeric_limits<double>::infinity());

  ProgressMonitor below(Parameters());
  below.Reset(AtSeconds(0), UnitLine(), {0.0, 0.0});
  EXPECT_FALSE(below.Update(AtSeconds(29), {just_below, 0.0}, false));
  EXPECT_TRUE(below.Update(AtSeconds(30), {just_below, 0.0}, false));

  ProgressMonitor exact(Parameters());
  exact.Reset(AtSeconds(0), UnitLine(), {0.0, 0.0});
  EXPECT_FALSE(exact.Update(AtSeconds(29), {0.2, 0.0}, false));
  EXPECT_FALSE(exact.Update(AtSeconds(58), {0.2, 0.0}, false));
  EXPECT_TRUE(exact.Update(AtSeconds(59), {0.2, 0.0}, false));

  ProgressMonitor above(Parameters());
  above.Reset(AtSeconds(0), UnitLine(), {0.0, 0.0});
  EXPECT_FALSE(above.Update(AtSeconds(30), {just_above, 0.0}, false));
  EXPECT_FALSE(above.Update(AtSeconds(59), {just_above, 0.0}, false));
  EXPECT_TRUE(above.Update(AtSeconds(60), {just_above, 0.0}, false));
}

TEST(ProgressMonitor, HistoricalMaximumNeverMovesBackward) {
  ProgressMonitor monitor(Parameters());
  monitor.Reset(AtSeconds(0), UnitLine(), {0.4, 0.0});
  EXPECT_FALSE(monitor.Update(AtSeconds(1), {0.1, 0.0}, false));
  EXPECT_DOUBLE_EQ(monitor.historical_max_arc_length_m(), 0.4);
}

TEST(ProgressMonitor, ComparesExtremeSteadyClockElapsedWithoutOverflow) {
  ProgressMonitor forward(Parameters());
  forward.Reset(TimePoint::min(), UnitLine(), {0.0, 0.0});
  EXPECT_TRUE(forward.Update(TimePoint::max(), {0.0, 0.0}, false));

  ProgressMonitor reverse(Parameters());
  reverse.Reset(TimePoint::max(), UnitLine(), {0.0, 0.0});
  EXPECT_THROW(reverse.Update(TimePoint::min(), {0.0, 0.0}, false),
               std::invalid_argument);
  EXPECT_FALSE(reverse.Update(TimePoint::max(), {0.0, 0.0}, false));
  EXPECT_DOUBLE_EQ(reverse.historical_max_arc_length_m(), 0.0);
}

TEST(ProgressMonitor, ResetWorksFromActiveAndStuckAndClearsWindow) {
  ProgressMonitor active(Parameters());
  active.Reset(AtSeconds(0), UnitLine(), {0.5, 0.0});
  ResetPath(active, AtSeconds(10), {{0.0, 0.0}, {0.0, 2.0}},
            {0.0, 1.0});
  EXPECT_DOUBLE_EQ(active.historical_max_arc_length_m(), 1.0);
  ExpectTimeoutFrom(active, 10, {0.0, 1.0});

  ProgressMonitor stuck(Parameters());
  stuck.Reset(AtSeconds(0), UnitLine(), {0.0, 0.0});
  EXPECT_TRUE(stuck.Update(AtSeconds(30), {0.0, 0.0}, false));
  stuck.Reset(AtSeconds(31), UnitLine(), {0.7, 0.0});
  EXPECT_DOUBLE_EQ(stuck.historical_max_arc_length_m(), 0.7);
  ExpectTimeoutFrom(stuck, 31, {0.7, 0.0});
}

TEST(ProgressMonitor, ResetAcceptsTheLastAcceptedTimeFromActiveAndStuck) {
  ProgressMonitor active(Parameters());
  active.Reset(AtSeconds(0), UnitLine(), {0.0, 0.0});
  EXPECT_FALSE(active.Update(AtSeconds(10), {0.1, 0.0}, false));
  EXPECT_NO_THROW(ResetPath(active, AtSeconds(10),
                            {{0.0, 0.0}, {0.0, 2.0}}, {0.0, 1.5}));
  EXPECT_DOUBLE_EQ(active.historical_max_arc_length_m(), 1.5);
  ExpectTimeoutFrom(active, 10, {0.0, 1.5});

  ProgressMonitor stuck(Parameters());
  stuck.Reset(AtSeconds(0), UnitLine(), {0.0, 0.0});
  EXPECT_TRUE(stuck.Update(AtSeconds(30), {0.0, 0.0}, false));
  EXPECT_NO_THROW(ResetPath(stuck, AtSeconds(30),
                            {{0.0, 0.0}, {0.0, 2.0}}, {0.0, 1.5}));
  EXPECT_DOUBLE_EQ(stuck.historical_max_arc_length_m(), 1.5);
  ExpectTimeoutFrom(stuck, 30, {0.0, 1.5});
}

TEST(ProgressMonitor, BackwardAndPausedResetPreserveCompleteBehavior) {
  ProgressMonitor backward(Parameters());
  backward.Reset(AtSeconds(0), UnitLine(), {0.0, 0.0});
  EXPECT_FALSE(backward.Update(AtSeconds(10), {0.1, 0.0}, false));
  EXPECT_THROW(ResetPath(backward, AtSeconds(9),
                         {{0.0, 0.0}, {0.0, 2.0}}, {0.0, 1.0}),
               std::invalid_argument);
  EXPECT_FALSE(backward.Update(AtSeconds(10), {0.0, 1.5}, false));
  EXPECT_DOUBLE_EQ(backward.historical_max_arc_length_m(), 0.1);
  EXPECT_FALSE(backward.Update(AtSeconds(29), {0.0, 1.5}, false));
  EXPECT_TRUE(backward.Update(AtSeconds(30), {0.0, 1.5}, false));

  ProgressMonitor paused(Parameters());
  paused.Reset(AtSeconds(0), UnitLine(), {0.0, 0.0});
  EXPECT_FALSE(paused.Update(AtSeconds(10), {0.9, 0.0}, true));
  EXPECT_THROW(ResetPath(paused, AtSeconds(20),
                         {{0.0, 0.0}, {0.0, 2.0}}, {0.0, 1.0}),
               std::logic_error);
  EXPECT_DOUBLE_EQ(paused.historical_max_arc_length_m(), 0.0);
  EXPECT_FALSE(paused.Update(AtSeconds(20), {0.5, 0.0}, false));
  EXPECT_DOUBLE_EQ(paused.historical_max_arc_length_m(), 0.5);
  ExpectTimeoutFrom(paused, 20, {0.5, 0.0});
}

TEST(ProgressMonitor, StuckRemainsLatchedUntilResume) {
  ProgressMonitor monitor(Parameters());
  monitor.Reset(AtSeconds(0), UnitLine(), {0.0, 0.0});
  EXPECT_TRUE(monitor.Update(AtSeconds(30), {0.0, 0.0}, false));
  EXPECT_TRUE(monitor.Update(AtSeconds(31), {1.0, 0.0}, false));
  EXPECT_DOUBLE_EQ(monitor.historical_max_arc_length_m(), 1.0);
  EXPECT_TRUE(monitor.Update(AtSeconds(32), {0.0, 0.0}, true));
  EXPECT_FALSE(monitor.Update(AtSeconds(33), {0.6, 0.0}, false));
  EXPECT_DOUBLE_EQ(monitor.historical_max_arc_length_m(), 0.6);
  ExpectTimeoutFrom(monitor, 33, {0.6, 0.0});
}

TEST(ProgressMonitor, PauseIgnoresFinitePositionAndResumeBuildsFullWindow) {
  ProgressMonitor monitor(Parameters());
  monitor.Reset(AtSeconds(0), UnitLine(), {0.1, 0.0});
  EXPECT_FALSE(monitor.Update(AtSeconds(20), {0.9, 0.0}, true));
  EXPECT_DOUBLE_EQ(monitor.historical_max_arc_length_m(), 0.1);
  EXPECT_FALSE(monitor.Update(AtSeconds(1000), {0.8, 0.0}, true));
  EXPECT_DOUBLE_EQ(monitor.historical_max_arc_length_m(), 0.1);
  EXPECT_FALSE(monitor.Update(AtSeconds(2000), {0.7, 0.0}, false));
  EXPECT_DOUBLE_EQ(monitor.historical_max_arc_length_m(), 0.7);
  ExpectTimeoutFrom(monitor, 2000, {0.7, 0.0});
}

enum class UpdateBranch { kActive, kFirstPause, kAlreadyPaused, kFirstResume };

void SetUpBranch(ProgressMonitor& monitor, UpdateBranch branch) {
  monitor.Reset(AtSeconds(0), UnitLine(), {0.0, 0.0});
  if (branch == UpdateBranch::kAlreadyPaused ||
      branch == UpdateBranch::kFirstResume) {
    EXPECT_FALSE(monitor.Update(AtSeconds(10), {0.0, 0.0}, true));
  }
}

void ProveUnchangedAfterRejectedUpdate(ProgressMonitor& monitor,
                                       UpdateBranch branch) {
  EXPECT_DOUBLE_EQ(monitor.historical_max_arc_length_m(), 0.0);
  switch (branch) {
    case UpdateBranch::kActive:
      ExpectTimeoutFrom(monitor, 0);
      return;
    case UpdateBranch::kFirstPause:
      EXPECT_FALSE(monitor.Update(AtSeconds(10), {0.0, 0.0}, true));
      EXPECT_FALSE(monitor.Update(AtSeconds(20), {0.0, 0.0}, false));
      ExpectTimeoutFrom(monitor, 20);
      return;
    case UpdateBranch::kAlreadyPaused:
      EXPECT_FALSE(monitor.Update(AtSeconds(20), {0.0, 0.0}, true));
      EXPECT_FALSE(monitor.Update(AtSeconds(30), {0.0, 0.0}, false));
      ExpectTimeoutFrom(monitor, 30);
      return;
    case UpdateBranch::kFirstResume:
      EXPECT_FALSE(monitor.Update(AtSeconds(20), {0.0, 0.0}, false));
      ExpectTimeoutFrom(monitor, 20);
      return;
  }
}

TEST(ProgressMonitor, RejectsNonfinitePositionBeforeAllFourPauseBranches) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  const std::vector<Vec2> invalid_positions{
      {nan, 0.0}, {0.0, nan}, {infinity, 0.0}, {-infinity, 0.0},
      {0.0, infinity}, {0.0, -infinity}};
  const std::vector<UpdateBranch> branches{
      UpdateBranch::kActive, UpdateBranch::kFirstPause,
      UpdateBranch::kAlreadyPaused, UpdateBranch::kFirstResume};

  for (const UpdateBranch branch : branches) {
    for (const Vec2 invalid : invalid_positions) {
      SCOPED_TRACE(static_cast<int>(branch));
      SCOPED_TRACE(invalid.x);
      SCOPED_TRACE(invalid.y);
      ProgressMonitor monitor(Parameters());
      SetUpBranch(monitor, branch);
      const bool paused = branch == UpdateBranch::kFirstPause ||
                          branch == UpdateBranch::kAlreadyPaused;
      const std::int64_t attempted_time =
          branch == UpdateBranch::kActive ? 10 : 20;
      EXPECT_THROW(
          monitor.Update(AtSeconds(attempted_time), invalid, paused),
          std::invalid_argument);
      ProveUnchangedAfterRejectedUpdate(monitor, branch);
    }
  }
}

TEST(ProgressMonitor, RejectsBackwardTimeInAllFourPauseBranches) {
  const std::vector<UpdateBranch> branches{
      UpdateBranch::kActive, UpdateBranch::kFirstPause,
      UpdateBranch::kAlreadyPaused, UpdateBranch::kFirstResume};
  for (const UpdateBranch branch : branches) {
    SCOPED_TRACE(static_cast<int>(branch));
    ProgressMonitor monitor(Parameters());
    SetUpBranch(monitor, branch);
    const bool paused = branch == UpdateBranch::kFirstPause ||
                        branch == UpdateBranch::kAlreadyPaused;
    const std::int64_t attempted_time =
        (branch == UpdateBranch::kActive ||
         branch == UpdateBranch::kFirstPause)
            ? -1
            : 9;
    EXPECT_THROW(monitor.Update(AtSeconds(attempted_time), {0.0, 0.0}, paused),
                 std::invalid_argument);
    ProveUnchangedAfterRejectedUpdate(monitor, branch);
  }
}

}  // namespace
}  // namespace lunar::pure_exploration
