#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stop_token>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "hierarchical/surface_global_search.hpp"
#include "shared/global_occupancy_projection.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::pure_planning::hierarchical {
namespace {

[[nodiscard]] std::shared_ptr<const shared::MapSnapshot> GlobalMap(
    std::vector<std::int8_t> occupancy, const std::size_t width,
    const double resolution_m = 1.0) {
  GridMap map{
      .frame_id = "map",
      .stamp = TimePoint{.nanoseconds_since_epoch = 1},
      .width = width,
      .height = occupancy.size() / width,
      .resolution_m = resolution_m,
      .origin_m = Vec3{},
      .layers = {},
  };
  map.layers.emplace("occupancy", GridLayer{.values = std::move(occupancy)});
  const auto result = shared::MapSnapshot::Create(
      std::move(map), shared::MapContract::kGlobalOccupancy);
  EXPECT_TRUE(result.ok()) << result.reason_code;
  return result.snapshot;
}

struct TestProblem final {
  shared::GlobalOccupancyProjection projection;
  SurfaceGlobalSearchProblem problem;
};

[[nodiscard]] TestProblem Problem(
    std::vector<std::int8_t> occupancy, const std::size_t width,
    const shared::GridCell start, const shared::GridCell goal,
    const double resolution_m = 1.0) {
  auto projection = shared::BuildGlobalOccupancyProjection(
      GlobalMap(std::move(occupancy), width, resolution_m));
  EXPECT_TRUE(projection.ok()) << projection.reason_code;
  TestProblem result{
      .projection = std::move(*projection.projection),
      .problem = {
      .projection = {},
      .start = start,
      .goal = goal,
      .start_pose_map = {},
      .goal_pose_map = {},
      .control = {},
      .search = {},
      },
  };
  result.problem.projection = result.projection.View();
  result.problem.start_pose_map.position_m =
      result.projection.View().map->CellCenter(start);
  result.problem.goal_pose_map.position_m =
      result.projection.View().map->CellCenter(goal);
  return result;
}

TEST(SurfaceGlobalSearch, UsesNativeInt8AndActualResolution) {
  auto projection = shared::BuildGlobalOccupancyProjection(
      GlobalMap({-1, 0, 49, 50, 100, 101}, 6U, 0.75));
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  const auto view = projection.projection->View();
  EXPECT_TRUE(view.HardFeasible({1, 0}));
  EXPECT_TRUE(view.HardFeasible({2, 0}));
  EXPECT_FALSE(view.HardFeasible({3, 0}));
  EXPECT_DOUBLE_EQ(view.map->resolution_m(), 0.75);

  auto problem = Problem(
      {0, 0, 0}, 3U, {.x = 0, .y = 0}, {.x = 2, .y = 0}, 0.75);
  const auto result = SearchSurfaceGlobal(problem.problem);
  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_DOUBLE_EQ(result.cost, 1.5);
}

TEST(SurfaceGlobalSearch, DoesNotCutBlockedDiagonalCorners) {
  auto problem = Problem(
      {0, 50, 50, 0}, 2U, {.x = 0, .y = 0}, {.x = 1, .y = 1});
  const auto result = SearchSurfaceGlobal(problem.problem);

  EXPECT_EQ(result.status, SurfaceGlobalSearchStatus::kNoPath);
  EXPECT_TRUE(result.raw_cells.empty());
}

TEST(SurfaceGlobalSearch, SimplifiesOnlyAcrossFreeSupercoverCells) {
  auto problem = Problem(
      {0, 0, 0, 0, 50, 0, 0, 0, 0}, 3U,
      {.x = 0, .y = 0}, {.x = 2, .y = 2});
  const auto result = SearchSurfaceGlobal(problem.problem);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GT(result.simplified_cells.size(), 2U);
  EXPECT_FALSE(std::ranges::any_of(result.simplified_cells,
                                   [](shared::GridCell cell) {
                                     return cell == shared::GridCell{1, 1};
                                   }));
}

TEST(SurfaceGlobalSearch, KeepsAnOffCenterGoalExactInPreview) {
  auto problem = Problem({0, 0, 0}, 3U, {.x = 0, .y = 0}, {.x = 2, .y = 0});
  problem.problem.start_pose_map.position_m = {.x = 0.2, .y = 0.3, .z = 0.0};
  problem.problem.goal_pose_map.position_m = {.x = 2.4, .y = 0.4, .z = 0.0};

  const auto result = SearchSurfaceGlobal(problem.problem);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_FALSE(result.preview.poses_map.empty());
  EXPECT_EQ(result.preview.poses_map.front(), problem.problem.start_pose_map);
  EXPECT_EQ(result.preview.poses_map.back(), problem.problem.goal_pose_map);
}

TEST(SurfaceGlobalSearch, KeepsNon45SupercoverWaypointForSingleSideObstacle) {
  std::vector<std::int8_t> occupancy(21U, 0);
  occupancy[18U] = 50;
  auto problem = Problem(std::move(occupancy), 7U,
                         {.x = 0, .y = 0}, {.x = 6, .y = 2});

  const auto result = SearchSurfaceGlobal(problem.problem);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GT(result.simplified_cells.size(), 2U);
}

TEST(SurfaceGlobalSearch, SimplifiesAxisAlignedAndReverseFreeSegments) {
  auto horizontal = Problem({0, 0, 0, 0}, 4U,
                            {.x = 0, .y = 0}, {.x = 3, .y = 0});
  auto vertical = Problem({0, 0, 0, 0}, 1U,
                          {.x = 0, .y = 0}, {.x = 0, .y = 3});
  auto reverse = Problem({0, 0, 0, 0}, 4U,
                          {.x = 3, .y = 0}, {.x = 0, .y = 0});

  EXPECT_EQ(SearchSurfaceGlobal(horizontal.problem).simplified_cells.size(), 2U);
  EXPECT_EQ(SearchSurfaceGlobal(vertical.problem).simplified_cells.size(), 2U);
  EXPECT_EQ(SearchSurfaceGlobal(reverse.problem).simplified_cells.size(), 2U);
}

TEST(SurfaceGlobalSearch, CertifiesExactEndpointSegmentsBeforeSimplifying) {
  auto problem = Problem({0, 0, 50, 0, 0, 0}, 3U,
                         {.x = 0, .y = 0}, {.x = 2, .y = 1});
  problem.problem.start_pose_map.position_m = {.x = 0.1, .y = 0.1, .z = 0.0};
  problem.problem.goal_pose_map.position_m = {.x = 2.9, .y = 1.1, .z = 0.0};

  const auto result = SearchSurfaceGlobal(problem.problem);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GT(result.preview.poses_map.size(), 2U);
}

TEST(SurfaceGlobalSearch, SupportsExactEndpointsInOneFreeCell) {
  auto problem = Problem({0}, 1U, {.x = 0, .y = 0}, {.x = 0, .y = 0});
  problem.problem.start_pose_map.position_m = {.x = 0.1, .y = 0.1, .z = 0.0};
  problem.problem.goal_pose_map.position_m = {.x = 0.9, .y = 0.9, .z = 0.0};

  const auto result = SearchSurfaceGlobal(problem.problem);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  ASSERT_EQ(result.preview.poses_map.size(), 2U);
  EXPECT_EQ(result.preview.poses_map.front(), problem.problem.start_pose_map);
  EXPECT_EQ(result.preview.poses_map.back(), problem.problem.goal_pose_map);
}

TEST(SurfaceGlobalSearch, RejectsExactPoseThatDoesNotBelongToItsCell) {
  auto problem = Problem({0, 0, 0}, 3U, {.x = 0, .y = 0}, {.x = 2, .y = 0});
  problem.problem.goal_pose_map.position_m = {.x = 1.2, .y = 0.2, .z = 0.0};

  const auto result = SearchSurfaceGlobal(problem.problem);

  EXPECT_EQ(result.status, SurfaceGlobalSearchStatus::kInvalidProblem);
  EXPECT_EQ(result.reason_code, "GLOBAL_SEARCH_POSE_CELL_MISMATCH");
}

TEST(SurfaceGlobalSearch, ReportsInfeasibleEndpointAsNoPath) {
  auto start_blocked = Problem({50, 0}, 2U,
                               {.x = 0, .y = 0}, {.x = 1, .y = 0});
  auto goal_unknown = Problem({0, -1}, 2U,
                              {.x = 0, .y = 0}, {.x = 1, .y = 0});

  const auto start = SearchSurfaceGlobal(start_blocked.problem);
  const auto goal = SearchSurfaceGlobal(goal_unknown.problem);

  EXPECT_EQ(start.status, SurfaceGlobalSearchStatus::kNoPath);
  EXPECT_EQ(start.reason_code, "GLOBAL_START_INFEASIBLE");
  EXPECT_EQ(goal.status, SurfaceGlobalSearchStatus::kNoPath);
  EXPECT_EQ(goal.reason_code, "GLOBAL_GOAL_INFEASIBLE");
}

TEST(SurfaceGlobalSearch, RejectsACentreDistanceShortcutAroundInflatedWallCorner) {
  constexpr std::size_t kWidth = 12U;
  std::vector<std::int8_t> occupancy(kWidth * 7U, 0);
  for (std::size_t y = 2U; y <= 4U; ++y) {
    occupancy[y * kWidth + 5U] = 100;
  }
  auto projection = shared::BuildInflatedGlobalOccupancyProjection(
      GlobalMap(std::move(occupancy), kWidth), 50, 0.9187);
  ASSERT_TRUE(projection.ok()) << projection.reason_code;

  SurfaceGlobalSearchProblem problem{
      .projection = projection.projection->View(),
      .start = {.x = 1, .y = 3},
      .goal = {.x = 10, .y = 3},
      .start_pose_map = {},
      .goal_pose_map = {},
      .control = {},
      .search = {},
  };
  problem.start_pose_map.position_m =
      problem.projection.map->CellCenter(problem.start);
  problem.goal_pose_map.position_m =
      problem.projection.map->CellCenter(problem.goal);

  const auto result = SearchSurfaceGlobal(problem);

  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_TRUE(std::ranges::all_of(
      result.raw_cells, [&problem](const shared::GridCell cell) {
        return problem.projection.HardFeasible(cell);
      }));
  EXPECT_TRUE(std::ranges::any_of(result.raw_cells,
                                  [](const shared::GridCell cell) {
                                    return cell.y == 0 || cell.y == 6;
                                  }));
}

TEST(SurfaceGlobalSearch, HandlesAThousandMeterMapWithSeededRandomObstacles) {
  constexpr std::size_t kWidth = 1000U;
  constexpr std::size_t kHeight = 1000U;
  constexpr shared::GridCell kStart{.x = 50, .y = 500};
  constexpr shared::GridCell kGoal{.x = 800, .y = 500};
  std::vector<std::int8_t> occupancy(kWidth * kHeight, 0);
  std::minstd_rand generator{0x5EED1234U};
  std::uniform_int_distribution<std::size_t> x_distribution{80U, 770U};
  std::uniform_int_distribution<std::size_t> y_distribution{40U, 960U};
  for (std::size_t index = 0U; index < 1800U; ++index) {
    const std::size_t x = x_distribution(generator);
    const std::size_t y = y_distribution(generator);
    // Keep endpoint neighborhoods valid; all other sampled obstacles are real
    // search obstacles, including those near the preferred direct corridor.
    if (std::hypot(static_cast<double>(x - kStart.x),
                   static_cast<double>(y - kStart.y)) > 8.0 &&
        std::hypot(static_cast<double>(x - kGoal.x),
                   static_cast<double>(y - kGoal.y)) > 8.0) {
      occupancy[y * kWidth + x] = 100;
    }
  }
  // Alternating gates guarantee that the 750 m request must perform several
  // obstacle-driven deviations instead of accepting a direct segment.
  for (const auto [x, gap_y] : {std::pair{180U, 350U}, std::pair{360U, 650U},
                                std::pair{540U, 350U}, std::pair{700U, 650U}}) {
    for (std::size_t y = 80U; y < 920U; ++y) {
      if (y < gap_y || y > gap_y + 80U) {
        occupancy[y * kWidth + x] = 100;
      }
    }
  }
  auto projection = shared::BuildInflatedGlobalOccupancyProjection(
      GlobalMap(std::move(occupancy), kWidth), 50, 1.0);
  ASSERT_TRUE(projection.ok()) << projection.reason_code;
  SurfaceGlobalSearchProblem problem{
      .projection = projection.projection->View(),
      .start = kStart,
      .goal = kGoal,
      .start_pose_map = {},
      .goal_pose_map = {},
      .control = {.deadline = SteadyClock::now() + std::chrono::seconds{20}},
      // Match the rolling dispatcher: this is a feasibility route, so the
      // first valid incumbent is sufficient and ARA* refinement is skipped.
      .search = {.stop_after_first_solution = true},
  };
  problem.start_pose_map.position_m = problem.projection.map->CellCenter(kStart);
  problem.goal_pose_map.position_m = problem.projection.map->CellCenter(kGoal);
  const auto result = SearchSurfaceGlobal(problem);
  ASSERT_TRUE(result.ok()) << result.reason_code;
  EXPECT_GT(result.cost, 750.0);
  EXPECT_GT(result.expanded_states, 750U);
  EXPECT_TRUE(std::ranges::all_of(result.raw_cells, [&problem](const auto cell) {
    return problem.projection.HardFeasible(cell);
  }));
}

TEST(SurfaceGlobalSearch, HonorsCancellationAfterAraReturnsAnIncumbent) {
  auto problem = Problem({0}, 1U, {.x = 0, .y = 0}, {.x = 0, .y = 0});
  std::stop_source stop_source;
  bool first_now = true;
  problem.problem.control.stop_token = stop_source.get_token();
  problem.problem.control.now = [&] {
    if (first_now) {
      first_now = false;
      stop_source.request_stop();
    }
    return SteadyClock::time_point::min();
  };

  const auto result = SearchSurfaceGlobal(problem.problem);

  EXPECT_EQ(result.status, SurfaceGlobalSearchStatus::kCanceled);
  EXPECT_EQ(result.reason_code, "REQUEST_CANCELED");
}

TEST(SurfaceGlobalSearch, TimesOutWhenDeadlineArrivesBeforeSingleCellCertification) {
  using namespace std::chrono_literals;
  auto problem = Problem({0}, 1U, {.x = 0, .y = 0}, {.x = 0, .y = 0});
  problem.problem.start_pose_map.position_m = {.x = 0.1, .y = 0.1, .z = 0.0};
  problem.problem.goal_pose_map.position_m = {.x = 0.9, .y = 0.9, .z = 0.0};
  const SteadyClock::time_point deadline{};
  std::size_t now_calls{};
  problem.problem.control.deadline = deadline;
  problem.problem.control.now = [&] {
    return now_calls++ == 0U ? deadline - 1ns : deadline;
  };

  const auto result = SearchSurfaceGlobal(problem.problem);

  ASSERT_EQ(result.status, SurfaceGlobalSearchStatus::kTimedOut);
  EXPECT_TRUE(result.deadline_reached);
  EXPECT_EQ(result.reason_code, "TIMEOUT");
  EXPECT_TRUE(result.preview.poses_map.empty());
}

TEST(SurfaceGlobalSearch, TimesOutWhenDeadlineInterruptsMultiCellRawCertification) {
  using namespace std::chrono_literals;
  constexpr auto kDeadline = SteadyClock::time_point{};
  auto problem = Problem({0, 0, 0, 0}, 4U,
                         {.x = 0, .y = 0}, {.x = 3, .y = 0});
  problem.problem.start_pose_map.position_m = {.x = 0.1, .y = 0.1, .z = 0.0};
  problem.problem.goal_pose_map.position_m = {.x = 3.9, .y = 0.9, .z = 0.0};
  std::size_t now_calls{};
  problem.problem.control.deadline = kDeadline;
  problem.problem.control.now = [&] {
    return now_calls++ < 102U ? kDeadline - 1ns : kDeadline;
  };

  const auto result = SearchSurfaceGlobal(problem.problem);

  EXPECT_EQ(result.status, SurfaceGlobalSearchStatus::kTimedOut);
  EXPECT_EQ(result.reason_code, "TIMEOUT");
  EXPECT_TRUE(result.deadline_reached);
  EXPECT_TRUE(result.preview.poses_map.empty());
}

TEST(SurfaceGlobalSearch, ReturnsRawRouteWhenDeadlineFollowsMultiCellCertification) {
  using namespace std::chrono_literals;
  constexpr auto kDeadline = SteadyClock::time_point{};
  auto problem = Problem({0, 0, 0, 0}, 4U,
                         {.x = 0, .y = 0}, {.x = 3, .y = 0});
  problem.problem.start_pose_map.position_m = {.x = 0.1, .y = 0.1, .z = 0.0};
  problem.problem.goal_pose_map.position_m = {.x = 3.9, .y = 0.9, .z = 0.0};
  std::size_t now_calls{};
  problem.problem.control.deadline = kDeadline;
  problem.problem.control.now = [&] {
    return now_calls++ < 103U ? kDeadline - 1ns : kDeadline;
  };

  const auto result = SearchSurfaceGlobal(problem.problem);

  EXPECT_EQ(result.status, SurfaceGlobalSearchStatus::kSolved);
  EXPECT_TRUE(result.deadline_reached);
  EXPECT_GT(result.raw_cells.size(), 2U);
  EXPECT_EQ(result.simplified_cells, result.raw_cells);
  EXPECT_FALSE(result.preview.poses_map.empty());
}

}  // namespace
}  // namespace lunar::pure_planning::hierarchical
