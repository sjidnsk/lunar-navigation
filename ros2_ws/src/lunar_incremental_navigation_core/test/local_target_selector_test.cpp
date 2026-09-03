#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/local_target_selector.hpp"
#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"
#include "local/local_target_selector_internal.hpp"

namespace lunar::incremental_navigation {
namespace {

using Cell = std::pair<GridIndex, std::pair<FineCellState, double>>;

static_assert(std::is_same_v<
              decltype(std::declval<const LocalTargetSelector&>().Select(
                  std::declval<const FineTraversabilitySnapshot&>(), Point2{},
                  FinalGoal{}, std::optional<GlobalRoute>{})),
              std::optional<LocalTarget>>);

[[nodiscard]] RigidTransform IdentityMapTransform() {
  return RigidTransform{.parent_frame = "map", .child_frame = "map"};
}

[[nodiscard]] std::shared_ptr<const ElevationSnapshot> MakeElevation(
    const std::size_t width, const std::size_t height,
    const double resolution_m) {
  PersistentElevationMap map;
  const GridGeometry geometry{.frame_id = "map",
                              .width = width,
                              .height = height,
                              .resolution_m = resolution_m};
  const std::vector<float> elevation(geometry.CellCount(), 0.0F);
  const ElevationUpdateResult update = map.Apply(ElevationEvidence{
      .geometry = geometry,
      .elevation_m = elevation,
      .map_from_source = IdentityMapTransform(),
  });
  if (update.status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("test elevation fixture was rejected");
  }
  return map.Snapshot();
}

[[nodiscard]] std::shared_ptr<const FineTraversabilitySnapshot> MakeFine(
    const std::initializer_list<Cell> cells, const double resolution_m = 1.0,
    const double hard_inflation_radius_m = 1.25,
    const double preferred_clearance_m = 0.8) {
  const auto elevation = MakeElevation(8U, 5U, resolution_m);
  FineTraversabilityTile::StateArray states;
  states.fill(FineCellState::kUnknown);
  FineTraversabilityTile::CostArray costs;
  costs.fill(0.0);
  for (const auto& [index, value] : cells) {
    states[TileCellOffset(index)] = value.first;
    costs[TileCellOffset(index)] = value.second;
  }
  const auto tile = std::make_shared<const FineTraversabilityTile>(
      std::move(states), std::move(costs));
  FineTraversabilityTileDirectory directory(elevation->geometry());
  directory = directory.WithTile(TileIndex{}, tile);
  return std::make_shared<const FineTraversabilitySnapshot>(
      elevation->geometry(), elevation->raw_elevation_revision(), 1U,
      "wheel-profile", hard_inflation_radius_m, preferred_clearance_m,
      TraversalCostWeights{}, elevation, std::move(directory),
      std::vector<TileIndex>{TileIndex{}},
      std::vector<TileIndex>{TileIndex{}}, FineSnapshotMetrics{});
}

[[nodiscard]] std::shared_ptr<const FineTraversabilitySnapshot>
MakeTwoTileSparseFine(const std::int64_t far_tile_x) {
  constexpr double kResolutionM = 1.0;
  PersistentElevationMap map;
  const std::array<float, 1U> elevation{0.0F};
  const auto apply = [&](const std::int64_t cell_x) {
    const GridGeometry geometry{
        .frame_id = "map",
        .width = 1U,
        .height = 1U,
        .resolution_m = kResolutionM,
        .origin_m = {.x = static_cast<double>(cell_x)}};
    return map.Apply(ElevationEvidence{
        .geometry = geometry,
        .elevation_m = elevation,
        .map_from_source = IdentityMapTransform(),
    });
  };
  const std::int64_t far_cell_x = far_tile_x * kGridTileWidthCells;
  if (apply(0).status != ElevationUpdateResult::Status::kApplied ||
      apply(far_cell_x).status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("sparse elevation fixture was rejected");
  }
  const auto raw = map.Snapshot();
  const auto make_tile = [](const GridIndex free_cell) {
    FineTraversabilityTile::StateArray states;
    states.fill(FineCellState::kUnknown);
    FineTraversabilityTile::CostArray costs;
    costs.fill(0.0);
    states[TileCellOffset(free_cell)] = FineCellState::kFree;
    return std::make_shared<const FineTraversabilityTile>(std::move(states),
                                                          std::move(costs));
  };
  FineTraversabilityTileDirectory directory(raw->geometry());
  directory = directory.WithTile(TileIndex{}, make_tile({.x = 0, .y = 0}));
  directory = directory.WithTile(
      TileIndex{.x = far_tile_x, .y = 0},
      make_tile({.x = far_cell_x, .y = 0}));
  return std::make_shared<const FineTraversabilitySnapshot>(
      raw->geometry(), raw->raw_elevation_revision(), 1U, "wheel-profile",
      0.1, 0.0, TraversalCostWeights{}, raw, std::move(directory),
      std::vector<TileIndex>{}, std::vector<TileIndex>{},
      FineSnapshotMetrics{});
}

[[nodiscard]] Pose3 RoutePose(const double x_m, const double y_m) {
  return Pose3{.position_m = {.x = x_m, .y = y_m, .z = 0.0}};
}

TEST(LocalTargetSelector, ReturnsTheTrueFreeFinalGoalAndItsOptionalYaw) {
  const auto fine = MakeFine({{{.x = 2, .y = 2},
                               {FineCellState::kFree, 0.4}}});
  const FinalGoal goal{.target_x_m = 2.2,
                       .target_y_m = 2.4,
                       .has_target_yaw = true,
                       .target_yaw_rad = 0.7};

  const auto selected =
      LocalTargetSelector{}.Select(*fine, {.x = 0.5, .y = 2.5}, goal,
                                   std::nullopt);

  ASSERT_TRUE(selected);
  EXPECT_EQ(selected->center, (Point2{.x = 2.2, .y = 2.4}));
  EXPECT_DOUBLE_EQ(selected->position_tolerance_m, 0.5);
  EXPECT_TRUE(selected->is_final_goal);
  ASSERT_TRUE(selected->terminal_yaw_rad);
  EXPECT_DOUBLE_EQ(*selected->terminal_yaw_rad, 0.7);
}

TEST(LocalTargetSelector,
     DoesNotUseTheHardInflationRadiusAsTheGoalPositionTolerance) {
  const auto fine = MakeFine({{{.x = 2, .y = 2},
                               {FineCellState::kFree, 0.4}}},
                             0.5, 1.25, 10.0);
  const FinalGoal goal{.target_x_m = 2.2, .target_y_m = 2.4};

  const auto selected =
      LocalTargetSelector{}.Select(*fine, {.x = 0.5, .y = 2.5}, goal,
                                   std::nullopt);

  ASSERT_TRUE(selected);
  EXPECT_DOUBLE_EQ(selected->position_tolerance_m, 0.25);
}

TEST(LocalTargetSelector,
     GuidanceSelectsOneFreeTargetNearKnownAreaExitWithoutMakingACorridor) {
  const auto fine = MakeFine({
      {{.x = 1, .y = 2}, {FineCellState::kFree, 0.1}},
      {{.x = 2, .y = 2}, {FineCellState::kFree, 0.1}},
      {{.x = 3, .y = 2}, {FineCellState::kFree, 0.1}},
      {{.x = 4, .y = 2}, {FineCellState::kBlocked, 0.0}},
      {{.x = 4, .y = 1}, {FineCellState::kFree, 9.0}},
  });
  const GlobalRoute guidance{
      .poses_map = {RoutePose(1.5, 2.5), RoutePose(7.5, 2.5)}};
  const FinalGoal goal{.target_x_m = 20.0,
                       .target_y_m = 2.5,
                       .has_target_yaw = true,
                       .target_yaw_rad = -0.4};

  const auto selected = LocalTargetSelector{}.Select(
      *fine, {.x = 1.5, .y = 2.5}, goal, guidance);

  ASSERT_TRUE(selected);
  EXPECT_EQ(selected->center, (Point2{.x = 4.5, .y = 1.5}));
  EXPECT_DOUBLE_EQ(selected->position_tolerance_m, 0.5);
  EXPECT_FALSE(selected->is_final_goal);
  EXPECT_FALSE(selected->terminal_yaw_rad);
}

TEST(LocalTargetSelector,
     WithoutGuidanceChoosesTheFurthestFreeFrontierTowardTheFinalGoal) {
  const auto fine = MakeFine({
      {{.x = 0, .y = 2}, {FineCellState::kFree, 0.0}},
      {{.x = 1, .y = 2}, {FineCellState::kFree, 0.0}},
      {{.x = 2, .y = 2}, {FineCellState::kFree, 4.0}},
      {{.x = 1, .y = 1}, {FineCellState::kBlocked, 0.0}},
  });
  const FinalGoal goal{.target_x_m = 20.0,
                       .target_y_m = 2.5,
                       .has_target_yaw = true,
                       .target_yaw_rad = 1.0};

  const auto selected = LocalTargetSelector{}.Select(
      *fine, {.x = 0.5, .y = 2.5}, goal, std::nullopt);

  ASSERT_TRUE(selected);
  EXPECT_EQ(selected->center, (Point2{.x = 2.5, .y = 2.5}));
  EXPECT_FALSE(selected->is_final_goal);
  EXPECT_FALSE(selected->terminal_yaw_rad);
}

TEST(LocalTargetSelector,
     SoftTraversalCostBreaksAnEqualGeometryTieWithoutDeletingCandidates) {
  const auto fine = MakeFine({
      {{.x = 6, .y = 1}, {FineCellState::kFree, 10.0}},
      {{.x = 6, .y = 3}, {FineCellState::kFree, 1.0}},
  }, 0.5, 0.1, 10.0);
  const FinalGoal goal{.target_x_m = 20.0, .target_y_m = 1.25};

  const auto selected = LocalTargetSelector{}.Select(
      *fine, {.x = 0.25, .y = 1.25}, goal, std::nullopt);

  ASSERT_TRUE(selected);
  EXPECT_EQ(selected->center, (Point2{.x = 3.25, .y = 1.75}));
}

TEST(LocalTargetSelector, SoftTraversalCostNeverDeletesTheOnlyFreeCandidate) {
  const auto fine = MakeFine(
      {{{.x = 6, .y = 2}, {FineCellState::kFree, 1000.0}}}, 0.5, 0.1,
      10.0);
  const FinalGoal goal{.target_x_m = 20.0, .target_y_m = 1.25};

  const auto selected = LocalTargetSelector{}.Select(
      *fine, {.x = 0.25, .y = 1.25}, goal, std::nullopt);

  ASSERT_TRUE(selected);
  EXPECT_EQ(selected->center, (Point2{.x = 3.25, .y = 1.25}));
  EXPECT_DOUBLE_EQ(selected->position_tolerance_m, 0.25);
}

TEST(LocalTargetSelector, ReturnsNoTargetWithoutAnyBaseFreeCandidate) {
  const auto fine = MakeFine({
      {{.x = 2, .y = 2}, {FineCellState::kBlocked, 0.0}},
      {{.x = 3, .y = 2}, {FineCellState::kUnknown, 0.0}},
  });
  const GlobalRoute guidance{
      .poses_map = {RoutePose(0.5, 2.5), RoutePose(7.5, 2.5)}};
  const FinalGoal goal{.target_x_m = 2.2,
                       .target_y_m = 2.4,
                       .has_target_yaw = true,
                       .target_yaw_rad = 0.7};

  EXPECT_FALSE(LocalTargetSelector{}.Select(
      *fine, {.x = 0.5, .y = 2.5}, goal, guidance));
}

TEST(LocalTargetSelector, RestrictsFinalAndFallbackTargetsToTheLocalWindow) {
  const auto fine = MakeFine({
      {{.x = 2, .y = 2}, {FineCellState::kFree, 0.0}},
      {{.x = 6, .y = 2}, {FineCellState::kFree, 0.0}},
  });
  const SparseGridGeometry local_window(
      "map", 1.0, {}, {.x = 0, .y = 0}, {.x = 4, .y = 5});
  const FinalGoal goal{.target_x_m = 6.2,
                       .target_y_m = 2.4,
                       .has_target_yaw = true,
                       .target_yaw_rad = 0.7};

  const auto selected = LocalTargetSelector{}.Select(
      *fine, local_window, {.x = 0.5, .y = 2.5}, goal, std::nullopt);

  ASSERT_TRUE(selected);
  EXPECT_EQ(selected->center, (Point2{.x = 2.5, .y = 2.5}));
  EXPECT_FALSE(selected->is_final_goal);
  EXPECT_FALSE(selected->terminal_yaw_rad);
}

TEST(LocalTargetSelector,
     SparseCandidateWorkDependsOnAllocatedTilesNotBoundingBoxHoles) {
  const auto near = MakeTwoTileSparseFine(4);
  const auto far = MakeTwoTileSparseFine(1'000'000);
  local_target_selector_internal::SelectionMetrics near_metrics;
  local_target_selector_internal::SelectionMetrics far_metrics;

  const auto near_selected = local_target_selector_internal::Select(
      *near, {.x = 0.5, .y = 0.5},
      FinalGoal{.target_x_m = 2000.0, .target_y_m = 0.5}, std::nullopt,
      &near_metrics);
  const auto far_selected = local_target_selector_internal::Select(
      *far, {.x = 0.5, .y = 0.5},
      FinalGoal{.target_x_m = 1.0e12, .target_y_m = 0.5}, std::nullopt,
      &far_metrics);

  ASSERT_TRUE(near_selected);
  ASSERT_TRUE(far_selected);
  EXPECT_EQ(near_metrics.candidate_tile_lookups, 2U);
  EXPECT_EQ(far_metrics.candidate_tile_lookups, 2U);
  EXPECT_EQ(near_metrics.candidate_cells_examined, 257U);
  EXPECT_EQ(far_metrics.candidate_cells_examined, 257U);
  EXPECT_EQ(near_metrics.candidate_tile_lookups,
            far_metrics.candidate_tile_lookups);
  EXPECT_EQ(near_metrics.candidate_cells_examined,
            far_metrics.candidate_cells_examined);
  EXPECT_EQ(near_selected->center.x,
            4.0 * static_cast<double>(kGridTileWidthCells) + 0.5);
  EXPECT_EQ(far_selected->center.x,
            1'000'000.0 * static_cast<double>(kGridTileWidthCells) + 0.5);
}

TEST(LocalTargetSelector,
     BoundedWindowDoesNotWalkPersistentHistoryTilesOutsideTheWindow) {
  const auto fine = MakeTwoTileSparseFine(1'000'000);
  const SparseGridGeometry local_window(
      "map", 1.0, {}, {.x = 0, .y = 0}, {.x = 320, .y = 1});
  local_target_selector_internal::SelectionMetrics metrics;

  const auto selected = local_target_selector_internal::Select(
      *fine, local_window, {.x = 0.5, .y = 0.5},
      FinalGoal{.target_x_m = 1.0e12, .target_y_m = 0.5}, std::nullopt,
      &metrics);

  ASSERT_TRUE(selected);
  EXPECT_EQ(selected->center, (Point2{.x = 0.5, .y = 0.5}));
  EXPECT_EQ(metrics.candidate_tile_lookups, 2U);
  EXPECT_EQ(metrics.candidate_cells_examined, 256U);
}

TEST(LocalTargetSelector,
     ClipsFarAndExtremeFiniteGuidanceFromAnUnknownStart) {
  const auto fine = MakeFine({
      {{.x = 1, .y = 2}, {FineCellState::kFree, 0.1}},
      {{.x = 2, .y = 2}, {FineCellState::kFree, 0.1}},
      {{.x = 3, .y = 2}, {FineCellState::kFree, 0.1}},
      {{.x = 4, .y = 2}, {FineCellState::kBlocked, 0.0}},
      {{.x = 4, .y = 1}, {FineCellState::kFree, 0.1}},
  });
  for (const double endpoint_x : {1.0e12, 1.0e308}) {
    const GlobalRoute guidance{
        .poses_map = {RoutePose(endpoint_x, 2.5)}};
    local_target_selector_internal::SelectionMetrics metrics;

    const auto selected = local_target_selector_internal::Select(
        *fine, {.x = 0.5, .y = 2.5},
        FinalGoal{.target_x_m = 0.5, .target_y_m = 100.0}, guidance,
        &metrics);

    ASSERT_TRUE(selected) << endpoint_x;
    EXPECT_EQ(selected->center, (Point2{.x = 4.5, .y = 1.5}))
        << endpoint_x;
    EXPECT_LE(metrics.guidance_cells_examined,
              fine->geometry().width() + fine->geometry().height())
        << endpoint_x;
  }
}

}  // namespace
}  // namespace lunar::incremental_navigation
