#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_ros/exploration_map_projector.hpp"

namespace lunar::incremental_navigation_ros {
namespace {

namespace core = lunar::incremental_navigation;

[[nodiscard]] core::GridGeometry DenseGeometry() {
  return core::GridGeometry{.frame_id = "map",
                            .width = 4U,
                            .height = 4U,
                            .resolution_m = 0.5,
                            .origin_m = {}};
}

[[nodiscard]] std::shared_ptr<const core::ElevationSnapshot> MakeElevation() {
  const core::GridGeometry geometry = DenseGeometry();
  const std::vector<float> elevation(geometry.CellCount(), 0.0F);
  core::PersistentElevationMap map;
  const auto result = map.Apply(core::ElevationEvidence{
      .geometry = geometry,
      .elevation_m = elevation,
      .map_from_source = {.parent_frame = "map",
                          .child_frame = "map",
                          .rotation = {.w = 1.0}},
  });
  if (result.status != core::ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("elevation fixture rejected");
  }
  return map.Snapshot();
}

[[nodiscard]] std::shared_ptr<const core::FineTraversabilitySnapshot> MakeFine(
    const std::vector<std::pair<core::GridIndex, core::FineCellState>>& cells,
    const std::uint64_t revision = 7U) {
  const auto elevation = MakeElevation();
  core::FineTraversabilityTile::StateArray states;
  states.fill(core::FineCellState::kUnknown);
  core::FineTraversabilityTile::CostArray costs;
  costs.fill(0.0);
  for (const auto& [index, state] : cells) {
    states[core::TileCellOffset(index)] = state;
  }
  auto tile = std::make_shared<const core::FineTraversabilityTile>(
      std::move(states), std::move(costs));
  core::FineTraversabilityTileDirectory directory(elevation->geometry());
  directory = directory.WithTile({.x = 0, .y = 0}, std::move(tile));
  return std::make_shared<const core::FineTraversabilitySnapshot>(
      elevation->geometry(), elevation->raw_elevation_revision(), revision,
      "wheel-profile", 0.0, 0.0, core::TraversalCostWeights{}, elevation,
      std::move(directory), std::vector<core::TileIndex>{},
      std::vector<core::TileIndex>{}, core::FineSnapshotMetrics{});
}

[[nodiscard]] core::SparseGridGeometry CoarseGeometry(
    const core::Vec3 origin = {}) {
  return core::SparseGridGeometry("map", 1.0, origin,
                                  {.x = 0, .y = 0},
                                  {.x = 2, .y = 2});
}

TEST(ExplorationMapProjectorTest, FreeEvidenceWinsAreaIntersection) {
  std::vector<std::pair<core::GridIndex, core::FineCellState>> cells;
  for (std::int64_t y = 0; y < 2; ++y) {
    for (std::int64_t x = 0; x < 2; ++x) {
      cells.push_back({{.x = x, .y = y}, core::FineCellState::kBlocked});
    }
  }
  cells.push_back({{.x = 2, .y = 0}, core::FineCellState::kBlocked});
  cells.push_back({{.x = 3, .y = 0}, core::FineCellState::kFree});
  const auto fine = MakeFine(cells);

  const auto projection = ExplorationMapProjector{}.Project(
      *fine, CoarseGeometry());

  ASSERT_EQ(projection.data.size(), 4U);
  EXPECT_EQ(projection.data[0], 100);
  EXPECT_EQ(projection.data[1], 0);
  EXPECT_EQ(projection.source_fine_traversability_revision, 7U);
}

TEST(ExplorationMapProjectorTest,
     BlockedRequiresCompleteEvidenceCoverage) {
  const auto fine = MakeFine({
      {{.x = 0, .y = 0}, core::FineCellState::kBlocked},
      {{.x = 1, .y = 0}, core::FineCellState::kBlocked},
      {{.x = 0, .y = 1}, core::FineCellState::kBlocked},
  });

  const auto projection = ExplorationMapProjector{}.Project(
      *fine, CoarseGeometry());

  ASSERT_EQ(projection.data.size(), 4U);
  EXPECT_EQ(projection.data[0], -1);
  EXPECT_EQ(projection.data[1], -1);
}

TEST(ExplorationMapProjectorTest,
     MisalignedOriginUsesWorldAreaIntersection) {
  std::vector<std::pair<core::GridIndex, core::FineCellState>> cells;
  for (std::int64_t y = 0; y < 3; ++y) {
    for (std::int64_t x = 0; x < 3; ++x) {
      cells.push_back({{.x = x, .y = y}, core::FineCellState::kBlocked});
    }
  }
  const auto fine = MakeFine(cells);
  const core::SparseGridGeometry shifted(
      "map", 1.0, {.x = 0.25, .y = 0.25},
      {.x = 0, .y = 0}, {.x = 1, .y = 1});

  const auto projection = ExplorationMapProjector{}.Project(*fine, shifted);

  ASSERT_EQ(projection.data.size(), 1U);
  EXPECT_EQ(projection.data.front(), 100);
  EXPECT_DOUBLE_EQ(projection.geometry.origin_m().x, 0.25);
  EXPECT_DOUBLE_EQ(projection.geometry.origin_m().y, 0.25);
}

}  // namespace
}  // namespace lunar::incremental_navigation_ros
