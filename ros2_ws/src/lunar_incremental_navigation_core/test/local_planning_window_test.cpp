#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/local_planning_window.hpp"

namespace lunar::incremental_navigation {
namespace {

[[nodiscard]] std::shared_ptr<const FineTraversabilitySnapshot> MakeFine(
    const std::size_t width, const std::size_t height,
    const double resolution_m = 0.2) {
  PersistentElevationMap map;
  const GridGeometry geometry{
      .frame_id = "map",
      .width = width,
      .height = height,
      .resolution_m = resolution_m,
      .origin_m = {.x = -100.0, .y = -90.0},
  };
  const std::vector<float> elevation(geometry.CellCount(), 0.0F);
  const ElevationUpdateResult update = map.Apply(ElevationEvidence{
      .geometry = geometry,
      .elevation_m = elevation,
      .map_from_source =
          RigidTransform{.parent_frame = "map", .child_frame = "map"},
  });
  if (update.status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("window fixture elevation was rejected");
  }
  const auto raw = map.Snapshot();
  FineTraversabilityTileDirectory directory(raw->geometry());
  return std::make_shared<const FineTraversabilitySnapshot>(
      raw->geometry(), raw->raw_elevation_revision(), 1U, "window-test",
      0.25, 0.0, TraversalCostWeights{}, raw, std::move(directory),
      std::vector<TileIndex>{}, std::vector<TileIndex>{}, FineSnapshotMetrics{});
}

[[nodiscard]] Vec2 CellCenter(const SparseGridGeometry& geometry,
                              const GridIndex index) {
  const Vec3 origin = geometry.origin_m();
  const double resolution_m = geometry.resolution_m();
  return {.x = origin.x + (static_cast<double>(index.x) + 0.5) * resolution_m,
          .y = origin.y + (static_cast<double>(index.y) + 0.5) * resolution_m};
}

TEST(LocalPlanningWindow, CentersAndClipsALatticeAlignedBoundedWindow) {
  const auto fine = MakeFine(1000U, 900U);
  const Vec2 start = CellCenter(fine->geometry(), {.x = 700, .y = 500});

  const auto window = BuildLocalPlanningWindow(*fine, start);

  ASSERT_TRUE(window);
  EXPECT_TRUE(IsLocalPlanningWindowFor(*window, fine->geometry()));
  EXPECT_EQ(window->width(), kMaximumLocalPlanningWindowAxisCells);
  EXPECT_EQ(window->height(), kMaximumLocalPlanningWindowAxisCells);
  EXPECT_EQ(window->min_inclusive(), (GridIndex{.x = 540, .y = 340}));
  EXPECT_EQ(window->max_exclusive(), (GridIndex{.x = 860, .y = 660}));
  EXPECT_TRUE(window->Contains(GridIndex{.x = 700, .y = 500}));
}

TEST(LocalPlanningWindow, ShiftsAtMapEdgesWithoutDroppingTheStartCell) {
  const auto fine = MakeFine(1000U, 900U);

  const auto lower = BuildLocalPlanningWindow(
      *fine, CellCenter(fine->geometry(), {.x = 2, .y = 3}));
  ASSERT_TRUE(lower);
  EXPECT_EQ(lower->min_inclusive(), (GridIndex{.x = 0, .y = 0}));
  EXPECT_EQ(lower->max_exclusive(), (GridIndex{.x = 320, .y = 320}));

  const auto upper = BuildLocalPlanningWindow(
      *fine, CellCenter(fine->geometry(), {.x = 997, .y = 897}));
  ASSERT_TRUE(upper);
  EXPECT_EQ(upper->min_inclusive(), (GridIndex{.x = 680, .y = 580}));
  EXPECT_EQ(upper->max_exclusive(), (GridIndex{.x = 1000, .y = 900}));
}

TEST(LocalPlanningWindow, PreservesSmallerMapsAndRejectsOutsideStarts) {
  const auto fine = MakeFine(8U, 5U, 1.0);
  const auto full = BuildLocalPlanningWindow(
      *fine, CellCenter(fine->geometry(), {.x = 4, .y = 2}));
  ASSERT_TRUE(full);
  EXPECT_EQ(full->min_inclusive(), fine->geometry().min_inclusive());
  EXPECT_EQ(full->max_exclusive(), fine->geometry().max_exclusive());
  EXPECT_FALSE(BuildLocalPlanningWindow(*fine, {.x = 1000.0, .y = 0.0}));
}

}  // namespace
}  // namespace lunar::incremental_navigation
