#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lunar_pure_exploration_core/candidate_generator.hpp"

namespace lunar::pure_exploration {
namespace {

constexpr double kPi = 3.14159265358979323846;

CandidateParameters StandardYawOffsets() {
  return CandidateParameters{{-kPi / 4.0, -kPi / 8.0, 0.0,
                              kPi / 8.0, kPi / 4.0}};
}

CandidateGenerator::Limits GenerousLimits() {
  return CandidateGenerator::Limits{1000U, 1000U, 1000000U};
}

PlatformGeometry WheelPlatform() {
  return PlatformGeometry{
      .platform_id = "wheel",
      .platform_type = "WHEELED",
      .base_frame_id = "base_footprint",
      .footprint_vertices = {{0.591, 0.409}, {0.591, -0.409},
                             {-0.591, -0.409}, {-0.591, 0.409}},
      .minimum_clearance_m = 0.2,
  };
}

PlatformGeometry CompactWheelPlatform() {
  return PlatformGeometry{
      .platform_id = "compact",
      .platform_type = "WHEELED",
      .base_frame_id = "base_footprint",
      .footprint_vertices = {{0.1, 0.1}, {-0.1, 0.1},
                             {-0.1, -0.1}, {0.1, -0.1}},
      .minimum_clearance_m = 0.0,
  };
}

PlatformGeometry OneMeterWheelPlatform() {
  return PlatformGeometry{
      .platform_id = "one_meter",
      .platform_type = "WHEELED",
      .base_frame_id = "base_footprint",
      .footprint_vertices = {{0.5, 0.1}, {0.5, -0.1},
                             {-0.5, -0.1}, {-0.5, 0.1}},
      .minimum_clearance_m = 0.0,
  };
}

PlatformGeometry RotationallySymmetricWheelPlatform(double phase = 0.0) {
  PlatformGeometry platform{
      .platform_id = "round",
      .platform_type = "WHEELED",
      .base_frame_id = "base_footprint",
      .footprint_vertices = {},
      .minimum_clearance_m = 0.0,
  };
  constexpr std::size_t kVertexCount = 16U;
  platform.footprint_vertices.reserve(kVertexCount);
  for (std::size_t index = 0U; index < kVertexCount; ++index) {
    const double angle = phase +
        2.0 * kPi * static_cast<double>(index) / kVertexCount;
    platform.footprint_vertices.push_back(
        Vec2{0.1 * std::cos(angle), 0.1 * std::sin(angle)});
  }
  return platform;
}

PlatformGeometry LargeWheelPlatform() {
  return PlatformGeometry{
      .platform_id = "large",
      .platform_type = "WHEELED",
      .base_frame_id = "base_footprint",
      .footprint_vertices = {{5.0, 2.0}, {5.0, -2.0},
                             {-5.0, -2.0}, {-5.0, 2.0}},
      .minimum_clearance_m = 0.0,
  };
}

enum class TangencyState {
  kFree,
  kOccupied,
  kUnknown,
  kOutsideMap,
  kOutsideTask,
};

enum class TangencyContact {
  kEdge,
  kCorner,
  kRoundedClearance,
};

struct TangencyFixture {
  TaskRaster raster;
  PlatformGeometry platform;
};

TangencyFixture MakeTangencyFixture(double map_yaw, double relative_yaw,
                                    TangencyContact contact,
                                    TangencyState state) {
  constexpr double kSquareRadius = 0.70710678118654752440;
  double final_phase = 0.0;
  double clearance = 1.0;
  GridIndex blocked{6, 7};
  if (contact == TangencyContact::kCorner) {
    blocked = GridIndex{7, 7};
    const double offset = kSquareRadius - 1.5;
    clearance = -(offset * offset + 1.0) / (2.0 * offset);
  } else if (contact == TangencyContact::kRoundedClearance) {
    final_phase = kPi / 8.0;
    const double top_extent =
        0.5 * (std::cos(final_phase) + std::sin(final_phase));
    clearance = 1.5 - top_extent;
  }

  const double base_phase = final_phase - relative_yaw;
  const double cosine = std::cos(base_phase);
  const double sine = std::sin(base_phase);
  std::vector<Vec2> footprint;
  for (const Vec2 vertex : {
           Vec2{0.5, 0.5}, Vec2{0.5, -0.5},
           Vec2{-0.5, -0.5}, Vec2{-0.5, 0.5}}) {
    footprint.push_back(
        Vec2{cosine * vertex.x - sine * vertex.y,
             sine * vertex.x + cosine * vertex.y});
  }

  const std::uint32_t height =
      state == TangencyState::kOutsideMap ? 7U : 12U;
  const GridGeometry geometry{12U, height, 1.0, 100.0, -50.0, map_yaw};
  std::vector<std::int8_t> data(12U * height, 0);
  data[5U * 12U + 8U] = -1;
  if (state == TangencyState::kOccupied ||
      state == TangencyState::kUnknown) {
    data[static_cast<std::size_t>(blocked.y) * 12U +
         static_cast<std::size_t>(blocked.x)] =
        state == TangencyState::kOccupied ? 100 : -1;
  }
  const OccupancyGridView map(geometry, data, 50);
  const auto world = [&geometry](double x, double y) {
    return *OccupancyGridView::GridToWorld(geometry, Vec2{x, y});
  };
  const double task_height =
      state == TangencyState::kOutsideTask ? 7.0 : 12.0;
  TaskRaster raster = TaskRaster::Build(
      map, Polygon2{{world(0.0, 0.0), world(12.0, 0.0),
                     world(12.0, task_height), world(0.0, task_height)}});
  return TangencyFixture{
      .raster = std::move(raster),
      .platform = PlatformGeometry{
          .platform_id = "tangency",
          .platform_type = "WHEELED",
          .base_frame_id = "base_footprint",
          .footprint_vertices = std::move(footprint),
          .minimum_clearance_m = clearance,
      },
  };
}

CandidateParameters SymmetricYawOffsets(double relative_yaw) {
  return CandidateParameters{{
      relative_yaw - 2.0 * kPi,
      relative_yaw - 1.5 * kPi,
      relative_yaw - kPi,
      relative_yaw - 0.5 * kPi,
      relative_yaw,
  }};
}

TaskRaster Raster(std::uint32_t width, std::uint32_t height,
                  std::vector<std::int8_t> data, double resolution = 0.2,
                  double origin_x = 0.0, double origin_y = 0.0,
                  double origin_yaw = 0.0) {
  const GridGeometry geometry{width, height, resolution, origin_x, origin_y,
                              origin_yaw};
  const OccupancyGridView map(geometry, data, 50);
  const auto corner = [&geometry](double x, double y) {
    return *OccupancyGridView::GridToWorld(geometry, Vec2{x, y});
  };
  return TaskRaster::Build(
      map, Polygon2{{corner(0.0, 0.0), corner(width, 0.0),
                     corner(width, height), corner(0.0, height)}});
}

FrontierCluster OneEdgeCluster(const TaskRaster& raster, GridIndex free_cell,
                               GridIndex unknown_cell, std::uint8_t direction,
                               std::int64_t key_seed = 1) {
  const Vec2 free_center = raster.CellCenter(free_cell);
  const Vec2 unknown_center = raster.CellCenter(unknown_cell);
  const Vec2 midpoint{(free_center.x + unknown_center.x) * 0.5,
                      (free_center.y + unknown_center.y) * 0.5};
  return FrontierCluster{
      .id = static_cast<std::uint64_t>(key_seed),
      .cells = {free_cell},
      .interface_edges = {{free_cell, unknown_cell, direction, midpoint}},
      .canonical_key = {key_seed, 0, direction},
      .centroid = midpoint,
      .length_m = raster.geometry().resolution,
  };
}

std::size_t PositionGroupCount(std::span<const CandidateView> views) {
  std::vector<std::pair<std::int64_t, std::int64_t>> positions;
  for (const CandidateView& view : views) {
    const auto position = std::pair{view.key.x_mm, view.key.y_mm};
    if (std::find(positions.begin(), positions.end(), position) ==
        positions.end()) {
      positions.push_back(position);
    }
  }
  return positions.size();
}

TEST(CandidateGeneratorTest, DerivesWheelDimensionsAndSearchBoundsAtPointTwoMeters) {
  const CandidateGenerator generator(WheelPlatform(), StandardYawOffsets(),
                                     GenerousLimits());
  EXPECT_DOUBLE_EQ(generator.platform_length_m(), 1.182);
  EXPECT_DOUBLE_EQ(generator.platform_width_m(), 0.818);
  EXPECT_NEAR(generator.footprint_circumscribed_radius_m(), 0.71872, 1.0e-5);
  EXPECT_DOUBLE_EQ(generator.minimum_spacing_m(0.2), 1.182);
  EXPECT_NEAR(generator.minimum_standoff_m(), 0.91872, 1.0e-5);
  EXPECT_DOUBLE_EQ(generator.maximum_extra_search_m(), 2.364);
  EXPECT_EQ(generator.maximum_search_step(0.2), 11U);

  const CandidateGenerator exact_division(
      OneMeterWheelPlatform(), StandardYawOffsets(), GenerousLimits());
  EXPECT_EQ(exact_division.maximum_search_step(0.2), 10U);

  const double size_boundary_resolution =
      exact_division.maximum_extra_search_m() /
      static_cast<double>(std::numeric_limits<std::size_t>::max());
  EXPECT_THROW(exact_division.maximum_search_step(size_boundary_resolution),
               std::overflow_error);
}

TEST(CandidateGeneratorTest, RejectsFinitePlatformInputsWhoseDerivedGeometryOverflows) {
  PlatformGeometry huge = WheelPlatform();
  huge.footprint_vertices = {
      {1.0e308, 1.0}, {1.0e308, -1.0},
      {-1.0e308, -1.0}, {-1.0e308, 1.0},
  };
  EXPECT_THROW(CandidateGenerator(huge, StandardYawOffsets(), GenerousLimits()),
               std::overflow_error);
}

TEST(CandidateGeneratorTest, ChargesQuadraticPlatformValidationBeforePairLoops) {
  auto exact = GenerousLimits();
  exact.maximum_collision_work_units = 12U;
  EXPECT_NO_THROW((void)CandidateGenerator(
      CompactWheelPlatform(), StandardYawOffsets(), exact));
  exact.maximum_collision_work_units = 11U;
  EXPECT_THROW(CandidateGenerator(CompactWheelPlatform(), StandardYawOffsets(), exact),
               std::length_error);

  PlatformGeometry many = CompactWheelPlatform();
  many.footprint_vertices.clear();
  constexpr std::size_t kVertexCount = 10000U;
  many.footprint_vertices.reserve(kVertexCount);
  for (std::size_t index = 0U; index < kVertexCount; ++index) {
    const double angle =
        2.0 * kPi * static_cast<double>(index) / kVertexCount;
    many.footprint_vertices.push_back(Vec2{std::cos(angle), std::sin(angle)});
  }
  auto tiny = GenerousLimits();
  tiny.maximum_collision_work_units = 1U;
  EXPECT_THROW(CandidateGenerator(std::move(many), StandardYawOffsets(), tiny),
               std::length_error);
}

TEST(CandidateGeneratorTest, RejectsInvalidPlatformPayloadYawOffsetsAndLimits) {
  auto platform = WheelPlatform();
  platform.platform_type = "LEGGED";
  EXPECT_THROW(CandidateGenerator(platform, StandardYawOffsets(), GenerousLimits()),
               std::invalid_argument);
  platform = WheelPlatform();
  platform.platform_id.clear();
  EXPECT_THROW(CandidateGenerator(platform, StandardYawOffsets(), GenerousLimits()),
               std::invalid_argument);
  platform = WheelPlatform();
  platform.footprint_vertices = {{0.0, 0.0}, {1.0, 0.0},
                                 {0.3, 0.3}, {1.0, 1.0}, {0.0, 1.0}};
  EXPECT_THROW(CandidateGenerator(platform, StandardYawOffsets(), GenerousLimits()),
               std::invalid_argument);
  platform = WheelPlatform();
  platform.footprint_vertices[1] = platform.footprint_vertices[0];
  EXPECT_THROW(CandidateGenerator(platform, StandardYawOffsets(), GenerousLimits()),
               std::invalid_argument);
  platform = WheelPlatform();
  platform.minimum_clearance_m = -0.1;
  EXPECT_THROW(CandidateGenerator(platform, StandardYawOffsets(), GenerousLimits()),
               std::invalid_argument);

  auto offsets = StandardYawOffsets();
  offsets.yaw_offsets_rad[2] = offsets.yaw_offsets_rad[1];
  EXPECT_THROW(CandidateGenerator(WheelPlatform(), offsets, GenerousLimits()),
               std::invalid_argument);
  offsets = StandardYawOffsets();
  offsets.yaw_offsets_rad[4] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(CandidateGenerator(WheelPlatform(), offsets, GenerousLimits()),
               std::invalid_argument);

  for (const auto limits : {
           CandidateGenerator::Limits{0U, 1U, 1U},
           CandidateGenerator::Limits{1U, 0U, 1U},
           CandidateGenerator::Limits{1U, 1U, 0U}}) {
    EXPECT_THROW(CandidateGenerator(WheelPlatform(), StandardYawOffsets(), limits),
                 std::invalid_argument);
  }
}

TEST(CandidateGeneratorTest, KeepsEverySafeYawAtOnePositionAndUsesLocalUnknownNormal) {
  std::vector<std::int8_t> data(30U * 20U, 0);
  data[10U * 30U + 20U] = -1;
  const TaskRaster raster = Raster(30U, 20U, data);
  const FrontierCluster frontier =
      OneEdgeCluster(raster, GridIndex{19, 10}, GridIndex{20, 10}, 0U);

  const auto views = CandidateGenerator(WheelPlatform(), StandardYawOffsets(),
                                        GenerousLimits())
                         .Generate(raster, std::span<const FrontierCluster>(&frontier, 1U));

  ASSERT_EQ(views.size(), 5U);
  const double radius = std::hypot(0.591, 0.409);
  const double expected_x = 4.0 - radius - 0.2;
  for (std::size_t index = 0U; index < views.size(); ++index) {
    EXPECT_NEAR(views[index].pose.x, expected_x, 1.0e-12);
    EXPECT_DOUBLE_EQ(views[index].pose.y, 2.1);
    EXPECT_NEAR(views[index].pose.yaw,
                StandardYawOffsets().yaw_offsets_rad[index], 1.0e-12);
    EXPECT_EQ(views[index].frontier_index, 0U);
    EXPECT_EQ(views[index].frontier_id, frontier.id);
    EXPECT_NEAR(views[index].frontier_distance_m, radius + 0.2, 1.0e-12);
  }

  const TaskRaster rebuilt_raster = Raster(30U, 20U, data);
  const FrontierCluster rebuilt_frontier = OneEdgeCluster(
      rebuilt_raster, GridIndex{19, 10}, GridIndex{20, 10}, 0U);
  const auto rebuilt =
      CandidateGenerator(WheelPlatform(), StandardYawOffsets(), GenerousLimits())
          .Generate(rebuilt_raster,
                    std::span<const FrontierCluster>(&rebuilt_frontier, 1U));
  ASSERT_EQ(rebuilt.size(), views.size());
  for (std::size_t index = 0U; index < views.size(); ++index) {
    EXPECT_EQ(rebuilt[index].id, views[index].id);
    EXPECT_EQ(rebuilt[index].key, views[index].key);
    EXPECT_DOUBLE_EQ(rebuilt[index].pose.x, views[index].pose.x);
    EXPECT_DOUBLE_EQ(rebuilt[index].pose.y, views[index].pose.y);
    EXPECT_DOUBLE_EQ(rebuilt[index].pose.yaw, views[index].pose.yaw);
  }
}

TEST(CandidateGeneratorTest,
     AdvancesPastGoalCellRejectedByGlobalCircumscribedInflation) {
  std::vector<std::int8_t> data(30U * 20U, 0);
  data[10U * 30U + 20U] = -1;
  const TaskRaster raster = Raster(30U, 20U, data);
  const FrontierCluster frontier =
      OneEdgeCluster(raster, GridIndex{19, 10}, GridIndex{20, 10}, 0U);

  const CandidateGenerator generator(WheelPlatform(), StandardYawOffsets(),
                                     GenerousLimits());
  const auto views = generator.Generate(
      raster, std::span<const FrontierCluster>(&frontier, 1U));

  ASSERT_EQ(views.size(), 5U);
  std::size_t collision_work = 0U;
  // The first continuous pose is body-safe, but its 0.2 m global goal cell
  // still intersects the global planner's circumscribed inflated unknown mask.
  EXPECT_FALSE(generator.GlobalGoalCellFeasible(
      raster, views.front().pose, collision_work));
  const Pose2 advanced{views.front().pose.x - 0.2, views.front().pose.y,
                       views.front().pose.yaw};
  EXPECT_TRUE(generator.GlobalGoalCellFeasible(raster, advanced,
                                               collision_work));

  const auto filtered = generator.Generate(
      raster, std::span<const FrontierCluster>(&frontier, 1U), true);
  ASSERT_EQ(filtered.size(), 5U);
  EXPECT_DOUBLE_EQ(filtered.front().pose.x, advanced.x);
  EXPECT_TRUE(generator.GlobalGoalCellFeasible(raster, filtered.front().pose,
                                               collision_work));
}

TEST(CandidateGeneratorTest, CanonicalOrderingPreservesOriginalFrontierIndex) {
  std::vector<std::int8_t> data(50U * 20U, 0);
  data[10U * 50U + 35U] = -1;
  data[10U * 50U + 15U] = -1;
  const TaskRaster raster = Raster(50U, 20U, data);
  FrontierCluster high =
      OneEdgeCluster(raster, GridIndex{34, 10}, GridIndex{35, 10}, 0U, 20);
  FrontierCluster low =
      OneEdgeCluster(raster, GridIndex{14, 10}, GridIndex{15, 10}, 0U, 10);
  const std::array<FrontierCluster, 2> reversed{high, low};

  const auto views = CandidateGenerator(WheelPlatform(), StandardYawOffsets(),
                                        GenerousLimits())
                         .Generate(raster, reversed);
  ASSERT_EQ(views.size(), 10U);
  for (std::size_t index = 0U; index < 5U; ++index) {
    EXPECT_EQ(views[index].frontier_index, 1U);
    EXPECT_EQ(views[index].frontier_id, low.id);
    ASSERT_NE(views[index].frontier_canonical_key, nullptr);
    EXPECT_EQ(*views[index].frontier_canonical_key, low.canonical_key);
    EXPECT_EQ(views[index].frontier_canonical_key,
              views.front().frontier_canonical_key);
  }
  for (std::size_t index = 5U; index < 10U; ++index) {
    EXPECT_EQ(views[index].frontier_index, 0U);
    EXPECT_EQ(views[index].frontier_id, high.id);
    ASSERT_NE(views[index].frontier_canonical_key, nullptr);
    EXPECT_EQ(*views[index].frontier_canonical_key, high.canonical_key);
    EXPECT_EQ(views[index].frontier_canonical_key,
              views[5U].frontier_canonical_key);
  }
  EXPECT_NE(views.front().frontier_canonical_key,
            views[5U].frontier_canonical_key);
}

TEST(CandidateGeneratorTest, RotatedMapAndLargeTranslationUseSharedRasterGeometry) {
  std::vector<std::int8_t> data(30U * 20U, 0);
  data[10U * 30U + 20U] = -1;
  const TaskRaster raster =
      Raster(30U, 20U, data, 0.2, 1.0e9, -1.0e9, kPi / 6.0);
  const FrontierCluster frontier =
      OneEdgeCluster(raster, GridIndex{19, 10}, GridIndex{20, 10}, 0U);

  const auto views = CandidateGenerator(WheelPlatform(), StandardYawOffsets(),
                                        GenerousLimits())
                         .Generate(raster,
                                   std::span<const FrontierCluster>(&frontier, 1U));
  ASSERT_FALSE(views.empty());
  const auto candidate_grid = raster.WorldToGrid(
      Vec2{views.front().pose.x, views.front().pose.y});
  ASSERT_TRUE(candidate_grid.has_value());
  EXPECT_NEAR(candidate_grid->x,
              20.0 - (std::hypot(0.591, 0.409) + 0.2) / 0.2,
              2.0e-6);
  EXPECT_NEAR(candidate_grid->y, 10.5, 2.0e-6);
  EXPECT_TRUE(std::any_of(views.begin(), views.end(), [](const CandidateView& view) {
    return std::abs(view.pose.yaw - kPi / 6.0) < 2.0e-6;
  }));
}

TEST(CandidateGeneratorTest, SymmetricUnknownRingAndBranchFallbackIsTranslationInvariant) {
  std::vector<std::int8_t> data(50U * 50U, 0);
  for (const GridIndex unknown : {
           GridIndex{21, 20}, GridIndex{20, 21},
           GridIndex{19, 20}, GridIndex{20, 19}}) {
    data[static_cast<std::size_t>(unknown.y) * 50U +
         static_cast<std::size_t>(unknown.x)] = -1;
  }
  const TaskRaster local = Raster(50U, 50U, data, 0.1, 0.0, 0.0, kPi / 6.0);
  const TaskRaster translated =
      Raster(50U, 50U, data, 0.1, 1.0e9, -1.0e9, kPi / 6.0);
  const FrontierCluster local_frontier =
      OneEdgeCluster(local, GridIndex{20, 20}, GridIndex{21, 20}, 0U);
  const FrontierCluster translated_frontier =
      OneEdgeCluster(translated, GridIndex{20, 20}, GridIndex{21, 20}, 0U);
  const CandidateGenerator generator(RotationallySymmetricWheelPlatform(),
                                     StandardYawOffsets(),
                                     GenerousLimits());
  const auto local_views = generator.Generate(
      local, std::span<const FrontierCluster>(&local_frontier, 1U));
  const auto translated_views = generator.Generate(
      translated,
      std::span<const FrontierCluster>(&translated_frontier, 1U));

  ASSERT_EQ(local_views.size(), 5U);
  ASSERT_EQ(translated_views.size(), local_views.size());
  for (std::size_t index = 0U; index < local_views.size(); ++index) {
    const auto local_grid = local.WorldToGrid(
        Vec2{local_views[index].pose.x, local_views[index].pose.y});
    const auto translated_grid = translated.WorldToGrid(
        Vec2{translated_views[index].pose.x, translated_views[index].pose.y});
    ASSERT_TRUE(local_grid.has_value());
    ASSERT_TRUE(translated_grid.has_value());
    EXPECT_NEAR(translated_grid->x, local_grid->x, 1.0e-6);
    EXPECT_NEAR(translated_grid->y, local_grid->y, 1.0e-6);
    EXPECT_NEAR(translated_views[index].pose.yaw,
                local_views[index].pose.yaw, 1.0e-9);
  }
}

TEST(CandidateGeneratorTest, LShapedCanonicalInterfaceSequenceIsDeterministic) {
  std::vector<std::int8_t> data(50U * 50U, 0);
  data[20U * 50U + 20U] = -1;
  data[21U * 50U + 20U] = -1;
  const TaskRaster first = Raster(50U, 50U, data, 0.1);
  const TaskRaster rebuilt = Raster(50U, 50U, data, 0.1);
  auto make_frontier = [](const TaskRaster& raster) {
    FrontierCluster cluster = OneEdgeCluster(
        raster, GridIndex{19, 20}, GridIndex{20, 20}, 0U, 1);
    cluster.interface_edges.push_back(
        OneEdgeCluster(raster, GridIndex{19, 21}, GridIndex{20, 21}, 0U, 2)
            .interface_edges.front());
    cluster.interface_edges.push_back(
        OneEdgeCluster(raster, GridIndex{20, 22}, GridIndex{20, 21}, 3U, 3)
            .interface_edges.front());
    cluster.canonical_key = {10, 20, 0, 10, 21, 0, 20, 21, 3};
    cluster.length_m = 0.3;
    return cluster;
  };
  const FrontierCluster first_frontier = make_frontier(first);
  const FrontierCluster rebuilt_frontier = make_frontier(rebuilt);
  const CandidateGenerator generator(RotationallySymmetricWheelPlatform(),
                                     StandardYawOffsets(), GenerousLimits());
  const auto first_views = generator.Generate(
      first, std::span<const FrontierCluster>(&first_frontier, 1U));
  const auto rebuilt_views = generator.Generate(
      rebuilt, std::span<const FrontierCluster>(&rebuilt_frontier, 1U));
  ASSERT_FALSE(first_views.empty());
  ASSERT_EQ(rebuilt_views.size(), first_views.size());
  for (std::size_t index = 0U; index < first_views.size(); ++index) {
    EXPECT_EQ(rebuilt_views[index].key, first_views[index].key);
    EXPECT_EQ(rebuilt_views[index].id, first_views[index].id);
  }
}

TEST(CandidateGeneratorTest, RealRingAndBranchSequencesUseCanonicalQuartiles) {
  auto verify = [](const std::vector<std::int8_t>& data) {
    const TaskRaster first = Raster(30U, 30U, data, 0.2);
    const TaskRaster rebuilt = Raster(30U, 30U, data, 0.2);
    const FrontierDetector detector({0.0});
    const FrontierDetection first_detection =
        detector.Detect(first, GridIndex{1, 1});
    const FrontierDetection rebuilt_detection =
        detector.Detect(rebuilt, GridIndex{1, 1});
    ASSERT_EQ(first_detection.clusters.size(), 1U);
    ASSERT_EQ(rebuilt_detection.clusters.size(), 1U);
    const FrontierCluster& frontier = first_detection.clusters.front();
    const FrontierCluster& rebuilt_frontier =
        rebuilt_detection.clusters.front();
    ASSERT_GE(frontier.interface_edges.size(), 8U);
    ASSERT_EQ(rebuilt_frontier.canonical_key, frontier.canonical_key);

    const CandidateGenerator generator(RotationallySymmetricWheelPlatform(),
                                       StandardYawOffsets(),
                                       GenerousLimits());
    const auto views = generator.Generate(
        first, std::span<const FrontierCluster>(&frontier, 1U));
    const auto rebuilt_views = generator.Generate(
        rebuilt,
        std::span<const FrontierCluster>(&rebuilt_frontier, 1U));
    ASSERT_EQ(PositionGroupCount(views), 3U);
    ASSERT_EQ(rebuilt_views.size(), views.size());

    const std::size_t count = frontier.interface_edges.size();
    const std::array<std::size_t, 3> selected{
        (count - 1U) / 4U,
        (count - 1U) / 2U,
        (count - 1U) - count / 4U,
    };
    std::vector<const CandidateView*> groups;
    for (const CandidateView& view : views) {
      if (groups.empty() ||
          groups.back()->key.x_mm != view.key.x_mm ||
          groups.back()->key.y_mm != view.key.y_mm) {
        groups.push_back(&view);
      }
    }
    ASSERT_EQ(groups.size(), selected.size());
    for (std::size_t index = 0U; index < groups.size(); ++index) {
      const CandidateView& view = *groups[index];
      const Vec2 midpoint = frontier.interface_edges[selected[index]].midpoint;
      EXPECT_NEAR(std::hypot(view.pose.x - midpoint.x,
                             view.pose.y - midpoint.y),
                  view.frontier_distance_m, 1.0e-12);
    }
    for (std::size_t index = 0U; index < views.size(); ++index) {
      EXPECT_EQ(rebuilt_views[index].key, views[index].key);
      EXPECT_EQ(rebuilt_views[index].id, views[index].id);
    }
  };

  std::vector<std::int8_t> ring(30U * 30U, 0);
  for (std::size_t y = 10U; y <= 19U; ++y) {
    for (std::size_t x = 10U; x <= 19U; ++x) {
      ring[y * 30U + x] = -1;
    }
  }
  verify(ring);

  std::vector<std::int8_t> branch(30U * 30U, 0);
  for (std::size_t y = 7U; y <= 22U; ++y) {
    branch[y * 30U + 15U] = -1;
  }
  for (std::size_t x = 7U; x <= 22U; ++x) {
    branch[15U * 30U + x] = -1;
  }
  verify(branch);
}

TEST(CandidateGeneratorTest, DeduplicatesQuartileEdgesAndRejectsDuplicateFullKeys) {
  std::vector<std::int8_t> data(60U * 20U, 0);
  for (const std::size_t x : {10U, 20U, 30U, 40U}) {
    data[10U * 60U + x] = -1;
  }
  const TaskRaster raster = Raster(60U, 20U, data);
  FrontierCluster cluster =
      OneEdgeCluster(raster, GridIndex{9, 10}, GridIndex{10, 10}, 0U, 1);
  for (const std::int32_t x : {19, 29, 39}) {
    const auto next = OneEdgeCluster(raster, GridIndex{x, 10},
                                     GridIndex{x + 1, 10}, 0U, x);
    cluster.interface_edges.push_back(next.interface_edges.front());
  }
  cluster.canonical_key = {1, 2, 3};

  const auto views = CandidateGenerator(CompactWheelPlatform(), StandardYawOffsets(),
                                        GenerousLimits())
                         .Generate(raster, std::span<const FrontierCluster>(&cluster, 1U));
  ASSERT_FALSE(views.empty());
  std::vector<std::int64_t> position_groups;
  for (const CandidateView& view : views) {
    if (position_groups.empty() || position_groups.back() != view.key.x_mm) {
      position_groups.push_back(view.key.x_mm);
    }
  }
  ASSERT_EQ(position_groups.size(), 3U);
  EXPECT_LT(position_groups[0], position_groups[1]);
  EXPECT_LT(position_groups[1], position_groups[2]);

  const std::array<FrontierCluster, 2> duplicates{cluster, cluster};
  EXPECT_THROW(CandidateGenerator(CompactWheelPlatform(), StandardYawOffsets(),
                                  GenerousLimits())
                   .Generate(raster, duplicates),
               std::invalid_argument);
}

TEST(CandidateGeneratorTest, QuartileSelectionCoversCanonicalEdgeCountsOneThroughFour) {
  std::vector<std::int8_t> data(100U * 30U, 0);
  for (const std::size_t x : {20U, 40U, 60U, 80U}) {
    data[15U * 100U + x] = -1;
  }
  const TaskRaster raster = Raster(100U, 30U, data, 0.1);
  std::vector<FrontierCluster::InterfaceEdge> edges;
  for (const std::int32_t x : {19, 39, 59, 79}) {
    edges.push_back(OneEdgeCluster(raster, GridIndex{x, 15},
                                   GridIndex{x + 1, 15}, 0U, x)
                        .interface_edges.front());
  }
  const CandidateGenerator generator(RotationallySymmetricWheelPlatform(),
                                     StandardYawOffsets(), GenerousLimits());
  for (std::size_t edge_count = 1U; edge_count <= 4U; ++edge_count) {
    FrontierCluster cluster{
        .id = edge_count,
        .cells = {},
        .interface_edges = {edges.begin(), edges.begin() + edge_count},
        .canonical_key = {static_cast<std::int64_t>(edge_count)},
        .centroid = edges.front().midpoint,
        .length_m = static_cast<double>(edge_count) * 0.1,
    };
    const auto views = generator.Generate(
        raster, std::span<const FrontierCluster>(&cluster, 1U));
    EXPECT_EQ(PositionGroupCount(views),
              std::min(edge_count, std::size_t{3U}));
  }
}

TEST(CandidateGeneratorTest, PartialYawGroupIsKeptAndAllYawFailureContinuesSearching) {
  std::vector<std::int8_t> data(30U * 20U, 0);
  data[10U * 30U + 20U] = -1;
  const TaskRaster raster = Raster(30U, 20U, data);
  const FrontierCluster frontier =
      OneEdgeCluster(raster, GridIndex{19, 10}, GridIndex{20, 10}, 0U);
  const auto partial =
      CandidateGenerator(CompactWheelPlatform(), StandardYawOffsets(),
                         GenerousLimits())
          .Generate(raster,
                    std::span<const FrontierCluster>(&frontier, 1U));
  ASSERT_GT(partial.size(), 0U);
  ASSERT_LT(partial.size(), 5U);
  EXPECT_EQ(PositionGroupCount(partial), 1U);

  for (std::size_t x = 10U; x <= 18U; ++x) {
    for (std::size_t y = 0U; y < 20U; ++y) {
      data[y * 30U + x] = 100;
    }
  }
  data[10U * 30U + 19U] = 0;
  const TaskRaster blocked = Raster(30U, 20U, data);
  const auto terminal =
      CandidateGenerator(OneMeterWheelPlatform(), StandardYawOffsets(),
                         GenerousLimits())
          .Generate(blocked,
                    std::span<const FrontierCluster>(&frontier, 1U));
  ASSERT_EQ(terminal.size(), 5U);
  EXPECT_NE(terminal.front().key.x_mm, partial.front().key.x_mm);
}

TEST(CandidateGeneratorTest, PositionSpacingHonorsStrictLessEqualityAndBucketBoundaries) {
  auto run = [](double resolution, double origin_y) {
    std::vector<std::int8_t> data(300U * 130U, 0);
    data[10U * 300U + 201U] = -1;
    data[110U * 300U + 201U] = -1;
    const TaskRaster raster =
        Raster(300U, 130U, data, resolution, 0.0, origin_y);
    const std::array<FrontierCluster, 2> frontiers{
        OneEdgeCluster(raster, GridIndex{200, 10}, GridIndex{201, 10}, 0U, 1),
        OneEdgeCluster(raster, GridIndex{200, 110}, GridIndex{201, 110}, 0U, 2),
    };
    auto limits = GenerousLimits();
    limits.maximum_position_probes = 2U;
    return CandidateGenerator(OneMeterWheelPlatform(), StandardYawOffsets(),
                              limits)
        .Generate(raster, frontiers);
  };

  EXPECT_THROW(run(std::nextafter(0.01, 0.0), -0.105), std::length_error);
  EXPECT_EQ(PositionGroupCount(run(0.01, -0.105)), 2U);
  EXPECT_EQ(PositionGroupCount(
                run(std::nextafter(0.01, 1.0), 0.895)),
            2U);
}

TEST(CandidateGeneratorTest, RejectsAnyClosedClearanceContactWithNonfreeCell) {
  std::vector<std::int8_t> free_data(30U * 20U, 0);
  free_data[10U * 30U + 20U] = -1;
  const TaskRaster free_raster = Raster(30U, 20U, free_data);
  const FrontierCluster frontier = OneEdgeCluster(
      free_raster, GridIndex{19, 10}, GridIndex{20, 10}, 0U);
  ASSERT_EQ(CandidateGenerator(WheelPlatform(), StandardYawOffsets(),
                               GenerousLimits())
                .Generate(free_raster,
                          std::span<const FrontierCluster>(&frontier, 1U))
                .size(),
            5U);

  // This occupied cell intersects the baseline candidate's closed inflated
  // footprint. The generator may search farther, but must never emit the
  // obstructed first position group.
  auto blocked_data = free_data;
  blocked_data[10U * 30U + 18U] = 100;
  const TaskRaster blocked_raster = Raster(30U, 20U, blocked_data);
  const auto blocked_views =
      CandidateGenerator(WheelPlatform(), StandardYawOffsets(), GenerousLimits())
          .Generate(blocked_raster,
                    std::span<const FrontierCluster>(&frontier, 1U));
  for (const CandidateView& view : blocked_views) {
    EXPECT_LT(view.pose.x, 3.0);
  }
}

TEST(CandidateGeneratorTest, RejectsClosedBodyOutsideMapAcrossRotatedOrigins) {
  for (const double map_yaw : {kPi / 6.0, kPi / 2.0}) {
    std::vector<std::int8_t> data(20U * 20U, 0);
    data[10U * 20U + 6U] = -1;
    const TaskRaster raster =
        Raster(20U, 20U, data, 0.2, 100.0, -50.0, map_yaw);
    const FrontierCluster frontier =
        OneEdgeCluster(raster, GridIndex{5, 10}, GridIndex{6, 10}, 0U);
    EXPECT_TRUE(CandidateGenerator(WheelPlatform(), StandardYawOffsets(),
                                   GenerousLimits())
                    .Generate(raster,
                              std::span<const FrontierCluster>(&frontier, 1U))
                    .empty());
  }
}

TEST(CandidateGeneratorTest, RejectsClosedBodyOutsideTaskWithFreeMapBackedCenter) {
  for (const double map_yaw : {kPi / 6.0, kPi / 2.0}) {
    const GridGeometry geometry{20U, 20U, 0.2, 10.0, -4.0, map_yaw};
    std::vector<std::int8_t> data(20U * 20U, 0);
    data[10U * 20U + 8U] = -1;
    const OccupancyGridView map(geometry, data, 50);
    const auto world = [&geometry](double x, double y) {
      return *OccupancyGridView::GridToWorld(geometry, Vec2{x, y});
    };
    const TaskRaster raster = TaskRaster::Build(
        map, Polygon2{{world(2.0, 0.0), world(20.0, 0.0),
                       world(20.0, 20.0), world(2.0, 20.0)}});
    const FrontierCluster frontier =
        OneEdgeCluster(raster, GridIndex{7, 10}, GridIndex{8, 10}, 0U);
    ASSERT_EQ(raster.Classify(GridIndex{3, 10}), CellState::kFree);
    EXPECT_TRUE(CandidateGenerator(WheelPlatform(), StandardYawOffsets(),
                                   GenerousLimits())
                    .Generate(raster,
                              std::span<const FrontierCluster>(&frontier, 1U))
                    .empty());
  }
}

TEST(CandidateGeneratorTest, ClosedTangencyMatrixRejectsEveryNonfreeState) {
  for (const double relative_yaw : {0.0, kPi / 4.0, kPi / 2.0}) {
    for (const double map_yaw : {kPi / 6.0, kPi / 2.0}) {
      for (const TangencyContact contact : {
               TangencyContact::kEdge,
               TangencyContact::kCorner,
               TangencyContact::kRoundedClearance}) {
        if (contact == TangencyContact::kRoundedClearance) {
          TangencyFixture free_fixture = MakeTangencyFixture(
              map_yaw, relative_yaw, contact, TangencyState::kFree);
          const FrontierCluster free_frontier = OneEdgeCluster(
              free_fixture.raster, GridIndex{7, 5}, GridIndex{8, 5}, 0U);
          const auto free_views =
              CandidateGenerator(
                  std::move(free_fixture.platform),
                  SymmetricYawOffsets(relative_yaw),
                  CandidateGenerator::Limits{1U, 5U, 100000U})
                  .Generate(
                      free_fixture.raster,
                      std::span<const FrontierCluster>(&free_frontier, 1U));
          ASSERT_EQ(free_views.size(), 5U);
        }
        for (const TangencyState state : {
                 TangencyState::kOccupied,
                 TangencyState::kUnknown,
                 TangencyState::kOutsideMap,
                 TangencyState::kOutsideTask}) {
          SCOPED_TRACE(relative_yaw);
          SCOPED_TRACE(map_yaw);
          SCOPED_TRACE(static_cast<int>(contact));
          SCOPED_TRACE(static_cast<int>(state));
          TangencyFixture fixture =
              MakeTangencyFixture(map_yaw, relative_yaw, contact, state);
          const FrontierCluster frontier = OneEdgeCluster(
              fixture.raster, GridIndex{7, 5}, GridIndex{8, 5}, 0U);
          EXPECT_THROW(
              CandidateGenerator(
                  std::move(fixture.platform),
                  SymmetricYawOffsets(relative_yaw),
                  CandidateGenerator::Limits{1U, 5U, 100000U})
                  .Generate(
                      fixture.raster,
                      std::span<const FrontierCluster>(&frontier, 1U)),
              std::length_error);
        }
      }
    }
  }
}

TEST(CandidateGeneratorTest, ResourceBudgetsThrowInsteadOfReturningEmptyEvidence) {
  std::vector<std::int8_t> data(30U * 20U, 0);
  data[10U * 30U + 20U] = -1;
  const TaskRaster raster = Raster(30U, 20U, data);
  const FrontierCluster frontier =
      OneEdgeCluster(raster, GridIndex{19, 10}, GridIndex{20, 10}, 0U);
  const auto span = std::span<const FrontierCluster>(&frontier, 1U);

  const auto exact =
      CandidateGenerator(WheelPlatform(), StandardYawOffsets(),
                         CandidateGenerator::Limits{1U, 5U, 483U})
          .Generate(raster, span);
  ASSERT_EQ(exact.size(), 5U);
  EXPECT_THROW(CandidateGenerator(WheelPlatform(), StandardYawOffsets(),
                                  CandidateGenerator::Limits{1U, 4U, 1000000U})
                   .Generate(raster, span),
               std::length_error);
  EXPECT_THROW(CandidateGenerator(WheelPlatform(), StandardYawOffsets(),
                                  CandidateGenerator::Limits{1U, 5U, 482U})
                   .Generate(raster, span),
               std::length_error);
  EXPECT_THROW(CandidateGenerator(WheelPlatform(), StandardYawOffsets(),
                                  CandidateGenerator::Limits{1U, 5U, 1U})
                   .Generate(raster, span),
               std::length_error);

  auto blocked_data = data;
  blocked_data[10U * 30U + 18U] = 100;
  const TaskRaster blocked = Raster(30U, 20U, blocked_data);
  EXPECT_THROW(CandidateGenerator(WheelPlatform(), StandardYawOffsets(),
                                  CandidateGenerator::Limits{1U, 100U, 1000000U})
                   .Generate(blocked, span),
               std::length_error);
}

TEST(CandidateGeneratorTest, BaselineSearchEnumeratesExactlyStepsZeroThroughEleven) {
  std::vector<std::int8_t> data(30U * 20U, 100);
  data[10U * 30U + 19U] = 0;
  data[10U * 30U + 20U] = -1;
  const TaskRaster raster = Raster(30U, 20U, data);
  const FrontierCluster frontier =
      OneEdgeCluster(raster, GridIndex{19, 10}, GridIndex{20, 10}, 0U);
  const auto span = std::span<const FrontierCluster>(&frontier, 1U);

  EXPECT_TRUE(CandidateGenerator(WheelPlatform(), StandardYawOffsets(),
                                 CandidateGenerator::Limits{12U, 5U, 12U})
                  .Generate(raster, span)
                  .empty());
  EXPECT_THROW(CandidateGenerator(WheelPlatform(), StandardYawOffsets(),
                                  CandidateGenerator::Limits{11U, 5U, 12U})
                   .Generate(raster, span),
               std::length_error);
}

TEST(CandidateGeneratorTest, ExactDoubleSearchEndpointCanBeTheOnlySafePosition) {
  std::vector<std::int8_t> data(30U * 20U, 0);
  data[10U * 30U + 20U] = -1;
  for (std::size_t x = 10U; x <= 18U; ++x) {
    for (std::size_t y = 0U; y < 20U; ++y) {
      data[y * 30U + x] = 100;
    }
  }
  data[10U * 30U + 19U] = 0;
  const TaskRaster raster = Raster(30U, 20U, data);
  const FrontierCluster frontier =
      OneEdgeCluster(raster, GridIndex{19, 10}, GridIndex{20, 10}, 0U);
  const auto views =
      CandidateGenerator(OneMeterWheelPlatform(), StandardYawOffsets(),
                         CandidateGenerator::Limits{11U, 5U, 1000000U})
          .Generate(raster,
                    std::span<const FrontierCluster>(&frontier, 1U));

  ASSERT_EQ(views.size(), 5U);
  const double expected_x =
      4.0 - std::hypot(0.5, 0.1) - 10.0 * 0.2;
  for (const CandidateView& view : views) {
    EXPECT_NEAR(view.pose.x, expected_x, 1.0e-12);
  }
}

TEST(CandidateGeneratorTest, DisplayIdentityDoesNotReplaceCandidateKeyIdentity) {
  const CandidateView first{42U, 7U, 3U, CandidateKey{1, 2, 3},
                            Pose2{0.0015, -0.0015, kPi}, 1.0};
  const CandidateView same_display_different_key{
      42U, 7U, 3U, CandidateKey{1, 2, 4}, Pose2{}, 1.0};
  EXPECT_EQ(first.id, same_display_different_key.id);
  EXPECT_NE(first.key, same_display_different_key.key);
}

TEST(CandidateGeneratorTest, FullFrontierKeyChangesCandidateDisplayIdThroughGenerator) {
  std::vector<std::int8_t> data(30U * 20U, 0);
  data[10U * 30U + 20U] = -1;
  const TaskRaster raster = Raster(30U, 20U, data);
  FrontierCluster first =
      OneEdgeCluster(raster, GridIndex{19, 10}, GridIndex{20, 10}, 0U);
  FrontierCluster second = first;
  first.id = 42U;
  second.id = 42U;
  first.canonical_key = {1, 2, 3};
  second.canonical_key = {1, 2, 4};
  const CandidateGenerator generator(WheelPlatform(), StandardYawOffsets(),
                                     GenerousLimits());
  const auto first_views = generator.Generate(
      raster, std::span<const FrontierCluster>(&first, 1U));
  const auto second_views = generator.Generate(
      raster, std::span<const FrontierCluster>(&second, 1U));

  ASSERT_EQ(first_views.size(), 5U);
  ASSERT_EQ(second_views.size(), first_views.size());
  for (std::size_t index = 0U; index < first_views.size(); ++index) {
    EXPECT_EQ(first_views[index].frontier_id, second_views[index].frontier_id);
    EXPECT_EQ(first_views[index].key, second_views[index].key);
    EXPECT_NE(first_views[index].id, second_views[index].id);
  }
}

TEST(CandidateGeneratorTest, QuantizesHalfBoundariesAndNormalizesPositivePi) {
  std::vector<std::int8_t> data(50U * 50U, 0);
  for (const GridIndex unknown : {
           GridIndex{21, 20}, GridIndex{20, 21},
           GridIndex{19, 20}, GridIndex{20, 19}}) {
    data[static_cast<std::size_t>(unknown.y) * 50U +
         static_cast<std::size_t>(unknown.x)] = -1;
  }
  constexpr double kHalfTenthDegree = kPi / 3600.0;
  const double origin_x = 0.0015000000001 - 1.8;
  const double origin_y = -0.0015000000001 - 2.05;
  const TaskRaster raster =
      Raster(50U, 50U, data, 0.1, origin_x, origin_y);
  const FrontierCluster frontier =
      OneEdgeCluster(raster, GridIndex{20, 20}, GridIndex{21, 20}, 0U);
  const CandidateParameters half_offsets{{
      -0.2,
      std::nextafter(-kHalfTenthDegree,
                     -std::numeric_limits<double>::infinity()),
      0.0,
      std::nextafter(kHalfTenthDegree,
                     std::numeric_limits<double>::infinity()),
      0.2}};
  const auto half_views =
      CandidateGenerator(RotationallySymmetricWheelPlatform(kPi / 16.0),
                         half_offsets,
                         GenerousLimits())
          .Generate(raster,
                    std::span<const FrontierCluster>(&frontier, 1U));
  ASSERT_EQ(half_views.size(), 5U);
  EXPECT_EQ(half_views.front().key.x_mm, 2);
  EXPECT_EQ(half_views.front().key.y_mm, -2);
  EXPECT_TRUE(std::any_of(half_views.begin(), half_views.end(),
                          [](const CandidateView& view) {
                            return view.key.yaw_tenth_deg == -1;
                          }));
  EXPECT_TRUE(std::any_of(half_views.begin(), half_views.end(),
                          [](const CandidateView& view) {
                            return view.key.yaw_tenth_deg == 1;
                          }));

  const CandidateParameters plus_pi{{-0.4, -0.2, 0.0, 0.2, kPi}};
  const CandidateParameters minus_pi{{-kPi, -0.2, 0.0, 0.2, 0.4}};
  const auto plus_views =
      CandidateGenerator(RotationallySymmetricWheelPlatform(kPi / 16.0),
                         plus_pi,
                         GenerousLimits())
          .Generate(raster,
                    std::span<const FrontierCluster>(&frontier, 1U));
  const auto minus_views =
      CandidateGenerator(RotationallySymmetricWheelPlatform(kPi / 16.0),
                         minus_pi,
                         GenerousLimits())
          .Generate(raster,
                    std::span<const FrontierCluster>(&frontier, 1U));
  const auto plus = std::find_if(
      plus_views.begin(), plus_views.end(), [](const CandidateView& view) {
        return view.key.yaw_tenth_deg == -1800;
      });
  const auto minus = std::find_if(
      minus_views.begin(), minus_views.end(), [](const CandidateView& view) {
        return view.key.yaw_tenth_deg == -1800;
      });
  ASSERT_NE(plus, plus_views.end());
  ASSERT_NE(minus, minus_views.end());
  EXPECT_DOUBLE_EQ(plus->pose.yaw, -kPi);
  EXPECT_DOUBLE_EQ(minus->pose.yaw, -kPi);
  EXPECT_EQ(plus->key, minus->key);
  EXPECT_EQ(plus->id, minus->id);
}

TEST(CandidateGeneratorTest, AcceptsNegativeZeroOffsetAndEmitsCanonicalZeroYaw) {
  std::vector<std::int8_t> data(50U * 50U, 0);
  data[20U * 50U + 21U] = -1;
  const TaskRaster raster = Raster(50U, 50U, data, 0.1);
  const FrontierCluster frontier =
      OneEdgeCluster(raster, GridIndex{20, 20}, GridIndex{21, 20}, 0U);
  const CandidateParameters offsets{{-0.4, -0.2, -0.0, 0.2, 0.4}};
  const auto views =
      CandidateGenerator(RotationallySymmetricWheelPlatform(kPi / 16.0),
                         offsets, GenerousLimits())
          .Generate(raster,
                    std::span<const FrontierCluster>(&frontier, 1U));

  const auto zero = std::find_if(
      views.begin(), views.end(), [](const CandidateView& view) {
        return view.key.yaw_tenth_deg == 0;
      });
  ASSERT_NE(zero, views.end());
  EXPECT_DOUBLE_EQ(zero->pose.yaw, 0.0);
  EXPECT_FALSE(std::signbit(zero->pose.yaw));
}

TEST(CandidateGeneratorTest, RasterLimitRejectsBeforeCandidateOutputWork) {
  const GridGeometry geometry{2U, 2U, 1.0, 0.0, 0.0, 0.0};
  const std::array<std::int8_t, 4> data{0, 0, 0, -1};
  const OccupancyGridView map(geometry, data, 50);
  const Polygon2 polygon{{Vec2{0.0, 0.0}, Vec2{2.0, 0.0},
                          Vec2{2.0, 2.0}, Vec2{0.0, 2.0}}};
  EXPECT_THROW(TaskRaster::Build(
                   map, polygon,
                   TaskRaster::Limits{.maximum_raster_cell_count = 3U}),
               std::length_error);

  const TaskRaster raster = TaskRaster::Build(
      map, polygon,
      TaskRaster::Limits{.maximum_raster_cell_count = 4U});
  const FrontierCluster frontier =
      OneEdgeCluster(raster, GridIndex{0, 1}, GridIndex{1, 1}, 0U);
  EXPECT_THROW(
      CandidateGenerator(
          CompactWheelPlatform(), StandardYawOffsets(),
          CandidateGenerator::Limits{100U, 1U, 100000U})
          .Generate(raster,
                    std::span<const FrontierCluster>(&frontier, 1U)),
      std::length_error);
}

TEST(CandidateGeneratorTest, FiniteCandidateCoordinatesOverflowBeforeQuantization) {
  std::vector<std::int8_t> data(50U * 50U, 0);
  data[20U * 50U + 21U] = -1;
  const TaskRaster raster = Raster(50U, 50U, data, 4.0, 1.0e16, 0.0);
  const FrontierCluster frontier =
      OneEdgeCluster(raster, GridIndex{20, 20}, GridIndex{21, 20}, 0U);
  EXPECT_THROW(
      CandidateGenerator(LargeWheelPlatform(),
                         StandardYawOffsets(), GenerousLimits())
          .Generate(raster,
                    std::span<const FrontierCluster>(&frontier, 1U)),
      std::overflow_error);
}

TEST(CandidateGeneratorTest, TinyResolutionTerminatesAtExplicitPositionBudget) {
  std::vector<std::int8_t> data(50U * 50U, 0);
  data[20U * 50U + 21U] = -1;
  const TaskRaster raster = Raster(50U, 50U, data, 1.0e-6);
  const FrontierCluster frontier =
      OneEdgeCluster(raster, GridIndex{20, 20}, GridIndex{21, 20}, 0U);
  EXPECT_THROW(
      CandidateGenerator(
          RotationallySymmetricWheelPlatform(), StandardYawOffsets(),
          CandidateGenerator::Limits{3U, 5U, 240U})
          .Generate(raster,
                    std::span<const FrontierCluster>(&frontier, 1U)),
      std::length_error);
}

}  // namespace
}  // namespace lunar::pure_exploration
