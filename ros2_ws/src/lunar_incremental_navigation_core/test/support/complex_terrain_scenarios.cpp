#include "support/complex_terrain_scenarios.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace lunar::incremental_navigation::test_support {
namespace {

[[nodiscard]] std::size_t Offset(const ComplexTerrainScenario& scenario,
                                 const GridIndex index) {
  if (index.x < 0 || index.y < 0 ||
      index.x >= static_cast<std::int64_t>(scenario.geometry.width) ||
      index.y >= static_cast<std::int64_t>(scenario.geometry.height)) {
    throw std::out_of_range("complex terrain cell lies outside its geometry");
  }
  return static_cast<std::size_t>(index.y) * scenario.geometry.width +
         static_cast<std::size_t>(index.x);
}

void SetBlocked(ComplexTerrainScenario& scenario, const GridIndex index) {
  const std::size_t offset = Offset(scenario, index);
  scenario.elevation_m[offset] = 1.0F;
  scenario.fine_states[offset] = FineCellState::kBlocked;
  scenario.traversal_costs[offset] = 0.0;
  scenario.guidance_states[offset] = GuidanceCellState::kProvenBlocked;
  scenario.guidance_risks[offset] = 0.0;
}

void SetFree(ComplexTerrainScenario& scenario, const GridIndex index) {
  const std::size_t offset = Offset(scenario, index);
  scenario.elevation_m[offset] = 0.0F;
  scenario.fine_states[offset] = FineCellState::kFree;
  scenario.traversal_costs[offset] = 0.0;
  scenario.guidance_states[offset] = GuidanceCellState::kCandidate;
  scenario.guidance_risks[offset] = 0.0;
}

void SetUnknown(ComplexTerrainScenario& scenario, const GridIndex index) {
  const std::size_t offset = Offset(scenario, index);
  scenario.elevation_m[offset] = std::numeric_limits<float>::quiet_NaN();
  scenario.fine_states[offset] = FineCellState::kUnknown;
  scenario.traversal_costs[offset] = 0.0;
  scenario.guidance_states[offset] = GuidanceCellState::kUnknown;
  scenario.guidance_risks[offset] = 0.0;
}

void SetRisk(ComplexTerrainScenario& scenario, const GridIndex index,
             const double cost) {
  const std::size_t offset = Offset(scenario, index);
  scenario.elevation_m[offset] = 0.0F;
  scenario.fine_states[offset] = FineCellState::kFree;
  scenario.traversal_costs[offset] = cost;
  scenario.guidance_states[offset] = GuidanceCellState::kCandidate;
  scenario.guidance_risks[offset] = cost;
}

void SetElevation(ComplexTerrainScenario& scenario, const GridIndex index,
                  const float elevation_m) {
  const std::size_t offset = Offset(scenario, index);
  scenario.elevation_m[offset] = elevation_m;
  scenario.fine_states[offset] = FineCellState::kFree;
  scenario.traversal_costs[offset] = 0.0;
  scenario.guidance_states[offset] = GuidanceCellState::kCandidate;
  scenario.guidance_risks[offset] = 0.0;
}

void AddBoundary(ComplexTerrainScenario& scenario) {
  const std::int64_t width =
      static_cast<std::int64_t>(scenario.geometry.width);
  const std::int64_t height =
      static_cast<std::int64_t>(scenario.geometry.height);
  for (std::int64_t x = 0; x < width; ++x) {
    SetBlocked(scenario, {.x = x, .y = 0});
    SetBlocked(scenario, {.x = x, .y = height - 1});
  }
  for (std::int64_t y = 1; y + 1 < height; ++y) {
    SetBlocked(scenario, {.x = 0, .y = y});
    SetBlocked(scenario, {.x = width - 1, .y = y});
  }
}

void AddAlternatingWallMaze(ComplexTerrainScenario& scenario) {
  const std::int64_t width =
      static_cast<std::int64_t>(scenario.geometry.width);
  const std::int64_t height =
      static_cast<std::int64_t>(scenario.geometry.height);
  bool gap_near_bottom = true;
  for (std::int64_t x = 8; x + 8 < width; x += 8) {
    const std::int64_t gap_center = gap_near_bottom ? 3 : height - 4;
    for (std::int64_t y = 1; y + 1 < height; ++y) {
      if (std::abs(y - gap_center) <= 1) {
        continue;
      }
      SetBlocked(scenario, {.x = x, .y = y});
    }
    gap_near_bottom = !gap_near_bottom;
  }
}

[[nodiscard]] std::uint64_t CellHash(const std::int64_t x,
                                     const std::int64_t y,
                                     const std::uint32_t seed) noexcept {
  std::uint64_t value = static_cast<std::uint64_t>(seed) ^
                        (static_cast<std::uint64_t>(x) << 32U) ^
                        static_cast<std::uint64_t>(y);
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] std::int64_t RockCorridorY(
    const ComplexTerrainScenario& scenario, const std::int64_t x) {
  const double width = static_cast<double>(scenario.geometry.width - 1U);
  const double phase = 2.0 * std::numbers::pi *
                       static_cast<double>(x) / width;
  const double amplitude =
      0.28 * static_cast<double>(scenario.geometry.height - 2U);
  const double center =
      0.5 * static_cast<double>(scenario.geometry.height - 1U);
  return std::clamp(
      static_cast<std::int64_t>(std::llround(center + amplitude * std::sin(phase))),
      std::int64_t{2},
      static_cast<std::int64_t>(scenario.geometry.height) - 3);
}

void AddDenseRockField(ComplexTerrainScenario& scenario,
                       const std::uint32_t seed) {
  const std::int64_t width =
      static_cast<std::int64_t>(scenario.geometry.width);
  const std::int64_t height =
      static_cast<std::int64_t>(scenario.geometry.height);
  for (std::int64_t y = 1; y + 1 < height; ++y) {
    for (std::int64_t x = 1; x + 1 < width; ++x) {
      if (CellHash(x, y, seed) % 100U < 38U) {
        SetBlocked(scenario, {.x = x, .y = y});
      }
    }
  }
  for (std::int64_t x = 1; x + 1 < width; ++x) {
    const std::int64_t center_y = RockCorridorY(scenario, x);
    for (std::int64_t dy = -2; dy <= 2; ++dy) {
      const std::int64_t y = center_y + dy;
      if (y > 0 && y + 1 < height) {
        SetFree(scenario, {.x = x, .y = y});
      }
    }
  }
  scenario.start_cell = {.x = 2, .y = RockCorridorY(scenario, 2)};
  scenario.goal_cell = {.x = width - 3,
                        .y = RockCorridorY(scenario, width - 3)};
}

void AddNarrowPassagesAndDeadEnds(ComplexTerrainScenario& scenario) {
  const std::int64_t width =
      static_cast<std::int64_t>(scenario.geometry.width);
  const std::int64_t height =
      static_cast<std::int64_t>(scenario.geometry.height);
  for (std::int64_t y = 0; y < height; ++y) {
    for (std::int64_t x = 0; x < width; ++x) {
      SetBlocked(scenario, {.x = x, .y = y});
    }
  }

  std::vector<std::int64_t> corridor_rows;
  for (std::int64_t y = 4; y + 4 < height; y += 8) {
    corridor_rows.push_back(y);
  }
  for (std::size_t row = 0; row < corridor_rows.size(); ++row) {
    const std::int64_t y = corridor_rows[row];
    for (std::int64_t x = 2; x + 2 < width; ++x) {
      SetFree(scenario, {.x = x, .y = y});
    }
    if (row + 1U < corridor_rows.size()) {
      const std::int64_t connector_x = row % 2U == 0U ? width - 3 : 2;
      for (std::int64_t connector_y = y;
           connector_y <= corridor_rows[row + 1U]; ++connector_y) {
        SetFree(scenario, {.x = connector_x, .y = connector_y});
      }
    }
    for (std::int64_t x = 10; x + 4 < width; x += 12) {
      if ((row % 2U == 0U && x + 3 >= width - 3) ||
          (row % 2U != 0U && x <= 3)) {
        continue;
      }
      for (std::int64_t length = 1; length <= 3; ++length) {
        const std::int64_t branch_y =
            row % 2U == 0U ? y - length : y + length;
        if (branch_y > 0 && branch_y + 1 < height) {
          SetFree(scenario, {.x = x, .y = branch_y});
        }
      }
    }
  }
  scenario.start_cell = {.x = 2, .y = corridor_rows.front()};
  const std::size_t last = corridor_rows.size() - 1U;
  scenario.goal_cell = {
      .x = last % 2U == 0U ? width - 3 : 2,
      .y = corridor_rows.back(),
  };
}

void AddRiskAndUnknownBands(ComplexTerrainScenario& scenario,
                            const std::uint32_t seed) {
  const std::int64_t width =
      static_cast<std::int64_t>(scenario.geometry.width);
  const std::int64_t height =
      static_cast<std::int64_t>(scenario.geometry.height);
  bool gap_near_bottom = true;
  for (std::int64_t x = 12; x + 8 < width; x += 16) {
    const std::int64_t gap_center = gap_near_bottom ? 5 : height - 6;
    for (std::int64_t band_x = x; band_x < x + 2; ++band_x) {
      for (std::int64_t y = 1; y + 1 < height; ++y) {
        if (std::abs(y - gap_center) <= 2) {
          SetFree(scenario, {.x = band_x, .y = y});
        } else {
          SetRisk(scenario, {.x = band_x, .y = y}, 30.0);
        }
      }
    }
    gap_near_bottom = !gap_near_bottom;
  }
  for (std::int64_t x = 18; x + 6 < width; x += 24) {
    const std::int64_t base_y =
        8 + static_cast<std::int64_t>(CellHash(x, x, seed) %
                                      static_cast<std::uint64_t>(height - 16));
    for (std::int64_t y = base_y - 2; y <= base_y + 2; ++y) {
      for (std::int64_t patch_x = x; patch_x < x + 5; ++patch_x) {
        SetUnknown(scenario, {.x = patch_x, .y = y});
      }
    }
  }
}

void AddEnclosedGoal(ComplexTerrainScenario& scenario) {
  for (std::int64_t dy = -2; dy <= 2; ++dy) {
    for (std::int64_t dx = -2; dx <= 2; ++dx) {
      if (std::max(std::abs(dx), std::abs(dy)) == 2) {
        SetBlocked(scenario, {.x = scenario.goal_cell.x + dx,
                              .y = scenario.goal_cell.y + dy});
      }
    }
  }
}

void AddLeggedStepGapField(ComplexTerrainScenario& scenario) {
  const std::int64_t width =
      static_cast<std::int64_t>(scenario.geometry.width);
  const std::int64_t height =
      static_cast<std::int64_t>(scenario.geometry.height);
  bool gap_near_bottom = true;
  for (std::int64_t x = 8; x + 8 < width; x += 8) {
    const std::int64_t gap_center = gap_near_bottom ? 4 : height - 5;
    for (std::int64_t y = 1; y + 1 < height; ++y) {
      if (std::abs(y - gap_center) > 1) {
        SetElevation(scenario, {.x = x, .y = y}, 0.8F);
      }
    }
    gap_near_bottom = !gap_near_bottom;
  }
}

}  // namespace

FineCellState ComplexTerrainScenario::State(const GridIndex index) const {
  return fine_states.at(Offset(*this, index));
}

ComplexTerrainScenario MakeComplexTerrainScenario(
    const ComplexTerrainSpec& spec) {
  if (spec.width < 32U || spec.height < 32U ||
      !std::isfinite(spec.resolution_m) || spec.resolution_m <= 0.0) {
    throw std::invalid_argument(
        "complex terrain requires at least 32x32 cells and positive resolution");
  }
  if (spec.height > std::numeric_limits<std::size_t>::max() / spec.width) {
    throw std::invalid_argument("complex terrain cell count overflows");
  }
  const std::size_t count = spec.width * spec.height;
  std::string name;
  switch (spec.pattern) {
    case ComplexTerrainPattern::kDenseRockField:
      name = "dense_rock_field";
      break;
    case ComplexTerrainPattern::kAlternatingWallMaze:
      name = "alternating_wall_maze";
      break;
    case ComplexTerrainPattern::kNarrowPassagesAndDeadEnds:
      name = "narrow_passages_and_dead_ends";
      break;
    case ComplexTerrainPattern::kRiskAndUnknownBands:
      name = "risk_and_unknown_bands";
      break;
    case ComplexTerrainPattern::kEnclosedGoal:
      name = "enclosed_goal";
      break;
    case ComplexTerrainPattern::kLeggedStepGapField:
      name = "legged_step_gap_field";
      break;
  }
  ComplexTerrainScenario scenario{
      .name = std::move(name),
      .geometry = GridGeometry{.frame_id = "map",
                               .width = spec.width,
                               .height = spec.height,
                               .resolution_m = spec.resolution_m},
      .elevation_m = std::vector<float>(count, 0.0F),
      .fine_states = std::vector<FineCellState>(count, FineCellState::kFree),
      .traversal_costs = std::vector<double>(count, 0.0),
      .guidance_states = std::vector<GuidanceCellState>(
          count, GuidanceCellState::kCandidate),
      .guidance_risks = std::vector<double>(count, 0.0),
      .start_cell = {.x = 2,
                     .y = static_cast<std::int64_t>(spec.height / 2U)},
      .goal_cell = {.x = static_cast<std::int64_t>(spec.width) - 3,
                    .y = static_cast<std::int64_t>(spec.height / 2U)},
      .expected_reachable = true,
  };

  AddBoundary(scenario);
  switch (spec.pattern) {
    case ComplexTerrainPattern::kDenseRockField:
      AddDenseRockField(scenario, spec.seed);
      break;
    case ComplexTerrainPattern::kAlternatingWallMaze:
      AddAlternatingWallMaze(scenario);
      break;
    case ComplexTerrainPattern::kNarrowPassagesAndDeadEnds:
      AddNarrowPassagesAndDeadEnds(scenario);
      break;
    case ComplexTerrainPattern::kRiskAndUnknownBands:
      AddRiskAndUnknownBands(scenario, spec.seed);
      break;
    case ComplexTerrainPattern::kEnclosedGoal:
      AddEnclosedGoal(scenario);
      scenario.expected_reachable = false;
      break;
    case ComplexTerrainPattern::kLeggedStepGapField:
      AddLeggedStepGapField(scenario);
      break;
  }
  SetFree(scenario, scenario.start_cell);
  SetFree(scenario, scenario.goal_cell);
  return scenario;
}

std::shared_ptr<const ElevationSnapshot> MakeElevationSnapshot(
    const ComplexTerrainScenario& scenario) {
  PersistentElevationMap map;
  const ElevationUpdateResult update = map.Apply(ElevationEvidence{
      .geometry = scenario.geometry,
      .elevation_m = scenario.elevation_m,
      .map_from_source =
          RigidTransform{.parent_frame = "map", .child_frame = "map"},
  });
  if (update.status != ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("complex terrain elevation was rejected");
  }
  return map.Snapshot();
}

std::shared_ptr<const FineTraversabilitySnapshot> MakeFineSnapshot(
    const ComplexTerrainScenario& scenario, std::string profile_hash,
    const std::uint64_t fine_revision,
    std::vector<TileIndex> changed_tiles) {
  const auto elevation = MakeElevationSnapshot(scenario);
  const SparseGridGeometry& geometry = elevation->geometry();
  if (scenario.fine_states.size() != geometry.CellCount() ||
      scenario.traversal_costs.size() != geometry.CellCount()) {
    throw std::invalid_argument("complex fine fixture dimensions do not match");
  }

  struct MutableTile final {
    FineTraversabilityTile::StateArray states;
    FineTraversabilityTile::CostArray costs;

    MutableTile() {
      states.fill(FineCellState::kUnknown);
      costs.fill(0.0);
    }
  };
  std::map<TileIndex, MutableTile> tiles;
  std::size_t input_offset = 0U;
  for (std::int64_t y = geometry.min_inclusive().y;
       y < geometry.max_exclusive().y; ++y) {
    for (std::int64_t x = geometry.min_inclusive().x;
         x < geometry.max_exclusive().x; ++x, ++input_offset) {
      const GridIndex index{.x = x, .y = y};
      MutableTile& tile = tiles[TileForCell(index)];
      tile.states[TileCellOffset(index)] = scenario.fine_states[input_offset];
      tile.costs[TileCellOffset(index)] =
          scenario.traversal_costs[input_offset];
    }
  }
  FineTraversabilityTileDirectory directory(geometry);
  for (auto& [index, tile] : tiles) {
    directory = directory.WithTile(
        index, std::make_shared<const FineTraversabilityTile>(
                   std::move(tile.states), std::move(tile.costs)));
  }
  if (changed_tiles.empty()) {
    changed_tiles = directory.tile_indices();
  }
  return std::make_shared<const FineTraversabilitySnapshot>(
      geometry, elevation->raw_elevation_revision(), fine_revision,
      std::move(profile_hash), 0.0, 0.0,
      TraversalCostWeights{.slope = 1.0,
                           .relief = 1.0,
                           .clearance = 1.0},
      elevation, std::move(directory), changed_tiles, changed_tiles,
      FineSnapshotMetrics{});
}

std::shared_ptr<const GlobalGuidanceSnapshot> MakeGuidanceSnapshot(
    const ComplexTerrainScenario& scenario, std::string profile_hash,
    const std::uint64_t guidance_revision,
    std::vector<TileIndex> changed_tiles) {
  const SparseGridGeometry geometry(
      scenario.geometry.frame_id, scenario.geometry.resolution_m,
      scenario.geometry.origin_m, GridIndex{},
      GridIndex{.x = static_cast<std::int64_t>(scenario.geometry.width),
                .y = static_cast<std::int64_t>(scenario.geometry.height)});
  if (scenario.guidance_states.size() != geometry.CellCount() ||
      scenario.guidance_risks.size() != geometry.CellCount()) {
    throw std::invalid_argument(
        "complex guidance fixture dimensions do not match");
  }

  struct MutableTile final {
    GlobalGuidanceTile::StateArray states;
    GlobalGuidanceTile::RiskArray risks;

    MutableTile() {
      states.fill(GuidanceCellState::kUnknown);
      risks.fill(0.0);
    }
  };
  std::map<TileIndex, MutableTile> tiles;
  std::size_t input_offset = 0U;
  for (std::int64_t y = 0;
       y < static_cast<std::int64_t>(scenario.geometry.height); ++y) {
    for (std::int64_t x = 0;
         x < static_cast<std::int64_t>(scenario.geometry.width);
         ++x, ++input_offset) {
      const GridIndex index{.x = x, .y = y};
      MutableTile& tile = tiles[TileForCell(index)];
      tile.states[TileCellOffset(index)] =
          scenario.guidance_states[input_offset];
      tile.risks[TileCellOffset(index)] = scenario.guidance_risks[input_offset];
    }
  }
  GlobalGuidanceTileDirectory directory(geometry);
  for (auto& [index, tile] : tiles) {
    directory = directory.WithTile(
        index, std::make_shared<const GlobalGuidanceTile>(
                   std::move(tile.states), std::move(tile.risks)));
  }
  if (changed_tiles.empty()) {
    changed_tiles = directory.tile_indices();
  }
  return std::make_shared<const GlobalGuidanceSnapshot>(
      geometry, 1U, 1U, guidance_revision, std::move(profile_hash),
      std::move(directory),
      changed_tiles, changed_tiles);
}

Pose2 StartPose(const ComplexTerrainScenario& scenario) {
  const double resolution = scenario.geometry.resolution_m;
  return Pose2{.position_m = {
                   .x = scenario.geometry.origin_m.x +
                        (static_cast<double>(scenario.start_cell.x) + 0.5) *
                            resolution,
                   .y = scenario.geometry.origin_m.y +
                        (static_cast<double>(scenario.start_cell.y) + 0.5) *
                            resolution,
               }};
}

LocalTarget LocalTargetForGoal(const ComplexTerrainScenario& scenario) {
  const double resolution = scenario.geometry.resolution_m;
  return LocalTarget{
      .center = {
          .x = scenario.geometry.origin_m.x +
               (static_cast<double>(scenario.goal_cell.x) + 0.5) * resolution,
          .y = scenario.geometry.origin_m.y +
               (static_cast<double>(scenario.goal_cell.y) + 0.5) * resolution,
      },
      .position_tolerance_m = 0.5 * resolution,
      .is_final_goal = true,
  };
}

WheeledCapability BenchmarkWheelCapability() {
  return WheeledCapability{
      .footprint_xy_m = {{-0.3, -0.2},
                         {0.3, -0.2},
                         {0.3, 0.2},
                         {-0.3, 0.2}},
      .body_extent_m = {.x = 0.6, .y = 0.4, .z = 0.3},
      .wheel_diameter_m = 0.3,
      .wheel_width_m = 0.08,
      .wheelbase_m = 0.4,
      .track_width_m = 0.3,
      .minimum_underbody_clearance_m = 0.3,
      .maximum_local_obstacle_relief_m = 0.2,
      .minimum_body_z_m = 0.0,
      .maximum_body_z_m = 1.0,
      .maximum_forward_speed_mps = 1.0,
      .maximum_reverse_speed_mps = 0.5,
      .maximum_spin_rate_radps = 1.0,
      .maximum_acceleration_mps2 = 1.0,
      .maximum_braking_deceleration_mps2 = 1.0,
      .maximum_yaw_acceleration_radps2 = 1.0,
      .maximum_lateral_acceleration_mps2 = 1.0,
      .maximum_curvature_per_m = 2.0,
      .maximum_slope_rad = std::numbers::pi / 3.0,
      .minimum_clearance_m = 0.2,
      .motion_primitives = {
          {.primitive_id = "forward", .kind = WheelPrimitiveKind::kForward},
          {.primitive_id = "reverse", .kind = WheelPrimitiveKind::kReverse},
          {.primitive_id = "spin-left",
           .kind = WheelPrimitiveKind::kSpinCounterclockwise},
          {.primitive_id = "spin-right",
           .kind = WheelPrimitiveKind::kSpinClockwise},
      },
  };
}

LeggedCapability BenchmarkLeggedCapability() {
  return LeggedCapability{
      .body_extent_m = {.x = 0.6, .y = 0.4, .z = 0.3},
      .nominal_body_height_m = 0.5,
      .platform_mass_kg = 20.0,
      .nominal_payload_kg = 2.0,
      .maximum_payload_kg = 10.0,
      .maximum_slope_rad = std::numbers::pi / 3.0,
      .maximum_step_height_m = 0.25,
      .maximum_gap_width_m = 0.2,
      .minimum_body_clearance_m = 0.3,
      .step_vertical_rate_mps = 0.2,
      .body_height_m = {.lower = 0.4, .upper = 0.6},
      .forward_speed_mps = {.lower = -0.5, .upper = 0.8},
      .lateral_speed_mps = {.lower = -0.4, .upper = 0.4},
      .yaw_rate_radps = {.lower = -0.8, .upper = 0.8},
      .maximum_linear_acceleration_mps2 = 1.0,
      .maximum_yaw_acceleration_radps2 = 1.0,
      .unknown_is_traversable = false,
      .motion_primitives = {
          {.primitive_id = "forward",
           .kind = LeggedPrimitiveKind::kForward,
           .body_frame_displacement_m = {.x = 1.0}},
          {.primitive_id = "backward",
           .kind = LeggedPrimitiveKind::kBackward,
           .body_frame_displacement_m = {.x = -1.0}},
          {.primitive_id = "left",
           .kind = LeggedPrimitiveKind::kLateralLeft,
           .body_frame_displacement_m = {.y = 1.0}},
          {.primitive_id = "right",
           .kind = LeggedPrimitiveKind::kLateralRight,
           .body_frame_displacement_m = {.y = -1.0}},
          {.primitive_id = "spin-left",
           .kind = LeggedPrimitiveKind::kSpin,
           .yaw_change_rad = std::numbers::pi / 2.0},
          {.primitive_id = "spin-right",
           .kind = LeggedPrimitiveKind::kSpin,
           .yaw_change_rad = -std::numbers::pi / 2.0},
      },
  };
}

TraversabilityProfile BenchmarkTraversabilityProfile() {
  return TraversabilityProfile{
      .maximum_slope_rad = std::numbers::pi / 3.0,
      .planar_envelope_xy_m = {{-0.3, -0.2},
                               {0.3, -0.2},
                               {0.3, 0.2},
                               {-0.3, 0.2}},
      .preferred_clearance_m = 0.4,
      .slope_weight = 1.0,
      .relief_weight = 1.0,
      .clearance_weight = 1.0,
      .start_blind_zone_margin_m = 0.1,
      .goal_position_tolerance_m = 0.1,
      .goal_yaw_tolerance_rad = std::numbers::pi / 12.0,
  };
}

}  // namespace lunar::incremental_navigation::test_support
