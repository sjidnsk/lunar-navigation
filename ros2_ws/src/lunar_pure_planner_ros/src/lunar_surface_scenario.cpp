#include "lunar_pure_planner_ros/lunar_surface_scenario.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <utility>
#include <vector>

namespace lunar::pure_planner_ros {
namespace {

// A kilometre-scale scene at one metre per cell contains one million cells.
// Keeping the demo at this resolution avoids publishing the 25 million cells
// required for a 0.2 m map every half second.
constexpr std::size_t kWidth = 1000U;
constexpr std::size_t kHeight = 1000U;
constexpr double kCellSizeM = 1.0;

[[nodiscard]] double SquaredDistance(const LunarSurfaceCell lhs,
                                     const LunarSurfaceCell rhs) noexcept {
  const double dx = static_cast<double>(lhs.x) - static_cast<double>(rhs.x);
  const double dy = static_cast<double>(lhs.y) - static_cast<double>(rhs.y);
  return dx * dx + dy * dy;
}

[[nodiscard]] std::vector<std::int32_t> FloodFill(
    const LunarSurfaceScenario& scenario, const LunarSurfaceCell start) {
  const std::size_t count = scenario.width * scenario.height;
  std::vector<std::int32_t> distance(count, -1);
  if (!scenario.InBounds(start) || scenario.Occupied(start)) {
    return distance;
  }
  std::queue<LunarSurfaceCell> frontier;
  distance[scenario.Index(start)] = 0;
  frontier.push(start);
  constexpr std::array<std::pair<int, int>, 4U> kOffsets{{{1, 0}, {-1, 0},
                                                            {0, 1}, {0, -1}}};
  while (!frontier.empty()) {
    const LunarSurfaceCell current = frontier.front();
    frontier.pop();
    for (const auto [dx, dy] : kOffsets) {
      const auto next_x = static_cast<std::ptrdiff_t>(current.x) + dx;
      const auto next_y = static_cast<std::ptrdiff_t>(current.y) + dy;
      if (next_x < 0 || next_y < 0 ||
          next_x >= static_cast<std::ptrdiff_t>(scenario.width) ||
          next_y >= static_cast<std::ptrdiff_t>(scenario.height)) {
        continue;
      }
      const LunarSurfaceCell next{static_cast<std::size_t>(next_x),
                                  static_cast<std::size_t>(next_y)};
      if (scenario.Occupied(next) || distance[scenario.Index(next)] >= 0) {
        continue;
      }
      distance[scenario.Index(next)] = distance[scenario.Index(current)] + 1;
      frontier.push(next);
    }
  }
  return distance;
}

}  // namespace

bool LunarSurfaceScenario::InBounds(const LunarSurfaceCell cell) const noexcept {
  return cell.x < width && cell.y < height;
}

std::size_t LunarSurfaceScenario::Index(const LunarSurfaceCell cell) const noexcept {
  return cell.y * width + cell.x;
}

bool LunarSurfaceScenario::Occupied(const LunarSurfaceCell cell) const noexcept {
  return !InBounds(cell) || occupancy[Index(cell)] >= 50;
}

std::optional<LunarSurfaceSample> LunarSurfaceScenario::Sample(
    const double x_m, const double y_m) const noexcept {
  if (!std::isfinite(x_m) || !std::isfinite(y_m) || x_m < origin_x_m ||
      y_m < origin_y_m ||
      x_m >= origin_x_m + static_cast<double>(width) * resolution_m ||
      y_m >= origin_y_m + static_cast<double>(height) * resolution_m) {
    return std::nullopt;
  }

  double elevation = 0.012 * x_m +
                     0.38 * std::sin(x_m * 0.11) * std::cos(y_m * 0.09);
  bool occupied = false;
  for (const auto& crater : craters) {
    const double distance =
        std::hypot(x_m - crater.center_x_m, y_m - crater.center_y_m);
    const double normalized = distance / crater.radius_m;
    if (normalized < 1.0) {
      elevation -= 0.45 * (1.0 - normalized * normalized);
    }
    occupied = occupied || (normalized >= 0.90 && normalized <= 1.15);
  }
  for (const auto& rock : rocks) {
    occupied = occupied ||
               std::hypot(x_m - rock.center_x_m, y_m - rock.center_y_m) <=
                   rock.radius_m;
  }

  const double start_x_m = origin_x_m +
      (static_cast<double>(start_cell.x) + 0.5) * resolution_m;
  const double start_y_m = origin_y_m +
      (static_cast<double>(start_cell.y) + 0.5) * resolution_m;
  const double start_distance = std::hypot(x_m - start_x_m, y_m - start_y_m);
  if (start_distance <= 25.0) {
    occupied = false;
  }
  if (start_distance <= 50.0) {
    elevation = 0.0;
  }
  return LunarSurfaceSample{.occupied = occupied,
                            .elevation_m = static_cast<float>(elevation)};
}

LunarSurfaceScenario BuildLunarSurfaceScenario(const std::uint32_t seed) {
  LunarSurfaceScenario scenario;
  scenario.width = kWidth;
  scenario.height = kHeight;
  scenario.resolution_m = kCellSizeM;
  scenario.origin_x_m = -500.0;
  scenario.origin_y_m = -500.0;
  scenario.start_cell = LunarSurfaceCell{150U, 500U};
  scenario.occupancy.assign(kWidth * kHeight, 0);
  scenario.elevation_m.resize(kWidth * kHeight);

  std::mt19937 generator(seed);
  std::uniform_int_distribution<std::size_t> x_distribution(80U, kWidth - 81U);
  std::uniform_int_distribution<std::size_t> y_distribution(80U, kHeight - 81U);
  std::uniform_real_distribution<double> radius_distribution(20.0, 45.0);
  scenario.craters.reserve(12U);
  for (std::size_t crater = 0U; crater < 12U; ++crater) {
    // Preserve the fixed-seed layout produced by the former emplace_back
    // expression on the supported GCC toolchain, which evaluated the radius
    // argument before the braced centre argument.
    const double radius_m =
        radius_distribution(generator) * scenario.resolution_m;
    const LunarSurfaceCell center{x_distribution(generator),
                                  y_distribution(generator)};
    scenario.craters.push_back(LunarSurfaceCrater{
        .center_x_m = scenario.origin_x_m +
            (static_cast<double>(center.x) + 0.5) * scenario.resolution_m,
        .center_y_m = scenario.origin_y_m +
            (static_cast<double>(center.y) + 0.5) * scenario.resolution_m,
        .radius_m = radius_m});
  }

  std::bernoulli_distribution rock_distribution(0.00015);
  for (std::size_t y = 1U; y + 1U < kHeight; ++y) {
    for (std::size_t x = 1U; x + 1U < kWidth; ++x) {
      if (rock_distribution(generator)) {
        scenario.rocks.push_back(LunarSurfaceRock{
            .center_x_m = scenario.origin_x_m +
                (static_cast<double>(x) + 0.5) * scenario.resolution_m,
            .center_y_m = scenario.origin_y_m +
                (static_cast<double>(y) + 0.5) * scenario.resolution_m,
            .radius_m = scenario.resolution_m});
      }
    }
  }

  for (std::size_t y = 0U; y < kHeight; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      const LunarSurfaceCell cell{x, y};
      const double world_x = scenario.origin_x_m +
                             (static_cast<double>(x) + 0.5) * kCellSizeM;
      const double world_y = scenario.origin_y_m +
                             (static_cast<double>(y) + 0.5) * kCellSizeM;
      const auto sample = scenario.Sample(world_x, world_y);
      scenario.occupancy[scenario.Index(cell)] =
          sample.has_value() && sample->occupied ? 100 : 0;
      scenario.elevation_m[scenario.Index(cell)] =
          sample.has_value() ? sample->elevation_m : 0.0F;
    }
  }

  scenario.default_goal_cell = LunarSurfaceCell{850U, 500U};

  const std::vector<std::int32_t> distance = FloodFill(scenario, scenario.start_cell);
  if (distance[scenario.Index(scenario.default_goal_cell)] < 0) {
    for (std::size_t index = 0U; index < distance.size(); ++index) {
      if (distance[index] > 0) {
        scenario.default_goal_cell = LunarSurfaceCell{index % kWidth, index / kWidth};
        break;
      }
    }
  }
  return scenario;
}

bool CellsConnected(const LunarSurfaceScenario& scenario, const LunarSurfaceCell start,
                    const LunarSurfaceCell goal) {
  const std::vector<std::int32_t> distance = FloodFill(scenario, start);
  return scenario.InBounds(goal) && distance[scenario.Index(goal)] >= 0;
}

}  // namespace lunar::pure_planner_ros
