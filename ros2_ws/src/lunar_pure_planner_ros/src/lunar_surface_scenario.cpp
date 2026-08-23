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

constexpr std::size_t kWidth = 500U;
constexpr std::size_t kHeight = 500U;
constexpr double kCellSizeM = 0.2;

[[nodiscard]] double SquaredDistance(const LunarSurfaceCell lhs,
                                     const LunarSurfaceCell rhs) noexcept {
  const double dx = static_cast<double>(lhs.x) - static_cast<double>(rhs.x);
  const double dy = static_cast<double>(lhs.y) - static_cast<double>(rhs.y);
  return dx * dx + dy * dy;
}

void MarkObstacleDisk(LunarSurfaceScenario& scenario, const LunarSurfaceCell center,
                      const double radius_m) {
  const double radius_squared = radius_m * radius_m;
  for (std::size_t y = 0U; y < scenario.height; ++y) {
    for (std::size_t x = 0U; x < scenario.width; ++x) {
      const LunarSurfaceCell cell{x, y};
      if (SquaredDistance(cell, center) <= radius_squared) {
        scenario.occupancy[scenario.Index(cell)] = 100;
      }
    }
  }
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

LunarSurfaceScenario BuildLunarSurfaceScenario(const std::uint32_t seed) {
  LunarSurfaceScenario scenario;
  scenario.width = kWidth;
  scenario.height = kHeight;
  scenario.resolution_m = kCellSizeM;
  scenario.origin_x_m = -50.0;
  scenario.origin_y_m = -50.0;
  scenario.occupancy.assign(kWidth * kHeight, 0);
  scenario.elevation_m.resize(kWidth * kHeight);

  std::mt19937 generator(seed);
  std::uniform_int_distribution<std::size_t> x_distribution(10U, kWidth - 11U);
  std::uniform_int_distribution<std::size_t> y_distribution(10U, kHeight - 11U);
  std::uniform_real_distribution<double> radius_distribution(1.0, 2.0);
  std::vector<std::pair<LunarSurfaceCell, double>> craters;
  craters.reserve(14U);
  for (std::size_t crater = 0U; crater < 14U; ++crater) {
    craters.emplace_back(LunarSurfaceCell{x_distribution(generator),
                                          y_distribution(generator)},
                         radius_distribution(generator) * 15.0);
  }

  for (std::size_t y = 0U; y < kHeight; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      const LunarSurfaceCell cell{x, y};
      const double world_x = scenario.origin_x_m +
                             (static_cast<double>(x) + 0.5) * kCellSizeM;
      const double world_y = scenario.origin_y_m +
                             (static_cast<double>(y) + 0.5) * kCellSizeM;
      double elevation = 0.012 * world_x +
                         0.38 * std::sin(world_x * 0.11) * std::cos(world_y * 0.09);
      for (const auto& [center, radius] : craters) {
        const double normalized = std::sqrt(SquaredDistance(cell, center)) / radius;
        if (normalized < 1.0) {
          elevation -= 0.45 * (1.0 - normalized * normalized);
        }
        if (normalized >= 0.90 && normalized <= 1.15) {
          scenario.occupancy[scenario.Index(cell)] = 100;
        }
      }
      scenario.elevation_m[scenario.Index(cell)] = static_cast<float>(elevation);
    }
  }

  std::bernoulli_distribution rock_distribution(0.001);
  for (std::size_t y = 1U; y + 1U < kHeight; ++y) {
    for (std::size_t x = 1U; x + 1U < kWidth; ++x) {
      if (rock_distribution(generator)) {
        MarkObstacleDisk(scenario, LunarSurfaceCell{x, y}, 1.0);
      }
    }
  }

  for (std::size_t y = 0U; y < kHeight; ++y) {
    for (std::size_t x = 0U; x < kWidth; ++x) {
      const LunarSurfaceCell cell{x, y};
      if (SquaredDistance(cell, scenario.start_cell) <= 225.0) {
        scenario.occupancy[scenario.Index(cell)] = 0;
      }
      if (SquaredDistance(cell, scenario.start_cell) <= 900.0) {
        scenario.elevation_m[scenario.Index(cell)] = 0.0F;
      }
    }
  }

  // A deterministic, visibly traversable lane makes the initial RViz request
  // repeatable while the rest of the 100 m scene remains randomly obstructed.
  for (std::size_t y = 25U; y <= 75U; ++y) {
    for (std::size_t x = 25U; x <= 175U; ++x) {
      const LunarSurfaceCell cell{x, y};
      scenario.occupancy[scenario.Index(cell)] = 0;
      scenario.elevation_m[scenario.Index(cell)] = 0.0F;
    }
  }
  scenario.default_goal_cell = LunarSurfaceCell{150U, 50U};

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
