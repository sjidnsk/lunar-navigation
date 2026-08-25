#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lunar::pure_planner_ros {

struct LunarSurfaceCell final {
  std::size_t x{};
  std::size_t y{};

  [[nodiscard]] friend bool operator==(const LunarSurfaceCell&, const LunarSurfaceCell&) = default;
};

struct LunarSurfaceScenario final {
  std::size_t width{1000U};
  std::size_t height{1000U};
  double resolution_m{1.0};
  double origin_x_m{-500.0};
  double origin_y_m{-500.0};
  std::vector<std::int8_t> occupancy;
  std::vector<float> elevation_m;
  LunarSurfaceCell start_cell{150U, 500U};
  LunarSurfaceCell default_goal_cell{};

  [[nodiscard]] bool InBounds(LunarSurfaceCell cell) const noexcept;
  [[nodiscard]] std::size_t Index(LunarSurfaceCell cell) const noexcept;
  [[nodiscard]] bool Occupied(LunarSurfaceCell cell) const noexcept;
};

[[nodiscard]] LunarSurfaceScenario BuildLunarSurfaceScenario(std::uint32_t seed);
[[nodiscard]] bool CellsConnected(const LunarSurfaceScenario& scenario,
                                  LunarSurfaceCell start,
                                  LunarSurfaceCell goal);

}  // namespace lunar::pure_planner_ros
