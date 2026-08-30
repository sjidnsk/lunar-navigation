#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace lunar::pure_planner_ros {

struct LunarSurfaceCell final {
  std::size_t x{};
  std::size_t y{};

  [[nodiscard]] friend bool operator==(const LunarSurfaceCell&, const LunarSurfaceCell&) = default;
};

struct LunarSurfaceSample final {
  bool occupied{};
  float elevation_m{};
};

struct LunarSurfaceCrater final {
  double center_x_m{};
  double center_y_m{};
  double radius_m{};
};

struct LunarSurfaceRock final {
  double center_x_m{};
  double center_y_m{};
  double radius_m{1.0};
};

struct LunarSurfaceScenario final {
  std::size_t width{1000U};
  std::size_t height{1000U};
  double resolution_m{1.0};
  double origin_x_m{-500.0};
  double origin_y_m{-500.0};
  std::vector<std::int8_t> occupancy;
  std::vector<float> elevation_m;
  std::vector<LunarSurfaceCrater> craters;
  std::vector<LunarSurfaceRock> rocks;
  LunarSurfaceCell start_cell{150U, 500U};
  LunarSurfaceCell default_goal_cell{};

  [[nodiscard]] bool InBounds(LunarSurfaceCell cell) const noexcept;
  [[nodiscard]] std::size_t Index(LunarSurfaceCell cell) const noexcept;
  [[nodiscard]] bool Occupied(LunarSurfaceCell cell) const noexcept;
  [[nodiscard]] std::optional<LunarSurfaceSample> Sample(
      double x_m, double y_m) const noexcept;
};

[[nodiscard]] LunarSurfaceScenario BuildLunarSurfaceScenario(std::uint32_t seed);
[[nodiscard]] bool CellsConnected(const LunarSurfaceScenario& scenario,
                                  LunarSurfaceCell start,
                                  LunarSurfaceCell goal);

}  // namespace lunar::pure_planner_ros
