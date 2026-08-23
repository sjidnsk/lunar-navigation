#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace lunar::pure_exploration_sim {

struct TruthSample {
  bool occupied{false};
  std::uint8_t semantic_id{0U};
  double elevation_m{0.0};
  double roughness{0.0};
};

class LunarScene;

[[nodiscard]] LunarScene BuildLunarScene(std::uint32_t seed);

class LunarScene {
 public:
  [[nodiscard]] double min_x_m() const noexcept;
  [[nodiscard]] double max_x_m() const noexcept;
  [[nodiscard]] std::size_t global_width() const noexcept;
  [[nodiscard]] std::size_t global_height() const noexcept;
  [[nodiscard]] double global_resolution_m() const noexcept;
  [[nodiscard]] const std::vector<std::int8_t>& GlobalOccupancy() const noexcept;
  [[nodiscard]] bool IsOccupied(double world_x_m, double world_y_m) const;
  [[nodiscard]] TruthSample Sample(double world_x_m, double world_y_m) const;

 private:
  struct Rock {
    double x_m;
    double y_m;
    double radius_m;
    double height_m;
  };

  struct Crater {
    double x_m;
    double y_m;
    double radius_m;
    double rim_half_width_m;
    double depth_m;
    double rim_height_m;
  };

  struct SurfaceSample {
    bool occupied;
    std::uint8_t semantic_id;
    double elevation_m;
  };

  explicit LunarScene(std::uint32_t seed);

  [[nodiscard]] SurfaceSample EvaluateSurface(double world_x_m,
                                              double world_y_m) const;
  [[nodiscard]] static bool IsReservedFree(double world_x_m,
                                           double world_y_m) noexcept;
  void BuildSpatialIndex();
  [[nodiscard]] std::optional<std::size_t> SpatialBucketIndex(
      double world_x_m, double world_y_m) const noexcept;

  std::vector<Rock> rocks_;
  std::vector<Crater> craters_;
  std::vector<std::vector<std::uint16_t>> rock_buckets_;
  std::vector<std::vector<std::uint16_t>> crater_buckets_;
  std::vector<std::int8_t> global_occupancy_;

  friend LunarScene BuildLunarScene(std::uint32_t seed);
};

}  // namespace lunar::pure_exploration_sim
