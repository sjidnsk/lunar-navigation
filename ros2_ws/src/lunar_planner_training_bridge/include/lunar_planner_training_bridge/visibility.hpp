#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lunar::planning::training {

struct GridShape final {
  std::size_t height{};
  std::size_t width{};
};

struct GridCell final {
  std::int32_t row{};
  std::int32_t column{};

  bool operator==(const GridCell&) const = default;
};

struct CandidateGain final {
  float roi{};
  float priority{};
};

class VisibilityKernel final {
 public:
  VisibilityKernel(double resolution_m, double range_m);

  [[nodiscard]] double resolution_m() const noexcept { return resolution_m_; }
  [[nodiscard]] double range_m() const noexcept { return range_m_; }
  [[nodiscard]] std::span<const GridCell> endpoint_offsets() const noexcept {
    return endpoint_offsets_;
  }

  [[nodiscard]] std::vector<CandidateGain> EstimateCandidateGains(
      GridShape shape, std::span<const std::uint8_t> observed,
      std::span<const float> obstacle_ratio,
      std::span<const float> roi_ratio,
      std::span<const float> priority_weight,
      std::span<const GridCell> candidates) const;

  [[nodiscard]] std::vector<std::uint8_t> RevealFromPose(
      GridShape shape, GridCell pose,
      std::span<const float> truth_obstacle_ratio) const;

 private:
  struct Ray final {
    GridCell endpoint;
    std::vector<GridCell> cells;
  };

  double resolution_m_{};
  double range_m_{};
  std::vector<GridCell> endpoint_offsets_;
  std::vector<Ray> rays_;
};

}  // namespace lunar::planning::training
