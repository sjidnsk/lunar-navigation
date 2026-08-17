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

  bool operator==(const GridCell &) const = default;
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
      std::span<const float> obstacle_ratio, std::span<const float> roi_ratio,
      std::span<const float> priority_weight,
      std::span<const GridCell> candidates) const;

  [[nodiscard]] std::vector<std::uint8_t>
  RevealFromPose(GridShape shape, GridCell pose,
                 std::span<const float> truth_obstacle_ratio) const;

  [[nodiscard]] std::vector<std::uint8_t>
  RevealFromPoses(GridShape shape, std::span<const GridCell> poses,
                  std::span<const float> truth_obstacle_ratios) const;

private:
  struct Ray final {
    std::uint32_t cell_offset{};
    std::uint32_t cell_count{};
  };

  double resolution_m_{};
  double range_m_{};
  std::int32_t radius_cells_{};
  std::vector<GridCell> endpoint_offsets_;
  std::vector<std::uint32_t> relative_cell_lookup_;
  std::vector<Ray> rays_;
  std::vector<std::uint32_t> ray_cell_indices_;
  std::vector<std::uint32_t> reverse_offsets_;
  std::vector<std::uint32_t> occurrence_rays_;
  std::vector<std::uint16_t> occurrence_positions_;

  void RevealFromPoseInto(GridShape shape, GridCell pose,
                          std::span<const float> truth_obstacle_ratio,
                          std::span<std::uint8_t> visible) const;
};

} // namespace lunar::planning::training
