#include "lunar_planner_training_bridge/visibility.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lunar::planning::training {
namespace {

[[nodiscard]] std::size_t CheckedCellCount(const GridShape shape) {
  if (shape.height == 0U || shape.width == 0U ||
      shape.height > std::numeric_limits<std::size_t>::max() / shape.width) {
    throw std::invalid_argument("visibility grid shape is invalid");
  }
  return shape.height * shape.width;
}

[[nodiscard]] bool InBounds(const GridShape shape,
                            const GridCell cell) noexcept {
  return cell.row >= 0 && cell.column >= 0 &&
         static_cast<std::size_t>(cell.row) < shape.height &&
         static_cast<std::size_t>(cell.column) < shape.width;
}

[[nodiscard]] std::size_t Index(const GridShape shape,
                                const GridCell cell) noexcept {
  return static_cast<std::size_t>(cell.row) * shape.width +
         static_cast<std::size_t>(cell.column);
}

[[nodiscard]] GridCell Add(const GridCell left,
                           const GridCell right) noexcept {
  return GridCell{
      .row = static_cast<std::int32_t>(left.row + right.row),
      .column = static_cast<std::int32_t>(left.column + right.column),
  };
}

[[nodiscard]] std::vector<GridCell> Bresenham(const GridCell endpoint) {
  std::int64_t row = 0;
  std::int64_t column = 0;
  const std::int64_t delta_column = std::abs(
      static_cast<std::int64_t>(endpoint.column));
  const std::int64_t delta_row =
      std::abs(static_cast<std::int64_t>(endpoint.row));
  const std::int64_t step_column = endpoint.column > 0 ? 1 : -1;
  const std::int64_t step_row = endpoint.row > 0 ? 1 : -1;
  std::int64_t error = delta_column - delta_row;
  std::vector<GridCell> cells;
  cells.reserve(static_cast<std::size_t>(
                    std::max(delta_column, delta_row)) +
                1U);
  while (true) {
    cells.push_back(GridCell{
        .row = static_cast<std::int32_t>(row),
        .column = static_cast<std::int32_t>(column),
    });
    if (row == endpoint.row && column == endpoint.column) {
      return cells;
    }
    const std::int64_t doubled = 2 * error;
    if (doubled > -delta_row) {
      error -= delta_row;
      column += step_column;
    }
    if (doubled < delta_column) {
      error += delta_column;
      row += step_row;
    }
  }
}

void ValidateFloatGrid(const std::span<const float> values,
                       const std::size_t expected_size,
                       const char* const name) {
  if (values.size() != expected_size) {
    throw std::invalid_argument(std::string{name} + " size mismatch");
  }
  if (!std::ranges::all_of(values, [](const float value) {
        return std::isfinite(value) && value >= 0.0F;
      })) {
    throw std::invalid_argument(std::string{name} + " values are invalid");
  }
}

}  // namespace

VisibilityKernel::VisibilityKernel(const double resolution_m,
                                   const double range_m)
    : resolution_m_(resolution_m), range_m_(range_m) {
  if (!std::isfinite(resolution_m) || resolution_m <= 0.0 ||
      !std::isfinite(range_m) || range_m <= 0.0) {
    throw std::invalid_argument("visibility geometry must be finite and positive");
  }
  const long double radius =
      std::floor(static_cast<long double>(range_m) / resolution_m);
  if (radius > std::numeric_limits<std::int32_t>::max()) {
    throw std::invalid_argument("visibility radius exceeds grid index domain");
  }
  const std::int64_t radius_cells = static_cast<std::int64_t>(radius);
  const long double range_squared =
      static_cast<long double>(range_m) * range_m;
  const long double resolution_squared =
      static_cast<long double>(resolution_m) * resolution_m;
  for (std::int64_t row = -radius_cells; row <= radius_cells; ++row) {
    for (std::int64_t column = -radius_cells; column <= radius_cells;
         ++column) {
      if (row == 0 && column == 0) {
        continue;
      }
      const long double cell_distance_squared =
          static_cast<long double>(row) * row +
          static_cast<long double>(column) * column;
      if (cell_distance_squared * resolution_squared > range_squared) {
        continue;
      }
      endpoint_offsets_.push_back(GridCell{
          .row = static_cast<std::int32_t>(row),
          .column = static_cast<std::int32_t>(column),
      });
    }
  }
  rays_.reserve(endpoint_offsets_.size());
  for (const GridCell endpoint : endpoint_offsets_) {
    rays_.push_back(Ray{
        .endpoint = endpoint,
        .cells = Bresenham(endpoint),
    });
  }
}

std::vector<CandidateGain> VisibilityKernel::EstimateCandidateGains(
    const GridShape shape, const std::span<const std::uint8_t> observed,
    const std::span<const float> obstacle_ratio,
    const std::span<const float> roi_ratio,
    const std::span<const float> priority_weight,
    const std::span<const GridCell> candidates) const {
  const std::size_t cell_count = CheckedCellCount(shape);
  if (observed.size() != cell_count) {
    throw std::invalid_argument("observed mask size mismatch");
  }
  if (!std::ranges::all_of(observed,
                           [](const auto value) { return value <= 1U; })) {
    throw std::invalid_argument("observed mask values are invalid");
  }
  ValidateFloatGrid(obstacle_ratio, cell_count, "obstacle ratio");
  ValidateFloatGrid(roi_ratio, cell_count, "ROI ratio");
  ValidateFloatGrid(priority_weight, cell_count, "priority weight");
  for (const GridCell candidate : candidates) {
    if (!InBounds(shape, candidate)) {
      throw std::out_of_range("visibility candidate is outside the grid");
    }
  }

  std::vector<CandidateGain> output;
  output.reserve(candidates.size());
  for (const GridCell candidate : candidates) {
    double roi_gain = 0.0;
    double priority_gain = 0.0;
    for (const Ray& ray : rays_) {
      const GridCell endpoint = Add(candidate, ray.endpoint);
      if (!InBounds(shape, endpoint)) {
        continue;
      }
      const std::size_t endpoint_index = Index(shape, endpoint);
      if (observed[endpoint_index] != 0U) {
        continue;
      }
      bool clear = true;
      for (auto cell = ray.cells.begin(); cell != std::prev(ray.cells.end());
           ++cell) {
        const GridCell absolute = Add(candidate, *cell);
        const std::size_t index = Index(shape, absolute);
        if (observed[index] == 0U || obstacle_ratio[index] > 0.0F) {
          clear = false;
          break;
        }
      }
      if (clear) {
        roi_gain += roi_ratio[endpoint_index];
        priority_gain += priority_weight[endpoint_index];
      }
    }
    output.push_back(CandidateGain{
        .roi = static_cast<float>(roi_gain),
        .priority = static_cast<float>(priority_gain),
    });
  }
  return output;
}

std::vector<std::uint8_t> VisibilityKernel::RevealFromPose(
    const GridShape shape, const GridCell pose,
    const std::span<const float> truth_obstacle_ratio) const {
  const std::size_t cell_count = CheckedCellCount(shape);
  ValidateFloatGrid(
      truth_obstacle_ratio, cell_count, "truth obstacle ratio");
  if (!InBounds(shape, pose)) {
    throw std::out_of_range("visibility pose is outside the grid");
  }
  std::vector<std::uint8_t> visible(cell_count, 0U);
  visible[Index(shape, pose)] = 1U;
  for (const Ray& ray : rays_) {
    if (!InBounds(shape, Add(pose, ray.endpoint))) {
      continue;
    }
    for (const GridCell relative : ray.cells) {
      const GridCell absolute = Add(pose, relative);
      const std::size_t index = Index(shape, absolute);
      visible[index] = 1U;
      if (truth_obstacle_ratio[index] > 0.0F) {
        break;
      }
    }
  }
  return visible;
}

}  // namespace lunar::planning::training
