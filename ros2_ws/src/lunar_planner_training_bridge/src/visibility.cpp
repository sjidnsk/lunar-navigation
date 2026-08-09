#include "lunar_planner_training_bridge/visibility.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <ranges>
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

[[nodiscard]] GridCell Add(const GridCell left, const GridCell right) noexcept {
  return GridCell{
      .row = static_cast<std::int32_t>(left.row + right.row),
      .column = static_cast<std::int32_t>(left.column + right.column),
  };
}

[[nodiscard]] std::vector<GridCell> Bresenham(const GridCell endpoint) {
  std::int64_t row = 0;
  std::int64_t column = 0;
  const std::int64_t delta_column =
      std::abs(static_cast<std::int64_t>(endpoint.column));
  const std::int64_t delta_row =
      std::abs(static_cast<std::int64_t>(endpoint.row));
  const std::int64_t step_column = endpoint.column > 0 ? 1 : -1;
  const std::int64_t step_row = endpoint.row > 0 ? 1 : -1;
  std::int64_t error = delta_column - delta_row;
  std::vector<GridCell> cells;
  cells.reserve(static_cast<std::size_t>(std::max(delta_column, delta_row)) +
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
                       const char *const name) {
  if (values.size() != expected_size) {
    throw std::invalid_argument(std::string{name} + " size mismatch");
  }
  if (!std::ranges::all_of(values, [](const float value) {
        return std::isfinite(value) && value >= 0.0F;
      })) {
    throw std::invalid_argument(std::string{name} + " values are invalid");
  }
}

} // namespace

VisibilityKernel::VisibilityKernel(const double resolution_m,
                                   const double range_m)
    : resolution_m_(resolution_m), range_m_(range_m) {
  if (!std::isfinite(resolution_m) || resolution_m <= 0.0 ||
      !std::isfinite(range_m) || range_m <= 0.0) {
    throw std::invalid_argument(
        "visibility geometry must be finite and positive");
  }
  const long double radius =
      std::floor(static_cast<long double>(range_m) / resolution_m);
  if (radius > std::numeric_limits<std::int32_t>::max()) {
    throw std::invalid_argument("visibility radius exceeds grid index domain");
  }
  radius_cells_ = static_cast<std::int32_t>(radius);
  const std::int64_t radius_cells = static_cast<std::int64_t>(radius_cells_);
  const long double range_squared = static_cast<long double>(range_m) * range_m;
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
  const std::size_t diameter =
      static_cast<std::size_t>(radius_cells_) * 2U + 1U;
  if (diameter > std::numeric_limits<std::size_t>::max() / diameter) {
    throw std::length_error("visibility offset lookup exceeds index domain");
  }
  const std::uint32_t missing = std::numeric_limits<std::uint32_t>::max();
  relative_cell_lookup_.assign(diameter * diameter, missing);
  for (std::size_t index = 0U; index < endpoint_offsets_.size(); ++index) {
    const GridCell cell = endpoint_offsets_[index];
    const std::size_t lookup_index =
        static_cast<std::size_t>(cell.row + radius_cells_) * diameter +
        static_cast<std::size_t>(cell.column + radius_cells_);
    if (index >= missing) {
      throw std::length_error("visibility endpoint count exceeds index domain");
    }
    relative_cell_lookup_[lookup_index] = static_cast<std::uint32_t>(index);
  }
  rays_.reserve(endpoint_offsets_.size());
  for (const GridCell endpoint : endpoint_offsets_) {
    const std::vector<GridCell> cells = Bresenham(endpoint);
    if (ray_cell_indices_.size() >= missing ||
        cells.size() - 1U > std::numeric_limits<std::uint16_t>::max()) {
      throw std::length_error("visibility ray storage exceeds index domain");
    }
    const std::uint32_t cell_offset =
        static_cast<std::uint32_t>(ray_cell_indices_.size());
    for (auto cell = std::next(cells.begin()); cell != cells.end(); ++cell) {
      const std::size_t lookup_index =
          static_cast<std::size_t>(cell->row + radius_cells_) * diameter +
          static_cast<std::size_t>(cell->column + radius_cells_);
      const std::uint32_t relative_index = relative_cell_lookup_[lookup_index];
      if (relative_index == missing) {
        throw std::logic_error("visibility ray cell is outside the disk");
      }
      ray_cell_indices_.push_back(relative_index);
    }
    rays_.push_back(Ray{
        .cell_offset = cell_offset,
        .cell_count = static_cast<std::uint32_t>(cells.size() - 1U),
    });
  }

  reverse_offsets_.resize(endpoint_offsets_.size() + 1U, 0U);
  for (const std::uint32_t cell_index : ray_cell_indices_) {
    if (reverse_offsets_[static_cast<std::size_t>(cell_index) + 1U] ==
        missing) {
      throw std::length_error("visibility reverse count exceeds index domain");
    }
    ++reverse_offsets_[static_cast<std::size_t>(cell_index) + 1U];
  }
  for (std::size_t index = 1U; index < reverse_offsets_.size(); ++index) {
    const std::uint64_t cumulative =
        static_cast<std::uint64_t>(reverse_offsets_[index - 1U]) +
        reverse_offsets_[index];
    if (cumulative >= missing) {
      throw std::length_error(
          "visibility reverse storage exceeds index domain");
    }
    reverse_offsets_[index] = static_cast<std::uint32_t>(cumulative);
  }
  occurrence_rays_.resize(ray_cell_indices_.size());
  occurrence_positions_.resize(ray_cell_indices_.size());
  std::vector<std::uint32_t> cursors(reverse_offsets_.begin(),
                                     std::prev(reverse_offsets_.end()));
  for (std::uint32_t ray_index = 0U; ray_index < rays_.size(); ++ray_index) {
    const Ray ray = rays_[ray_index];
    for (std::uint32_t position = 0U; position < ray.cell_count; ++position) {
      const std::uint32_t cell_index =
          ray_cell_indices_[ray.cell_offset + position];
      const std::uint32_t occurrence_index = cursors[cell_index]++;
      occurrence_rays_[occurrence_index] = ray_index;
      occurrence_positions_[occurrence_index] =
          static_cast<std::uint16_t>(position);
    }
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

  std::vector<std::uint8_t> potential_endpoint_mask(cell_count, 0U);
  std::vector<GridCell> potential_endpoints;
  for (std::size_t row = 0U; row < shape.height; ++row) {
    for (std::size_t column = 0U; column < shape.width; ++column) {
      const GridCell observed_cell{
          .row = static_cast<std::int32_t>(row),
          .column = static_cast<std::int32_t>(column),
      };
      const std::size_t observed_index = Index(shape, observed_cell);
      if (observed[observed_index] == 0U ||
          obstacle_ratio[observed_index] != 0.0F) {
        continue;
      }
      for (std::int32_t delta_row = -1; delta_row <= 1; ++delta_row) {
        for (std::int32_t delta_column = -1; delta_column <= 1;
             ++delta_column) {
          if (delta_row == 0 && delta_column == 0) {
            continue;
          }
          const GridCell endpoint{
              .row = observed_cell.row + delta_row,
              .column = observed_cell.column + delta_column,
          };
          if (!InBounds(shape, endpoint)) {
            continue;
          }
          const std::size_t endpoint_index = Index(shape, endpoint);
          if (observed[endpoint_index] == 0U &&
              (roi_ratio[endpoint_index] != 0.0F ||
               priority_weight[endpoint_index] != 0.0F) &&
              potential_endpoint_mask[endpoint_index] == 0U) {
            potential_endpoint_mask[endpoint_index] = 1U;
            potential_endpoints.push_back(endpoint);
          }
        }
      }
    }
  }
  std::ranges::sort(potential_endpoints, {}, [](const GridCell cell) {
    return std::pair{cell.row, cell.column};
  });

  std::vector<CandidateGain> output;
  output.reserve(candidates.size());
  const std::uint32_t missing = std::numeric_limits<std::uint32_t>::max();
  const std::size_t diameter =
      static_cast<std::size_t>(radius_cells_) * 2U + 1U;
  for (const GridCell candidate : candidates) {
    double roi_gain = 0.0;
    double priority_gain = 0.0;
    const std::size_t candidate_index = Index(shape, candidate);
    if (observed[candidate_index] == 0U ||
        obstacle_ratio[candidate_index] != 0.0F) {
      output.push_back(CandidateGain{});
      continue;
    }
    for (const GridCell endpoint : potential_endpoints) {
      const std::int64_t relative_row =
          static_cast<std::int64_t>(endpoint.row) - candidate.row;
      const std::int64_t relative_column =
          static_cast<std::int64_t>(endpoint.column) - candidate.column;
      if (std::abs(relative_row) > radius_cells_ ||
          std::abs(relative_column) > radius_cells_) {
        continue;
      }
      const std::size_t endpoint_index = Index(shape, endpoint);
      const std::size_t lookup_index =
          static_cast<std::size_t>(relative_row + radius_cells_) * diameter +
          static_cast<std::size_t>(relative_column + radius_cells_);
      const std::uint32_t ray_index = relative_cell_lookup_[lookup_index];
      if (ray_index == missing) {
        continue;
      }
      const Ray ray = rays_[ray_index];
      if (ray.cell_count > 1U) {
        const GridCell penultimate =
            Add(candidate,
                endpoint_offsets_[ray_cell_indices_[ray.cell_offset +
                                                    ray.cell_count - 2U]]);
        const std::size_t penultimate_index = Index(shape, penultimate);
        if (observed[penultimate_index] == 0U ||
            obstacle_ratio[penultimate_index] != 0.0F) {
          continue;
        }
      }
      bool clear = true;
      for (std::uint32_t position = 0U; clear && position + 1U < ray.cell_count;
           ++position) {
        const GridCell absolute = Add(
            candidate,
            endpoint_offsets_[ray_cell_indices_[ray.cell_offset + position]]);
        const std::size_t cell_index = Index(shape, absolute);
        clear =
            observed[cell_index] != 0U && obstacle_ratio[cell_index] == 0.0F;
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
  ValidateFloatGrid(truth_obstacle_ratio, cell_count, "truth obstacle ratio");
  if (!InBounds(shape, pose)) {
    throw std::out_of_range("visibility pose is outside the grid");
  }
  std::vector<std::uint8_t> visible(cell_count, 0U);
  const std::size_t pose_index = Index(shape, pose);
  visible[pose_index] = 1U;
  if (truth_obstacle_ratio[pose_index] > 0.0F) {
    return visible;
  }
  const bool full_disk_in_bounds =
      pose.row >= radius_cells_ && pose.column >= radius_cells_ &&
      static_cast<std::size_t>(pose.row + radius_cells_) < shape.height &&
      static_cast<std::size_t>(pose.column + radius_cells_) < shape.width;
  if (full_disk_in_bounds) {
    thread_local std::vector<std::uint32_t> nearest_obstacle;
    nearest_obstacle.assign(rays_.size(),
                            std::numeric_limits<std::uint32_t>::max());
    bool has_obstacle = false;
    for (std::size_t cell_index = 0U; cell_index < endpoint_offsets_.size();
         ++cell_index) {
      const std::size_t absolute_index =
          Index(shape, Add(pose, endpoint_offsets_[cell_index]));
      if (truth_obstacle_ratio[absolute_index] <= 0.0F) {
        continue;
      }
      has_obstacle = true;
      const std::uint32_t occurrence_begin = reverse_offsets_[cell_index];
      const std::uint32_t occurrence_end = reverse_offsets_[cell_index + 1U];
      for (std::uint32_t occurrence = occurrence_begin;
           occurrence < occurrence_end; ++occurrence) {
        const std::uint32_t ray_index = occurrence_rays_[occurrence];
        nearest_obstacle[ray_index] = std::min(
            nearest_obstacle[ray_index],
            static_cast<std::uint32_t>(occurrence_positions_[occurrence]));
      }
    }
    if (!has_obstacle) {
      for (const GridCell cell : endpoint_offsets_) {
        visible[Index(shape, Add(pose, cell))] = 1U;
      }
      return visible;
    }
    for (std::size_t cell_index = 0U; cell_index < endpoint_offsets_.size();
         ++cell_index) {
      const std::uint32_t occurrence_begin = reverse_offsets_[cell_index];
      const std::uint32_t occurrence_end = reverse_offsets_[cell_index + 1U];
      for (std::uint32_t occurrence = occurrence_begin;
           occurrence < occurrence_end; ++occurrence) {
        const std::uint32_t ray_index = occurrence_rays_[occurrence];
        const std::uint32_t position = occurrence_positions_[occurrence];
        if (position <= nearest_obstacle[ray_index]) {
          visible[Index(shape, Add(pose, endpoint_offsets_[cell_index]))] = 1U;
          break;
        }
      }
    }
    return visible;
  }
  for (std::size_t ray_index = 0U; ray_index < rays_.size(); ++ray_index) {
    if (!InBounds(shape, Add(pose, endpoint_offsets_[ray_index]))) {
      continue;
    }
    const Ray ray = rays_[ray_index];
    for (std::uint32_t position = 0U; position < ray.cell_count; ++position) {
      const GridCell absolute =
          Add(pose,
              endpoint_offsets_[ray_cell_indices_[ray.cell_offset + position]]);
      const std::size_t index = Index(shape, absolute);
      visible[index] = 1U;
      if (truth_obstacle_ratio[index] > 0.0F) {
        break;
      }
    }
  }
  return visible;
}

} // namespace lunar::planning::training
