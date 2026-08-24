#include "lunar_pure_planner_ros/center_distance_transform.hpp"

#include <cmath>
#include <limits>
#include <optional>

namespace lunar::pure_planner_ros {
namespace {

[[nodiscard]] std::optional<std::vector<double>> TransformLine(
    const std::span<const double> input) {
  const double infinity = std::numeric_limits<double>::infinity();
  std::optional<std::size_t> first_seed;
  for (std::size_t index = 0U; index < input.size(); ++index) {
    if (std::isfinite(input[index])) {
      first_seed = index;
      break;
    }
  }
  if (!first_seed.has_value()) {
    return std::vector<double>(input.size(), infinity);
  }

  std::vector<std::size_t> sites(input.size());
  std::vector<double> boundaries(input.size() + 1U, infinity);
  std::size_t envelope_size = 1U;
  sites[0] = *first_seed;
  boundaries[0] = -infinity;
  boundaries[1] = infinity;
  for (std::size_t candidate = *first_seed + 1U;
       candidate < input.size(); ++candidate) {
    if (!std::isfinite(input[candidate])) continue;
    double boundary{};
    while (envelope_size > 0U) {
      const std::size_t previous = sites[envelope_size - 1U];
      const double candidate_coordinate = static_cast<double>(candidate);
      const double previous_coordinate = static_cast<double>(previous);
      boundary = ((input[candidate] + candidate_coordinate * candidate_coordinate) -
                  (input[previous] + previous_coordinate * previous_coordinate)) /
                 (2.0 * (candidate_coordinate - previous_coordinate));
      if (boundary > boundaries[envelope_size - 1U]) break;
      --envelope_size;
    }
    if (envelope_size == 0U) {
      sites[0] = candidate;
      boundaries[0] = -infinity;
      boundaries[1] = infinity;
      envelope_size = 1U;
      continue;
    }
    sites[envelope_size] = candidate;
    boundaries[envelope_size] = boundary;
    boundaries[envelope_size + 1U] = infinity;
    ++envelope_size;
  }

  std::vector<double> result(input.size());
  std::size_t envelope_index{};
  for (std::size_t coordinate = 0U; coordinate < input.size(); ++coordinate) {
    while (envelope_index + 1U < envelope_size &&
           boundaries[envelope_index + 1U] <= static_cast<double>(coordinate)) {
      ++envelope_index;
    }
    const double delta = static_cast<double>(coordinate) -
                         static_cast<double>(sites[envelope_index]);
    result[coordinate] = delta * delta + input[sites[envelope_index]];
  }
  return result;
}

}  // namespace

CenterDistanceResult BuildCenterSquaredDistance(
    const std::size_t width, const std::size_t height,
    const std::span<const std::uint8_t> seeds) {
  if (width == 0U || height == 0U ||
      height > std::numeric_limits<std::size_t>::max() / width ||
      seeds.size() != width * height) {
    return {.squared_cells = {}, .reason_code = "CENTER_DISTANCE_INVALID"};
  }
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> temporary(width * height, infinity);
  std::vector<double> line(width, infinity);
  for (std::size_t y = 0U; y < height; ++y) {
    for (std::size_t x = 0U; x < width; ++x) {
      line[x] = seeds[y * width + x] == 0U ? infinity : 0.0;
    }
    const auto transformed = TransformLine(line);
    if (!transformed.has_value()) {
      return {.squared_cells = {}, .reason_code = "CENTER_DISTANCE_INVALID"};
    }
    for (std::size_t x = 0U; x < width; ++x) {
      temporary[y * width + x] = (*transformed)[x];
    }
  }

  CenterDistanceResult result;
  result.squared_cells.resize(width * height, infinity);
  line.resize(height);
  for (std::size_t x = 0U; x < width; ++x) {
    for (std::size_t y = 0U; y < height; ++y) line[y] = temporary[y * width + x];
    const auto transformed = TransformLine(line);
    if (!transformed.has_value()) {
      return {.squared_cells = {}, .reason_code = "CENTER_DISTANCE_INVALID"};
    }
    for (std::size_t y = 0U; y < height; ++y) {
      result.squared_cells[y * width + x] = (*transformed)[y];
    }
  }
  return result;
}

}  // namespace lunar::pure_planner_ros
