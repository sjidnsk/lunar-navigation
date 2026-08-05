#include "hierarchical/landing_support_field.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stop_token>
#include <utility>
#include <vector>

namespace lunar::planning::hierarchical {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double kTolerance = 1.0e-9;

[[nodiscard]] double Elevation(const shared::MapSnapshot &map,
                               const shared::GridCell cell) noexcept {
  if (!map.InBounds(cell)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(map.FloatLayer("elevation")[map.Index(cell)]);
}

[[nodiscard]] double AxisGradient(const shared::MapSnapshot &map,
                                  const shared::GridCell cell,
                                  const std::int32_t dx,
                                  const std::int32_t dy) noexcept {
  const shared::GridCell negative{.x = cell.x - dx, .y = cell.y - dy};
  const shared::GridCell positive{.x = cell.x + dx, .y = cell.y + dy};
  const double center = Elevation(map, cell);
  if (map.InBounds(negative) && map.InBounds(positive)) {
    return (Elevation(map, positive) - Elevation(map, negative)) /
           (2.0 * map.resolution_m());
  }
  if (map.InBounds(positive)) {
    return (Elevation(map, positive) - center) / map.resolution_m();
  }
  if (map.InBounds(negative)) {
    return (center - Elevation(map, negative)) / map.resolution_m();
  }
  return 0.0;
}

[[nodiscard]] double PlaneResidual(const shared::SafeProjection &projection,
                                   const shared::GridCell cell) noexcept {
  if (projection.source_map() == nullptr) {
    return std::numeric_limits<double>::infinity();
  }
  const shared::MapSnapshot &map = *projection.source_map();
  const double center_elevation = Elevation(map, cell);
  const Vec3 center = map.CellCenter(cell);
  const double gradient_x = AxisGradient(map, cell, 1, 0);
  const double gradient_y = AxisGradient(map, cell, 0, 1);
  if (!std::isfinite(center_elevation) || !std::isfinite(gradient_x) ||
      !std::isfinite(gradient_y)) {
    return std::numeric_limits<double>::infinity();
  }
  double residual = 0.0;
  for (std::int32_t dy = -1; dy <= 1; ++dy) {
    for (std::int32_t dx = -1; dx <= 1; ++dx) {
      const shared::GridCell neighbor{.x = cell.x + dx, .y = cell.y + dy};
      if (!map.InBounds(neighbor)) {
        continue;
      }
      if (!projection.Known(neighbor) || !projection.HardFeasible(neighbor)) {
        return std::numeric_limits<double>::infinity();
      }
      const Vec3 point = map.CellCenter(neighbor);
      const double predicted = center_elevation +
                               gradient_x * (point.x - center.x) +
                               gradient_y * (point.y - center.y);
      residual =
          std::max(residual, std::abs(Elevation(map, neighbor) - predicted));
    }
  }
  return residual;
}

[[nodiscard]] bool
LandingBaseSafe(const shared::SafeProjection &projection,
                const shared::GridCell cell,
                const HopperCapability &capability) noexcept {
  if (projection.source_map() == nullptr || !projection.HardFeasible(cell)) {
    return false;
  }
  const double required_clearance =
      std::max(capability.minimum_landing_clearance_m,
               std::hypot(capability.body_half_extent_m.x,
                          capability.body_half_extent_m.y) +
                   capability.minimum_lateral_clearance_m);
  return static_cast<double>(projection.SlopeRadians(cell)) <=
             capability.maximum_landing_slope_rad + kTolerance &&
         static_cast<double>(projection.RoughnessMeters(cell)) <=
             capability.maximum_landing_roughness_m + kTolerance &&
         PlaneResidual(projection, cell) <=
             capability.maximum_plane_residual_m + kTolerance &&
         static_cast<double>(projection.ClearanceMeters(cell)) + kTolerance >=
             required_clearance;
}

void DistanceTransform1D(const std::vector<double> &input,
                         std::vector<double> &output,
                         std::vector<std::size_t> &sites,
                         std::vector<double> &boundaries) {
  const std::size_t count = input.size();
  output.assign(count, std::numeric_limits<double>::infinity());
  if (count == 0U) {
    return;
  }
  std::size_t first = 0U;
  while (first < count && !std::isfinite(input[first])) {
    ++first;
  }
  if (first == count) {
    return;
  }

  sites.resize(count);
  boundaries.resize(count + 1U);
  std::size_t envelope = 0U;
  sites[0] = first;
  boundaries[0] = -std::numeric_limits<double>::infinity();
  boundaries[1] = std::numeric_limits<double>::infinity();
  for (std::size_t q = first + 1U; q < count; ++q) {
    if (!std::isfinite(input[q])) {
      continue;
    }
    double intersection = 0.0;
    while (true) {
      const std::size_t site = sites[envelope];
      const double q_coordinate = static_cast<double>(q);
      const double site_coordinate = static_cast<double>(site);
      intersection = ((input[q] + q_coordinate * q_coordinate) -
                      (input[site] + site_coordinate * site_coordinate)) /
                     (2.0 * (q_coordinate - site_coordinate));
      if (intersection > boundaries[envelope] || envelope == 0U) {
        break;
      }
      --envelope;
    }
    ++envelope;
    sites[envelope] = q;
    boundaries[envelope] = intersection;
    boundaries[envelope + 1U] = std::numeric_limits<double>::infinity();
  }

  std::size_t selected = 0U;
  for (std::size_t q = 0U; q < count; ++q) {
    const double coordinate = static_cast<double>(q);
    while (selected < envelope && boundaries[selected + 1U] < coordinate) {
      ++selected;
    }
    const double delta = coordinate - static_cast<double>(sites[selected]);
    output[q] = delta * delta + input[sites[selected]];
  }
}

[[nodiscard]] bool SquaredEuclideanDistanceToUnsafe(
    const shared::MapSnapshot &map, const std::vector<std::uint8_t> &base_safe,
    std::vector<double> &distance, const std::stop_token stop_token) {
  const double infinity = std::numeric_limits<double>::infinity();
  std::vector<double> first_pass(map.cell_count(), infinity);
  std::vector<double> input;
  std::vector<double> output;
  std::vector<std::size_t> sites;
  std::vector<double> boundaries;
  input.reserve(std::max(map.width(), map.height()));

  for (std::size_t y = 0U; y < map.height(); ++y) {
    if (stop_token.stop_requested()) {
      return false;
    }
    input.resize(map.width());
    for (std::size_t x = 0U; x < map.width(); ++x) {
      input[x] = base_safe[y * map.width() + x] == 0U ? 0.0 : infinity;
    }
    DistanceTransform1D(input, output, sites, boundaries);
    for (std::size_t x = 0U; x < map.width(); ++x) {
      first_pass[y * map.width() + x] = output[x];
    }
  }

  distance.assign(map.cell_count(), infinity);
  for (std::size_t x = 0U; x < map.width(); ++x) {
    if (stop_token.stop_requested()) {
      return false;
    }
    input.resize(map.height());
    for (std::size_t y = 0U; y < map.height(); ++y) {
      input[y] = first_pass[y * map.width() + x];
    }
    DistanceTransform1D(input, output, sites, boundaries);
    for (std::size_t y = 0U; y < map.height(); ++y) {
      distance[y * map.width() + x] = output[y];
    }
  }
  return true;
}

[[nodiscard]] LandingSupportFieldBuildResult
Failure(const Clock::time_point started, std::string reason_code) {
  return LandingSupportFieldBuildResult{
      .field = std::nullopt,
      .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - started),
      .reason_code = std::move(reason_code),
  };
}

} // namespace

LandingSupportFieldBuildResult
BuildLandingSupportField(const shared::SafeProjection &projection,
                         const HopperCapability &capability,
                         const std::stop_token stop_token) {
  const Clock::time_point started = Clock::now();
  if (stop_token.stop_requested()) {
    return Failure(started, "REQUEST_CANCELED");
  }
  if (projection.source_map() == nullptr ||
      !std::isfinite(capability.minimum_landing_region_area_m2) ||
      capability.minimum_landing_region_area_m2 <= 0.0) {
    return Failure(started, "HOPPER_GLOBAL_CONFIGURATION_INVALID");
  }

  LandingSupportField field;
  field.map_ = projection.source_map();
  field.required_radius_m_ =
      std::sqrt(capability.minimum_landing_region_area_m2 / std::numbers::pi);
  if (!std::isfinite(field.required_radius_m_)) {
    return Failure(started, "HOPPER_GLOBAL_CONFIGURATION_INVALID");
  }

  field.base_safe_.assign(field.map_->cell_count(), 0U);
  for (std::size_t index = 0U; index < field.map_->cell_count(); ++index) {
    if (stop_token.stop_requested()) {
      return Failure(started, "REQUEST_CANCELED");
    }
    const shared::GridCell cell{
        .x = static_cast<std::int32_t>(index % field.map_->width()),
        .y = static_cast<std::int32_t>(index / field.map_->width()),
    };
    field.base_safe_[index] = static_cast<std::uint8_t>(
        LandingBaseSafe(projection, cell, capability));
  }
  if (!SquaredEuclideanDistanceToUnsafe(*field.map_, field.base_safe_,
                                        field.squared_distance_cells_,
                                        stop_token)) {
    return Failure(started, "REQUEST_CANCELED");
  }

  const double resolution = field.map_->resolution_m();
  const double half_diagonal = resolution * std::numbers::sqrt2 / 2.0;
  field.center_safe_.assign(field.map_->cell_count(), 0U);
  for (std::size_t index = 0U; index < field.map_->cell_count(); ++index) {
    if (stop_token.stop_requested()) {
      return Failure(started, "REQUEST_CANCELED");
    }
    if (field.base_safe_[index] == 0U) {
      continue;
    }
    const std::size_t x = index % field.map_->width();
    const std::size_t y = index / field.map_->width();
    const double boundary_distance_m = std::min(
        {(static_cast<double>(x) + 0.5) * resolution,
         (static_cast<double>(y) + 0.5) * resolution,
         (static_cast<double>(field.map_->width() - x) - 0.5) * resolution,
         (static_cast<double>(field.map_->height() - y) - 0.5) * resolution});
    double unsafe_distance_m = std::numeric_limits<double>::infinity();
    if (std::isfinite(field.squared_distance_cells_[index])) {
      unsafe_distance_m = std::max(
          0.0, std::sqrt(field.squared_distance_cells_[index]) * resolution -
                   half_diagonal);
    }
    const double support_radius_m =
        std::min(boundary_distance_m, unsafe_distance_m);
    field.center_safe_[index] = static_cast<std::uint8_t>(
        support_radius_m + kTolerance >= field.required_radius_m_);
  }

  return LandingSupportFieldBuildResult{
      .field = std::move(field),
      .elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now() - started),
      .reason_code = {},
  };
}

bool LandingSupportField::BaseSafe(const shared::GridCell cell) const noexcept {
  return map_ != nullptr && map_->InBounds(cell) &&
         base_safe_[map_->Index(cell)] != 0U;
}

bool LandingSupportField::CenterSafe(
    const shared::GridCell cell) const noexcept {
  return map_ != nullptr && map_->InBounds(cell) &&
         center_safe_[map_->Index(cell)] != 0U;
}

double LandingSupportField::RequiredRadiusMeters() const noexcept {
  return required_radius_m_;
}

std::size_t LandingSupportField::EstimatedWorkMemoryBytes() const noexcept {
  if (map_ == nullptr) {
    return 0U;
  }
  const std::size_t maximum_axis = std::max(map_->width(), map_->height());
  return base_safe_.capacity() * sizeof(std::uint8_t) +
         center_safe_.capacity() * sizeof(std::uint8_t) +
         squared_distance_cells_.capacity() * sizeof(double) +
         map_->cell_count() * sizeof(double) +
         maximum_axis *
             (2U * sizeof(double) + sizeof(std::size_t) + sizeof(double));
}

} // namespace lunar::planning::hierarchical
