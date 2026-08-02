#include "lunar_path_planner/v3/hopper/safe_pose_mask.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <ranges>
#include <string>
#include <tuple>

namespace lunar::planning::v3 {
namespace {

[[nodiscard]] Error Invalid(
    std::string path, std::string message) {
  return {
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool InBounds(
    const GridGeometry& geometry, const Cell cell) {
  return cell.x >= 0 && cell.y >= 0 &&
         static_cast<std::size_t>(cell.x) < geometry.width &&
         static_cast<std::size_t>(cell.y) < geometry.height;
}

[[nodiscard]] bool BaseCellSafe(
    const ImmutableMapSnapshot& map,
    const HopperCapabilityView& capability,
    const Cell cell) {
  const GridGeometry& geometry = map.geometry();
  if (!InBounds(geometry, cell)) {
    return false;
  }
  const std::size_t index = landing_cell_index(geometry, cell);
  if (map.KnownMask()[index] == 0U ||
      map.HardObstacleMask()[index] != 0U) {
    return false;
  }
  const auto normals = map.SurfaceNormals();
  const Eigen::Vector3d normal{
      normals.x[index], normals.y[index], normals.z[index]};
  if (!normal.allFinite() || normal.norm() <= 1.0e-12) {
    return false;
  }
  const double slope = std::acos(std::clamp(
      normal.normalized().z(), -1.0, 1.0));
  const auto& thresholds =
      capability.landing_terrain_thresholds;
  if (slope > thresholds.maximum_slope_rad + 1.0e-9 ||
      map.RoughnessMeters()[index] >
          thresholds.maximum_roughness_m + 1.0e-9) {
    return false;
  }
  const auto esdf = map.EsdfMeters();
  if (!esdf.empty()) {
    const double required = std::max(
        thresholds.minimum_overhead_clearance_m,
        thresholds.minimum_lateral_clearance_m);
    if (esdf[index] + 1.0e-9 < required) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::size_t landing_cell_index(
    const GridGeometry& geometry, const Cell cell) noexcept {
  return static_cast<std::size_t>(cell.y) * geometry.width +
         static_cast<std::size_t>(cell.x);
}

bool is_landing_cell_safe(
    const SafePoseMask& mask, const Cell cell) noexcept {
  if (cell.x < 0 || cell.y < 0 ||
      static_cast<std::size_t>(cell.x) >= mask.geometry.width ||
      static_cast<std::size_t>(cell.y) >= mask.geometry.height) {
    return false;
  }
  const std::size_t index =
      landing_cell_index(mask.geometry, cell);
  return index < mask.cells.size() &&
         mask.cells[index] == LandingCellState::kSafe;
}

Result<SafePoseMask> build_safe_pose_mask(
    const ImmutableMapSnapshot& map,
    const HopperCapabilityView& capability,
    const SweptFootprintEnvelope& footprint_envelope) {
  const GridGeometry& geometry = map.geometry();
  if (geometry.frame_id != capability.frame_id ||
      geometry.CellCount() == 0U ||
      !validate_convex_polygon(
           footprint_envelope.vertices_ccw).ok()) {
    return Invalid(
        "map", "INVALID_HOPPER_SAFE_POSE_INPUT");
  }
  SafePoseMask mask{
      .geometry = geometry,
      .cells =
          std::vector<LandingCellState>(
              geometry.CellCount(), LandingCellState::kUnsafe),
      .clearance_m =
          std::vector<double>(geometry.CellCount(), 0.0),
      .certified_yaw_interval =
          footprint_envelope.certified_yaw_interval,
      .source_snapshot_ref = map.snapshot_ref(),
  };

  double minimum_x = std::numeric_limits<double>::infinity();
  double maximum_x = -minimum_x;
  double minimum_y = minimum_x;
  double maximum_y = -minimum_x;
  for (const Eigen::Vector2d& vertex :
       footprint_envelope.vertices_ccw.vertices_ccw) {
    minimum_x = std::min(minimum_x, vertex.x());
    maximum_x = std::max(maximum_x, vertex.x());
    minimum_y = std::min(minimum_y, vertex.y());
    maximum_y = std::max(maximum_y, vertex.y());
  }
  const int minimum_dx = static_cast<int>(
      std::floor(minimum_x / geometry.resolution_m - 0.5));
  const int maximum_dx = static_cast<int>(
      std::ceil(maximum_x / geometry.resolution_m + 0.5));
  const int minimum_dy = static_cast<int>(
      std::floor(minimum_y / geometry.resolution_m - 0.5));
  const int maximum_dy = static_cast<int>(
      std::ceil(maximum_y / geometry.resolution_m + 0.5));

  for (std::size_t y = 0U; y < geometry.height; ++y) {
    for (std::size_t x = 0U; x < geometry.width; ++x) {
      const Cell center{
          static_cast<std::int32_t>(x),
          static_cast<std::int32_t>(y)};
      bool safe = true;
      for (int dy = minimum_dy;
           dy <= maximum_dy && safe; ++dy) {
        for (int dx = minimum_dx;
             dx <= maximum_dx; ++dx) {
          const Cell covered{
              center.x + dx, center.y + dy};
          if (!BaseCellSafe(map, capability, covered)) {
            safe = false;
            break;
          }
        }
      }
      if (safe) {
        mask.cells[landing_cell_index(geometry, center)] =
            LandingCellState::kSafe;
      }
    }
  }

  std::vector<Cell> unsafe;
  unsafe.reserve(geometry.CellCount());
  for (std::size_t y = 0U; y < geometry.height; ++y) {
    for (std::size_t x = 0U; x < geometry.width; ++x) {
      const Cell cell{
          static_cast<std::int32_t>(x),
          static_cast<std::int32_t>(y)};
      if (!is_landing_cell_safe(mask, cell)) {
        unsafe.push_back(cell);
      }
    }
  }
  for (std::size_t y = 0U; y < geometry.height; ++y) {
    for (std::size_t x = 0U; x < geometry.width; ++x) {
      const Cell cell{
          static_cast<std::int32_t>(x),
          static_cast<std::int32_t>(y)};
      if (!is_landing_cell_safe(mask, cell)) {
        continue;
      }
      double distance_cells = std::min(
          {static_cast<double>(x + 1U),
           static_cast<double>(y + 1U),
           static_cast<double>(geometry.width - x),
           static_cast<double>(geometry.height - y)});
      for (const Cell blocked : unsafe) {
        distance_cells = std::min(
            distance_cells,
            std::hypot(
                static_cast<double>(cell.x - blocked.x),
                static_cast<double>(cell.y - blocked.y)));
      }
      mask.clearance_m[
          landing_cell_index(geometry, cell)] =
          distance_cells * geometry.resolution_m;
    }
  }
  return mask;
}

std::vector<LandingSeed> select_landing_seeds(
    const SafePoseMask& mask,
    const std::size_t maximum_seed_count) {
  std::vector<LandingSeed> seeds;
  if (maximum_seed_count == 0U) {
    return seeds;
  }
  for (std::size_t y = 0U; y < mask.geometry.height; ++y) {
    for (std::size_t x = 0U; x < mask.geometry.width; ++x) {
      const Cell cell{
          static_cast<std::int32_t>(x),
          static_cast<std::int32_t>(y)};
      if (is_landing_cell_safe(mask, cell)) {
        const std::size_t index =
            landing_cell_index(mask.geometry, cell);
        seeds.push_back(
            {cell, mask.clearance_m[index],
             static_cast<std::uint64_t>(index)});
      }
    }
  }
  std::sort(
      seeds.begin(), seeds.end(),
      [](const LandingSeed& lhs, const LandingSeed& rhs) {
        return std::tuple{-lhs.clearance_m, lhs.stable_id} <
               std::tuple{-rhs.clearance_m, rhs.stable_id};
      });
  if (seeds.size() > maximum_seed_count) {
    seeds.resize(maximum_seed_count);
  }
  return seeds;
}

}  // namespace lunar::planning::v3
