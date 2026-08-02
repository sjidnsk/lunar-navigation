#include "lunar_path_planner/v3/hopper/flight_tube_certifier.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "lunar_path_planner/v3/hopper/landing_geometry.hpp"

namespace lunar::planning::v3 {
namespace {

struct Bounds3 final {
  Eigen::Vector3d minimum{
      Eigen::Vector3d::Constant(
          std::numeric_limits<double>::infinity())};
  Eigen::Vector3d maximum{
      Eigen::Vector3d::Constant(
          -std::numeric_limits<double>::infinity())};
};

[[nodiscard]] Eigen::Vector3d ToEigen(const Vec3 value) {
  return {value.x, value.y, value.z};
}

[[nodiscard]] Eigen::Vector3d BoxRadius(
    const AxisAlignedBox3& box) {
  return ToEigen(box.center).cwiseAbs() +
         ToEigen(box.half_extent);
}

[[nodiscard]] bool ValidBox(const AxisAlignedBox3& box) {
  const Eigen::Vector3d center = ToEigen(box.center);
  const Eigen::Vector3d half = ToEigen(box.half_extent);
  return center.allFinite() && half.allFinite() &&
         (half.array() >= 0.0).all();
}

[[nodiscard]] std::optional<Bounds3> AxisAlignedBodyBounds(
    const ConvexPolytope3& polytope) {
  Bounds3 bounds{};
  bounds.minimum = Eigen::Vector3d::Constant(
      -std::numeric_limits<double>::infinity());
  bounds.maximum = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::infinity());
  std::array<bool, 3> has_min{false, false, false};
  std::array<bool, 3> has_max{false, false, false};
  for (const Halfspace3& halfspace : polytope.halfspaces) {
    const Eigen::Vector3d normal = ToEigen(halfspace.normal);
    if (!normal.allFinite() ||
        !std::isfinite(halfspace.offset_m)) {
      return std::nullopt;
    }
    for (Eigen::Index axis = 0; axis < 3; ++axis) {
      bool is_axis = true;
      for (Eigen::Index other = 0; other < 3; ++other) {
        if (other != axis &&
            std::abs(normal[other]) > 1.0e-12) {
          is_axis = false;
        }
      }
      if (!is_axis || std::abs(normal[axis]) <= 1.0e-12) {
        continue;
      }
      const double bound =
          halfspace.offset_m / normal[axis];
      if (normal[axis] > 0.0) {
        bounds.maximum[axis] =
            std::min(bounds.maximum[axis], bound);
        has_max[static_cast<std::size_t>(axis)] = true;
      } else {
        bounds.minimum[axis] =
            std::max(bounds.minimum[axis], bound);
        has_min[static_cast<std::size_t>(axis)] = true;
      }
    }
  }
  for (std::size_t axis = 0U; axis < 3U; ++axis) {
    if (!has_min[axis] || !has_max[axis] ||
        bounds.minimum[static_cast<Eigen::Index>(axis)] >
            bounds.maximum[static_cast<Eigen::Index>(axis)]) {
      return std::nullopt;
    }
  }
  return bounds;
}

[[nodiscard]] std::pair<double, double> CoordinateRange(
    const NominalBallisticArc& arc, const Eigen::Index axis,
    const double begin, const double end) {
  auto value = [&arc, axis](const double time) {
    return arc.launch_position_m[axis] +
           arc.launch_velocity_mps[axis] * time +
           0.5 * arc.gravity_mps2[axis] * time * time;
  };
  double minimum = std::min(value(begin), value(end));
  double maximum = std::max(value(begin), value(end));
  if (std::abs(arc.gravity_mps2[axis]) > 1.0e-15) {
    const double stationary =
        -arc.launch_velocity_mps[axis] /
        arc.gravity_mps2[axis];
    if (stationary > begin && stationary < end) {
      minimum = std::min(minimum, value(stationary));
      maximum = std::max(maximum, value(stationary));
    }
  }
  return {minimum, maximum};
}

[[nodiscard]] ConvexPolytope3 MakeBoundsPolytope(
    const Bounds3& bounds) {
  return {
      .halfspaces =
          {
              {{1.0, 0.0, 0.0}, bounds.maximum.x()},
              {{-1.0, 0.0, 0.0}, -bounds.minimum.x()},
              {{0.0, 1.0, 0.0}, bounds.maximum.y()},
              {{0.0, -1.0, 0.0}, -bounds.minimum.y()},
              {{0.0, 0.0, 1.0}, bounds.maximum.z()},
              {{0.0, 0.0, -1.0}, -bounds.minimum.z()},
          },
  };
}

[[nodiscard]] DurationNanoseconds FloorDuration(
    const double seconds) {
  return DurationNanoseconds{
      std::chrono::nanoseconds{
          static_cast<std::int64_t>(
              std::floor(seconds * 1.0e9))}};
}

[[nodiscard]] DurationNanoseconds CeilDuration(
    const double seconds) {
  return DurationNanoseconds{
      std::chrono::nanoseconds{
          static_cast<std::int64_t>(
              std::ceil(seconds * 1.0e9))}};
}

[[nodiscard]] bool CellBelongsToRegion(
    const Cell cell, const GridGeometry& geometry,
    const TerrainCertifiedLandingRegion& region) {
  const Eigen::Vector3d point{
      geometry.origin_m.x +
          (static_cast<double>(cell.x) + 0.5) *
              geometry.resolution_m,
      geometry.origin_m.y +
          (static_cast<double>(cell.y) + 0.5) *
              geometry.resolution_m,
      region.landing_plane.origin_m.z};
  const Eigen::Vector2d uv =
      world_to_plane_uv(region.landing_plane, point);
  return point_in_convex_polygon(
      uv, region.vertices_uv_ccw, 0.0);
}

}  // namespace

FlightTubeCertifier::FlightTubeCertifier(
    HopperCapabilityView capability,
    HopperPlannerLimits limits)
    : capability_(std::move(capability)),
      limits_(std::move(limits)) {}

FlightTubeCertificationResult FlightTubeCertifier::certify(
    const FlightTubeCertificationInput& input) const {
  FlightTubeCertificationResult result{};
  result.diagnostics.minimum_clearance_m =
      std::numeric_limits<double>::infinity();
  if (input.map == nullptr) {
    result.diagnostics.rejection_reason = "missing_map";
    return result;
  }
  if (input.source_snapshot_ref != input.map->snapshot_ref() ||
      input.arc.frame_id != input.map->frame_id() ||
      input.source_region.frame_id != input.arc.frame_id ||
      input.target_region.frame_id != input.arc.frame_id ||
      input.body_rotation_envelope_ref !=
          capability_.body_rotation_envelope.content_ref ||
      input.error_model_ref !=
          capability_.error_model.content_ref) {
    result.diagnostics.rejection_reason =
        "certification_source_mismatch";
    return result;
  }
  if (!std::isfinite(input.arc.flight_time_s) ||
      input.arc.flight_time_s <= 0.0 ||
      !ValidBox(input.initial_position_error_m) ||
      !ValidBox(input.initial_velocity_error_mps) ||
      !ValidBox(input.launch_execution_velocity_error_mps) ||
      !ValidBox(input.gravity_error_mps2)) {
    result.diagnostics.rejection_reason = "invalid_input";
    return result;
  }
  const auto body_bounds = AxisAlignedBodyBounds(
      input.arbitrary_attitude_body_envelope);
  if (!body_bounds.has_value()) {
    result.diagnostics.rejection_reason =
        "unsupported_body_rotation_envelope";
    return result;
  }
  if (limits_.maximum_flight_tube_sections == 0U) {
    result.diagnostics.rejection_reason =
        "flight_tube_section_resource_exhausted";
    return result;
  }

  const std::size_t section_count =
      std::min<std::size_t>(
          16U, limits_.maximum_flight_tube_sections);
  const GridGeometry& geometry = input.map->geometry();
  const auto known = input.map->KnownMask();
  const auto obstacles = input.map->HardObstacleMask();
  const auto elevations = input.map->ElevationMeters();
  std::vector<FlightTubeSection> sections;
  sections.reserve(section_count);
  const Eigen::Vector3d position_radius =
      BoxRadius(input.initial_position_error_m);
  const Eigen::Vector3d velocity_radius =
      BoxRadius(input.initial_velocity_error_mps) +
      BoxRadius(input.launch_execution_velocity_error_mps);
  const Eigen::Vector3d gravity_radius =
      BoxRadius(input.gravity_error_mps2);

  for (std::size_t section_index = 0U;
       section_index < section_count; ++section_index) {
    const double begin =
        input.arc.flight_time_s *
        static_cast<double>(section_index) /
        static_cast<double>(section_count);
    const double end =
        input.arc.flight_time_s *
        static_cast<double>(section_index + 1U) /
        static_cast<double>(section_count);
    const Eigen::Vector3d error_radius =
        position_radius + velocity_radius * end +
        0.5 * gravity_radius * end * end;
    Bounds3 bounds{};
    for (Eigen::Index axis = 0; axis < 3; ++axis) {
      const auto [minimum, maximum] =
          CoordinateRange(input.arc, axis, begin, end);
      bounds.minimum[axis] =
          minimum - error_radius[axis] +
          body_bounds->minimum[axis];
      bounds.maximum[axis] =
          maximum + error_radius[axis] +
          body_bounds->maximum[axis];
    }
    if (!bounds.minimum.allFinite() ||
        !bounds.maximum.allFinite()) {
      result.diagnostics.rejection_reason =
          "flight_tube_numerical_failure";
      return result;
    }

    const long long minimum_x = static_cast<long long>(
        std::floor(
            (bounds.minimum.x() - geometry.origin_m.x) /
            geometry.resolution_m));
    const long long maximum_x = static_cast<long long>(
        std::floor(
            (bounds.maximum.x() - geometry.origin_m.x) /
            geometry.resolution_m));
    const long long minimum_y = static_cast<long long>(
        std::floor(
            (bounds.minimum.y() - geometry.origin_m.y) /
            geometry.resolution_m));
    const long long maximum_y = static_cast<long long>(
        std::floor(
            (bounds.maximum.y() - geometry.origin_m.y) /
            geometry.resolution_m));
    if (minimum_x < 0 || minimum_y < 0 ||
        maximum_x >= static_cast<long long>(geometry.width) ||
        maximum_y >= static_cast<long long>(geometry.height)) {
      result.diagnostics.rejection_reason =
          "flight_tube_outside_map";
      return result;
    }
    for (long long y = minimum_y; y <= maximum_y; ++y) {
      for (long long x = minimum_x; x <= maximum_x; ++x) {
        ++result.diagnostics.overlapped_cell_count;
        const Cell cell{
            static_cast<std::int32_t>(x),
            static_cast<std::int32_t>(y)};
        const std::size_t cell_index =
            landing_cell_index(geometry, cell);
        if (cell_index >= known.size() || known[cell_index] == 0U) {
          result.diagnostics.rejection_reason =
              "flight_tube_overlaps_unknown_cell";
          return result;
        }
        if (cell_index >= obstacles.size() ||
            obstacles[cell_index] != 0U) {
          result.diagnostics.rejection_reason =
              "continuous_swept_volume_intersects_obstacle";
          return result;
        }
        if (cell_index >= elevations.size() ||
            !std::isfinite(elevations[cell_index])) {
          result.diagnostics.rejection_reason =
              "flight_tube_overlaps_unknown_cell";
          return result;
        }
        const double clearance =
            bounds.minimum.z() -
            static_cast<double>(elevations[cell_index]);
        result.diagnostics.minimum_clearance_m =
            std::min(
                result.diagnostics.minimum_clearance_m,
                clearance);
        if (clearance <= 0.0) {
          const bool source_contact =
              section_index == 0U &&
              CellBelongsToRegion(
                  cell, geometry, input.source_region);
          const bool target_contact =
              section_index + 1U == section_count &&
              CellBelongsToRegion(
                  cell, geometry, input.target_region);
          if (!source_contact && !target_contact) {
            result.diagnostics.rejection_reason =
                "continuous_swept_volume_intersects_terrain";
            return result;
          }
        }
      }
    }
    sections.push_back(
        {
            .time_interval =
                {
                    .start_offset = FloorDuration(begin),
                    .end_offset = CeilDuration(end),
                },
            .envelope = MakeBoundsPolytope(bounds),
        });
  }
  result.diagnostics.time_slab_count = sections.size();
  result.diagnostics.subdivision_count =
      sections.empty() ? 0U : sections.size() - 1U;
  if (!std::isfinite(result.diagnostics.minimum_clearance_m)) {
    result.diagnostics.minimum_clearance_m = 0.0;
  }
  result.certified_tube = CertifiedFlightTube{
      .frame_id = input.arc.frame_id,
      .sections = std::move(sections),
      .source_map_snapshot_ref = input.source_snapshot_ref,
      .body_rotation_envelope_ref =
          input.body_rotation_envelope_ref,
      .error_model_ref = input.error_model_ref,
      .minimum_certified_clearance_m =
          std::max(0.0, result.diagnostics.minimum_clearance_m),
  };
  return result;
}

}  // namespace lunar::planning::v3
