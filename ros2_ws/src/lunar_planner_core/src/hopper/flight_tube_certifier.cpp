#include "hopper/flight_tube_certifier.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "hopper/ballistic_kinematics.hpp"

namespace lunar::planning::hopper {
namespace {

constexpr double kTolerance = 1.0e-9;

struct Bounds3 final {
  Vec3 minimum;
  Vec3 maximum;
};

[[nodiscard]] bool IsFinite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

[[nodiscard]] std::pair<double, double> CoordinateRange(
    const BallisticArc& arc, const std::size_t axis,
    const double begin_s, const double end_s) noexcept {
  const auto coordinate = [axis](const Vec3 value) {
    if (axis == 0U) {
      return value.x;
    }
    if (axis == 1U) {
      return value.y;
    }
    return value.z;
  };
  const double p0 = coordinate(arc.launch_position_m);
  const double velocity = coordinate(arc.launch_velocity_mps);
  const double acceleration = coordinate(arc.gravity_mps2);
  const auto value = [&](const double time_s) {
    return p0 + velocity * time_s +
        0.5 * acceleration * time_s * time_s;
  };
  double minimum = std::min(value(begin_s), value(end_s));
  double maximum = std::max(value(begin_s), value(end_s));
  if (std::abs(acceleration) > 1.0e-15) {
    const double stationary = -velocity / acceleration;
    if (stationary > begin_s && stationary < end_s) {
      minimum = std::min(minimum, value(stationary));
      maximum = std::max(maximum, value(stationary));
    }
  }
  return {minimum, maximum};
}

[[nodiscard]] bool CertificationDataKnown(
    const shared::MapSnapshot& map,
    const std::size_t index,
    const MapSafetyConfig& config) noexcept {
  const auto valid = map.ByteLayer("valid_mask");
  const auto elevation_variance = map.FloatLayer("elevation_variance");
  const auto obstacle_variance = map.FloatLayer("obstacle_variance");
  const auto observation_age = map.FloatLayer("observation_age_s");
  const auto observation_quality = map.FloatLayer("observation_quality");
  const auto observation_count = map.CountLayer("observation_count");
  return index < valid.size() && valid[index] != 0U &&
      index < elevation_variance.size() &&
      static_cast<double>(elevation_variance[index]) <=
          config.maximum_elevation_variance_m2 + kTolerance &&
      index < obstacle_variance.size() &&
      static_cast<double>(obstacle_variance[index]) <=
          config.maximum_obstacle_variance_m2 + kTolerance &&
      index < observation_age.size() &&
      static_cast<double>(observation_age[index]) <=
          config.maximum_observation_age_s + kTolerance &&
      index < observation_quality.size() &&
      static_cast<double>(observation_quality[index]) + kTolerance >=
          config.minimum_observation_quality &&
      index < observation_count.size() &&
      observation_count[index] >= config.minimum_observation_count;
}

[[nodiscard]] bool ContactFootprintContains(
    const shared::MapSnapshot& map,
    const shared::GridCell cell,
    const Vec3 contact_position,
    const HopperCapability& capability,
    const double additional_radius_m) noexcept {
  const Vec3 center = map.CellCenter(cell);
  const double half_cell = 0.5 * map.resolution_m();
  return std::abs(center.x - contact_position.x) <=
      capability.body_half_extent_m.x +
          capability.minimum_lateral_clearance_m + additional_radius_m +
          half_cell +
                 kTolerance &&
      std::abs(center.y - contact_position.y) <=
      capability.body_half_extent_m.y +
          capability.minimum_lateral_clearance_m + additional_radius_m +
          half_cell +
                 kTolerance;
}

[[nodiscard]] FlightTubeCertificationResult Failure(
    std::string reason_code,
    const std::size_t section_count,
    const std::size_t overlapped_cell_count,
    const double minimum_clearance_m,
    const double radius_m,
    const bool canceled = false) {
  return FlightTubeCertificationResult{
      .certified = false,
      .canceled = canceled,
      .section_count = section_count,
      .overlapped_cell_count = overlapped_cell_count,
      .minimum_clearance_m = minimum_clearance_m,
      .radius_m = radius_m,
      .reason_code = std::move(reason_code),
  };
}

}  // namespace

FlightTubeCertificationResult CertifyFlightTube(
    const BallisticArc& arc,
    const shared::MapSnapshot& map,
    const CertifiedLandingRegion& source_region,
    const CertifiedLandingRegion& target_region,
    const HopperCapability& capability,
    const PlannerConfig& config,
    const std::stop_token stop_token,
    const double additional_radius_m) {
  const double horizontal_x = capability.body_half_extent_m.x +
      capability.minimum_lateral_clearance_m + additional_radius_m;
  const double horizontal_y = capability.body_half_extent_m.y +
      capability.minimum_lateral_clearance_m + additional_radius_m;
  const double radius = std::hypot(horizontal_x, horizontal_y);
  if (stop_token.stop_requested()) {
    return Failure("REQUEST_CANCELED", 0U, 0U, 0.0, radius, true);
  }
  if (!IsFinite(arc.launch_position_m) || !IsFinite(arc.landing_position_m) ||
      !IsFinite(arc.launch_velocity_mps) || !IsFinite(arc.gravity_mps2) ||
      !std::isfinite(arc.flight_time_s) || arc.flight_time_s <= 0.0 ||
      !std::isfinite(additional_radius_m) || additional_radius_m < 0.0 ||
      !std::isfinite(radius) || radius <= 0.0) {
    return Failure(
        "HOPPER_FLIGHT_TUBE_INPUT_INVALID", 0U, 0U, 0.0, radius);
  }
  if (config.hopper.maximum_flight_tube_sections < 2U) {
    return Failure(
        "HOPPER_FLIGHT_TUBE_RESOURCE_EXHAUSTED", 0U, 0U, 0.0, radius);
  }
  const std::size_t section_count = std::min<std::size_t>(
      config.hopper.maximum_flight_tube_sections, 64U);
  const auto elevations = map.FloatLayer("elevation");
  const auto obstacles = map.ByteLayer("obstacle");
  const auto forbidden = map.ByteLayer("forbidden");
  const auto obstacle_heights = map.FloatLayer("obstacle_height");
  std::size_t overlapped = 0U;
  double minimum_clearance = std::numeric_limits<double>::infinity();

  for (std::size_t section = 0U; section < section_count; ++section) {
    if (stop_token.stop_requested()) {
      return Failure(
          "REQUEST_CANCELED", section, overlapped,
          std::isfinite(minimum_clearance) ? minimum_clearance : 0.0,
          radius, true);
    }
    const double begin_s = arc.flight_time_s *
        static_cast<double>(section) / static_cast<double>(section_count);
    const double end_s = arc.flight_time_s *
        static_cast<double>(section + 1U) /
        static_cast<double>(section_count);
    const auto [minimum_x, maximum_x] =
        CoordinateRange(arc, 0U, begin_s, end_s);
    const auto [minimum_y, maximum_y] =
        CoordinateRange(arc, 1U, begin_s, end_s);
    const auto [minimum_z, maximum_z] =
        CoordinateRange(arc, 2U, begin_s, end_s);
    const Bounds3 bounds{
        .minimum = Vec3{
            minimum_x - horizontal_x,
            minimum_y - horizontal_y,
            minimum_z - capability.body_half_extent_m.z,
        },
        .maximum = Vec3{
            maximum_x + horizontal_x,
            maximum_y + horizontal_y,
            maximum_z + capability.body_half_extent_m.z +
                capability.minimum_overhead_clearance_m,
        },
    };
    if (!IsFinite(bounds.minimum) || !IsFinite(bounds.maximum)) {
      return Failure(
          "HOPPER_FLIGHT_TUBE_NUMERICAL_INDETERMINATE", section, overlapped,
          0.0, radius);
    }
    const auto cell_index = [&](const double coordinate, const double origin) {
      return static_cast<long long>(
          std::floor((coordinate - origin) / map.resolution_m()));
    };
    const long long minimum_cell_x =
        cell_index(bounds.minimum.x, map.origin_m().x);
    const long long maximum_cell_x =
        cell_index(bounds.maximum.x, map.origin_m().x);
    const long long minimum_cell_y =
        cell_index(bounds.minimum.y, map.origin_m().y);
    const long long maximum_cell_y =
        cell_index(bounds.maximum.y, map.origin_m().y);
    if (minimum_cell_x < 0 || minimum_cell_y < 0 ||
        maximum_cell_x >= static_cast<long long>(map.width()) ||
        maximum_cell_y >= static_cast<long long>(map.height())) {
      return Failure(
          "HOPPER_FLIGHT_TUBE_OUTSIDE_MAP", section, overlapped,
          std::isfinite(minimum_clearance) ? minimum_clearance : 0.0,
          radius);
    }

    for (long long y = minimum_cell_y; y <= maximum_cell_y; ++y) {
      for (long long x = minimum_cell_x; x <= maximum_cell_x; ++x) {
        ++overlapped;
        const shared::GridCell cell{
            .x = static_cast<std::int32_t>(x),
            .y = static_cast<std::int32_t>(y),
        };
        const std::size_t index = map.Index(cell);
        if (!CertificationDataKnown(map, index, config.map_safety)) {
          return Failure(
              "HOPPER_FLIGHT_TUBE_UNKNOWN", section, overlapped,
              std::isfinite(minimum_clearance) ? minimum_clearance : 0.0,
              radius);
        }
        const double terrain = static_cast<double>(elevations[index]);
        const double clearance = bounds.minimum.z - terrain;
        minimum_clearance = std::min(minimum_clearance, clearance);
        const bool source_contact = section == 0U &&
            ContactFootprintContains(
                map, cell, source_region.aim_position_on_surface_m,
                capability, additional_radius_m);
        const bool target_contact = section + 1U == section_count &&
            ContactFootprintContains(
                map, cell, target_region.aim_position_on_surface_m,
                capability, additional_radius_m);
        if (clearance < -kTolerance && !source_contact && !target_contact) {
          return Failure(
              "HOPPER_FLIGHT_TUBE_TERRAIN_COLLISION", section, overlapped,
              clearance, radius);
        }
        if (forbidden[index] != 0U) {
          return Failure(
              "HOPPER_FLIGHT_TUBE_FORBIDDEN", section, overlapped,
              clearance, radius);
        }
        if (obstacles[index] != 0U) {
          const double obstacle_top = terrain + std::max(
              static_cast<double>(obstacle_heights[index]),
              map.resolution_m());
          if (obstacle_top + capability.minimum_overhead_clearance_m >=
              bounds.minimum.z - kTolerance) {
            return Failure(
                "HOPPER_FLIGHT_TUBE_OBSTACLE_COLLISION", section,
                overlapped, clearance, radius);
          }
        }
      }
    }
  }

  return FlightTubeCertificationResult{
      .certified = true,
      .canceled = false,
      .section_count = section_count,
      .overlapped_cell_count = overlapped,
      .minimum_clearance_m =
          std::isfinite(minimum_clearance)
              ? std::max(0.0, minimum_clearance)
              : 0.0,
      .radius_m = radius,
      .reason_code = {},
  };
}

}  // namespace lunar::planning::hopper
