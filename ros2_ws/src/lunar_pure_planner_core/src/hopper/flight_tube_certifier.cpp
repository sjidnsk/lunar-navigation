#include "hopper/flight_tube_certifier.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "hopper/ballistic_kinematics.hpp"

namespace lunar::pure_planning::hopper {
namespace {

constexpr double kTolerance = 1.0e-9;

struct TimeSection final {
  double begin_s{};
  double end_s{};
  bool launch_contact{};
  bool landing_contact{};
};

struct Bounds3 final {
  Vec3 minimum;
  Vec3 maximum;
};

[[nodiscard]] bool Finite(const Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
      std::isfinite(value.z);
}

[[nodiscard]] double Distance(const Vec3 lhs, const Vec3 rhs) noexcept {
  return std::hypot(
      std::hypot(lhs.x - rhs.x, lhs.y - rhs.y), lhs.z - rhs.z);
}

[[nodiscard]] double Norm(const Vec3 value) noexcept {
  return std::hypot(std::hypot(value.x, value.y), value.z);
}

[[nodiscard]] double PlanarPointSegmentDistance(
    const Vec3 point, const Vec3 begin, const Vec3 end) noexcept {
  const double dx = end.x - begin.x;
  const double dy = end.y - begin.y;
  const double squared_length = dx * dx + dy * dy;
  if (squared_length <= std::numeric_limits<double>::epsilon()) {
    return std::hypot(point.x - begin.x, point.y - begin.y);
  }
  const double projection = std::clamp(
      ((point.x - begin.x) * dx + (point.y - begin.y) * dy) /
          squared_length,
      0.0, 1.0);
  return std::hypot(
      point.x - (begin.x + projection * dx),
      point.y - (begin.y + projection * dy));
}

[[nodiscard]] double PointRectangleDistance(
    const Vec3 point, const double x0, const double x1,
    const double y0, const double y1) noexcept {
  const double nearest_x = std::clamp(point.x, x0, x1);
  const double nearest_y = std::clamp(point.y, y0, y1);
  return std::hypot(point.x - nearest_x, point.y - nearest_y);
}

[[nodiscard]] bool SegmentIntersectsRectangle(
    const Vec3 begin, const Vec3 end, const double x0, const double x1,
    const double y0, const double y1) noexcept {
  double lower = 0.0;
  double upper = 1.0;
  const double dx = end.x - begin.x;
  const double dy = end.y - begin.y;
  const auto clip = [&](const double p, const double q) {
    if (std::abs(p) <= std::numeric_limits<double>::epsilon()) {
      return q >= 0.0;
    }
    const double ratio = q / p;
    if (p < 0.0) {
      lower = std::max(lower, ratio);
    } else {
      upper = std::min(upper, ratio);
    }
    return lower <= upper;
  };
  return clip(-dx, begin.x - x0) && clip(dx, x1 - begin.x) &&
      clip(-dy, begin.y - y0) && clip(dy, y1 - begin.y);
}

[[nodiscard]] double SegmentRectangleDistance(
    const Vec3 begin, const Vec3 end, const double x0, const double x1,
    const double y0, const double y1) noexcept {
  if (SegmentIntersectsRectangle(begin, end, x0, x1, y0, y1)) {
    return 0.0;
  }
  double distance = std::min(
      PointRectangleDistance(begin, x0, x1, y0, y1),
      PointRectangleDistance(end, x0, x1, y0, y1));
  for (const Vec3 corner : {
           Vec3{x0, y0, 0.0}, Vec3{x1, y0, 0.0},
           Vec3{x1, y1, 0.0}, Vec3{x0, y1, 0.0}}) {
    distance = std::min(
        distance, PlanarPointSegmentDistance(corner, begin, end));
  }
  return distance;
}

[[nodiscard]] std::pair<double, double> CoordinateRange(
    const BallisticArc& arc, const std::size_t axis,
    const double begin_s, const double end_s) noexcept {
  const auto coordinate = [axis](const Vec3 value) {
    return axis == 0U ? value.x : (axis == 1U ? value.y : value.z);
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
  if (std::abs(acceleration) > std::numeric_limits<double>::epsilon()) {
    const double stationary = -velocity / acceleration;
    if (stationary > begin_s && stationary < end_s) {
      minimum = std::min(minimum, value(stationary));
      maximum = std::max(maximum, value(stationary));
    }
  }
  return {minimum, maximum};
}

[[nodiscard]] bool Known(
    const shared::MapSnapshot& map, const std::size_t index,
    const MapSafetyConfig& config) noexcept {
  return map.ByteLayer("valid_mask")[index] != 0U &&
      static_cast<double>(map.FloatLayer("obstacle_variance")[index]) <=
          config.maximum_obstacle_variance_m2 + kTolerance &&
      static_cast<double>(map.FloatLayer("observation_age_s")[index]) <=
          config.maximum_observation_age_s + kTolerance &&
      static_cast<double>(map.FloatLayer("observation_quality")[index]) +
              kTolerance >=
          config.minimum_observation_quality &&
      map.CountLayer("observation_count")[index] >=
          config.minimum_observation_count;
}

[[nodiscard]] bool ContactCell(
    const shared::MapSnapshot& map, const shared::GridCell cell,
    const Vec3 contact, const double radius_m) noexcept {
  const double x0 = map.origin_m().x + static_cast<double>(cell.x) *
      map.resolution_m();
  const double y0 = map.origin_m().y + static_cast<double>(cell.y) *
      map.resolution_m();
  return PointRectangleDistance(
             contact, x0, x0 + map.resolution_m(), y0,
             y0 + map.resolution_m()) <=
      radius_m + kTolerance;
}

[[nodiscard]] FlightTubeCertificationResult Failure(
    std::string reason_code, const std::size_t section_count,
    const std::size_t overlapped_cell_count, const double minimum_clearance_m,
    const double radius_m, const bool canceled = false) {
  return {
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
    const BallisticArc& arc, const shared::MapSnapshot& map,
    const HopperCapability& capability, const MapSafetyConfig& map_safety,
    const std::stop_token stop_token, const double additional_radius_m) {
  const double radius = capability.flight_collision_radius_m +
      capability.flight_map_margin_m + additional_radius_m;
  if (stop_token.stop_requested()) {
    return Failure("REQUEST_CANCELED", 0U, 0U, 0.0, radius, true);
  }
  if (!Finite(arc.launch_position_m) || !Finite(arc.landing_position_m) ||
      !Finite(arc.launch_velocity_mps) || !Finite(arc.gravity_mps2) ||
      !std::isfinite(arc.flight_time_s) || arc.flight_time_s <= 0.0 ||
      !std::isfinite(radius) || radius <= 0.0 ||
      !std::isfinite(additional_radius_m) || additional_radius_m < 0.0) {
    return Failure(
        "HOPPER_FLIGHT_TUBE_INPUT_INVALID", 0U, 0U, 0.0, radius);
  }

  const double spatial_tolerance = 0.25 * map.resolution_m();
  std::vector<TimeSection> pending{
      TimeSection{
          .begin_s = 0.0,
          .end_s = arc.flight_time_s,
          .launch_contact = true,
          .landing_contact = true,
      },
  };
  std::vector<TimeSection> sections;
  while (!pending.empty()) {
    if (stop_token.stop_requested()) {
      return Failure(
          "REQUEST_CANCELED", sections.size(), 0U, 0.0, radius, true);
    }
    const TimeSection section = pending.back();
    pending.pop_back();
    const double midpoint = std::midpoint(section.begin_s, section.end_s);
    if (!(midpoint > section.begin_s && midpoint < section.end_s)) {
      return Failure(
          "HOPPER_FLIGHT_TUBE_NUMERICAL_INDETERMINATE", sections.size(),
          0U, 0.0, radius);
    }
    const Vec3 begin = EvaluateBallisticState(arc, section.begin_s).position_m;
    const Vec3 end = EvaluateBallisticState(arc, section.end_s).position_m;
    const double duration = section.end_s - section.begin_s;
    const double chord_length = Distance(begin, end);
    const double curvature_deviation =
        Norm(arc.gravity_mps2) * duration * duration / 8.0;
    if (!std::isfinite(chord_length) || !std::isfinite(curvature_deviation)) {
      return Failure(
          "HOPPER_FLIGHT_TUBE_NUMERICAL_INDETERMINATE", sections.size(),
          0U, 0.0, radius);
    }
    if (std::max(chord_length, curvature_deviation) > spatial_tolerance) {
      pending.push_back(TimeSection{
          .begin_s = midpoint,
          .end_s = section.end_s,
          .launch_contact = false,
          .landing_contact = section.landing_contact,
      });
      pending.push_back(TimeSection{
          .begin_s = section.begin_s,
          .end_s = midpoint,
          .launch_contact = section.launch_contact,
          .landing_contact = false,
      });
      continue;
    }
    sections.push_back(section);
  }
  std::ranges::sort(sections, {}, &TimeSection::begin_s);

  std::size_t overlapped = 0U;
  double minimum_clearance = std::numeric_limits<double>::infinity();
  for (std::size_t section_index = 0U; section_index < sections.size();
       ++section_index) {
    if (stop_token.stop_requested()) {
      return Failure(
          "REQUEST_CANCELED", section_index, overlapped,
          std::isfinite(minimum_clearance) ? minimum_clearance : 0.0,
          radius, true);
    }
    const TimeSection& section = sections[section_index];
    const Vec3 section_begin =
        EvaluateBallisticState(arc, section.begin_s).position_m;
    const Vec3 section_end =
        EvaluateBallisticState(arc, section.end_s).position_m;
    const auto [minimum_x, maximum_x] =
        CoordinateRange(arc, 0U, section.begin_s, section.end_s);
    const auto [minimum_y, maximum_y] =
        CoordinateRange(arc, 1U, section.begin_s, section.end_s);
    const auto [minimum_z, maximum_z] =
        CoordinateRange(arc, 2U, section.begin_s, section.end_s);
    const Bounds3 bounds{
        .minimum = {minimum_x - radius, minimum_y - radius,
                    minimum_z - radius},
        .maximum = {maximum_x + radius, maximum_y + radius,
                    maximum_z + radius},
    };
    if (!Finite(bounds.minimum) || !Finite(bounds.maximum)) {
      return Failure(
          "HOPPER_FLIGHT_TUBE_NUMERICAL_INDETERMINATE", section_index,
          overlapped, 0.0, radius);
    }
    const auto cell_index = [&](const double coordinate, const double origin) {
      return static_cast<long long>(
          std::floor((coordinate - origin) / map.resolution_m()));
    };
    const long long x0 = cell_index(bounds.minimum.x, map.origin_m().x);
    const long long x1 = cell_index(bounds.maximum.x, map.origin_m().x);
    const long long y0 = cell_index(bounds.minimum.y, map.origin_m().y);
    const long long y1 = cell_index(bounds.maximum.y, map.origin_m().y);
    if (x0 < 0 || y0 < 0 || x1 >= static_cast<long long>(map.width()) ||
        y1 >= static_cast<long long>(map.height())) {
      return Failure(
          "HOPPER_FLIGHT_TUBE_OUTSIDE_MAP", section_index, overlapped,
          std::isfinite(minimum_clearance) ? minimum_clearance : 0.0, radius);
    }
    for (long long y = y0; y <= y1; ++y) {
      for (long long x = x0; x <= x1; ++x) {
        ++overlapped;
        const shared::GridCell cell{
            .x = static_cast<std::int32_t>(x),
            .y = static_cast<std::int32_t>(y),
        };
        const Vec3 cell_center = map.CellCenter(cell);
        const double cell_x0 = map.origin_m().x +
            static_cast<double>(cell.x) * map.resolution_m();
        const double cell_y0 = map.origin_m().y +
            static_cast<double>(cell.y) * map.resolution_m();
        const double planar_distance = SegmentRectangleDistance(
            section_begin, section_end, cell_x0,
            cell_x0 + map.resolution_m(), cell_y0,
            cell_y0 + map.resolution_m());
        if (planar_distance > radius + kTolerance) {
          continue;
        }
        const std::size_t index = map.Index(cell);
        if (!Known(map, index, map_safety)) {
          return Failure(
              "HOPPER_FLIGHT_TUBE_UNKNOWN", section_index, overlapped,
              std::isfinite(minimum_clearance) ? minimum_clearance : 0.0,
              radius);
        }
        if (map.ByteLayer("forbidden")[index] != 0U) {
          return Failure(
              "HOPPER_FLIGHT_TUBE_FORBIDDEN", section_index, overlapped,
              std::isfinite(minimum_clearance) ? minimum_clearance : 0.0,
              radius);
        }
        const double terrain =
            static_cast<double>(map.FloatLayer("elevation")[index]);
        const double vertical_radius = std::sqrt(std::max(
            0.0, radius * radius - planar_distance * planar_distance));
        const double lower_tube_z = minimum_z - vertical_radius;
        const double clearance = lower_tube_z - terrain;
        minimum_clearance = std::min(minimum_clearance, clearance);
        const bool contact =
            ContactCell(
                map, cell, arc.launch_position_m,
                radius + std::numbers::sqrt2 * 0.5 * map.resolution_m()) ||
            ContactCell(
                map, cell, arc.landing_position_m,
                radius + std::numbers::sqrt2 * 0.5 * map.resolution_m());
        if (clearance < -kTolerance && !contact) {
          return Failure(
              "HOPPER_FLIGHT_TUBE_TERRAIN_COLLISION", section_index,
              overlapped, clearance, radius);
        }
        if (map.ByteLayer("obstacle")[index] != 0U) {
          const double obstacle_top = terrain + std::max(
              static_cast<double>(map.FloatLayer("obstacle_height")[index]),
              map.resolution_m());
          if (obstacle_top >= lower_tube_z - kTolerance) {
            return Failure(
                "HOPPER_FLIGHT_TUBE_OBSTACLE_COLLISION", section_index,
                overlapped, clearance, radius);
          }
        }
      }
    }
  }

  return {
      .certified = true,
      .section_count = sections.size(),
      .overlapped_cell_count = overlapped,
      .minimum_clearance_m = std::isfinite(minimum_clearance)
          ? std::max(0.0, minimum_clearance)
          : 0.0,
      .radius_m = radius,
  };
}

FlightTubeCertificationResult CertifyFlightTube(
    const BallisticArc& arc, const shared::MapSnapshot& map,
    const CertifiedLandingRegion&, const CertifiedLandingRegion&,
    const HopperCapability& capability, const PlannerConfig& config,
    const std::stop_token stop_token, const double additional_radius_m) {
  return CertifyFlightTube(
      arc, map, capability, config.map_safety, stop_token,
      additional_radius_m);
}

}  // namespace lunar::pure_planning::hopper
