#include "shared/request_local_start_patch.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include "shared/controlled_work.hpp"

namespace lunar::pure_planning::shared {
namespace {

constexpr double kTolerance = 1.0e-9;
constexpr std::uint64_t kPatchPolicyVersion = 1U;
constexpr std::uint64_t kFnvOffset = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t kFnvPrime = 1'099'511'628'211ULL;

void HashByte(std::uint64_t& hash, const std::uint8_t value) noexcept {
  hash ^= value;
  hash *= kFnvPrime;
}

template <typename Value>
void HashPod(std::uint64_t& hash, const Value value) noexcept {
  static_assert(std::is_trivially_copyable_v<Value>);
  const auto bytes =
      std::bit_cast<std::array<std::uint8_t, sizeof(Value)>>(value);
  for (const std::uint8_t byte : bytes) {
    HashByte(hash, byte);
  }
}

[[nodiscard]] std::optional<double> YawFromQuaternion(
    const Quaternion& quaternion) noexcept {
  const double norm = std::sqrt(
      quaternion.w * quaternion.w + quaternion.x * quaternion.x +
      quaternion.y * quaternion.y + quaternion.z * quaternion.z);
  if (!std::isfinite(norm) || std::abs(norm - 1.0) > 1.0e-6) {
    return std::nullopt;
  }
  const double yaw = std::atan2(
      2.0 * (quaternion.w * quaternion.z +
             quaternion.x * quaternion.y),
      1.0 - 2.0 * (quaternion.y * quaternion.y +
                   quaternion.z * quaternion.z));
  return std::isfinite(yaw) ? std::optional<double>{yaw} : std::nullopt;
}

[[nodiscard]] double Cross(const Vec2& first, const Vec2& second,
                           const Vec2& third) noexcept {
  return (second.x - first.x) * (third.y - first.y) -
         (second.y - first.y) * (third.x - first.x);
}

[[nodiscard]] bool PointOnSegment(const Vec2& point, const Vec2& start,
                                  const Vec2& finish) noexcept {
  return std::abs(Cross(start, finish, point)) <= kTolerance &&
         point.x + kTolerance >= std::min(start.x, finish.x) &&
         point.x <= std::max(start.x, finish.x) + kTolerance &&
         point.y + kTolerance >= std::min(start.y, finish.y) &&
         point.y <= std::max(start.y, finish.y) + kTolerance;
}

[[nodiscard]] bool SegmentsIntersect(const Vec2& first_start,
                                     const Vec2& first_finish,
                                     const Vec2& second_start,
                                     const Vec2& second_finish) noexcept {
  const double first_side_start =
      Cross(first_start, first_finish, second_start);
  const double first_side_finish =
      Cross(first_start, first_finish, second_finish);
  const double second_side_start =
      Cross(second_start, second_finish, first_start);
  const double second_side_finish =
      Cross(second_start, second_finish, first_finish);
  const bool proper =
      ((first_side_start > kTolerance && first_side_finish < -kTolerance) ||
       (first_side_start < -kTolerance && first_side_finish > kTolerance)) &&
      ((second_side_start > kTolerance && second_side_finish < -kTolerance) ||
       (second_side_start < -kTolerance && second_side_finish > kTolerance));
  return proper || PointOnSegment(second_start, first_start, first_finish) ||
         PointOnSegment(second_finish, first_start, first_finish) ||
         PointOnSegment(first_start, second_start, second_finish) ||
         PointOnSegment(first_finish, second_start, second_finish);
}

[[nodiscard]] bool PointInPolygon(const Vec2& point,
                                  const std::vector<Vec2>& polygon) noexcept {
  bool inside = false;
  for (std::size_t current = 0U, previous = polygon.size() - 1U;
       current < polygon.size(); previous = current++) {
    const Vec2& start = polygon[previous];
    const Vec2& finish = polygon[current];
    if (PointOnSegment(point, start, finish)) {
      return true;
    }
    if ((start.y > point.y) == (finish.y > point.y)) {
      continue;
    }
    const double crossing_x =
        start.x + (point.y - start.y) * (finish.x - start.x) /
                      (finish.y - start.y);
    if (point.x < crossing_x) {
      inside = !inside;
    }
  }
  return inside;
}

[[nodiscard]] bool PolygonIntersectsCell(const std::vector<Vec2>& polygon,
                                         const double minimum_x,
                                         const double minimum_y,
                                         const double maximum_x,
                                         const double maximum_y) noexcept {
  const auto inside_cell = [&](const Vec2& point) {
    return point.x + kTolerance >= minimum_x &&
           point.x <= maximum_x + kTolerance &&
           point.y + kTolerance >= minimum_y &&
           point.y <= maximum_y + kTolerance;
  };
  if (std::ranges::any_of(polygon, inside_cell)) {
    return true;
  }
  const std::array<Vec2, 4U> corners{
      Vec2{.x = minimum_x, .y = minimum_y},
      Vec2{.x = maximum_x, .y = minimum_y},
      Vec2{.x = maximum_x, .y = maximum_y},
      Vec2{.x = minimum_x, .y = maximum_y},
  };
  if (std::ranges::any_of(corners, [&](const Vec2& corner) {
        return PointInPolygon(corner, polygon);
      })) {
    return true;
  }
  for (std::size_t polygon_index = 0U; polygon_index < polygon.size();
       ++polygon_index) {
    for (std::size_t cell_index = 0U; cell_index < corners.size();
         ++cell_index) {
      if (SegmentsIntersect(
              polygon[polygon_index],
              polygon[(polygon_index + 1U) % polygon.size()],
              corners[cell_index],
              corners[(cell_index + 1U) % corners.size()])) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] double SquaredDistanceToSegment(
    const Vec2& point, const Vec2& start, const Vec2& finish) noexcept {
  const double dx = finish.x - start.x;
  const double dy = finish.y - start.y;
  const double squared_length = dx * dx + dy * dy;
  if (squared_length <= kTolerance) {
    const double px = point.x - start.x;
    const double py = point.y - start.y;
    return px * px + py * py;
  }
  const double ratio = std::clamp(
      ((point.x - start.x) * dx + (point.y - start.y) * dy) /
          squared_length,
      0.0, 1.0);
  const double px = point.x - (start.x + ratio * dx);
  const double py = point.y - (start.y + ratio * dy);
  return px * px + py * py;
}

[[nodiscard]] double PolygonDistanceToCell(
    const std::vector<Vec2>& polygon, const double minimum_x,
    const double minimum_y, const double maximum_x,
    const double maximum_y) noexcept {
  if (PolygonIntersectsCell(polygon, minimum_x, minimum_y, maximum_x,
                            maximum_y)) {
    return 0.0;
  }
  const std::array<Vec2, 4U> corners{
      Vec2{.x = minimum_x, .y = minimum_y},
      Vec2{.x = maximum_x, .y = minimum_y},
      Vec2{.x = maximum_x, .y = maximum_y},
      Vec2{.x = minimum_x, .y = maximum_y},
  };
  double squared_distance = std::numeric_limits<double>::infinity();
  for (std::size_t polygon_index = 0U; polygon_index < polygon.size();
       ++polygon_index) {
    const Vec2& polygon_start = polygon[polygon_index];
    const Vec2& polygon_finish =
        polygon[(polygon_index + 1U) % polygon.size()];
    for (std::size_t cell_index = 0U; cell_index < corners.size();
         ++cell_index) {
      const Vec2& cell_start = corners[cell_index];
      const Vec2& cell_finish = corners[(cell_index + 1U) % corners.size()];
      squared_distance = std::min(
          {squared_distance,
           SquaredDistanceToSegment(polygon_start, cell_start, cell_finish),
           SquaredDistanceToSegment(polygon_finish, cell_start, cell_finish),
           SquaredDistanceToSegment(cell_start, polygon_start,
                                    polygon_finish),
           SquaredDistanceToSegment(cell_finish, polygon_start,
                                    polygon_finish)});
    }
  }
  return std::sqrt(squared_distance);
}

[[nodiscard]] bool InsideDiscreteStartEnvelope(
    const std::vector<Vec2>& polygon, const double clearance_m,
    const double minimum_x, const double minimum_y, const double maximum_x,
    const double maximum_y) noexcept {
  const double distance = PolygonDistanceToCell(
      polygon, minimum_x, minimum_y, maximum_x, maximum_y);
  return distance <= kTolerance ||
         distance + kTolerance < clearance_m;
}

struct PlaneAccumulator final {
  std::size_t count{};
  double x{};
  double y{};
  double z{};
  double xx{};
  double xy{};
  double yy{};
  double xz{};
  double yz{};
};

[[nodiscard]] std::optional<std::array<double, 3U>> FitSupportPlane(
    const GridMap& map, const std::vector<float>& occupancy,
    const std::vector<float>& elevation, const double occupancy_threshold,
    const std::int64_t minimum_x, const std::int64_t minimum_y,
    const std::int64_t maximum_x, const std::int64_t maximum_y,
    const SearchControl& control) {
  PlaneAccumulator values;
  std::size_t work{};
  for (std::int64_t y = minimum_y; y <= maximum_y; ++y) {
    for (std::int64_t x = minimum_x; x <= maximum_x; ++x) {
      if (ControlCheckDue(work++) && StopReason(control).has_value()) {
        return std::nullopt;
      }
      const std::size_t index = static_cast<std::size_t>(y) * map.width +
                                static_cast<std::size_t>(x);
      if (!std::isfinite(occupancy[index]) || occupancy[index] < 0.0F ||
          occupancy[index] >= occupancy_threshold ||
          !std::isfinite(elevation[index])) {
        continue;
      }
      const double sample_x = map.origin_m.x +
                              (static_cast<double>(x) + 0.5) *
                                  map.resolution_m;
      const double sample_y = map.origin_m.y +
                              (static_cast<double>(y) + 0.5) *
                                  map.resolution_m;
      const double sample_z = elevation[index];
      ++values.count;
      values.x += sample_x;
      values.y += sample_y;
      values.z += sample_z;
      values.xx += sample_x * sample_x;
      values.xy += sample_x * sample_y;
      values.yy += sample_y * sample_y;
      values.xz += sample_x * sample_z;
      values.yz += sample_y * sample_z;
    }
  }
  if (values.count == 0U) {
    return std::nullopt;
  }
  const double count = static_cast<double>(values.count);
  const double mean_x = values.x / count;
  const double mean_y = values.y / count;
  const double mean_z = values.z / count;
  const double centered_xx = values.xx - count * mean_x * mean_x;
  const double centered_xy = values.xy - count * mean_x * mean_y;
  const double centered_yy = values.yy - count * mean_y * mean_y;
  const double centered_xz = values.xz - count * mean_x * mean_z;
  const double centered_yz = values.yz - count * mean_y * mean_z;
  const double determinant =
      centered_xx * centered_yy - centered_xy * centered_xy;
  double x_slope{};
  double y_slope{};
  const double scale = std::max(
      {1.0, std::abs(centered_xx), std::abs(centered_xy),
       std::abs(centered_yy)});
  if (values.count >= 3U &&
      std::abs(determinant) > scale * scale * 1.0e-12) {
    x_slope = (centered_xz * centered_yy -
               centered_yz * centered_xy) /
              determinant;
    y_slope = (centered_yz * centered_xx -
               centered_xz * centered_xy) /
              determinant;
  }
  const double offset = mean_z - x_slope * mean_x - y_slope * mean_y;
  if (!std::isfinite(x_slope) || !std::isfinite(y_slope) ||
      !std::isfinite(offset)) {
    return std::nullopt;
  }
  return std::array<double, 3U>{x_slope, y_slope, offset};
}

[[nodiscard]] RequestLocalStartPatchResult Failure(std::string reason_code) {
  return {.reason_code = std::move(reason_code)};
}

[[nodiscard]] std::optional<std::int64_t> ClampedCellCoordinate(
    const double position_m, const double origin_m,
    const double resolution_m, const std::size_t cell_count) noexcept {
  const double relative =
      std::floor((position_m - origin_m) / resolution_m);
  if (!std::isfinite(relative) || cell_count == 0U) {
    return std::nullopt;
  }
  const double clamped = std::clamp(
      relative, 0.0, static_cast<double>(cell_count - 1U));
  return static_cast<std::int64_t>(clamped);
}

}  // namespace

RequestLocalStartPatchResult AnalyzeWheelStartPatch(
    const GridMap& map, const WheeledState& state,
    const WheeledCapability& capability, const double occupancy_threshold,
    const SearchControl& control) {
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return Failure(std::string{*stopped});
  }
  const auto occupancy_found = map.layers.find("occupancy");
  const auto elevation_found = map.layers.find("elevation");
  const auto* occupancy = occupancy_found == map.layers.end()
      ? nullptr
      : std::get_if<std::vector<float>>(&occupancy_found->second.values);
  const auto* elevation = elevation_found == map.layers.end()
      ? nullptr
      : std::get_if<std::vector<float>>(&elevation_found->second.values);
  const auto yaw = YawFromQuaternion(state.pose.orientation);
  if (occupancy == nullptr || elevation == nullptr ||
      occupancy->size() != map.CellCount() ||
      elevation->size() != map.CellCount() || map.width == 0U ||
      map.height == 0U || !std::isfinite(map.resolution_m) ||
      map.resolution_m <= 0.0 ||
      map.width >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      map.height >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      !std::isfinite(map.origin_m.x) || !std::isfinite(map.origin_m.y) ||
      !std::isfinite(map.origin_m.z) ||
      capability.footprint_xy_m.size() < 3U ||
      !std::isfinite(capability.minimum_clearance_m) ||
      capability.minimum_clearance_m < 0.0 ||
      !std::isfinite(capability.maximum_slope_rad) ||
      capability.maximum_slope_rad < 0.0 ||
      !std::isfinite(occupancy_threshold) || occupancy_threshold <= 0.0 ||
      occupancy_threshold > 1.0 || !yaw.has_value() ||
      !std::isfinite(state.pose.position_m.x) ||
      !std::isfinite(state.pose.position_m.y)) {
    return Failure("INVALID_INPUT");
  }

  const double cosine = std::cos(*yaw);
  const double sine = std::sin(*yaw);
  std::vector<Vec2> polygon;
  polygon.reserve(capability.footprint_xy_m.size());
  double minimum_x = std::numeric_limits<double>::infinity();
  double minimum_y = std::numeric_limits<double>::infinity();
  double maximum_x = -std::numeric_limits<double>::infinity();
  double maximum_y = -std::numeric_limits<double>::infinity();
  std::size_t work{};
  for (const Vec2 vertex : capability.footprint_xy_m) {
    if (ControlCheckDue(work++)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return Failure(std::string{*stopped});
      }
    }
    if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y)) {
      return Failure("INVALID_INPUT");
    }
    const Vec2 world{
        .x = state.pose.position_m.x + cosine * vertex.x - sine * vertex.y,
        .y = state.pose.position_m.y + sine * vertex.x + cosine * vertex.y,
    };
    if (!std::isfinite(world.x) || !std::isfinite(world.y)) {
      return Failure("INVALID_INPUT");
    }
    polygon.push_back(world);
    minimum_x = std::min(minimum_x, world.x);
    minimum_y = std::min(minimum_y, world.y);
    maximum_x = std::max(maximum_x, world.x);
    maximum_y = std::max(maximum_y, world.y);
  }
  const double clearance = capability.minimum_clearance_m;
  const double envelope_minimum_x = minimum_x - clearance;
  const double envelope_minimum_y = minimum_y - clearance;
  const double envelope_maximum_x = maximum_x + clearance;
  const double envelope_maximum_y = maximum_y + clearance;
  const auto minimum_cell_x = ClampedCellCoordinate(
      envelope_minimum_x, map.origin_m.x, map.resolution_m, map.width);
  const auto minimum_cell_y = ClampedCellCoordinate(
      envelope_minimum_y, map.origin_m.y, map.resolution_m, map.height);
  const auto maximum_cell_x = ClampedCellCoordinate(
      envelope_maximum_x, map.origin_m.x, map.resolution_m, map.width);
  const auto maximum_cell_y = ClampedCellCoordinate(
      envelope_maximum_y, map.origin_m.y, map.resolution_m, map.height);
  if (!minimum_cell_x.has_value() || !minimum_cell_y.has_value() ||
      !maximum_cell_x.has_value() || !maximum_cell_y.has_value()) {
    return Failure("INVALID_INPUT");
  }

  RequestLocalStartPatch patch;
  work = 0U;
  for (std::int64_t y = *minimum_cell_y; y <= *maximum_cell_y; ++y) {
    for (std::int64_t x = *minimum_cell_x; x <= *maximum_cell_x; ++x) {
      if (ControlCheckDue(work++)) {
        if (const auto stopped = StopReason(control); stopped.has_value()) {
          return Failure(std::string{*stopped});
        }
      }
      const double cell_minimum_x =
          map.origin_m.x + static_cast<double>(x) * map.resolution_m;
      const double cell_minimum_y =
          map.origin_m.y + static_cast<double>(y) * map.resolution_m;
      if (!InsideDiscreteStartEnvelope(
              polygon, clearance, cell_minimum_x, cell_minimum_y,
              cell_minimum_x + map.resolution_m,
              cell_minimum_y + map.resolution_m)) {
        continue;
      }
      const std::size_t index = static_cast<std::size_t>(y) * map.width +
                                static_cast<std::size_t>(x);
      const float occupancy_value = (*occupancy)[index];
      const bool occupancy_unknown = std::isnan(occupancy_value);
      const bool occupancy_free = std::isfinite(occupancy_value) &&
                                  occupancy_value >= 0.0F &&
                                  occupancy_value < occupancy_threshold;
      if (occupancy_unknown) {
        patch.occupancy_offsets.push_back(index);
      }
      if (std::isnan((*elevation)[index]) &&
          (occupancy_unknown || occupancy_free)) {
        patch.elevation_offsets.push_back(index);
      }
    }
  }

  if (!patch.required()) {
    return {.patch = std::move(patch)};
  }
  if (!patch.elevation_offsets.empty()) {
    constexpr std::int64_t kAnchorHaloCells = 2;
    const std::int64_t anchor_minimum_x = std::max<std::int64_t>(
        0, *minimum_cell_x - kAnchorHaloCells);
    const std::int64_t anchor_minimum_y = std::max<std::int64_t>(
        0, *minimum_cell_y - kAnchorHaloCells);
    const std::int64_t anchor_maximum_x = std::min<std::int64_t>(
        static_cast<std::int64_t>(map.width) - 1,
        *maximum_cell_x + kAnchorHaloCells);
    const std::int64_t anchor_maximum_y = std::min<std::int64_t>(
        static_cast<std::int64_t>(map.height) - 1,
        *maximum_cell_y + kAnchorHaloCells);
    const auto plane = FitSupportPlane(
        map, *occupancy, *elevation, occupancy_threshold, anchor_minimum_x,
        anchor_minimum_y, anchor_maximum_x, anchor_maximum_y, control);
    if (!plane.has_value()) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return Failure(std::string{*stopped});
      }
      return Failure("WHEEL_START_SUPPORT_PLANE_UNAVAILABLE");
    }
    patch.elevation_x_slope = (*plane)[0];
    patch.elevation_y_slope = (*plane)[1];
    patch.elevation_offset = (*plane)[2];
    if (std::atan(std::hypot(patch.elevation_x_slope,
                             patch.elevation_y_slope)) >
        capability.maximum_slope_rad + kTolerance) {
      return Failure("WHEEL_START_SUPPORT_PLANE_INFEASIBLE");
    }
  }

  std::uint64_t identity = kFnvOffset;
  HashPod(identity, kPatchPolicyVersion);
  HashPod(identity, state.pose.position_m.x);
  HashPod(identity, state.pose.position_m.y);
  HashPod(identity, state.pose.orientation.w);
  HashPod(identity, state.pose.orientation.x);
  HashPod(identity, state.pose.orientation.y);
  HashPod(identity, state.pose.orientation.z);
  HashPod(identity, occupancy_threshold);
  HashPod(identity, clearance);
  HashPod(identity, capability.footprint_xy_m.size());
  work = 0U;
  for (const Vec2 vertex : capability.footprint_xy_m) {
    if (ControlCheckDue(work++)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return Failure(std::string{*stopped});
      }
    }
    HashPod(identity, vertex.x);
    HashPod(identity, vertex.y);
  }
  HashPod(identity, patch.elevation_x_slope);
  HashPod(identity, patch.elevation_y_slope);
  HashPod(identity, patch.elevation_offset);
  HashPod(identity, patch.occupancy_offsets.size());
  for (const std::size_t offset : patch.occupancy_offsets) {
    if (ControlCheckDue(work++)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return Failure(std::string{*stopped});
      }
    }
    HashPod(identity, offset);
  }
  HashPod(identity, patch.elevation_offsets.size());
  for (const std::size_t offset : patch.elevation_offsets) {
    if (ControlCheckDue(work++)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return Failure(std::string{*stopped});
      }
    }
    HashPod(identity, offset);
  }
  patch.identity = identity == 0U ? 1U : identity;
  return {.patch = std::move(patch)};
}

std::string ApplyWheelStartPatch(
    GridMap& map, const RequestLocalStartPatch& patch,
    const SearchControl& control) {
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return std::string{*stopped};
  }
  const auto occupancy_found = map.layers.find("occupancy");
  const auto elevation_found = map.layers.find("elevation");
  auto* occupancy = occupancy_found == map.layers.end()
      ? nullptr
      : std::get_if<std::vector<float>>(&occupancy_found->second.values);
  auto* elevation = elevation_found == map.layers.end()
      ? nullptr
      : std::get_if<std::vector<float>>(&elevation_found->second.values);
  if (occupancy == nullptr || elevation == nullptr ||
      occupancy->size() != map.CellCount() ||
      elevation->size() != map.CellCount()) {
    return "INVALID_INPUT";
  }
  std::size_t work{};
  for (const std::size_t offset : patch.occupancy_offsets) {
    if (ControlCheckDue(work++)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return std::string{*stopped};
      }
    }
    if (offset >= occupancy->size() || !std::isnan((*occupancy)[offset])) {
      return "INVALID_INPUT";
    }
    (*occupancy)[offset] = 0.0F;
  }
  for (const std::size_t offset : patch.elevation_offsets) {
    if (ControlCheckDue(work++)) {
      if (const auto stopped = StopReason(control); stopped.has_value()) {
        return std::string{*stopped};
      }
    }
    if (offset >= elevation->size() || !std::isnan((*elevation)[offset])) {
      return "INVALID_INPUT";
    }
    const std::size_t x = offset % map.width;
    const std::size_t y = offset / map.width;
    const double sample_x = map.origin_m.x +
                            (static_cast<double>(x) + 0.5) * map.resolution_m;
    const double sample_y = map.origin_m.y +
                            (static_cast<double>(y) + 0.5) * map.resolution_m;
    const double height = patch.elevation_x_slope * sample_x +
                          patch.elevation_y_slope * sample_y +
                          patch.elevation_offset;
    if (!std::isfinite(height) ||
        height < -static_cast<double>(std::numeric_limits<float>::max()) ||
        height > static_cast<double>(std::numeric_limits<float>::max())) {
      return "INVALID_INPUT";
    }
    (*elevation)[offset] = static_cast<float>(height);
  }
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return std::string{*stopped};
  }
  return {};
}

RequestLocalPatchedMapResult BuildWheelStartPatchedMap(
    const GridMap& map, const RequestLocalStartPatch& patch,
    const SearchControl& control) {
  if (const auto stopped = StopReason(control); stopped.has_value()) {
    return {.reason_code = std::string{*stopped}};
  }
  GridMap copied{
      .frame_id = map.frame_id,
      .stamp = map.stamp,
      .width = map.width,
      .height = map.height,
      .resolution_m = map.resolution_m,
      .origin_m = map.origin_m,
      .layers = {},
  };
  for (const auto& [name, layer] : map.layers) {
    GridLayer copied_layer;
    std::string stopped_reason;
    copied_layer.values = std::visit(
        [&](const auto& source_values) -> GridLayerValues {
          using Values = std::decay_t<decltype(source_values)>;
          Values values;
          values.reserve(source_values.size());
          for (std::size_t offset = 0U; offset < source_values.size();
               offset += kControlCheckStride) {
            if (const auto stopped = StopReason(control);
                stopped.has_value()) {
              stopped_reason = *stopped;
              return values;
            }
            const std::size_t finish = std::min(
                source_values.size(), offset + kControlCheckStride);
            values.insert(values.end(), source_values.begin() + offset,
                          source_values.begin() + finish);
          }
          return values;
        },
        layer.values);
    if (!stopped_reason.empty()) {
      return {.reason_code = std::move(stopped_reason)};
    }
    copied.layers.emplace(name, std::move(copied_layer));
  }
  if (const std::string patch_reason =
          ApplyWheelStartPatch(copied, patch, control);
      !patch_reason.empty()) {
    return {.reason_code = patch_reason};
  }
  return {.map = std::move(copied)};
}

}  // namespace lunar::pure_planning::shared
