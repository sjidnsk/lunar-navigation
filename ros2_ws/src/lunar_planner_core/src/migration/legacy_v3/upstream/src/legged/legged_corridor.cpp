#include "lunar_path_planner/v3/legged/legged_corridor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace lunar::planning::v3 {
namespace {

constexpr double kGeometryTolerance = 1.0e-10;

[[nodiscard]] Error Invalid(std::string path, std::string message) {
  return Error{
      .code = ErrorCode::kInvalidArgument,
      .field_path = std::move(path),
      .message = std::move(message),
  };
}

[[nodiscard]] bool IsFinite(const Vec2& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y);
}

[[nodiscard]] bool IsFinite(const Vec3& value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] double Evaluate(const HalfPlane2& plane,
                              const Vec2& point) noexcept {
  return plane.outward_unit_normal.x * point.x +
         plane.outward_unit_normal.y * point.y -
         plane.upper_offset_m;
}

[[nodiscard]] bool Inside(
    std::span<const HalfPlane2> planes,
    const Vec2& point) noexcept {
  return std::ranges::all_of(
      planes, [&](const HalfPlane2& plane) {
        return Evaluate(plane, point) <= kGeometryTolerance;
      });
}

[[nodiscard]] std::vector<Vec2> PolygonVertices(
    const ConvexCorridorCell& cell) {
  std::vector<Vec2> vertices;
  for (std::size_t first = 0U;
       first < cell.half_planes.size(); ++first) {
    for (std::size_t second = first + 1U;
         second < cell.half_planes.size(); ++second) {
      const auto& a = cell.half_planes[first];
      const auto& b = cell.half_planes[second];
      const double determinant =
          a.outward_unit_normal.x * b.outward_unit_normal.y -
          a.outward_unit_normal.y * b.outward_unit_normal.x;
      if (std::abs(determinant) <= kGeometryTolerance) {
        continue;
      }
      const Vec2 intersection{
          .x =
              (a.upper_offset_m * b.outward_unit_normal.y -
               a.outward_unit_normal.y * b.upper_offset_m) /
              determinant,
          .y =
              (a.outward_unit_normal.x * b.upper_offset_m -
               a.upper_offset_m * b.outward_unit_normal.x) /
              determinant,
      };
      if (IsFinite(intersection) &&
          Inside(cell.half_planes, intersection)) {
        vertices.push_back(intersection);
      }
    }
  }
  std::ranges::sort(
      vertices, [](const Vec2& lhs, const Vec2& rhs) {
        return std::tie(lhs.x, lhs.y) < std::tie(rhs.x, rhs.y);
      });
  vertices.erase(
      std::unique(
          vertices.begin(), vertices.end(),
          [](const Vec2& lhs, const Vec2& rhs) {
            return std::hypot(lhs.x - rhs.x, lhs.y - rhs.y) <=
                   kGeometryTolerance;
          }),
      vertices.end());
  return vertices;
}

[[nodiscard]] double Determinant(
    const Vec3& first, const Vec3& second,
    const Vec3& third) noexcept {
  return first.x * (second.y * third.z -
                    second.z * third.y) -
         first.y * (second.x * third.z -
                    second.z * third.x) +
         first.z * (second.x * third.y -
                    second.y * third.x);
}

[[nodiscard]] std::optional<Vec3> Intersection(
    const Halfspace3& first, const Halfspace3& second,
    const Halfspace3& third) noexcept {
  const double determinant =
      Determinant(first.normal, second.normal, third.normal);
  if (!std::isfinite(determinant) ||
      std::abs(determinant) <= kGeometryTolerance) {
    return std::nullopt;
  }
  const Vec3 offsets{
      first.offset_m, second.offset_m, third.offset_m};
  const Vec3 x_column{
      first.normal.y, second.normal.y, third.normal.y};
  const Vec3 y_column{
      first.normal.z, second.normal.z, third.normal.z};
  const Vec3 z_column{
      first.normal.x, second.normal.x, third.normal.x};
  const Vec3 point{
      .x = Determinant(offsets, x_column, y_column) /
           determinant,
      .y = Determinant(z_column, offsets, y_column) /
           determinant,
      .z = Determinant(z_column, x_column, offsets) /
           determinant,
  };
  return IsFinite(point) ? std::optional<Vec3>{point}
                         : std::nullopt;
}

[[nodiscard]] std::optional<double> HorizontalBodyRadius(
    const BodyConvexPolytope& envelope) noexcept {
  const auto& halfspaces = envelope.body_frame_halfspaces.halfspaces;
  double radius = 0.0;
  bool found_vertex = false;
  for (std::size_t first = 0U; first < halfspaces.size(); ++first) {
    for (std::size_t second = first + 1U;
         second < halfspaces.size(); ++second) {
      for (std::size_t third = second + 1U;
           third < halfspaces.size(); ++third) {
        const auto point = Intersection(
            halfspaces[first], halfspaces[second],
            halfspaces[third]);
        if (!point.has_value()) {
          continue;
        }
        const bool inside = std::ranges::all_of(
            halfspaces, [&](const Halfspace3& halfspace) {
              return halfspace.normal.x * point->x +
                         halfspace.normal.y * point->y +
                         halfspace.normal.z * point->z <=
                     halfspace.offset_m + kGeometryTolerance;
            });
        if (inside) {
          found_vertex = true;
          radius = std::max(radius,
                            std::hypot(point->x, point->y));
        }
      }
    }
  }
  if (!found_vertex || !std::isfinite(radius)) {
    return std::nullopt;
  }
  return radius;
}

[[nodiscard]] double HorizontalErrorRadius(
    const DeterministicVectorSet3& bound) noexcept {
  return std::visit(
      [](const auto& concrete) {
        using Bound = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<Bound, AxisAlignedBox3>) {
          return std::hypot(
              std::abs(concrete.center.x) +
                  concrete.half_extent.x,
              std::abs(concrete.center.y) +
                  concrete.half_extent.y);
        } else {
          return std::hypot(concrete.center.x,
                            concrete.center.y) +
                 concrete.radius;
        }
      },
      bound);
}

[[nodiscard]] Vec2 StatePosition(
    const LeggedLatticeState& state,
    const GridGeometry& geometry) noexcept {
  return {
      .x = geometry.origin_m.x +
           (static_cast<double>(state.ix) + 0.5) *
               geometry.resolution_m,
      .y = geometry.origin_m.y +
           (static_cast<double>(state.iy) + 0.5) *
               geometry.resolution_m,
  };
}

[[nodiscard]] double UnwrapNear(double yaw,
                                double previous) noexcept {
  const double period = 2.0 * std::numbers::pi;
  while (yaw - previous > std::numbers::pi) {
    yaw -= period;
  }
  while (yaw - previous < -std::numbers::pi) {
    yaw += period;
  }
  return yaw;
}

[[nodiscard]] std::optional<Cell> PositionCell(
    const GridGeometry& geometry, const Vec2& position) noexcept {
  const double x =
      (position.x - geometry.origin_m.x) / geometry.resolution_m;
  const double y =
      (position.y - geometry.origin_m.y) / geometry.resolution_m;
  if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0 ||
      y < 0.0 || x >= static_cast<double>(geometry.width) ||
      y >= static_cast<double>(geometry.height)) {
    return std::nullopt;
  }
  return Cell{
      .x = static_cast<std::int32_t>(std::floor(x)),
      .y = static_cast<std::int32_t>(std::floor(y)),
  };
}

class AnalyticProductCertifier final
    : public LeggedCartesianProductCertifier {
 public:
  AnalyticProductCertifier(
      const SafeProjection& projection,
      const LeggedCapabilityView& capability) noexcept
      : projection_(&projection), capability_(&capability) {}

  [[nodiscard]] bool Certify(
      const LeggedProductBox& box) const noexcept override {
    if (projection_ == nullptr || capability_ == nullptr ||
        !IsValidHeightInterval(box.z) ||
        !std::isfinite(box.yaw.min_unwrapped_rad) ||
        !std::isfinite(box.yaw.max_unwrapped_rad) ||
        box.yaw.min_unwrapped_rad > box.yaw.max_unwrapped_rad ||
        box.yaw.max_unwrapped_rad -
                box.yaw.min_unwrapped_rad >
            2.0 * std::numbers::pi + kGeometryTolerance) {
      return false;
    }
    const auto vertices = PolygonVertices(box.xy);
    if (vertices.size() < 3U) {
      return false;
    }
    const auto elevation = projection_->ElevationMeters();
    const auto& limits = capability_->terrain_thresholds;
    for (const auto& vertex : vertices) {
      const auto cell =
          PositionCell(projection_->geometry, vertex);
      if (!cell.has_value() || !projection_->Known(*cell) ||
          !projection_->HardFeasible(*cell)) {
        return false;
      }
      const std::size_t index =
          static_cast<std::size_t>(cell->y) *
              projection_->geometry.width +
          static_cast<std::size_t>(cell->x);
      const double terrain_height =
          static_cast<double>(elevation[index]);
      if (box.z.min_m <
              terrain_height + limits.minimum_body_height_m -
                  kGeometryTolerance ||
          box.z.max_m >
              terrain_height + limits.maximum_body_height_m +
                  kGeometryTolerance) {
        return false;
      }
    }
    return true;
  }

 private:
  const SafeProjection* projection_;
  const LeggedCapabilityView* capability_;
};

[[nodiscard]] const ConvexCorridorCell* FindCell(
    const CorridorResult& corridor, double s) noexcept {
  for (const auto& cell : corridor.cells) {
    if (s >= cell.centerline_s_begin - kGeometryTolerance &&
        s <= cell.centerline_s_end + kGeometryTolerance) {
      return &cell;
    }
  }
  return corridor.cells.empty() ? nullptr : &corridor.cells.back();
}

struct PendingSection final {
  LeggedCorridorSection section;
  std::size_t depth{};
  bool split_yaw_next{true};
};

}  // namespace

Result<LeggedCorridor> BuildLeggedCorridor(
    const LeggedCorridorRequest& request) {
  if (request.projection.platform_type() != PlatformType::kLegged ||
      request.discrete_plan.states.empty() ||
      request.discrete_plan.grid.yaw_bin_count == 0U ||
      request.config.maximum_regions == 0U ||
      request.config.maximum_halfplanes_per_region < 4U ||
      request.config.maximum_inflation_iterations == 0U) {
    return Invalid("legged_corridor_request",
                   "legged corridor request is incomplete");
  }
  const auto body_radius =
      HorizontalBodyRadius(request.capability.collision_envelope);
  const double error_radius = HorizontalErrorRadius(
      request.capability.certified_state_error_bounds
          .position_bound_m);
  if (!body_radius.has_value() || !std::isfinite(error_radius) ||
      error_radius < 0.0) {
    return Invalid("legged_corridor_request.capability",
                   "body or tracking envelope is invalid");
  }

  std::vector<double> state_s(
      request.discrete_plan.states.size(), 0.0);
  std::vector<PathControlPoint> centerline;
  centerline.reserve(request.discrete_plan.states.size());
  for (std::size_t index = 0U;
       index < request.discrete_plan.states.size(); ++index) {
    const Vec2 position = StatePosition(
        request.discrete_plan.states[index],
        request.projection.geometry);
    if (index > 0U) {
      const Vec2 previous = StatePosition(
          request.discrete_plan.states[index - 1U],
          request.projection.geometry);
      state_s[index] =
          state_s[index - 1U] +
          std::hypot(position.x - previous.x,
                     position.y - previous.y);
    }
    if (centerline.empty() ||
        state_s[index] >
            centerline.back().s_m + kGeometryTolerance) {
      centerline.push_back(
          PathControlPoint{state_s[index], position});
    }
  }

  const auto base = BuildConvexCorridor(
      CorridorRequest{
          .projection = request.projection,
          .validated_centerline = centerline,
          .tightening =
              CorridorTightening{
                  .footprint_support_radius_m = *body_radius,
                  .tracking_error_bound_m = error_radius,
                  .additional_margin_m =
                      request.capability.terrain_thresholds
                          .minimum_body_clearance_m,
                  .platform_half_planes = {},
              },
          .max_planes =
              request.config.maximum_halfplanes_per_region,
          .max_iterations =
              request.config.maximum_inflation_iterations,
      });
  if (base.status != CorridorStatus::kCertified ||
      base.cells.empty() ||
      base.cells.size() > request.config.maximum_regions) {
    return Invalid("legged_corridor.xy",
                   "shared xy corridor requires primitive fallback");
  }

  std::vector<double> unwrapped_yaw(
      request.discrete_plan.states.size(), 0.0);
  const double yaw_step =
      2.0 * std::numbers::pi /
      static_cast<double>(
          request.discrete_plan.grid.yaw_bin_count);
  for (std::size_t index = 0U; index < unwrapped_yaw.size();
       ++index) {
    const double raw =
        static_cast<double>(
            request.discrete_plan.states[index].iyaw) *
        yaw_step;
    unwrapped_yaw[index] =
        index == 0U ? raw
                    : UnwrapNear(raw, unwrapped_yaw[index - 1U]);
  }

  std::vector<LeggedCorridorSection> initial_sections;
  const std::size_t segment_count =
      std::max<std::size_t>(
          request.discrete_plan.states.size() - 1U, 1U);
  initial_sections.reserve(segment_count);
  for (std::size_t segment = 0U; segment < segment_count;
       ++segment) {
    const std::size_t finish =
        std::min(segment + 1U,
                 request.discrete_plan.states.size() - 1U);
    const double middle =
        std::midpoint(state_s[segment], state_s[finish]);
    const auto* xy = FindCell(base, middle);
    const auto z = IntersectHeightIntervals(
        request.discrete_plan.states[segment].reachable_z,
        request.discrete_plan.states[finish].reachable_z);
    if (xy == nullptr || !z.has_value()) {
      return Invalid("legged_corridor.product",
                     "height interval cannot form a certified product");
    }
    ConvexCorridorCell segment_xy = *xy;
    segment_xy.centerline_s_begin = state_s[segment];
    segment_xy.centerline_s_end = state_s[finish];
    initial_sections.push_back(
        LeggedCorridorSection{
            .xy = std::move(segment_xy),
            .z = *z,
            .yaw =
                YawInterval{
                    .min_unwrapped_rad =
                        std::min(unwrapped_yaw[segment],
                                 unwrapped_yaw[finish]),
                    .max_unwrapped_rad =
                        std::max(unwrapped_yaw[segment],
                                 unwrapped_yaw[finish]),
                },
            .terrain_normal =
                TerrainNormalEnvelope{
                    .maximum_normal_deviation_rad =
                        request.capability.terrain_thresholds
                            .maximum_slope_rad,
                    .source_terrain_certification_ref =
                        request.projection.source_map()
                            ->snapshot_ref(),
                },
            .cartesian_product_certified = false,
        });
  }

  const AnalyticProductCertifier analytic(
      request.projection, request.capability);
  const LeggedCartesianProductCertifier& certifier =
      request.product_certifier == nullptr
          ? static_cast<const LeggedCartesianProductCertifier&>(
                analytic)
          : *request.product_certifier;
  std::deque<PendingSection> pending;
  for (auto& section : initial_sections) {
    pending.push_back(
        PendingSection{.section = std::move(section)});
  }

  LeggedCorridor result;
  while (!pending.empty()) {
    PendingSection current = std::move(pending.front());
    pending.pop_front();
    const LeggedProductBox box{
        .xy = current.section.xy,
        .z = current.section.z,
        .yaw = current.section.yaw,
    };
    if (certifier.Certify(box)) {
      current.section.cartesian_product_certified = true;
      result.sections.push_back(std::move(current.section));
      if (result.sections.size() > request.config.maximum_regions) {
        return Invalid("legged_corridor.product",
                       "certified product region limit exceeded");
      }
      continue;
    }
    if (current.depth >= request.config.maximum_split_depth) {
      return Invalid("legged_corridor.product",
                     "cartesian product certification failed");
    }

    const double yaw_width =
        current.section.yaw.max_unwrapped_rad -
        current.section.yaw.min_unwrapped_rad;
    const double z_width =
        current.section.z.max_m - current.section.z.min_m;
    const bool split_yaw =
        (current.split_yaw_next &&
         yaw_width > kGeometryTolerance) ||
        z_width <= kGeometryTolerance;
    if (split_yaw && yaw_width > kGeometryTolerance) {
      const double middle = std::midpoint(
          current.section.yaw.min_unwrapped_rad,
          current.section.yaw.max_unwrapped_rad);
      auto lower = current.section;
      auto upper = current.section;
      lower.yaw.max_unwrapped_rad = middle;
      upper.yaw.min_unwrapped_rad = middle;
      const double s_middle = std::midpoint(
          current.section.xy.centerline_s_begin,
          current.section.xy.centerline_s_end);
      lower.xy.centerline_s_end = s_middle;
      upper.xy.centerline_s_begin = s_middle;
      pending.push_back(
          PendingSection{
              .section = std::move(lower),
              .depth = current.depth + 1U,
              .split_yaw_next = false,
          });
      pending.push_back(
          PendingSection{
              .section = std::move(upper),
              .depth = current.depth + 1U,
              .split_yaw_next = false,
          });
      continue;
    }
    if (z_width > kGeometryTolerance) {
      const double middle =
          std::midpoint(current.section.z.min_m,
                        current.section.z.max_m);
      auto lower = current.section;
      auto upper = current.section;
      lower.z.max_m = middle;
      upper.z.min_m = middle;
      pending.push_back(
          PendingSection{
              .section = std::move(lower),
              .depth = current.depth + 1U,
              .split_yaw_next = true,
          });
      pending.push_back(
          PendingSection{
              .section = std::move(upper),
              .depth = current.depth + 1U,
              .split_yaw_next = true,
          });
      continue;
    }
    return Invalid("legged_corridor.product",
                   "unsafe point product cannot be split");
  }
  return result;
}

}  // namespace lunar::planning::v3
