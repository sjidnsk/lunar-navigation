#include "lunar_path_planner/v3/wheel/wheel_corridor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace lunar::planning::v3 {
namespace {

constexpr double kGeometryTolerance = 1.0e-9;

[[nodiscard]] CorridorResult Fallback(
    std::string reason_code) {
  return {
      .status = CorridorStatus::kFallbackRequired,
      .fallback =
          CorridorFallback::kUseDiscreteValidatedPrimitives,
      .cells = {},
      .reason_code = std::move(reason_code),
      .iterations = 0U,
  };
}

[[nodiscard]] CorridorResult Invalid(
    std::string reason_code) {
  return {
      .status = CorridorStatus::kInvalidRequest,
      .fallback =
          CorridorFallback::kUseDiscreteValidatedPrimitives,
      .cells = {},
      .reason_code = std::move(reason_code),
      .iterations = 0U,
  };
}

[[nodiscard]] bool FinitePose(const PoseXyzYaw& pose) noexcept {
  return std::isfinite(pose.position_m.x) &&
         std::isfinite(pose.position_m.y) &&
         std::isfinite(pose.position_m.z) &&
         std::isfinite(pose.yaw_rad);
}

[[nodiscard]] bool SamePosePosition(
    const PoseXyzYaw& lhs, const PoseXyzYaw& rhs) noexcept {
  return std::hypot(lhs.position_m.x - rhs.position_m.x,
                    lhs.position_m.y - rhs.position_m.y) <=
             kGeometryTolerance &&
         std::abs(lhs.position_m.z - rhs.position_m.z) <=
             kGeometryTolerance;
}

[[nodiscard]] double FootprintSupport(
    const ExtrudedConvexFootprint& footprint) noexcept {
  double support = 0.0;
  for (const Vec2& vertex : footprint.vertices_xy_m) {
    if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    support = std::max(support, std::hypot(vertex.x, vertex.y));
  }
  return support;
}

}  // namespace

CorridorResult BuildWheelCorridor(
    const WheelCorridorRequest& request) {
  const auto& config = request.config;
  if (request.edges.empty() ||
      request.envelope.footprint.vertices_xy_m.size() < 3U ||
      config.maximum_regions == 0U ||
      config.maximum_inflation_iterations == 0U ||
      config.maximum_halfplanes_per_region < 4U ||
      config.maximum_centerline_samples == 0U ||
      !std::isfinite(config.sampling_spacing_m) ||
      config.sampling_spacing_m <= 0.0 ||
      !std::isfinite(config.additional_margin_m) ||
      config.additional_margin_m < 0.0 ||
      !std::isfinite(config.maximum_curvature_per_m) ||
      config.maximum_curvature_per_m <= 0.0 ||
      !std::isfinite(
          request.envelope.horizontal_tracking_error_m) ||
      request.envelope.horizontal_tracking_error_m < 0.0 ||
      !std::isfinite(request.envelope.minimum_clearance_m) ||
      request.envelope.minimum_clearance_m < 0.0) {
    return Invalid("invalid_wheel_corridor_request");
  }
  const double support =
      FootprintSupport(request.envelope.footprint);
  if (!std::isfinite(support)) {
    return Invalid("invalid_wheel_collision_envelope");
  }

  std::vector<PathControlPoint> centerline;
  centerline.reserve(
      std::min(config.maximum_centerline_samples,
               request.edges.size() * 2U + 1U));
  double arc_length_m = 0.0;
  const auto append_point = [&](const Vec2& point,
                                double s_m,
                                auto& output) {
    if (!output.empty() &&
        std::hypot(output.back().position_m.x - point.x,
                   output.back().position_m.y - point.y) <=
            kGeometryTolerance) {
      return;
    }
    output.push_back({.s_m = s_m, .position_m = point});
  };

  const WheelLatticeEdge* previous = nullptr;
  for (const WheelLatticeEdge& edge : request.edges) {
    if (!FinitePose(edge.source_pose) ||
        !FinitePose(edge.target_pose) ||
        (previous != nullptr &&
         !SamePosePosition(previous->target_pose,
                           edge.source_pose))) {
      return Invalid("invalid_wheel_edge_chain");
    }
    previous = &edge;
    if (edge.primitive_kind == WheelPrimitiveKind::kSpin) {
      return Invalid("spin_not_supported_by_wheel_corridor");
    }

    const double delta_x =
        edge.target_pose.position_m.x -
        edge.source_pose.position_m.x;
    const double delta_y =
        edge.target_pose.position_m.y -
        edge.source_pose.position_m.y;
    const double distance_m = std::hypot(delta_x, delta_y);
    const double yaw_change =
        std::abs(edge.target_pose.yaw_rad -
                 edge.source_pose.yaw_rad);
    if (distance_m <= kGeometryTolerance) {
      if (yaw_change > kGeometryTolerance &&
          edge.primitive_kind !=
              WheelPrimitiveKind::kModeSwitch) {
        return Fallback("zero_length_drive_with_yaw_change");
      }
      if (centerline.empty()) {
        append_point(
            {edge.source_pose.position_m.x,
             edge.source_pose.position_m.y},
            arc_length_m, centerline);
      }
      continue;
    }
    if (yaw_change / distance_m >
        config.maximum_curvature_per_m +
            kGeometryTolerance) {
      return Fallback("curvature_limit_exceeded");
    }

    const auto subdivisions = static_cast<std::size_t>(
        std::max(
            1.0,
            std::ceil(distance_m /
                      config.sampling_spacing_m)));
    if (subdivisions >
            config.maximum_centerline_samples ||
        centerline.size() >
            config.maximum_centerline_samples - subdivisions -
                (centerline.empty() ? 1U : 0U)) {
      return Fallback("centerline_sample_limit_exceeded");
    }
    if (centerline.empty()) {
      append_point(
          {edge.source_pose.position_m.x,
           edge.source_pose.position_m.y},
          arc_length_m, centerline);
    }
    for (std::size_t index = 1U; index <= subdivisions;
         ++index) {
      const double fraction =
          static_cast<double>(index) /
          static_cast<double>(subdivisions);
      append_point(
          {
              edge.source_pose.position_m.x +
                  fraction * delta_x,
              edge.source_pose.position_m.y +
                  fraction * delta_y,
          },
          arc_length_m + fraction * distance_m,
          centerline);
    }
    arc_length_m += distance_m;
  }
  if (centerline.empty()) {
    return Invalid("empty_wheel_centerline");
  }

  auto result = BuildConvexCorridor(
      {
          .projection = request.projection,
          .validated_centerline = centerline,
          .tightening =
              {
                  .footprint_support_radius_m = support,
                  .tracking_error_bound_m =
                      request.envelope
                          .horizontal_tracking_error_m,
                  .additional_margin_m =
                      request.envelope.minimum_clearance_m +
                      config.additional_margin_m,
                  .platform_half_planes = {},
              },
          .max_planes =
              config.maximum_halfplanes_per_region,
          .max_iterations =
              config.maximum_inflation_iterations,
      });
  if (result.status == CorridorStatus::kCertified &&
      result.cells.size() > config.maximum_regions) {
    return Fallback("corridor_region_limit_exceeded");
  }
  return result;
}

}  // namespace lunar::planning::v3
