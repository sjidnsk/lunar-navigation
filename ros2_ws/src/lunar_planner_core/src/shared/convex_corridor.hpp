#pragma once

#include <cstddef>
#include <stop_token>
#include <string>
#include <vector>

#include "lunar_planner_core/types/geometry.hpp"
#include "lunar_planner_core/types/planner_config.hpp"
#include "shared/safe_projection.hpp"

namespace lunar::planning::shared {

struct HalfPlane2 final {
  Vec2 outward_unit_normal;
  double upper_offset_m{};

  bool operator==(const HalfPlane2&) const = default;
};

struct CorridorTightening final {
  double footprint_support_radius_m{};
  double tracking_error_bound_m{};
  double additional_margin_m{};
};

struct ConvexCorridorCell final {
  std::size_t stable_index{};
  double centerline_s_begin_m{};
  double centerline_s_end_m{};
  std::vector<HalfPlane2> half_planes;

  bool operator==(const ConvexCorridorCell&) const = default;
};

enum class CorridorStatus {
  kCertified,
  kFallbackRequired,
  kCanceled,
  kInvalidRequest,
};

enum class CorridorFallback {
  kNone,
  kUseDiscreteValidatedPrimitives,
};

struct CorridorResult final {
  CorridorStatus status{CorridorStatus::kInvalidRequest};
  CorridorFallback fallback{
      CorridorFallback::kUseDiscreteValidatedPrimitives};
  std::vector<ConvexCorridorCell> cells;
  std::string reason_code;
  std::size_t iterations{};
};

[[nodiscard]] CorridorResult BuildConvexCorridor(
    const SafeProjection& projection,
    const std::vector<Vec2>& validated_centerline,
    const CorridorTightening& tightening,
    const CorridorConfig& config,
    std::stop_token stop_token);

}  // namespace lunar::planning::shared
