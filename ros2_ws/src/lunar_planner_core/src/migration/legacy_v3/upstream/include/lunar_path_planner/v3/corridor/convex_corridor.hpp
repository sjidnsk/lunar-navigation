#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/contracts/base_types.hpp"
#include "lunar_path_planner/v3/map/safe_projection.hpp"

namespace lunar::planning::v3 {

struct PathControlPoint final {
  double s_m{};
  Vec2 position_m;
};

struct HalfPlane2 final {
  Vec2 outward_unit_normal;
  double upper_offset_m{};
};

struct CorridorTightening final {
  double footprint_support_radius_m{};
  double tracking_error_bound_m{};
  double additional_margin_m{};
  std::vector<HalfPlane2> platform_half_planes;
};

struct ConvexCorridorCell final {
  std::string stable_cell_id;
  double centerline_s_begin{};
  double centerline_s_end{};
  std::vector<HalfPlane2> half_planes;
};

struct CorridorRequest final {
  const SafeProjection& projection;
  std::span<const PathControlPoint> validated_centerline;
  CorridorTightening tightening;
  std::size_t max_planes{};
  std::size_t max_iterations{};
};

enum class CorridorStatus {
  kCertified,
  kFallbackRequired,
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
    const CorridorRequest& request);

}  // namespace lunar::planning::v3
