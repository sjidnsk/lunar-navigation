#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "lunar_planner_core/types/motion_reference.hpp"
#include "shared/map_snapshot.hpp"

namespace lunar::planning::hopper {

struct BallisticState final {
  Vec3 position_m;
  Vec3 velocity_mps;
};

struct BallisticArc final {
  Vec3 launch_position_m;
  Vec3 landing_position_m;
  Vec3 gravity_mps2;
  Vec3 launch_velocity_mps;
  Vec3 landing_velocity_mps;
  double flight_time_s{};
};

struct BallisticSolveResult final {
  std::optional<BallisticArc> arc;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return arc.has_value() && reason_code.empty();
  }
};

struct CertifiedLandingRegion final {
  shared::GridCell seed_cell;
  Vec3 aim_position_on_surface_m;
  Vec3 plane_normal;
  std::vector<Vec3> boundary_m;
  double area_m2{};
  double maximum_slope_rad{};
  double maximum_roughness_m{};
  double maximum_plane_residual_m{};
  double minimum_clearance_m{};
};

enum class LandingRegionStatus {
  kCertified,
  kInfeasible,
  kInvalidRequest,
  kCanceled,
  kResourceExhausted,
};

struct LandingRegionResult final {
  LandingRegionStatus status{LandingRegionStatus::kInvalidRequest};
  std::optional<CertifiedLandingRegion> region;
  std::size_t inspected_cells{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return status == LandingRegionStatus::kCertified && region.has_value() &&
        reason_code.empty();
  }
};

struct FlightTubeCertificationResult final {
  bool certified{};
  bool canceled{};
  std::size_t section_count{};
  std::size_t overlapped_cell_count{};
  double minimum_clearance_m{};
  double radius_m{};
  std::string reason_code;
};

struct HopCertificationResult final {
  std::optional<HopSegment> segment;
  bool canceled{};
  bool resource_exhausted{};
  std::size_t attempted_candidates{};
  double cost{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return segment.has_value() && !canceled && !resource_exhausted &&
        reason_code.empty();
  }
};

}  // namespace lunar::planning::hopper
