#pragma once

#include <span>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/hopper/safe_pose_mask.hpp"

namespace lunar::planning::v3 {

struct TerrainCertification final {
  std::string certification_id;
  ContentRef source_snapshot_ref;
  double maximum_slope_rad{};
  double maximum_roughness_m{};
  double maximum_plane_residual_m{};
  double minimum_clearance_m{};
};

struct TerrainCertifiedLandingRegion final {
  std::string region_id;
  FrameId frame_id;
  LandingPlane landing_plane;
  ConvexPolygon2d vertices_uv_ccw;
  CircularYawInterval allowed_yaw_interval;
  TerrainCertification terrain_certification;
  double inward_safety_margin_m{};
  Cell source_seed;
};

class LandingRegionGenerator final {
 public:
  LandingRegionGenerator(
      HopperCapabilityView capability,
      HopperPlannerLimits limits);

  [[nodiscard]] std::vector<TerrainCertifiedLandingRegion> generate(
      const ImmutableMapSnapshot& map,
      const SafePoseMask& mask,
      std::span<const LandingSeed> seeds) const;

 private:
  HopperCapabilityView capability_;
  HopperPlannerLimits limits_;
};

[[nodiscard]] ValidationReport validate_terrain_certified_region(
    const TerrainCertifiedLandingRegion& region,
    const ImmutableMapSnapshot& map,
    const SafePoseMask& mask,
    const HopperCapabilityView& capability);

[[nodiscard]] TerrainCertifiedLandingRegion simplify_region_inward(
    const TerrainCertifiedLandingRegion& region,
    std::size_t maximum_vertex_count);

}  // namespace lunar::planning::v3
