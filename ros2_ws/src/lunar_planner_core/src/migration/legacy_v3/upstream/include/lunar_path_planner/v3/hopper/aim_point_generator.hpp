#pragma once

#include <optional>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/hopper/landing_region.hpp"

namespace lunar::planning::v3 {

enum class AimPointSource : std::uint8_t {
  kChebyshevCenter,
  kGoalProjection,
  kMissionDirectionInset,
  kSupportDirectionInset,
};

struct AimPointCandidate final {
  std::string aim_point_id;
  std::string region_id;
  FrameId frame_id;
  Eigen::Vector2d position_uv;
  Eigen::Vector3d position_m;
  AimPointSource source{AimPointSource::kChebyshevCenter};
  double minimum_boundary_distance_m{};
};

class AimPointGenerator final {
 public:
  AimPointGenerator(
      HopperCapabilityView capability,
      HopperPlannerLimits limits);

  [[nodiscard]] std::vector<AimPointCandidate> generate(
      const TerrainCertifiedLandingRegion& region,
      const GoalRegion& goal,
      const std::optional<Eigen::Vector3d>&
          mission_direction_frame) const;

 private:
  HopperCapabilityView capability_;
  HopperPlannerLimits limits_;
};

[[nodiscard]] bool point_strictly_inside_region(
    const AimPointCandidate& point,
    const TerrainCertifiedLandingRegion& region);

}  // namespace lunar::planning::v3
