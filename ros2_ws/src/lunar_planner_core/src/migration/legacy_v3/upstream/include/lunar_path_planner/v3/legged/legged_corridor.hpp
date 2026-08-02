#pragma once

#include <vector>

#include "lunar_path_planner/v3/corridor/convex_corridor.hpp"
#include "lunar_path_planner/v3/legged/legged_search.hpp"

namespace lunar::planning::v3 {

struct YawInterval final {
  double min_unwrapped_rad{};
  double max_unwrapped_rad{};
};

struct LeggedProductBox final {
  ConvexCorridorCell xy;
  HeightInterval z;
  YawInterval yaw;
};

class LeggedCartesianProductCertifier {
 public:
  virtual ~LeggedCartesianProductCertifier() = default;

  [[nodiscard]] virtual bool Certify(
      const LeggedProductBox& box) const noexcept = 0;
};

struct LeggedCorridorSection final {
  ConvexCorridorCell xy;
  HeightInterval z;
  YawInterval yaw;
  TerrainNormalEnvelope terrain_normal;
  bool cartesian_product_certified{};
};

struct LeggedCorridor final {
  std::vector<LeggedCorridorSection> sections;
};

struct LeggedCorridorRequest final {
  const SafeProjection& projection;
  const LeggedDiscretePlan& discrete_plan;
  const LeggedCapabilityView& capability;
  CorridorConfig config;
  const LeggedCartesianProductCertifier* product_certifier{};
};

[[nodiscard]] Result<LeggedCorridor> BuildLeggedCorridor(
    const LeggedCorridorRequest& request);

}  // namespace lunar::planning::v3
