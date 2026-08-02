#pragma once

#include "lunar_path_planner/v3/contracts/platform_reference.hpp"
#include "lunar_path_planner/v3/contracts/profiles.hpp"
#include "lunar_path_planner/v3/contracts/status.hpp"

namespace lunar::planning::v3 {

struct LeggedCapabilityView final {
  ContentRef content_ref;
  FrameId frame_id;
  ReferencePointId reference_point_id;
  BodyConvexPolytope collision_envelope;
  LeggedTerrainThresholds terrain_thresholds;
  BodyFrameVelocityEnvelope velocity_envelope;
  WheeledOrLeggedErrorBounds certified_state_error_bounds;
  double preferred_body_height_m{};
  double maximum_linear_acceleration_mps2{};
  double maximum_yaw_acceleration_radps2{};

  [[nodiscard]] static Result<LeggedCapabilityView> Create(
      const SafetyCapabilityProfile& profile);
};

}  // namespace lunar::planning::v3
