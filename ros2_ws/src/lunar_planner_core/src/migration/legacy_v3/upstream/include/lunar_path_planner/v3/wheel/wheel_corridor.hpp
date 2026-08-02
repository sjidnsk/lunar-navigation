#pragma once

#include <cstddef>
#include <span>

#include "lunar_path_planner/v3/corridor/convex_corridor.hpp"
#include "lunar_path_planner/v3/wheel/wheel_lattice.hpp"

namespace lunar::planning::v3 {

struct WheelCollisionEnvelope final {
  ExtrudedConvexFootprint footprint;
  double horizontal_tracking_error_m{};
  double minimum_clearance_m{};
};

struct WheelCorridorConfig final {
  std::size_t maximum_regions{};
  std::size_t maximum_inflation_iterations{};
  std::size_t maximum_halfplanes_per_region{};
  std::size_t maximum_centerline_samples{};
  double sampling_spacing_m{};
  double additional_margin_m{};
  double maximum_curvature_per_m{};
};

struct WheelCorridorRequest final {
  std::span<const WheelLatticeEdge> edges;
  const SafeProjection& projection;
  const WheelCollisionEnvelope& envelope;
  WheelCorridorConfig config;
};

[[nodiscard]] CorridorResult BuildWheelCorridor(
    const WheelCorridorRequest& request);

}  // namespace lunar::planning::v3
