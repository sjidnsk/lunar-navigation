#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace lunar::planning {

struct SearchResourceLimits final {
  std::size_t maximum_expanded_states{100'000U};
  std::size_t maximum_reopened_states{50'000U};
  std::size_t maximum_generated_candidates{100'000U};
  std::size_t maximum_open_states{100'000U};
  std::size_t maximum_memory_bytes{256U * 1024U * 1024U};
};

struct AraStarConfig final {
  double initial_epsilon{2.0};
  double epsilon_decrement{0.25};
  double target_epsilon{1.0};
  SearchResourceLimits resources;
};

struct CorridorConfig final {
  std::size_t maximum_regions{64U};
  std::size_t maximum_inflation_iterations{128U};
  std::size_t maximum_halfplanes_per_region{32U};
  std::size_t maximum_split_depth{8U};
  double minimum_overlap_m{0.05};
  double sampling_spacing_m{0.1};
};

struct OptimizationConfig final {
  std::size_t maximum_iterations{128U};
  std::size_t maximum_trust_region_reductions{8U};
  double initial_trust_region_m{0.25};
  double minimum_trust_region_m{0.005};
  double constraint_tolerance{1.0e-6};
};

struct MapSafetyConfig final {
  double project_maximum_slope_rad{std::numbers::pi / 6.0};
  double maximum_elevation_variance_m2{0.04};
  double maximum_obstacle_variance_m2{0.04};
  double maximum_observation_age_s{2.0};
  double minimum_observation_quality{0.8};
  std::uint32_t minimum_observation_count{1U};
};

struct WheelPlannerConfig final {
  double xy_resolution_m{0.25};
  std::size_t yaw_bin_count{32U};
  std::size_t maximum_terminal_candidates{64U};
  std::size_t continuous_validation_maximum_subdivisions{32U};
};

struct LeggedPlannerConfig final {
  double xy_resolution_m{0.25};
  std::size_t yaw_bin_count{32U};
  std::size_t maximum_terminal_candidates{64U};
  std::size_t maximum_height_interval_splits{16U};
  std::size_t continuous_validation_maximum_subdivisions{32U};
};

struct HopperPlannerConfig final {
  std::size_t maximum_landing_regions{64U};
  std::size_t maximum_graph_nodes{128U};
  std::size_t maximum_graph_out_degree{8U};
  std::size_t maximum_nominal_aim_points_per_region{16U};
  std::size_t maximum_certification_attempts{64U};
  std::size_t maximum_flight_tube_sections{128U};
  std::size_t maximum_authorized_hops{1U};
};

struct PlannerConfig final {
  std::chrono::nanoseconds maximum_input_skew{std::chrono::seconds{1}};
  AraStarConfig search;
  CorridorConfig corridor;
  OptimizationConfig optimization;
  MapSafetyConfig map_safety;
  WheelPlannerConfig wheel;
  LeggedPlannerConfig legged;
  HopperPlannerConfig hopper;
  bool stable_candidate_order{true};
};

}  // namespace lunar::planning
