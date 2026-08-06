#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <string>

namespace lunar::planning {

struct AraStarConfig final {
  double initial_epsilon{2.0};
  double epsilon_decrement{0.25};
  double target_epsilon{1.0};
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
  std::size_t maximum_smoothing_control_points{64U};
  std::size_t maximum_smoothing_samples{512U};
  std::size_t maximum_iterations{128U};
  std::size_t maximum_trust_region_reductions{8U};
  double initial_trust_region_m{0.25};
  double minimum_trust_region_m{0.005};
  double constraint_tolerance{1.0e-6};
  bool require_smoothed_execution{false};
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
};

struct LeggedPlannerConfig final {
  double xy_resolution_m{0.25};
  std::size_t yaw_bin_count{32U};
};

struct HopperPlannerConfig final {};

struct GlobalMapConfig final {
  double base_resolution_m{0.2};
  std::size_t maximum_level{4U};
  std::size_t maximum_cells{1'048'576U};
  std::size_t maximum_axis_cells{4'096U};
};

struct GlobalSearchConfig final {
  std::size_t maximum_preview_points{4'096U};
  double slope_weight{1.0};
  double roughness_weight{1.0};
  double clearance_weight{1.0};
};

struct LocalFrontierConfig final {
  double wheel_horizon_m{4.0};
  double legged_horizon_m{3.0};
  double additional_corridor_margin_m{0.4};
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
  GlobalMapConfig global_map;
  GlobalSearchConfig global_search;
  LocalFrontierConfig local_frontier;
  bool stable_candidate_order{true};
};

struct WorldSnapshot;

namespace hierarchical {

struct ExpectedMapLevelResult final {
  std::optional<std::size_t> level;
  std::size_t width{};
  std::size_t height{};
  double resolution_m{};
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return level.has_value() && reason_code.empty();
  }
};

struct MapLevelValidationResult final {
  std::optional<std::size_t> global_level;
  std::string reason_code;

  [[nodiscard]] bool ok() const noexcept {
    return global_level.has_value() && reason_code.empty();
  }
};

[[nodiscard]] ExpectedMapLevelResult ExpectedGlobalMapLevel(
    double size_x_m,
    double size_y_m,
    const GlobalMapConfig& config) noexcept;

[[nodiscard]] MapLevelValidationResult ValidateMapLevels(
    const WorldSnapshot& world,
    const GlobalMapConfig& config) noexcept;

}  // namespace hierarchical

} // namespace lunar::planning
