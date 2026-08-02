#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "lunar_path_planner/v3/api/planner_v3.hpp"
#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"

namespace lunar::planning::v3::system_test {

enum class MapScale {
  kTenMeter,
  kHundredMeter,
  kThousandMeter,
};

enum class MapScenario {
  kOpenKnown,
  kDetour,
  kKnownObstacleAtGoal,
  kUnknownGoalWithSafeFrontier,
};

enum class G1ReferenceScenario {
  kLowKnown,
  kMediumKnown,
  kHighFrontier,
};

enum class ExpectedExperimentOutcome {
  kNewReferenceReady,
  kSafeFrontierReferenceReady,
};

struct ScenarioRegion final {
  enum class Kind {
    kHardObstacle,
    kSyntheticTerrainObstacleProxy,
    kUnknown,
    kHighSlope,
    kHighRoughness,
  };

  Kind kind{};
  std::vector<Vec2> polygon_xy_m;
};

struct G1ReferenceProvenance final {
  G1ReferenceScenario reference{G1ReferenceScenario::kLowKnown};
  std::string source_scenario_id;
  std::string source_scenario_hash;
  std::string proxy_seed_hex;
  double source_hard_obstacle_fraction{};
  std::string source_kind;
  bool physical_obstacle_cells_written{};
  std::string derivation;
};

struct ScenarioDescription final {
  MapScale scale{MapScale::kTenMeter};
  MapScenario scene{MapScenario::kOpenKnown};
  std::optional<G1ReferenceProvenance> g1_reference;
  double width_m{};
  double height_m{};
  Vec2 start_xy_m;
  Vec2 goal_xy_m;
  double nominal_plan_distance_m{};
  ExpectedExperimentOutcome expected_outcome{
      ExpectedExperimentOutcome::kNewReferenceReady};
  std::vector<ScenarioRegion> regions;
};

enum class RegistryFault {
  kNone,
  kMissingCapability,
  kCapabilityBindingsMismatch,
};

struct SystemScenario final {
  PlanningRequest request;
  std::shared_ptr<const ContractObjectRegistry> registry;
  std::unique_ptr<SafeProjectionCache> projection_cache;
  ScenarioDescription description;
};

[[nodiscard]] SystemScenario MakeSystemScenario(
    PlatformType platform_type,
    MapScenario map_scenario = MapScenario::kOpenKnown,
    RegistryFault registry_fault = RegistryFault::kNone);

[[nodiscard]] SystemScenario MakeMultiscaleSystemScenario(
    PlatformType platform_type,
    MapScale map_scale,
    MapScenario map_scenario);

[[nodiscard]] SystemScenario MakeMultiscaleSystemScenario(
    PlatformType platform_type,
    MapScale map_scale,
    G1ReferenceScenario reference_scenario);

[[nodiscard]] std::unique_ptr<PlannerV3> MakePlanner(
    SystemScenario& scenario);

[[nodiscard]] PreviousExecutionContext MakeGroundExecutionContext(
    const SystemScenario& scenario,
    bool stale_source_map = false);

[[nodiscard]] Result<Sha256Digest> ResponseHash(
    const PlanningResponse& response);

[[nodiscard]] std::string DescribeIssues(
    const ValidationReport& report);

}  // namespace lunar::planning::v3::system_test
