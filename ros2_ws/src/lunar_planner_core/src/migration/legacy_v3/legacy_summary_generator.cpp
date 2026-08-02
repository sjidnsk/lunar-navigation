#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "fixtures/system/planner_v3_system_fixture.hpp"
#include "lunar_path_planner/v3/codec/json_codec.hpp"

namespace lunar::planning::v3::migration {
namespace {

using Json = nlohmann::json;
using system_test::MapScenario;

class ThrowingRegistry final : public ContractObjectRegistry {
 public:
  explicit ThrowingRegistry(const ContractObjectRegistry& delegate)
      : delegate_(delegate) {}

  [[nodiscard]] std::shared_ptr<const ImmutableMapSnapshot> FindMapSnapshot(
      const ContentRef&, std::string_view) const override {
    throw std::runtime_error{"injected legacy numerical failure"};
  }

  [[nodiscard]] std::shared_ptr<const SafetyCapabilityProfile>
  FindSafetyCapability(const ContentRef& ref) const override {
    return delegate_.FindSafetyCapability(ref);
  }

  [[nodiscard]] std::shared_ptr<const PlannerAlgorithmConfig>
  FindAlgorithmConfig(const ContentRef& ref) const override {
    return delegate_.FindAlgorithmConfig(ref);
  }

  [[nodiscard]] std::shared_ptr<const LearnedCostSnapshot> FindLearnedCost(
      const ContentRef& ref, std::string_view handle) const override {
    return delegate_.FindLearnedCost(ref, handle);
  }

  [[nodiscard]] Result<ResolvedCapabilityBindings> ResolveCapabilityBindings(
      const SafetyCapabilityProfile& profile) const override {
    return delegate_.ResolveCapabilityBindings(profile);
  }

  [[nodiscard]] Result<std::shared_ptr<const ImmutableContractObject>> Resolve(
      const ContentRef& ref, ContractObjectKind kind) const override {
    return delegate_.Resolve(ref, kind);
  }

 private:
  const ContractObjectRegistry& delegate_;
};

class MapOverrideRegistry final : public ContractObjectRegistry {
 public:
  MapOverrideRegistry(
      const ContractObjectRegistry& delegate,
      std::shared_ptr<const ImmutableMapSnapshot> map)
      : delegate_(delegate), map_(std::move(map)) {}

  [[nodiscard]] std::shared_ptr<const ImmutableMapSnapshot> FindMapSnapshot(
      const ContentRef& ref, std::string_view handle) const override {
    return map_->snapshot_ref() == ref && map_->immutable_data_handle() == handle
               ? map_
               : nullptr;
  }

  [[nodiscard]] std::shared_ptr<const SafetyCapabilityProfile>
  FindSafetyCapability(const ContentRef& ref) const override {
    return delegate_.FindSafetyCapability(ref);
  }

  [[nodiscard]] std::shared_ptr<const PlannerAlgorithmConfig>
  FindAlgorithmConfig(const ContentRef& ref) const override {
    return delegate_.FindAlgorithmConfig(ref);
  }

  [[nodiscard]] std::shared_ptr<const LearnedCostSnapshot> FindLearnedCost(
      const ContentRef& ref, std::string_view handle) const override {
    return delegate_.FindLearnedCost(ref, handle);
  }

  [[nodiscard]] Result<ResolvedCapabilityBindings> ResolveCapabilityBindings(
      const SafetyCapabilityProfile& profile) const override {
    return delegate_.ResolveCapabilityBindings(profile);
  }

  [[nodiscard]] Result<std::shared_ptr<const ImmutableContractObject>> Resolve(
      const ContentRef& ref, ContractObjectKind kind) const override {
    return delegate_.Resolve(ref, kind);
  }

 private:
  const ContractObjectRegistry& delegate_;
  std::shared_ptr<const ImmutableMapSnapshot> map_;
};

[[nodiscard]] PlatformType ParsePlatform(const std::string& platform) {
  if (platform == "wheel") {
    return PlatformType::kWheeled;
  }
  if (platform == "legged") {
    return PlatformType::kLegged;
  }
  if (platform == "hopper") {
    return PlatformType::kHopper;
  }
  throw std::invalid_argument{"unsupported platform: " + platform};
}

[[nodiscard]] std::string OutcomeName(const PlanningOutcome outcome) {
  switch (outcome) {
    case PlanningOutcome::kNewReferenceReady:
      return "NEW_REFERENCE_AVAILABLE";
    case PlanningOutcome::kSafeFrontierReferenceReady:
      return "SAFE_FRONTIER_REFERENCE_AVAILABLE";
    case PlanningOutcome::kNoKnownSafeRoute:
      return "NO_KNOWN_SAFE_ROUTE";
    case PlanningOutcome::kGoalInfeasible:
      return "GOAL_INFEASIBLE";
    case PlanningOutcome::kInvalidRequest:
      return "INVALID_REQUEST";
    case PlanningOutcome::kStaleInput:
      return "STALE_INPUT";
    case PlanningOutcome::kNumericalFailure:
      return "NUMERICAL_FAILURE";
    case PlanningOutcome::kResourceLimit:
      return "RESOURCE_EXHAUSTED";
    case PlanningOutcome::kActiveReferenceInvalidated:
      return "ACTIVE_REFERENCE_INVALIDATED";
  }
  throw std::logic_error{"unhandled legacy planning outcome"};
}

void MoveStateOutsideMap(PlanningRequest& request) {
  std::visit(
      [](auto& state) {
        state.position_m.x = -10.0;
        state.position_m.y = -10.0;
      },
      request.current_state);
}

template <typename Value>
[[nodiscard]] std::vector<Value> CopySpan(std::span<const Value> values) {
  return {values.begin(), values.end()};
}

[[nodiscard]] std::shared_ptr<const ImmutableMapSnapshot>
MakeHopperGoalKnownInfeasible(PlanningRequest& request) {
  auto* point = std::get_if<PointGoal>(&request.goal.target);
  if (point == nullptr) {
    throw std::logic_error{"hopper baseline expected a point goal"};
  }
  point->position_m = {4.25, 3.25, 0.0};
  point->position_tolerance_m = 0.05;

  const auto& source = *request.map_snapshot;
  const auto normals = source.SurfaceNormals();
  MapSnapshotInput input{
      .snapshot_ref =
          ContentRef{
              .id = "system-hopper-goal-infeasible-map",
              .revision = 1U,
              .content_hash = std::string(64U, 'd'),
          },
      .map_revision = source.map_revision() + 1U,
      .immutable_data_handle = "system-hopper-goal-infeasible-map-handle",
      .source_time = source.source_time(),
      .bounds = source.bounds(),
      .geometry = source.geometry(),
      .layer_manifest = CopySpan(source.layer_manifest()),
      .known_mask = CopySpan(source.KnownMask()),
      .elevation_m = CopySpan(source.ElevationMeters()),
      .normal_x = CopySpan(normals.x),
      .normal_y = CopySpan(normals.y),
      .normal_z = CopySpan(normals.z),
      .roughness_m = CopySpan(source.RoughnessMeters()),
      .hard_obstacle_mask = CopySpan(source.HardObstacleMask()),
      .confidence = CopySpan(source.Confidence()),
      .esdf_m = CopySpan(source.EsdfMeters()),
      .static_speed_limit_mps = CopySpan(source.StaticSpeedLimitMps()),
  };
  constexpr std::size_t kGoalCellX = 8U;
  constexpr std::size_t kGoalCellY = 6U;
  input.hard_obstacle_mask[
      kGoalCellY * input.geometry.width + kGoalCellX] = 1U;
  auto created = ImmutableMapSnapshot::Create(input);
  if (!IsOk(created)) {
    const Error& error = std::get<Error>(created);
    throw std::runtime_error{"cannot create goal-infeasible map: " +
                             error.field_path + ": " + error.message};
  }
  request.map_snapshot =
      std::get<std::shared_ptr<const ImmutableMapSnapshot>>(std::move(created));
  return request.map_snapshot;
}

[[nodiscard]] PlanningResponse RunOnce(
    const PlatformType platform, const std::string& setup) {
  const MapScenario map_scenario =
      setup == "goal_infeasible" && platform != PlatformType::kHopper
          ? MapScenario::kKnownObstacleAtGoal
          : MapScenario::kOpenKnown;
  auto scenario = system_test::MakeSystemScenario(platform, map_scenario);
  if (setup == "no_safe_route") {
    MoveStateOutsideMap(scenario.request);
  }

  if (setup == "numerical_failure") {
    static const SemanticValidator validator;
    ThrowingRegistry registry{*scenario.registry};
    auto planner = MakeDefaultPlannerV3(
        validator, registry, *scenario.projection_cache);
    return planner->Plan(scenario.request);
  }
  if (setup == "goal_infeasible" && platform == PlatformType::kHopper) {
    auto map = MakeHopperGoalKnownInfeasible(scenario.request);
    static const SemanticValidator validator;
    MapOverrideRegistry registry{*scenario.registry, std::move(map)};
    auto planner = MakeDefaultPlannerV3(
        validator, registry, *scenario.projection_cache);
    return planner->Plan(scenario.request);
  }
  auto planner = system_test::MakePlanner(scenario);
  return planner->Plan(scenario.request);
}

[[nodiscard]] Json Normalize(const PlanningResponse& response) {
  const bool has_reference = response.new_reference_bundle.has_value();
  bool collision_free = false;
  bool constraints_satisfied = false;
  if (has_reference) {
    const ValidationSummary& validation =
        response.new_reference_bundle->validation_summary;
    collision_free = validation.hard_constraints_passed &&
                     validation.continuous_validation_passed;
    constraints_satisfied = validation.hard_constraints_passed;
  }

  Json cost = nullptr;
  if (response.call_diagnostics.expected_execution_time.has_value()) {
    constexpr double kNanosecondsPerSecond = 1.0e9;
    cost = static_cast<double>(
               response.call_diagnostics.expected_execution_time->value
                   .count()) /
           kNanosecondsPerSecond;
  }
  return Json{
      {"reachable", has_reference},
      {"planning_outcome", OutcomeName(response.planning_outcome)},
      {"collision_free", collision_free},
      {"goal_reached",
       response.planning_outcome == PlanningOutcome::kNewReferenceReady},
      {"platform_constraints_satisfied", constraints_satisfied},
      {"cost", std::move(cost)},
  };
}

[[nodiscard]] std::string ExpectedOutcomeFor(const std::string& setup) {
  if (setup == "open_known" || setup == "deterministic_repeat") {
    return "NEW_REFERENCE_AVAILABLE";
  }
  if (setup == "no_safe_route") {
    return "NO_KNOWN_SAFE_ROUTE";
  }
  if (setup == "goal_infeasible") {
    return "GOAL_INFEASIBLE";
  }
  if (setup == "numerical_failure") {
    return "NUMERICAL_FAILURE";
  }
  throw std::invalid_argument{"unsupported setup: " + setup};
}

[[nodiscard]] Json GenerateCase(
    const PlatformType platform, const Json& definition) {
  const std::string case_id = definition.at("case_id").get<std::string>();
  const std::string setup = definition.at("setup").get<std::string>();
  const int repetitions = definition.at("repetitions").get<int>();
  if (repetitions < 1) {
    throw std::invalid_argument{"repetitions must be positive"};
  }

  Json stable;
  for (int repetition = 0; repetition < repetitions; ++repetition) {
    Json current = Normalize(RunOnce(platform, setup));
    if (repetition == 0) {
      stable = std::move(current);
    } else if (current != stable) {
      throw std::runtime_error{"legacy semantic summary is nondeterministic: " +
                               case_id};
    }
  }
  const std::string actual = stable.at("planning_outcome").get<std::string>();
  const std::string expected = ExpectedOutcomeFor(setup);
  if (actual != expected) {
    throw std::runtime_error{"unexpected outcome for " + case_id + ": " +
                             actual + " (expected " + expected + ")"};
  }
  stable["case_id"] = case_id;
  return stable;
}

[[nodiscard]] Json LoadJson(const std::string& path) {
  std::ifstream stream{path};
  if (!stream) {
    throw std::runtime_error{"cannot open fixture: " + path};
  }
  return Json::parse(stream);
}

void Generate(const std::string& output_path, char** fixture_paths) {
  Json summaries = Json::array();
  for (int index = 0; index < 3; ++index) {
    const Json document = LoadJson(fixture_paths[index]);
    if (document.at("schema_version") !=
        "lunar-v3-differential-cases/v1") {
      throw std::runtime_error{"unsupported fixture schema"};
    }
    const PlatformType platform =
        ParsePlatform(document.at("platform").get<std::string>());
    for (const Json& definition : document.at("cases")) {
      summaries.push_back(GenerateCase(platform, definition));
    }
  }

  std::ofstream output{output_path};
  if (!output) {
    throw std::runtime_error{"cannot open output: " + output_path};
  }
  output << Json{
                {"schema_version", "lunar-v3-legacy-summary/v1"},
                {"summaries", std::move(summaries)},
            }
                 .dump(2)
         << '\n';
}

}  // namespace
}  // namespace lunar::planning::v3::migration

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: lunar_planner_legacy_summary OUTPUT "
                 "WHEEL_CASES LEGGED_CASES HOPPER_CASES\n";
    return 2;
  }
  try {
    lunar::planning::v3::migration::Generate(argv[1], argv + 2);
  } catch (const std::exception& error) {
    std::cerr << "legacy summary error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
