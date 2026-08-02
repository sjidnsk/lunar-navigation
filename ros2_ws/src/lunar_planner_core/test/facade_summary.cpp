#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include <nlohmann/json.hpp>

#include "migration/legacy_v3_adapter.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::migration {
namespace {

using Json = nlohmann::json;

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

[[nodiscard]] PlannerInput MakeInput(const PlatformType platform) {
  switch (platform) {
    case PlatformType::kWheeled:
      return test::MakeValidWheelInput();
    case PlatformType::kLegged:
      return test::MakeValidLeggedInput();
    case PlatformType::kHopper:
      return test::MakeValidHopperInput();
  }
  throw std::logic_error{"unhandled platform"};
}

void MoveStateOutsideMap(PlannerInput& input) {
  std::visit(
      [](auto& state) {
        using State = std::decay_t<decltype(state)>;
        Vec3* position = nullptr;
        if constexpr (std::is_same_v<State, WheeledState> ||
                      std::is_same_v<State, HopperState>) {
          position = &state.pose.position_m;
        } else {
          position = &state.body_pose.position_m;
        }
        position->x = -10.0;
        position->y = -10.0;
      },
      input.current_state);
}

void MakeGoalKnownInfeasible(
    PlannerInput& input, const PlatformType platform) {
  auto& forbidden = std::get<std::vector<std::uint8_t>>(
      input.world.local_map.layers.at("forbidden").values);
  if (platform == PlatformType::kHopper) {
    input.goal.target = PointGoal{
        .position_m = {4.25, 3.25, 0.0},
        .tolerance_m = 0.05,
    };
    forbidden[6U * input.world.local_map.width + 8U] = 1U;
    return;
  }
  input.goal.target = PointGoal{
      .position_m = {4.5, 3.5, 0.0},
      .tolerance_m = 0.2,
  };
  forbidden[3U * input.world.local_map.width + 4U] = 1U;
}

[[nodiscard]] std::string OutcomeName(const PlanningOutcome outcome) {
  switch (outcome) {
    case PlanningOutcome::kNewReferenceAvailable:
      return "NEW_REFERENCE_AVAILABLE";
    case PlanningOutcome::kSafeFrontierReferenceAvailable:
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
    case PlanningOutcome::kResourceExhausted:
      return "RESOURCE_EXHAUSTED";
    case PlanningOutcome::kActiveReferenceInvalidated:
      return "ACTIVE_REFERENCE_INVALIDATED";
    case PlanningOutcome::kCanceled:
      return "CANCELED";
  }
  throw std::logic_error{"unhandled planning outcome"};
}

[[nodiscard]] std::string DirectiveName(const ExecutionDirective directive) {
  switch (directive) {
    case ExecutionDirective::kActivateNewReference:
      return "ACTIVATE_NEW_REFERENCE";
    case ExecutionDirective::kContinueActiveReference:
      return "CONTINUE_ACTIVE_REFERENCE";
    case ExecutionDirective::kHoldPosition:
      return "HOLD_POSITION";
    case ExecutionDirective::kContinueCommittedHop:
      return "CONTINUE_COMMITTED_HOP";
    case ExecutionDirective::kNoSafeReference:
      return "NO_SAFE_REFERENCE";
  }
  throw std::logic_error{"unhandled execution directive"};
}

[[nodiscard]] bool ReferenceSatisfiesPlatformContract(
    const std::optional<MotionReference>& reference,
    const PlatformType platform) {
  if (!reference.has_value() || reference->platform_type != platform ||
      reference->plan_id.empty()) {
    return false;
  }
  if (platform == PlatformType::kHopper) {
    const auto* hop = std::get_if<HopReference>(&reference->data);
    return hop != nullptr && !hop->segments.empty() &&
           !hop->segments.front().landing_region_boundary_m.empty();
  }
  const auto* trajectory = std::get_if<TrajectoryReference>(&reference->data);
  if (trajectory == nullptr || trajectory->points.empty()) {
    return false;
  }
  const auto expected =
      platform == PlatformType::kWheeled
          ? TrajectorySemantics::kWheeledBase
          : TrajectorySemantics::kLeggedBodyReference;
  return trajectory->semantics == expected;
}

[[nodiscard]] PlannerOutput RunOnce(
    const PlatformType platform, const std::string& setup) {
  PlannerInput input = MakeInput(platform);
  input.request_id += "-" + setup;
  if (setup == "no_safe_route") {
    MoveStateOutsideMap(input);
  } else if (setup == "goal_infeasible") {
    MakeGoalKnownInfeasible(input, platform);
  }

  const LegacyV3FaultMode fault_mode =
      setup == "numerical_failure"
          ? LegacyV3FaultMode::kThrowingRegistry
          : LegacyV3FaultMode::kNone;
  LegacyV3Adapter adapter{fault_mode};
  return adapter.Plan(input);
}

[[nodiscard]] Json Normalize(
    const PlannerOutput& output, const PlatformType platform) {
  const bool has_valid_reference =
      ReferenceSatisfiesPlatformContract(output.reference, platform);
  Json cost = nullptr;
  if (output.diagnostics.best_cost.has_value()) {
    cost = *output.diagnostics.best_cost;
  }
  return Json{
      {"reachable", output.reference.has_value()},
      {"planning_outcome", OutcomeName(output.outcome)},
      {"execution_directive", DirectiveName(output.directive)},
      {"collision_free", has_valid_reference},
      {"goal_reached",
       output.outcome == PlanningOutcome::kNewReferenceAvailable},
      {"platform_constraints_satisfied", has_valid_reference},
      {"cost", std::move(cost)},
  };
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
    Json current = Normalize(RunOnce(platform, setup), platform);
    if (repetition == 0) {
      stable = std::move(current);
    } else if (current != stable) {
      throw std::runtime_error{
          "facade semantic summary is nondeterministic: " + case_id};
    }
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
                {"schema_version", "lunar-v3-facade-summary/v1"},
                {"summaries", std::move(summaries)},
            }
                 .dump(2)
         << '\n';
}

}  // namespace
}  // namespace lunar::planning::migration

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: lunar_planner_facade_summary OUTPUT "
                 "WHEEL_CASES LEGGED_CASES HOPPER_CASES\n";
    return 2;
  }
  try {
    lunar::planning::migration::Generate(argv[1], argv + 2);
  } catch (const std::exception& error) {
    std::cerr << "facade summary error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
