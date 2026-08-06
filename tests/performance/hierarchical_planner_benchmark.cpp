#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#endif

#include <nlohmann/json.hpp>

#include "hierarchical/global_route_planner.hpp"
#include "lunar_planner_core/planner.hpp"
#include "test_fixtures.hpp"

namespace lunar::planning::benchmark
{
namespace
{

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

constexpr std::string_view kSchemaVersion =
  "lunar-hierarchical-benchmark/v2";
constexpr std::string_view kBuildType = LUNAR_BUILD_TYPE;
constexpr std::size_t kWarmupRuns = 1U;
constexpr std::size_t kMeasuredRuns = 30U;
constexpr double kResolutionM = 0.2;

struct Arguments final
{
  std::string output_path;
};

struct SearchMetrics final
{
  std::uint64_t expanded_states{};
  std::size_t open_peak{};
  std::size_t peak_memory_bytes{};
  std::size_t safe_landing_nodes{};
  std::size_t candidate_edges_evaluated{};
  std::size_t coarse_edges_rejected{};
  std::size_t full_edges_certified{};
  std::size_t full_edges_invalidated{};
  std::size_t edge_certificate_cache_hits{};

  auto operator<=>(const SearchMetrics &) const = default;
};

struct Timings final
{
  double p50_s{};
  double p95_s{};
  double maximum_s{};
};

[[nodiscard]] Arguments ParseArguments(const int argc, char ** argv)
{
  if (argc != 3 || std::string_view{argv[1]} != "--output" ||
    std::string_view{argv[2]}.empty())
  {
    throw std::invalid_argument{
            "usage: hierarchical_planner_benchmark --output PATH"};
  }
  return Arguments{.output_path = argv[2]};
}

void SetObstacle(GridMap & map, const std::size_t x, const std::size_t y)
{
  const std::size_t index = y * map.width + x;
  std::get<std::vector<std::uint8_t>>(
    map.layers.at("obstacle").values)[index] = 1U;
  std::get<std::vector<float>>(
    map.layers.at("obstacle_height").values)[index] = 1.0F;
}

void AddFixedObstacles(GridMap & map)
{
  const std::size_t center_y = map.height / 2U;
  const std::size_t first_x = map.width / 3U;
  const std::size_t second_x = 2U * map.width / 3U;
  for (std::size_t y = 0U; y < map.height; ++y) {
    if (y + 8U < center_y || y > center_y + 8U) {
      SetObstacle(map, first_x, y);
    }
    if (y + 32U < center_y || y > center_y - 16U) {
      SetObstacle(map, second_x, y);
    }
  }
}

void AddNarrowChannel(GridMap & map)
{
  const std::size_t center_y = map.height / 2U;
  for (std::size_t y = 0U; y < map.height; ++y) {
    if (y + 4U >= center_y && y <= center_y + 4U) {
      continue;
    }
    for (std::size_t x = 0U; x < map.width; ++x) {
      SetObstacle(map, x, y);
    }
  }
}

void AddNoRouteWall(GridMap & map)
{
  const std::size_t wall_x = map.width / 2U;
  for (std::size_t y = 0U; y < map.height; ++y) {
    SetObstacle(map, wall_x, y);
  }
}

[[nodiscard]] PlannerInput MakeInput(
  const std::size_t axis,
  const std::string_view fixture)
{
  PlannerInput input = test::MakeValidWheelInput();
  input.request_id = "hierarchical-benchmark-" + std::to_string(axis) + "-" +
    std::string{fixture};
  input.world.global_map = test::MakeFlatMap(
    "map", axis, axis, kResolutionM);
  input.world.local_map = test::MakeFlatMap("odom", 64U, 64U, kResolutionM);
  input.config.global_map.base_resolution_m = kResolutionM;
  input.config.wheel.xy_resolution_m = kResolutionM;
  auto & capability = std::get<WheeledCapability>(input.capability);
  capability.footprint_xy_m =
  {{-0.05, -0.05}, {0.05, -0.05}, {0.05, 0.05}, {-0.05, 0.05}};
  capability.minimum_clearance_m = 0.0;
  for (auto & primitive : capability.motion_primitives) {
    primitive.relative_end_pose.position_m.x *= kResolutionM;
    primitive.relative_end_pose.position_m.y *= kResolutionM;
  }

  const std::size_t start_x = 10U;
  const std::size_t start_y = axis / 2U;
  const std::size_t goal_x = axis - 11U;
  const double start_map_x =
    (static_cast<double>(start_x) + 0.5) * kResolutionM;
  const double start_map_y =
    (static_cast<double>(start_y) + 0.5) * kResolutionM;
  const double goal_map_x =
    (static_cast<double>(goal_x) + 0.5) * kResolutionM;
  const double goal_map_y = start_map_y;
  std::get<WheeledState>(input.current_state).pose.position_m = {
    start_map_x, start_map_y, 0.0};
  input.goal_map = GoalRegion{
    .goal_id = "benchmark-goal",
    .target = PointGoal{.position_m = {goal_map_x, goal_map_y, 0.0},
      .tolerance_m = 0.05},
  };
  input.world.local_map.origin_m = {
    start_map_x - 32.5 * kResolutionM,
    start_map_y - 32.5 * kResolutionM,
    0.0,
  };

  if (fixture == "fixed-obstacle") {
    AddFixedObstacles(input.world.global_map);
  } else if (fixture == "narrow-channel") {
    AddNarrowChannel(input.world.global_map);
  } else if (fixture == "no-route") {
    AddNoRouteWall(input.world.global_map);
  } else if (fixture != "open") {
    throw std::invalid_argument{"unknown benchmark fixture"};
  }
  return input;
}

void ClearValidity(GridMap & map)
{
  auto & valid = std::get<std::vector<std::uint8_t>>(
    map.layers.at("valid_mask").values);
  std::ranges::fill(valid, 0U);
}

void MarkValidRectangle(
  GridMap & map, const std::size_t minimum_x, const std::size_t maximum_x,
  const std::size_t minimum_y, const std::size_t maximum_y)
{
  auto & valid = std::get<std::vector<std::uint8_t>>(
    map.layers.at("valid_mask").values);
  for (std::size_t y = minimum_y; y <= maximum_y; ++y) {
    for (std::size_t x = minimum_x; x <= maximum_x; ++x) {
      valid.at(y * map.width + x) = 1U;
    }
  }
}

[[nodiscard]] double CellCenter(const std::size_t cell)
{
  return (static_cast<double>(cell) + 0.5) * kResolutionM;
}

[[nodiscard]] PlannerInput MakeFixedGroundInput(const bool legged)
{
  PlannerInput input = MakeInput(250U, "fixed-obstacle");
  input.request_id = legged ? "benchmark-legged-positive" :
    "benchmark-wheel-positive";
  if (!legged) {
    return input;
  }

  const PlannerInput source = test::MakeValidLeggedInput();
  const Vec3 start = std::get<WheeledState>(input.current_state).pose.position_m;
  input.platform_id = "benchmark-legged";
  input.capability_version = "test-only-legged-v1";
  input.current_state = LeggedState{
    .body_pose = Pose3{.position_m = {start.x, start.y, 0.5}},
  };
  input.capability = std::get<LeggedCapability>(source.capability);
  auto & capability = std::get<LeggedCapability>(input.capability);
  for (auto & primitive : capability.motion_primitives) {
    primitive.body_frame_displacement_m.x *= kResolutionM;
    primitive.body_frame_displacement_m.y *= kResolutionM;
  }
  input.config.legged.xy_resolution_m = kResolutionM;
  return input;
}

[[nodiscard]] PlannerInput MakeFixedHopperInput(
  const std::string_view fixture)
{
  PlannerInput input = test::MakeValidHopperInput();
  input.request_id = "benchmark-" + std::string{fixture};
  input.platform_id = "benchmark-hopper";
  input.capability_version = "test-only-hopper-v1";
  input.world.global_map = test::MakeFlatMap(
    "map", 250U, 250U, kResolutionM);
  input.world.local_map = test::MakeFlatMap(
    "odom", 250U, 250U, kResolutionM);
  input.config.global_map.base_resolution_m = kResolutionM;
  input.position_uncertainty_m = 0.0;
  input.velocity_uncertainty_mps = 0.0;
  input.hopper_propellant = HopperPropellantState{
    .stamp = input.state_time,
    .platform_id = input.platform_id,
    .capability_version = input.capability_version,
    .total_mass_kg = 20.0,
    .remaining_usable_fuel_mass_kg = 0.20,
  };

  constexpr std::size_t start_x = 30U;
  constexpr std::size_t center_y = 125U;
  std::size_t goal_x = 40U;
  ClearValidity(input.world.global_map);
  if (fixture == "hopper_direct_positive") {
    MarkValidRectangle(input.world.global_map, 20U, 60U, 112U, 138U);
  } else if (fixture == "hopper_multihop_positive") {
    goal_x = 90U;
    MarkValidRectangle(input.world.global_map, 20U, 100U, 112U, 138U);
  } else if (fixture == "hopper_complete_negative") {
    goal_x = 100U;
    MarkValidRectangle(input.world.global_map, 20U, 55U, 112U, 138U);
    MarkValidRectangle(input.world.global_map, 90U, 120U, 112U, 138U);
  } else {
    throw std::invalid_argument{"unknown hopper benchmark fixture"};
  }

  input.current_state = HopperState{
    .pose = Pose3{
      .position_m = {CellCenter(start_x), CellCenter(center_y), 0.5}},
  };
  input.goal_map = GoalRegion{
    .goal_id = "benchmark-hopper-goal",
    .target = PointGoal{
      .position_m = {CellCenter(goal_x), CellCenter(center_y), 0.0},
      .tolerance_m = 0.0},
  };
  return input;
}

[[nodiscard]] PlannerInput MakeFixedInput(const std::string_view fixture)
{
  if (fixture == "wheel_positive") {
    return MakeFixedGroundInput(false);
  }
  if (fixture == "legged_positive") {
    return MakeFixedGroundInput(true);
  }
  return MakeFixedHopperInput(fixture);
}

[[nodiscard]] std::string OutcomeName(const PlanningOutcome outcome)
{
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
  throw std::logic_error{"unknown planning outcome"};
}

[[nodiscard]] Json PoseJson(const Pose3 & pose)
{
  return Json::array(
    {pose.position_m.x, pose.position_m.y, pose.position_m.z,
      pose.orientation.w, pose.orientation.x, pose.orientation.y,
      pose.orientation.z});
}

[[nodiscard]] Json StableSignature(const PlannerOutput & output)
{
  Json signature{
    {"outcome", OutcomeName(output.outcome)},
    {"directive", static_cast<std::uint8_t>(output.directive)},
    {"reason", output.reason_code},
    {"best_cost",
      output.diagnostics.best_cost ?
      Json{*output.diagnostics.best_cost} :
      Json{nullptr}},
    {"warnings", output.diagnostics.warning_codes},
  };
  if (output.diagnostics.local_trajectory) {
    const auto & local = *output.diagnostics.local_trajectory;
    signature["trajectory_mode"] = ToString(local.trajectory_mode);
    signature["start_anchor_error_m"] = local.start_anchor_error_m;
    signature["endpoint_error_m"] = local.endpoint_error_m;
    signature["maximum_curvature_per_m"] = local.maximum_curvature_per_m;
    signature["collision_validation"] = ToString(local.collision_validation);
  }
  if (output.reference) {
    signature["plan_id"] = output.reference->plan_id;
    for (const auto & pose : output.reference->preview.poses_map) {
      signature["preview"].push_back(PoseJson(pose));
    }
    if (const auto * trajectory =
      std::get_if<TrajectoryReference>(&output.reference->data))
    {
      for (const auto & point : trajectory->points) {
        signature["local"].push_back(
          {{"time_ns", point.time_from_start.count()},
            {"pose", PoseJson(point.pose)},
            {"linear",
              {point.velocity.linear_mps.x, point.velocity.linear_mps.y,
                point.velocity.linear_mps.z}},
            {"angular",
              {point.velocity.angular_radps.x, point.velocity.angular_radps.y,
                point.velocity.angular_radps.z}}});
      }
    }
  }
  return signature;
}

[[nodiscard]] std::string HashSignature(const Json & signature)
{
  constexpr std::uint64_t kOffsetBasis = 14'695'981'039'346'656'037ULL;
  constexpr std::uint64_t kPrime = 1'099'511'628'211ULL;
  std::uint64_t hash = kOffsetBasis;
  for (const unsigned char byte : signature.dump()) {
    hash ^= static_cast<std::uint64_t>(byte);
    hash *= kPrime;
  }
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << hash;
  return stream.str();
}

[[nodiscard]] SearchMetrics Metrics(const PlannerOutput & output)
{
  SearchMetrics metrics;
  if (output.diagnostics.hierarchical) {
    metrics.expanded_states = std::max(
      metrics.expanded_states,
      output.diagnostics.hierarchical->global_expanded_states);
    metrics.open_peak = std::max(
      metrics.open_peak, output.diagnostics.hierarchical->global_open_peak);
    metrics.peak_memory_bytes = std::max(
      metrics.peak_memory_bytes,
      output.diagnostics.hierarchical->estimated_work_memory_bytes);
    metrics.safe_landing_nodes =
      output.diagnostics.hierarchical->safe_landing_nodes;
    metrics.candidate_edges_evaluated =
      output.diagnostics.hierarchical->candidate_edges_evaluated;
    metrics.coarse_edges_rejected =
      output.diagnostics.hierarchical->coarse_edges_rejected;
    metrics.full_edges_certified =
      output.diagnostics.hierarchical->full_edges_certified;
    metrics.full_edges_invalidated =
      output.diagnostics.hierarchical->full_edges_invalidated;
    metrics.edge_certificate_cache_hits =
      output.diagnostics.hierarchical->edge_certificate_cache_hits;
  }
  return metrics;
}

void RequireFixedExpected(
  const std::string_view fixture, const PlannerOutput & output)
{
  if (fixture == "hopper_complete_negative") {
    if (output.outcome != PlanningOutcome::kNoKnownSafeRoute ||
      output.reason_code != "GLOBAL_NO_KNOWN_SAFE_ROUTE" ||
      output.reference.has_value())
    {
      throw std::runtime_error{
              "complete hopper negative produced unexpected result: " +
              output.reason_code};
    }
    return;
  }
  if (output.outcome != PlanningOutcome::kNewReferenceAvailable ||
    !output.reference.has_value())
  {
    throw std::runtime_error{
            "fixed positive produced unexpected result: " + output.reason_code};
  }
  if (fixture == "wheel_positive" &&
    output.reason_code != "WHEEL_PLAN_AVAILABLE")
  {
    throw std::runtime_error{"wheel positive reason mismatch"};
  }
  if (fixture == "legged_positive" &&
    output.reason_code != "LEGGED_BODY_PLAN_AVAILABLE")
  {
    throw std::runtime_error{"legged positive reason mismatch"};
  }
  if (fixture == "hopper_direct_positive") {
    if (output.reason_code != "HOPPER_FIRST_HOP_AVAILABLE" ||
      output.certified_hops.size() != 1U)
    {
      throw std::runtime_error{"direct hopper route is not exactly one hop"};
    }
  }
  if (fixture == "hopper_multihop_positive") {
    if (output.reason_code != "HOPPER_FIRST_HOP_AVAILABLE" ||
      output.certified_hops.size() <= 1U)
    {
      throw std::runtime_error{"hopper multihop fixture did not use multiple hops"};
    }
  }
}

[[nodiscard]] std::string FixedPlatformName(const std::string_view fixture)
{
  if (fixture == "wheel_positive") {
    return "WHEELED";
  }
  if (fixture == "legged_positive") {
    return "LEGGED";
  }
  return "HOPPER";
}

[[nodiscard]] double FixedThreshold(const std::string_view fixture)
{
  if (fixture == "wheel_positive" || fixture == "legged_positive") {
    return 2.0;
  }
  if (fixture == "hopper_direct_positive") {
    return 1.0;
  }
  return 5.0;
}

[[nodiscard]] double Seconds(const Clock::duration duration)
{
  return std::chrono::duration<double>{duration}.count();
}

[[nodiscard]] Timings Summarize(std::vector<double> values)
{
  if (values.size() != kMeasuredRuns) {
    throw std::logic_error{"benchmark timing count is invalid"};
  }
  std::ranges::sort(values);
  const std::size_t middle = values.size() / 2U;
  const double median = values.size() % 2U == 0U ?
    (values[middle - 1U] + values[middle]) / 2.0 :
    values[middle];
  const std::size_t p95_index =
    static_cast<std::size_t>(std::ceil(0.95 * values.size())) - 1U;
  return Timings{
    .p50_s = median,
    .p95_s = values[p95_index],
    .maximum_s = values.back(),
  };
}

[[nodiscard]] Json TimingJson(const Timings & timing)
{
  return Json{
    {"p50_s", timing.p50_s},
    {"p95_s", timing.p95_s},
    {"maximum_s", timing.maximum_s},
  };
}

[[nodiscard]] std::size_t PeakResidentMemoryBytes()
{
#if defined(__linux__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss >= 0) {
    return static_cast<std::size_t>(usage.ru_maxrss) * 1024U;
  }
#endif
  return 0U;
}

[[nodiscard]] Json RunFixedCase(const std::string_view fixture)
{
  const PlannerInput input = MakeFixedInput(fixture);
  Planner planner;
  for (std::size_t run = 0U; run < kWarmupRuns; ++run) {
    RequireFixedExpected(fixture, planner.Plan(input));
  }

  std::vector<double> complete_seconds;
  std::vector<double> global_seconds;
  std::vector<double> local_seconds;
  std::vector<double> landing_field_seconds;
  std::vector<double> spatial_index_seconds;
  std::vector<double> ballistic_solve_seconds;
  std::vector<double> flight_tube_seconds;
  for (auto * values : {
      &complete_seconds, &global_seconds, &local_seconds,
      &landing_field_seconds, &spatial_index_seconds,
      &ballistic_solve_seconds, &flight_tube_seconds})
  {
    values->reserve(kMeasuredRuns);
  }

  bool first = true;
  std::string stable_hash;
  SearchMetrics stable_metrics;
  PlanningOutcome stable_outcome = PlanningOutcome::kInvalidRequest;
  std::string stable_reason;
  for (std::size_t run = 0U; run < kMeasuredRuns; ++run) {
    const Clock::time_point complete_started = Clock::now();
    const PlannerOutput output = planner.Plan(input);
    complete_seconds.push_back(Seconds(Clock::now() - complete_started));
    RequireFixedExpected(fixture, output);
    if (!output.diagnostics.hierarchical.has_value()) {
      throw std::runtime_error{"fixed benchmark hierarchical metrics missing"};
    }
    const auto & hierarchical = *output.diagnostics.hierarchical;
    global_seconds.push_back(
      std::chrono::duration<double>{hierarchical.global_elapsed}.count());
    local_seconds.push_back(
      std::chrono::duration<double>{hierarchical.local_elapsed}.count());
    landing_field_seconds.push_back(
      std::chrono::duration<double>{hierarchical.landing_field_elapsed}.count());
    spatial_index_seconds.push_back(
      std::chrono::duration<double>{hierarchical.spatial_index_elapsed}.count());
    ballistic_solve_seconds.push_back(
      std::chrono::duration<double>{hierarchical.ballistic_solve_elapsed}.count());
    flight_tube_seconds.push_back(
      std::chrono::duration<double>{
        hierarchical.flight_tube_certification_elapsed}.count());

    const std::string hash = HashSignature(StableSignature(output));
    const SearchMetrics metrics = Metrics(output);
    if (first) {
      first = false;
      stable_hash = hash;
      stable_metrics = metrics;
      stable_outcome = output.outcome;
      stable_reason = output.reason_code;
    } else if (hash != stable_hash || metrics != stable_metrics ||
      output.outcome != stable_outcome || output.reason_code != stable_reason)
    {
      throw std::runtime_error{
              "fixed benchmark route, outcome, or diagnostic counts changed"};
    }
  }

  const Timings complete = Summarize(std::move(complete_seconds));
  const double threshold_s = FixedThreshold(fixture);
  return Json{
    {"schema_version", kSchemaVersion},
    {"platform", FixedPlatformName(fixture)},
    {"fixture", fixture},
    {"authority", "test-only/non-authoritative"},
    {"cells", 250U * 250U},
    {"width_m", 50.0},
    {"height_m", 50.0},
    {"resolution_m", kResolutionM},
    {"runs", kMeasuredRuns},
    {"p50_s", complete.p50_s},
    {"p95_s", complete.p95_s},
    {"maximum_s", complete.maximum_s},
    {"stage_timings_s",
      {{"global_search", TimingJson(Summarize(std::move(global_seconds)))},
        {"local_planning", TimingJson(Summarize(std::move(local_seconds)))},
        {"landing_field", TimingJson(
            Summarize(std::move(landing_field_seconds)))},
        {"spatial_index", TimingJson(
            Summarize(std::move(spatial_index_seconds)))},
        {"ballistic_solve", TimingJson(
            Summarize(std::move(ballistic_solve_seconds)))},
        {"flight_tube_certification", TimingJson(
            Summarize(std::move(flight_tube_seconds)))}}},
    {"expanded_states", stable_metrics.expanded_states},
    {"open_peak", stable_metrics.open_peak},
    {"peak_work_memory_bytes", stable_metrics.peak_memory_bytes},
    {"peak_resident_memory_bytes", PeakResidentMemoryBytes()},
    {"safe_landing_nodes", stable_metrics.safe_landing_nodes},
    {"candidate_edges_evaluated",
      stable_metrics.candidate_edges_evaluated},
    {"coarse_edges_rejected", stable_metrics.coarse_edges_rejected},
    {"full_edges_certified", stable_metrics.full_edges_certified},
    {"full_edges_invalidated", stable_metrics.full_edges_invalidated},
    {"edge_certificate_cache_hits",
      stable_metrics.edge_certificate_cache_hits},
    {"route_hash", stable_hash},
    {"planning_outcome", OutcomeName(stable_outcome)},
    {"reason_code", stable_reason},
    {"deterministic", true},
    {"ubuntu_release_p95_threshold_s", threshold_s},
    {"ubuntu_threshold_passed", complete.p95_s <= threshold_s},
  };
}

[[nodiscard]] Json Run()
{
  if (kBuildType != "Release") {
    throw std::runtime_error{"PERFORMANCE_BUILD_NOT_RELEASE"};
  }
  constexpr std::array<std::string_view, 5U> fixed_fixtures{
    "wheel_positive",
    "legged_positive",
    "hopper_direct_positive",
    "hopper_multihop_positive",
    "hopper_complete_negative",
  };
  Json results = Json::array();
  Json document{
    {"schema_version", kSchemaVersion},
    {"build_type", kBuildType},
    {"warmup_runs", kWarmupRuns},
    {"measured_runs", kMeasuredRuns},
    {"timing_unit", "s"},
    {"map", {{"width_m", 50.0}, {"height_m", 50.0},
        {"resolution_m", kResolutionM}, {"cells", 250U * 250U}}},
    {"capability_authority", "test-only/non-authoritative"},
    {"ubuntu_amd64_release_evaluated", true},
    {"jetson_agx_orin_evaluated", false},
  };
  for (const std::string_view fixture : fixed_fixtures) {
    try {
      Json result = RunFixedCase(fixture);
      document[std::string{fixture}] = result;
      results.push_back(std::move(result));
    } catch (const std::exception & error) {
      throw std::runtime_error{
              "fixture " + std::string{fixture} + ": " + error.what()};
    }
  }
  document["results"] = std::move(results);
  return document;
}

void Write(const std::string & path, const Json & document)
{
  std::ofstream stream{path, std::ios::binary | std::ios::trunc};
  if (!stream) {
    throw std::runtime_error{"cannot open benchmark output: " + path};
  }
  stream << document.dump(2) << '\n';
  stream.flush();
  if (!stream) {
    throw std::runtime_error{"cannot write benchmark output: " + path};
  }
}

} // namespace
} // namespace lunar::planning::benchmark

int main(int argc, char ** argv)
{
  try {
    const auto arguments =
      lunar::planning::benchmark::ParseArguments(argc, argv);
    lunar::planning::benchmark::Write(
      arguments.output_path, lunar::planning::benchmark::Run());
  } catch (const std::exception & error) {
    std::cerr << "hierarchical planner benchmark error: " << error.what()
              << '\n';
    return 1;
  }
  return 0;
}
