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
  "lunar-hierarchical-benchmark/v1";
constexpr std::string_view kBuildType = LUNAR_BUILD_TYPE;
constexpr std::size_t kWarmupRuns = 1U;
constexpr std::size_t kMeasuredRuns = 30U;
constexpr std::size_t kMaximumBenchmarkMemoryBytes =
  256U * 1024U * 1024U;
constexpr double kResolutionM = 0.2;
constexpr std::array<std::size_t, 3> kAxisTiers{256U, 512U, 1'024U};
constexpr std::array<std::string_view, 4> kFixtures{
  "open", "fixed-obstacle", "narrow-channel", "no-route"};
constexpr std::array<std::string_view, 5> kTrajectoryModes{
  "STATIONARY", "OPTIMIZED", "DISCRETE_FALLBACK", "CERTIFIED_HOP", "NONE"};

struct Thresholds final
{
  double global_s{};
  double complete_s{};
};

struct Arguments final
{
  std::string output_path;
};

struct SearchMetrics final
{
  std::uint64_t expanded_states{};
  std::size_t open_peak{};
  std::size_t peak_memory_bytes{};

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

[[nodiscard]] std::size_t TrajectoryModeIndex(const PlannerOutput & output)
{
  if (!output.diagnostics.local_trajectory) {
    return kTrajectoryModes.size() - 1U;
  }
  switch (output.diagnostics.local_trajectory->trajectory_mode) {
    case TrajectoryMode::kStationary:
      return 0U;
    case TrajectoryMode::kOptimized:
      return 1U;
    case TrajectoryMode::kDiscreteFallback:
      return 2U;
    case TrajectoryMode::kCertifiedHop:
      return 3U;
  }
  throw std::logic_error{"unknown trajectory mode"};
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
  }
  return metrics;
}

void RequireExpected(
  const std::string_view fixture,
  const PlannerOutput & output)
{
  if (fixture == "no-route") {
    if (output.outcome != PlanningOutcome::kNoKnownSafeRoute ||
      output.reason_code != "GLOBAL_NO_KNOWN_SAFE_ROUTE" ||
      output.reference.has_value())
    {
      throw std::runtime_error{"no-route fixture produced unexpected result"};
    }
    return;
  }
  if (output.outcome != PlanningOutcome::kNewReferenceAvailable ||
    output.reason_code != "WHEEL_PLAN_AVAILABLE" ||
    !output.reference.has_value())
  {
    throw std::runtime_error{
            "success fixture produced unexpected result: " + output.reason_code};
  }
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

[[nodiscard]] Thresholds UbuntuThresholds(const std::size_t cells)
{
  if (cells == 65'536U) {
    return Thresholds{.global_s = 0.5, .complete_s = 2.0};
  }
  if (cells == 262'144U) {
    return Thresholds{.global_s = 1.0, .complete_s = 3.0};
  }
  if (cells == 1'048'576U) {
    return Thresholds{.global_s = 2.0, .complete_s = 4.0};
  }
  throw std::logic_error{"unsupported benchmark tier"};
}

[[nodiscard]] Json RunCase(
  const std::size_t axis,
  const std::string_view fixture)
{
  const PlannerInput input = MakeInput(axis, fixture);
  Planner planner;
  for (std::size_t run = 0U; run < kWarmupRuns; ++run) {
    const auto output = planner.Plan(input);
    RequireExpected(fixture, output);
  }

  std::vector<double> global_seconds;
  std::vector<double> complete_seconds;
  std::vector<double> smoothing_seconds;
  std::vector<double> landing_field_seconds;
  global_seconds.reserve(kMeasuredRuns);
  complete_seconds.reserve(kMeasuredRuns);
  smoothing_seconds.reserve(kMeasuredRuns);
  landing_field_seconds.reserve(kMeasuredRuns);
  std::array<std::size_t, kTrajectoryModes.size()> trajectory_mode_counts{};
  std::string stable_hash;
  SearchMetrics stable_metrics;
  bool first = true;
  PlanningOutcome stable_outcome = PlanningOutcome::kInvalidRequest;
  std::string stable_reason;
  for (std::size_t run = 0U; run < kMeasuredRuns; ++run) {
    const auto complete_started = Clock::now();
    const PlannerOutput output = planner.Plan(input);
    complete_seconds.push_back(Seconds(Clock::now() - complete_started));
    RequireExpected(fixture, output);
    if (!output.diagnostics.hierarchical) {
      throw std::runtime_error{"hierarchical metrics are missing"};
    }
    global_seconds.push_back(
      std::chrono::duration<double>{
          output.diagnostics.hierarchical->global_elapsed}.count());
    if (output.diagnostics.local_trajectory) {
      smoothing_seconds.push_back(
        output.diagnostics.local_trajectory->smoothing_elapsed_s);
      landing_field_seconds.push_back(
        output.diagnostics.local_trajectory->landing_field_elapsed_s);
    } else {
      smoothing_seconds.push_back(0.0);
      landing_field_seconds.push_back(0.0);
    }
    ++trajectory_mode_counts[TrajectoryModeIndex(output)];
    const std::string hash = HashSignature(StableSignature(output));
    const SearchMetrics metrics = Metrics(output);
    if (first) {
      stable_hash = hash;
      stable_metrics = metrics;
      stable_outcome = output.outcome;
      stable_reason = output.reason_code;
      first = false;
    } else if (hash != stable_hash || metrics != stable_metrics ||
      output.outcome != stable_outcome ||
      output.reason_code != stable_reason)
    {
      throw std::runtime_error{"benchmark result is not deterministic"};
    }
    if (metrics.peak_memory_bytes > kMaximumBenchmarkMemoryBytes)
    {
      throw std::runtime_error{"benchmark memory ceiling exceeded"};
    }
  }

  const Timings global = Summarize(std::move(global_seconds));
  const Timings complete = Summarize(std::move(complete_seconds));
  const Timings smoothing = Summarize(std::move(smoothing_seconds));
  const Timings landing_field = Summarize(std::move(landing_field_seconds));
  Json mode_counts = Json::object();
  for (std::size_t index = 0U; index < kTrajectoryModes.size(); ++index) {
    mode_counts[std::string{kTrajectoryModes[index]}] =
      trajectory_mode_counts[index];
  }
  const std::size_t cells = axis * axis;
  const Thresholds thresholds = UbuntuThresholds(cells);
  return Json{
    {"schema_version", kSchemaVersion},
    {"platform", "WHEELED"},
    {"fixture", fixture},
    {"cells", cells},
    {"resolution_m", kResolutionM},
    {"runs", kMeasuredRuns},
    {"p50_s", complete.p50_s},
    {"p95_s", complete.p95_s},
    {"maximum_s", complete.maximum_s},
    {"global_p50_s", global.p50_s},
    {"global_p95_s", global.p95_s},
    {"global_maximum_s", global.maximum_s},
    {"smoothing_p50_s", smoothing.p50_s},
    {"smoothing_p95_s", smoothing.p95_s},
    {"smoothing_maximum_s", smoothing.maximum_s},
    {"landing_field_p50_s", landing_field.p50_s},
    {"landing_field_p95_s", landing_field.p95_s},
    {"landing_field_maximum_s", landing_field.maximum_s},
    {"trajectory_mode_counts", std::move(mode_counts)},
    {"expanded_states", stable_metrics.expanded_states},
    {"open_peak", stable_metrics.open_peak},
    {"peak_memory_bytes", stable_metrics.peak_memory_bytes},
    {"route_hash", stable_hash},
    {"planning_outcome", OutcomeName(stable_outcome)},
    {"reason_code", stable_reason},
    {"deterministic", true},
    {"ubuntu_threshold_passed",
      global.p95_s <= thresholds.global_s &&
      complete.p95_s <= thresholds.complete_s},
  };
}

[[nodiscard]] Json ThresholdProfile(
  const bool evaluated,
  const std::array<double, 3> global,
  const std::array<double, 3> complete)
{
  return Json{
    {"evaluated", evaluated},
    {"global_p95_s",
      {{"65536", global[0]},
        {"262144", global[1]},
        {"1048576", global[2]}}},
    {"complete_core_p95_s",
      {{"65536", complete[0]},
        {"262144", complete[1]},
        {"1048576", complete[2]}}},
  };
}

[[nodiscard]] Json Run()
{
  if (kBuildType != "Release") {
    throw std::runtime_error{"PERFORMANCE_BUILD_NOT_RELEASE"};
  }
  Json results = Json::array();
  for (const std::size_t axis : kAxisTiers) {
    for (const std::string_view fixture : kFixtures) {
      try {
        results.push_back(RunCase(axis, fixture));
      } catch (const std::exception & error) {
        throw std::runtime_error{"tier " + std::to_string(axis * axis) +
                " fixture " + std::string{fixture} + ": " +
                error.what()};
      }
    }
  }
  return Json{
    {"schema_version", kSchemaVersion},
    {"build_type", kBuildType},
    {"warmup_runs", kWarmupRuns},
    {"measured_runs", kMeasuredRuns},
    {"timing_unit", "s"},
    {"memory_semantics", "planner_estimated_peak_work_memory_bytes"},
    {"threshold_profiles",
      {{"ubuntu_amd64",
        ThresholdProfile(true, {0.5, 1.0, 2.0}, {2.0, 3.0, 4.0})},
        {"jetson_agx_orin",
          ThresholdProfile(false, {1.0, 2.0, 4.0}, {3.0, 4.0, 6.0})}}},
    {"results", std::move(results)},
  };
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
