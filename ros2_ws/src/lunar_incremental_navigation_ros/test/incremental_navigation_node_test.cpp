#include "lunar_incremental_navigation_ros/incremental_navigation_node.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <lunar_planning_msgs/action/navigate_to_pose.hpp>
#include <lunar_planning_msgs/msg/path_reference.hpp>
#include <lunar_planning_msgs/msg/tracking_status.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"
#if defined(LUNAR_BUILD_DEMO)
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include "lunar_incremental_navigation_ros/planning_debug_publisher.hpp"
#endif
#include "lunar_incremental_navigation_ros/request_diagnostics.hpp"

namespace lunar::incremental_navigation_ros {
namespace {

using Action = lunar_planning_msgs::action::NavigateToPose;
namespace core = lunar::incremental_navigation;
using namespace std::chrono_literals;

class RosEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    if (!rclcpp::ok()) {
      int argc = 0;
      rclcpp::init(argc, nullptr);
    }
  }
  void TearDown() override {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

const auto* const kRosEnvironment =
    ::testing::AddGlobalTestEnvironment(new RosEnvironment{});

template <typename Predicate>
bool WaitFor(Predicate&& predicate,
             const std::chrono::milliseconds timeout = 3s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(2ms);
  }
  return predicate();
}

[[nodiscard]] std::string GoalUuidHex(
    const rclcpp_action::GoalUUID& uuid) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string value;
  value.reserve(uuid.size() * 2U);
  for (const std::uint8_t byte : uuid) {
    value.push_back(kHex[byte >> 4U]);
    value.push_back(kHex[byte & 0x0fU]);
  }
  return value;
}

[[nodiscard]] std_msgs::msg::Float32MultiArray Task3ElevationLayer() {
  std_msgs::msg::Float32MultiArray layer;
  std_msgs::msg::MultiArrayDimension outer;
  outer.label = "column_index";
  outer.size = 2U;
  outer.stride = 6U;
  std_msgs::msg::MultiArrayDimension inner;
  inner.label = "row_index";
  inner.size = 3U;
  inner.stride = 3U;
  layer.layout.dim = {outer, inner};
  layer.data.assign(6U, 0.0F);
  return layer;
}

[[nodiscard]] grid_map_msgs::msg::GridMap Task3LocalMap() {
  grid_map_msgs::msg::GridMap map;
  map.header.frame_id = "odom";
  map.info.resolution = 0.2;
  map.info.length_x = 0.6;
  map.info.length_y = 0.4;
  map.info.pose.orientation.w = 1.0;
  map.layers = {"elevation"};
  map.data = {Task3ElevationLayer()};
  return map;
}

[[nodiscard]] tf2_msgs::msg::TFMessage DirectMapFromOdom() {
  tf2_msgs::msg::TFMessage message;
  message.transforms.resize(1U);
  auto& transform = message.transforms.front();
  transform.header.frame_id = "map";
  transform.child_frame_id = "odom";
  transform.transform.rotation.w = 1.0;
  return message;
}

[[nodiscard]] std::shared_ptr<const core::FineTraversabilitySnapshot>
MakeFine(const std::uint64_t revision = 1U,
         const std::optional<core::GridIndex> blocked = std::nullopt) {
  core::PersistentElevationMap map;
  const core::GridGeometry geometry{.frame_id = "map",
                                    .width = 8U,
                                    .height = 8U,
                                    .resolution_m = 1.0};
  const std::vector<float> elevation(geometry.CellCount(), 0.0F);
  const auto update = map.Apply(core::ElevationEvidence{
      .geometry = geometry,
      .elevation_m = elevation,
      .map_from_source = {.parent_frame = "map", .child_frame = "map"},
  });
  if (update.status != core::ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("fine fixture elevation rejected");
  }
  auto raw = map.Snapshot();
  core::FineTraversabilityTile::StateArray states;
  states.fill(core::FineCellState::kUnknown);
  core::FineTraversabilityTile::CostArray costs;
  costs.fill(0.0);
  for (std::int64_t y = 0; y < 8; ++y) {
    for (std::int64_t x = 0; x < 8; ++x) {
      states[core::TileCellOffset({.x = x, .y = y})] =
          core::FineCellState::kFree;
    }
  }
  if (blocked) states[core::TileCellOffset(*blocked)] = core::FineCellState::kBlocked;
  auto tile = std::make_shared<const core::FineTraversabilityTile>(
      std::move(states), std::move(costs));
  core::FineTraversabilityTileDirectory directory(raw->geometry());
  directory = directory.WithTile(core::TileIndex{}, std::move(tile));
  return std::make_shared<const core::FineTraversabilitySnapshot>(
      raw->geometry(), raw->raw_elevation_revision(), revision, "test-profile",
      0.25, 0.0, core::TraversalCostWeights{}, raw, std::move(directory),
      std::vector<core::TileIndex>{core::TileIndex{}},
      std::vector<core::TileIndex>{core::TileIndex{}},
      core::FineSnapshotMetrics{});
}

[[nodiscard]] std::shared_ptr<const core::GlobalGuidanceSnapshot>
MakeGuidance(const std::shared_ptr<const core::FineTraversabilitySnapshot>& fine) {
  core::GlobalGuidanceTileDirectory directory(fine->geometry());
  return std::make_shared<const core::GlobalGuidanceSnapshot>(
      fine->geometry(), fine->raw_elevation_revision(),
      fine->fine_traversability_revision(), 1U, "test-profile",
      std::move(directory), std::vector<core::TileIndex>{},
      std::vector<core::TileIndex>{});
}

[[nodiscard]] std::shared_ptr<const core::FineTraversabilitySnapshot>
MakeDebugFine() {
  core::PersistentElevationMap map;
  const core::GridGeometry geometry{.frame_id = "map",
                                    .width = 8U,
                                    .height = 8U,
                                    .resolution_m = 1.0};
  const auto update = map.Apply(core::ElevationEvidence{
      .geometry = geometry,
      .elevation_m = std::vector<float>(geometry.CellCount(), 0.0F),
      .map_from_source = {.parent_frame = "map", .child_frame = "map"},
  });
  if (update.status != core::ElevationUpdateResult::Status::kApplied) {
    throw std::runtime_error("debug fine fixture elevation rejected");
  }
  const auto raw = map.Snapshot();
  core::FineTraversabilityTile::StateArray states;
  states.fill(core::FineCellState::kUnknown);
  for (std::int64_t y = 0; y < 8; ++y) {
    for (std::int64_t x = 0; x < 8; ++x) {
      states[core::TileCellOffset({.x = x, .y = y})] =
          core::FineCellState::kFree;
    }
  }
  states[core::TileCellOffset({.x = 1, .y = 1})] =
      core::FineCellState::kUnknown;
  states[core::TileCellOffset({.x = 2, .y = 1})] =
      core::FineCellState::kBlocked;
  core::FineTraversabilityTile::CostArray costs;
  costs.fill(0.0);
  for (std::int64_t y = 0; y < 8; ++y) {
    for (std::int64_t x = 0; x < 8; ++x) {
      costs[core::TileCellOffset({.x = x, .y = y})] = 0.25;
    }
  }
  core::FineTraversabilityTileDirectory directory(raw->geometry());
  directory = directory.WithTile(
      core::TileIndex{}, std::make_shared<const core::FineTraversabilityTile>(
                             std::move(states), std::move(costs)));
  return std::make_shared<const core::FineTraversabilitySnapshot>(
      raw->geometry(), raw->raw_elevation_revision(), 17U, "debug-profile",
      0.25, 0.0, core::TraversalCostWeights{}, raw, std::move(directory),
      std::vector<core::TileIndex>{core::TileIndex{}},
      std::vector<core::TileIndex>{core::TileIndex{}},
      core::FineSnapshotMetrics{});
}

[[nodiscard]] std::shared_ptr<const core::GlobalGuidanceSnapshot>
MakeDebugGuidance(
    const std::shared_ptr<const core::FineTraversabilitySnapshot>& fine) {
  core::GlobalGuidanceTile::StateArray states;
  states.fill(core::GuidanceCellState::kUnknown);
  states[core::TileCellOffset({.x = 1, .y = 1})] =
      core::GuidanceCellState::kCandidate;
  states[core::TileCellOffset({.x = 2, .y = 1})] =
      core::GuidanceCellState::kProvenBlocked;
  core::GlobalGuidanceTile::RiskArray risks;
  risks.fill(0.0);
  core::GlobalGuidanceTileDirectory directory(fine->geometry());
  directory = directory.WithTile(
      core::TileIndex{}, std::make_shared<const core::GlobalGuidanceTile>(
                             std::move(states), std::move(risks)));
  return std::make_shared<const core::GlobalGuidanceSnapshot>(
      fine->geometry(), fine->raw_elevation_revision(),
      fine->fine_traversability_revision(), 23U, "debug-profile",
      std::move(directory), std::vector<core::TileIndex>{core::TileIndex{}},
      std::vector<core::TileIndex>{core::TileIndex{}});
}

struct FakeControl final {
  bool reaches_final_goal{};
  bool publish_global_route{};
  bool make_debug_start_patch{};
  std::chrono::milliseconds local_delay{};
  std::vector<core::Point2> segment_ends;
  std::size_t local_calls{};
  std::atomic<std::int64_t> global_deadline_remaining_ms{-1};
  std::atomic<std::int64_t> local_deadline_remaining_ms{-1};
  std::optional<std::size_t> fail_local_after_calls;
  std::optional<std::size_t> block_local_call;
  std::mutex block_mutex;
  std::condition_variable block_condition;
  bool solver_blocked{};

  [[nodiscard]] bool WaitUntilBlocked() {
    std::unique_lock lock{block_mutex};
    return block_condition.wait_for(lock, 3s,
                                    [this] { return solver_blocked; });
  }
};

[[nodiscard]] SessionPortsFactory FakePorts(
    std::shared_ptr<FakeControl> control) {
  return [control](const core::PlatformCapability&) {
    core::PlanningSessionPorts ports;
    ports.plan_global =
        [control](const core::GlobalGuidanceSnapshot&, const core::Point2 start,
                  const core::Point2 goal, const core::SearchDeadline deadline,
                  const core::StopToken&) {
          control->global_deadline_remaining_ms.store(
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  deadline - std::chrono::steady_clock::now())
                  .count());
          core::GlobalRouteResult result{
              .status = control->publish_global_route
                            ? core::GuidanceStatus::kAvailable
                            : core::GuidanceStatus::kNoRoute,
              .statistics = {.expanded_states = 7U, .open_peak = 5U},
          };
          if (control->publish_global_route) {
            result.route = core::GlobalRoute{
                .poses_map = {
                    {.position_m = {.x = start.x, .y = start.y}},
                    {.position_m = {.x = goal.x, .y = goal.y}},
                },
                .expanded_states = 7U,
            };
          }
          return result;
        };
    ports.select_target =
        [](const core::FineTraversabilitySnapshot&,
           const core::SparseGridGeometry&, core::Point2,
           const core::FinalGoal& goal,
           const std::optional<core::GlobalRoute>&) {
          return std::optional<core::LocalTarget>(core::LocalTarget{
              .center = {.x = goal.target_x_m, .y = goal.target_y_m},
              .position_tolerance_m = 0.5,
              .is_final_goal = true,
              .terminal_yaw_rad = goal.has_target_yaw
                  ? std::optional(goal.target_yaw_rad)
                  : std::nullopt,
          });
        };
    ports.build_start_patch =
        [control](std::shared_ptr<const core::FineTraversabilitySnapshot> fine,
           const core::SparseGridGeometry& local_window,
           const core::Pose2& p0,
           const core::PlatformCapability&,
           const core::TraversabilityProfile&) {
          std::vector<core::LocalCellOverride> overrides;
          if (control->make_debug_start_patch) {
            overrides.push_back({.index = {.x = 1, .y = 1},
                                 .source = core::LocalCellSource::kStartAssumedFree,
                                 .traversal_cost = 0.25});
          }
          return core::StartPatchResult{
              .status = core::StartPatchResult::Status::kNotNeeded,
              .view = std::make_shared<const core::RequestLocalPlanningView>(
                  std::move(fine), local_window, p0,
                  control->make_debug_start_patch ? 1.0 : 0.0,
                  std::move(overrides)),
          };
        };
    ports.plan_wheel =
        [control](const core::RequestLocalPlanningView&,
                  const core::Pose2& start, const core::LocalTarget& target,
                  const core::SearchDeadline deadline,
                  const core::StopToken& stop) {
          control->local_deadline_remaining_ms.store(
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  deadline - std::chrono::steady_clock::now())
                  .count());
          if (control->local_delay > std::chrono::milliseconds::zero()) {
            std::this_thread::sleep_for(control->local_delay);
          }
          const std::size_t call_index = control->local_calls;
          if (control->fail_local_after_calls &&
              call_index >= *control->fail_local_after_calls) {
            ++control->local_calls;
            return core::LocalPlanResult{
                .status = core::LocalPlanResult::Status::kNoPath};
          }
          if (control->block_local_call &&
              call_index == *control->block_local_call) {
            ++control->local_calls;
            std::unique_lock lock{control->block_mutex};
            control->solver_blocked = true;
            control->block_condition.notify_all();
            std::stop_callback wake_solver{
                stop, [control] { control->block_condition.notify_all(); }};
            control->block_condition.wait(
                lock, [&stop] { return stop.stop_requested(); });
            return core::LocalPlanResult{
                .status = core::LocalPlanResult::Status::kCanceled};
          }
          const core::Point2 end = call_index < control->segment_ends.size()
                                       ? control->segment_ends[call_index]
                                       : target.center;
          ++control->local_calls;
          core::LocalPlanResult result{
              .status = core::LocalPlanResult::Status::kPlanFound,
              .reaches_final_goal = control->reaches_final_goal,
              .statistics = {.expanded_states = 11U, .open_peak = 9U},
              .postprocess_elapsed = 3ms,
          };
          result.path = {
              core::PathPoint{.pose = {
                  .position_m = {.x = start.position_m.x,
                                 .y = start.position_m.y},
                  .orientation = {.w = 1.0}}},
              core::PathPoint{.pose = {
                  .position_m = {.x = end.x, .y = end.y},
                  .orientation = {.w = 1.0}}},
          };
          return result;
        };
    return ports;
  };
}

class StateQueue final {
 public:
  void Push(const core::Pose2& pose) {
    std::scoped_lock lock{mutex_};
    values_.push_back(core::StateInput{.base_link_pose = pose});
  }

  [[nodiscard]] std::optional<core::StateInput> Take() {
    std::scoped_lock lock{mutex_};
    if (values_.empty()) {
      return std::nullopt;
    }
    auto value = values_.front();
    values_.erase(values_.begin());
    return value;
  }

 private:
  std::mutex mutex_;
  std::vector<core::StateInput> values_;
};

struct TimingOverrides final {
  std::int64_t planning_sla_ms;
  std::int64_t planning_hard_timeout_ms;
  std::int64_t global_subdeadline_ms;
};

[[nodiscard]] rclcpp::NodeOptions ServerOptions(
    const std::string& suffix, const bool debug_enabled = false,
    const std::optional<TimingOverrides>& timing = std::nullopt) {
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-r", "__node:=incremental_navigation_" + suffix});
  options.parameter_overrides({
      rclcpp::Parameter{"enable_tracking_feedback", suffix.starts_with("tracking_")},
      rclcpp::Parameter{"tracking_status_topic", "/test/" + suffix + "/tracking"},
      rclcpp::Parameter{"platform_type", "wheel"},
      rclcpp::Parameter{"platform_config",
                        std::string{LUNAR_INCREMENTAL_NAVIGATION_CONFIG_DIR} +
                            "/wheel.yaml"},
      rclcpp::Parameter{"action_name",
                        "/test/" + suffix + "/navigate_to_pose"},
      rclcpp::Parameter{"path_reference_topic",
                        "/test/" + suffix + "/path_reference"},
      rclcpp::Parameter{"local_path_topic",
                        "/test/" + suffix + "/local_path"},
      rclcpp::Parameter{"global_route_topic",
                        "/test/" + suffix + "/global_route"},
      rclcpp::Parameter{"diagnostics_topic",
                        "/test/" + suffix + "/diagnostics"},
      rclcpp::Parameter{"local_map_topic", "/test/" + suffix + "/local"},
      rclcpp::Parameter{"exploration_map_topic",
                        "/test/" + suffix + "/exploration_map"},
      rclcpp::Parameter{"odometry_topic", "/test/" + suffix + "/odom"},
      rclcpp::Parameter{"tf_topic", "/test/" + suffix + "/tf"},
  });
  if (timing) {
    options.append_parameter_override("planning_sla_ms",
                                      timing->planning_sla_ms);
    options.append_parameter_override("planning_hard_timeout_ms",
                                      timing->planning_hard_timeout_ms);
    options.append_parameter_override("global_subdeadline_ms",
                                      timing->global_subdeadline_ms);
  }
#if defined(LUNAR_BUILD_DEMO)
  if (debug_enabled) {
    options.append_parameter_override("enable_debug_visualization", true);
    options.append_parameter_override("debug_topic_prefix",
                                      "/planning_demo/debug/" + suffix);
  }
#endif
  return options;
}

struct EventLog final {
  void Push(const std::string& value) {
    std::scoped_lock lock{mutex};
    events.push_back(value);
  }
  [[nodiscard]] std::vector<std::string> Copy() const {
    std::scoped_lock lock{mutex};
    return events;
  }
  mutable std::mutex mutex;
  std::vector<std::string> events;
};

class CallbackGate final {
 public:
  void Arm() {
    std::scoped_lock lock{mutex_};
    armed_ = true;
    entered_ = false;
    released_ = false;
  }

  void WaitIfArmed() {
    std::unique_lock lock{mutex_};
    if (!armed_) {
      return;
    }
    entered_ = true;
    condition_.notify_all();
    condition_.wait(lock, [this] { return released_; });
    armed_ = false;
  }

  [[nodiscard]] bool WaitUntilEntered() {
    std::unique_lock lock{mutex_};
    return condition_.wait_for(lock, 3s, [this] { return entered_; });
  }

  void Release() {
    std::scoped_lock lock{mutex_};
    released_ = true;
    condition_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  bool armed_{};
  bool entered_{};
  bool released_{};
};

class RunningSystem final {
 public:
  RunningSystem(std::string suffix, IncrementalNavigationNodeDependencies hooks,
                const bool debug_enabled = false,
                const std::optional<TimingOverrides>& timing = std::nullopt)
      : suffix_(std::move(suffix)),
        server_(std::make_shared<IncrementalNavigationNode>(
            ServerOptions(suffix_, debug_enabled, timing), std::move(hooks))),
        client_node_(std::make_shared<rclcpp::Node>("client_" + suffix_)),
        client_(rclcpp_action::create_client<Action>(
            client_node_, "/test/" + suffix_ + "/navigate_to_pose")) {
    path_subscription_ =
        client_node_->create_subscription<lunar_planning_msgs::msg::PathReference>(
            "/test/" + suffix_ + "/path_reference", PathReferenceQos(),
            [this](const lunar_planning_msgs::msg::PathReference& path) {
              std::scoped_lock lock{mutex_};
              paths_.push_back(path);
            });
    local_path_subscription_ = client_node_->create_subscription<nav_msgs::msg::Path>(
        "/test/" + suffix_ + "/local_path", PathReferenceQos(),
        [this](const nav_msgs::msg::Path& path) {
          std::scoped_lock lock{mutex_};
          local_paths_.push_back(path);
        });
    global_subscription_ = client_node_->create_subscription<nav_msgs::msg::Path>(
        "/test/" + suffix_ + "/global_route", PathReferenceQos(),
        [this](const nav_msgs::msg::Path& path) {
          std::scoped_lock lock{mutex_};
          globals_.push_back(path);
        });
    diagnostics_subscription_ =
        client_node_->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
            "/test/" + suffix_ + "/diagnostics",
            rclcpp::QoS{rclcpp::KeepLast{10}}.reliable(),
            [this](const diagnostic_msgs::msg::DiagnosticArray& diagnostics) {
              std::scoped_lock lock{mutex_};
              diagnostics_.push_back(diagnostics);
            });
#if defined(LUNAR_BUILD_DEMO)
    if (debug_enabled) {
      const std::string debug_prefix = "/planning_demo/debug/" + suffix_;
      fine_state_subscription_ =
          client_node_->create_subscription<nav_msgs::msg::OccupancyGrid>(
              debug_prefix + "/fine_state", DebugVisualizationQos(),
              [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr message) {
                std::scoped_lock lock{mutex_};
                fine_states_.push_back(*message);
              });
      guidance_state_subscription_ =
          client_node_->create_subscription<nav_msgs::msg::OccupancyGrid>(
              debug_prefix + "/guidance_state", DebugVisualizationQos(),
              [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr message) {
                std::scoped_lock lock{mutex_};
                guidance_states_.push_back(*message);
              });
      start_patch_subscription_ = client_node_->create_subscription<
          visualization_msgs::msg::MarkerArray>(
          debug_prefix + "/start_patch_cells", DebugVisualizationQos(),
          [this](visualization_msgs::msg::MarkerArray::ConstSharedPtr message) {
            std::scoped_lock lock{mutex_};
            start_patches_.push_back(*message);
          });
    }
#endif
    executor_.add_node(server_);
    executor_.add_node(client_node_);
    spinner_ = std::jthread([this] { executor_.spin(); });
    if (!client_->wait_for_action_server(3s)) {
      throw std::runtime_error("action server not ready");
    }
  }

  ~RunningSystem() {
    executor_.cancel();
    spinner_.join();
  }

  [[nodiscard]] rclcpp_action::ClientGoalHandle<Action>::SharedPtr Send(
      const double x, const double y,
      std::vector<Action::Feedback>* feedback = nullptr) {
    Action::Goal goal;
    goal.target_x_m = x;
    goal.target_y_m = y;
    rclcpp_action::Client<Action>::SendGoalOptions options;
    if (feedback) {
      options.feedback_callback =
          [feedback](auto, const std::shared_ptr<const Action::Feedback> value) {
            feedback->push_back(*value);
          };
    }
    auto future = client_->async_send_goal(goal, options);
    if (future.wait_for(3s) != std::future_status::ready) {
      return nullptr;
    }
    return future.get();
  }

  [[nodiscard]] auto Result(
      const rclcpp_action::ClientGoalHandle<Action>::SharedPtr& handle) {
    return client_->async_get_result(handle);
  }

  auto Cancel(const rclcpp_action::ClientGoalHandle<Action>::SharedPtr& handle) {
    return client_->async_cancel_goal(handle);
  }

  [[nodiscard]] std::vector<lunar_planning_msgs::msg::PathReference> Paths()
      const {
    std::scoped_lock lock{mutex_};
    return paths_;
  }
  [[nodiscard]] std::vector<nav_msgs::msg::Path> Globals() const {
    std::scoped_lock lock{mutex_};
    return globals_;
  }
  [[nodiscard]] std::vector<nav_msgs::msg::Path> LocalPaths() const {
    std::scoped_lock lock{mutex_};
    return local_paths_;
  }
  [[nodiscard]] std::vector<diagnostic_msgs::msg::DiagnosticArray> Diagnostics()
      const {
    std::scoped_lock lock{mutex_};
    return diagnostics_;
  }
  [[nodiscard]] std::shared_ptr<IncrementalNavigationNode> server() const {
    return server_;
  }
#if defined(LUNAR_BUILD_DEMO)
  [[nodiscard]] std::vector<nav_msgs::msg::OccupancyGrid> FineStates() const {
    std::scoped_lock lock{mutex_};
    return fine_states_;
  }
  [[nodiscard]] std::vector<nav_msgs::msg::OccupancyGrid> GuidanceStates()
      const {
    std::scoped_lock lock{mutex_};
    return guidance_states_;
  }
  [[nodiscard]] std::vector<visualization_msgs::msg::MarkerArray>
  StartPatches() const {
    std::scoped_lock lock{mutex_};
    return start_patches_;
  }
#endif

 private:
  std::string suffix_;
  std::shared_ptr<IncrementalNavigationNode> server_;
  rclcpp::Node::SharedPtr client_node_;
  rclcpp_action::Client<Action>::SharedPtr client_;
  rclcpp::Subscription<lunar_planning_msgs::msg::PathReference>::SharedPtr
      path_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr
      local_path_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_subscription_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_subscription_;
#if defined(LUNAR_BUILD_DEMO)
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
      fine_state_subscription_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
      guidance_state_subscription_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr
      start_patch_subscription_;
#endif
  mutable std::mutex mutex_;
  std::vector<lunar_planning_msgs::msg::PathReference> paths_;
  std::vector<nav_msgs::msg::Path> local_paths_;
  std::vector<nav_msgs::msg::Path> globals_;
  std::vector<diagnostic_msgs::msg::DiagnosticArray> diagnostics_;
#if defined(LUNAR_BUILD_DEMO)
  std::vector<nav_msgs::msg::OccupancyGrid> fine_states_;
  std::vector<nav_msgs::msg::OccupancyGrid> guidance_states_;
  std::vector<visualization_msgs::msg::MarkerArray> start_patches_;
#endif
  rclcpp::executors::MultiThreadedExecutor executor_;
  std::jthread spinner_;
};

TEST(IncrementalNavigationNode, TerminalMappingIsExact) {
  EXPECT_EQ(TerminalStateFor(core::SessionOutcome::kGoalReached),
            ActionTerminalState::kSucceeded);
  EXPECT_EQ(TerminalStateFor(core::SessionOutcome::kCanceled),
            ActionTerminalState::kCanceled);
  for (const auto outcome : {
           core::SessionOutcome::kNoPath, core::SessionOutcome::kInvalidGoal,
           core::SessionOutcome::kMapUnavailable,
           core::SessionOutcome::kTimeout,
           core::SessionOutcome::kInternalError}) {
    EXPECT_EQ(TerminalStateFor(outcome), ActionTerminalState::kAborted);
  }
}

TEST(IncrementalNavigationNode, ActiveCancelIsDeferredUntilGoalHandleIsCanceling) {
  EXPECT_EQ(ClassifyCancelHandling(true, false),
            CancelHandlingDecision::kDefer);
  EXPECT_EQ(ClassifyCancelHandling(true, true),
            CancelHandlingDecision::kFinalize);
  EXPECT_EQ(ClassifyCancelHandling(false, false),
            CancelHandlingDecision::kIgnore);
  EXPECT_EQ(ClassifyCancelHandling(false, true),
            CancelHandlingDecision::kIgnore);
}

TEST(IncrementalNavigationNode, PathPublisherQosIsReliableTransientLocalDepthOne) {
  const auto profile = PathReferenceQos().get_rmw_qos_profile();
  EXPECT_EQ(profile.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  EXPECT_EQ(profile.depth, 1U);
  EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
}

TEST(IncrementalNavigationNode,
     HasExactlyOneGridMapInputWithConfiguredLocalQos) {
  RunningSystem system("map_qos", IncrementalNavigationNodeDependencies{});

  const auto local = system.server()->get_subscriptions_info_by_topic(
      "/test/map_qos/local");
  ASSERT_EQ(local.size(), 1U);
  const auto local_profile = local.front().qos_profile().get_rmw_qos_profile();
  EXPECT_EQ(local_profile.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  EXPECT_EQ(local_profile.durability,
            RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

  std::size_t grid_map_subscription_count = 0U;
  for (const auto& [topic, unused_types] :
       system.server()->get_topic_names_and_types()) {
    static_cast<void>(unused_types);
    for (const auto& subscription :
         system.server()->get_subscriptions_info_by_topic(topic)) {
      if (subscription.topic_type() == "grid_map_msgs/msg/GridMap") {
        ++grid_map_subscription_count;
      }
    }
  }
  EXPECT_EQ(grid_map_subscription_count, 1U);
}

TEST(IncrementalNavigationNode, LocalMapQosCanBeOverriddenForVolatileReplay) {
  auto options = ServerOptions("volatile_map_qos");
  options.append_parameter_override("local_map_qos_reliability", "best_effort");
  options.append_parameter_override("local_map_qos_durability", "volatile");
  auto server =
      std::make_shared<IncrementalNavigationNode>(options,
                                             IncrementalNavigationNodeDependencies{});

  const auto subscriptions = server->get_subscriptions_info_by_topic(
      "/test/volatile_map_qos/local");
  ASSERT_EQ(subscriptions.size(), 1U);
  const auto profile =
      subscriptions.front().qos_profile().get_rmw_qos_profile();
  EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
  EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_VOLATILE);
}

TEST(IncrementalNavigationNode, StartupParametersRequireRestart) {
  auto server = std::make_shared<IncrementalNavigationNode>(
      ServerOptions("startup_parameters"), IncrementalNavigationNodeDependencies{});
  const auto result = server->set_parameters_atomically(
      {rclcpp::Parameter("map_frame", "world")});
  EXPECT_FALSE(result.successful);
  EXPECT_NE(result.reason.find("map_frame"), std::string::npos);
  EXPECT_NE(result.reason.find("restart"), std::string::npos);
  EXPECT_EQ(server->get_parameter("map_frame").as_string(), "map");
}

TEST(IncrementalNavigationNode, RejectsUnsupportedLocalMapQosParameters) {
  auto options = ServerOptions("invalid_map_qos");
  options.append_parameter_override("local_map_qos_durability", "volatilee");

  EXPECT_THROW(
      std::make_shared<IncrementalNavigationNode>(
          options, IncrementalNavigationNodeDependencies{}),
      std::runtime_error);
}

TEST(IncrementalNavigationNode, ReportsIncompatibleLocalMapQosWithoutAMapSample) {
  const std::string suffix = "incompatible_map_qos";
  RunningSystem system(suffix, IncrementalNavigationNodeDependencies{});
  auto producer = std::make_shared<rclcpp::Node>("producer_" + suffix);
  auto incompatible_publisher =
      producer->create_publisher<grid_map_msgs::msg::GridMap>(
          "/test/" + suffix + "/local",
          rclcpp::QoS{rclcpp::KeepLast{1}}
              .best_effort()
              .durability_volatile());

  ASSERT_TRUE(WaitFor([&] {
    for (const auto& diagnostics : system.Diagnostics()) {
      if (FindDiagnosticValue(diagnostics, "incompatible_qos_count") == "1") {
        return true;
      }
    }
    return false;
  }));
  const auto diagnostics = system.Diagnostics();
  EXPECT_EQ(FindDiagnosticValue(diagnostics.back(), "received_map_count"),
            "0");
  EXPECT_EQ(diagnostics.back().status.front().level,
            diagnostic_msgs::msg::DiagnosticStatus::ERROR);
}

TEST(IncrementalNavigationNode,
     LateJoinedTask3MapIsAppliedAfterDirectTfArrives) {
  const std::string suffix = "late_map_tf";
  auto producer = std::make_shared<rclcpp::Node>("producer_" + suffix);
  auto local_publisher = producer->create_publisher<grid_map_msgs::msg::GridMap>(
      "/test/" + suffix + "/local",
      rclcpp::QoS{rclcpp::KeepLast{1}}.reliable().transient_local());
  auto tf_publisher = producer->create_publisher<tf2_msgs::msg::TFMessage>(
      "/test/" + suffix + "/tf",
      rclcpp::QoS{rclcpp::KeepLast{10}}.reliable().durability_volatile());

  local_publisher->publish(Task3LocalMap());
  RunningSystem system(suffix, IncrementalNavigationNodeDependencies{});
  ASSERT_TRUE(WaitFor(
      [&] { return local_publisher->get_subscription_count() == 1U; }));
  ASSERT_TRUE(WaitFor([&] {
    for (const auto& diagnostics : system.Diagnostics()) {
      if (FindDiagnosticValue(diagnostics, "received_map_count") == "1" &&
          FindDiagnosticValue(diagnostics, "applied_map_count") == "0" &&
          FindDiagnosticValue(diagnostics, "rejected_map_count") == "0") {
        return true;
      }
    }
    return false;
  }));

  ASSERT_TRUE(
      WaitFor([&] { return tf_publisher->get_subscription_count() == 1U; }));
  tf_publisher->publish(DirectMapFromOdom());
  ASSERT_TRUE(WaitFor([&] {
    for (const auto& diagnostics : system.Diagnostics()) {
      if (FindDiagnosticValue(diagnostics, "received_map_count") == "1" &&
          FindDiagnosticValue(diagnostics, "applied_map_count") == "1") {
        return true;
      }
    }
    return false;
  }));
}

TEST(IncrementalNavigationNode, OdometryAndTfUseDeploymentBestEffortDepthTen) {
  const auto expected = StateInputQos().get_rmw_qos_profile();
  EXPECT_EQ(expected.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  EXPECT_EQ(expected.depth, 10U);
  EXPECT_EQ(expected.reliability, RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
  EXPECT_EQ(expected.durability, RMW_QOS_POLICY_DURABILITY_VOLATILE);

  RunningSystem system("state_qos", IncrementalNavigationNodeDependencies{});
  for (const std::string topic : {"/test/state_qos/odom",
                                  "/test/state_qos/tf"}) {
    const auto subscriptions =
        system.server()->get_subscriptions_info_by_topic(topic);
    ASSERT_EQ(subscriptions.size(), 1U) << topic;
    const auto profile =
        subscriptions.front().qos_profile().get_rmw_qos_profile();
    EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        << topic;
    EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_VOLATILE)
        << topic;
  }
}

#if defined(LUNAR_BUILD_DEMO)
TEST(IncrementalNavigationNode, DebugTopicsAreAbsentByDefault) {
  RunningSystem system("debug_disabled", IncrementalNavigationNodeDependencies{});

  EXPECT_TRUE(system.server()->has_parameter("enable_debug_visualization"));
  EXPECT_TRUE(system.server()->has_parameter("debug_topic_prefix"));
  EXPECT_TRUE(system.server()->has_parameter("debug_fine_window_m"));
  EXPECT_TRUE(system.server()->has_parameter("debug_cost_display_max"));
  EXPECT_TRUE(system.server()
                  ->get_publishers_info_by_topic(
                      "/planning_demo/debug/fine_state")
                  .empty());
}

TEST(IncrementalNavigationNode, EnabledDebugUsesConfiguredTransientLocalTopics) {
  auto control = std::make_shared<FakeControl>();
  const auto fine = MakeFine();
  auto server = std::make_shared<IncrementalNavigationNode>(
      ServerOptions("debug_enabled", true),
      IncrementalNavigationNodeDependencies{
          .ports_factory = FakePorts(control),
          .snapshots = core::SnapshotBundle{
              .fine = fine, .guidance = MakeGuidance(fine)},
          .state = core::StateInput{.base_link_pose = {
              .position_m = {.x = 0.5, .y = 0.5}}},
      });

  ASSERT_FALSE(server->get_publishers_info_by_topic(
                   "/planning_demo/debug/debug_enabled/fine_state")
                   .empty());
  const auto profile = DebugVisualizationQos().get_rmw_qos_profile();
  EXPECT_EQ(profile.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  EXPECT_EQ(profile.depth, 1U);
  EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
}

[[nodiscard]] std::int8_t GridCell(const nav_msgs::msg::OccupancyGrid& grid,
                                   const std::size_t x,
                                   const std::size_t y) {
  return grid.data.at(y * grid.info.width + x);
}

[[nodiscard]] bool HasMarkerCell(
    const visualization_msgs::msg::MarkerArray& markers, const double x,
    const double y) {
  for (const auto& marker : markers.markers) {
    if (marker.ns != "start_patch_assumed_unknown") {
      continue;
    }
    for (const auto& point : marker.points) {
      if (point.x == x && point.y == y) {
        return true;
      }
    }
  }
  return false;
}

TEST(IncrementalNavigationNode,
     EnabledDebugPublishesCycleSnapshotAndActualStartPatchView) {
  auto control = std::make_shared<FakeControl>();
  control->make_debug_start_patch = true;
  const auto fine = MakeDebugFine();
  RunningSystem system("debug_cycle", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{
          .fine = fine, .guidance = MakeDebugGuidance(fine)},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
  }, true);

  auto handle = system.Send(5.5, 0.5);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(WaitFor([&] {
    return !system.FineStates().empty() && !system.GuidanceStates().empty() &&
           !system.StartPatches().empty() && !system.Diagnostics().empty() &&
           !system.Paths().empty();
  }));

  const auto fine_states = system.FineStates();
  const auto guidance_states = system.GuidanceStates();
  const auto patches = system.StartPatches();
  const auto diagnostics = system.Diagnostics();
  ASSERT_FALSE(system.Paths().empty());
  EXPECT_EQ(system.Paths().back().state,
            lunar_planning_msgs::msg::PathReference::ACTIVE);
  EXPECT_EQ(FindDiagnosticValue(diagnostics.back(), "reason_code"),
            "EXECUTING");
  EXPECT_EQ(FindDiagnosticValue(diagnostics.back(),
                                "fine_traversability_revision"),
            "17");
  EXPECT_EQ(FindDiagnosticValue(diagnostics.back(), "global_guidance_revision"),
            "23");
  EXPECT_EQ(system.Paths().back().traversability_revision, 17U);

  ASSERT_FALSE(fine_states.empty());
  EXPECT_EQ(fine_states.back().header.frame_id, "map");
  EXPECT_EQ(GridCell(fine_states.back(), 1U, 1U), -1);
  EXPECT_EQ(GridCell(fine_states.back(), 2U, 1U), 100);
  ASSERT_FALSE(guidance_states.empty());
  EXPECT_EQ(guidance_states.back().header.frame_id, "map");
  EXPECT_EQ(GridCell(guidance_states.back(), 1U, 1U), 0);
  EXPECT_EQ(GridCell(guidance_states.back(), 2U, 1U), 100);
  ASSERT_FALSE(patches.empty());
  EXPECT_TRUE(HasMarkerCell(patches.back(), 1.5, 1.5));
  EXPECT_FALSE(HasMarkerCell(patches.back(), 2.5, 1.5));
  system.Cancel(handle);
}
#endif

TEST(IncrementalNavigationNode, GoalTolerancesAreFrozenPerPlatformProfile) {
  const auto wheel = PlatformProfileFor(
      core::WheeledCapability{.minimum_clearance_m = 0.2}, 0.4);
  const auto legged = PlatformProfileFor(
      core::LeggedCapability{.minimum_body_clearance_m = 0.3}, 0.6);
  EXPECT_GT(wheel.goal_position_tolerance_m, 0.0);
  EXPECT_GT(wheel.goal_yaw_tolerance_rad, 0.0);
  EXPECT_GT(legged.goal_position_tolerance_m, 0.0);
  EXPECT_GT(legged.goal_yaw_tolerance_rad, 0.0);
  EXPECT_NE(wheel.goal_position_tolerance_m,
            legged.goal_position_tolerance_m);
  EXPECT_NE(wheel.goal_yaw_tolerance_rad, legged.goal_yaw_tolerance_rad);
  EXPECT_DOUBLE_EQ(wheel.start_blind_zone_margin_m, 0.4);
  EXPECT_DOUBLE_EQ(legged.start_blind_zone_margin_m, 0.6);
  EXPECT_DOUBLE_EQ(wheel.preferred_clearance_m, 0.2);
  EXPECT_DOUBLE_EQ(legged.preferred_clearance_m, 0.3);
  EXPECT_DOUBLE_EQ(wheel.clearance_weight, 0.0);
  EXPECT_DOUBLE_EQ(legged.clearance_weight, 0.0);
}

TEST(IncrementalNavigationNode,
     ActiveLocalPathIsRepublishedAsStandardControllerPath) {
  auto control = std::make_shared<FakeControl>();
  const auto fine = MakeFine();
  RunningSystem system("standard_local_path", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = fine},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
  });

  auto handle = system.Send(3.5, 0.5);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(WaitFor([&] {
    return !system.Paths().empty() && !system.LocalPaths().empty();
  }));

  const auto reference = system.Paths().back();
  const auto path = system.LocalPaths().back();
  ASSERT_EQ(reference.state, lunar_planning_msgs::msg::PathReference::ACTIVE);
  ASSERT_FALSE(reference.path.poses.empty());
  EXPECT_EQ(path.header.frame_id, "map");
  ASSERT_EQ(path.poses.size(), reference.path.poses.size());
  EXPECT_DOUBLE_EQ(path.poses.front().pose.position.x,
                   reference.path.poses.front().pose.position.x);
  EXPECT_DOUBLE_EQ(path.poses.front().pose.position.y,
                   reference.path.poses.front().pose.position.y);
  EXPECT_DOUBLE_EQ(path.poses.back().pose.position.x,
                   reference.path.poses.back().pose.position.x);
  EXPECT_DOUBLE_EQ(path.poses.back().pose.position.y,
                   reference.path.poses.back().pose.position.y);
  EXPECT_DOUBLE_EQ(path.poses.back().pose.orientation.w,
                   reference.path.poses.back().pose.orientation.w);
  system.Cancel(handle);
}

TEST(IncrementalNavigationNode, PlanningDiagnosticsUseMeasuredSolverStatistics) {
  auto control = std::make_shared<FakeControl>();
  control->publish_global_route = true;
  const auto fine = MakeFine();
  RunningSystem system("measured_diagnostics", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{
          .fine = fine, .guidance = MakeGuidance(fine)},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
  });
  auto handle = system.Send(5.5, 0.5);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(WaitFor([&] { return !system.Diagnostics().empty(); }));
  const auto diagnostics = system.Diagnostics();
  EXPECT_EQ(FindDiagnosticValue(diagnostics.front(), "global_open_peak"), "5");
  EXPECT_EQ(FindDiagnosticValue(diagnostics.front(), "local_open_peak"), "9");
  EXPECT_EQ(FindDiagnosticValue(diagnostics.front(), "postprocess_elapsed_ms"),
            "3");
  system.Cancel(handle);
}

TEST(IncrementalNavigationNode,
     DefaultRuntimeBudgetsReachGlobalAndLocalSolvers) {
  auto control = std::make_shared<FakeControl>();
  control->publish_global_route = true;
  const auto fine = MakeFine();
  RunningSystem system("default_timing", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{
          .fine = fine, .guidance = MakeGuidance(fine)},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
  });

  auto handle = system.Send(5.5, 0.5);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(WaitFor([&] {
    return !system.Paths().empty() &&
           control->global_deadline_remaining_ms.load() >= 0 &&
           control->local_deadline_remaining_ms.load() >= 0;
  }));
  EXPECT_GT(control->global_deadline_remaining_ms.load(), 400);
  EXPECT_LT(control->global_deadline_remaining_ms.load(), 550);
  EXPECT_GT(control->local_deadline_remaining_ms.load(), 2500);
  EXPECT_LT(control->local_deadline_remaining_ms.load(), 3100);
  auto result = system.Result(handle);
  system.Cancel(handle);
  ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
}

TEST(IncrementalNavigationNode,
     SlaMissKeepsACompletePathAndPublishesWarningDiagnostics) {
  auto control = std::make_shared<FakeControl>();
  control->local_delay = 5ms;
  RunningSystem system("sla_miss", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
  }, false, TimingOverrides{.planning_sla_ms = 2,
                            .planning_hard_timeout_ms = 100,
                            .global_subdeadline_ms = 1});

  auto handle = system.Send(5.5, 0.5);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(WaitFor([&] {
    const auto diagnostics = system.Diagnostics();
    return !system.Paths().empty() &&
           std::any_of(diagnostics.begin(), diagnostics.end(),
                       [](const auto& message) {
                         return FindDiagnosticValue(message, "latency_class") ==
                                "SLA_MISSED";
                       });
  }));
  EXPECT_EQ(system.Paths().back().state,
            lunar_planning_msgs::msg::PathReference::ACTIVE);
  const auto diagnostics = system.Diagnostics();
  const auto sla_miss = std::find_if(
      diagnostics.begin(), diagnostics.end(), [](const auto& message) {
        return FindDiagnosticValue(message, "latency_class") == "SLA_MISSED";
      });
  ASSERT_NE(sla_miss, diagnostics.end());
  EXPECT_EQ(sla_miss->status.front().level,
            diagnostic_msgs::msg::DiagnosticStatus::WARN);
  auto result = system.Result(handle);
  system.Cancel(handle);
  ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
}

TEST(IncrementalNavigationNode,
     ConfiguredHardTimeoutAbortsWithoutPublishingAPath) {
  auto control = std::make_shared<FakeControl>();
  control->local_delay = 20ms;
  RunningSystem system("hard_timeout", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
  }, false, TimingOverrides{.planning_sla_ms = 5,
                            .planning_hard_timeout_ms = 10,
                            .global_subdeadline_ms = 1});

  auto handle = system.Send(5.5, 0.5);
  ASSERT_TRUE(handle);
  auto result = system.Result(handle);
  ASSERT_EQ(result.wait_for(500ms), std::future_status::ready);
  const auto wrapped = result.get();
  EXPECT_EQ(wrapped.code, rclcpp_action::ResultCode::ABORTED);
  ASSERT_TRUE(wrapped.result);
  EXPECT_EQ(wrapped.result->outcome, Action::Result::TIMEOUT);
  EXPECT_EQ(wrapped.result->reason_code, "TIMEOUT");
  EXPECT_TRUE(system.Paths().empty());
  ASSERT_TRUE(WaitFor([&] {
    const auto diagnostics = system.Diagnostics();
    return !diagnostics.empty() &&
           FindDiagnosticValue(diagnostics.back(), "latency_class") ==
               "HARD_TIMEOUT";
  }));
}

TEST(IncrementalNavigationNode, RejectsInvalidPlanningTimingParameters) {
  EXPECT_NO_THROW(std::make_shared<IncrementalNavigationNode>(
      ServerOptions("global_after_sla", false,
                    TimingOverrides{.planning_sla_ms = 2000,
                                    .planning_hard_timeout_ms = 3000,
                                    .global_subdeadline_ms = 2500}),
      IncrementalNavigationNodeDependencies{}));
  for (const auto& [suffix, timing] :
       std::vector<std::pair<std::string, TimingOverrides>>{
           {"zero_sla", {.planning_sla_ms = 0,
                         .planning_hard_timeout_ms = 3000,
                         .global_subdeadline_ms = 500}},
           {"equal_sla_hard", {.planning_sla_ms = 3000,
                               .planning_hard_timeout_ms = 3000,
                               .global_subdeadline_ms = 500}},
           {"global_at_hard", {.planning_sla_ms = 2000,
                               .planning_hard_timeout_ms = 3000,
                               .global_subdeadline_ms = 3000}},
       }) {
    EXPECT_THROW(
        std::make_shared<IncrementalNavigationNode>(
            ServerOptions(suffix, false, timing),
            IncrementalNavigationNodeDependencies{}),
        std::runtime_error)
        << suffix;
  }
}

TEST(IncrementalNavigationNode,
     NewGoalAlwaysPreemptsAndEveryTerminalInvalidatesTheActiveSegmentFirst) {
  auto control = std::make_shared<FakeControl>();
  auto log = std::make_shared<EventLog>();
  RunningSystem system("preempt", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
      .event_sink = [log](const std::string& event) { log->Push(event); },
  });

  std::vector<Action::Feedback> feedback;
  auto first = system.Send(3.5, 0.5, &feedback);
  ASSERT_TRUE(first);
  ASSERT_TRUE(WaitFor([&] {
    const auto paths = system.Paths();
    return !paths.empty() && paths.back().state == paths.back().ACTIVE;
  }));
  const auto first_uuid = first->get_goal_id();
  ASSERT_EQ(system.Paths().front().session_id.uuid, first_uuid);

  auto first_result = system.Result(first);
  auto second = system.Send(4.5, 0.5);
  ASSERT_TRUE(second);
  ASSERT_EQ(first_result.wait_for(3s), std::future_status::ready);
  const auto replaced = first_result.get();
  EXPECT_EQ(replaced.code, rclcpp_action::ResultCode::CANCELED);
  ASSERT_TRUE(replaced.result);
  EXPECT_EQ(replaced.result->outcome, Action::Result::CANCELED);
  EXPECT_EQ(replaced.result->reason_code, "PREEMPTED");

  auto second_result = system.Result(second);
  system.Cancel(second);
  ASSERT_EQ(second_result.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(second_result.get().code, rclcpp_action::ResultCode::CANCELED);

  const auto events = log->Copy();
  ASSERT_GE(events.size(), 6U);
  EXPECT_EQ(events[0], "path:ACTIVE");
  EXPECT_EQ(events[1], "path:INVALIDATED");
  EXPECT_EQ(events[2], "terminal:PREEMPTED");
  EXPECT_EQ(events[3], "path:ACTIVE");
  EXPECT_EQ(events[4], "path:INVALIDATED");
  EXPECT_EQ(events[5], "terminal:CANCELED");
  ASSERT_FALSE(feedback.empty());
  EXPECT_EQ(feedback.front().session_state, Action::Feedback::EXECUTING);
  EXPECT_EQ(feedback.front().planning_cycle, 1U);
  EXPECT_EQ(feedback.front().active_segment_revision, 1U);
  EXPECT_EQ(feedback.front().reason_code, "PLAN_FOUND");
  ASSERT_FALSE(system.Globals().empty());
  EXPECT_TRUE(system.Globals().back().poses.empty());
}

TEST(IncrementalNavigationNode, GoalReachedInvalidatesThenSucceeds) {
  auto control = std::make_shared<FakeControl>();
  control->reaches_final_goal = true;
  auto states = std::make_shared<StateQueue>();
  auto log = std::make_shared<EventLog>();
  RunningSystem system("reached", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
      .state_source = [states] { return states->Take(); },
      .event_sink = [log](const std::string& event) { log->Push(event); },
  });
  auto handle = system.Send(3.5, 0.5);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(WaitFor([&] { return system.Paths().size() >= 1U; }));
  auto result = system.Result(handle);
  states->Push({.position_m = {.x = 3.5, .y = 0.5}});
  ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
  const auto wrapped = result.get();
  EXPECT_EQ(wrapped.code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_TRUE(wrapped.result);
  EXPECT_EQ(wrapped.result->outcome, Action::Result::GOAL_REACHED);
  EXPECT_EQ(log->Copy(), (std::vector<std::string>{
      "path:ACTIVE", "path:INVALIDATED", "terminal:GOAL_REACHED"}));
  ASSERT_TRUE(WaitFor([&] {
    const auto diagnostics = system.Diagnostics();
    return !diagnostics.empty() &&
           FindDiagnosticValue(diagnostics.back(), "reason_code") ==
               "GOAL_REACHED";
  }));
  const auto diagnostics = system.Diagnostics();
  const auto& terminal_diagnostics = diagnostics.back();
  EXPECT_EQ(FindDiagnosticValue(terminal_diagnostics, "segment_revision"),
            "1");
  EXPECT_EQ(FindDiagnosticValue(terminal_diagnostics, "path_state"),
            "INVALIDATED");
  EXPECT_EQ(FindDiagnosticValue(terminal_diagnostics, "reason_code"),
            "GOAL_REACHED");
  EXPECT_EQ(FindDiagnosticValue(terminal_diagnostics, "session_id"),
            GoalUuidHex(handle->get_goal_id()));
  EXPECT_FALSE(FindDiagnosticValue(terminal_diagnostics, "session_id").empty());
  EXPECT_EQ(FindDiagnosticValue(terminal_diagnostics, "path_points"), "0");
  EXPECT_EQ(wrapped.result->last_segment_revision, 1U);
}

TEST(IncrementalNavigationNode,
     SegmentEndPublishesASecondMonotonicActiveSegmentInTheSameSession) {
  auto control = std::make_shared<FakeControl>();
  control->segment_ends = {{.x = 2.5, .y = 0.5}, {.x = 4.5, .y = 0.5}};
  auto states = std::make_shared<StateQueue>();
  RunningSystem system("segment_end", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
      .state_source = [states] { return states->Take(); },
  });
  auto handle = system.Send(7.5, 0.5);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(WaitFor([&] { return system.Paths().size() >= 1U; }));

  states->Push({.position_m = {.x = 2.5, .y = 0.5}});
  ASSERT_TRUE(WaitFor([&] { return system.Paths().size() >= 2U; }));
  const auto paths = system.Paths();
  ASSERT_EQ(paths[0].state, paths[0].ACTIVE);
  ASSERT_EQ(paths[1].state, paths[1].ACTIVE);
  EXPECT_EQ(paths[0].session_id.uuid, paths[1].session_id.uuid);
  EXPECT_EQ(paths[0].segment_revision, 1U);
  EXPECT_EQ(paths[1].segment_revision, 2U);
  system.Cancel(handle);
}

TEST(IncrementalNavigationNode, TrackingCompletionRequiresMatchingStoppedFeedbackAfterPoseArrival) {
  auto control = std::make_shared<FakeControl>();
  control->reaches_final_goal = true;
  auto states = std::make_shared<StateQueue>();
  RunningSystem system("tracking_completed", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {.position_m = {.x = 0.5, .y = 0.5}}},
      .state_source = [states] { return states->Take(); },
  });
  auto producer = rclcpp::Node::make_shared("tracking_completed_publisher");
  auto publisher = producer->create_publisher<lunar_planning_msgs::msg::TrackingStatus>(
      "/test/tracking_completed/tracking", 10);
  ASSERT_TRUE(WaitFor([&] { return publisher->get_subscription_count() > 0; }));
  auto goal = system.Send(3.5, 0.5);
  ASSERT_TRUE(goal);
  ASSERT_TRUE(WaitFor([&] { return !system.Paths().empty(); }));
  auto result = system.Result(goal);
  states->Push({.position_m = {.x = 3.5, .y = 0.5}});
  EXPECT_EQ(result.wait_for(150ms), std::future_status::timeout);
  lunar_planning_msgs::msg::TrackingStatus status;
  status.session_id.uuid = goal->get_goal_id();
  status.segment_revision = 1;
  status.state = status.COMPLETED;
  status.linear_speed_mps = 0.2;
  publisher->publish(status);
  EXPECT_EQ(result.wait_for(80ms), std::future_status::timeout);
  status.linear_speed_mps = 0.;
  status.segment_revision = 0;
  publisher->publish(status);
  EXPECT_EQ(result.wait_for(80ms), std::future_status::timeout);
  status.segment_revision = 1;
  status.session_id.uuid[0] ^= 1;
  publisher->publish(status);
  EXPECT_EQ(result.wait_for(80ms), std::future_status::timeout);
  status.session_id.uuid = goal->get_goal_id();
  publisher->publish(status);
  ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
  const auto wrapped = result.get();
  EXPECT_EQ(wrapped.code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_TRUE(wrapped.result);
  EXPECT_EQ(wrapped.result->outcome, Action::Result::GOAL_REACHED);
  EXPECT_EQ(wrapped.result->last_segment_revision, 1U);
}

TEST(IncrementalNavigationNode, PendingFineCollisionInvalidatesBeforeMatchingCompletion) {
  auto control = std::make_shared<FakeControl>();
  control->reaches_final_goal = true;
  // Once the old reference is invalidated, a replacement cannot be certified.
  control->fail_local_after_calls = 1U;
  auto states = std::make_shared<StateQueue>();
  auto gate = std::make_shared<CallbackGate>();
  auto log = std::make_shared<EventLog>();
  auto inject_map = std::make_shared<std::atomic<bool>>(false);
  const auto blocked = MakeFine(2U, core::GridIndex{.x = 3, .y = 0});
  RunningSystem system("tracking_collision", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {.position_m = {.x = 0.5, .y = 0.5}}},
      .state_source = [states, gate] { gate->WaitIfArmed(); return states->Take(); },
      .snapshot_source = [inject_map, blocked]() -> std::optional<core::SnapshotBundle> {
        if (inject_map->exchange(false)) return core::SnapshotBundle{.fine = blocked};
        return std::nullopt;
      },
      .event_sink = [log](const std::string& event) { log->Push(event); },
  });
  auto producer = rclcpp::Node::make_shared("tracking_collision_publisher");
  auto publisher = producer->create_publisher<lunar_planning_msgs::msg::TrackingStatus>(
      "/test/tracking_collision/tracking", 10);
  ASSERT_TRUE(WaitFor([&] { return publisher->get_subscription_count() > 0; }));
  auto goal = system.Send(3.5, 0.5);
  ASSERT_TRUE(goal);
  ASSERT_TRUE(WaitFor([&] { return !system.Paths().empty(); }));
  auto result = system.Result(goal);
  gate->Arm();
  const bool entered = gate->WaitUntilEntered();
  if (!entered) gate->Release();
  ASSERT_TRUE(entered);
  states->Push({.position_m = {.x = 3.5, .y = 0.5}});
  inject_map->store(true);
  lunar_planning_msgs::msg::TrackingStatus status;
  status.session_id.uuid = goal->get_goal_id();
  status.segment_revision = 1;
  status.state = status.COMPLETED;
  publisher->publish(status);
  // The planning tick is held before capturing either event. A callback-level
  // receipt proves completion is pending before the new snapshot is captured.
  const bool received = WaitFor([&] {
    const auto events = log->Copy();
    return std::find(events.begin(), events.end(), "tracking:RECEIVED") != events.end();
  });
  gate->Release();
  ASSERT_TRUE(received);
  ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
  const auto wrapped = result.get();
  EXPECT_EQ(wrapped.code, rclcpp_action::ResultCode::ABORTED);
  ASSERT_TRUE(wrapped.result);
  EXPECT_EQ(wrapped.result->outcome, Action::Result::NO_PATH);
  ASSERT_TRUE(WaitFor([&] { return system.Paths().size() >= 2U; }));
  const auto paths = system.Paths();
  EXPECT_EQ(paths[1].state, paths[1].INVALIDATED);
  EXPECT_EQ(paths[1].traversability_revision, 2U);
  EXPECT_EQ(control->local_calls, 2U);
}

TEST(IncrementalNavigationNode, StoppedMatchingTrackingFailureReplansAndStaleFeedbackIsIgnored) {
  auto control = std::make_shared<FakeControl>();
  control->segment_ends = {{.x = 3.5, .y = 0.5}};
  RunningSystem system("tracking_failure", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {.position_m = {.x = 0.5, .y = 0.5}}},
  });
  auto publisher_node = rclcpp::Node::make_shared("tracking_failure_publisher");
  auto publisher = publisher_node->create_publisher<lunar_planning_msgs::msg::TrackingStatus>(
      "/test/tracking_failure/tracking", 10);
  ASSERT_TRUE(WaitFor([&] { return publisher->get_subscription_count() > 0; }));
  auto goal = system.Send(7.5, 0.5);
  ASSERT_TRUE(goal);
  ASSERT_TRUE(WaitFor([&] { return !system.Paths().empty(); }));
  lunar_planning_msgs::msg::TrackingStatus status;
  status.session_id.uuid = goal->get_goal_id();
  status.segment_revision = 1;
  status.state = status.FAILED;
  status.linear_speed_mps = 0.2;
  publisher->publish(status);
  std::this_thread::sleep_for(80ms);
  EXPECT_EQ(system.Paths().size(), 1U);
  status.linear_speed_mps = 0.;
  status.session_id.uuid[0] ^= 1;
  publisher->publish(status);
  std::this_thread::sleep_for(80ms);
  EXPECT_EQ(system.Paths().size(), 1U);
  status.session_id.uuid = goal->get_goal_id();
  publisher->publish(status);
  ASSERT_TRUE(WaitFor([&] { return system.Paths().size() >= 3U; }));
  EXPECT_EQ(system.Paths().back().segment_revision, 2U);
  publisher->publish(status);
  std::this_thread::sleep_for(80ms);
  EXPECT_EQ(system.Paths().size(), 3U);
  system.Cancel(goal);
}

TEST(IncrementalNavigationNode,
     DeviationInvalidatesThenReplansWithANewerActiveSegment) {
  auto control = std::make_shared<FakeControl>();
  control->segment_ends = {{.x = 3.5, .y = 0.5}, {.x = 3.5, .y = 5.5}};
  auto states = std::make_shared<StateQueue>();
  auto log = std::make_shared<EventLog>();
  RunningSystem system("deviation", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
      .state_source = [states] { return states->Take(); },
      .event_sink = [log](const std::string& event) { log->Push(event); },
  });
  auto handle = system.Send(7.5, 0.5);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(WaitFor([&] { return system.Paths().size() >= 1U; }));

  states->Push({.position_m = {.x = 0.5, .y = 5.5}});
  ASSERT_TRUE(WaitFor([&] { return log->Copy().size() >= 3U; }));
  const auto events = log->Copy();
  EXPECT_EQ(events[0], "path:ACTIVE");
  EXPECT_EQ(events[1], "path:INVALIDATED");
  EXPECT_EQ(events[2], "path:ACTIVE");
  ASSERT_TRUE(WaitFor([&] {
    const auto paths = system.Paths();
    return !paths.empty() && paths.back().state == paths.back().ACTIVE &&
           paths.back().segment_revision == 2U;
  }));
  const auto paths = system.Paths();
  ASSERT_EQ(paths[0].state, paths[0].ACTIVE);
  EXPECT_EQ(paths[0].segment_revision, 1U);
  EXPECT_EQ(paths.back().segment_revision, 2U);
  system.Cancel(handle);
}

TEST(IncrementalNavigationNode,
     CancelDuringBlockedReplanInvalidatesThenCancelsAndNodeStaysUsable) {
  auto control = std::make_shared<FakeControl>();
  control->segment_ends = {{.x = 2.5, .y = 0.5}, {.x = 4.5, .y = 0.5}};
  control->block_local_call = 1U;
  auto states = std::make_shared<StateQueue>();
  auto log = std::make_shared<EventLog>();
  RunningSystem system("cancel_race", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
      .state_source = [states] { return states->Take(); },
      .event_sink = [log](const std::string& event) { log->Push(event); },
  });
  auto handle = system.Send(7.5, 0.5);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(WaitFor([&] { return log->Copy().size() >= 1U; }));
  states->Push({.position_m = {.x = 2.5, .y = 0.5}});
  ASSERT_TRUE(control->WaitUntilBlocked());

  auto result = system.Result(handle);
  system.Cancel(handle);
  ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
  const auto canceled = result.get();
  EXPECT_EQ(canceled.code, rclcpp_action::ResultCode::CANCELED);
  ASSERT_TRUE(canceled.result);
  EXPECT_EQ(canceled.result->outcome, Action::Result::CANCELED);
  ASSERT_TRUE(WaitFor([&] { return log->Copy().size() >= 3U; }));
  const auto canceled_events = log->Copy();
  EXPECT_EQ(canceled_events[0], "path:ACTIVE");
  EXPECT_EQ(canceled_events[1], "path:INVALIDATED");
  EXPECT_EQ(canceled_events[2], "terminal:CANCELED");

  auto recovery = system.Send(6.5, 0.5);
  ASSERT_TRUE(recovery);
  ASSERT_TRUE(WaitFor([&] {
    const auto events = log->Copy();
    return events.size() >= 4U && events.back() == "path:ACTIVE";
  }));
  auto recovery_result = system.Result(recovery);
  system.Cancel(recovery);
  ASSERT_EQ(recovery_result.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(recovery_result.get().code, rclcpp_action::ResultCode::CANCELED);
}

TEST(IncrementalNavigationNode,
     QueuedGoalCanceledBeforeProcessingNeverPreemptsTheActiveGoal) {
  auto control = std::make_shared<FakeControl>();
  auto gate = std::make_shared<CallbackGate>();
  RunningSystem system("queued_cancel", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
      .before_goal_processing = [gate] { gate->WaitIfArmed(); },
  });
  auto first = system.Send(6.5, 0.5);
  ASSERT_TRUE(first);
  ASSERT_TRUE(WaitFor([&] { return system.Paths().size() >= 1U; }));
  const auto first_uuid = first->get_goal_id();
  auto first_result = system.Result(first);

  gate->Arm();
  auto queued = system.Send(7.5, 0.5);
  ASSERT_TRUE(queued);
  ASSERT_TRUE(gate->WaitUntilEntered());
  auto queued_result = system.Result(queued);
  auto cancel_response = system.Cancel(queued);
  ASSERT_EQ(cancel_response.wait_for(3s), std::future_status::ready);
  gate->Release();

  ASSERT_EQ(queued_result.wait_for(3s), std::future_status::ready);
  const auto canceled = queued_result.get();
  EXPECT_EQ(canceled.code, rclcpp_action::ResultCode::CANCELED);
  ASSERT_TRUE(canceled.result);
  EXPECT_EQ(canceled.result->outcome, Action::Result::CANCELED);
  EXPECT_EQ(canceled.result->last_segment_revision, 0U);
  EXPECT_EQ(first_result.wait_for(100ms), std::future_status::timeout);
  for (const auto& path : system.Paths()) {
    if (path.state == path.ACTIVE) {
      EXPECT_EQ(path.session_id.uuid, first_uuid);
    }
  }

  system.Cancel(first);
  ASSERT_EQ(first_result.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(first_result.get().code, rclcpp_action::ResultCode::CANCELED);
}

TEST(IncrementalNavigationNode,
     CancelWinningTerminalCommitRaceProducesOneTerminalAndNodeRecovers) {
  auto control = std::make_shared<FakeControl>();
  control->reaches_final_goal = true;
  auto states = std::make_shared<StateQueue>();
  auto gate = std::make_shared<CallbackGate>();
  auto log = std::make_shared<EventLog>();
  RunningSystem system("terminal_cancel_race", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
      .state_source = [states] { return states->Take(); },
      .event_sink = [log](const std::string& event) { log->Push(event); },
      .before_terminal_commit = [gate] { gate->WaitIfArmed(); },
  });
  auto handle = system.Send(3.5, 0.5);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(WaitFor([&] { return log->Copy().size() >= 1U; }));
  auto result = system.Result(handle);

  gate->Arm();
  states->Push({.position_m = {.x = 3.5, .y = 0.5}});
  ASSERT_TRUE(gate->WaitUntilEntered());
  ASSERT_TRUE(WaitFor([&] {
    const auto events = log->Copy();
    return events.size() >= 2U && events[1] == "path:INVALIDATED";
  }));
  auto cancel_response = system.Cancel(handle);
  ASSERT_EQ(cancel_response.wait_for(3s), std::future_status::ready);
  gate->Release();

  ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
  const auto canceled = result.get();
  EXPECT_EQ(canceled.code, rclcpp_action::ResultCode::CANCELED);
  ASSERT_TRUE(canceled.result);
  EXPECT_EQ(canceled.result->outcome, Action::Result::CANCELED);
  ASSERT_TRUE(WaitFor([&] {
    const auto events = log->Copy();
    return std::count(events.begin(), events.end(), "terminal:CANCELED") == 1;
  }));
  const auto events = log->Copy();
  ASSERT_GE(events.size(), 3U);
  EXPECT_EQ(events[0], "path:ACTIVE");
  EXPECT_EQ(events[1], "path:INVALIDATED");
  EXPECT_EQ(events[2], "terminal:CANCELED");
  EXPECT_EQ(std::count_if(events.begin(), events.end(), [](const auto& event) {
              return event.starts_with("terminal:");
            }),
            1);
  ASSERT_TRUE(WaitFor([&] {
    const auto diagnostics = system.Diagnostics();
    return !diagnostics.empty() &&
           FindDiagnosticValue(diagnostics.back(), "reason_code") ==
               "CANCELED";
  }));
  EXPECT_EQ(FindDiagnosticValue(system.Diagnostics().back(), "reason_code"),
            "CANCELED");
  EXPECT_EQ(FindDiagnosticValue(system.Diagnostics().back(), "session_id"),
            GoalUuidHex(handle->get_goal_id()));
  EXPECT_FALSE(
      FindDiagnosticValue(system.Diagnostics().back(), "session_id").empty());

  auto recovery = system.Send(6.5, 0.5);
  ASSERT_TRUE(recovery);
  ASSERT_TRUE(WaitFor([&] {
    const auto paths = system.Paths();
    return !paths.empty() && paths.back().state == paths.back().ACTIVE &&
           paths.back().session_id.uuid == recovery->get_goal_id();
  }));
  auto recovery_result = system.Result(recovery);
  system.Cancel(recovery);
  ASSERT_EQ(recovery_result.wait_for(3s), std::future_status::ready);
  EXPECT_EQ(recovery_result.get().code, rclcpp_action::ResultCode::CANCELED);
}

TEST(IncrementalNavigationNode,
     FailedTerminalCommitRetriesWithoutRepublishingInvalidation) {
  auto control = std::make_shared<FakeControl>();
  control->reaches_final_goal = true;
  auto states = std::make_shared<StateQueue>();
  auto log = std::make_shared<EventLog>();
  auto commit_attempts = std::make_shared<std::atomic<std::size_t>>(0U);
  RunningSystem system("terminal_retry", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
      .state_source = [states] { return states->Take(); },
      .event_sink = [log](const std::string& event) { log->Push(event); },
      .before_terminal_commit = [commit_attempts] {
        if (commit_attempts->fetch_add(1U) == 0U) {
          throw std::runtime_error("injected terminal transition failure");
        }
      },
  });
  auto handle = system.Send(3.5, 0.5);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(WaitFor([&] { return log->Copy().size() >= 1U; }));
  auto result = system.Result(handle);

  states->Push({.position_m = {.x = 3.5, .y = 0.5}});
  ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
  const auto succeeded = result.get();
  EXPECT_EQ(succeeded.code, rclcpp_action::ResultCode::SUCCEEDED);
  ASSERT_TRUE(succeeded.result);
  EXPECT_EQ(succeeded.result->outcome, Action::Result::GOAL_REACHED);
  EXPECT_EQ(commit_attempts->load(), 2U);
  EXPECT_EQ(log->Copy(), (std::vector<std::string>{
                            "path:ACTIVE", "path:INVALIDATED",
                            "terminal:GOAL_REACHED"}));
}

TEST(IncrementalNavigationNode,
     FailedReplanDiagnosticsRetainTheAlreadyInvalidatedSegmentRevision) {
  auto control = std::make_shared<FakeControl>();
  control->segment_ends = {{.x = 3.5, .y = 0.5}};
  control->fail_local_after_calls = 1U;
  auto states = std::make_shared<StateQueue>();
  RunningSystem system("failed_replan", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{.fine = MakeFine()},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
      .state_source = [states] { return states->Take(); },
  });
  auto handle = system.Send(7.5, 0.5);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(WaitFor([&] { return system.Paths().size() >= 1U; }));
  auto result = system.Result(handle);

  states->Push({.position_m = {.x = 0.5, .y = 5.5}});
  ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
  const auto wrapped = result.get();
  ASSERT_TRUE(wrapped.result);
  EXPECT_EQ(wrapped.result->last_segment_revision, 1U);
  ASSERT_TRUE(WaitFor([&] {
    const auto diagnostics = system.Diagnostics();
    return !diagnostics.empty() &&
           FindDiagnosticValue(diagnostics.back(), "reason_code") ==
               "NO_PATH";
  }));
  const auto diagnostics = system.Diagnostics();
  EXPECT_EQ(FindDiagnosticValue(diagnostics.back(), "segment_revision"), "1");
  EXPECT_EQ(FindDiagnosticValue(diagnostics.back(), "path_state"),
            "INVALIDATED");
  EXPECT_EQ(FindDiagnosticValue(diagnostics.back(), "path_points"), "0");
  EXPECT_EQ(FindDiagnosticValue(diagnostics.back(), "session_id"),
            GoalUuidHex(handle->get_goal_id()));
  EXPECT_FALSE(FindDiagnosticValue(diagnostics.back(), "session_id").empty());
}

TEST(IncrementalNavigationNode,
     GlobalRouteIsRetainedDuringExecutionAndClearedAtTerminal) {
  auto control = std::make_shared<FakeControl>();
  control->publish_global_route = true;
  control->segment_ends = {{.x = 4.5, .y = 0.5}};
  auto states = std::make_shared<StateQueue>();
  const auto fine = MakeFine();
  RunningSystem system("global_lifecycle", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .snapshots = core::SnapshotBundle{
          .fine = fine, .guidance = MakeGuidance(fine)},
      .state = core::StateInput{.base_link_pose = {
          .position_m = {.x = 0.5, .y = 0.5}}},
      .state_source = [states] { return states->Take(); },
  });
  auto handle = system.Send(7.5, 0.5);
  ASSERT_TRUE(handle);
  ASSERT_TRUE(WaitFor([&] {
    const auto routes = system.Globals();
    return !routes.empty() && !routes.back().poses.empty();
  }));
  const std::size_t published_before_tick = system.Globals().size();

  states->Push({.position_m = {.x = 1.0, .y = 0.5}});
  std::this_thread::sleep_for(100ms);
  EXPECT_EQ(system.Globals().size(), published_before_tick);

  auto result = system.Result(handle);
  system.Cancel(handle);
  ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
  ASSERT_TRUE(WaitFor([&] {
    const auto routes = system.Globals();
    return routes.size() > published_before_tick && routes.back().poses.empty();
  }));
}

TEST(IncrementalNavigationNode, MissingFineSnapshotAbortsWithoutWaiting) {
  auto control = std::make_shared<FakeControl>();
  RunningSystem system("missing_map", IncrementalNavigationNodeDependencies{
      .ports_factory = FakePorts(control),
      .state = core::StateInput{.base_link_pose = {}},
  });
  auto handle = system.Send(2.0, 2.0);
  ASSERT_TRUE(handle);
  auto result = system.Result(handle);
  ASSERT_EQ(result.wait_for(3s), std::future_status::ready);
  const auto wrapped = result.get();
  EXPECT_EQ(wrapped.code, rclcpp_action::ResultCode::ABORTED);
  ASSERT_TRUE(wrapped.result);
  EXPECT_EQ(wrapped.result->outcome, Action::Result::MAP_UNAVAILABLE);
  EXPECT_EQ(wrapped.result->reason_code, "MAP_UNAVAILABLE");
}

}  // namespace
}  // namespace lunar::incremental_navigation_ros
