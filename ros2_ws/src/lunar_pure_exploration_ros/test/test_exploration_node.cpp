#include "lunar_pure_exploration_ros/exploration_node.hpp"
#include "lunar_pure_exploration_ros/planner_client.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_status.hpp>
#include <lunar_pure_exploration_msgs/msg/pure_exploration_task.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace lunar::pure_exploration_ros {

class ExplorationNodeTestPeer final {
 public:
  static FrozenPlanningCyclePtr ActiveCycle(const ExplorationNode& node) {
    return node.SnapshotActiveCycleForTest();
  }
  static std::optional<lunar_planning_msgs::msg::MotionReference>
  ActiveReference(const ExplorationNode& node) {
    return node.SnapshotActiveReferenceForTest();
  }
  static std::optional<std::string> ActiveRequestId(
      const ExplorationNode& node) {
    return node.SnapshotActiveRequestIdForTest();
  }
  static std::optional<lunar::pure_exploration::Pose2> ActiveTarget(
      const ExplorationNode& node) { return node.SnapshotActiveTargetForTest(); }
  static std::vector<lunar::pure_exploration::Vec2> ExecutablePolyline(
      const ExplorationNode& node) {
    return node.SnapshotExecutablePolylineForTest();
  }
  static std::optional<lunar::pure_exploration::Vec2> ExecutableEndpoint(
      const ExplorationNode& node) {
    return node.SnapshotExecutableEndpointForTest();
  }
  static double PlatformWidth(const ExplorationNode& node) {
    return node.PlatformWidthForTest();
  }
  static std::size_t CoarseCursor(const ExplorationNode& node) {
    return node.CoarseCursorForTest();
  }
  static std::optional<std::size_t> CandidateIndexForRequest(
      const ExplorationNode& node, const std::string& request_id) {
    return node.CandidateIndexForRequestForTest(request_id);
  }
  static std::optional<lunar::pure_exploration::Pose2> LatestPose(
      const ExplorationNode& node) {
    return node.LatestPoseForTest();
  }
  static std::optional<double> LatestMapResolution(
      const ExplorationNode& node) {
    return node.LatestMapResolutionForTest();
  }
  static void InjectEvaluation(ExplorationNode& node,
                               PlannerEvaluation evaluation) {
    node.InjectEvaluationForTest(std::move(evaluation));
  }
  static void ResetActiveCycle(ExplorationNode& node) {
    node.ResetActiveCycleForTest();
  }
};

namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using Task = lunar_pure_exploration_msgs::msg::PureExplorationTask;
using Status = lunar_pure_exploration_msgs::msg::PureExplorationStatus;
using ServerGoalHandle = rclcpp_action::ServerGoalHandle<Action>;
using namespace std::chrono_literals;

constexpr std::array<const char*, 7U> kStopWaitDiagnosticKeys{
    "stop_wait_elapsed_ms",
    "stop_entry_linear_mps",
    "stop_entry_angular_radps",
    "stop_confirmed_linear_mps",
    "stop_confirmed_angular_radps",
    "stop_confirmation_samples",
    "planning_epoch",
};

constexpr std::array<const char*, 26U> kPlannerOutputTimingKeys{
    "candidate_call_count",
    "candidate_record_count",
    "candidate_global_elapsed_ms",
    "candidate_global_call_count",
    "candidate_local_elapsed_ms",
    "candidate_local_call_count",
    "candidate_total_elapsed_ms",
    "rolling_call_count",
    "rolling_record_count",
    "rolling_global_elapsed_ms",
    "rolling_global_call_count",
    "rolling_local_elapsed_ms",
    "rolling_local_call_count",
    "rolling_total_elapsed_ms",
    "stuck_call_count",
    "stuck_record_count",
    "stuck_global_elapsed_ms",
    "stuck_global_call_count",
    "stuck_local_elapsed_ms",
    "stuck_local_call_count",
    "stuck_total_elapsed_ms",
    "global_elapsed_ms",
    "global_call_count",
    "local_elapsed_ms",
    "local_call_count",
    "total_elapsed_ms",
};

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
             const std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(2ms);
  }
  return predicate();
}

geometry_msgs::msg::Quaternion YawQuaternion(const double yaw,
                                             const double scale = 1.0) {
  geometry_msgs::msg::Quaternion quaternion;
  quaternion.z = scale * std::sin(yaw / 2.0);
  quaternion.w = scale * std::cos(yaw / 2.0);
  return quaternion;
}

nav_msgs::msg::OccupancyGrid GlobalMap(
    const std::uint32_t width = 20U, const std::uint32_t height = 20U,
    const double resolution = 0.5, const double origin_x = 0.0,
    const double origin_y = 0.0, const double origin_yaw = 0.0) {
  nav_msgs::msg::OccupancyGrid map;
  map.header.frame_id = "map";
  map.info.width = width;
  map.info.height = height;
  map.info.resolution = static_cast<float>(resolution);
  map.info.origin.position.x = origin_x;
  map.info.origin.position.y = origin_y;
  map.info.origin.orientation = YawQuaternion(origin_yaw, 7.0);
  map.data.assign(static_cast<std::size_t>(width) * height, 0);
  for (std::uint32_t y = 0U; y < height; ++y) {
    for (std::uint32_t x = width / 2U; x < width; ++x) {
      map.data[static_cast<std::size_t>(y) * width + x] = -1;
    }
  }
  return map;
}

nav_msgs::msg::Odometry Odometry(const double x = 2.0,
                                 const double y = 2.0,
                                 const double yaw = 0.0) {
  nav_msgs::msg::Odometry odometry;
  odometry.header.frame_id = "odom";
  odometry.child_frame_id = "base_link";
  odometry.pose.pose.position.x = x;
  odometry.pose.pose.position.y = y;
  odometry.pose.pose.orientation = YawQuaternion(yaw);
  return odometry;
}

tf2_msgs::msg::TFMessage MapFromOdom(const double x = 0.0,
                                    const double y = 0.0,
                                    const double yaw = 0.0) {
  tf2_msgs::msg::TFMessage transforms;
  geometry_msgs::msg::TransformStamped transform;
  transform.header.frame_id = "map";
  transform.child_frame_id = "odom";
  transform.transform.translation.x = x;
  transform.transform.translation.y = y;
  transform.transform.rotation = YawQuaternion(yaw);
  transforms.transforms.push_back(std::move(transform));
  return transforms;
}

Task StartTask(std::string task_id = "task-alpha") {
  Task task;
  task.header.frame_id = "map";
  task.task_id = std::move(task_id);
  task.command = Task::START;
  for (const std::pair<float, float> point : {
           std::pair{0.0F, 0.0F}, std::pair{10.0F, 0.0F},
           std::pair{10.0F, 10.0F}, std::pair{0.0F, 10.0F}}) {
    geometry_msgs::msg::Point32 vertex;
    vertex.x = point.first;
    vertex.y = point.second;
    task.boundary.points.push_back(vertex);
  }
  return task;
}

lunar::pure_exploration::FrontierCluster Frontier(
    std::vector<std::int64_t> canonical_key = {10, 20, 0},
    const std::uint64_t display_id = 9U) {
  return {.id = display_id,
          .cells = {{1, 1}},
          .interface_edges = {},
          .canonical_key = std::move(canonical_key),
          .centroid = {1.0, 1.0},
          .length_m = 1.0};
}

lunar::pure_exploration::CandidateView Candidate(
    const std::size_t frontier_index, const std::int64_t key_seed,
    const std::uint64_t display_id = 77U,
    const std::uint64_t frontier_display_id = 9U,
    std::vector<std::int64_t> canonical_key = {10, 20, 0}) {
  auto key = std::make_shared<const std::vector<std::int64_t>>(
      std::move(canonical_key));
  return {.id = display_id,
          .frontier_id = frontier_display_id,
          .frontier_index = frontier_index,
          .key = {.x_mm = key_seed, .y_mm = 0, .yaw_tenth_deg = 0},
          .pose = {.x = static_cast<double>(key_seed), .y = 0.0, .yaw = 0.0},
          .frontier_distance_m = 1.0,
          .frontier_canonical_key = std::move(key)};
}

FrozenGlobalMapContent FrozenMap(const nav_msgs::msg::OccupancyGrid& map) {
  return FrozenGlobalMapContent{
      .geometry = {.width = map.info.width,
                   .height = map.info.height,
                   .resolution = static_cast<double>(map.info.resolution),
                   .origin_x = map.info.origin.position.x,
                   .origin_y = map.info.origin.position.y,
                   .origin_yaw = 0.0},
      .data = map.data};
}

FrozenCandidateBatchPtr BatchWithCount(const std::size_t count) {
  std::vector<lunar::pure_exploration::FrontierCluster> frontiers{
      Frontier()};
  std::vector<lunar::pure_exploration::CandidateView> candidates;
  candidates.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    candidates.push_back(Candidate(0U, static_cast<std::int64_t>(index + 1U)));
  }
  return std::make_shared<FrozenCandidateBatch>(
      FrozenMap(GlobalMap(1U, 1U)), std::move(frontiers),
      std::move(candidates));
}

std::vector<lunar::pure_exploration::CandidateView> ControlledCandidates(
    std::span<const lunar::pure_exploration::FrontierCluster> frontiers,
    const std::size_t count) {
  if (frontiers.empty()) {
    throw std::logic_error{"controlled candidates require a frontier"};
  }
  std::vector<lunar::pure_exploration::CandidateView> candidates;
  candidates.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    candidates.push_back(Candidate(
        0U, static_cast<std::int64_t>(index + 1U),
        static_cast<std::uint64_t>(1000U + index), frontiers.front().id,
        frontiers.front().canonical_key));
    candidates.back().pose = {4.5, 2.0, 0.0};
  }
  return candidates;
}

std::vector<lunar::pure_exploration::CandidateGain> Gains(
    const std::size_t count) {
  std::vector<lunar::pure_exploration::CandidateGain> gains;
  gains.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    gains.push_back({index, static_cast<double>(count - index)});
  }
  return gains;
}

std::vector<lunar::pure_exploration::RankedCandidate> Order(
    const std::size_t count) {
  std::vector<lunar::pure_exploration::RankedCandidate> order;
  order.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    order.push_back({.candidate_index = index,
                     .information_gain_m2 = static_cast<double>(count - index),
                     .euclidean_distance_m = static_cast<double>(index),
                     .path_length_m = 0.0,
                     .heading_change_rad = 0.0,
                     .revisit_penalty = 0.0,
                     .rank_value = static_cast<double>(count - index)});
  }
  return order;
}

TEST(FrozenCandidateBatchTest, OwnsIdentityAndAssociatesOnlyByRequestId) {
  auto source_map = GlobalMap(2U, 2U, 0.5, 1.0, 2.0);
  source_map.header.stamp.sec = 100;
  auto batch = std::make_shared<FrozenCandidateBatch>(
      FrozenMap(source_map), std::vector{Frontier()},
      std::vector{Candidate(0U, 1, 55U), Candidate(0U, 2, 55U)});

  source_map.header.stamp.sec = 0;
  EXPECT_TRUE(batch->GlobalMapContentEquals(source_map));
  source_map.data[0] = 100;
  EXPECT_FALSE(batch->GlobalMapContentEquals(source_map));
  source_map.data[0] = 0;
  source_map.info.origin.position.x = 3.0;
  EXPECT_FALSE(batch->GlobalMapContentEquals(source_map));

  batch->RegisterRequest("request-for-index-1", 1U);
  batch->RegisterRequest("request-for-index-0", 0U);
  EXPECT_EQ(batch->CandidateIndexForRequest("request-for-index-1"), 1U);
  EXPECT_EQ(batch->CandidateIndexForRequest("request-for-index-0"), 0U);
  EXPECT_EQ(batch->LatestRequestIdForCandidate(0U),
            "request-for-index-0");
  EXPECT_EQ(batch->LatestRequestIdForCandidate(1U),
            "request-for-index-1");
  EXPECT_FALSE(batch->CandidateIndexForRequest("55").has_value());
  EXPECT_THROW(batch->RegisterRequest("", 0U), std::invalid_argument);
  EXPECT_THROW(batch->RegisterRequest("request-for-index-0", 1U),
               std::invalid_argument);
  EXPECT_THROW(batch->RegisterRequest("out-of-range", 2U),
               std::out_of_range);
}

TEST(FrozenCandidateBatchTest, RejectsEveryCandidateFrontierIdentityMismatch) {
  const auto map = FrozenMap(GlobalMap(1U, 1U));
  const std::vector frontiers{Frontier({10, 20, 0}, 9U)};

  EXPECT_THROW(FrozenCandidateBatch(map, frontiers,
                                    {Candidate(1U, 1)}),
               std::invalid_argument);
  EXPECT_THROW(FrozenCandidateBatch(map, frontiers,
                                    {Candidate(0U, 1, 77U, 8U)}),
               std::invalid_argument);
  EXPECT_THROW(FrozenCandidateBatch(
                   map, frontiers,
                   {Candidate(0U, 1, 77U, 9U, {10, 21, 0})}),
               std::invalid_argument);
  auto empty_key = Candidate(0U, 1);
  empty_key.frontier_canonical_key =
      std::make_shared<const std::vector<std::int64_t>>();
  EXPECT_THROW(FrozenCandidateBatch(map, frontiers, {empty_key}),
               std::invalid_argument);
}

TEST(FrozenPlanningCycleTest, ValidatesAuthorityAndChunksThirtyThreeExactly) {
  auto cycle = std::make_shared<FrozenPlanningCycle>(
      BatchWithCount(33U), lunar::pure_exploration::Pose2{1.0, 2.0, 0.5},
      0.2, Gains(33U), Order(33U));
  EXPECT_EQ(cycle->coarse_cursor(), 0U);
  EXPECT_EQ(cycle->TakeNextCandidateIndices().size(), 16U);
  EXPECT_EQ(cycle->coarse_cursor(), 16U);
  EXPECT_EQ(cycle->TakeNextCandidateIndices().size(), 16U);
  EXPECT_EQ(cycle->coarse_cursor(), 32U);
  const auto tail = cycle->TakeNextCandidateIndices();
  ASSERT_EQ(tail.size(), 1U);
  EXPECT_EQ(tail.front(), 32U);
  EXPECT_TRUE(cycle->TakeNextCandidateIndices().empty());

  EXPECT_THROW(FrozenPlanningCycle(nullptr, {}, 0.2, {}, {}),
               std::invalid_argument);
  EXPECT_THROW(FrozenPlanningCycle(
                   BatchWithCount(1U),
                   {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0},
                   0.2, Gains(1U), Order(1U)),
               std::invalid_argument);
  EXPECT_THROW(FrozenPlanningCycle(BatchWithCount(1U), {}, 0.0, Gains(1U),
                                   Order(1U)),
               std::invalid_argument);

  auto duplicate_gains = Gains(2U);
  duplicate_gains[1].candidate_index = 0U;
  EXPECT_THROW(FrozenPlanningCycle(BatchWithCount(2U), {}, 0.2,
                                   duplicate_gains, Order(2U)),
               std::invalid_argument);
  auto missing_order = Order(2U);
  missing_order.pop_back();
  EXPECT_THROW(FrozenPlanningCycle(BatchWithCount(2U), {}, 0.2, Gains(2U),
                                   missing_order),
               std::invalid_argument);
  auto mismatched_order = Order(2U);
  mismatched_order[1].information_gain_m2 = 99.0;
  EXPECT_THROW(FrozenPlanningCycle(BatchWithCount(2U), {}, 0.2, Gains(2U),
                                   mismatched_order),
               std::invalid_argument);
}

TEST(FrozenPlanningCycleTest, OwnsUniqueReachableRowsAndReferences) {
  auto batch = BatchWithCount(2U);
  FrozenPlanningCycle cycle(batch, {1.0, 2.0, 0.5}, 0.2, Gains(2U),
                            Order(2U));
  lunar_planning_msgs::msg::MotionReference reference;
  reference.plan_id = "plan-owned";
  cycle.AddReachable({.metrics = {1U, 3.5}, .reference = reference});
  ASSERT_EQ(cycle.reachable().size(), 1U);
  EXPECT_EQ(cycle.reachable().front().reference.plan_id, "plan-owned");
  EXPECT_THROW(
      cycle.AddReachable({.metrics = {1U, 4.0}, .reference = reference}),
      std::invalid_argument);
  EXPECT_THROW(
      cycle.AddReachable({.metrics = {2U, 4.0}, .reference = reference}),
      std::out_of_range);
  EXPECT_THROW(cycle.AddReachable(
                   {.metrics = {0U, std::numeric_limits<double>::infinity()},
                    .reference = reference}),
               std::invalid_argument);
}

class FakePlannerServer final {
 public:
  enum class Mode { kDelayed, kReachable, kExhaustive, kRetryable,
                    kSizedExecutableReference, kSizedPreviewReference,
                    kRollingReachable, kContractError, kResourceError };

  FakePlannerServer(std::shared_ptr<rclcpp::Node> node,
                    const std::string& action_name, const Mode mode,
                    const std::size_t executable_points = 1U,
                    const std::size_t preview_poses = 1U)
      : node_(std::move(node)), mode_(mode),
        executable_points_(executable_points),
        preview_poses_(preview_poses) {
    server_ = rclcpp_action::create_server<Action>(
        node_, action_name,
        [this](const rclcpp_action::GoalUUID&,
               const std::shared_ptr<const Action::Goal> goal) {
          std::scoped_lock lock{mutex_};
          goals_.push_back(*goal);
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [this](const std::shared_ptr<ServerGoalHandle> handle) {
          std::scoped_lock lock{mutex_};
          ++cancel_count_;
          canceled_request_ids_.insert(handle->get_goal()->request_id);
          return rclcpp_action::CancelResponse::ACCEPT;
        },
        [this](const std::shared_ptr<ServerGoalHandle> handle) {
          Mode mode;
          {
            std::scoped_lock lock{mutex_};
            handles_.insert_or_assign(handle->get_goal()->request_id, handle);
            ++outstanding_;
            maximum_outstanding_ = std::max(maximum_outstanding_, outstanding_);
            mode = mode_;
          }
          if (mode != Mode::kDelayed) {
            Finish(handle->get_goal()->request_id, mode);
          }
        });
  }

  std::vector<Action::Goal> Goals() const {
    std::scoped_lock lock{mutex_};
    return goals_;
  }

  std::size_t MaximumOutstanding() const {
    std::scoped_lock lock{mutex_};
    return maximum_outstanding_;
  }

  std::size_t CancelCount() const {
    std::scoped_lock lock{mutex_};
    return cancel_count_;
  }
  void SetMode(const Mode mode) {
    std::scoped_lock lock{mutex_};
    mode_ = mode;
  }

  void SetFixedPlanId(std::string plan_id) {
    std::scoped_lock lock{mutex_};
    fixed_plan_id_ = std::move(plan_id);
  }

  bool HasHandle() const {
    std::scoped_lock lock{mutex_};
    return !handles_.empty();
  }

  bool HasHandle(const std::string& request_id) const {
    std::scoped_lock lock{mutex_};
    return handles_.contains(request_id);
  }

  std::string LatestRequestId() const {
    std::scoped_lock lock{mutex_};
    return goals_.empty() ? std::string{} : goals_.back().request_id;
  }

  void Finish(const std::string& request_id, const Mode mode) {
    std::shared_ptr<ServerGoalHandle> handle;
    {
      std::scoped_lock lock{mutex_};
      handle = handles_.at(request_id);
      if (outstanding_ > 0U) {
        --outstanding_;
      }
    }
    auto result = std::make_shared<Action::Result>();
    if (mode == Mode::kReachable ||
        mode == Mode::kSizedExecutableReference ||
        mode == Mode::kSizedPreviewReference || mode == Mode::kRollingReachable ||
        mode == Mode::kContractError || mode == Mode::kResourceError) {
      result->planning_outcome = Action::Result::NEW_REFERENCE_AVAILABLE;
      result->execution_directive = mode == Mode::kContractError
                                        ? Action::Result::NO_SAFE_REFERENCE
                                        : Action::Result::ACTIVATE_NEW_REFERENCE;
      result->reason_code = "PLAN_FOUND";
      result->has_reference = true;
      result->reference.header.frame_id = "map";
      result->reference.header.stamp.sec = 31;
      {
        std::scoped_lock lock{mutex_};
        result->reference.plan_id = fixed_plan_id_.value_or(
            "plan:" + request_id);
      }
      result->reference.platform_type = result->reference.WHEELED;
      result->reference.input_time.sec = 29;
      result->reference.path_preview.header.frame_id = "map";
      const std::size_t preview_count =
          mode == Mode::kSizedPreviewReference ? preview_poses_ : 1U;
      for (std::size_t index = 0U; index < preview_count; ++index) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp.sec = 23;
        pose.pose.position.x = handle->get_goal()->goal.point.x;
        pose.pose.position.y = handle->get_goal()->goal.point.y;
        pose.pose.orientation.w = 1.0;
        result->reference.path_preview.poses.push_back(std::move(pose));
      }
      const std::size_t count = mode == Mode::kResourceError ? 65U : mode == Mode::kSizedExecutableReference
                                    ? executable_points_
                                    : 1U;
      result->reference.trajectory.points.resize(count);
      for (auto& point : result->reference.trajectory.points) {
        geometry_msgs::msg::Transform transform;
        transform.translation.x = mode == Mode::kRollingReachable
                                      ? handle->get_goal()->goal.point.x - 1.0
                                      : handle->get_goal()->goal.point.x;
        transform.translation.y = handle->get_goal()->goal.point.y;
        transform.rotation.w = 1.0;
        point.transforms.push_back(std::move(transform));
      }
      handle->succeed(result);
    } else if (mode == Mode::kExhaustive) {
      result->planning_outcome = Action::Result::GOAL_INFEASIBLE;
      result->execution_directive = Action::Result::NO_SAFE_REFERENCE;
      result->reason_code = "NO_PATH";
      result->has_reference = false;
      handle->abort(result);
    } else if (mode == Mode::kRetryable) {
      result->planning_outcome = Action::Result::RESOURCE_EXHAUSTED;
      result->execution_directive = Action::Result::NO_SAFE_REFERENCE;
      result->reason_code = "TIMEOUT";
      result->has_reference = false;
      handle->abort(result);
    } else {
      throw std::logic_error{"delayed result requires explicit terminal mode"};
    }
  }

  void FinishCanceled(const std::string& request_id) {
    std::shared_ptr<ServerGoalHandle> handle;
    {
      std::scoped_lock lock{mutex_};
      handle = handles_.at(request_id);
      if (outstanding_ > 0U) {
        --outstanding_;
      }
    }
    auto result = std::make_shared<Action::Result>();
    result->planning_outcome = Action::Result::CANCELED;
    result->execution_directive = Action::Result::NO_SAFE_REFERENCE;
    result->reason_code = "REQUEST_CANCELED";
    result->has_reference = false;
    handle->canceled(result);
  }

 private:
  std::shared_ptr<rclcpp::Node> node_;
  Mode mode_;
  std::size_t executable_points_;
  std::size_t preview_poses_;
  rclcpp_action::Server<Action>::SharedPtr server_;
  mutable std::mutex mutex_;
  std::vector<Action::Goal> goals_;
  std::unordered_map<std::string, std::shared_ptr<ServerGoalHandle>> handles_;
  std::set<std::string> canceled_request_ids_;
  std::optional<std::string> fixed_plan_id_;
  std::size_t outstanding_{0U};
  std::size_t maximum_outstanding_{0U};
  std::size_t cancel_count_{0U};
};

ExplorationNodeParameters TestParameters(const std::string& prefix) {
  return ExplorationNodeParameters{
      .platform = {.platform_id = "compact-wheel",
                   .platform_type = "WHEELED",
                   .base_frame_id = "base_footprint",
                   .footprint_vertices = {{0.1, 0.1}, {0.1, -0.1},
                                          {-0.1, -0.1}, {-0.1, 0.1}},
                   .minimum_clearance_m = 0.0},
      .candidate_parameters = {{-std::numbers::pi / 4.0,
                                -std::numbers::pi / 8.0, 0.0,
                                std::numbers::pi / 8.0,
                                std::numbers::pi / 4.0}},
      .candidate_limits = {4096U, 64U, 100000U},
      .task_raster_limits = {1048576U},
      .sensor_model = {10.0, std::numbers::pi / 2.0},
      .information_gain_limits = {100000U},
      .score_weights = {},
      .failure_memory_limits = {8U, 256U, 1024U},
      .global_occupied_threshold = 50,
      .maximum_path_preview_poses = 64U,
      .maximum_executable_path_points = 64U,
      .maximum_replans = 2U,
      .goal_yaw_tolerance_rad = std::numbers::pi / 16.0,
      .filter_global_goal_cell = false,
      .stationary_gate =
          {.maximum_linear_speed_mps = 0.01,
           .maximum_angular_speed_radps = 0.02,
           .confirmation_samples = 3U,
           .diagnostic_period = 5s},
      .planner_result_timeout = 2s,
      .global_map_topic = prefix + "/global_map",
      .odometry_topic = prefix + "/odometry",
      .tf_topic = prefix + "/tf",
      .task_topic = prefix + "/task",
      .planner_action = prefix + "/plan_motion",
      .planner_diagnostics_topic = prefix + "/planner_diagnostics",
      .motion_reference_topic = prefix + "/motion_reference",
      .execution_cancel_topic = prefix + "/execution_cancel",
      .status_topic = prefix + "/status",
      .current_goal_topic = prefix + "/current_goal",
      .frontiers_topic = prefix + "/frontiers",
      .diagnostics_topic = prefix + "/diagnostics"};
}

class ExplorationNodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto suffix = sequence_.fetch_add(1U);
    prefix_ = "/exploration_node_test_" + std::to_string(suffix);
    parameters_ = TestParameters(prefix_);
    server_node_ = std::make_shared<rclcpp::Node>(
        "exploration_fake_server_" + std::to_string(suffix));
    io_node_ = std::make_shared<rclcpp::Node>(
        "exploration_test_io_" + std::to_string(suffix));
    task_publisher_ = io_node_->create_publisher<Task>(parameters_.task_topic, 10);
    map_publisher_ = io_node_->create_publisher<nav_msgs::msg::OccupancyGrid>(
        parameters_.global_map_topic, 10);
    odometry_publisher_ = io_node_->create_publisher<nav_msgs::msg::Odometry>(
        parameters_.odometry_topic, 10);
    tf_publisher_ = io_node_->create_publisher<tf2_msgs::msg::TFMessage>(
        parameters_.tf_topic, 10);
    planner_diagnostics_publisher_ =
        io_node_->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
            parameters_.planner_diagnostics_topic, 10);
    reference_subscription_ =
        io_node_->create_subscription<lunar_planning_msgs::msg::MotionReference>(
            parameters_.motion_reference_topic, 10,
            [this](lunar_planning_msgs::msg::MotionReference::SharedPtr value) {
              std::scoped_lock lock{messages_mutex_};
              references_.push_back(*value);
            });
    cancel_subscription_ = io_node_->create_subscription<std_msgs::msg::String>(
        parameters_.execution_cancel_topic, 10,
        [this](std_msgs::msg::String::SharedPtr value) {
          std::scoped_lock lock{messages_mutex_};
          execution_cancels_.push_back(value->data);
        });
    status_subscription_ = io_node_->create_subscription<Status>(
        parameters_.status_topic,
        rclcpp::QoS{10}.reliable().transient_local(),
        [this](Status::SharedPtr value) {
          std::scoped_lock lock{messages_mutex_};
          statuses_.push_back(*value);
        });
    current_goal_subscription_ =
        io_node_->create_subscription<geometry_msgs::msg::PoseStamped>(
            parameters_.current_goal_topic,
            rclcpp::QoS{1}.reliable().transient_local(),
            [this](geometry_msgs::msg::PoseStamped::SharedPtr value) {
              std::scoped_lock lock{messages_mutex_};
              current_goals_.push_back(*value);
            });
    frontiers_subscription_ =
        io_node_->create_subscription<visualization_msgs::msg::MarkerArray>(
            parameters_.frontiers_topic,
            rclcpp::QoS{1}.reliable().transient_local(),
            [this](visualization_msgs::msg::MarkerArray::SharedPtr value) {
              std::scoped_lock lock{messages_mutex_};
              frontier_markers_.push_back(*value);
            });
    diagnostics_subscription_ =
        io_node_->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
            parameters_.diagnostics_topic,
            rclcpp::QoS{1}.reliable().transient_local(),
            [this](diagnostic_msgs::msg::DiagnosticArray::SharedPtr value) {
              std::scoped_lock lock{messages_mutex_};
              diagnostics_.push_back(*value);
            });
  }

  void Start(FakePlannerServer::Mode mode,
             std::size_t executable_points = 1U,
             std::size_t preview_poses = 1U) {
    server_ = std::make_unique<FakePlannerServer>(
        server_node_, parameters_.planner_action, mode, executable_points,
        preview_poses);
    explorer_ = std::make_shared<ExplorationNode>(parameters_);
    executor_ = std::make_unique<rclcpp::executors::MultiThreadedExecutor>(
        rclcpp::ExecutorOptions{}, 4U);
    executor_->add_node(server_node_);
    executor_->add_node(io_node_);
    executor_->add_node(explorer_);
    spin_thread_ = std::jthread([this] { executor_->spin(); });
    auto probe = rclcpp_action::create_client<Action>(
        io_node_, parameters_.planner_action);
    ASSERT_TRUE(probe->wait_for_action_server(3s));
    ASSERT_TRUE(WaitFor([this] {
      return task_publisher_->get_subscription_count() == 1U &&
             map_publisher_->get_subscription_count() == 1U &&
             odometry_publisher_->get_subscription_count() == 1U &&
             tf_publisher_->get_subscription_count() == 1U;
    }));
  }

  rclcpp::NodeOptions StopGateParameterOptions(
      const double linear_threshold = 0.01,
      const double angular_threshold = 0.02,
      const std::int64_t confirmation_samples = 3,
      const double diagnostic_period_s = 5.0) const {
    const auto platform_config =
        (std::filesystem::path{__FILE__}.parent_path() /
         "../../../../config/wheel.yaml")
            .lexically_normal()
            .string();
    rclcpp::NodeOptions options;
    options.parameter_overrides({
        rclcpp::Parameter("platform_selector", "wheel"),
        rclcpp::Parameter("platform_config", platform_config),
        rclcpp::Parameter("maximum_position_probes", 4096),
        rclcpp::Parameter("maximum_candidate_views", 64),
        rclcpp::Parameter("maximum_collision_work_units", 100000),
        rclcpp::Parameter("maximum_visibility_work_units", 100000),
        rclcpp::Parameter("maximum_failure_entries", 8),
        rclcpp::Parameter("maximum_failure_patch_cells_per_entry", 256),
        rclcpp::Parameter("maximum_failure_total_patch_cells", 1024),
        rclcpp::Parameter("maximum_path_preview_poses", 64),
        rclcpp::Parameter("maximum_executable_path_points", 64),
        rclcpp::Parameter("stop_before_planning", true),
        rclcpp::Parameter("stationary_linear_speed_mps", linear_threshold),
        rclcpp::Parameter("stationary_angular_speed_radps", angular_threshold),
        rclcpp::Parameter("stationary_confirmation_samples",
                          confirmation_samples),
        rclcpp::Parameter("stop_wait_diagnostic_s", diagnostic_period_s),
        rclcpp::Parameter("global_map_topic", parameters_.global_map_topic),
        rclcpp::Parameter("odometry_topic", parameters_.odometry_topic),
        rclcpp::Parameter("tf_topic", parameters_.tf_topic),
        rclcpp::Parameter("task_topic", parameters_.task_topic),
        rclcpp::Parameter("planner_action", parameters_.planner_action),
        rclcpp::Parameter("planner_diagnostics_topic",
                          parameters_.planner_diagnostics_topic),
        rclcpp::Parameter("motion_reference_topic",
                          parameters_.motion_reference_topic),
        rclcpp::Parameter("execution_cancel_topic",
                          parameters_.execution_cancel_topic),
    });
    return options;
  }

  void StartWithStopGateParameters(FakePlannerServer::Mode mode) {
    server_ = std::make_unique<FakePlannerServer>(
        server_node_, parameters_.planner_action, mode);
    explorer_ =
        std::make_shared<ExplorationNode>(StopGateParameterOptions());
    executor_ = std::make_unique<rclcpp::executors::MultiThreadedExecutor>(
        rclcpp::ExecutorOptions{}, 4U);
    executor_->add_node(server_node_);
    executor_->add_node(io_node_);
    executor_->add_node(explorer_);
    spin_thread_ = std::jthread([this] { executor_->spin(); });
    auto probe = rclcpp_action::create_client<Action>(
        io_node_, parameters_.planner_action);
    ASSERT_TRUE(probe->wait_for_action_server(3s));
    ASSERT_TRUE(WaitFor([this] {
      return task_publisher_->get_subscription_count() == 1U &&
             map_publisher_->get_subscription_count() == 1U &&
             odometry_publisher_->get_subscription_count() == 1U &&
             tf_publisher_->get_subscription_count() == 1U;
    }));
  }

  void TearDown() override {
    if (executor_) {
      executor_->cancel();
    }
    if (spin_thread_.joinable()) {
      spin_thread_.join();
    }
    if (executor_) {
      if (explorer_) {
        executor_->remove_node(explorer_);
      }
      executor_->remove_node(io_node_);
      executor_->remove_node(server_node_);
    }
    explorer_.reset();
    server_.reset();
    io_node_.reset();
    server_node_.reset();
  }

  void PublishAllInputs(Task task = StartTask(),
                        nav_msgs::msg::OccupancyGrid map = GlobalMap(),
                        nav_msgs::msg::Odometry odometry = Odometry(),
                        tf2_msgs::msg::TFMessage transforms = MapFromOdom()) {
    map_publisher_->publish(std::move(map));
    odometry_publisher_->publish(std::move(odometry));
    tf_publisher_->publish(std::move(transforms));
    task_publisher_->publish(std::move(task));
  }

  void PublishPlannerTiming(const std::string& request_id) {
    diagnostic_msgs::msg::DiagnosticStatus status;
    const auto add = [&status](std::string key, std::string value) {
      status.values.push_back(
          diagnostic_msgs::msg::KeyValue{}.set__key(std::move(key)).set__value(
              std::move(value)));
    };
    add("request_id", request_id);
    add("platform_type", "WHEELED");
    add("environment_mode", "1");
    add("planning_outcome", "0");
    add("reason_code", "OK");
    add("global_elapsed_ms", "11.0");
    add("global_call_count", "2");
    add("local_elapsed_ms", "7.0");
    add("local_call_count", "3");
    add("total_elapsed_ms", "18.0");
    diagnostic_msgs::msg::DiagnosticArray message;
    message.status.push_back(std::move(status));
    planner_diagnostics_publisher_->publish(std::move(message));
  }

  std::optional<Status> LatestStatus() const {
    std::scoped_lock lock{messages_mutex_};
    if (statuses_.empty()) {
      return std::nullopt;
    }
    return statuses_.back();
  }

  std::size_t ReferenceCount() const {
    std::scoped_lock lock{messages_mutex_};
    return references_.size();
  }

  std::vector<lunar_planning_msgs::msg::MotionReference> References() const {
    std::scoped_lock lock{messages_mutex_};
    return references_;
  }

  std::vector<std::string> ExecutionCancels() const {
    std::scoped_lock lock{messages_mutex_};
    return execution_cancels_;
  }

  std::size_t StatusCount() const {
    std::scoped_lock lock{messages_mutex_};
    return statuses_.size();
  }

  std::vector<Status> Statuses() const {
    std::scoped_lock lock{messages_mutex_};
    return statuses_;
  }

  std::size_t CurrentGoalCount() const {
    std::scoped_lock lock{messages_mutex_};
    return current_goals_.size();
  }

  std::optional<geometry_msgs::msg::PoseStamped> LatestCurrentGoal() const {
    std::scoped_lock lock{messages_mutex_};
    return current_goals_.empty() ? std::nullopt
                                  : std::optional{current_goals_.back()};
  }

  std::optional<visualization_msgs::msg::MarkerArray> LatestMarkers() const {
    std::scoped_lock lock{messages_mutex_};
    return frontier_markers_.empty() ? std::nullopt
                                     : std::optional{frontier_markers_.back()};
  }

  std::optional<diagnostic_msgs::msg::DiagnosticArray> LatestDiagnostics() const {
    std::scoped_lock lock{messages_mutex_};
    return diagnostics_.empty() ? std::nullopt : std::optional{diagnostics_.back()};
  }

  std::size_t DiagnosticsCount() const {
    std::scoped_lock lock{messages_mutex_};
    return diagnostics_.size();
  }

  std::optional<std::string> DiagnosticValue(const std::string& key) const {
    const auto diagnostics = LatestDiagnostics();
    if (!diagnostics || diagnostics->status.size() != 1U) {
      return std::nullopt;
    }
    const auto found = std::ranges::find_if(
        diagnostics->status.front().values, [&key](const auto& value) {
          return value.key == key;
        });
    return found == diagnostics->status.front().values.end()
               ? std::nullopt
               : std::optional<std::string>{found->value};
  }

  std::vector<std::string> PlannerTimingDiagnosticBytes() const {
    const auto diagnostics = LatestDiagnostics();
    if (!diagnostics || diagnostics->status.size() != 1U) {
      return {};
    }
    std::vector<std::string> result;
    result.reserve(kPlannerOutputTimingKeys.size());
    for (const char* key : kPlannerOutputTimingKeys) {
      const auto found = std::ranges::find_if(
          diagnostics->status.front().values,
          [key](const auto& value) { return value.key == key; });
      if (found == diagnostics->status.front().values.end()) {
        return {};
      }
      std::string bytes{found->key};
      bytes.push_back('\0');
      bytes += found->value;
      result.push_back(std::move(bytes));
    }
    return result;
  }

  void ExpectWaitingWithMissingInput(const int missing) {
    Start(FakePlannerServer::Mode::kDelayed);
    task_publisher_->publish(StartTask("missing-" + std::to_string(missing)));
    if (missing != 0) {
      map_publisher_->publish(GlobalMap());
    }
    if (missing != 1) {
      odometry_publisher_->publish(Odometry());
    }
    if (missing != 2) {
      tf_publisher_->publish(MapFromOdom());
    }
    ASSERT_TRUE(WaitFor([this] {
      const auto status = LatestStatus();
      return status.has_value() && status->state == Status::WAITING_FOR_INPUT;
    }));
    std::this_thread::sleep_for(50ms);
    const auto status = LatestStatus();
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(status->state, Status::WAITING_FOR_INPUT);
    EXPECT_EQ(status->reason_code, "WAITING_FOR_INPUT");
    EXPECT_TRUE(server_->Goals().empty());
    EXPECT_EQ(ReferenceCount(), 0U);
  }

  static std::atomic<std::uint64_t> sequence_;
  std::string prefix_;
  ExplorationNodeParameters parameters_;
  std::shared_ptr<rclcpp::Node> server_node_;
  std::shared_ptr<rclcpp::Node> io_node_;
  std::shared_ptr<ExplorationNode> explorer_;
  std::unique_ptr<FakePlannerServer> server_;
  std::unique_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
  std::jthread spin_thread_;
  rclcpp::Publisher<Task>::SharedPtr task_publisher_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      planner_diagnostics_publisher_;
  rclcpp::Subscription<lunar_planning_msgs::msg::MotionReference>::SharedPtr
      reference_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr cancel_subscription_;
  rclcpp::Subscription<Status>::SharedPtr status_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      current_goal_subscription_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr
      frontiers_subscription_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
      diagnostics_subscription_;
  mutable std::mutex messages_mutex_;
  std::vector<Status> statuses_;
  std::vector<lunar_planning_msgs::msg::MotionReference> references_;
  std::vector<std::string> execution_cancels_;
  std::vector<geometry_msgs::msg::PoseStamped> current_goals_;
  std::vector<visualization_msgs::msg::MarkerArray> frontier_markers_;
  std::vector<diagnostic_msgs::msg::DiagnosticArray> diagnostics_;
};

std::atomic<std::uint64_t> ExplorationNodeTest::sequence_{0U};

enum class ExecutionMapUpdate {
  kStampOnly,
  kOdometryCovarianceOnly,
  kGeometry,
  kData,
};

class ExecutionReplanMapGateTest
    : public ExplorationNodeTest,
      public ::testing::WithParamInterface<ExecutionMapUpdate> {};

enum class ExecutionControl {
  kPause,
  kCancel,
  kReplacementStart,
};

class ExecutionReplanControlGateTest
    : public ExplorationNodeTest,
      public ::testing::WithParamInterface<ExecutionControl> {};

TEST_F(ExplorationNodeTest, MissingGlobalMapWaitsIndefinitely) {
  ExpectWaitingWithMissingInput(0);
}

TEST_F(ExplorationNodeTest, MissingOdometryWaitsIndefinitely) {
  ExpectWaitingWithMissingInput(1);
}

TEST_F(ExplorationNodeTest, MissingMapToOdomTransformWaitsIndefinitely) {
  ExpectWaitingWithMissingInput(2);
}

TEST_F(ExplorationNodeTest,
       StopGateWaitsForThreeConsecutiveStationaryOdometrySamples) {
  parameters_.stop_before_planning = true;
  Start(FakePlannerServer::Mode::kDelayed);
  auto moving = Odometry();
  moving.twist.twist.linear.x = 0.2;
  PublishAllInputs(StartTask("stationary-three"), GlobalMap(), moving);
  ASSERT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    const auto status = LatestStatus();
    return pose && pose->x == 2.0 && status &&
           status->reason_code == "WAITING_FOR_STOP";
  }));
  EXPECT_EQ(server_->Goals().size(), 0U);

  odometry_publisher_->publish(Odometry(2.1, 2.0));
  ASSERT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == 2.1;
  }));
  EXPECT_EQ(server_->Goals().size(), 0U);
  odometry_publisher_->publish(Odometry(2.2, 2.0));
  ASSERT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == 2.2;
  }));
  EXPECT_EQ(server_->Goals().size(), 0U);

  odometry_publisher_->publish(Odometry(2.3, 2.0));
  ASSERT_TRUE(WaitFor([this] { return server_->Goals().size() == 1U; }));
  odometry_publisher_->publish(Odometry(2.4, 2.0));
  ASSERT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == 2.4;
  }));
  EXPECT_EQ(server_->Goals().size(), 1U);
}

TEST_F(ExplorationNodeTest,
       MovingOdometryResetsConsecutiveStationaryAdmissionSamples) {
  parameters_.stop_before_planning = true;
  Start(FakePlannerServer::Mode::kDelayed);
  auto moving = Odometry();
  moving.twist.twist.linear.x = 0.2;
  PublishAllInputs(StartTask("stationary-reset"), GlobalMap(), moving);
  ASSERT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    const auto status = LatestStatus();
    return pose && pose->x == 2.0 && status &&
           status->reason_code == "WAITING_FOR_STOP";
  }));
  EXPECT_EQ(server_->Goals().size(), 0U);

  odometry_publisher_->publish(Odometry(2.1, 2.0));
  ASSERT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == 2.1;
  }));
  odometry_publisher_->publish(Odometry(2.2, 2.0));
  ASSERT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == 2.2;
  }));
  auto reset = Odometry(2.3, 2.0);
  reset.twist.twist.angular.z = 0.2;
  odometry_publisher_->publish(reset);
  ASSERT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == 2.3;
  }));

  odometry_publisher_->publish(Odometry(2.4, 2.0));
  ASSERT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == 2.4;
  }));
  odometry_publisher_->publish(Odometry(2.5, 2.0));
  ASSERT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == 2.5;
  }));
  EXPECT_EQ(server_->Goals().size(), 0U);

  odometry_publisher_->publish(Odometry(2.6, 2.0));
  ASSERT_TRUE(WaitFor([this] { return server_->Goals().size() == 1U; }));
  odometry_publisher_->publish(Odometry(2.7, 2.0));
  ASSERT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == 2.7;
  }));
  EXPECT_EQ(server_->Goals().size(), 1U);
}

TEST_F(ExplorationNodeTest,
       PublishesExactStopWaitDiagnosticsFromInjectedSteadyClock) {
  auto now_ns = std::make_shared<std::atomic<std::int64_t>>(0);
  parameters_.stop_before_planning = true;
  parameters_.steady_now = [now_ns] {
    return std::chrono::steady_clock::time_point{
        std::chrono::nanoseconds{now_ns->load()}};
  };
  Start(FakePlannerServer::Mode::kDelayed);

  auto moving = Odometry();
  moving.twist.twist.linear.x = 0.2;
  moving.twist.twist.angular.z = 0.1;
  map_publisher_->publish(GlobalMap());
  tf_publisher_->publish(MapFromOdom());
  odometry_publisher_->publish(moving);
  ASSERT_TRUE(WaitFor([this] {
    return ExplorationNodeTestPeer::LatestPose(*explorer_).has_value();
  }));
  task_publisher_->publish(StartTask("stop-wait-diagnostics"));
  ASSERT_TRUE(WaitFor([this] {
    return DiagnosticValue("reason_code") == "WAITING_FOR_STOP";
  }));

  auto diagnostics = LatestDiagnostics();
  ASSERT_TRUE(diagnostics.has_value());
  ASSERT_EQ(diagnostics->status.size(), 1U);
  for (const char* key : kStopWaitDiagnosticKeys) {
    EXPECT_EQ(std::ranges::count_if(
                  diagnostics->status.front().values,
                  [key](const auto& value) { return value.key == key; }),
              1U)
        << key;
  }
  EXPECT_EQ(DiagnosticValue("stop_wait_elapsed_ms"), "0.000000");
  EXPECT_EQ(DiagnosticValue("stop_entry_linear_mps"), "0.200000");
  EXPECT_EQ(DiagnosticValue("stop_entry_angular_radps"), "0.100000");
  EXPECT_EQ(DiagnosticValue("stop_confirmed_linear_mps"), "unavailable");
  EXPECT_EQ(DiagnosticValue("stop_confirmed_angular_radps"), "unavailable");
  EXPECT_EQ(DiagnosticValue("stop_confirmation_samples"), "0");
  EXPECT_EQ(DiagnosticValue("planning_epoch"), "0");

  now_ns->store(std::chrono::duration_cast<std::chrono::nanoseconds>(5s)
                    .count());
  auto diagnostics_count = DiagnosticsCount();
  odometry_publisher_->publish(Odometry(2.1, 2.0));
  ASSERT_TRUE(WaitFor([this, diagnostics_count] {
    return DiagnosticsCount() > diagnostics_count;
  }));
  EXPECT_EQ(DiagnosticValue("stop_wait_elapsed_ms"), "5000.000000");
  EXPECT_EQ(DiagnosticValue("stop_confirmation_samples"), "1");
  EXPECT_EQ(DiagnosticValue("planning_epoch"), "0");
  const auto waiting_status = LatestStatus();
  ASSERT_TRUE(waiting_status.has_value());
  EXPECT_EQ(waiting_status->reason_code, "WAITING_FOR_STOP");
  EXPECT_EQ(waiting_status->failed_candidate_count, 0U);
  EXPECT_TRUE(server_->Goals().empty());

  now_ns->store(std::chrono::duration_cast<std::chrono::nanoseconds>(5100ms)
                    .count());
  auto reset = Odometry(2.2, 2.0);
  reset.twist.twist.linear.x = 0.2;
  diagnostics_count = DiagnosticsCount();
  odometry_publisher_->publish(reset);
  ASSERT_TRUE(WaitFor([this, diagnostics_count] {
    return DiagnosticsCount() > diagnostics_count;
  }));
  EXPECT_EQ(DiagnosticValue("stop_confirmation_samples"), "0");
  EXPECT_EQ(DiagnosticValue("planning_epoch"), "0");

  for (const auto [x, elapsed] :
       {std::pair{2.3, 5200ms}, std::pair{2.4, 5300ms},
        std::pair{2.5, 5400ms}}) {
    now_ns->store(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    diagnostics_count = DiagnosticsCount();
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, diagnostics_count] {
      return DiagnosticsCount() > diagnostics_count;
    }));
  }
  ASSERT_TRUE(WaitFor([this] { return server_->Goals().size() == 1U; }));
  EXPECT_EQ(DiagnosticValue("stop_wait_elapsed_ms"), "5400.000000");
  EXPECT_EQ(DiagnosticValue("stop_entry_linear_mps"), "0.200000");
  EXPECT_EQ(DiagnosticValue("stop_entry_angular_radps"), "0.100000");
  EXPECT_EQ(DiagnosticValue("stop_confirmed_linear_mps"), "0.000000");
  EXPECT_EQ(DiagnosticValue("stop_confirmed_angular_radps"), "0.000000");
  EXPECT_EQ(DiagnosticValue("stop_confirmation_samples"), "3");
  EXPECT_EQ(DiagnosticValue("planning_epoch"), "1");
}

TEST_F(ExplorationNodeTest,
       DiagnosticDuePublishesWithoutChangingPlannerTimingOrStopState) {
  auto now_ns = std::make_shared<std::atomic<std::int64_t>>(0);
  parameters_.stop_before_planning = true;
  parameters_.steady_now = [now_ns] {
    return std::chrono::steady_clock::time_point{
        std::chrono::nanoseconds{now_ns->load()}};
  };
  Start(FakePlannerServer::Mode::kRollingReachable);
  PublishAllInputs(StartTask("diagnostic-due-timing-isolation"));
  ASSERT_TRUE(WaitFor([this] {
    return DiagnosticValue("reason_code") == "WAITING_FOR_STOP";
  }));
  for (const double x : {2.1, 2.2, 2.3}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const auto request_id = ExplorationNodeTestPeer::ActiveRequestId(*explorer_);
  ASSERT_TRUE(request_id.has_value());
  PublishPlannerTiming(*request_id);
  ASSERT_TRUE(WaitFor([this, &request_id] {
    return DiagnosticValue("request_id") == request_id &&
           DiagnosticValue("candidate_record_count") == "1";
  }));
  const auto timing_before_wait = PlannerTimingDiagnosticBytes();
  ASSERT_EQ(timing_before_wait.size(), kPlannerOutputTimingKeys.size());

  const auto reference = References().front();
  const auto endpoint =
      reference.trajectory.points.back().transforms.front().translation;
  now_ns->store(std::chrono::duration_cast<std::chrono::nanoseconds>(1s)
                    .count());
  auto moving_at_endpoint = Odometry(endpoint.x, endpoint.y);
  moving_at_endpoint.twist.twist.linear.x = 0.2;
  odometry_publisher_->publish(moving_at_endpoint);
  ASSERT_TRUE(WaitFor([this, &endpoint] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == endpoint.x && pose->y == endpoint.y;
  }));
  explorer_->PollExecution();
  ASSERT_TRUE(WaitFor([this] {
    return DiagnosticValue("reason_code") == "WAITING_FOR_STOP";
  }));
  EXPECT_EQ(PlannerTimingDiagnosticBytes(), timing_before_wait);
  const auto planner_goal_count = server_->Goals().size();

  now_ns->store(std::chrono::duration_cast<std::chrono::nanoseconds>(6s)
                    .count());
  const auto diagnostics_count = DiagnosticsCount();
  odometry_publisher_->publish(Odometry(endpoint.x, endpoint.y + 0.01));
  ASSERT_TRUE(WaitFor([this, diagnostics_count] {
    return DiagnosticsCount() > diagnostics_count;
  }));
  EXPECT_EQ(PlannerTimingDiagnosticBytes(), timing_before_wait);
  EXPECT_EQ(DiagnosticValue("stop_wait_elapsed_ms"), "5000.000000");
  EXPECT_EQ(DiagnosticValue("stop_confirmation_samples"), "1");
  EXPECT_EQ(DiagnosticValue("planning_epoch"), "1");
  EXPECT_EQ(DiagnosticValue("stop_confirmed_linear_mps"), "unavailable");
  EXPECT_EQ(DiagnosticValue("stop_confirmed_angular_radps"), "unavailable");
  EXPECT_EQ(server_->Goals().size(), planner_goal_count);
  const auto status = LatestStatus();
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->state, Status::REPLANNING);
  EXPECT_EQ(status->reason_code, "WAITING_FOR_STOP");
  EXPECT_EQ(status->failed_candidate_count, 0U);
}

TEST_F(ExplorationNodeTest,
       DisabledStopGateNeverCreatesAStopRecordOrPlanningEpoch) {
  parameters_.stop_before_planning = false;
  Start(FakePlannerServer::Mode::kDelayed);
  auto moving = Odometry();
  moving.twist.twist.linear.x = 0.2;
  PublishAllInputs(StartTask("stop-gate-disabled"), GlobalMap(), moving);
  ASSERT_TRUE(WaitFor([this] { return server_->Goals().size() == 1U; }));
  for (const char* key : kStopWaitDiagnosticKeys) {
    ASSERT_TRUE(DiagnosticValue(key).has_value()) << key;
  }
  EXPECT_EQ(DiagnosticValue("stop_wait_elapsed_ms"), "unavailable");
  EXPECT_EQ(DiagnosticValue("stop_entry_linear_mps"), "unavailable");
  EXPECT_EQ(DiagnosticValue("stop_entry_angular_radps"), "unavailable");
  EXPECT_EQ(DiagnosticValue("stop_confirmed_linear_mps"), "unavailable");
  EXPECT_EQ(DiagnosticValue("stop_confirmed_angular_radps"), "unavailable");
  EXPECT_EQ(DiagnosticValue("stop_confirmation_samples"), "unavailable");
  EXPECT_EQ(DiagnosticValue("planning_epoch"), "0");
}

TEST_F(ExplorationNodeTest, NonFiniteStopEntryIsPublishedAsUnavailable) {
  parameters_.stop_before_planning = true;
  Start(FakePlannerServer::Mode::kDelayed);
  auto nonfinite = Odometry();
  nonfinite.twist.twist.linear.x =
      std::numeric_limits<double>::quiet_NaN();
  map_publisher_->publish(GlobalMap());
  tf_publisher_->publish(MapFromOdom());
  odometry_publisher_->publish(nonfinite);
  ASSERT_TRUE(WaitFor([this] {
    return ExplorationNodeTestPeer::LatestPose(*explorer_).has_value();
  }));
  task_publisher_->publish(StartTask("nonfinite-stop-entry"));
  ASSERT_TRUE(WaitFor([this] {
    return DiagnosticValue("reason_code") == "WAITING_FOR_STOP";
  }));
  EXPECT_EQ(DiagnosticValue("stop_entry_linear_mps"), "unavailable");
  EXPECT_EQ(DiagnosticValue("planning_epoch"), "0");

  for (const double x : {2.1, 2.2, 2.3}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this] { return server_->Goals().size() == 1U; }));
  EXPECT_EQ(DiagnosticValue("stop_entry_linear_mps"), "unavailable");
  EXPECT_EQ(DiagnosticValue("stop_confirmed_linear_mps"), "0.000000");
  EXPECT_EQ(DiagnosticValue("planning_epoch"), "1");
}

TEST_F(ExplorationNodeTest,
       GateCancelRetainsLastRecordButTaskCancelResetsStopLifetime) {
  parameters_.stop_before_planning = true;
  Start(FakePlannerServer::Mode::kRollingReachable);
  auto moving = Odometry();
  moving.twist.twist.linear.x = 0.2;
  moving.twist.twist.angular.z = 0.1;
  map_publisher_->publish(GlobalMap());
  tf_publisher_->publish(MapFromOdom());
  odometry_publisher_->publish(moving);
  ASSERT_TRUE(WaitFor([this] {
    return ExplorationNodeTestPeer::LatestPose(*explorer_).has_value();
  }));
  task_publisher_->publish(StartTask("stop-record-lifetime"));
  ASSERT_TRUE(WaitFor([this] {
    return DiagnosticValue("reason_code") == "WAITING_FOR_STOP";
  }));
  for (const double x : {2.1, 2.2, 2.3}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));

  const auto endpoint = References()
                            .front()
                            .trajectory.points.back()
                            .transforms.front()
                            .translation;
  auto moving_at_endpoint = Odometry(endpoint.x, endpoint.y);
  moving_at_endpoint.twist.twist.linear.x = 0.4;
  moving_at_endpoint.twist.twist.angular.z = 0.3;
  odometry_publisher_->publish(moving_at_endpoint);
  ASSERT_TRUE(WaitFor([this, &endpoint] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == endpoint.x && pose->y == endpoint.y;
  }));
  explorer_->PollExecution();
  ASSERT_TRUE(WaitFor([this] {
    return DiagnosticValue("stop_entry_linear_mps") == "0.400000";
  }));
  EXPECT_EQ(DiagnosticValue("stop_confirmed_linear_mps"), "unavailable");
  EXPECT_EQ(DiagnosticValue("planning_epoch"), "1");

  Task pause;
  pause.command = Task::PAUSE;
  task_publisher_->publish(pause);
  ASSERT_TRUE(WaitFor([this] {
    return DiagnosticValue("reason_code") == "PAUSED";
  }));
  EXPECT_EQ(DiagnosticValue("stop_entry_linear_mps"), "0.200000");
  EXPECT_EQ(DiagnosticValue("stop_entry_angular_radps"), "0.100000");
  EXPECT_EQ(DiagnosticValue("stop_confirmed_linear_mps"), "0.000000");
  EXPECT_EQ(DiagnosticValue("stop_confirmation_samples"), "3");
  EXPECT_EQ(DiagnosticValue("planning_epoch"), "1");

  Task cancel;
  cancel.command = Task::CANCEL;
  task_publisher_->publish(cancel);
  ASSERT_TRUE(WaitFor([this] {
    return DiagnosticValue("planning_epoch") == "0";
  }));
  EXPECT_EQ(DiagnosticValue("stop_wait_elapsed_ms"), "unavailable");
  EXPECT_EQ(DiagnosticValue("stop_entry_linear_mps"), "unavailable");
  EXPECT_EQ(DiagnosticValue("stop_confirmed_linear_mps"), "unavailable");
  EXPECT_EQ(DiagnosticValue("stop_confirmation_samples"), "unavailable");
}

TEST_F(ExplorationNodeTest,
       MapsDuringFreshBuildDoNotRearmConfirmedStationaryAdmission) {
  parameters_.stop_before_planning = true;
  auto build_entered = std::make_shared<std::promise<void>>();
  auto build_entered_future = build_entered->get_future();
  auto release_build = std::make_shared<std::promise<void>>();
  const auto release_build_future = release_build->get_future().share();
  auto build_calls = std::make_shared<std::atomic<std::size_t>>(0U);
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [build_entered, release_build_future, build_calls](
          const lunar::pure_exploration::TaskRaster&,
          std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        if (build_calls->fetch_add(1U) == 0U) {
          build_entered->set_value();
          release_build_future.wait();
        }
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kDelayed);

  auto moving = Odometry();
  moving.twist.twist.linear.x = 0.2;
  PublishAllInputs(StartTask("maps-during-fresh-build"), GlobalMap(), moving);
  EXPECT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    const auto status = LatestStatus();
    return pose && pose->x == 2.0 && status &&
           status->reason_code == "WAITING_FOR_STOP";
  }));
  EXPECT_EQ(server_->Goals().size(), 0U);

  for (const double x : {2.1, 2.2, 2.3}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    EXPECT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  EXPECT_EQ(build_entered_future.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(server_->Goals().size(), 0U);

  const auto publish_and_observe_map = [this](
                                          const std::uint32_t width,
                                          const double resolution) {
    map_publisher_->publish(GlobalMap(width, width, resolution));
    return WaitFor([this, resolution] {
      const auto observed =
          ExplorationNodeTestPeer::LatestMapResolution(*explorer_);
      return observed && std::abs(*observed - resolution) < 1.0e-6;
    });
  };
  EXPECT_TRUE(publish_and_observe_map(40U, 0.25));
  EXPECT_TRUE(publish_and_observe_map(50U, 0.20));
  EXPECT_TRUE(publish_and_observe_map(25U, 0.40));
  EXPECT_TRUE(publish_and_observe_map(80U, 0.125));
  EXPECT_EQ(server_->Goals().size(), 0U);
  const auto status_during_build = LatestStatus();
  EXPECT_TRUE(status_during_build.has_value());
  if (status_during_build) {
    EXPECT_NE(status_during_build->reason_code, "WAITING_FOR_STOP");
  }

  release_build->set_value();
  ASSERT_TRUE(WaitFor([this] { return server_->Goals().size() == 1U; }));
  EXPECT_EQ(build_calls->load(), 1U);

  for (const double x : {2.4, 2.5, 2.6, 2.7}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
    EXPECT_EQ(server_->Goals().size(), 1U);
  }
  EXPECT_EQ(build_calls->load(), 1U);
}

TEST_F(ExplorationNodeTest, StopGateParameterLoadRejectsInvalidValues) {
  EXPECT_THROW(ExplorationNode(StopGateParameterOptions(-0.01, 0.02, 3, 5.0)),
               std::invalid_argument);
  EXPECT_THROW(ExplorationNode(StopGateParameterOptions(0.01, -0.02, 3, 5.0)),
               std::invalid_argument);
  EXPECT_THROW(ExplorationNode(StopGateParameterOptions(0.01, 0.02, 0, 5.0)),
               std::invalid_argument);
  EXPECT_THROW(ExplorationNode(StopGateParameterOptions(0.01, 0.02, 3, 0.0)),
               std::invalid_argument);
}

TEST_F(ExplorationNodeTest,
       FreshPlanningRetainsCanceledPlanIdentityUntilStationaryConfirmation) {
  parameters_.stop_before_planning = true;
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster&,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  server_->SetFixedPlanId("wheel-active");
  PublishAllInputs(StartTask("active-before-stop"));
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return status && status->reason_code == "WAITING_FOR_STOP" && pose &&
           pose->x == 2.0 && pose->y == 2.0;
  }));
  for (const double x : {2.1, 2.2, 2.3}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this] { return !server_->Goals().empty(); }));
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  ASSERT_EQ(References().front().plan_id, "wheel-active");

  server_->SetMode(FakePlannerServer::Mode::kDelayed);
  const auto goal_count_before_stop = server_->Goals().size();
  auto moving = Odometry(2.4, 2.0);
  moving.twist.twist.linear.x = 0.2;
  odometry_publisher_->publish(moving);
  ASSERT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == 2.4;
  }));
  task_publisher_->publish(StartTask("fresh-after-stop"));

  ASSERT_TRUE(WaitFor([this] {
    return ExecutionCancels() == std::vector<std::string>{"wheel-active"};
  }));
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP";
  }));
  auto retained = ExplorationNodeTestPeer::ActiveReference(*explorer_);
  ASSERT_TRUE(retained.has_value());
  EXPECT_EQ(retained->plan_id, "wheel-active");
  EXPECT_EQ(server_->Goals().size(), goal_count_before_stop);

  for (const double x : {2.5, 2.6}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
    EXPECT_TRUE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
    EXPECT_EQ(server_->Goals().size(), goal_count_before_stop);
  }

  odometry_publisher_->publish(Odometry(2.7, 2.0));
  ASSERT_TRUE(WaitFor([this, goal_count_before_stop] {
    return server_->Goals().size() > goal_count_before_stop;
  }));
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
  EXPECT_EQ(ExecutionCancels(), std::vector<std::string>{"wheel-active"});
}

TEST_F(ExplorationNodeTest,
       GlobalGoalCellFilterKeepsKnownFreeCandidateAdjacentToUnknown) {
  // Match the deployed wheeled envelope: 0.7187 m circumscribed footprint
  // radius plus 0.2 m clearance.  The test map resolution is 0.5 m, so the
  // candidate at x=4.5 is known free and immediately adjacent to unknown.
  parameters_.platform.footprint_vertices = {{0.591, 0.409},
                                             {0.591, -0.409},
                                             {-0.591, -0.409},
                                             {-0.591, 0.409}};
  parameters_.platform.minimum_clearance_m = 0.2;
  parameters_.filter_global_goal_cell = true;
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster&,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        auto candidates = ControlledCandidates(frontiers, 2U);
        // x=5.5 is unknown and must be rejected; x=4.5 is a known free cell
        // adjacent to that unknown cell and must not be obstacle-inflated.
        candidates[0].pose = {5.5, 2.0, 0.0};
        candidates[1].pose = {4.5, 2.0, 0.0};
        return candidates;
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);

  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();

  ASSERT_TRUE(WaitFor([this] { return !server_->Goals().empty(); }));
  const auto goals = server_->Goals();
  ASSERT_FALSE(goals.empty());
  // The fake server completes immediately, so the node may construct a
  // follow-up cycle before this assertion samples the request history.  Every
  // dispatched goal must nevertheless be the globally feasible candidate:
  // x=5.5 is unknown, whereas x=4.5 is free beside unknown.
  for (const auto& goal : goals) {
    EXPECT_DOUBLE_EQ(goal.goal.point.x, 4.5);
    EXPECT_DOUBLE_EQ(goal.goal.point.y, 2.0);
  }
}

TEST_F(ExplorationNodeTest,
       HeavyBuildRunsOffCallbackMutexAndCanceledWorkCannotCommit) {
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  std::promise<void> build_entered;
  auto entered = build_entered.get_future();
  std::promise<void> release_build;
  const auto release = release_build.get_future().share();
  seams->generate_candidates =
      [&build_entered, release](
          const lunar::pure_exploration::TaskRaster&,
          std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        build_entered.set_value();
        release.wait();
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kDelayed);
  PublishAllInputs();
  ASSERT_EQ(entered.wait_for(3s), std::future_status::ready);

  Task cancel;
  cancel.command = Task::CANCEL;
  task_publisher_->publish(cancel);
  const bool canceled_while_build_blocked = WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() && status->state == Status::IDLE;
  }, 500ms);

  release_build.set_value();
  EXPECT_TRUE(canceled_while_build_blocked);
  EXPECT_TRUE(server_->Goals().empty());
  std::this_thread::sleep_for(50ms);
  const auto status = LatestStatus();
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->state, Status::IDLE);
  EXPECT_TRUE(server_->Goals().empty());
}

TEST_F(ExplorationNodeTest,
       BuildsFrozenCycleSeriallyAndPublishesOnlySelectedSavedReference) {
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->state == Status::EXECUTING;
  }));

  const auto statuses = Statuses();
  const auto planning = std::ranges::find_if(
      statuses, [](const Status& status) { return status.state == Status::PLANNING; });
  const auto executing = std::ranges::find_if(
      statuses, [](const Status& status) { return status.state == Status::EXECUTING; });
  ASSERT_NE(planning, statuses.end());
  ASSERT_NE(executing, statuses.end());
  EXPECT_LT(std::distance(statuses.begin(), planning),
            std::distance(statuses.begin(), executing));

  const auto cycle = ExplorationNodeTestPeer::ActiveCycle(*explorer_);
  ASSERT_NE(cycle, nullptr);
  EXPECT_DOUBLE_EQ(cycle->frozen_robot_pose().x, 2.0);
  EXPECT_DOUBLE_EQ(cycle->frozen_robot_pose().y, 2.0);
  EXPECT_DOUBLE_EQ(cycle->frozen_resolution_m(), 0.5);
  EXPECT_EQ(cycle->complete_gains().size(), cycle->batch()->candidates().size());
  for (const auto& candidate : cycle->batch()->candidates()) {
    ASSERT_LT(candidate.frontier_index, cycle->batch()->frontiers().size());
    EXPECT_EQ(*candidate.frontier_canonical_key,
              cycle->batch()->frontiers()[candidate.frontier_index].canonical_key);
  }
  EXPECT_LE(server_->Goals().size(), 16U);
  EXPECT_EQ(server_->MaximumOutstanding(), 1U);

  const auto published = References().front();
  const auto saved = ExplorationNodeTestPeer::ActiveReference(*explorer_);
  ASSERT_TRUE(saved.has_value());
  EXPECT_EQ(published, *saved);
  ASSERT_TRUE(published.plan_id.starts_with("plan:"));
  const std::string selected_request_id = published.plan_id.substr(5U);
  EXPECT_EQ(ExplorationNodeTestPeer::ActiveRequestId(*explorer_),
            selected_request_id);
  EXPECT_TRUE(std::ranges::any_of(server_->Goals(), [&](const auto& goal) {
    return goal.request_id == selected_request_id;
  }));
  EXPECT_EQ(ExplorationNodeTestPeer::ExecutablePolyline(*explorer_).size(), 1U);
  const auto endpoint =
      ExplorationNodeTestPeer::ExecutableEndpoint(*explorer_);
  ASSERT_TRUE(endpoint.has_value());
  const auto polyline =
      ExplorationNodeTestPeer::ExecutablePolyline(*explorer_);
  ASSERT_FALSE(polyline.empty());
  EXPECT_DOUBLE_EQ(endpoint->x, polyline.back().x);
  EXPECT_DOUBLE_EQ(endpoint->y, polyline.back().y);
  EXPECT_DOUBLE_EQ(ExplorationNodeTestPeer::PlatformWidth(*explorer_), 0.2);
  explorer_->PollExecution();  // Explicit test seam; production also polls on a wall timer.
}

TEST_F(ExplorationNodeTest, WallTimerReleasesFinalArrivalWithoutManualPoll) {
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const auto final = ExplorationNodeTestPeer::ActiveTarget(*explorer_);
  ASSERT_TRUE(final.has_value());
  odometry_publisher_->publish(
      Odometry(final->x, final->y, final->yaw));
  ASSERT_TRUE(WaitFor([this, &final] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose.has_value() && pose->x == final->x && pose->y == final->y;
  }));
  ASSERT_TRUE(WaitFor([this] {
    const auto latest = LatestStatus();
    return latest.has_value() && latest->completed_goal_count == 1U;
  }));
}

TEST_F(ExplorationNodeTest,
       PublishesCommittedGoalFrozenMarkersAndCompleteOutputDiagnostics) {
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  ASSERT_TRUE(WaitFor([this] {
    return CurrentGoalCount() > 0U && LatestMarkers().has_value() &&
           LatestDiagnostics().has_value();
  }));

  const auto target = ExplorationNodeTestPeer::ActiveTarget(*explorer_);
  const auto goal = LatestCurrentGoal();
  ASSERT_TRUE(target.has_value());
  ASSERT_TRUE(goal.has_value());
  EXPECT_EQ(goal->header.frame_id, "map");
  EXPECT_DOUBLE_EQ(goal->pose.position.x, target->x);
  EXPECT_DOUBLE_EQ(goal->pose.position.y, target->y);

  const auto markers = LatestMarkers();
  ASSERT_TRUE(markers.has_value());
  EXPECT_TRUE(std::ranges::any_of(markers->markers, [](const auto& marker) {
    return marker.ns == "candidates" &&
           marker.action == visualization_msgs::msg::Marker::ADD &&
           marker.color.r == 1.0F;
  }));
  const auto diagnostics = LatestDiagnostics();
  ASSERT_TRUE(diagnostics.has_value());
  ASSERT_EQ(diagnostics->status.size(), 1U);
  const auto has_key = [&diagnostics](const std::string& key) {
    return std::ranges::any_of(diagnostics->status.front().values,
                               [&key](const auto& value) {
                                 return value.key == key;
                               });
  };
  EXPECT_TRUE(has_key("task_id"));
  EXPECT_TRUE(has_key("active_candidate_id"));
  EXPECT_TRUE(has_key("frontier_detection_call_count"));
  EXPECT_TRUE(has_key("information_gain_accumulated_elapsed_ms"));
  EXPECT_TRUE(has_key("global_elapsed_ms"));
}

TEST_F(ExplorationNodeTest,
       MissingTimingForCommittedRequestDoesNotReuseAnotherCandidateRecord) {
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const auto committed = ExplorationNodeTestPeer::ActiveRequestId(*explorer_);
  ASSERT_TRUE(committed.has_value());
  const auto goals = server_->Goals();
  const auto other = std::ranges::find_if(
      goals, [&committed](const auto& goal) {
        return goal.request_id != *committed;
      });
  ASSERT_NE(other, goals.end());
  PublishPlannerTiming(other->request_id);
  ASSERT_TRUE(WaitFor([this] {
    return DiagnosticValue("candidate_record_count") == "1";
  }));
  EXPECT_EQ(DiagnosticValue("request_id"), committed);
  EXPECT_EQ(DiagnosticValue("global_elapsed_ms"), "unavailable");
  EXPECT_EQ(DiagnosticValue("local_elapsed_ms"), "unavailable");
  EXPECT_EQ(DiagnosticValue("total_elapsed_ms"), "unavailable");
}

TEST_F(ExplorationNodeTest,
       PlannerCallCountExistsBeforeAnyPlannerTimingRecordArrives) {
  Start(FakePlannerServer::Mode::kDelayed);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return server_->HasHandle(); }));
  ASSERT_TRUE(WaitFor([this] {
    return DiagnosticValue("candidate_call_count") == "1";
  }));
  EXPECT_EQ(DiagnosticValue("candidate_call_count"), "1");
  EXPECT_EQ(DiagnosticValue("candidate_record_count"), "0");
}

TEST_F(ExplorationNodeTest,
       SnapshotToGoalTimesOnlyTheCommittedFrozenGeneration) {
  auto now = std::make_shared<std::chrono::steady_clock::time_point>();
  parameters_.steady_now = [now] { return *now; };
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->final_rank = [now](
                          std::span<const lunar::pure_exploration::CandidateView>
                              candidates,
                          std::span<const lunar::pure_exploration::FrontierCluster>
                              frontiers,
                          std::span<const lunar::pure_exploration::CandidateGain>
                              gains,
                          std::span<const lunar::pure_exploration::PlannedCandidate>
                              planned,
                          lunar::pure_exploration::Pose2 pose,
                          std::span<const lunar::pure_exploration::Vec2> completed,
                          double resolution) {
    *now += 1ms;
    return lunar::pure_exploration::CandidateRanker(
               {.information_gain = 0.60,
                .path_cost = 0.30,
                .heading_change = 0.05,
                .revisit = 0.05},
               0.2)
        .FinalRank(candidates, frontiers, gains, planned, pose, completed,
                   resolution);
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  ASSERT_TRUE(WaitFor([this] {
    return DiagnosticValue("snapshot_to_goal_call_count") == "1";
  }));
  ASSERT_TRUE(DiagnosticValue("snapshot_to_goal_last_elapsed_ms").has_value());
  EXPECT_GT(std::stod(*DiagnosticValue("snapshot_to_goal_last_elapsed_ms")),
            0.0);
}

TEST_F(ExplorationNodeTest,
       ContinuousPoseUpdatesDoNotInvalidateAnInProgressFrozenBuild) {
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster&,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        std::this_thread::sleep_for(20ms);
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();

  std::atomic<bool> stop_updates{false};
  std::jthread pose_updates{[this, &stop_updates] {
    while (!stop_updates.load()) {
      odometry_publisher_->publish(Odometry());
      std::this_thread::sleep_for(1ms);
    }
  }};
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }, 2s));
  stop_updates.store(true);
}

TEST_F(ExplorationNodeTest,
       FinalEndpointWithUnreachedYawDoesNotTriggerRollingReplan) {
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const auto target = ExplorationNodeTestPeer::ActiveTarget(*explorer_);
  ASSERT_TRUE(target.has_value());

  odometry_publisher_->publish(Odometry(target->x, target->y, target->yaw + 1.0));
  ASSERT_TRUE(WaitFor([this, &target] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == target->x && pose->y == target->y;
  }));
  std::this_thread::sleep_for(250ms);
  EXPECT_TRUE(ExecutionCancels().empty());
  EXPECT_EQ(server_->CancelCount(), 0U);
  EXPECT_EQ(ReferenceCount(), 1U);
}

TEST_F(ExplorationNodeTest,
       RetryableCandidateEvaluationIsRetriedWithoutNewInputs) {
  Start(FakePlannerServer::Mode::kRetryable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return !server_->Goals().empty(); }));
  const auto first_goal_count = server_->Goals().size();
  server_->SetMode(FakePlannerServer::Mode::kReachable);

  ASSERT_TRUE(WaitFor([this, first_goal_count] {
    return server_->Goals().size() > first_goal_count && ReferenceCount() == 1U;
  }, 2s));
  EXPECT_EQ(ExecutionCancels().size(), 0U);
}

TEST_F(ExplorationNodeTest,
       RepeatedRetryableEvaluationAdvancesToTheNextCandidate) {
  Start(FakePlannerServer::Mode::kRetryable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return server_->Goals().size() >= 3U; }, 2s));

  const auto goals = server_->Goals();
  const auto first_candidate = ExplorationNodeTestPeer::CandidateIndexForRequest(
      *explorer_, goals[0].request_id);
  const auto third_candidate = ExplorationNodeTestPeer::CandidateIndexForRequest(
      *explorer_, goals[2].request_id);
  ASSERT_TRUE(first_candidate.has_value());
  ASSERT_TRUE(third_candidate.has_value());
  EXPECT_NE(*first_candidate, *third_candidate);
}

TEST_F(ExplorationNodeTest,
       RemoteMapChangeDoesNotCancelAStillValidCommittedGoal) {
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster&,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  auto map = GlobalMap();
  PublishAllInputs(StartTask(), map);
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const auto original_goal_count = server_->Goals().size();
  const auto status_count = StatusCount();

  map.data[0] = 100;
  map_publisher_->publish(map);
  ASSERT_TRUE(WaitFor([this, status_count] { return StatusCount() > status_count; }));
  EXPECT_TRUE(ExecutionCancels().empty());
  EXPECT_EQ(server_->CancelCount(), 0U);
  EXPECT_EQ(server_->Goals().size(), original_goal_count);
  EXPECT_EQ(ReferenceCount(), 1U);
  EXPECT_TRUE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
}

TEST_F(ExplorationNodeTest,
       MapInvalidationRetainsCanceledReferenceUntilStopThenBuildsFresh) {
  parameters_.stop_before_planning = true;
  auto build_calls = std::make_shared<std::atomic<std::size_t>>(0U);
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [build_calls](const lunar::pure_exploration::TaskRaster& raster,
                    std::span<const lunar::pure_exploration::FrontierCluster>
                        frontiers) {
        ++*build_calls;
        auto candidates = ControlledCandidates(frontiers, 1U);
        const auto old_target = raster.WorldToCell({4.5, 2.0});
        const bool old_target_free =
            old_target && raster.Classify(*old_target) ==
                              lunar::pure_exploration::CellState::kFree;
        candidates.front().pose =
            old_target_free ? lunar::pure_exploration::Pose2{4.5, 2.0, 0.0}
                            : lunar::pure_exploration::Pose2{3.5, 2.0, 0.0};
        candidates.front().key.x_mm = old_target_free ? 4500 : 3500;
        return candidates;
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  auto map = GlobalMap();
  PublishAllInputs(StartTask("map-stop-refresh"), map);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP";
  }));
  for (const double x : {2.1, 2.2, 2.3}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const auto first = References().front();
  const auto initial_goals = server_->Goals().size();
  const auto initial_build_calls = build_calls->load();
  server_->SetMode(FakePlannerServer::Mode::kDelayed);

  auto moving = Odometry(2.4, 2.0);
  moving.twist.twist.linear.x = 0.2;
  odometry_publisher_->publish(moving);
  ASSERT_TRUE(WaitFor([this] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == 2.4;
  }));
  map.data[4U * map.info.width + 9U] = 100;
  map_publisher_->publish(map);

  ASSERT_TRUE(WaitFor([this, &first] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP" &&
           ExecutionCancels() == std::vector<std::string>{first.plan_id};
  }));
  const auto retained = ExplorationNodeTestPeer::ActiveReference(*explorer_);
  ASSERT_TRUE(retained.has_value());
  EXPECT_EQ(retained->plan_id, first.plan_id);
  EXPECT_EQ(server_->Goals().size(), initial_goals);
  // The map callback performs one foreground validity check through the same
  // candidate seam, but must not start a new planner batch before stop.
  EXPECT_EQ(build_calls->load(), initial_build_calls + 1U);

  for (const double x : {2.41, 2.42}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
    EXPECT_TRUE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
    EXPECT_EQ(server_->Goals().size(), initial_goals);
  }
  odometry_publisher_->publish(Odometry(2.43, 2.0));
  ASSERT_TRUE(WaitFor([this, build_calls, initial_build_calls] {
    const auto status = LatestStatus();
    return status && status->reason_code == "CANDIDATE_INVALID" &&
           !ExplorationNodeTestPeer::ActiveReference(*explorer_) &&
           build_calls->load() > initial_build_calls + 1U;
  }));
  ASSERT_TRUE(WaitFor([this, initial_goals] {
    return server_->Goals().size() > initial_goals;
  }));
  const auto latest_goal = server_->Goals().back();
  EXPECT_DOUBLE_EQ(latest_goal.goal.point.x, 3.5);
  EXPECT_EQ(LatestStatus()->failed_candidate_count, 0U);
  EXPECT_EQ(ExecutionCancels(), std::vector<std::string>{first.plan_id});
}

TEST_F(ExplorationNodeTest,
       LocalEndpointFreshAuthorityOutranksOlderBlockedMapValidation) {
  parameters_.stop_before_planning = true;
  auto block_next_validation = std::make_shared<std::atomic<bool>>(false);
  auto validation_entered = std::make_shared<std::promise<void>>();
  auto release_validation = std::make_shared<std::promise<void>>();
  const auto entered = validation_entered->get_future().share();
  const auto release = release_validation->get_future().share();
  auto build_calls = std::make_shared<std::atomic<std::size_t>>(0U);
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [block_next_validation, validation_entered, release, build_calls](
          const lunar::pure_exploration::TaskRaster& raster,
          std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        ++*build_calls;
        if (block_next_validation->exchange(false)) {
          validation_entered->set_value();
          release.wait();
        }
        auto candidates = ControlledCandidates(frontiers, 1U);
        const auto old_target = raster.WorldToCell({4.5, 2.0});
        const bool old_target_free =
            old_target && raster.Classify(*old_target) ==
                              lunar::pure_exploration::CellState::kFree;
        candidates.front().pose =
            old_target_free ? lunar::pure_exploration::Pose2{4.5, 2.0, 0.0}
                            : lunar::pure_exploration::Pose2{3.5, 2.0, 0.0};
        candidates.front().key.x_mm = old_target_free ? 4500 : 3500;
        return candidates;
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kRollingReachable);
  auto map = GlobalMap();
  PublishAllInputs(StartTask("endpoint-outranks-map-validation"), map);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP";
  }));
  for (const double x : {2.1, 2.2, 2.3}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const auto first = References().front();
  ASSERT_FALSE(first.trajectory.points.empty());
  const auto endpoint =
      first.trajectory.points.back().transforms.front().translation;
  const auto initial_cycle = ExplorationNodeTestPeer::ActiveCycle(*explorer_);
  ASSERT_NE(initial_cycle, nullptr);
  const auto initial_goals = server_->Goals().size();
  const auto initial_build_calls = build_calls->load();
  server_->SetMode(FakePlannerServer::Mode::kDelayed);

  block_next_validation->store(true);
  map.data[4U * map.info.width + 9U] = 100;
  map_publisher_->publish(map);
  const bool validation_is_blocked =
      entered.wait_for(1s) == std::future_status::ready;
  if (!validation_is_blocked) {
    release_validation->set_value();
  }
  ASSERT_TRUE(validation_is_blocked);

  auto moving_at_endpoint = Odometry(endpoint.x, endpoint.y, 0.0);
  moving_at_endpoint.twist.twist.linear.x = 0.2;
  odometry_publisher_->publish(moving_at_endpoint);
  EXPECT_TRUE(WaitFor([this, &endpoint] {
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return pose && pose->x == endpoint.x && pose->y == endpoint.y;
  }));
  explorer_->PollExecution();
  EXPECT_TRUE(WaitFor([this, &first] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP" &&
           ExecutionCancels() == std::vector<std::string>{first.plan_id};
  }));
  for (const double y : {2.01, 2.02}) {
    odometry_publisher_->publish(Odometry(endpoint.x, y, 0.0));
    EXPECT_TRUE(WaitFor([this, &endpoint, y] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == endpoint.x && pose->y == y;
    }));
  }

  const auto statuses_before_release = StatusCount();
  release_validation->set_value();
  ASSERT_TRUE(WaitFor([this, statuses_before_release] {
    return StatusCount() > statuses_before_release;
  }));
  EXPECT_EQ(ExecutionCancels(), std::vector<std::string>{first.plan_id});
  const auto retained = ExplorationNodeTestPeer::ActiveReference(*explorer_);
  ASSERT_TRUE(retained.has_value());
  EXPECT_EQ(retained->plan_id, first.plan_id);
  EXPECT_EQ(server_->Goals().size(), initial_goals);

  odometry_publisher_->publish(Odometry(endpoint.x, 2.03, 0.0));
  ASSERT_TRUE(WaitFor([this, build_calls, initial_build_calls] {
    const auto status = LatestStatus();
    return status && status->reason_code == "LOCAL_SEGMENT_COMPLETED" &&
           build_calls->load() > initial_build_calls + 1U;
  }));
  ASSERT_TRUE(WaitFor([this, initial_goals] {
    return server_->Goals().size() > initial_goals;
  }));
  const auto refreshed_cycle = ExplorationNodeTestPeer::ActiveCycle(*explorer_);
  ASSERT_NE(refreshed_cycle, nullptr);
  ASSERT_NE(refreshed_cycle, initial_cycle);
  ASSERT_EQ(refreshed_cycle->batch()->candidates().size(), 1U);
  EXPECT_DOUBLE_EQ(refreshed_cycle->batch()->candidates().front().pose.x, 3.5);
  const auto status = LatestStatus();
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->completed_goal_count, 0U);
  EXPECT_EQ(status->failed_candidate_count, 0U);
  EXPECT_EQ(ExecutionCancels(), std::vector<std::string>{first.plan_id});
}

TEST_F(ExplorationNodeTest,
       FinalRankCannotCommitCandidateBlockedByNewerGlobalMap) {
  auto final_rank_started = std::make_shared<std::atomic<bool>>(false);
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster& raster,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        const auto target = raster.WorldToCell({4.5, 2.0});
        if (!target || raster.Classify(*target) !=
                           lunar::pure_exploration::CellState::kFree) {
          return std::vector<lunar::pure_exploration::CandidateView>{};
        }
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  seams->final_rank = [final_rank_started](
                          std::span<const lunar::pure_exploration::CandidateView>
                              candidates,
                          std::span<const lunar::pure_exploration::FrontierCluster>
                              frontiers,
                          std::span<const lunar::pure_exploration::CandidateGain>
                              gains,
                          std::span<const lunar::pure_exploration::PlannedCandidate>
                              planned,
                          lunar::pure_exploration::Pose2 pose,
                          std::span<const lunar::pure_exploration::Vec2> completed,
                          double resolution) {
    final_rank_started->store(true);
    std::this_thread::sleep_for(150ms);
    return lunar::pure_exploration::CandidateRanker(
               {.information_gain = 0.60,
                .path_cost = 0.30,
                .heading_change = 0.05,
                .revisit = 0.05},
               0.2)
        .FinalRank(candidates, frontiers, gains, planned, pose, completed,
                   resolution);
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  auto map = GlobalMap();
  PublishAllInputs(StartTask(), map);
  ASSERT_TRUE(WaitFor([&final_rank_started] {
    return final_rank_started->load();
  }));

  map.data[4U * map.info.width + 9U] = 100;
  map_publisher_->publish(map);
  std::this_thread::sleep_for(250ms);
  EXPECT_EQ(ReferenceCount(), 0U);
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
}

TEST_F(ExplorationNodeTest,
       MapChangeAfterFinalRankObservesSameMapCannotCommitOldReference) {
  auto observed = std::make_shared<std::promise<void>>();
  auto release = std::make_shared<std::promise<void>>();
  const auto observed_future = observed->get_future().share();
  const auto release_future = release->get_future().share();
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster& raster,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        const auto target = raster.WorldToCell({4.5, 2.0});
        if (!target || raster.Classify(*target) !=
                           lunar::pure_exploration::CellState::kFree) {
          return std::vector<lunar::pure_exploration::CandidateView>{};
        }
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  seams->after_final_rank_map_observed = [observed, release_future] {
    observed->set_value();
    release_future.wait();
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  auto map = GlobalMap();
  PublishAllInputs(StartTask(), map);
  ASSERT_EQ(observed_future.wait_for(1s), std::future_status::ready);

  const auto statuses_before_map = StatusCount();
  map.data[4U * map.info.width + 9U] = 100;
  map_publisher_->publish(map);
  ASSERT_TRUE(WaitFor([this, statuses_before_map] {
    return StatusCount() > statuses_before_map;
  }));
  release->set_value();
  std::this_thread::sleep_for(200ms);
  EXPECT_EQ(ReferenceCount(), 0U);
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
}

TEST_F(ExplorationNodeTest,
       FineMapValidationOverBudgetFailsClosedWithoutLeavingFinalRankBusy) {
  parameters_.task_raster_limits = {400U};
  auto rank_entered = std::make_shared<std::promise<void>>();
  auto release_rank = std::make_shared<std::promise<void>>();
  const auto entered = rank_entered->get_future().share();
  const auto release = release_rank->get_future().share();
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->final_rank = [rank_entered, release](
                          std::span<const lunar::pure_exploration::CandidateView>
                              candidates,
                          std::span<const lunar::pure_exploration::FrontierCluster>
                              frontiers,
                          std::span<const lunar::pure_exploration::CandidateGain>
                              gains,
                          std::span<const lunar::pure_exploration::PlannedCandidate>
                              planned,
                          lunar::pure_exploration::Pose2 pose,
                          std::span<const lunar::pure_exploration::Vec2> completed,
                          double resolution) {
    rank_entered->set_value();
    release.wait();
    return lunar::pure_exploration::CandidateRanker(
               {.information_gain = 0.60,
                .path_cost = 0.30,
                .heading_change = 0.05,
                .revisit = 0.05},
               0.2)
        .FinalRank(candidates, frontiers, gains, planned, pose, completed,
                   resolution);
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs(StartTask(), GlobalMap(20U, 20U, 0.5));
  ASSERT_EQ(entered.wait_for(1s), std::future_status::ready);

  const auto statuses_before_map = StatusCount();
  map_publisher_->publish(GlobalMap(40U, 40U, 0.25));
  ASSERT_TRUE(WaitFor([this, statuses_before_map] {
    return StatusCount() > statuses_before_map;
  }));
  release_rank->set_value();
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->state == Status::ERROR;
  }));
  EXPECT_EQ(ReferenceCount(), 0U);
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
}

TEST_F(ExplorationNodeTest,
       MapCancellationCannotOvertakeBlockedCommittedReferencePublication) {
  auto publish_entered = std::make_shared<std::promise<void>>();
  auto release_publish = std::make_shared<std::promise<void>>();
  const auto entered = publish_entered->get_future().share();
  const auto release = release_publish->get_future().share();
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster& raster,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        const auto target = raster.WorldToCell({4.5, 2.0});
        if (!target || raster.Classify(*target) !=
                           lunar::pure_exploration::CellState::kFree) {
          return std::vector<lunar::pure_exploration::CandidateView>{};
        }
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  seams->before_reference_publish = [publish_entered, release] {
    publish_entered->set_value();
    release.wait();
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  auto map = GlobalMap();
  PublishAllInputs(StartTask(), map);
  ASSERT_EQ(entered.wait_for(1s), std::future_status::ready);

  map.data[4U * map.info.width + 9U] = 100;
  map_publisher_->publish(map);
  std::this_thread::sleep_for(50ms);
  EXPECT_TRUE(ExecutionCancels().empty());
  release_publish->set_value();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  ASSERT_TRUE(WaitFor([this] { return ExecutionCancels().size() == 1U; }));
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
}

TEST_F(ExplorationNodeTest,
       PauseIsNotBlockedByLongLatestMapValidation) {
  auto block_validation = std::make_shared<std::atomic<bool>>(false);
  auto validation_entered = std::make_shared<std::promise<void>>();
  auto release_validation = std::make_shared<std::promise<void>>();
  const auto entered = validation_entered->get_future().share();
  const auto release = release_validation->get_future().share();
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [block_validation, validation_entered, release](
          const lunar::pure_exploration::TaskRaster&,
          std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        if (block_validation->load()) {
          validation_entered->set_value();
          release.wait();
        }
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  auto map = GlobalMap();
  PublishAllInputs(StartTask(), map);
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));

  block_validation->store(true);
  map.data[0] = 100;
  map_publisher_->publish(map);
  ASSERT_EQ(entered.wait_for(1s), std::future_status::ready);
  Task pause;
  pause.command = Task::PAUSE;
  task_publisher_->publish(pause);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->state == Status::PAUSED;
  }, 200ms));
  ASSERT_TRUE(WaitFor([this] { return ExecutionCancels().size() == 1U; },
                      200ms));
  release_validation->set_value();
}

TEST_F(ExplorationNodeTest,
       InvalidGlobalMapCancelsPublishedReferenceExactlyOnceBeforeError) {
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const std::string plan_id = References().front().plan_id;
  auto invalid_map = GlobalMap();
  invalid_map.data.pop_back();
  map_publisher_->publish(std::move(invalid_map));

  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->state == Status::ERROR;
  }));
  ASSERT_TRUE(WaitFor([this] { return ExecutionCancels().size() == 1U; }));
  EXPECT_EQ(ExecutionCancels(), std::vector<std::string>{plan_id});
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
}

TEST_F(ExplorationNodeTest,
       InvalidCommandCancelsPublishedReferenceExactlyOnceBeforeError) {
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const std::string plan_id = References().front().plan_id;
  Task invalid_command;
  invalid_command.command = std::numeric_limits<std::uint8_t>::max();
  task_publisher_->publish(invalid_command);

  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->state == Status::ERROR;
  }));
  ASSERT_TRUE(WaitFor([this] { return ExecutionCancels().size() == 1U; }));
  EXPECT_EQ(ExecutionCancels(), std::vector<std::string>{plan_id});
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
}

TEST_F(ExplorationNodeTest,
       PendingPauseFreezesElapsedBeforePlannerCancelTerminal) {
  auto now = std::make_shared<std::chrono::steady_clock::time_point>();
  parameters_.steady_now = [now] { return *now; };
  Start(FakePlannerServer::Mode::kDelayed);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return server_->HasHandle(); }));
  *now += 5s;
  Task pause;
  pause.command = Task::PAUSE;
  task_publisher_->publish(pause);
  ASSERT_TRUE(WaitFor([this] { return server_->CancelCount() == 1U; }));
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->active_elapsed_s == 5.0;
  }));
  *now += 10s;
  const auto statuses_before_second_pause = StatusCount();
  task_publisher_->publish(pause);
  ASSERT_TRUE(WaitFor([this, statuses_before_second_pause] {
    return StatusCount() > statuses_before_second_pause;
  }));
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->active_elapsed_s == 5.0;
  }));
}

TEST_F(ExplorationNodeTest,
       LocalEndpointStopsThenBuildsAFreshFrontierBatch) {
  parameters_.stop_before_planning = true;
  auto build_calls = std::make_shared<std::atomic<std::size_t>>(0U);
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [build_calls](const lunar::pure_exploration::TaskRaster&,
                    std::span<const lunar::pure_exploration::FrontierCluster>
                        frontiers) {
        ++*build_calls;
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kRollingReachable);
  PublishAllInputs();

  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP";
  }));
  for (const double x : {2.1, 2.2, 2.3}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const auto first = References().front();
  ASSERT_FALSE(first.trajectory.points.empty());
  const auto& endpoint =
      first.trajectory.points.back().transforms.front().translation;
  const auto initial_goals = server_->Goals().size();
  const auto initial_build_calls = build_calls->load();
  const auto initial_cycle = ExplorationNodeTestPeer::ActiveCycle(*explorer_);
  ASSERT_NE(initial_cycle, nullptr);
  server_->SetMode(FakePlannerServer::Mode::kDelayed);

  auto moving_at_endpoint = Odometry(endpoint.x, endpoint.y, 0.0);
  moving_at_endpoint.twist.twist.linear.x = 0.2;
  odometry_publisher_->publish(moving_at_endpoint);
  ASSERT_TRUE(WaitFor([this, &first] {
    const auto status = LatestStatus();
    return status && status->state == Status::REPLANNING &&
           status->reason_code == "WAITING_FOR_STOP" &&
           ExecutionCancels() == std::vector<std::string>{first.plan_id};
  }));
  EXPECT_EQ(server_->Goals().size(), initial_goals);
  EXPECT_EQ(build_calls->load(), initial_build_calls);
  const auto retained = ExplorationNodeTestPeer::ActiveReference(*explorer_);
  ASSERT_TRUE(retained.has_value());
  EXPECT_EQ(retained->plan_id, first.plan_id);

  for (const double y : {2.01, 2.02}) {
    odometry_publisher_->publish(Odometry(endpoint.x, y, 0.0));
    ASSERT_TRUE(WaitFor([this, &endpoint, y] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == endpoint.x && pose->y == y;
    }));
    EXPECT_EQ(server_->Goals().size(), initial_goals);
    EXPECT_TRUE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
  }
  odometry_publisher_->publish(Odometry(endpoint.x, 2.03, 0.0));
  ASSERT_TRUE(WaitFor([this, build_calls, initial_build_calls] {
    const auto status = LatestStatus();
    return status && status->reason_code == "LOCAL_SEGMENT_COMPLETED" &&
           !ExplorationNodeTestPeer::ActiveReference(*explorer_) &&
           build_calls->load() > initial_build_calls;
  }));
  ASSERT_TRUE(WaitFor([this, initial_goals] {
    return server_->Goals().size() > initial_goals;
  }));
  const auto refreshed_cycle =
      ExplorationNodeTestPeer::ActiveCycle(*explorer_);
  ASSERT_NE(refreshed_cycle, nullptr);
  EXPECT_NE(refreshed_cycle, initial_cycle);
  const auto status = LatestStatus();
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->completed_goal_count, 0U);
  EXPECT_EQ(status->failed_candidate_count, 0U);
  EXPECT_EQ(status->replan_count, 0U);
  EXPECT_EQ(ExecutionCancels(), std::vector<std::string>{first.plan_id});
}

TEST_F(ExplorationNodeTest,
       PauseAndResumeInterruptLocalEndpointReleaseAuthority) {
  parameters_.stop_before_planning = true;
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster&,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kRollingReachable);
  PublishAllInputs(StartTask("endpoint-control-interrupt"));
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP";
  }));
  for (const double x : {2.1, 2.2, 2.3}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const auto first = References().front();
  const auto endpoint =
      first.trajectory.points.back().transforms.front().translation;
  auto moving_at_endpoint = Odometry(endpoint.x, endpoint.y, 0.0);
  moving_at_endpoint.twist.twist.linear.x = 0.2;
  odometry_publisher_->publish(moving_at_endpoint);
  ASSERT_TRUE(WaitFor([this, &first] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP" &&
           ExecutionCancels() == std::vector<std::string>{first.plan_id};
  }));

  Task pause;
  pause.command = Task::PAUSE;
  task_publisher_->publish(pause);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->state == Status::PAUSED &&
           status->reason_code == "PAUSED";
  }));
  Task resume;
  resume.command = Task::RESUME;
  task_publisher_->publish(resume);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP";
  }));
  server_->SetMode(FakePlannerServer::Mode::kDelayed);
  const auto goals_before_resume = server_->Goals().size();
  const auto statuses_before_resume = StatusCount();
  for (const double y : {2.01, 2.02, 2.03}) {
    odometry_publisher_->publish(Odometry(endpoint.x, y, 0.0));
    ASSERT_TRUE(WaitFor([this, &endpoint, y] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == endpoint.x && pose->y == y;
    }));
  }
  ASSERT_TRUE(WaitFor([this, goals_before_resume] {
    return server_->Goals().size() > goals_before_resume;
  }));
  const auto statuses = Statuses();
  ASSERT_LE(statuses_before_resume, statuses.size());
  EXPECT_TRUE(std::ranges::none_of(
      statuses.begin() + static_cast<std::ptrdiff_t>(statuses_before_resume),
      statuses.end(), [](const Status& status) {
        return status.reason_code == "LOCAL_SEGMENT_COMPLETED";
      }));
  EXPECT_EQ(LatestStatus()->completed_goal_count, 0U);
  EXPECT_EQ(LatestStatus()->failed_candidate_count, 0U);
  EXPECT_EQ(ExecutionCancels(), std::vector<std::string>{first.plan_id});
}

TEST_F(ExplorationNodeTest,
       LocalEndpointFreshBatchDoesNotCreateRollingPlannerTiming) {
  Start(FakePlannerServer::Mode::kRollingReachable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const auto first = References().front();
  ASSERT_FALSE(first.trajectory.points.empty());
  const auto& endpoint =
      first.trajectory.points.back().transforms.front().translation;
  const auto initial_goals = server_->Goals().size();
  server_->SetMode(FakePlannerServer::Mode::kDelayed);
  odometry_publisher_->publish(Odometry(endpoint.x, endpoint.y, 0.0));
  ASSERT_TRUE(WaitFor([this, initial_goals] {
    return server_->Goals().size() > initial_goals;
  }));
  const auto fresh_candidate_request = server_->LatestRequestId();
  ASSERT_FALSE(fresh_candidate_request.empty());
  PublishPlannerTiming(fresh_candidate_request);
  ASSERT_TRUE(WaitFor([this, &fresh_candidate_request] {
    return DiagnosticValue("request_id") == fresh_candidate_request &&
           DiagnosticValue("candidate_record_count") == "1";
  }));
  EXPECT_EQ(DiagnosticValue("candidate_record_count"), "1");
  EXPECT_EQ(DiagnosticValue("rolling_record_count"), "0");
  EXPECT_EQ(DiagnosticValue("global_elapsed_ms"), "11.000000");
  EXPECT_EQ(DiagnosticValue("local_elapsed_ms"), "7.000000");
  EXPECT_EQ(DiagnosticValue("total_elapsed_ms"), "18.000000");
}

TEST_F(ExplorationNodeTest, StuckTwiceReplansThenRecordsOnePersistentFailure) {
  auto now = std::make_shared<std::chrono::steady_clock::time_point>();
  parameters_.steady_now = [now] { return *now; };
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  server_->SetMode(FakePlannerServer::Mode::kDelayed);
  for (const std::uint8_t count : {1U, 2U}) {
    const auto references_before = ReferenceCount();
    *now += 30s;
    explorer_->PollExecution();
    ASSERT_TRUE(WaitFor([this, count] {
      const auto status = LatestStatus();
      return status.has_value() && status->replan_count == count &&
             server_->HasHandle();
    }));
    server_->Finish(server_->LatestRequestId(), FakePlannerServer::Mode::kReachable);
    ASSERT_TRUE(WaitFor([this, references_before] {
      return ReferenceCount() > references_before;
    }));
  }
  const auto plan_id = References().back().plan_id;
  *now += 30s;
  explorer_->PollExecution();
  ASSERT_TRUE(WaitFor([this] { return ExecutionCancels().size() >= 3U; }));
  EXPECT_EQ(ExecutionCancels().back(), plan_id);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() && status->failed_candidate_count == 1U;
  }));
}

TEST_F(ExplorationNodeTest,
       ExhaustedStuckRetriesRetainReferenceUntilFreshBuildStopConfirmation) {
  auto now = std::make_shared<std::chrono::steady_clock::time_point>();
  parameters_.steady_now = [now] { return *now; };
  parameters_.stop_before_planning = true;
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster&,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs(StartTask("stuck-stop-gated"));
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP";
  }));
  for (const double x : {2.1, 2.2, 2.3}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));

  for (const std::uint8_t retry : {1U, 2U}) {
    server_->SetMode(FakePlannerServer::Mode::kDelayed);
    const auto goals_before = server_->Goals().size();
    const auto references_before = ReferenceCount();
    *now += 30s;
    explorer_->PollExecution();
    ASSERT_TRUE(WaitFor([this, retry] {
      const auto status = LatestStatus();
      return status && status->state == Status::REPLANNING &&
             status->reason_code == "WAITING_FOR_STOP" &&
             status->replan_count == retry;
    }));
    EXPECT_EQ(server_->Goals().size(), goals_before);
    for (std::size_t sample = 1U; sample <= 3U; ++sample) {
      const double x =
          2.3 + 0.1 * static_cast<double>(retry) +
          0.01 * static_cast<double>(sample);
      odometry_publisher_->publish(Odometry(x, 2.0));
      ASSERT_TRUE(WaitFor([this, x] {
        const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
        return pose && pose->x == x;
      }));
    }
    ASSERT_TRUE(WaitFor([this, goals_before] {
      return server_->Goals().size() > goals_before;
    }));
    const auto request = server_->LatestRequestId();
    ASSERT_TRUE(WaitFor(
        [this, &request] { return server_->HasHandle(request); }));
    server_->Finish(request, FakePlannerServer::Mode::kReachable);
    ASSERT_TRUE(WaitFor([this, references_before] {
      return ReferenceCount() > references_before;
    }));
  }

  const auto exhausted_plan = References().back().plan_id;
  const auto goals_before_exhaustion = server_->Goals().size();
  *now += 30s;
  explorer_->PollExecution();
  ASSERT_TRUE(WaitFor([this, &exhausted_plan] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP" &&
           status->failed_candidate_count == 1U &&
           !ExecutionCancels().empty() &&
           ExecutionCancels().back() == exhausted_plan;
  }));
  const auto retained = ExplorationNodeTestPeer::ActiveReference(*explorer_);
  ASSERT_TRUE(retained.has_value());
  EXPECT_EQ(retained->plan_id, exhausted_plan);
  EXPECT_EQ(server_->Goals().size(), goals_before_exhaustion);

  for (const double x : {2.61, 2.62}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
    EXPECT_TRUE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
  }
  odometry_publisher_->publish(Odometry(2.63, 2.0));
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->state == Status::COMPLETED &&
           status->reason_code == "COMPLETED_NO_REACHABLE_FRONTIER" &&
           !ExplorationNodeTestPeer::ActiveReference(*explorer_);
  }));
  EXPECT_EQ(server_->Goals().size(), goals_before_exhaustion);
  EXPECT_EQ(LatestStatus()->failed_candidate_count, 1U);
}

TEST_F(ExplorationNodeTest,
       FailureMemoryLimitStillCancelsThirdStuckPlanBeforeStableError) {
  auto now = std::make_shared<std::chrono::steady_clock::time_point>();
  parameters_.steady_now = [now] { return *now; };
  parameters_.failure_memory_limits = {8U, 1U, 1024U};
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  server_->SetMode(FakePlannerServer::Mode::kDelayed);
  for (const std::uint8_t count : {1U, 2U}) {
    const auto references_before = ReferenceCount();
    *now += 30s;
    explorer_->PollExecution();
    ASSERT_TRUE(WaitFor([this, count] {
      const auto status = LatestStatus();
      return status.has_value() && status->replan_count == count &&
             server_->HasHandle();
    }));
    server_->Finish(server_->LatestRequestId(),
                    FakePlannerServer::Mode::kReachable);
    ASSERT_TRUE(WaitFor([this, references_before] {
      return ReferenceCount() > references_before;
    }));
  }

  const auto plan_id = References().back().plan_id;
  const auto goals_before_third = server_->Goals().size();
  *now += 30s;
  explorer_->PollExecution();

  ASSERT_TRUE(WaitFor([this] { return ExecutionCancels().size() == 3U; }));
  EXPECT_EQ(ExecutionCancels().back(), plan_id);
  EXPECT_EQ(server_->Goals().size(), goals_before_third);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() && status->state == Status::ERROR;
  }));
  const auto status = LatestStatus();
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->reason_code, "EXECUTION_REPLAN_FAILURE_MEMORY_ERROR");
  std::this_thread::sleep_for(200ms);
  const auto stable_status = LatestStatus();
  ASSERT_TRUE(stable_status.has_value());
  EXPECT_EQ(stable_status->state, Status::ERROR);
  EXPECT_EQ(stable_status->reason_code,
            "EXECUTION_REPLAN_FAILURE_MEMORY_ERROR");
  EXPECT_EQ(server_->Goals().size(), goals_before_third);
}

TEST_F(ExplorationNodeTest, ExecutionReplanTimeoutPreservesCommittedGoal) {
  auto now = std::make_shared<std::chrono::steady_clock::time_point>();
  parameters_.steady_now = [now] { return *now; };
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const auto committed = ExplorationNodeTestPeer::ActiveTarget(*explorer_);
  ASSERT_TRUE(committed.has_value());
  server_->SetMode(FakePlannerServer::Mode::kDelayed);
  const auto goals_before = server_->Goals().size();
  *now += 30s;
  explorer_->PollExecution();
  ASSERT_TRUE(WaitFor([this, goals_before] {
    return server_->Goals().size() > goals_before;
  }));
  const std::string request = server_->LatestRequestId();
  ASSERT_TRUE(WaitFor(
      [this, &request] { return server_->HasHandle(request); }));
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->state == Status::REPLANNING;
  }));
  const auto first_replan_goal = server_->Goals().back();
  const auto references_before_timeout = ReferenceCount();
  const auto first_replan_status = LatestStatus();
  ASSERT_TRUE(first_replan_status.has_value());
  server_->Finish(request, FakePlannerServer::Mode::kRetryable);
  ASSERT_TRUE(WaitFor([this, goals_before] {
    return server_->Goals().size() > goals_before + 1U;
  }));
  const auto second_replan_goal = server_->Goals().back();
  EXPECT_NE(second_replan_goal.request_id, first_replan_goal.request_id);
  EXPECT_EQ(second_replan_goal.mission_id, first_replan_goal.mission_id);
  EXPECT_DOUBLE_EQ(second_replan_goal.goal.point.x,
                   first_replan_goal.goal.point.x);
  EXPECT_DOUBLE_EQ(second_replan_goal.goal.point.y,
                   first_replan_goal.goal.point.y);
  EXPECT_DOUBLE_EQ(second_replan_goal.goal.yaw_rad,
                   first_replan_goal.goal.yaw_rad);
  EXPECT_EQ(ReferenceCount(), references_before_timeout);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() && status->state == Status::REPLANNING &&
           status->replan_count == 1U;
  }));
  const auto status = LatestStatus();
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->reason_code, first_replan_status->reason_code);
  EXPECT_NE(status->reason_code, "NO_PATH");
  EXPECT_NE(status->state, Status::COMPLETED);
  const auto retained = ExplorationNodeTestPeer::ActiveTarget(*explorer_);
  ASSERT_TRUE(retained.has_value());
  EXPECT_DOUBLE_EQ(retained->x, committed->x);
  EXPECT_DOUBLE_EQ(retained->y, committed->y);
  EXPECT_DOUBLE_EQ(retained->yaw, committed->yaw);
}

TEST_F(ExplorationNodeTest, ExecutionReplanNoPathReleasesGoal) {
  auto now = std::make_shared<std::chrono::steady_clock::time_point>();
  parameters_.steady_now = [now] { return *now; };
  Start(FakePlannerServer::Mode::kReachable); PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  server_->SetMode(FakePlannerServer::Mode::kDelayed); const auto before = server_->Goals().size();
  *now += 30s; explorer_->PollExecution();
  ASSERT_TRUE(WaitFor([this, before] { return server_->Goals().size() > before; }));
  const auto request = server_->LatestRequestId();
  ASSERT_TRUE(WaitFor([this, &request] { return server_->HasHandle(request); }));
  server_->Finish(request, FakePlannerServer::Mode::kExhaustive);
  ASSERT_TRUE(WaitFor([this] { return !ExplorationNodeTestPeer::ActiveTarget(*explorer_); }));
}

TEST_F(ExplorationNodeTest,
       OldEpochNoPathCannotFailOrCompleteTheFreshStationaryBatch) {
  auto now = std::make_shared<std::chrono::steady_clock::time_point>();
  parameters_.steady_now = [now] { return *now; };
  parameters_.stop_before_planning = true;
  auto build_calls = std::make_shared<std::atomic<std::size_t>>(0U);
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [build_calls](const lunar::pure_exploration::TaskRaster&,
                    std::span<const lunar::pure_exploration::FrontierCluster>
                        frontiers) {
        ++*build_calls;
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  auto map = GlobalMap();
  PublishAllInputs(StartTask("stale-no-path"), map);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP";
  }));
  for (const double x : {2.1, 2.2, 2.3}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));

  server_->SetMode(FakePlannerServer::Mode::kDelayed);
  const auto goals_before_replan = server_->Goals().size();
  *now += 30s;
  explorer_->PollExecution();
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP";
  }));
  for (const double x : {2.4, 2.5, 2.6}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this, goals_before_replan] {
    return server_->Goals().size() > goals_before_replan;
  }));
  const std::string stale_request = server_->LatestRequestId();
  ASSERT_TRUE(WaitFor(
      [this, &stale_request] { return server_->HasHandle(stale_request); }));

  map.data[0] = 100;
  map_publisher_->publish(map);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP" &&
           server_->CancelCount() == 1U;
  }));
  const auto old_build_calls = build_calls->load();
  for (const double x : {2.7, 2.8, 2.9}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([build_calls, old_build_calls] {
    return build_calls->load() > old_build_calls;
  }));
  const auto goals_before_stale_terminal = server_->Goals().size();
  EXPECT_EQ(goals_before_stale_terminal, goals_before_replan + 1U);

  server_->Finish(stale_request, FakePlannerServer::Mode::kExhaustive);
  ASSERT_TRUE(WaitFor([this, goals_before_stale_terminal] {
    return server_->Goals().size() > goals_before_stale_terminal;
  }));
  const auto after_stale = LatestStatus();
  ASSERT_TRUE(after_stale.has_value());
  EXPECT_NE(after_stale->state, Status::COMPLETED);
  EXPECT_EQ(after_stale->failed_candidate_count, 0U);
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveTarget(*explorer_));

  const std::string current_request = server_->LatestRequestId();
  ASSERT_NE(current_request, stale_request);
  ASSERT_TRUE(WaitFor(
      [this, &current_request] { return server_->HasHandle(current_request); }));
  server_->Finish(current_request, FakePlannerServer::Mode::kExhaustive);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->state == Status::COMPLETED &&
           status->reason_code == "COMPLETED_NO_REACHABLE_FRONTIER";
  }));
  EXPECT_EQ(LatestStatus()->failed_candidate_count, 0U);
}

TEST_F(ExplorationNodeTest,
       OldPlannerTerminalBeforeMapStopCannotDiscardTheActiveCycle) {
  auto now = std::make_shared<std::chrono::steady_clock::time_point>();
  parameters_.steady_now = [now] { return *now; };
  parameters_.stop_before_planning = true;
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster&,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  auto map = GlobalMap();
  PublishAllInputs(StartTask("terminal-before-stop"), map);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP";
  }));
  for (const double x : {2.1, 2.2, 2.3}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));

  server_->SetMode(FakePlannerServer::Mode::kDelayed);
  const auto goals_before_replan = server_->Goals().size();
  *now += 30s;
  explorer_->PollExecution();
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP";
  }));
  for (const double x : {2.4, 2.5, 2.6}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this, goals_before_replan] {
    return server_->Goals().size() > goals_before_replan;
  }));
  const std::string old_request = server_->LatestRequestId();
  ASSERT_TRUE(WaitFor(
      [this, &old_request] { return server_->HasHandle(old_request); }));
  const auto active_cycle = ExplorationNodeTestPeer::ActiveCycle(*explorer_);
  ASSERT_NE(active_cycle, nullptr);

  map.data[0] = 100;
  map_publisher_->publish(map);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->reason_code == "WAITING_FOR_STOP" &&
           server_->CancelCount() == 1U;
  }));
  const auto goals_before_old_terminal = server_->Goals().size();
  const auto statuses_before_old_terminal = StatusCount();
  server_->Finish(old_request, FakePlannerServer::Mode::kExhaustive);
  ASSERT_TRUE(WaitFor([this, statuses_before_old_terminal] {
    return StatusCount() > statuses_before_old_terminal;
  }));
  const auto waiting_after_terminal = LatestStatus();
  ASSERT_TRUE(waiting_after_terminal.has_value());
  EXPECT_EQ(waiting_after_terminal->reason_code, "WAITING_FOR_STOP");
  EXPECT_EQ(ExplorationNodeTestPeer::ActiveCycle(*explorer_), active_cycle);
  EXPECT_TRUE(ExplorationNodeTestPeer::ActiveTarget(*explorer_));
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
  EXPECT_EQ(server_->Goals().size(), goals_before_old_terminal);
  EXPECT_EQ(LatestStatus()->failed_candidate_count, 0U);

  for (const double x : {2.7, 2.8, 2.9}) {
    odometry_publisher_->publish(Odometry(x, 2.0));
    ASSERT_TRUE(WaitFor([this, x] {
      const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
      return pose && pose->x == x;
    }));
  }
  ASSERT_TRUE(WaitFor([this, goals_before_old_terminal] {
    return server_->Goals().size() > goals_before_old_terminal;
  }));
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
  EXPECT_NE(server_->LatestRequestId(), old_request);
  EXPECT_EQ(LatestStatus()->failed_candidate_count, 0U);
}

TEST_F(ExplorationNodeTest, ExecutionReplanContractErrorFailsClosed) {
  auto now = std::make_shared<std::chrono::steady_clock::time_point>();
  parameters_.steady_now = [now] { return *now; };
  Start(FakePlannerServer::Mode::kReachable); PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  server_->SetMode(FakePlannerServer::Mode::kDelayed); const auto before = server_->Goals().size();
  *now += 30s; explorer_->PollExecution();
  ASSERT_TRUE(WaitFor([this, before] { return server_->Goals().size() > before; }));
  const auto request = server_->LatestRequestId(); ASSERT_TRUE(WaitFor([this, &request] { return server_->HasHandle(request); }));
  server_->Finish(request, FakePlannerServer::Mode::kContractError);
  ASSERT_TRUE(WaitFor([this] { const auto s = LatestStatus(); return s && s->state == Status::ERROR; }));
}

TEST_F(ExplorationNodeTest, ExecutionReplanResourceErrorFailsClosed) {
  auto now = std::make_shared<std::chrono::steady_clock::time_point>();
  parameters_.steady_now = [now] { return *now; };
  Start(FakePlannerServer::Mode::kReachable); PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  server_->SetMode(FakePlannerServer::Mode::kDelayed); const auto before = server_->Goals().size();
  *now += 30s; explorer_->PollExecution();
  ASSERT_TRUE(WaitFor([this, before] { return server_->Goals().size() > before; }));
  const auto request = server_->LatestRequestId(); ASSERT_TRUE(WaitFor([this, &request] { return server_->HasHandle(request); }));
  server_->Finish(request, FakePlannerServer::Mode::kResourceError);
  ASSERT_TRUE(WaitFor([this] { const auto s = LatestStatus(); return s && s->state == Status::ERROR; }));
}

TEST_P(ExecutionReplanMapGateTest,
       AppliesOnlyContentChangesAfterThePlannerTerminalGate) {
  auto now = std::make_shared<std::chrono::steady_clock::time_point>();
  parameters_.steady_now = [now] { return *now; };
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster&,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  auto original_map = GlobalMap();
  PublishAllInputs(StartTask("map-gate"), original_map);
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const auto original_target = ExplorationNodeTestPeer::ActiveTarget(*explorer_);
  ASSERT_TRUE(original_target.has_value());
  const auto original_plan_id = References().front().plan_id;
  const auto original_goal_count = server_->Goals().size();
  const auto status_count_before_update = StatusCount();

  if (GetParam() == ExecutionMapUpdate::kStampOnly) {
    original_map.header.stamp.sec = 123;
    map_publisher_->publish(original_map);
    ASSERT_TRUE(WaitFor([this, status_count_before_update] {
      return StatusCount() > status_count_before_update;
    }));
    EXPECT_TRUE(ExecutionCancels().empty());
    EXPECT_EQ(server_->CancelCount(), 0U);
    EXPECT_EQ(server_->Goals().size(), original_goal_count);
    EXPECT_EQ(ReferenceCount(), 1U);
    const auto retained = ExplorationNodeTestPeer::ActiveTarget(*explorer_);
    ASSERT_TRUE(retained.has_value());
    EXPECT_DOUBLE_EQ(retained->x, original_target->x);
    EXPECT_DOUBLE_EQ(retained->y, original_target->y);
    EXPECT_DOUBLE_EQ(retained->yaw, original_target->yaw);
    return;
  }
  if (GetParam() == ExecutionMapUpdate::kOdometryCovarianceOnly) {
    auto covariance_only = Odometry();
    covariance_only.pose.covariance[0] = 999999.0;
    covariance_only.pose.covariance[7] = 999999.0;
    covariance_only.pose.covariance[35] = 999999.0;
    odometry_publisher_->publish(covariance_only);
    ASSERT_TRUE(WaitFor([this, status_count_before_update] {
      return StatusCount() > status_count_before_update;
    }));
    EXPECT_TRUE(ExecutionCancels().empty());
    EXPECT_EQ(server_->CancelCount(), 0U);
    EXPECT_EQ(server_->Goals().size(), original_goal_count);
    EXPECT_EQ(ReferenceCount(), 1U);
    const auto retained = ExplorationNodeTestPeer::ActiveTarget(*explorer_);
    ASSERT_TRUE(retained.has_value());
    EXPECT_DOUBLE_EQ(retained->x, original_target->x);
    EXPECT_DOUBLE_EQ(retained->y, original_target->y);
    EXPECT_DOUBLE_EQ(retained->yaw, original_target->yaw);
    return;
  }

  server_->SetMode(FakePlannerServer::Mode::kDelayed);
  *now += 30s;
  explorer_->PollExecution();
  ASSERT_TRUE(WaitFor([this, original_goal_count] {
    return server_->Goals().size() > original_goal_count;
  }));
  const std::string stale_request = server_->LatestRequestId();
  ASSERT_TRUE(WaitFor(
      [this, &stale_request] { return server_->HasHandle(stale_request); }));
  ASSERT_EQ(ExecutionCancels(), std::vector<std::string>{original_plan_id});

  auto changed_map = original_map;
  if (GetParam() == ExecutionMapUpdate::kGeometry) {
    changed_map.info.resolution = 0.25F;
  } else {
    changed_map.data[0] = 100;
  }
  map_publisher_->publish(changed_map);
  ASSERT_TRUE(WaitFor([this] { return server_->CancelCount() == 1U; }));
  const auto goals_before_terminal = server_->Goals().size();
  std::this_thread::sleep_for(100ms);
  EXPECT_EQ(server_->Goals().size(), goals_before_terminal);
  EXPECT_EQ(ReferenceCount(), 1U);

  server_->FinishCanceled(stale_request);
  ASSERT_TRUE(WaitFor([this, goals_before_terminal] {
    return server_->Goals().size() > goals_before_terminal;
  }));
  const auto goals = server_->Goals();
  EXPECT_FALSE(goals.back().replace_active_request);
  EXPECT_NE(goals.back().request_id, stale_request);
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveTarget(*explorer_));
  EXPECT_EQ(ReferenceCount(), 1U);
  EXPECT_EQ(ExecutionCancels(), std::vector<std::string>{original_plan_id});
  const auto rebuilt_cycle = ExplorationNodeTestPeer::ActiveCycle(*explorer_);
  ASSERT_NE(rebuilt_cycle, nullptr);
  EXPECT_TRUE(rebuilt_cycle->batch()->GlobalMapContentEquals(changed_map));

  server_->SetMode(FakePlannerServer::Mode::kReachable);
  const std::string rebuilt_request = server_->LatestRequestId();
  ASSERT_TRUE(WaitFor(
      [this, &rebuilt_request] { return server_->HasHandle(rebuilt_request); }));
  server_->Finish(rebuilt_request, FakePlannerServer::Mode::kReachable);
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 2U; }));
  const std::string rebuilt_plan_id = References().back().plan_id;
  ASSERT_NE(rebuilt_plan_id, original_plan_id);
  const auto rebuilt_reference =
      ExplorationNodeTestPeer::ActiveReference(*explorer_);
  ASSERT_TRUE(rebuilt_reference.has_value());
  EXPECT_EQ(rebuilt_reference->plan_id, rebuilt_plan_id);
  const auto rebuilt_target = ExplorationNodeTestPeer::ActiveTarget(*explorer_);
  ASSERT_TRUE(rebuilt_target.has_value());
  EXPECT_DOUBLE_EQ(rebuilt_target->x, 4.5);
  EXPECT_DOUBLE_EQ(rebuilt_target->y, 2.0);
  Task pause;
  pause.command = Task::PAUSE;
  task_publisher_->publish(pause);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status && status->state == Status::PAUSED;
  }));
  ASSERT_TRUE(WaitFor([this] { return ExecutionCancels().size() == 2U; }));
  EXPECT_EQ(ExecutionCancels(),
            (std::vector<std::string>{original_plan_id, rebuilt_plan_id}));
}

INSTANTIATE_TEST_SUITE_P(
    MapAndPoseMatrix, ExecutionReplanMapGateTest,
    ::testing::Values(ExecutionMapUpdate::kStampOnly,
                      ExecutionMapUpdate::kOdometryCovarianceOnly));

TEST_P(ExecutionReplanControlGateTest,
       OldTerminalOnlyClosesGateAndCompletesThePendingTransition) {
  auto now = std::make_shared<std::chrono::steady_clock::time_point>();
  parameters_.steady_now = [now] { return *now; };
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs(StartTask("old-task"));
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const auto old_plan_id = References().front().plan_id;
  const auto goals_before_replan = server_->Goals().size();
  server_->SetMode(FakePlannerServer::Mode::kDelayed);
  *now += 30s;
  explorer_->PollExecution();
  ASSERT_TRUE(WaitFor([this, goals_before_replan] {
    return server_->Goals().size() > goals_before_replan;
  }));
  const std::string stale_request = server_->LatestRequestId();
  ASSERT_TRUE(WaitFor(
      [this, &stale_request] { return server_->HasHandle(stale_request); }));
  ASSERT_TRUE(WaitFor([this, &old_plan_id] {
    return ExecutionCancels() == std::vector<std::string>{old_plan_id};
  }));

  Task control;
  if (GetParam() == ExecutionControl::kPause) {
    control.command = Task::PAUSE;
  } else if (GetParam() == ExecutionControl::kCancel) {
    control.command = Task::CANCEL;
  } else {
    control = StartTask("new-task");
  }
  task_publisher_->publish(control);
  ASSERT_TRUE(WaitFor([this] { return server_->CancelCount() == 1U; }));
  const auto goals_before_terminal = server_->Goals().size();
  const auto before_terminal = LatestStatus();
  ASSERT_TRUE(before_terminal.has_value());
  EXPECT_EQ(before_terminal->task_id, "old-task");
  EXPECT_EQ(before_terminal->state, Status::REPLANNING);
  EXPECT_EQ(server_->Goals().size(), goals_before_terminal);

  server_->FinishCanceled(stale_request);
  if (GetParam() == ExecutionControl::kPause) {
    ASSERT_TRUE(WaitFor([this] {
      const auto status = LatestStatus();
      return status && status->state == Status::PAUSED;
    }));
    EXPECT_EQ(server_->Goals().size(), goals_before_terminal);
    EXPECT_EQ(LatestStatus()->task_id, "old-task");
  } else if (GetParam() == ExecutionControl::kCancel) {
    ASSERT_TRUE(WaitFor([this] {
      const auto status = LatestStatus();
      return status && status->state == Status::IDLE;
    }));
    EXPECT_EQ(server_->Goals().size(), goals_before_terminal);
    EXPECT_TRUE(LatestStatus()->task_id.empty());
  } else {
    ASSERT_TRUE(WaitFor([this, goals_before_terminal] {
      return server_->Goals().size() > goals_before_terminal;
    }));
    const auto goals = server_->Goals();
    EXPECT_EQ(goals.back().mission_id, "new-task");
    EXPECT_NE(goals.back().request_id, stale_request);
    EXPECT_FALSE(goals.back().replace_active_request);
    const auto status = LatestStatus();
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(status->task_id, "new-task");
    EXPECT_EQ(status->state, Status::SELECTING_FRONTIER);
  }
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveTarget(*explorer_));
  EXPECT_EQ(ReferenceCount(), 1U);
  EXPECT_EQ(ExecutionCancels(), std::vector<std::string>{old_plan_id});
}

INSTANTIATE_TEST_SUITE_P(
    PauseCancelAndReplacementStart, ExecutionReplanControlGateTest,
    ::testing::Values(ExecutionControl::kPause, ExecutionControl::kCancel,
                      ExecutionControl::kReplacementStart));

TEST_F(ExplorationNodeTest, LatestCacheNeverChangesInFlightCycleAuthority) {
  Start(FakePlannerServer::Mode::kDelayed);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return server_->HasHandle(); }));
  const auto cycle = ExplorationNodeTestPeer::ActiveCycle(*explorer_);
  ASSERT_NE(cycle, nullptr);
  const auto frozen_candidates = cycle->batch()->candidates().size();
  const auto request_id = server_->LatestRequestId();

  auto changed_map = GlobalMap();
  changed_map.data[0] = 100;
  map_publisher_->publish(changed_map);
  odometry_publisher_->publish(Odometry());
  server_->Finish(request_id, FakePlannerServer::Mode::kReachable);

  ASSERT_TRUE(WaitFor([this] { return server_->Goals().size() >= 2U; }));
  EXPECT_EQ(ExplorationNodeTestPeer::ActiveCycle(*explorer_), cycle);
  EXPECT_DOUBLE_EQ(cycle->frozen_robot_pose().x, 2.0);
  EXPECT_DOUBLE_EQ(cycle->frozen_robot_pose().y, 2.0);
  EXPECT_DOUBLE_EQ(cycle->frozen_resolution_m(), 0.5);
  EXPECT_EQ(cycle->batch()->candidates().size(), frozen_candidates);

  while (ReferenceCount() == 0U) {
    const auto next = server_->LatestRequestId();
    if (!next.empty() && next != request_id) {
      server_->Finish(next, FakePlannerServer::Mode::kReachable);
    }
    if (!WaitFor([this, next] {
          return ReferenceCount() != 0U || server_->LatestRequestId() != next;
        }, 1s)) {
      break;
    }
  }
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  EXPECT_EQ(ExplorationNodeTestPeer::ActiveCycle(*explorer_), cycle);
}

TEST_F(ExplorationNodeTest,
       ActiveCompletionStronglyOwnsCycleAndBatchUntilTerminalReturns) {
  Start(FakePlannerServer::Mode::kDelayed);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return server_->HasHandle(); }));
  auto cycle = ExplorationNodeTestPeer::ActiveCycle(*explorer_);
  ASSERT_NE(cycle, nullptr);
  std::weak_ptr<FrozenPlanningCycle> weak_cycle{cycle};
  std::weak_ptr<FrozenCandidateBatch> weak_batch{cycle->batch()};
  cycle.reset();

  ExplorationNodeTestPeer::ResetActiveCycle(*explorer_);
  EXPECT_FALSE(weak_cycle.expired());
  EXPECT_FALSE(weak_batch.expired());

  server_->Finish(server_->LatestRequestId(),
                  FakePlannerServer::Mode::kExhaustive);
  EXPECT_TRUE(WaitFor([&] {
    return weak_cycle.expired() && weak_batch.expired();
  }));
  EXPECT_EQ(ReferenceCount(), 0U);
}

TEST_F(ExplorationNodeTest, RetryableNeverAdvancesToCompletionOrError) {
  Start(FakePlannerServer::Mode::kRetryable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return !server_->Goals().empty(); }));
  std::this_thread::sleep_for(50ms);
  const auto status = LatestStatus();
  ASSERT_TRUE(status.has_value());
  EXPECT_NE(status->state, Status::COMPLETED);
  EXPECT_NE(status->state, Status::ERROR);
  EXPECT_EQ(ReferenceCount(), 0U);
  EXPECT_EQ(server_->Goals().size(), 1U);
}

TEST_F(ExplorationNodeTest, ExhaustiveNoPathAloneCanCompleteCurrentContent) {
  Start(FakePlannerServer::Mode::kExhaustive);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() && status->state == Status::COMPLETED;
  }));
  const auto status = LatestStatus();
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->reason_code, "COMPLETED_NO_REACHABLE_FRONTIER");
  EXPECT_EQ(ReferenceCount(), 0U);
  EXPECT_EQ(server_->MaximumOutstanding(), 1U);
}

TEST_F(ExplorationNodeTest,
       NoReachableFreeStartWaitsForContentAndNeverProvesCompletion) {
  Start(FakePlannerServer::Mode::kExhaustive);
  auto blocked = GlobalMap();
  std::fill(blocked.data.begin(), blocked.data.end(), 100);
  PublishAllInputs(StartTask(), std::move(blocked));
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() &&
           status->state == Status::SELECTING_FRONTIER;
  }));
  std::this_thread::sleep_for(50ms);
  const auto status = LatestStatus();
  ASSERT_TRUE(status.has_value());
  EXPECT_NE(status->state, Status::COMPLETED);
  EXPECT_NE(status->state, Status::ERROR);
  EXPECT_TRUE(server_->Goals().empty());
  EXPECT_EQ(ReferenceCount(), 0U);
}

TEST_F(ExplorationNodeTest,
       PauseCancelsPlannerAndWaitsForSingleWinnerTerminalBeforeTransition) {
  Start(FakePlannerServer::Mode::kDelayed);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return server_->HasHandle(); }));
  const std::string request_id = server_->LatestRequestId();

  Task pause;
  pause.command = Task::PAUSE;
  pause.boundary.points.push_back(
      geometry_msgs::msg::Point32{}.set__x(
          std::numeric_limits<float>::quiet_NaN()));
  task_publisher_->publish(pause);
  ASSERT_TRUE(WaitFor([this] { return server_->CancelCount() == 1U; }));
  const auto before_terminal = LatestStatus();
  ASSERT_TRUE(before_terminal.has_value());
  EXPECT_NE(before_terminal->state, Status::PAUSED);

  server_->FinishCanceled(request_id);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() && status->state == Status::PAUSED;
  }));
  EXPECT_EQ(server_->Goals().size(), 1U);
}

TEST_F(ExplorationNodeTest,
       ThirtyThreeCandidatesUseSixteenSixteenOneWithoutDuplicates) {
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster&,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        return ControlledCandidates(frontiers, 33U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView& row) {
    return static_cast<double>(row.key.x_mm);
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kDelayed);
  PublishAllInputs();

  std::set<std::size_t> seen;
  for (std::size_t goal_index = 0U; goal_index < 33U; ++goal_index) {
    ASSERT_TRUE(WaitFor([this, goal_index] {
      return server_->Goals().size() > goal_index;
    }));
    const auto goals = server_->Goals();
    const std::string request_id = goals[goal_index].request_id;
    ASSERT_TRUE(WaitFor(
        [this, &request_id] { return server_->HasHandle(request_id); }));
    const auto candidate = ExplorationNodeTestPeer::CandidateIndexForRequest(
        *explorer_, request_id);
    ASSERT_TRUE(candidate.has_value());
    EXPECT_TRUE(seen.insert(*candidate).second);
    if (goal_index == 0U) {
      EXPECT_EQ(ExplorationNodeTestPeer::CoarseCursor(*explorer_), 16U);
    } else if (goal_index == 16U) {
      EXPECT_EQ(ExplorationNodeTestPeer::CoarseCursor(*explorer_), 32U);
    } else if (goal_index == 32U) {
      EXPECT_EQ(ExplorationNodeTestPeer::CoarseCursor(*explorer_), 33U);
    }
    server_->Finish(request_id, FakePlannerServer::Mode::kExhaustive);
  }

  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() && status->state == Status::COMPLETED;
  }));
  EXPECT_EQ(seen.size(), 33U);
  for (std::size_t index = 0U; index < 33U; ++index) {
    EXPECT_TRUE(seen.contains(index));
  }
  EXPECT_EQ(server_->MaximumOutstanding(), 1U);
  EXPECT_EQ(ReferenceCount(), 0U);
}

TEST_F(ExplorationNodeTest,
       LatestMapAndPoseCannotChangeFrozenFinalRankAuthority) {
  struct Capture final {
    std::mutex mutex;
    std::vector<lunar::pure_exploration::CandidateGain> gains;
    lunar::pure_exploration::Pose2 pose{};
    double resolution{0.0};
    std::size_t candidate_count{0U};
    std::size_t planned_count{0U};
  } capture;
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster&,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        return ControlledCandidates(frontiers, 3U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView& row) {
    return 100.0 + static_cast<double>(row.key.x_mm);
  };
  seams->final_rank =
      [&capture](
          std::span<const lunar::pure_exploration::CandidateView> candidates,
          std::span<const lunar::pure_exploration::FrontierCluster>,
          std::span<const lunar::pure_exploration::CandidateGain> gains,
          std::span<const lunar::pure_exploration::PlannedCandidate> planned,
          const lunar::pure_exploration::Pose2 pose,
          std::span<const lunar::pure_exploration::Vec2>,
          const double resolution) {
        {
          std::scoped_lock lock{capture.mutex};
          capture.gains.assign(gains.begin(), gains.end());
          capture.pose = pose;
          capture.resolution = resolution;
          capture.candidate_count = candidates.size();
          capture.planned_count = planned.size();
        }
        if (planned.empty()) {
          return std::vector<lunar::pure_exploration::RankedCandidate>{};
        }
        return std::vector{lunar::pure_exploration::RankedCandidate{
            .candidate_index = planned.front().candidate_index,
            .information_gain_m2 = 1.0,
            .euclidean_distance_m = 1.0,
            .path_length_m = planned.front().path_length_m,
            .heading_change_rad = 0.0,
            .revisit_penalty = 0.0,
            .rank_value = 1.0}};
      };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kDelayed);
  PublishAllInputs();

  ASSERT_TRUE(WaitFor([this] { return server_->Goals().size() == 1U; }));
  const std::string first = server_->Goals().front().request_id;
  ASSERT_TRUE(WaitFor([this, &first] { return server_->HasHandle(first); }));
  server_->Finish(first, FakePlannerServer::Mode::kExhaustive);
  ASSERT_TRUE(WaitFor([this] { return server_->Goals().size() == 2U; }));
  const std::string second = server_->Goals()[1].request_id;
  ASSERT_TRUE(WaitFor([this, &second] { return server_->HasHandle(second); }));

  auto changed_map = GlobalMap();
  changed_map.data[0] = 100;
  map_publisher_->publish(changed_map);
  odometry_publisher_->publish(Odometry());
  ASSERT_TRUE(WaitFor([this] {
    const auto resolution =
        ExplorationNodeTestPeer::LatestMapResolution(*explorer_);
    const auto pose = ExplorationNodeTestPeer::LatestPose(*explorer_);
    return resolution == 0.5 && pose.has_value() && pose->x == 2.0 &&
           pose->y == 2.0;
  }));

  const auto cycle = ExplorationNodeTestPeer::ActiveCycle(*explorer_);
  ASSERT_NE(cycle, nullptr);
  server_->Finish(second, FakePlannerServer::Mode::kReachable);
  ASSERT_TRUE(WaitFor([this] { return server_->Goals().size() == 3U; }));
  const std::string third = server_->Goals()[2].request_id;
  ASSERT_TRUE(WaitFor([this, &third] { return server_->HasHandle(third); }));
  server_->Finish(third, FakePlannerServer::Mode::kExhaustive);
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));

  std::scoped_lock capture_lock{capture.mutex};
  ASSERT_EQ(capture.gains.size(), 3U);
  EXPECT_DOUBLE_EQ(capture.gains[0].information_gain_m2, 101.0);
  EXPECT_DOUBLE_EQ(capture.gains[1].information_gain_m2, 102.0);
  EXPECT_DOUBLE_EQ(capture.gains[2].information_gain_m2, 103.0);
  EXPECT_DOUBLE_EQ(capture.pose.x, 2.0);
  EXPECT_DOUBLE_EQ(capture.pose.y, 2.0);
  EXPECT_DOUBLE_EQ(capture.resolution, 0.5);
  EXPECT_EQ(capture.candidate_count, 3U);
  EXPECT_EQ(capture.planned_count, 1U);
  EXPECT_EQ(ExplorationNodeTestPeer::ActiveCycle(*explorer_), cycle);
  EXPECT_EQ(cycle->coarse_order().size(), 3U);
  EXPECT_EQ(ExplorationNodeTestPeer::CoarseCursor(*explorer_), 3U);
}

TEST_F(ExplorationNodeTest,
       ThrowDuringSecondFinalReferenceCopyFailsClosedBeforeGoalCommit) {
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  auto copy_count = std::make_shared<std::atomic<std::size_t>>(0U);
  seams->copy_reference =
      [copy_count](
          const lunar_planning_msgs::msg::MotionReference& reference) {
        if (copy_count->fetch_add(1U) == 1U) {
          throw std::length_error{"controlled second reference copy"};
        }
        return reference;
      };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();

  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() && status->state == Status::ERROR;
  }));
  const auto status = LatestStatus();
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->reason_code, "RESOURCE_RESULT_COPY");
  EXPECT_TRUE(status->current_plan_id.empty());
  EXPECT_EQ(ReferenceCount(), 0U);
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveReference(*explorer_));
  EXPECT_FALSE(ExplorationNodeTestPeer::ActiveRequestId(*explorer_));
  EXPECT_TRUE(ExplorationNodeTestPeer::ExecutablePolyline(*explorer_).empty());
  EXPECT_FALSE(ExplorationNodeTestPeer::ExecutableEndpoint(*explorer_));
  EXPECT_EQ(copy_count->load(), 2U);
}

TEST_F(ExplorationNodeTest,
       CancelWaitsForPlannerTerminalAndCannotRestoreOldCycle) {
  Start(FakePlannerServer::Mode::kDelayed);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return server_->HasHandle(); }));
  const std::string request_id = server_->LatestRequestId();

  Task cancel;
  cancel.command = Task::CANCEL;
  task_publisher_->publish(cancel);
  ASSERT_TRUE(WaitFor([this] { return server_->CancelCount() == 1U; }));
  const auto before_terminal = LatestStatus();
  ASSERT_TRUE(before_terminal.has_value());
  EXPECT_NE(before_terminal->state, Status::IDLE);

  server_->FinishCanceled(request_id);
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() && status->state == Status::IDLE;
  }));
  EXPECT_EQ(server_->Goals().size(), 1U);
  EXPECT_EQ(ExplorationNodeTestPeer::ActiveCycle(*explorer_), nullptr);
}

TEST_F(ExplorationNodeTest,
       SameTaskReplacementUsesNewRequestIdAfterOldTerminalGate) {
  Start(FakePlannerServer::Mode::kDelayed);
  PublishAllInputs(StartTask("same-task"));
  ASSERT_TRUE(WaitFor([this] { return server_->HasHandle(); }));
  const std::string first = server_->LatestRequestId();
  const auto first_cycle = ExplorationNodeTestPeer::ActiveCycle(*explorer_);
  ASSERT_NE(first_cycle, nullptr);

  task_publisher_->publish(StartTask("same-task"));
  ASSERT_TRUE(WaitFor([this] { return server_->CancelCount() == 1U; }));
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(server_->Goals().size(), 1U);
  server_->FinishCanceled(first);

  ASSERT_TRUE(WaitFor([this] { return server_->Goals().size() == 2U; }));
  const std::string second = server_->Goals()[1].request_id;
  ASSERT_TRUE(WaitFor([this, &second] { return server_->HasHandle(second); }));
  EXPECT_NE(first, second);
  EXPECT_NE(ExplorationNodeTestPeer::ActiveCycle(*explorer_), first_cycle);
  const auto status = LatestStatus();
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->task_id, "same-task");
  EXPECT_EQ(status->state, Status::SELECTING_FRONTIER);
}

TEST_F(ExplorationNodeTest, CancelingPublishedPlanUsesExactPlanId) {
  Start(FakePlannerServer::Mode::kReachable);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  const std::string plan_id = References().front().plan_id;

  Task cancel;
  cancel.command = Task::CANCEL;
  task_publisher_->publish(cancel);
  ASSERT_TRUE(WaitFor([this, &plan_id] {
    const auto cancels = ExecutionCancels();
    return std::ranges::find(cancels, plan_id) != cancels.end();
  }));
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() && status->state == Status::IDLE;
  }));
}

TEST_F(ExplorationNodeTest, UnknownRequestIdFailsClosed) {
  Start(FakePlannerServer::Mode::kDelayed);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return server_->HasHandle(); }));

  ExplorationNodeTestPeer::InjectEvaluation(
      *explorer_, PlannerEvaluation{
                      .request_id = "not-the-active-request",
                      .candidate_id = 0U,
                      .kind = PlannerEvaluationKind::kExhaustiveNoPath,
                      .reason_code = "NO_PATH"});
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() && status->state == Status::ERROR;
  }));
  EXPECT_EQ(LatestStatus()->reason_code, "UNKNOWN_PLANNER_REQUEST_ID");
  EXPECT_EQ(ReferenceCount(), 0U);
}

TEST_F(ExplorationNodeTest,
       ChangedMapContentBeforeExhaustionRebuildsInsteadOfCompleting) {
  auto seams = std::make_shared<ExplorationPipelineSeams>();
  seams->generate_candidates =
      [](const lunar::pure_exploration::TaskRaster&,
         std::span<const lunar::pure_exploration::FrontierCluster> frontiers) {
        return ControlledCandidates(frontiers, 1U);
      };
  seams->evaluate_gain = [](const lunar::pure_exploration::TaskRaster&,
                            const lunar::pure_exploration::CandidateView&) {
    return 1.0;
  };
  parameters_.pipeline_seams = std::move(seams);
  Start(FakePlannerServer::Mode::kDelayed);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return server_->HasHandle(); }));
  const std::string first = server_->LatestRequestId();
  const auto first_cycle = ExplorationNodeTestPeer::ActiveCycle(*explorer_);
  ASSERT_NE(first_cycle, nullptr);

  auto changed_map = GlobalMap(20U, 20U, 0.25);
  changed_map.data[0] = 100;
  map_publisher_->publish(changed_map);
  ASSERT_TRUE(WaitFor([this] {
    return ExplorationNodeTestPeer::LatestMapResolution(*explorer_) == 0.25;
  }));
  server_->Finish(first, FakePlannerServer::Mode::kExhaustive);

  ASSERT_TRUE(WaitFor([this] { return server_->Goals().size() == 2U; }));
  const auto second_cycle = ExplorationNodeTestPeer::ActiveCycle(*explorer_);
  ASSERT_NE(second_cycle, nullptr);
  EXPECT_NE(second_cycle, first_cycle);
  EXPECT_TRUE(second_cycle->batch()->GlobalMapContentEquals(changed_map));
  const auto status = LatestStatus();
  ASSERT_TRUE(status.has_value());
  EXPECT_NE(status->state, Status::COMPLETED);
}

TEST_F(ExplorationNodeTest,
       ExactExecutableAndPreviewLimitsPublishReferences) {
  parameters_.maximum_executable_path_points = 2U;
  parameters_.maximum_path_preview_poses = 2U;
  Start(FakePlannerServer::Mode::kSizedExecutableReference, 2U);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  EXPECT_NE(LatestStatus()->state, Status::ERROR);
}

TEST_F(ExplorationNodeTest, ExactPreviewLimitPublishesReference) {
  parameters_.maximum_path_preview_poses = 2U;
  Start(FakePlannerServer::Mode::kSizedPreviewReference, 1U, 2U);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] { return ReferenceCount() == 1U; }));
  EXPECT_NE(LatestStatus()->state, Status::ERROR);
}

TEST_F(ExplorationNodeTest,
       OversizedPreviewReferenceIsResourceErrorAndNeverCompletion) {
  parameters_.maximum_path_preview_poses = 2U;
  Start(FakePlannerServer::Mode::kSizedPreviewReference, 1U, 3U);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() && status->state == Status::ERROR;
  }));
  EXPECT_EQ(LatestStatus()->reason_code, "RESOURCE_PATH_PREVIEW_LIMIT");
  EXPECT_EQ(ReferenceCount(), 0U);
}

TEST_F(ExplorationNodeTest,
       OversizedExecutableReferenceIsResourceErrorAndNeverCompletion) {
  parameters_.maximum_executable_path_points = 2U;
  Start(FakePlannerServer::Mode::kSizedExecutableReference, 3U);
  PublishAllInputs();
  ASSERT_TRUE(WaitFor([this] {
    const auto status = LatestStatus();
    return status.has_value() && status->state == Status::ERROR;
  }));
  const auto status = LatestStatus();
  ASSERT_TRUE(status.has_value());
  EXPECT_EQ(status->reason_code, "RESOURCE_EXECUTABLE_PATH_LIMIT");
  EXPECT_EQ(ReferenceCount(), 0U);
}

}  // namespace
}  // namespace lunar::pure_exploration_ros
