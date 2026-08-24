#include "lunar_pure_exploration_ros/planner_client.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <action_msgs/msg/goal_status.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <gtest/gtest.h>
#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <variant>

#include "accepted_goal_finalizer.hpp"
#include "lunar_pure_planner_ros/message_conversion.hpp"

namespace lunar::pure_exploration_ros {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using ClientGoalHandle = rclcpp_action::ClientGoalHandle<Action>;
using ServerGoalHandle = rclcpp_action::ServerGoalHandle<Action>;
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

class FakePlannerServer final {
 public:
  FakePlannerServer(std::shared_ptr<rclcpp::Node> node, std::string action_name)
      : node_(std::move(node)), action_name_(std::move(action_name)) {
    server_ = rclcpp_action::create_server<Action>(
        node_, action_name_,
        [this](const rclcpp_action::GoalUUID&,
               const std::shared_ptr<const Action::Goal> goal) {
          std::unique_lock lock{mutex_};
          goals_.insert_or_assign(goal->request_id, *goal);
          goal_callbacks_entered_.insert(goal->request_id);
          condition_.notify_all();
          if (delay_next_goal_response_) {
            delay_next_goal_response_ = false;
            condition_.wait(lock, [this] { return release_goal_responses_; });
          }
          if (reject_next_) {
            reject_next_ = false;
            return rclcpp_action::GoalResponse::REJECT;
          }
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        },
        [this](const std::shared_ptr<ServerGoalHandle> handle) {
          std::unique_lock lock{mutex_};
          ++cancel_counts_[handle->get_goal()->request_id];
          condition_.notify_all();
          if (delay_cancel_response_) {
            condition_.wait(lock, [this] { return release_cancel_responses_; });
          }
          return accept_cancel_ ? rclcpp_action::CancelResponse::ACCEPT
                                : rclcpp_action::CancelResponse::REJECT;
        },
        [this](const std::shared_ptr<ServerGoalHandle> handle) {
          std::scoped_lock lock{mutex_};
          handles_.insert_or_assign(handle->get_goal()->request_id, handle);
        });
  }

  void RejectNext() {
    std::scoped_lock lock{mutex_};
    reject_next_ = true;
  }

  void DelayNextGoalResponse() {
    std::scoped_lock lock{mutex_};
    delay_next_goal_response_ = true;
    release_goal_responses_ = false;
  }

  void ReleaseGoalResponses() {
    std::scoped_lock lock{mutex_};
    release_goal_responses_ = true;
    condition_.notify_all();
  }

  bool GoalCallbackEntered(const std::string& request_id) const {
    std::scoped_lock lock{mutex_};
    return goal_callbacks_entered_.contains(request_id);
  }

  void DelayCancelResponse() {
    std::scoped_lock lock{mutex_};
    delay_cancel_response_ = true;
    release_cancel_responses_ = false;
  }

  void ReleaseCancelResponses() {
    std::scoped_lock lock{mutex_};
    delay_cancel_response_ = false;
    release_cancel_responses_ = true;
    condition_.notify_all();
  }

  void SetAcceptCancel(const bool accept) {
    std::scoped_lock lock{mutex_};
    accept_cancel_ = accept;
  }

  bool HasGoal(const std::string& request_id) const {
    std::scoped_lock lock{mutex_};
    return goals_.contains(request_id);
  }

  bool HasHandle(const std::string& request_id) const {
    std::scoped_lock lock{mutex_};
    return handles_.contains(request_id);
  }

  Action::Goal Goal(const std::string& request_id) const {
    std::scoped_lock lock{mutex_};
    return goals_.at(request_id);
  }

  std::uint64_t CancelCount(const std::string& request_id) const {
    std::scoped_lock lock{mutex_};
    const auto found = cancel_counts_.find(request_id);
    return found == cancel_counts_.end() ? 0U : found->second;
  }

  void Finish(const std::string& request_id,
              const rclcpp_action::ResultCode code,
              const Action::Result& result) {
    std::shared_ptr<ServerGoalHandle> handle;
    {
      std::scoped_lock lock{mutex_};
      handle = handles_.at(request_id);
    }
    const auto shared_result = std::make_shared<Action::Result>(result);
    switch (code) {
      case rclcpp_action::ResultCode::SUCCEEDED:
        handle->succeed(shared_result);
        return;
      case rclcpp_action::ResultCode::ABORTED:
        handle->abort(shared_result);
        return;
      case rclcpp_action::ResultCode::CANCELED:
        handle->canceled(shared_result);
        return;
      default:
        throw std::invalid_argument{"unsupported fake terminal"};
    }
  }

  void PublishFeedback(const std::string& request_id) {
    std::shared_ptr<ServerGoalHandle> handle;
    {
      std::scoped_lock lock{mutex_};
      handle = handles_.at(request_id);
    }
    auto feedback = std::make_shared<Action::Feedback>();
    feedback->phase = Action::Feedback::SEARCHING;
    feedback->elapsed_s = 999.0;
    feedback->expanded_states = std::numeric_limits<std::uint64_t>::max();
    feedback->has_best_cost = true;
    feedback->best_cost = -std::numeric_limits<double>::infinity();
    handle->publish_feedback(feedback);
  }

 private:
  std::shared_ptr<rclcpp::Node> node_;
  std::string action_name_;
  rclcpp_action::Server<Action>::SharedPtr server_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool reject_next_{false};
  bool accept_cancel_{true};
  bool delay_next_goal_response_{false};
  bool release_goal_responses_{true};
  bool delay_cancel_response_{false};
  bool release_cancel_responses_{true};
  std::unordered_map<std::string, Action::Goal> goals_;
  std::unordered_set<std::string> goal_callbacks_entered_;
  std::unordered_map<std::string, std::shared_ptr<ServerGoalHandle>> handles_;
  std::unordered_map<std::string, std::uint64_t> cancel_counts_;
};

lunar::pure_exploration::CandidateView Candidate(
    const std::uint64_t id = 41U, const double x = 1.25,
    const double y = -2.5, const double yaw = 0.75) {
  return {.id = id,
          .frontier_id = 9U,
          .frontier_index = 0U,
          .key = {.x_mm = 1250, .y_mm = -2500, .yaw_tenth_deg = 430},
          .pose = {.x = x, .y = y, .yaw = yaw},
          .frontier_distance_m = 2.0};
}

Action::Result FailureResult(const std::uint8_t outcome,
                             std::string reason_code) {
  Action::Result result;
  result.planning_outcome = outcome;
  result.execution_directive = Action::Result::NO_SAFE_REFERENCE;
  result.reason_code = std::move(reason_code);
  result.has_reference = false;
  return result;
}

lunar_planning_msgs::msg::MotionReference ReferenceWithPath(
    const std::vector<std::pair<double, double>>& points) {
  lunar_planning_msgs::msg::MotionReference reference;
  reference.header.frame_id = "map";
  reference.header.stamp.sec = 31;
  reference.header.stamp.nanosec = 17U;
  reference.plan_id = "preserve-complete-reference";
  reference.platform_type = reference.HOPPER;
  reference.input_time.sec = 29;
  reference.input_time.nanosec = 19U;
  reference.path_preview.header.frame_id = "map";
  reference.path_preview.header.stamp.sec = 23;
  reference.path_preview.header.stamp.nanosec = 21U;
  for (std::size_t index = 0U; index < points.size(); ++index) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = "stamp-is-ignored";
    pose.header.stamp.sec = static_cast<std::int32_t>(100U - index);
    pose.pose.position.x = points[index].first;
    pose.pose.position.y = points[index].second;
    pose.pose.position.z = std::numeric_limits<double>::quiet_NaN();
    pose.pose.orientation.x = std::numeric_limits<double>::infinity();
    reference.path_preview.poses.push_back(std::move(pose));
  }
  reference.trajectory.header.frame_id = "trajectory-frame-preserved";
  reference.trajectory.header.stamp.sec = 37;
  reference.trajectory.joint_names = {"base_link", "payload"};
  lunar_planning_msgs::msg::HopSegment hop;
  hop.header.frame_id = "hop-frame-preserved";
  hop.header.stamp.sec = 41;
  hop.segment_id = "hop-1";
  hop.flight_time.sec = 3;
  hop.flight_tube_radius_m = 0.8;
  hop.capability_version = "frozen-v2";
  hop.global_map_generation = 11U;
  hop.local_map_generation = 12U;
  reference.hops.push_back(std::move(hop));
  return reference;
}

Action::Result SuccessResult(
    const std::vector<std::pair<double, double>>& points) {
  Action::Result result;
  result.planning_outcome = Action::Result::NEW_REFERENCE_AVAILABLE;
  result.execution_directive = Action::Result::ACTIVATE_NEW_REFERENCE;
  result.reason_code = "PLAN_FOUND";
  result.has_reference = true;
  result.reference = ReferenceWithPath(points);
  result.global_map_stamp.sec = 71;
  result.local_map_stamp.sec = 61;
  result.state_stamp.sec = 51;
  return result;
}

std::vector<std::uint8_t> SerializedReference(
    const lunar_planning_msgs::msg::MotionReference& reference) {
  rclcpp::Serialization<lunar_planning_msgs::msg::MotionReference>
      serialization;
  rclcpp::SerializedMessage serialized;
  serialization.serialize_message(&reference, &serialized);
  const auto& message = serialized.get_rcl_serialized_message();
  return {message.buffer, message.buffer + message.buffer_length};
}

class PlannerClientTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto suffix = sequence_.fetch_add(1U);
    action_name_ = "/planner_client_test_" + std::to_string(suffix);
    server_node_ = std::make_shared<rclcpp::Node>(
        "planner_client_fake_server_" + std::to_string(suffix));
    client_node_ = std::make_shared<rclcpp::Node>(
        "planner_client_under_test_" + std::to_string(suffix));
    server_ = std::make_unique<FakePlannerServer>(server_node_, action_name_);
    executor_ = std::make_unique<rclcpp::executors::MultiThreadedExecutor>(
        rclcpp::ExecutorOptions{}, 4U);
    executor_->add_node(server_node_);
    executor_->add_node(client_node_);
    spin_thread_ = std::jthread([this] { executor_->spin(); });
    probe_ = rclcpp_action::create_client<Action>(client_node_, action_name_);
    ASSERT_TRUE(probe_->wait_for_action_server(3s));
    client_ = std::make_unique<PlannerClient>(
        *client_node_, action_name_,
        PlannerClientParameters{.maximum_path_preview_poses = 4U,
                                .maximum_executable_path_points = 3U,
                                .goal_response_timeout = 1s,
                                .result_timeout = 3500ms},
        [this] {
          return std::chrono::steady_clock::time_point{
              std::chrono::nanoseconds{now_ns_.load()}};
        });
  }

  void TearDown() override {
    client_.reset();
    executor_->cancel();
    if (spin_thread_.joinable()) {
      spin_thread_.join();
    }
    executor_->remove_node(client_node_);
    executor_->remove_node(server_node_);
    probe_.reset();
    server_.reset();
    client_node_.reset();
    server_node_.reset();
  }

  void Evaluate(const std::string& request_id,
                const lunar::pure_exploration::CandidateView& candidate) {
    client_->Evaluate(
        "task-alpha", request_id, candidate, 0.25, 0.125,
        [this](PlannerEvaluation evaluation) {
          std::scoped_lock lock{completion_mutex_};
          completions_.push_back(std::move(evaluation));
        });
  }

  PlannerEvaluation FinishAndWait(const std::string& request_id,
                                  const rclcpp_action::ResultCode code,
                                  const Action::Result& result,
                                  const bool request_cancel = false) {
    if (!WaitFor([&] { return server_->HasHandle(request_id); })) {
      throw std::runtime_error{"fake server did not accept " + request_id};
    }
    if (request_cancel) {
      client_->CancelActive();
      if (!WaitFor([&] { return server_->CancelCount(request_id) == 1U; })) {
        throw std::runtime_error{"fake server did not receive cancel " +
                                 request_id};
      }
    }
    const std::size_t before = CompletionCount();
    server_->Finish(request_id, code, result);
    EXPECT_TRUE(WaitFor([&] { return CompletionCount() == before + 1U; }));
    std::scoped_lock lock{completion_mutex_};
    return completions_.back();
  }

  std::size_t CompletionCount() const {
    std::scoped_lock lock{completion_mutex_};
    return completions_.size();
  }

  static std::atomic<std::uint64_t> sequence_;
  std::string action_name_;
  std::shared_ptr<rclcpp::Node> server_node_;
  std::shared_ptr<rclcpp::Node> client_node_;
  std::unique_ptr<FakePlannerServer> server_;
  std::unique_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
  std::jthread spin_thread_;
  rclcpp_action::Client<Action>::SharedPtr probe_;
  std::unique_ptr<PlannerClient> client_;
  std::atomic<std::int64_t> now_ns_{0};
  mutable std::mutex completion_mutex_;
  std::vector<PlannerEvaluation> completions_;
};

std::atomic<std::uint64_t> PlannerClientTest::sequence_{0U};

TEST_F(PlannerClientTest, BuildsExactCanonicalGoalAndCorrelatesOnlyByRequestId) {
  EXPECT_EQ(PlannerClient::MakeRequestId("task-alpha", 0U),
            "task-alpha/candidate/0");
  EXPECT_EQ(PlannerClient::MakeRequestId("task-alpha", 42U),
            "task-alpha/candidate/42");

  auto first_candidate = Candidate(77U, 1.25, -2.5, 0.75);
  Evaluate("task-alpha/candidate/0", first_candidate);
  first_candidate.id = 999U;
  first_candidate.pose = {.x = 100.0, .y = 200.0, .yaw = 3.0};
  ASSERT_TRUE(WaitFor(
      [&] { return server_->HasGoal("task-alpha/candidate/0"); }));
  const auto goal = server_->Goal("task-alpha/candidate/0");
  EXPECT_EQ(goal.request_id, "task-alpha/candidate/0");
  EXPECT_EQ(goal.mission_id, "task-alpha");
  EXPECT_EQ(goal.mission_revision, 0U);
  EXPECT_EQ(goal.environment_mode, Action::Goal::LUNAR_SURFACE);
  EXPECT_FALSE(goal.replace_active_request);
  EXPECT_EQ(goal.goal.header.frame_id, "map");
  EXPECT_EQ(goal.goal.header.stamp.sec, 0);
  EXPECT_EQ(goal.goal.header.stamp.nanosec, 0U);
  EXPECT_EQ(goal.goal.goal_id, "task-alpha/candidate/0/goal");
  EXPECT_EQ(goal.goal.goal_type, goal.goal.POINT);
  EXPECT_DOUBLE_EQ(goal.goal.point.x, 1.25);
  EXPECT_DOUBLE_EQ(goal.goal.point.y, -2.5);
  EXPECT_DOUBLE_EQ(goal.goal.point.z, 0.0);
  EXPECT_FALSE(std::signbit(goal.goal.point.z));
  EXPECT_TRUE(goal.goal.planar_region.points.empty());
  EXPECT_DOUBLE_EQ(goal.goal.position_tolerance_m, 0.25);
  EXPECT_TRUE(goal.goal.has_yaw_constraint);
  EXPECT_DOUBLE_EQ(goal.goal.yaw_rad, 0.75);
  EXPECT_DOUBLE_EQ(goal.goal.yaw_tolerance_rad, 0.125);

  const auto first = FinishAndWait(
      "task-alpha/candidate/0", rclcpp_action::ResultCode::ABORTED,
      FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH"));
  EXPECT_EQ(first.request_id, "task-alpha/candidate/0");
  EXPECT_EQ(first.candidate_id, 77U);

  Evaluate("task-alpha/candidate/1", Candidate(77U, 3.0, 4.0, -0.5));
  const auto second = FinishAndWait(
      "task-alpha/candidate/1", rclcpp_action::ResultCode::SUCCEEDED,
      SuccessResult({{3.0, 4.0}}));
  EXPECT_EQ(second.request_id, "task-alpha/candidate/1");
  EXPECT_EQ(second.candidate_id, 77U);
  EXPECT_NE(first.request_id, second.request_id);
}

TEST_F(PlannerClientTest, ClassifiesTheOnlyFourAcceptedTypedCombinations) {
  Evaluate("reachable", Candidate());
  const auto reference = ReferenceWithPath({{0.0, 0.0}, {3.0, 4.0}, {6.0, 8.0}});
  auto success = SuccessResult({{0.0, 0.0}, {3.0, 4.0}, {6.0, 8.0}});
  success.reference = reference;
  const auto reachable = FinishAndWait(
      "reachable", rclcpp_action::ResultCode::SUCCEEDED, success);
  EXPECT_EQ(reachable.kind, PlannerEvaluationKind::kReachable);
  EXPECT_EQ(reachable.reason_code, "PLAN_FOUND");
  ASSERT_TRUE(reachable.path_length_m.has_value());
  EXPECT_DOUBLE_EQ(*reachable.path_length_m, 10.0);
  ASSERT_TRUE(reachable.reference.has_value());
  EXPECT_EQ(SerializedReference(*reachable.reference),
            SerializedReference(reference));

  for (const auto& [id, reason] :
       {std::pair{"no-path", "NO_PATH"},
        std::pair{"outside", "GOAL_OUTSIDE_LOCAL_MAP"}}) {
    Evaluate(id, Candidate());
    const auto evaluation = FinishAndWait(
        id, rclcpp_action::ResultCode::ABORTED,
        FailureResult(Action::Result::GOAL_INFEASIBLE, reason));
    EXPECT_EQ(evaluation.kind, PlannerEvaluationKind::kExhaustiveNoPath);
    EXPECT_EQ(evaluation.reason_code, reason);
    EXPECT_FALSE(evaluation.path_length_m.has_value());
    EXPECT_FALSE(evaluation.reference.has_value());
  }

  Evaluate("timeout", Candidate());
  const auto timeout = FinishAndWait(
      "timeout", rclcpp_action::ResultCode::ABORTED,
      FailureResult(Action::Result::RESOURCE_EXHAUSTED, "TIMEOUT"));
  EXPECT_EQ(timeout.kind, PlannerEvaluationKind::kRetryable);
  EXPECT_EQ(timeout.reason_code, "TIMEOUT");
  EXPECT_FALSE(timeout.path_length_m.has_value());
  EXPECT_FALSE(timeout.reference.has_value());

  Evaluate("canceled", Candidate());
  const auto canceled = FinishAndWait(
      "canceled", rclcpp_action::ResultCode::CANCELED,
      FailureResult(Action::Result::CANCELED, "REQUEST_CANCELED"), true);
  EXPECT_EQ(canceled.kind, PlannerEvaluationKind::kCanceled);
  EXPECT_EQ(canceled.reason_code, "REQUEST_CANCELED");
  EXPECT_FALSE(canceled.path_length_m.has_value());
  EXPECT_FALSE(canceled.reference.has_value());
}

TEST_F(PlannerClientTest,
       RejectsEveryOutcomeDirectiveWrapperReasonAndReferenceContradiction) {
  struct ContractCase final {
    std::string id;
    rclcpp_action::ResultCode wrapper;
    Action::Result result;
  };
  std::vector<ContractCase> cases;
  for (const std::uint8_t outcome : {
           Action::Result::SAFE_FRONTIER_REFERENCE_AVAILABLE,
           Action::Result::NO_KNOWN_SAFE_ROUTE,
           Action::Result::INVALID_REQUEST,
           Action::Result::STALE_INPUT,
           Action::Result::NUMERICAL_FAILURE,
           Action::Result::ACTIVE_REFERENCE_INVALIDATED,
           static_cast<std::uint8_t>(255U)}) {
    cases.push_back({"outcome-" + std::to_string(outcome),
                     rclcpp_action::ResultCode::ABORTED,
                     FailureResult(outcome, "UNSUPPORTED")});
  }
  cases.push_back({"goal-reason", rclcpp_action::ResultCode::ABORTED,
                   FailureResult(Action::Result::GOAL_INFEASIBLE, "OTHER")});
  cases.push_back({"resource-reason", rclcpp_action::ResultCode::ABORTED,
                   FailureResult(Action::Result::RESOURCE_EXHAUSTED,
                                 "MEMORY")});
  cases.push_back({"empty-reason", rclcpp_action::ResultCode::ABORTED,
                   FailureResult(Action::Result::GOAL_INFEASIBLE, "")});
  cases.push_back({"success-aborted", rclcpp_action::ResultCode::ABORTED,
                   SuccessResult({{0.0, 0.0}})});
  cases.push_back({"failure-succeeded", rclcpp_action::ResultCode::SUCCEEDED,
                   FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH")});
  auto wrong_directive = FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH");
  wrong_directive.execution_directive = Action::Result::HOLD_POSITION;
  cases.push_back({"wrong-directive", rclcpp_action::ResultCode::ABORTED,
                   wrong_directive});
  auto unknown_directive = FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH");
  unknown_directive.execution_directive = 255U;
  cases.push_back({"unknown-directive", rclcpp_action::ResultCode::ABORTED,
                   unknown_directive});
  auto false_success = SuccessResult({{0.0, 0.0}});
  false_success.has_reference = false;
  cases.push_back({"success-without-reference", rclcpp_action::ResultCode::SUCCEEDED,
                   false_success});
  auto true_failure = FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH");
  true_failure.has_reference = true;
  cases.push_back({"failure-with-reference-flag", rclcpp_action::ResultCode::ABORTED,
                   true_failure});
  auto payload_failure = FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH");
  payload_failure.reference = ReferenceWithPath({{0.0, 0.0}});
  cases.push_back({"failure-with-reference-payload",
                   rclcpp_action::ResultCode::ABORTED, payload_failure});

  for (auto& contract_case : cases) {
    SCOPED_TRACE(contract_case.id);
    Evaluate(contract_case.id, Candidate());
    const auto evaluation = FinishAndWait(
        contract_case.id, contract_case.wrapper, contract_case.result);
    EXPECT_EQ(evaluation.kind, PlannerEvaluationKind::kContractError);
    EXPECT_FALSE(evaluation.path_length_m.has_value());
    EXPECT_FALSE(evaluation.reference.has_value());
  }
}

TEST_F(PlannerClientTest, AdmitsOnlyBoundedFiniteXyPathAndPreservesIgnoredFields) {
  Evaluate("one-pose", Candidate());
  const auto one_pose_result = SuccessResult({{2.0, -3.0}});
  const auto one_pose = FinishAndWait(
      "one-pose", rclcpp_action::ResultCode::SUCCEEDED, one_pose_result);
  ASSERT_EQ(one_pose.kind, PlannerEvaluationKind::kReachable);
  ASSERT_TRUE(one_pose.path_length_m.has_value());
  EXPECT_DOUBLE_EQ(*one_pose.path_length_m, 0.0);
  EXPECT_FALSE(std::signbit(*one_pose.path_length_m));
  ASSERT_TRUE(one_pose.reference.has_value());
  EXPECT_EQ(SerializedReference(*one_pose.reference),
            SerializedReference(one_pose_result.reference));

  std::vector<std::pair<std::string, Action::Result>> invalid;
  invalid.emplace_back("empty-path", SuccessResult({}));
  invalid.emplace_back("nonfinite-x",
                       SuccessResult({{0, 0},
                                      {std::numeric_limits<double>::quiet_NaN(),
                                       1}}));
  invalid.emplace_back("nonfinite-y",
                       SuccessResult({{0, 0},
                                      {1, std::numeric_limits<double>::infinity()}}));
  invalid.emplace_back(
      "double-overflow",
      SuccessResult({{-std::numeric_limits<double>::max(), 0},
                     {std::numeric_limits<double>::max(), 0}}));
  for (auto& [id, result] : invalid) {
    SCOPED_TRACE(id);
    Evaluate(id, Candidate());
    const auto evaluation = FinishAndWait(
        id, rclcpp_action::ResultCode::SUCCEEDED, result);
    EXPECT_EQ(evaluation.kind, PlannerEvaluationKind::kContractError);
    EXPECT_FALSE(evaluation.path_length_m.has_value());
    EXPECT_FALSE(evaluation.reference.has_value());
  }
}

TEST_F(PlannerClientTest,
       ClassifiesPreviewAndExecutableExactLimitsBeforeReferenceCopy) {
  Evaluate("preview-exact", Candidate());
  const auto preview_exact = FinishAndWait(
      "preview-exact", rclcpp_action::ResultCode::SUCCEEDED,
      SuccessResult({{0, 0}, {1, 0}, {2, 0}, {3, 0}}));
  EXPECT_EQ(preview_exact.kind, PlannerEvaluationKind::kReachable);
  ASSERT_TRUE(preview_exact.reference.has_value());
  EXPECT_EQ(preview_exact.reference->path_preview.poses.size(), 4U);

  Evaluate("preview-over", Candidate());
  const auto preview_over = FinishAndWait(
      "preview-over", rclcpp_action::ResultCode::SUCCEEDED,
      SuccessResult({{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}}));
  EXPECT_EQ(preview_over.kind, PlannerEvaluationKind::kResourceError);
  EXPECT_EQ(preview_over.resource_kind,
            PlannerResourceKind::kPathPreviewPoses);
  EXPECT_EQ(preview_over.reason_code, "RESOURCE_PATH_PREVIEW_LIMIT");
  EXPECT_FALSE(preview_over.reference.has_value());

  for (const auto platform : {
           lunar_planning_msgs::msg::MotionReference::WHEELED,
           lunar_planning_msgs::msg::MotionReference::LEGGED}) {
    const std::string prefix = platform ==
                                       lunar_planning_msgs::msg::MotionReference::WHEELED
                                   ? "wheel"
                                   : "legged";
    auto exact = SuccessResult({{0, 0}});
    exact.reference.platform_type = platform;
    exact.reference.hops.clear();
    exact.reference.trajectory.points.resize(3U);
    Evaluate(prefix + "-exact", Candidate());
    const auto exact_evaluation = FinishAndWait(
        prefix + "-exact", rclcpp_action::ResultCode::SUCCEEDED, exact);
    EXPECT_EQ(exact_evaluation.kind, PlannerEvaluationKind::kReachable);
    ASSERT_TRUE(exact_evaluation.reference.has_value());
    EXPECT_EQ(exact_evaluation.reference->trajectory.points.size(), 3U);

    exact.reference.trajectory.points.resize(4U);
    Evaluate(prefix + "-over", Candidate());
    const auto over_evaluation = FinishAndWait(
        prefix + "-over", rclcpp_action::ResultCode::SUCCEEDED, exact);
    EXPECT_EQ(over_evaluation.kind, PlannerEvaluationKind::kResourceError);
    EXPECT_EQ(over_evaluation.resource_kind,
              PlannerResourceKind::kExecutablePathPoints);
    EXPECT_EQ(over_evaluation.reason_code,
              "RESOURCE_EXECUTABLE_PATH_LIMIT");
    EXPECT_FALSE(over_evaluation.reference.has_value());
  }

  auto hopper = SuccessResult({{0, 0}});
  hopper.reference.hops.resize(3U);
  Evaluate("hopper-exact", Candidate());
  const auto hopper_exact = FinishAndWait(
      "hopper-exact", rclcpp_action::ResultCode::SUCCEEDED, hopper);
  EXPECT_EQ(hopper_exact.kind, PlannerEvaluationKind::kReachable);
  ASSERT_TRUE(hopper_exact.reference.has_value());
  EXPECT_EQ(hopper_exact.reference->hops.size(), 3U);

  hopper.reference.hops.resize(4U);
  Evaluate("hopper-over", Candidate());
  const auto hopper_over = FinishAndWait(
      "hopper-over", rclcpp_action::ResultCode::SUCCEEDED, hopper);
  EXPECT_EQ(hopper_over.kind, PlannerEvaluationKind::kResourceError);
  EXPECT_EQ(hopper_over.resource_kind,
            PlannerResourceKind::kExecutablePathPoints);
  EXPECT_EQ(hopper_over.reason_code, "RESOURCE_EXECUTABLE_PATH_LIMIT");
  EXPECT_FALSE(hopper_over.reference.has_value());
}

TEST_F(PlannerClientTest,
       RejectsConcurrentEvaluationAndClassifiesGoalRejectionAsRetryable) {
  Evaluate("active", Candidate());
  ASSERT_TRUE(WaitFor([&] { return server_->HasHandle("active"); }));
  EXPECT_THROW(Evaluate("concurrent", Candidate()), std::logic_error);
  const auto active = FinishAndWait(
      "active", rclcpp_action::ResultCode::ABORTED,
      FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH"));
  EXPECT_EQ(active.kind, PlannerEvaluationKind::kExhaustiveNoPath);

  server_->RejectNext();
  Evaluate("rejected", Candidate());
  ASSERT_TRUE(WaitFor([&] { return CompletionCount() == 2U; }));
  std::scoped_lock lock{completion_mutex_};
  EXPECT_EQ(completions_.back().request_id, "rejected");
  EXPECT_EQ(completions_.back().kind, PlannerEvaluationKind::kRetryable);
  EXPECT_FALSE(completions_.back().path_length_m.has_value());
  EXPECT_FALSE(completions_.back().reference.has_value());
}

TEST_F(PlannerClientTest,
       CancelBeforeGoalResponseIsIdempotentAndAcceptedHandleGetsOneCancel) {
  server_->DelayNextGoalResponse();
  Evaluate("cancel-before-handle", Candidate());
  ASSERT_TRUE(WaitFor(
      [&] { return server_->GoalCallbackEntered("cancel-before-handle"); }));
  client_->CancelActive();
  client_->CancelActive();
  EXPECT_EQ(server_->CancelCount("cancel-before-handle"), 0U);

  server_->ReleaseGoalResponses();
  ASSERT_TRUE(WaitFor(
      [&] { return server_->CancelCount("cancel-before-handle") == 1U; }));
  const auto evaluation = FinishAndWait(
      "cancel-before-handle", rclcpp_action::ResultCode::CANCELED,
      FailureResult(Action::Result::CANCELED, "REQUEST_CANCELED"));
  EXPECT_EQ(evaluation.kind, PlannerEvaluationKind::kCanceled);
  EXPECT_EQ(CompletionCount(), 1U);
  EXPECT_EQ(server_->CancelCount("cancel-before-handle"), 1U);
}

TEST_F(PlannerClientTest,
       GoalRejectionWinsEvenWhenCancelIntentPrecedesDelayedResponse) {
  server_->DelayNextGoalResponse();
  server_->RejectNext();
  Evaluate("cancel-then-reject", Candidate());
  ASSERT_TRUE(WaitFor(
      [&] { return server_->GoalCallbackEntered("cancel-then-reject"); }));
  client_->CancelActive();
  server_->ReleaseGoalResponses();

  ASSERT_TRUE(WaitFor([&] { return CompletionCount() == 1U; }));
  std::scoped_lock lock{completion_mutex_};
  EXPECT_EQ(completions_.back().kind, PlannerEvaluationKind::kRetryable);
  EXPECT_EQ(completions_.back().reason_code, "GOAL_REJECTED");
  EXPECT_EQ(server_->CancelCount("cancel-then-reject"), 0U);
}

TEST_F(PlannerClientTest,
       RejectedCancelResponseDoesNotCompleteAndTypedResultStillWins) {
  server_->SetAcceptCancel(false);
  Evaluate("rejected-cancel", Candidate());
  ASSERT_TRUE(WaitFor([&] { return server_->HasHandle("rejected-cancel"); }));
  client_->CancelActive();
  ASSERT_TRUE(WaitFor(
      [&] { return server_->CancelCount("rejected-cancel") == 1U; }));
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(CompletionCount(), 0U);

  const auto evaluation = FinishAndWait(
      "rejected-cancel", rclcpp_action::ResultCode::ABORTED,
      FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH"));
  EXPECT_EQ(evaluation.kind, PlannerEvaluationKind::kExhaustiveNoPath);
  EXPECT_EQ(CompletionCount(), 1U);
}

TEST_F(PlannerClientTest,
       UnsolicitedTypedCanceledPayloadIsAContractError) {
  Evaluate("unsolicited-canceled", Candidate());
  ASSERT_TRUE(
      WaitFor([&] { return server_->HasHandle("unsolicited-canceled"); }));
  const auto external_cancel = probe_->async_cancel_all_goals();
  ASSERT_EQ(external_cancel.wait_for(3s), std::future_status::ready);
  ASSERT_TRUE(WaitFor(
      [&] { return server_->CancelCount("unsolicited-canceled") == 1U; }));
  const auto evaluation = FinishAndWait(
      "unsolicited-canceled", rclcpp_action::ResultCode::CANCELED,
      FailureResult(Action::Result::CANCELED, "REQUEST_CANCELED"));
  EXPECT_EQ(evaluation.kind, PlannerEvaluationKind::kContractError);
  EXPECT_FALSE(evaluation.path_length_m.has_value());
  EXPECT_FALSE(evaluation.reference.has_value());
}

TEST_F(PlannerClientTest,
       ClosedDeadlineWithAcceptedHandleClaimsOnceAndScopesCleanupToOldGoal) {
  Evaluate("timeout-with-handle", Candidate());
  ASSERT_TRUE(
      WaitFor([&] { return server_->HasHandle("timeout-with-handle"); }));
  now_ns_ =
      std::chrono::duration_cast<std::chrono::nanoseconds>(3499ms).count();
  client_->PollTimeout();
  EXPECT_EQ(CompletionCount(), 0U);

  now_ns_ =
      std::chrono::duration_cast<std::chrono::nanoseconds>(3500ms).count();
  client_->PollTimeout();
  ASSERT_TRUE(WaitFor([&] { return CompletionCount() == 1U; }));
  ASSERT_TRUE(WaitFor(
      [&] { return server_->CancelCount("timeout-with-handle") == 1U; }));
  client_->PollTimeout();
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(CompletionCount(), 1U);
  EXPECT_EQ(server_->CancelCount("timeout-with-handle"), 1U);
  std::scoped_lock lock{completion_mutex_};
  EXPECT_EQ(completions_.front().kind, PlannerEvaluationKind::kRetryable);
  EXPECT_EQ(completions_.front().reason_code, "CLIENT_RESULT_TIMEOUT");
  EXPECT_FALSE(completions_.front().path_length_m.has_value());
  EXPECT_FALSE(completions_.front().reference.has_value());
}

TEST_F(PlannerClientTest, ResultDeadlineStartsWhenGoalIsAccepted) {
  server_->DelayNextGoalResponse();
  Evaluate("accepted-starts-result-deadline", Candidate());
  ASSERT_TRUE(WaitFor([&] {
    return server_->GoalCallbackEntered("accepted-starts-result-deadline");
  }));

  now_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(750ms).count();
  server_->ReleaseGoalResponses();
  ASSERT_TRUE(WaitFor(
      [&] { return server_->HasHandle("accepted-starts-result-deadline"); }));
  std::this_thread::sleep_for(20ms);

  now_ns_ =
      std::chrono::duration_cast<std::chrono::nanoseconds>(3500ms).count();
  client_->PollTimeout();
  EXPECT_EQ(CompletionCount(), 0U);
  EXPECT_EQ(server_->CancelCount("accepted-starts-result-deadline"), 0U);

  now_ns_ =
      std::chrono::duration_cast<std::chrono::nanoseconds>(4250ms).count();
  client_->PollTimeout();
  ASSERT_TRUE(WaitFor([&] { return CompletionCount() == 1U; }));
  ASSERT_TRUE(WaitFor([&] {
    return server_->CancelCount("accepted-starts-result-deadline") == 1U;
  }));
  std::scoped_lock lock{completion_mutex_};
  EXPECT_EQ(completions_.front().kind, PlannerEvaluationKind::kRetryable);
  EXPECT_EQ(completions_.front().reason_code, "CLIENT_RESULT_TIMEOUT");
}

TEST_F(PlannerClientTest, LateCertifiedPathWinsBeforeResultWatchdog) {
  Evaluate("plan-found-late", Candidate());
  ASSERT_TRUE(WaitFor([&] { return server_->HasHandle("plan-found-late"); }));

  now_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(2500ms).count();
  client_->PollTimeout();
  EXPECT_EQ(CompletionCount(), 0U);
  EXPECT_EQ(server_->CancelCount("plan-found-late"), 0U);

  auto result = SuccessResult({{0.0, 0.0}, {1.0, 0.0}});
  result.reason_code = "PLAN_FOUND_LATE";
  const auto evaluation = FinishAndWait(
      "plan-found-late", rclcpp_action::ResultCode::SUCCEEDED, result);
  EXPECT_EQ(evaluation.kind, PlannerEvaluationKind::kReachable);
  EXPECT_EQ(evaluation.reason_code, "PLAN_FOUND_LATE");
  EXPECT_EQ(server_->CancelCount("plan-found-late"), 0U);
}

TEST_F(PlannerClientTest, ServerTimeoutWinsBeforeResultWatchdog) {
  Evaluate("server-timeout", Candidate());
  ASSERT_TRUE(WaitFor([&] { return server_->HasHandle("server-timeout"); }));

  now_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(3s).count();
  client_->PollTimeout();
  EXPECT_EQ(CompletionCount(), 0U);
  EXPECT_EQ(server_->CancelCount("server-timeout"), 0U);

  const auto evaluation = FinishAndWait(
      "server-timeout", rclcpp_action::ResultCode::ABORTED,
      FailureResult(Action::Result::RESOURCE_EXHAUSTED, "TIMEOUT"));
  EXPECT_EQ(evaluation.kind, PlannerEvaluationKind::kRetryable);
  EXPECT_EQ(evaluation.reason_code, "TIMEOUT");
  EXPECT_EQ(server_->CancelCount("server-timeout"), 0U);
}

TEST_F(PlannerClientTest, MissingGoalResponseUsesGoalWatchdogWithoutCancel) {
  server_->DelayNextGoalResponse();
  Evaluate("missing-goal-response", Candidate());
  ASSERT_TRUE(WaitFor([&] {
    return server_->GoalCallbackEntered("missing-goal-response");
  }));

  now_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(999ms).count();
  client_->PollTimeout();
  EXPECT_EQ(CompletionCount(), 0U);

  now_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count();
  client_->PollTimeout();
  ASSERT_TRUE(WaitFor([&] { return CompletionCount() == 1U; }));
  EXPECT_EQ(server_->CancelCount("missing-goal-response"), 0U);
  std::scoped_lock lock{completion_mutex_};
  EXPECT_EQ(completions_.front().kind, PlannerEvaluationKind::kRetryable);
  EXPECT_EQ(completions_.front().reason_code, "CLIENT_RESULT_TIMEOUT");
  server_->ReleaseGoalResponses();
}

TEST_F(PlannerClientTest,
       ProductionTimerTimesOutOnceAndReentrantGenerationIgnoresOldCallbacks) {
  auto runtime_client = std::make_unique<PlannerClient>(
      *client_node_, action_name_,
      PlannerClientParameters{.maximum_path_preview_poses = 4U,
                              .maximum_executable_path_points = 3U,
                              .goal_response_timeout = 100ms,
                              .result_timeout = 250ms});
  runtime_client->Evaluate(
      "task-alpha", "automatic-timeout", Candidate(), 0.25, 0.125,
      [&, this](PlannerEvaluation evaluation) {
        {
          std::scoped_lock lock{completion_mutex_};
          completions_.push_back(std::move(evaluation));
        }
        runtime_client->Evaluate(
            "task-alpha", "automatic-reentrant", Candidate(42U), 0.25,
            0.125, [this](PlannerEvaluation next) {
              std::scoped_lock lock{completion_mutex_};
              completions_.push_back(std::move(next));
            });
      });
  ASSERT_TRUE(
      WaitFor([&] { return server_->HasHandle("automatic-timeout"); }));

  ASSERT_TRUE(WaitFor([&] { return CompletionCount() == 1U; }));
  ASSERT_TRUE(WaitFor(
      [&] { return server_->CancelCount("automatic-timeout") == 1U; }));
  {
    std::scoped_lock lock{completion_mutex_};
    EXPECT_EQ(completions_[0].request_id, "automatic-timeout");
    EXPECT_EQ(completions_[0].kind, PlannerEvaluationKind::kRetryable);
    EXPECT_EQ(completions_[0].reason_code, "CLIENT_RESULT_TIMEOUT");
  }
  ASSERT_TRUE(
      WaitFor([&] { return server_->HasHandle("automatic-reentrant"); }));

  server_->Finish("automatic-timeout", rclcpp_action::ResultCode::ABORTED,
                  FailureResult(Action::Result::RESOURCE_EXHAUSTED,
                                "TIMEOUT"));
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(CompletionCount(), 1U);
  EXPECT_EQ(server_->CancelCount("automatic-reentrant"), 0U);

  server_->Finish("automatic-reentrant", rclcpp_action::ResultCode::ABORTED,
                  FailureResult(Action::Result::GOAL_INFEASIBLE,
                                "NO_PATH"));
  ASSERT_TRUE(WaitFor([&] { return CompletionCount() == 2U; }));
  std::this_thread::sleep_for(300ms);
  EXPECT_EQ(CompletionCount(), 2U);
  EXPECT_EQ(server_->CancelCount("automatic-timeout"), 1U);
  EXPECT_EQ(server_->CancelCount("automatic-reentrant"), 0U);
  std::scoped_lock lock{completion_mutex_};
  EXPECT_EQ(completions_[1].request_id, "automatic-reentrant");
  EXPECT_EQ(completions_[1].kind,
            PlannerEvaluationKind::kExhaustiveNoPath);
}

TEST_F(PlannerClientTest,
       CancelThenTimeoutBeforeHandleSendsNoCancelAndLateHandleIsANoop) {
  server_->DelayNextGoalResponse();
  Evaluate("timeout-before-handle", Candidate());
  ASSERT_TRUE(WaitFor(
      [&] { return server_->GoalCallbackEntered("timeout-before-handle"); }));
  client_->CancelActive();
  now_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count();
  client_->PollTimeout();
  ASSERT_TRUE(WaitFor([&] { return CompletionCount() == 1U; }));
  EXPECT_EQ(server_->CancelCount("timeout-before-handle"), 0U);

  Evaluate("new-generation", Candidate(41U, 8.0, 9.0, 1.0));
  server_->ReleaseGoalResponses();
  ASSERT_TRUE(
      WaitFor([&] { return server_->HasHandle("new-generation"); }));
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(server_->CancelCount("timeout-before-handle"), 0U);
  EXPECT_EQ(server_->CancelCount("new-generation"), 0U);
  EXPECT_EQ(CompletionCount(), 1U);

  const auto next = FinishAndWait(
      "new-generation", rclcpp_action::ResultCode::ABORTED,
      FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH"));
  EXPECT_EQ(next.kind, PlannerEvaluationKind::kExhaustiveNoPath);
  EXPECT_EQ(CompletionCount(), 2U);
}

TEST_F(PlannerClientTest,
       OldDelayedCancelResponseAndOldResultCannotMutateNewGeneration) {
  server_->DelayCancelResponse();
  Evaluate("old-generation", Candidate());
  ASSERT_TRUE(WaitFor([&] { return server_->HasHandle("old-generation"); }));
  client_->CancelActive();
  ASSERT_TRUE(WaitFor(
      [&] { return server_->CancelCount("old-generation") == 1U; }));
  now_ns_ =
      std::chrono::duration_cast<std::chrono::nanoseconds>(3500ms).count();
  client_->PollTimeout();
  ASSERT_TRUE(WaitFor([&] { return CompletionCount() == 1U; }));

  Evaluate("replacement-generation", Candidate(42U));
  server_->ReleaseCancelResponses();
  ASSERT_TRUE(WaitFor(
      [&] { return server_->HasHandle("replacement-generation"); }));
  EXPECT_EQ(server_->CancelCount("replacement-generation"), 0U);
  server_->Finish("old-generation", rclcpp_action::ResultCode::ABORTED,
                  FailureResult(Action::Result::RESOURCE_EXHAUSTED,
                                "TIMEOUT"));
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(CompletionCount(), 1U);
  EXPECT_EQ(server_->CancelCount("replacement-generation"), 0U);

  const auto replacement = FinishAndWait(
      "replacement-generation", rclcpp_action::ResultCode::ABORTED,
      FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH"));
  EXPECT_EQ(replacement.kind, PlannerEvaluationKind::kExhaustiveNoPath);
  EXPECT_EQ(CompletionCount(), 2U);
}

TEST_F(PlannerClientTest,
       CancelResponseAloneNeverCompletesAndAcceptedCancelWaitsForWatchdog) {
  Evaluate("accepted-cancel-no-result", Candidate());
  ASSERT_TRUE(WaitFor(
      [&] { return server_->HasHandle("accepted-cancel-no-result"); }));
  client_->CancelActive();
  ASSERT_TRUE(WaitFor([&] {
    return server_->CancelCount("accepted-cancel-no-result") == 1U;
  }));
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(CompletionCount(), 0U);

  now_ns_ =
      std::chrono::duration_cast<std::chrono::nanoseconds>(3500ms).count();
  client_->PollTimeout();
  ASSERT_TRUE(WaitFor([&] { return CompletionCount() == 1U; }));
  std::scoped_lock lock{completion_mutex_};
  EXPECT_EQ(completions_.front().kind, PlannerEvaluationKind::kRetryable);
  EXPECT_EQ(completions_.front().reason_code, "CLIENT_RESULT_TIMEOUT");
}

TEST_F(PlannerClientTest,
       ResultBeforeDelayedCancelResponseWinsAndLateResponseIsIgnored) {
  server_->DelayCancelResponse();
  Evaluate("result-before-cancel-response", Candidate());
  ASSERT_TRUE(WaitFor(
      [&] { return server_->HasHandle("result-before-cancel-response"); }));
  client_->CancelActive();
  ASSERT_TRUE(WaitFor([&] {
    return server_->CancelCount("result-before-cancel-response") == 1U;
  }));
  const auto evaluation = FinishAndWait(
      "result-before-cancel-response", rclcpp_action::ResultCode::ABORTED,
      FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH"));
  EXPECT_EQ(evaluation.kind, PlannerEvaluationKind::kExhaustiveNoPath);
  server_->ReleaseCancelResponses();
  now_ns_ =
      std::chrono::duration_cast<std::chrono::nanoseconds>(3500ms).count();
  client_->PollTimeout();
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(CompletionCount(), 1U);
}

TEST_F(PlannerClientTest,
       FeedbackNeverChangesResultTimingCancelOrWatchdogState) {
  Evaluate("feedback-ignored", Candidate());
  ASSERT_TRUE(WaitFor([&] { return server_->HasHandle("feedback-ignored"); }));
  for (std::size_t index = 0U; index < 5U; ++index) {
    server_->PublishFeedback("feedback-ignored");
  }
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(CompletionCount(), 0U);
  EXPECT_EQ(server_->CancelCount("feedback-ignored"), 0U);
  const auto evaluation = FinishAndWait(
      "feedback-ignored", rclcpp_action::ResultCode::ABORTED,
      FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH"));
  EXPECT_EQ(evaluation.kind, PlannerEvaluationKind::kExhaustiveNoPath);
  EXPECT_EQ(server_->CancelCount("feedback-ignored"), 0U);
}

TEST_F(PlannerClientTest,
       CompletionMayReenterEvaluateAndThrowWithoutUndoingEitherGeneration) {
  client_->Evaluate(
      "task-alpha", "reentrant-first", Candidate(), 0.25, 0.125,
      [this](PlannerEvaluation evaluation) {
        {
          std::scoped_lock lock{completion_mutex_};
          completions_.push_back(std::move(evaluation));
        }
        client_->Evaluate(
            "task-alpha", "reentrant-second", Candidate(43U), 0.25, 0.125,
            [this](PlannerEvaluation second) {
              std::scoped_lock lock{completion_mutex_};
              completions_.push_back(std::move(second));
            });
        throw std::runtime_error{"completion failure"};
      });
  ASSERT_TRUE(WaitFor([&] { return server_->HasHandle("reentrant-first"); }));
  server_->Finish("reentrant-first", rclcpp_action::ResultCode::ABORTED,
                  FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH"));
  ASSERT_TRUE(WaitFor([&] { return server_->HasHandle("reentrant-second"); }));
  server_->Finish("reentrant-second", rclcpp_action::ResultCode::ABORTED,
                  FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH"));
  ASSERT_TRUE(WaitFor([&] { return CompletionCount() == 2U; }));
  std::scoped_lock lock{completion_mutex_};
  EXPECT_EQ(completions_[0].request_id, "reentrant-first");
  EXPECT_EQ(completions_[1].request_id, "reentrant-second");
}

TEST_F(PlannerClientTest,
       DestructionClearsCompletionAndDelayedTransportCallbacksDoNothing) {
  server_->DelayNextGoalResponse();
  Evaluate("destroyed-client", Candidate());
  ASSERT_TRUE(WaitFor(
      [&] { return server_->GoalCallbackEntered("destroyed-client"); }));
  client_.reset();
  server_->ReleaseGoalResponses();
  ASSERT_TRUE(WaitFor([&] { return server_->HasHandle("destroyed-client"); }));
  server_->Finish("destroyed-client", rclcpp_action::ResultCode::ABORTED,
                  FailureResult(Action::Result::GOAL_INFEASIBLE, "NO_PATH"));
  std::this_thread::sleep_for(50ms);
  EXPECT_EQ(CompletionCount(), 0U);
  EXPECT_EQ(server_->CancelCount("destroyed-client"), 0U);
}

TEST_F(PlannerClientTest,
       ServerUnavailableIsRetryableWithoutBlockingOrRetainingCandidate) {
  PlannerClient unavailable{
      *client_node_, action_name_ + "_missing",
      PlannerClientParameters{.maximum_path_preview_poses = 1U,
                              .maximum_executable_path_points = 1U,
                              .goal_response_timeout = 1s,
                              .result_timeout = 3500ms},
      [] { return std::chrono::steady_clock::time_point{}; }};
  std::optional<PlannerEvaluation> completion;
  auto candidate = Candidate(55U);
  unavailable.Evaluate(
      "task-alpha", "unavailable", candidate, 0.0, 0.0,
      [&](PlannerEvaluation evaluation) { completion = std::move(evaluation); });
  candidate.id = 99U;
  ASSERT_TRUE(completion.has_value());
  EXPECT_EQ(completion->request_id, "unavailable");
  EXPECT_EQ(completion->candidate_id, 55U);
  EXPECT_EQ(completion->kind, PlannerEvaluationKind::kRetryable);
  EXPECT_EQ(completion->reason_code, "ACTION_SERVER_UNAVAILABLE");
}

TEST(FinalPlannerCompatibilityTest,
     IntegratedAdaptersAndFinalizerPreserveTheReviewedClientContract) {
  lunar::pure_planning::RigidTransform identity;
  identity.parent_frame = "map";
  identity.child_frame = "odom";
  identity.rotation.w = 1.0;
  for (const double incoming_z :
       {-std::numeric_limits<double>::max(), -1.0, 0.0, 1.0,
        std::numeric_limits<double>::max()}) {
    Action::Goal goal;
    goal.environment_mode = Action::Goal::LUNAR_SURFACE;
    goal.goal.header.frame_id = "map";
    goal.goal.goal_id = "finite-z-is-not-a-planar-input";
    goal.goal.goal_type = goal.goal.POINT;
    goal.goal.point.x = 2.0;
    goal.goal.point.y = 3.0;
    goal.goal.point.z = incoming_z;
    goal.goal.position_tolerance_m = 0.25;
    goal.goal.yaw_tolerance_rad = 0.125;
    const auto converted =
        lunar::pure_planner_ros::ConvertGoal(goal, identity);
    ASSERT_TRUE(converted.ok());
    const auto& point = std::get<lunar::pure_planning::PointGoal>(
        converted.goal->target);
    EXPECT_DOUBLE_EQ(point.position_m.x, 2.0);
    EXPECT_DOUBLE_EQ(point.position_m.y, 3.0);
    EXPECT_DOUBLE_EQ(point.position_m.z, 0.0);
    EXPECT_FALSE(std::signbit(point.position_m.z));
  }

  struct MappingCase final {
    lunar::pure_planning::PlanningStatus status;
    std::uint8_t outcome;
    const char* reason;
  };
  for (const auto& mapping : {
           MappingCase{lunar::pure_planning::PlanningStatus::kNoPath,
                       Action::Result::GOAL_INFEASIBLE, "NO_PATH"},
           MappingCase{lunar::pure_planning::PlanningStatus::kTimedOut,
                       Action::Result::RESOURCE_EXHAUSTED, "TIMEOUT"},
           MappingCase{lunar::pure_planning::PlanningStatus::kInvalidInput,
                       Action::Result::INVALID_REQUEST, "INVALID_INPUT"},
       }) {
    const auto converted = lunar::pure_planner_ros::ConvertResult(
        lunar::pure_planning::PlanningResult{.status = mapping.status,
                                             .reason_code = mapping.reason,
                                             .reference = std::nullopt,
                                             .timing = {},
                                             .expanded_states = 0U},
        0U);
    EXPECT_EQ(converted.planning_outcome, mapping.outcome);
    EXPECT_EQ(converted.execution_directive,
              Action::Result::NO_SAFE_REFERENCE);
    EXPECT_EQ(converted.reason_code, mapping.reason);
    EXPECT_FALSE(converted.has_reference);
    EXPECT_EQ(converted.reference,
              lunar_planning_msgs::msg::MotionReference{});
  }

  std::vector<std::string> order;
  std::uint64_t abort_count = 0U;
  lunar::pure_planner_ros::detail::FinalizeAcceptedGoal(
      [&] {
        order.emplace_back("result");
        return lunar::pure_planner_ros::detail::
            MakeMinimalPlannerErrorResult<Action>(0U, 1ms);
      },
      [&](const std::shared_ptr<Action::Result>& result) {
        order.emplace_back("terminal");
        ++abort_count;
        EXPECT_EQ(result->planning_outcome,
                  Action::Result::NUMERICAL_FAILURE);
      },
      [&] {
        order.emplace_back("diagnostic-build");
        return diagnostic_msgs::msg::DiagnosticArray{};
      },
      [&](const diagnostic_msgs::msg::DiagnosticArray&) {
        order.emplace_back("diagnostic-publish");
      },
      [&](const diagnostic_msgs::msg::DiagnosticArray&) {
        order.emplace_back("diagnostic-format");
      },
      [](lunar::pure_planner_ros::detail::
             AcceptedGoalFinalizationFailure) {});
  EXPECT_EQ(order,
            (std::vector<std::string>{"result", "terminal",
                                      "diagnostic-build",
                                      "diagnostic-publish",
                                      "diagnostic-format"}));
  EXPECT_EQ(abort_count, 1U);
}

TEST(PlannerClientTraitsTest, IsNothrowDestructibleAndNeitherCopyableNorMovable) {
  static_assert(std::is_nothrow_destructible_v<PlannerClient>);
  static_assert(!std::is_copy_constructible_v<PlannerClient>);
  static_assert(!std::is_copy_assignable_v<PlannerClient>);
  static_assert(!std::is_move_constructible_v<PlannerClient>);
  static_assert(!std::is_move_assignable_v<PlannerClient>);
}

}  // namespace
}  // namespace lunar::pure_exploration_ros
