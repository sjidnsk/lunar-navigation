#include "lunar_pure_exploration_ros/planner_client.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <lunar_planning_msgs/action/plan_motion.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp/timer.hpp>

namespace lunar::pure_exploration_ros {
namespace {

using Action = lunar_planning_msgs::action::PlanMotion;
using ActionClient = rclcpp_action::Client<Action>;
using GoalHandle = rclcpp_action::ClientGoalHandle<Action>;

PlannerEvaluation ContractError(std::string request_id,
                                const std::uint64_t candidate_id,
                                std::string reason_code) {
  return {.request_id = std::move(request_id),
          .candidate_id = candidate_id,
          .kind = PlannerEvaluationKind::kContractError,
          .reason_code = std::move(reason_code),
          .path_length_m = std::nullopt,
          .reference = std::nullopt};
}

PlannerEvaluation FailureEvaluation(std::string request_id,
                                    const std::uint64_t candidate_id,
                                    const PlannerEvaluationKind kind,
                                    std::string reason_code) {
  return {.request_id = std::move(request_id),
          .candidate_id = candidate_id,
          .kind = kind,
          .reason_code = std::move(reason_code),
          .path_length_m = std::nullopt,
          .reference = std::nullopt};
}

PlannerEvaluation ResourceEvaluation(
    std::string request_id, const std::uint64_t candidate_id,
    const PlannerResourceKind resource_kind, std::string reason_code) {
  return {.request_id = std::move(request_id),
          .candidate_id = candidate_id,
          .kind = PlannerEvaluationKind::kResourceError,
          .resource_kind = resource_kind,
          .reason_code = std::move(reason_code),
          .path_length_m = std::nullopt,
          .reference = std::nullopt};
}

bool IsEmptyReference(
    const lunar_planning_msgs::msg::MotionReference& reference) {
  return reference == lunar_planning_msgs::msg::MotionReference{};
}

std::optional<double> PathLength(
    const lunar_planning_msgs::msg::MotionReference& reference,
    const std::size_t maximum_path_preview_poses) {
  const auto& poses = reference.path_preview.poses;
  if (poses.empty() || poses.size() > maximum_path_preview_poses) {
    return std::nullopt;
  }

  long double total = 0.0L;
  long double previous_x{};
  long double previous_y{};
  for (std::size_t index = 0U; index < poses.size(); ++index) {
    const double source_x = poses[index].pose.position.x;
    const double source_y = poses[index].pose.position.y;
    if (!std::isfinite(source_x) || !std::isfinite(source_y)) {
      return std::nullopt;
    }
    const long double x = static_cast<long double>(source_x);
    const long double y = static_cast<long double>(source_y);
    if (index != 0U) {
      const long double segment = std::hypot(x - previous_x, y - previous_y);
      if (!std::isfinite(segment)) {
        return std::nullopt;
      }
      total += segment;
      if (!std::isfinite(total)) {
        return std::nullopt;
      }
    }
    previous_x = x;
    previous_y = y;
  }
  if (total > static_cast<long double>(std::numeric_limits<double>::max())) {
    return std::nullopt;
  }
  double result = static_cast<double>(total);
  if (!std::isfinite(result) || result < 0.0) {
    return std::nullopt;
  }
  if (result == 0.0) {
    result = 0.0;
  }
  return result;
}

PlannerEvaluation Classify(
    const GoalHandle::WrappedResult& wrapped, const std::string& request_id,
    const std::uint64_t candidate_id, const bool locally_requested_cancel,
    const std::size_t maximum_path_preview_poses,
    const std::size_t maximum_executable_path_points) {
  if (!wrapped.result) {
    return FailureEvaluation(request_id, candidate_id,
                             PlannerEvaluationKind::kRetryable,
                             "TRANSPORT_RESULT_MISSING");
  }

  const auto& result = *wrapped.result;
  if (result.reason_code.empty()) {
    return ContractError(request_id, candidate_id, "EMPTY_REASON_CODE");
  }

  if (result.planning_outcome == Action::Result::NEW_REFERENCE_AVAILABLE) {
    if (wrapped.code != rclcpp_action::ResultCode::SUCCEEDED ||
        result.execution_directive !=
            Action::Result::ACTIVATE_NEW_REFERENCE ||
        !result.has_reference) {
      return ContractError(request_id, candidate_id,
                           "RESULT_CONTRACT_MISMATCH");
    }
    if (result.reference.path_preview.poses.size() >
        maximum_path_preview_poses) {
      return ResourceEvaluation(
          request_id, candidate_id, PlannerResourceKind::kPathPreviewPoses,
          "RESOURCE_PATH_PREVIEW_LIMIT");
    }
    if (result.reference.trajectory.points.size() >
            maximum_executable_path_points ||
        result.reference.hops.size() > maximum_executable_path_points) {
      return ResourceEvaluation(
          request_id, candidate_id,
          PlannerResourceKind::kExecutablePathPoints,
          "RESOURCE_EXECUTABLE_PATH_LIMIT");
    }
    std::optional<double> path_length;
    if (result.diagnostics.has_best_cost) {
      const double best_cost = result.diagnostics.best_cost;
      if (!std::isfinite(best_cost) || best_cost < 0.0) {
        return ContractError(request_id, candidate_id, "BEST_COST_INVALID");
      }
      path_length = best_cost;
    } else {
      path_length = PathLength(result.reference, maximum_path_preview_poses);
    }
    if (!path_length.has_value()) {
      return ContractError(request_id, candidate_id,
                           "PATH_PREVIEW_INVALID");
    }
    return {.request_id = request_id,
            .candidate_id = candidate_id,
            .kind = PlannerEvaluationKind::kReachable,
            .reason_code = result.reason_code,
            .path_length_m = path_length,
            .reference = result.reference};
  }

  const bool empty_failure_reference =
      !result.has_reference && IsEmptyReference(result.reference);
  const bool ordinary_failure_wrapper =
      wrapped.code == rclcpp_action::ResultCode::ABORTED;
  const std::string_view reason_code{result.reason_code};
  if (result.planning_outcome == Action::Result::GOAL_INFEASIBLE &&
      result.execution_directive == Action::Result::NO_SAFE_REFERENCE &&
      empty_failure_reference && ordinary_failure_wrapper &&
      (reason_code == "NO_PATH" || reason_code == "GOAL_OUTSIDE_LOCAL_MAP" ||
       reason_code == "GOAL_NOT_FREE" || reason_code == "GLOBAL_NO_PATH" ||
       reason_code == "LOCAL_NO_CANDIDATE" || reason_code == "LOCAL_NO_PATH")) {
    return FailureEvaluation(request_id, candidate_id,
                             PlannerEvaluationKind::kExhaustiveNoPath,
                             result.reason_code);
  }
  if (result.planning_outcome == Action::Result::RESOURCE_EXHAUSTED &&
      result.execution_directive == Action::Result::NO_SAFE_REFERENCE &&
      empty_failure_reference && ordinary_failure_wrapper &&
      result.reason_code == "TIMEOUT") {
    return FailureEvaluation(request_id, candidate_id,
                             PlannerEvaluationKind::kRetryable,
                             result.reason_code);
  }
  if (result.planning_outcome == Action::Result::ACTIVE_REFERENCE_INVALIDATED &&
      result.execution_directive == Action::Result::NO_SAFE_REFERENCE &&
      empty_failure_reference && ordinary_failure_wrapper &&
      reason_code == "STALE_PATH_INVALIDATED") {
    return FailureEvaluation(request_id, candidate_id,
                             PlannerEvaluationKind::kRetryable,
                             result.reason_code);
  }
  if (result.planning_outcome == Action::Result::CANCELED &&
      result.execution_directive == Action::Result::NO_SAFE_REFERENCE &&
      empty_failure_reference && locally_requested_cancel &&
      wrapped.code == rclcpp_action::ResultCode::CANCELED &&
      result.reason_code == "REQUEST_CANCELED") {
    return FailureEvaluation(request_id, candidate_id,
                             PlannerEvaluationKind::kCanceled,
                             result.reason_code);
  }
  if (((result.planning_outcome == Action::Result::INVALID_REQUEST &&
        (reason_code == "START_NOT_FREE" || reason_code == "INVALID_INPUT" ||
         reason_code == "MAP_RESOLUTION_MISMATCH")) ||
       (result.planning_outcome == Action::Result::NUMERICAL_FAILURE &&
        (reason_code == "POSTCHECK_FAILED" || reason_code == "PLANNER_ERROR"))) &&
      result.execution_directive == Action::Result::NO_SAFE_REFERENCE &&
      empty_failure_reference && ordinary_failure_wrapper) {
    return ContractError(request_id, candidate_id, result.reason_code);
  }
  return ContractError(request_id, candidate_id, "RESULT_CONTRACT_MISMATCH");
}

void InvokeCompletion(PlannerClient::Completion completion,
                      PlannerEvaluation evaluation) noexcept {
  try {
    completion(std::move(evaluation));
  } catch (...) {
  }
}

}  // namespace

struct PlannerClient::CallbackState final {
  struct ActiveRequest final {
    std::uint64_t generation;
    std::string request_id;
    std::uint64_t candidate_id;
    Completion completion;
    std::chrono::steady_clock::time_point deadline;
    GoalHandle::SharedPtr goal_handle;
    bool cancel_requested{false};
    bool cancel_sent{false};
  };

  std::mutex mutex;
  bool teardown{false};
  std::uint64_t next_generation{0U};
  std::optional<ActiveRequest> active;
  PlannerClientParameters parameters;
  SteadyNow now;
  ActionClient::SharedPtr action_client;
  rclcpp::TimerBase::SharedPtr goal_response_timer;
  rclcpp::TimerBase::SharedPtr result_timer;
};

namespace {

struct CompletionDispatch final {
  PlannerClient::Completion completion;
  PlannerEvaluation evaluation;
};

template <typename State>
void CancelTimerNoexcept(State& state) noexcept {
  try {
    if (state.goal_response_timer) {
      state.goal_response_timer->cancel();
    }
    if (state.result_timer) {
      state.result_timer->cancel();
    }
  } catch (...) {
  }
}

template <typename State>
std::optional<CompletionDispatch> ClaimLocked(
    State& state, const std::uint64_t generation,
    PlannerEvaluation evaluation) {
  if (state.teardown || !state.active.has_value() ||
      state.active->generation != generation) {
    return std::nullopt;
  }
  CompletionDispatch dispatch{.completion = std::move(state.active->completion),
                              .evaluation = std::move(evaluation)};
  CancelTimerNoexcept(state);
  state.active.reset();
  return dispatch;
}

void Dispatch(std::optional<CompletionDispatch> dispatch) noexcept {
  if (dispatch.has_value()) {
    InvokeCompletion(std::move(dispatch->completion),
                     std::move(dispatch->evaluation));
  }
}

template <typename State>
void SendScopedCancel(
    const std::shared_ptr<State>& state,
    const std::uint64_t generation,
    const GoalHandle::SharedPtr& goal_handle) noexcept {
  try {
    const std::weak_ptr<State> weak_state{state};
    state->action_client->async_cancel_goal(
        goal_handle,
        [weak_state, generation](const ActionClient::CancelResponse::SharedPtr) {
          const auto locked = weak_state.lock();
          if (!locked) {
            return;
          }
          std::scoped_lock lock{locked->mutex};
          if (locked->teardown || !locked->active.has_value() ||
              locked->active->generation != generation) {
            return;
          }
        });
  } catch (...) {
  }
}

template <typename State>
void PollTimeoutState(const std::shared_ptr<State>& state) {
  const auto now = state->now();
  GoalHandle::SharedPtr expired_handle;
  std::uint64_t generation{};
  std::optional<CompletionDispatch> dispatch;
  {
    std::scoped_lock lock{state->mutex};
    if (state->teardown || !state->active.has_value() ||
        now < state->active->deadline) {
      return;
    }
    generation = state->active->generation;
    expired_handle = std::move(state->active->goal_handle);
    dispatch = ClaimLocked(
        *state, generation,
        FailureEvaluation(state->active->request_id,
                          state->active->candidate_id,
                          PlannerEvaluationKind::kRetryable,
                          "CLIENT_RESULT_TIMEOUT"));
  }
  if (expired_handle) {
    SendScopedCancel(state, generation, expired_handle);
  }
  Dispatch(std::move(dispatch));
}

}  // namespace

PlannerClient::PlannerClient(rclcpp::Node& node, std::string action_name,
                             PlannerClientParameters parameters, SteadyNow now) {
  if (action_name.empty() || parameters.maximum_path_preview_poses == 0U ||
      parameters.maximum_executable_path_points == 0U ||
      parameters.goal_response_timeout <=
          std::chrono::steady_clock::duration::zero() ||
      parameters.result_timeout <= std::chrono::steady_clock::duration::zero() ||
      !now) {
    throw std::invalid_argument{"invalid planner client parameters"};
  }
  auto state = std::make_shared<CallbackState>();
  state->parameters = parameters;
  state->now = std::move(now);
  state->action_client =
      rclcpp_action::create_client<Action>(&node, std::move(action_name));
  const std::weak_ptr<CallbackState> weak_state{state};
  state->goal_response_timer = node.create_wall_timer(
      parameters.goal_response_timeout, [weak_state] {
        try {
          const auto locked = weak_state.lock();
          if (locked) {
            PollTimeoutState(locked);
          }
        } catch (...) {
        }
      });
  state->result_timer = node.create_wall_timer(
      parameters.result_timeout, [weak_state] {
        try {
          const auto locked = weak_state.lock();
          if (locked) {
            PollTimeoutState(locked);
          }
        } catch (...) {
        }
      });
  state->goal_response_timer->cancel();
  state->result_timer->cancel();
  state_ = std::move(state);
}

PlannerClient::~PlannerClient() noexcept {
  if (!state_) {
    return;
  }
  {
    std::scoped_lock lock{state_->mutex};
    state_->teardown = true;
    CancelTimerNoexcept(*state_);
    state_->active.reset();
  }
  state_.reset();
}

std::string PlannerClient::MakeRequestId(const std::string_view task_id,
                                         const std::uint64_t sequence) {
  return std::string{task_id} + "/candidate/" + std::to_string(sequence);
}

void PlannerClient::Evaluate(
    std::string task_id, std::string request_id,
    const lunar::pure_exploration::CandidateView& candidate,
    const double position_tolerance_m, const double yaw_tolerance_rad,
    Completion completion) {
  if (task_id.empty() || request_id.empty() || !completion ||
      !std::isfinite(candidate.pose.x) ||
      !std::isfinite(candidate.pose.y) ||
      !std::isfinite(candidate.pose.yaw) ||
      !std::isfinite(position_tolerance_m) || position_tolerance_m < 0.0 ||
      !std::isfinite(yaw_tolerance_rad) || yaw_tolerance_rad < 0.0) {
    throw std::invalid_argument{"invalid planner evaluation"};
  }

  Action::Goal goal;
  goal.environment_mode = Action::Goal::LUNAR_SURFACE;
  goal.request_id = request_id;
  goal.mission_id = std::move(task_id);
  goal.mission_revision = 0U;
  goal.goal.header.frame_id = "map";
  goal.goal.goal_id = request_id + "/goal";
  goal.goal.goal_type = goal.goal.POINT;
  goal.goal.point.x = candidate.pose.x;
  goal.goal.point.y = candidate.pose.y;
  goal.goal.point.z = 0.0;
  goal.goal.planar_region.points.clear();
  goal.goal.position_tolerance_m = position_tolerance_m;
  goal.goal.has_yaw_constraint = true;
  goal.goal.yaw_rad = candidate.pose.yaw;
  goal.goal.yaw_tolerance_rad = yaw_tolerance_rad;
  goal.replace_active_request = false;

  const auto now = state_->now();
  const auto deadline = now + state_->parameters.goal_response_timeout;
  const std::weak_ptr<CallbackState> weak_state{state_};
  std::uint64_t generation{};
  std::optional<CompletionDispatch> immediate;
  {
    std::scoped_lock lock{state_->mutex};
    if (state_->teardown) {
      throw std::logic_error{"planner client is tearing down"};
    }
    if (state_->active.has_value()) {
      throw std::logic_error{"planner evaluation already active"};
    }
    generation = state_->next_generation++;
    state_->active.emplace(CallbackState::ActiveRequest{
        .generation = generation,
        .request_id = request_id,
        .candidate_id = candidate.id,
        .completion = std::move(completion),
        .deadline = deadline,
        .goal_handle = GoalHandle::SharedPtr{},
        .cancel_requested = false,
        .cancel_sent = false});
    try {
      state_->goal_response_timer->reset();
    } catch (...) {
      state_->active.reset();
      throw;
    }

    if (!state_->action_client->action_server_is_ready()) {
      immediate = ClaimLocked(
          *state_, generation,
          FailureEvaluation(request_id, candidate.id,
                            PlannerEvaluationKind::kRetryable,
                            "ACTION_SERVER_UNAVAILABLE"));
    } else {
      ActionClient::SendGoalOptions options;
      options.goal_response_callback =
          [weak_state, generation](const GoalHandle::SharedPtr goal_handle) {
            const auto state = weak_state.lock();
            if (!state) {
              return;
            }
            GoalHandle::SharedPtr cancel_handle;
            std::optional<CompletionDispatch> dispatch;
            {
              std::scoped_lock lock{state->mutex};
              if (state->teardown || !state->active.has_value() ||
                  state->active->generation != generation) {
                return;
              }
              if (!goal_handle) {
                dispatch = ClaimLocked(
                    *state, generation,
                    FailureEvaluation(
                        state->active->request_id,
                        state->active->candidate_id,
                        PlannerEvaluationKind::kRetryable, "GOAL_REJECTED"));
              } else {
                state->active->deadline =
                    state->now() + state->parameters.result_timeout;
                state->goal_response_timer->cancel();
                state->result_timer->reset();
                state->active->goal_handle = goal_handle;
                if (state->active->cancel_requested &&
                    !state->active->cancel_sent) {
                  state->active->cancel_sent = true;
                  cancel_handle = goal_handle;
                }
              }
            }
            if (cancel_handle) {
              SendScopedCancel(state, generation, cancel_handle);
            }
            Dispatch(std::move(dispatch));
          };
      options.result_callback =
          [weak_state, generation](
              const GoalHandle::WrappedResult& wrapped) noexcept {
            try {
              const auto state = weak_state.lock();
              if (!state) {
                return;
              }
              std::optional<CompletionDispatch> dispatch;
              {
                std::scoped_lock lock{state->mutex};
                if (state->teardown || !state->active.has_value() ||
                    state->active->generation != generation) {
                  return;
                }
                try {
                  dispatch = ClaimLocked(
                      *state, generation,
                      Classify(
                          wrapped, state->active->request_id,
                          state->active->candidate_id,
                          state->active->cancel_requested,
                          state->parameters.maximum_path_preview_poses,
                          state->parameters.maximum_executable_path_points));
                } catch (const std::length_error&) {
                  dispatch = ClaimLocked(
                      *state, generation,
                      ResourceEvaluation(
                          std::move(state->active->request_id),
                          state->active->candidate_id,
                          PlannerResourceKind::kResultCopyFailure,
                          "RESOURCE_RESULT_COPY"));
                } catch (const std::bad_alloc&) {
                  dispatch = ClaimLocked(
                      *state, generation,
                      ResourceEvaluation(
                          std::move(state->active->request_id),
                          state->active->candidate_id,
                          PlannerResourceKind::kResultCopyFailure,
                          "RESOURCE_RESULT_COPY"));
                }
              }
              Dispatch(std::move(dispatch));
            } catch (...) {
              // ROS transport callbacks must never propagate allocation or
              // user-visible classification failures into the executor.
            }
          };
      try {
        state_->action_client->async_send_goal(goal, options);
      } catch (...) {
        immediate = ClaimLocked(
            *state_, generation,
            FailureEvaluation(request_id, candidate.id,
                              PlannerEvaluationKind::kRetryable,
                              "GOAL_SEND_FAILED"));
      }
    }
  }
  Dispatch(std::move(immediate));
}

void PlannerClient::CancelActive() {
  GoalHandle::SharedPtr cancel_handle;
  std::uint64_t generation{};
  {
    std::scoped_lock lock{state_->mutex};
    if (state_->teardown || !state_->active.has_value() ||
        state_->active->cancel_requested) {
      return;
    }
    state_->active->cancel_requested = true;
    generation = state_->active->generation;
    if (state_->active->goal_handle && !state_->active->cancel_sent) {
      state_->active->cancel_sent = true;
      cancel_handle = state_->active->goal_handle;
    }
  }
  if (cancel_handle) {
    SendScopedCancel(state_, generation, cancel_handle);
  }
}

void PlannerClient::PollTimeout() {
  PollTimeoutState(state_);
}

}  // namespace lunar::pure_exploration_ros
