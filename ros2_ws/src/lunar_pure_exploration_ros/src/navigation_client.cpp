#include "lunar_pure_exploration_ros/navigation_client.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

#include <lunar_planning_msgs/action/navigate_to_pose.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

namespace lunar::pure_exploration_ros {
namespace {

using Action = lunar_planning_msgs::action::NavigateToPose;
using ActionClient = rclcpp_action::Client<Action>;
using GoalHandle = rclcpp_action::ClientGoalHandle<Action>;

std::optional<NavigationFeedbackState> MapFeedback(const std::uint8_t state) {
  switch (state) {
    case Action::Feedback::PLANNING:
      return NavigationFeedbackState::kPlanning;
    case Action::Feedback::EXECUTING:
      return NavigationFeedbackState::kExecuting;
    case Action::Feedback::REPLANNING:
      return NavigationFeedbackState::kReplanning;
    default:
      return std::nullopt;
  }
}

void InvokeFeedback(const NavigationClient::FeedbackCallback& callback,
                    const NavigationFeedbackState state) noexcept {
  try {
    callback(state);
  } catch (...) {
  }
}

void InvokeTerminal(const NavigationClient::TerminalCallback& callback,
                    NavigationTerminal terminal) noexcept {
  try {
    callback(std::move(terminal));
  } catch (...) {
  }
}

NavigationTerminal MissingResult() {
  return {.outcome = Action::Result::INTERNAL_ERROR,
          .reason_code = "NAVIGATION_RESULT_MISSING"};
}

NavigationTerminal RejectedGoal() {
  return {.outcome = Action::Result::INTERNAL_ERROR,
          .reason_code = "NAVIGATION_GOAL_REJECTED"};
}

}  // namespace

struct NavigationClient::CallbackState final {
  struct ActiveGoal final {
    std::uint64_t generation;
    GoalHandle::SharedPtr handle;
    bool cancel_requested{false};
  };

  std::mutex mutex;
  bool teardown{false};
  std::uint64_t next_generation{0U};
  std::optional<ActiveGoal> active;
  FeedbackCallback feedback_callback;
  TerminalCallback terminal_callback;
  ActionClient::SharedPtr action_client;
};

namespace {

template <typename State>
void SendCancel(const std::shared_ptr<State>& state,
                const std::uint64_t generation,
                const GoalHandle::SharedPtr& handle) noexcept {
  try {
    const std::weak_ptr<State> weak_state{state};
    state->action_client->async_cancel_goal(
        handle, [weak_state, generation](
                    const ActionClient::CancelResponse::SharedPtr) {
          const auto locked = weak_state.lock();
          if (!locked) {
            return;
          }
          std::scoped_lock lock{locked->mutex};
          if (locked->teardown || !locked->active ||
              locked->active->generation != generation) {
            return;
          }
        });
  } catch (...) {
  }
}

}  // namespace

NavigationClient::NavigationClient(
    rclcpp::Node& node, std::string action_name,
    FeedbackCallback feedback_callback, TerminalCallback terminal_callback) {
  if (action_name.empty() || action_name.front() != '/' ||
      !feedback_callback || !terminal_callback) {
    throw std::invalid_argument{"invalid navigation client configuration"};
  }
  auto state = std::make_shared<CallbackState>();
  state->feedback_callback = std::move(feedback_callback);
  state->terminal_callback = std::move(terminal_callback);
  state->action_client =
      rclcpp_action::create_client<Action>(&node, std::move(action_name));
  state_ = std::move(state);
}

NavigationClient::~NavigationClient() noexcept {
  if (!state_) {
    return;
  }
  std::scoped_lock lock{state_->mutex};
  state_->teardown = true;
  state_->active.reset();
}

void NavigationClient::Navigate(const NavigationTarget target) {
  if (!std::isfinite(target.pose.x) || !std::isfinite(target.pose.y) ||
      !std::isfinite(target.pose.yaw)) {
    throw std::invalid_argument{"navigation target must be finite"};
  }

  std::uint64_t generation{};
  {
    std::scoped_lock lock{state_->mutex};
    if (state_->teardown) {
      throw std::logic_error{"navigation client is shutting down"};
    }
    if (state_->active) {
      throw std::logic_error{"navigation goal already active"};
    }
    if (state_->next_generation == std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error{"navigation goal generation exhausted"};
    }
    generation = ++state_->next_generation;
    state_->active.emplace(
        CallbackState::ActiveGoal{.generation = generation});
  }

  Action::Goal goal;
  goal.target_x_m = target.pose.x;
  goal.target_y_m = target.pose.y;
  goal.has_target_yaw = target.has_target_yaw;
  goal.target_yaw_rad = target.pose.yaw;

  ActionClient::SendGoalOptions options;
  const std::weak_ptr<CallbackState> weak_state{state_};
  options.goal_response_callback =
      [weak_state, generation](const GoalHandle::SharedPtr& handle) {
        const auto state = weak_state.lock();
        if (!state) {
          return;
        }
        bool cancel = false;
        TerminalCallback terminal_callback;
        {
          std::scoped_lock lock{state->mutex};
          if (state->teardown || !state->active ||
              state->active->generation != generation) {
            return;
          }
          if (!handle) {
            terminal_callback = state->terminal_callback;
            state->active.reset();
          } else {
            state->active->handle = handle;
            cancel = state->active->cancel_requested;
          }
        }
        if (!handle) {
          InvokeTerminal(terminal_callback, RejectedGoal());
        } else if (cancel) {
          SendCancel(state, generation, handle);
        }
      };
  options.feedback_callback =
      [weak_state, generation](
          GoalHandle::SharedPtr,
          const std::shared_ptr<const Action::Feedback> feedback) {
        if (!feedback) {
          return;
        }
        const auto mapped = MapFeedback(feedback->session_state);
        if (!mapped) {
          return;
        }
        const auto state = weak_state.lock();
        if (!state) {
          return;
        }
        FeedbackCallback callback;
        {
          std::scoped_lock lock{state->mutex};
          if (state->teardown || !state->active ||
              state->active->generation != generation) {
            return;
          }
          callback = state->feedback_callback;
        }
        InvokeFeedback(callback, *mapped);
      };
  options.result_callback =
      [weak_state, generation](const GoalHandle::WrappedResult& wrapped) {
        const auto state = weak_state.lock();
        if (!state) {
          return;
        }
        TerminalCallback callback;
        {
          std::scoped_lock lock{state->mutex};
          if (state->teardown || !state->active ||
              state->active->generation != generation) {
            return;
          }
          callback = state->terminal_callback;
          state->active.reset();
        }
        if (!wrapped.result) {
          InvokeTerminal(callback, MissingResult());
          return;
        }
        InvokeTerminal(
            callback,
            NavigationTerminal{
                .outcome = wrapped.result->outcome,
                .reason_code = wrapped.result->reason_code});
      };

  try {
    static_cast<void>(state_->action_client->async_send_goal(goal, options));
  } catch (...) {
    std::scoped_lock lock{state_->mutex};
    if (state_->active && state_->active->generation == generation) {
      state_->active.reset();
    }
    throw;
  }
}

void NavigationClient::CancelActive() {
  GoalHandle::SharedPtr handle;
  std::uint64_t generation{};
  {
    std::scoped_lock lock{state_->mutex};
    if (state_->teardown || !state_->active) {
      return;
    }
    state_->active->cancel_requested = true;
    generation = state_->active->generation;
    handle = state_->active->handle;
  }
  if (handle) {
    SendCancel(state_, generation, handle);
  }
}

}  // namespace lunar::pure_exploration_ros
