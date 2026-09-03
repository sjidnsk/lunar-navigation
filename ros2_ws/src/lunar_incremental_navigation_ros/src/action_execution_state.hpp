#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>

namespace lunar::incremental_navigation_ros::detail {

class ActionExecutionState final {
 public:
  enum class Phase {
    kIdle,
    kRunning,
    kTerminalClaimed,
    kTerminalDelivered,
    kTerminalFailed,
  };

  struct TerminalClaim final {
    bool claimed{};
    bool client_cancel_accepted{};
    bool replacement_accepted{};
  };

  [[nodiscard]] bool MayAccept(const bool goal_present,
                               const bool replace_active) const noexcept {
    return goal_present &&
           (phase_ == Phase::kIdle ||
            (phase_ == Phase::kRunning && !cancel_accepted_ &&
             replace_active));
  }

  [[nodiscard]] std::optional<std::uint64_t> TryStart(
      const void* goal_identity) noexcept {
    if (phase_ != Phase::kIdle || goal_identity == nullptr) {
      return std::nullopt;
    }
    active_goal_identity_ = goal_identity;
    phase_ = Phase::kRunning;
    cancel_accepted_ = false;
    replacement_requested_ = false;
    return ++generation_;
  }

  [[nodiscard]] bool AcceptCancel(const void* goal_identity,
                                  const bool worker_present) noexcept {
    if (phase_ != Phase::kRunning || !worker_present ||
        goal_identity != active_goal_identity_) {
      return false;
    }
    cancel_accepted_ = true;
    return true;
  }

  [[nodiscard]] bool RequestReplacement() noexcept {
    if (phase_ != Phase::kRunning) {
      return false;
    }
    replacement_requested_ = true;
    return true;
  }

  [[nodiscard]] TerminalClaim ClaimTerminal(
      const std::uint64_t generation, const void* goal_identity) noexcept {
    if (!Matches(generation, goal_identity) || phase_ != Phase::kRunning) {
      return {};
    }
    phase_ = Phase::kTerminalClaimed;
    return {.claimed = true,
            .client_cancel_accepted = cancel_accepted_,
            .replacement_accepted = replacement_requested_};
  }

  void MarkTerminalDelivered(const std::uint64_t generation,
                             const void* goal_identity) noexcept {
    if (Matches(generation, goal_identity) &&
        phase_ == Phase::kTerminalClaimed) {
      phase_ = Phase::kTerminalDelivered;
    }
  }

  void MarkTerminalFailed(const std::uint64_t generation,
                          const void* goal_identity) noexcept {
    if (Matches(generation, goal_identity)) {
      phase_ = Phase::kTerminalFailed;
    }
  }

  void Finish(const std::uint64_t generation,
              const void* goal_identity) noexcept {
    if (!Matches(generation, goal_identity)) {
      return;
    }
    if (phase_ == Phase::kTerminalDelivered) {
      phase_ = Phase::kIdle;
      active_goal_identity_ = nullptr;
      cancel_accepted_ = false;
      replacement_requested_ = false;
    } else if (phase_ == Phase::kRunning ||
               phase_ == Phase::kTerminalClaimed) {
      phase_ = Phase::kTerminalFailed;
    }
  }

  [[nodiscard]] Phase phase() const noexcept { return phase_; }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }

 private:
  [[nodiscard]] bool Matches(const std::uint64_t generation,
                             const void* goal_identity) const noexcept {
    return generation == generation_ && goal_identity != nullptr &&
           goal_identity == active_goal_identity_;
  }

  Phase phase_{Phase::kIdle};
  const void* active_goal_identity_{};
  std::uint64_t generation_{};
  bool cancel_accepted_{};
  bool replacement_requested_{};
};

class CancelTransitionWaiter final {
 public:
  enum class Result {
    kCanceling,
    kStopped,
    kPredicateError,
  };

  template <typename IsCanceling, typename ContextIsValid,
            typename TeardownRequested>
  [[nodiscard]] Result Wait(IsCanceling&& is_canceling,
                            ContextIsValid&& context_is_valid,
                            TeardownRequested&& teardown_requested) noexcept {
    using namespace std::chrono_literals;
    for (;;) {
      try {
        if (teardown_requested() || !context_is_valid()) {
          return Result::kStopped;
        }
        if (is_canceling()) {
          return Result::kCanceling;
        }
      } catch (...) {
        return Result::kPredicateError;
      }
      std::unique_lock lock{mutex_};
      wake_.wait_for(lock, 2ms);
    }
  }

  void Notify() noexcept { wake_.notify_all(); }

 private:
  std::mutex mutex_;
  std::condition_variable wake_;
};

}  // namespace lunar::incremental_navigation_ros::detail
