#include "lunar_incremental_navigation_core/planning_session_coordinator.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <utility>

#include "local/grid_supercover.hpp"

namespace lunar::incremental_navigation {
namespace {

[[nodiscard]] bool FinitePose(const Pose2& pose) noexcept {
  return std::isfinite(pose.position_m.x) &&
         std::isfinite(pose.position_m.y) && std::isfinite(pose.yaw_rad);
}

[[nodiscard]] double NormalizeAngle(double angle) noexcept {
  constexpr double kPi = 3.14159265358979323846;
  constexpr double kTwoPi = 2.0 * kPi;
  angle = std::fmod(angle + kPi, kTwoPi);
  if (angle < 0.0) {
    angle += kTwoPi;
  }
  return angle - kPi;
}

[[nodiscard]] double GoalPositionTolerance(
    const FineTraversabilitySnapshot& fine,
    const double profile_tolerance_m) noexcept {
  return std::max(0.5 * fine.geometry().resolution_m(),
                  profile_tolerance_m);
}

[[nodiscard]] SessionState FeedbackState(
    const CoordinatorState state) noexcept {
  switch (state) {
    case CoordinatorState::kExecuting:
      return SessionState::kExecuting;
    case CoordinatorState::kReplanning:
      return SessionState::kReplanning;
    case CoordinatorState::kIdle:
    case CoordinatorState::kPlanning:
      return SessionState::kPlanning;
  }
  return SessionState::kPlanning;
}

class CurrentCycleStopRegistration final {
 public:
  explicit CurrentCycleStopRegistration(
      std::shared_ptr<std::stop_source>& slot)
      : slot_(slot), source_(std::make_shared<std::stop_source>()) {
    std::atomic_store_explicit(&slot_, source_, std::memory_order_release);
  }

  ~CurrentCycleStopRegistration() {
    auto expected = source_;
    static_cast<void>(std::atomic_compare_exchange_strong_explicit(
        &slot_, &expected, std::shared_ptr<std::stop_source>{},
        std::memory_order_acq_rel, std::memory_order_acquire));
  }

  [[nodiscard]] const StopToken& token() {
    token_ = source_->get_token();
    return token_;
  }

  [[nodiscard]] bool stop_requested() const noexcept {
    return source_->stop_requested();
  }

 private:
  std::shared_ptr<std::stop_source>& slot_;
  std::shared_ptr<std::stop_source> source_;
  StopToken token_;
};

}  // namespace

struct PlanningSessionCoordinator::Impl final {
  Impl(PlatformCapability capability_value,
       TraversabilityProfile profile_value,
       PlanningSessionCoordinatorConfig config_value,
       PlanningSessionPorts ports_value)
      : capability(std::move(capability_value)),
        profile(std::move(profile_value)),
        config(std::move(config_value)),
        ports(std::move(ports_value)) {}

  PlatformCapability capability;
  TraversabilityProfile profile;
  PlanningSessionCoordinatorConfig config;
  PlanningSessionPorts ports;
  CoordinatorState state{CoordinatorState::kIdle};
  std::optional<SessionId> session_id;
  std::optional<FinalGoal> goal;
  std::uint64_t planning_cycle{};
  std::uint64_t segment_revision{};
  std::uint64_t captured_fine_revision{};
  std::uint64_t last_safety_evaluated_fine_revision{};
  std::optional<PathReference> active_path;
  std::shared_ptr<const RequestLocalPlanningView> active_view;
  std::optional<StateInput> latest_state;
  std::shared_ptr<std::stop_source> current_cycle_stop;

  [[nodiscard]] SteadyClock::time_point Now() const {
    return config.now ? config.now() : SteadyClock::now();
  }

  [[nodiscard]] NavigateToPoseFeedback Feedback(std::string reason = {}) const {
    return NavigateToPoseFeedback{
        .session_state = FeedbackState(state),
        .planning_cycle = planning_cycle,
        .active_segment_revision =
            active_path ? active_path->segment_revision : 0U,
        .reason_code = std::move(reason),
    };
  }

  [[nodiscard]] std::optional<PathReference> Invalidate(
      const std::uint64_t evaluated_fine_revision) {
    if (!active_path) {
      return std::nullopt;
    }
    PathReference invalidated{
        .session_id = active_path->session_id,
        .segment_revision = active_path->segment_revision,
        .traversability_revision = evaluated_fine_revision,
        .state = PathState::kInvalidated,
        .reaches_final_goal = false,
    };
    active_path.reset();
    active_view.reset();
    return invalidated;
  }

  [[nodiscard]] SessionTerminal Finish(
      const SessionOutcome outcome, std::string reason_code,
      const std::uint64_t evaluated_fine_revision) {
    SessionTerminal terminal{
        .session_id = session_id.value_or(SessionId{}),
        .result = NavigateToPoseResult{
            .outcome = outcome,
            .reason_code = std::move(reason_code),
            .last_segment_revision = segment_revision,
        },
        .invalidated = Invalidate(evaluated_fine_revision),
    };
    state = CoordinatorState::kIdle;
    session_id.reset();
    goal.reset();
    latest_state.reset();
    return terminal;
  }

  [[nodiscard]] bool GoalReached(const StateInput& input,
                                 const FineTraversabilitySnapshot& fine)
      const noexcept {
    if (!active_path || !active_path->reaches_final_goal || !goal) {
      return false;
    }
    const double position_tolerance_m = GoalPositionTolerance(
        fine, config.goal_position_tolerance_m);
    if (std::hypot(input.base_link_pose.position_m.x - goal->target_x_m,
                   input.base_link_pose.position_m.y - goal->target_y_m) >
        position_tolerance_m) {
      return false;
    }
    return !goal->has_target_yaw ||
           std::abs(NormalizeAngle(input.base_link_pose.yaw_rad -
                                   goal->target_yaw_rad)) <=
               config.goal_yaw_tolerance_rad;
  }

  struct RemainingPathEvaluation final {
    bool complete{};
    bool blocked{};
  };

  [[nodiscard]] RemainingPathEvaluation EvaluateRemainingPath(
      const FineTraversabilitySnapshot& fine) const noexcept {
    if (!active_path || active_path->path.poses.empty()) {
      return {};
    }
    const auto& poses = active_path->path.poses;
    std::size_t first_segment = 0U;
    double first_fraction = 0.0;
    if (latest_state.has_value()) {
      double nearest = std::numeric_limits<double>::infinity();
      for (std::size_t index = 0U; index + 1U < poses.size(); ++index) {
        const Vec3& begin = poses[index].position_m;
        const Vec3& end = poses[index + 1U].position_m;
        const double dx = end.x - begin.x;
        const double dy = end.y - begin.y;
        const double length_squared = dx * dx + dy * dy;
        const double fraction =
            length_squared > 0.0
                ? std::clamp(((latest_state->base_link_pose.position_m.x -
                               begin.x) *
                                  dx +
                              (latest_state->base_link_pose.position_m.y -
                               begin.y) *
                                  dy) /
                                 length_squared,
                             0.0, 1.0)
                : 0.0;
        const double projected_x = std::lerp(begin.x, end.x, fraction);
        const double projected_y = std::lerp(begin.y, end.y, fraction);
        const double distance = std::hypot(
            projected_x - latest_state->base_link_pose.position_m.x,
            projected_y - latest_state->base_link_pose.position_m.y);
        if (distance < nearest) {
          nearest = distance;
          first_segment = index;
          first_fraction = fraction;
        }
      }
    }

    bool blocked = false;
    const auto visit = [&](const GridIndex index) {
      if (fine.State(index) == FineCellState::kBlocked) {
        blocked = true;
        return false;
      }
      return true;
    };
    if (poses.size() == 1U) {
      const bool complete = local::VisitSupercoverCells(
          fine.geometry(), poses.front().position_m,
          poses.front().position_m, visit);
      return {.complete = complete, .blocked = blocked};
    }

    for (std::size_t index = first_segment; index + 1U < poses.size();
         ++index) {
      const Vec3& begin = poses[index].position_m;
      const Vec3& end = poses[index + 1U].position_m;
      const double begin_fraction =
          index == first_segment ? first_fraction : 0.0;
      const Vec3 remaining_begin{
          .x = std::lerp(begin.x, end.x, begin_fraction),
          .y = std::lerp(begin.y, end.y, begin_fraction),
          .z = std::lerp(begin.z, end.z, begin_fraction),
      };
      if (!local::VisitSupercoverCells(fine.geometry(), remaining_begin, end,
                                       visit)) {
        return {.complete = false, .blocked = blocked};
      }
    }
    return {.complete = true, .blocked = false};
  }
};

PlanningSessionCoordinator::PlanningSessionCoordinator(
    PlatformCapability capability, TraversabilityProfile profile,
    PlanningSessionCoordinatorConfig config, PlanningSessionPorts ports)
    : impl_(std::make_unique<Impl>(
          std::move(capability), std::move(profile), std::move(config),
          std::move(ports))) {
  if (!std::isfinite(impl_->config.goal_position_tolerance_m) ||
      impl_->config.goal_position_tolerance_m < 0.0 ||
      !std::isfinite(impl_->config.goal_yaw_tolerance_rad) ||
      impl_->config.goal_yaw_tolerance_rad < 0.0 ||
      !std::isfinite(impl_->config.local_window_size_m) ||
      impl_->config.local_window_size_m <= 0.0 ||
      impl_->config.global_subdeadline < std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument(
        "planning session tolerances, local window, and subdeadline are invalid");
  }
}

PlanningSessionCoordinator::~PlanningSessionCoordinator() = default;
PlanningSessionCoordinator::PlanningSessionCoordinator(
    PlanningSessionCoordinator&&) noexcept = default;
PlanningSessionCoordinator& PlanningSessionCoordinator::operator=(
    PlanningSessionCoordinator&&) noexcept = default;

StartSessionResult PlanningSessionCoordinator::Start(const SessionId id,
                                                     const FinalGoal goal) {
  if (!IsValidFinalGoal(goal)) {
    return StartSessionResult{
        .terminal = SessionTerminal{
            .session_id = id,
            .result = NavigateToPoseResult{
                .outcome = SessionOutcome::kInvalidGoal,
                .reason_code = "INVALID_GOAL",
            },
        },
    };
  }

  StartSessionResult result{.accepted = true};
  if (impl_->state != CoordinatorState::kIdle && impl_->session_id) {
    result.replaced_session = impl_->Finish(
        SessionOutcome::kCanceled, "PREEMPTED",
        impl_->last_safety_evaluated_fine_revision);
  }

  impl_->session_id = id;
  impl_->goal = goal;
  impl_->planning_cycle = 0U;
  impl_->segment_revision = 0U;
  impl_->captured_fine_revision = 0U;
  impl_->last_safety_evaluated_fine_revision = 0U;
  impl_->active_path.reset();
  impl_->active_view.reset();
  impl_->latest_state.reset();
  impl_->state = CoordinatorState::kPlanning;
  return result;
}

CycleOutput PlanningSessionCoordinator::PlanCycle(
    const StateInput state, const SnapshotBundle snapshots,
    const CycleTrigger trigger, const SearchDeadline deadline) {
  CycleOutput output;
  if (impl_->state == CoordinatorState::kIdle || !impl_->session_id ||
      !impl_->goal) {
    return output;
  }
  CurrentCycleStopRegistration cycle_stop(impl_->current_cycle_stop);
  ++impl_->planning_cycle;
  impl_->latest_state = state;
  if (snapshots.fine) {
    impl_->captured_fine_revision = std::max(
        impl_->captured_fine_revision,
        snapshots.fine->fine_traversability_revision());
  }

  if (!FinitePose(state.base_link_pose)) {
    output.terminal = impl_->Finish(SessionOutcome::kInternalError,
                                    "INVALID_STATE",
                                    impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback("INVALID_STATE");
    return output;
  }

  if (snapshots.fine && impl_->GoalReached(state, *snapshots.fine)) {
    output.terminal = impl_->Finish(SessionOutcome::kGoalReached,
                                    "GOAL_REACHED",
                                    impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback("GOAL_REACHED");
    return output;
  }

  if (trigger == CycleTrigger::kDeviation && impl_->active_path) {
    output.path_reference =
        impl_->Invalidate(impl_->last_safety_evaluated_fine_revision);
    impl_->state = CoordinatorState::kReplanning;
    output.feedback = impl_->Feedback("PATH_DEVIATED");
    return output;
  }

  const bool planning_required =
      impl_->state == CoordinatorState::kPlanning ||
      impl_->state == CoordinatorState::kReplanning ||
      trigger == CycleTrigger::kSegmentEnd;
  if (!planning_required) {
    output.feedback = impl_->Feedback("EXECUTING");
    return output;
  }
  if (trigger == CycleTrigger::kSegmentEnd) {
    impl_->state = CoordinatorState::kReplanning;
  }

  if (!snapshots.fine) {
    output.terminal = impl_->Finish(SessionOutcome::kMapUnavailable,
                                    "MAP_UNAVAILABLE",
                                    impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback("MAP_UNAVAILABLE");
    return output;
  }

  try {
    if (impl_->Now() >= deadline) {
      output.terminal = impl_->Finish(SessionOutcome::kTimeout, "TIMEOUT",
                                      impl_->last_safety_evaluated_fine_revision);
      output.feedback = impl_->Feedback("TIMEOUT");
      return output;
    }
  } catch (...) {
    output.terminal = impl_->Finish(SessionOutcome::kInternalError,
                                    "INTERNAL_ERROR",
                                    impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback("INTERNAL_ERROR");
    return output;
  }

  const Point2 start = state.base_link_pose.position_m;
  const Point2 final_goal{.x = impl_->goal->target_x_m,
                          .y = impl_->goal->target_y_m};
  std::optional<GlobalRoute> guidance_route;
  output.guidance_status = GuidanceStatus::kUnavailable;
  if (snapshots.guidance && impl_->ports.plan_global) {
    try {
      const SearchDeadline global_deadline = std::min(
          deadline, impl_->Now() + impl_->config.global_subdeadline);
      GlobalRouteResult global = impl_->ports.plan_global(
          *snapshots.guidance, start, final_goal, global_deadline,
          cycle_stop.token());
      output.guidance_status = global.status;
      if (global.status == GuidanceStatus::kAvailable && global.route) {
        guidance_route = std::move(global.route);
        output.global_route = guidance_route;
      }
    } catch (...) {
      output.guidance_status = GuidanceStatus::kUnavailable;
    }
  }
  if (cycle_stop.stop_requested()) {
    output.feedback = impl_->Feedback("CYCLE_STOPPED");
    return output;
  }

  if (!impl_->ports.select_target || !impl_->ports.build_start_patch) {
    output.terminal = impl_->Finish(SessionOutcome::kInternalError,
                                    "INTERNAL_ERROR",
                                    impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback("INTERNAL_ERROR");
    return output;
  }

  const LocalPlanningWindowResult local_window = BuildLocalPlanningWindow(
      *snapshots.fine, start, impl_->config.local_window_size_m);
  if (!local_window.geometry) {
    const std::string reason =
        local_window.status == LocalPlanningWindowStatus::kCapacityExceeded
            ? "LOCAL_WINDOW_CAPACITY_EXCEEDED"
            : "START_OUTSIDE_FINE_MAP";
    output.terminal = impl_->Finish(
        SessionOutcome::kNoPath, reason,
        impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback(reason);
    return output;
  }

  std::optional<LocalTarget> target;
  try {
    target = impl_->ports.select_target(*snapshots.fine, *local_window.geometry, start,
                                        *impl_->goal, guidance_route);
  } catch (...) {
    output.terminal = impl_->Finish(SessionOutcome::kInternalError,
                                    "INTERNAL_ERROR",
                                    impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback("INTERNAL_ERROR");
    return output;
  }
  if (cycle_stop.stop_requested()) {
    output.feedback = impl_->Feedback("CYCLE_STOPPED");
    return output;
  }
  if (!target) {
    output.terminal = impl_->Finish(SessionOutcome::kNoPath, "NO_PATH",
                                    impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback("NO_PATH");
    return output;
  }
  // Keep planning termination and execution completion on one internal
  // contract.  Obstacle inflation has already been consumed by the fine-map
  // state and must not enlarge this arrival region.
  target->position_tolerance_m = GoalPositionTolerance(
      *snapshots.fine, impl_->config.goal_position_tolerance_m);

  StartPatchResult patch;
  try {
    patch = impl_->ports.build_start_patch(
        snapshots.fine, *local_window.geometry, state.base_link_pose, impl_->capability,
        impl_->profile);
  } catch (...) {
    output.terminal = impl_->Finish(SessionOutcome::kInternalError,
                                    "INTERNAL_ERROR",
                                    impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback("INTERNAL_ERROR");
    return output;
  }
  if (cycle_stop.stop_requested()) {
    output.feedback = impl_->Feedback("CYCLE_STOPPED");
    return output;
  }
  if (patch.status == StartPatchResult::Status::kStartBlocked) {
    output.terminal = impl_->Finish(SessionOutcome::kNoPath, "START_BLOCKED",
                                    impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback("START_BLOCKED");
    return output;
  }
  if (patch.status == StartPatchResult::Status::kUnresolved || !patch.view) {
    output.terminal = impl_->Finish(SessionOutcome::kNoPath,
                                    "START_BLIND_ZONE_UNRESOLVED",
                                    impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback("START_BLIND_ZONE_UNRESOLVED");
    return output;
  }

  LocalPlanResult local;
  const PlatformType platform = CapabilityPlatform(impl_->capability);
  try {
    if (platform == PlatformType::kWheeled && impl_->ports.plan_wheel) {
      local = impl_->ports.plan_wheel(
          *patch.view, state.base_link_pose, *target, deadline,
          cycle_stop.token());
    } else if (platform == PlatformType::kLegged &&
               impl_->ports.plan_legged) {
      local = impl_->ports.plan_legged(
          *patch.view, state.base_link_pose, *target, deadline,
          cycle_stop.token());
    } else {
      output.terminal = impl_->Finish(SessionOutcome::kInternalError,
                                      "INTERNAL_ERROR",
                                      impl_->last_safety_evaluated_fine_revision);
      output.feedback = impl_->Feedback("INTERNAL_ERROR");
      return output;
    }
  } catch (...) {
    output.terminal = impl_->Finish(SessionOutcome::kInternalError,
                                    "INTERNAL_ERROR",
                                    impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback("INTERNAL_ERROR");
    return output;
  }
  if (cycle_stop.stop_requested()) {
    output.feedback = impl_->Feedback("CYCLE_STOPPED");
    return output;
  }

  try {
    if (impl_->Now() >= deadline &&
        local.status != LocalPlanResult::Status::kCanceled) {
      local.status = LocalPlanResult::Status::kTimeout;
    }
  } catch (...) {
    local.status = LocalPlanResult::Status::kTimeout;
  }

  if (local.status != LocalPlanResult::Status::kPlanFound) {
    SessionOutcome outcome = SessionOutcome::kNoPath;
    std::string reason = local.reason_code.empty() ? "NO_PATH"
                                                    : local.reason_code;
    if (local.status == LocalPlanResult::Status::kTimeout) {
      outcome = SessionOutcome::kTimeout;
      reason = "TIMEOUT";
    } else if (local.status == LocalPlanResult::Status::kCanceled) {
      outcome = SessionOutcome::kCanceled;
      reason = "CANCELED";
    }
    output.terminal =
        impl_->Finish(outcome, reason,
                      impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback(reason);
    return output;
  }

  const std::vector<PathPoint>& selected_path =
      local.path.empty() ? local.raw_path : local.path;
  if (selected_path.empty()) {
    output.terminal = impl_->Finish(SessionOutcome::kInternalError,
                                    "EMPTY_LOCAL_PATH",
                                    impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback("EMPTY_LOCAL_PATH");
    return output;
  }

  PathReference reference{
      .session_id = *impl_->session_id,
      .segment_revision = impl_->segment_revision + 1U,
      .traversability_revision =
          snapshots.fine->fine_traversability_revision(),
      .state = PathState::kActive,
      .reaches_final_goal = local.reaches_final_goal,
  };
  reference.path.poses.reserve(selected_path.size());
  for (const PathPoint& point : selected_path) {
    reference.path.poses.push_back(point.pose);
  }
  try {
    if (impl_->Now() >= deadline) {
      output.terminal = impl_->Finish(
          SessionOutcome::kTimeout, "TIMEOUT",
          impl_->last_safety_evaluated_fine_revision);
      output.feedback = impl_->Feedback("TIMEOUT");
      return output;
    }
  } catch (...) {
    output.terminal = impl_->Finish(
        SessionOutcome::kTimeout, "TIMEOUT",
        impl_->last_safety_evaluated_fine_revision);
    output.feedback = impl_->Feedback("TIMEOUT");
    return output;
  }
  impl_->last_safety_evaluated_fine_revision =
      snapshots.fine->fine_traversability_revision();
  impl_->segment_revision = reference.segment_revision;
  impl_->active_path = reference;
  impl_->active_view = std::move(patch.view);
  impl_->state = CoordinatorState::kExecuting;
  output.path_reference = std::move(reference);
  output.feedback = impl_->Feedback("PLAN_FOUND");
  return output;
}

std::optional<PathReference> PlanningSessionCoordinator::OnFineSnapshot(
    std::shared_ptr<const FineTraversabilitySnapshot> fine) {
  if (fine) {
    impl_->captured_fine_revision = std::max(
        impl_->captured_fine_revision,
        fine->fine_traversability_revision());
  }
  if (!fine || impl_->state != CoordinatorState::kExecuting ||
      !impl_->active_path ||
      fine->fine_traversability_revision() <=
          impl_->last_safety_evaluated_fine_revision) {
    return std::nullopt;
  }
  const Impl::RemainingPathEvaluation evaluation =
      impl_->EvaluateRemainingPath(*fine);
  if (!evaluation.complete && !evaluation.blocked) {
    return std::nullopt;
  }
  impl_->last_safety_evaluated_fine_revision =
      fine->fine_traversability_revision();
  if (!evaluation.blocked) {
    return std::nullopt;
  }
  auto invalidated =
      impl_->Invalidate(impl_->last_safety_evaluated_fine_revision);
  impl_->state = CoordinatorState::kReplanning;
  return invalidated;
}

bool PlanningSessionCoordinator::RequestStopCurrentCycle() noexcept {
  const auto source = std::atomic_load_explicit(
      &impl_->current_cycle_stop, std::memory_order_acquire);
  return source && source->request_stop();
}

SessionTerminal PlanningSessionCoordinator::Cancel(
    const CancelReason reason) {
  if (impl_->state == CoordinatorState::kIdle || !impl_->session_id) {
    return SessionTerminal{
        .result = NavigateToPoseResult{
            .outcome = SessionOutcome::kCanceled,
            .reason_code = reason == CancelReason::kPreempted ? "PREEMPTED"
                                                               : "CANCELED"},
    };
  }
  return impl_->Finish(SessionOutcome::kCanceled,
                       reason == CancelReason::kPreempted ? "PREEMPTED"
                                                          : "CANCELED",
                       impl_->last_safety_evaluated_fine_revision);
}

CoordinatorState PlanningSessionCoordinator::state() const noexcept {
  return impl_->state;
}

std::uint64_t PlanningSessionCoordinator::planning_cycle() const noexcept {
  return impl_->planning_cycle;
}

std::uint64_t PlanningSessionCoordinator::active_segment_revision()
    const noexcept {
  return impl_->active_path ? impl_->active_path->segment_revision : 0U;
}

}  // namespace lunar::incremental_navigation
