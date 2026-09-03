#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include "lunar_incremental_navigation_core/global_route_planner.hpp"
#include "lunar_incremental_navigation_core/legged_local_planner.hpp"
#include "lunar_incremental_navigation_core/local_planning_window.hpp"
#include "lunar_incremental_navigation_core/local_target_selector.hpp"
#include "lunar_incremental_navigation_core/request_local_start_patch.hpp"
#include "lunar_incremental_navigation_core/types/path_reference.hpp"
#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"

namespace lunar::incremental_navigation {

enum class CoordinatorState : std::uint8_t {
  kIdle,
  kPlanning,
  kExecuting,
  kReplanning,
};

enum class CycleTrigger : std::uint8_t {
  kContinue,
  kSegmentEnd,
  kDeviation,
};

enum class CancelReason : std::uint8_t {
  kCanceled,
  kPreempted,
};

struct StateInput final {
  Pose2 base_link_pose;
};

struct SessionTerminal final {
  SessionId session_id;
  NavigateToPoseResult result;
  std::optional<PathReference> invalidated;
};

struct StartSessionResult final {
  bool accepted{};
  std::optional<SessionTerminal> replaced_session;
  std::optional<SessionTerminal> terminal;
};

struct CycleOutput final {
  NavigateToPoseFeedback feedback;
  GuidanceStatus guidance_status{GuidanceStatus::kUnavailable};
  std::optional<GlobalRoute> global_route;
  std::optional<PathReference> path_reference;
  std::optional<SessionTerminal> terminal;
};

struct PlanningSessionCoordinatorConfig final {
  double goal_position_tolerance_m{};
  double goal_yaw_tolerance_rad{};
  std::chrono::nanoseconds global_subdeadline{std::chrono::milliseconds(500)};
  NowFn now{[] { return SteadyClock::now(); }};
};

struct PlanningSessionPorts final {
  std::function<GlobalRouteResult(
      const GlobalGuidanceSnapshot&, Point2, Point2, SearchDeadline,
      const StopToken&)>
      plan_global;
  std::function<std::optional<LocalTarget>(
      const FineTraversabilitySnapshot&, const SparseGridGeometry&, Point2,
      const FinalGoal&,
      const std::optional<GlobalRoute>&)>
      select_target;
  std::function<StartPatchResult(
      std::shared_ptr<const FineTraversabilitySnapshot>,
      const SparseGridGeometry&, const Pose2&,
      const PlatformCapability&, const TraversabilityProfile&)>
      build_start_patch;
  std::function<LocalPlanResult(
      const RequestLocalPlanningView&, const Pose2&, const LocalTarget&,
      SearchDeadline, const StopToken&)>
      plan_wheel;
  std::function<LocalPlanResult(
      const RequestLocalPlanningView&, const Pose2&, const LocalTarget&,
      SearchDeadline, const StopToken&)>
      plan_legged;
};

class PlanningSessionCoordinator final {
 public:
  PlanningSessionCoordinator(
      PlatformCapability capability, TraversabilityProfile profile,
      PlanningSessionCoordinatorConfig config, PlanningSessionPorts ports);
  ~PlanningSessionCoordinator();

  PlanningSessionCoordinator(const PlanningSessionCoordinator&) = delete;
  PlanningSessionCoordinator& operator=(const PlanningSessionCoordinator&) =
      delete;
  PlanningSessionCoordinator(PlanningSessionCoordinator&&) noexcept;
  PlanningSessionCoordinator& operator=(PlanningSessionCoordinator&&) noexcept;

  [[nodiscard]] StartSessionResult Start(SessionId id, FinalGoal goal);
  [[nodiscard]] CycleOutput PlanCycle(StateInput state,
                                      SnapshotBundle snapshots,
                                      CycleTrigger trigger,
                                      SearchDeadline deadline);
  [[nodiscard]] std::optional<PathReference> OnFineSnapshot(
      std::shared_ptr<const FineTraversabilitySnapshot> fine);
  // Thread-safe interruption only. Start, Cancel and all state transitions
  // remain serialized on the coordinator thread after PlanCycle returns.
  [[nodiscard]] bool RequestStopCurrentCycle() noexcept;
  [[nodiscard]] SessionTerminal Cancel(CancelReason reason);

  [[nodiscard]] CoordinatorState state() const noexcept;
  [[nodiscard]] std::uint64_t planning_cycle() const noexcept;
  [[nodiscard]] std::uint64_t active_segment_revision() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lunar::incremental_navigation
