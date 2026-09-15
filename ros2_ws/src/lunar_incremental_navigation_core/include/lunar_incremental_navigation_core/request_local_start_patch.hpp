#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "lunar_incremental_navigation_core/fine_traversability_builder.hpp"
#include "lunar_incremental_navigation_core/local_planning.hpp"
#include "lunar_incremental_navigation_core/types/platform_capability.hpp"

namespace lunar::incremental_navigation {

struct StartPatchResult final {
  enum class Status {
    kNotNeeded,
    kReady,
    kStartBlocked,
    kUnresolved,
  };

  Status status{Status::kUnresolved};
  std::shared_ptr<const RequestLocalPlanningView> view;
  std::size_t assumed_cells{};
};

struct StartConnection final {
  GridIndex index;
  StartPhase phase{StartPhase::kNormal};
};

struct StartConnectionsResult final {
  StartPatchResult::Status status{StartPatchResult::Status::kUnresolved};
  std::vector<StartConnection> connections;
};

class RequestLocalStartPatchBuilder final {
 public:
  // Stationary wheel support is bounded at the actual anchor and does not
  // certify any translation endpoint or mutate the persistent evidence.
  [[nodiscard]] StartPatchResult BuildStationary(
      std::shared_ptr<const FineTraversabilitySnapshot> fine, const Pose2& p0,
      const PlatformCapability& capability, const TraversabilityProfile& profile) const;
  [[nodiscard]] StartPatchResult Build(
      std::shared_ptr<const FineTraversabilitySnapshot> fine,
      const SparseGridGeometry& local_window, const Pose2& p0,
      const PlatformCapability& capability,
      const TraversabilityProfile& profile) const;

  [[nodiscard]] StartConnectionsResult BuildStartConnections(
      std::shared_ptr<const FineTraversabilitySnapshot> fine,
      const SparseGridGeometry& local_window, const Pose2& p0,
      const PlatformCapability& capability,
      const TraversabilityProfile& profile) const;

  // Compatibility entry point for callers that intentionally use the full
  // fine snapshot. Session planning always supplies a bounded window.
  [[nodiscard]] StartPatchResult Build(
      std::shared_ptr<const FineTraversabilitySnapshot> fine,
      const Pose2& p0, const PlatformCapability& capability,
      const TraversabilityProfile& profile) const;
};

}  // namespace lunar::incremental_navigation
