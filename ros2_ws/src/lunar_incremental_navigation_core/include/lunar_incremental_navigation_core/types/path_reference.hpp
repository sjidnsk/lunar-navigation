#pragma once

#include <cstdint>
#include <vector>

#include "lunar_incremental_navigation_core/types/geometry.hpp"
#include "lunar_incremental_navigation_core/types/planning_cycle.hpp"

namespace lunar::incremental_navigation {

enum class PathState : std::uint8_t {
  kActive = 0,
  kInvalidated = 1,
};

struct GeometricPath final {
  std::vector<Pose3> poses;
};

struct PathReference final {
  SessionId session_id;
  std::uint64_t segment_revision{0};
  std::uint64_t traversability_revision{0};
  PathState state{PathState::kInvalidated};
  bool reaches_final_goal{false};
  GeometricPath path;
};

[[nodiscard]] inline bool IsValidPathReference(
    const PathReference& reference) noexcept {
  switch (reference.state) {
    case PathState::kActive:
      return !reference.path.poses.empty();
    case PathState::kInvalidated:
      return reference.path.poses.empty();
  }
  return false;
}

[[nodiscard]] inline bool IsStrictlyNewerSegment(
    const PathReference& candidate,
    const PathReference& current) noexcept {
  return candidate.session_id == current.session_id &&
         candidate.segment_revision > current.segment_revision;
}

}  // namespace lunar::incremental_navigation
