#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"

#include <algorithm>
#include <vector>

#include "local/grid_supercover.hpp"

namespace lunar::incremental_navigation {
namespace {

[[nodiscard]] bool Interrupted(const SearchControl* control) {
  return control != nullptr &&
         (control->canceled() || control->expired());
}

[[nodiscard]] bool Allowed(const RequestLocalPlanningView& view,
                           const GridIndex index,
                           const StartPhase phase) noexcept {
  const LocalCellSource source = view.Source(index);
  return phase == StartPhase::kStartPrefix
             ? source == LocalCellSource::kStartAssumedFree
             : source == LocalCellSource::kEvidenceFree;
}

[[nodiscard]] bool HasLineOfSight(const RequestLocalPlanningView& view,
                                  const PathPoint& from,
                                  const PathPoint& to,
                                  const SearchControl* control) {
  if (Interrupted(control)) {
    return false;
  }
  if (from.phase != to.phase) {
    return false;
  }
  const SparseGridGeometry& geometry = view.geometry();
  return local::VisitSupercoverCells(
      geometry, from.pose.position_m, to.pose.position_m,
      [&](const GridIndex index) {
        return !Interrupted(control) && Allowed(view, index, from.phase);
      });
}

void SimplifyOnePhase(const RequestLocalPlanningView& view,
                      const std::span<const PathPoint> phase_path,
                      std::vector<PathPoint>& output,
                      const SearchControl* control) {
  if (phase_path.empty()) {
    return;
  }
  if (output.empty() ||
      output.back().pose != phase_path.front().pose ||
      output.back().phase != phase_path.front().phase) {
    output.push_back(phase_path.front());
  }
  std::size_t anchor = 0U;
  while (anchor + 1U < phase_path.size()) {
    if (Interrupted(control)) {
      return;
    }
    std::size_t selected = anchor + 1U;
    for (std::size_t candidate = phase_path.size() - 1U;
         candidate > anchor + 1U; --candidate) {
      if (HasLineOfSight(view, phase_path[anchor], phase_path[candidate],
                         control)) {
        selected = candidate;
        break;
      }
      if (Interrupted(control)) {
        return;
      }
    }
    output.push_back(phase_path[selected]);
    anchor = selected;
  }
}

}  // namespace

std::vector<PathPoint> SimplifyPhaseAwarePath(
    const RequestLocalPlanningView& view,
    const std::span<const PathPoint> raw_path,
    const SearchControl& control) {
  if (Interrupted(&control)) {
    return {};
  }
  if (raw_path.size() < 3U) {
    return {raw_path.begin(), raw_path.end()};
  }
  std::vector<PathPoint> simplified;
  simplified.reserve(raw_path.size());
  std::size_t begin = 0U;
  while (begin < raw_path.size()) {
    if (Interrupted(&control)) {
      return {};
    }
    std::size_t end = begin + 1U;
    while (end < raw_path.size() &&
           raw_path[end].phase == raw_path[begin].phase) {
      ++end;
    }
    SimplifyOnePhase(view, raw_path.subspan(begin, end - begin), simplified,
                     &control);
    if (Interrupted(&control)) {
      return {};
    }
    begin = end;
  }
  return simplified;
}

std::vector<PathPoint> SimplifyPhaseAwarePath(
    const RequestLocalPlanningView& view,
    const std::span<const PathPoint> raw_path) {
  return SimplifyPhaseAwarePath(
      view, raw_path,
      SearchControl{.deadline = SteadyClock::time_point::max()});
}

}  // namespace lunar::incremental_navigation
