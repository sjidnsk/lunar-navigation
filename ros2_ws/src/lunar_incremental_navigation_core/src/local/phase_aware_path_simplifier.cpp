#include "lunar_incremental_navigation_core/wheel_local_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

[[nodiscard]] bool SameDirectionCollinear(const PathPoint& first,
                                          const PathPoint& middle,
                                          const PathPoint& last) noexcept {
  if (first.phase != middle.phase || middle.phase != last.phase) {
    return false;
  }
  const double first_dx = middle.pose.position_m.x - first.pose.position_m.x;
  const double first_dy = middle.pose.position_m.y - first.pose.position_m.y;
  const double second_dx = last.pose.position_m.x - middle.pose.position_m.x;
  const double second_dy = last.pose.position_m.y - middle.pose.position_m.y;
  const double first_length_squared =
      std::fma(first_dx, first_dx, first_dy * first_dy);
  const double second_length_squared =
      std::fma(second_dx, second_dx, second_dy * second_dy);
  if (first_length_squared == 0.0 || second_length_squared == 0.0) {
    return false;
  }
  const double dot = std::fma(first_dx, second_dx, first_dy * second_dy);
  if (dot <= 0.0) {
    return false;
  }
  const double cross = std::fma(first_dx, second_dy,
                                -first_dy * second_dx);
  const double cross_scale =
      std::abs(first_dx * second_dy) +
      std::abs(first_dy * second_dx);
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * cross_scale;
  return std::abs(cross) <= tolerance;
}

[[nodiscard]] bool CollectPhaseCorners(
    const std::span<const PathPoint> phase_path,
    std::vector<PathPoint>& corners, const SearchControl* control) {
  corners.clear();
  corners.reserve(phase_path.size());
  for (const PathPoint& point : phase_path) {
    if (Interrupted(control)) {
      corners.clear();
      return false;
    }
    if (corners.size() >= 2U &&
        SameDirectionCollinear(corners[corners.size() - 2U], corners.back(),
                               point)) {
      corners.back() = point;
    } else {
      corners.push_back(point);
    }
  }
  return true;
}

[[nodiscard]] bool AdjacentEdgesAreCertified(
    const RequestLocalPlanningView& view,
    const std::span<const PathPoint> path, const SearchControl* control) {
  for (std::size_t index = 1U; index < path.size(); ++index) {
    if (!HasLineOfSight(view, path[index - 1U], path[index], control)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool SimplifyOnePhase(
    const RequestLocalPlanningView& view,
    const std::span<const PathPoint> phase_path,
    std::vector<PathPoint>& output, const SearchControl* control) {
  if (phase_path.empty()) {
    return true;
  }
  std::vector<PathPoint> corners;
  if (!CollectPhaseCorners(phase_path, corners, control)) {
    return false;
  }
  std::span<const PathPoint> candidate_path(corners);
  if (!AdjacentEdgesAreCertified(view, candidate_path, control)) {
    if (Interrupted(control)) {
      return false;
    }
    candidate_path = phase_path;
    if (!AdjacentEdgesAreCertified(view, candidate_path, control)) {
      return false;
    }
  }
  if (output.empty() ||
      output.back().pose != candidate_path.front().pose ||
      output.back().phase != candidate_path.front().phase) {
    output.push_back(candidate_path.front());
  }
  std::size_t anchor = 0U;
  while (anchor + 1U < candidate_path.size()) {
    if (Interrupted(control)) {
      return false;
    }
    std::size_t selected = anchor + 1U;
    for (std::size_t candidate = candidate_path.size() - 1U;
         candidate > anchor + 1U; --candidate) {
      if (HasLineOfSight(view, candidate_path[anchor], candidate_path[candidate],
                         control)) {
        selected = candidate;
        break;
      }
      if (Interrupted(control)) {
        return false;
      }
    }
    output.push_back(candidate_path[selected]);
    anchor = selected;
  }
  return true;
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
    if (!SimplifyOnePhase(view, raw_path.subspan(begin, end - begin),
                          simplified, &control)) {
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
