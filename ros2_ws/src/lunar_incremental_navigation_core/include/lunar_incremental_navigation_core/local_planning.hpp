#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "lunar_incremental_navigation_core/global_guidance_snapshot.hpp"
#include "lunar_incremental_navigation_core/search_control.hpp"
#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"
#include "lunar_incremental_navigation_core/types/grid_geometry.hpp"

namespace lunar::incremental_navigation {

enum class LocalCellSource : std::uint8_t {
  kUnknown,
  kEvidenceFree,
  kStartAssumedFree,
  kBlocked,
};

enum class StartPhase : std::uint8_t {
  kStartPrefix,
  kNormal,
};

struct PathPoint final {
  // Wheel/legged local navigation is an SE(2) map-plane reference. Pose3 is
  // retained as the shared container, and position_m.z is always 0.0.
  Pose3 pose;
  StartPhase phase{StartPhase::kNormal};
};

struct LocalPlanResult final {
  enum class Status : std::uint8_t {
    kPlanFound,
    kNoPath,
    kTimeout,
    kCanceled,
  };

  Status status{Status::kNoPath};
  std::vector<PathPoint> raw_path;
  std::vector<PathPoint> path;
  bool reaches_final_goal{};
  // Empty on success and generic search failure. A nonempty value preserves a
  // certified local failure cause for the session diagnostic.
  std::string reason_code;
  SearchStatistics statistics;
  std::chrono::nanoseconds postprocess_elapsed{};
};

struct LocalCellOverride final {
  GridIndex index;
  LocalCellSource source{LocalCellSource::kUnknown};
  double traversal_cost{};
};

class RequestLocalPlanningView final {
 public:
  RequestLocalPlanningView(
      std::shared_ptr<const FineTraversabilitySnapshot> base,
      const Pose2 patch_anchor, const double start_patch_radius_m,
      std::vector<LocalCellOverride> overrides)
      : RequestLocalPlanningView(base,
                                 base ? base->geometry()
                                      : SparseGridGeometry{},
                                 patch_anchor, start_patch_radius_m,
                                 std::move(overrides)) {}

  RequestLocalPlanningView(
      std::shared_ptr<const FineTraversabilitySnapshot> base,
      SparseGridGeometry geometry, const Pose2 patch_anchor,
      const double start_patch_radius_m,
      std::vector<LocalCellOverride> overrides)
      : base_(std::move(base)),
        geometry_(std::move(geometry)),
        patch_anchor_(patch_anchor),
        start_patch_radius_m_(start_patch_radius_m),
        overrides_(std::move(overrides)) {
    if (!base_) {
      throw std::invalid_argument("request-local view requires a base snapshot");
    }
    const SparseGridGeometry& base_geometry = base_->geometry();
    if (!geometry_.valid() ||
        geometry_.frame_id() != base_geometry.frame_id() ||
        geometry_.resolution_m() != base_geometry.resolution_m() ||
        geometry_.origin_m() != base_geometry.origin_m() ||
        geometry_.min_inclusive().x < base_geometry.min_inclusive().x ||
        geometry_.min_inclusive().y < base_geometry.min_inclusive().y ||
        geometry_.max_exclusive().x > base_geometry.max_exclusive().x ||
        geometry_.max_exclusive().y > base_geometry.max_exclusive().y) {
      throw std::invalid_argument(
          "request-local geometry must be a subset of the base lattice");
    }
    if (!std::isfinite(patch_anchor_.position_m.x) ||
        !std::isfinite(patch_anchor_.position_m.y) ||
        !std::isfinite(patch_anchor_.yaw_rad) ||
        !std::isfinite(start_patch_radius_m_) || start_patch_radius_m_ < 0.0) {
      throw std::invalid_argument(
          "request-local patch geometry must be finite and nonnegative");
    }
    for (const auto& cell : overrides_) {
      if (!geometry_.Contains(cell.index)) {
        throw std::invalid_argument(
            "request-local override lies outside the local geometry");
      }
      if (cell.source != LocalCellSource::kStartAssumedFree ||
          base_->State(cell.index) != FineCellState::kUnknown) {
        throw std::invalid_argument(
            "request-local overrides may only assume unknown cells free");
      }
      if (!CellIntersectsPatchDisk(cell.index)) {
        throw std::invalid_argument(
            "request-local override lies outside the start patch disk");
      }
      if (!std::isfinite(cell.traversal_cost) || cell.traversal_cost < 0.0) {
        throw std::invalid_argument(
            "request-local traversal costs must be finite and nonnegative");
      }
    }
    std::sort(overrides_.begin(), overrides_.end(), OverrideLess);
    if (std::adjacent_find(overrides_.begin(), overrides_.end(),
                           [](const LocalCellOverride& left,
                              const LocalCellOverride& right) {
                             return left.index == right.index;
                           }) != overrides_.end()) {
      throw std::invalid_argument(
          "request-local overrides must contain unique cells");
    }
  }

  [[nodiscard]] std::shared_ptr<const FineTraversabilitySnapshot> base()
      const noexcept {
    return base_;
  }

  [[nodiscard]] const SparseGridGeometry& geometry() const noexcept {
    return geometry_;
  }

  [[nodiscard]] const Pose2& patch_anchor() const noexcept {
    return patch_anchor_;
  }

  [[nodiscard]] double start_patch_radius_m() const noexcept {
    return start_patch_radius_m_;
  }

  [[nodiscard]] LocalCellSource Source(const GridIndex index) const noexcept {
    if (!geometry_.Contains(index)) {
      return LocalCellSource::kUnknown;
    }
    if (const auto* cell = FindOverride(index); cell != nullptr) {
      return cell->source;
    }
    switch (base_->State(index)) {
      case FineCellState::kFree:
        return LocalCellSource::kEvidenceFree;
      case FineCellState::kBlocked:
        return LocalCellSource::kBlocked;
      case FineCellState::kUnknown:
        return LocalCellSource::kUnknown;
    }
    return LocalCellSource::kUnknown;
  }

  [[nodiscard]] double TraversalCost(const GridIndex index) const noexcept {
    if (!geometry_.Contains(index)) {
      return 0.0;
    }
    if (const auto* cell = FindOverride(index); cell != nullptr) {
      return cell->traversal_cost;
    }
    return base_->TraversalCost(index);
  }

  [[nodiscard]] std::optional<StartPhase> AdvancePhase(
      const StartPhase phase, const GridIndex index) const noexcept {
    switch (Source(index)) {
      case LocalCellSource::kEvidenceFree:
        return StartPhase::kNormal;
      case LocalCellSource::kStartAssumedFree:
        return phase == StartPhase::kStartPrefix
                   ? std::optional<StartPhase>(StartPhase::kStartPrefix)
                   : std::nullopt;
      case LocalCellSource::kUnknown:
      case LocalCellSource::kBlocked:
        return std::nullopt;
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<StartPhase> AdvanceGridStep(
      const StartPhase phase, const GridIndex current,
      const GridIndex next) const noexcept {
    const double dx = static_cast<double>(next.x) - static_cast<double>(current.x);
    const double dy = static_cast<double>(next.y) - static_cast<double>(current.y);
    if ((dx == 0.0 && dy == 0.0) || std::abs(dx) > 1.0 || std::abs(dy) > 1.0) {
      return std::nullopt;
    }
    const auto next_phase = AdvancePhase(phase, next);
    if (!next_phase) return std::nullopt;
    if (dx != 0.0 && dy != 0.0 &&
        (!AdvancePhase(phase, {.x = next.x, .y = current.y}) ||
         !AdvancePhase(phase, {.x = current.x, .y = next.y}))) {
      return std::nullopt;
    }
    return next_phase;
  }

  [[nodiscard]] bool CanBeEndpoint(const GridIndex index) const noexcept {
    return geometry_.Contains(index) &&
           base_->State(index) == FineCellState::kFree;
  }

  [[nodiscard]] bool CanCertifyLeggedSupport(
      const GridIndex index, const bool is_initial_support) const noexcept {
    const LocalCellSource source = Source(index);
    return source == LocalCellSource::kEvidenceFree ||
           (is_initial_support &&
            source == LocalCellSource::kStartAssumedFree);
  }

  [[nodiscard]] std::span<const LocalCellOverride> overrides() const noexcept {
    return overrides_;
  }

 private:
  [[nodiscard]] static bool OverrideLess(const LocalCellOverride& left,
                                         const LocalCellOverride& right) {
    if (left.index.y != right.index.y) {
      return left.index.y < right.index.y;
    }
    return left.index.x < right.index.x;
  }

  [[nodiscard]] bool CellIntersectsPatchDisk(
      const GridIndex index) const noexcept {
    const SparseGridGeometry& geometry = geometry_;
    const Vec3 origin = geometry.origin_m();
    const double resolution_m = geometry.resolution_m();
    const double cell_min_x =
        std::fma(static_cast<double>(index.x), resolution_m, origin.x);
    const double cell_min_y =
        std::fma(static_cast<double>(index.y), resolution_m, origin.y);
    const double cell_max_x = cell_min_x + resolution_m;
    const double cell_max_y = cell_min_y + resolution_m;
    const double nearest_x = std::clamp(patch_anchor_.position_m.x,
                                        cell_min_x, cell_max_x);
    const double nearest_y = std::clamp(patch_anchor_.position_m.y,
                                        cell_min_y, cell_max_y);
    return std::hypot(patch_anchor_.position_m.x - nearest_x,
                      patch_anchor_.position_m.y - nearest_y) <=
           start_patch_radius_m_;
  }

  [[nodiscard]] const LocalCellOverride* FindOverride(
      const GridIndex index) const noexcept {
    const LocalCellOverride key{.index = index};
    const auto found =
        std::lower_bound(overrides_.begin(), overrides_.end(), key, OverrideLess);
    return found != overrides_.end() && found->index == index ? &*found : nullptr;
  }

  std::shared_ptr<const FineTraversabilitySnapshot> base_;
  SparseGridGeometry geometry_;
  Pose2 patch_anchor_;
  double start_patch_radius_m_{};
  std::vector<LocalCellOverride> overrides_;
};

struct SnapshotBundle final {
  std::shared_ptr<const FineTraversabilitySnapshot> fine;
  std::shared_ptr<const GlobalGuidanceSnapshot> guidance;
};

}  // namespace lunar::incremental_navigation
