#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

#include "lunar_incremental_navigation_core/elevation_map.hpp"
#include "lunar_incremental_navigation_core/traversability_snapshot.hpp"
#include "lunar_incremental_navigation_core/types/platform_capability.hpp"

namespace lunar::incremental_navigation {

enum class IntrinsicCellState : std::uint8_t {
  kUnknown,
  kFree,
  kBlocked,
};

struct IntrinsicTraversalEvaluation final {
  IntrinsicCellState state{IntrinsicCellState::kUnknown};
  double slope_rad{};
  double relief_m{};
};

struct FineCellEvaluation final {
  FineCellState state{FineCellState::kUnknown};
  double traversal_cost{};
};

class PlatformElevationEvaluator {
 public:
  virtual ~PlatformElevationEvaluator() = default;

  [[nodiscard]] virtual IntrinsicTraversalEvaluation Evaluate(
      const ElevationRangeView& elevation, GridIndex index) const = 0;
};

[[nodiscard]] std::unique_ptr<const PlatformElevationEvaluator>
MakePlatformElevationEvaluator(const PlatformCapability& capability);

class FineCellEvaluator final {
 public:
  FineCellEvaluator(const ElevationRangeView& elevation,
                    const PlatformCapability& capability,
                    const TraversabilityProfile& profile);
  ~FineCellEvaluator();

  FineCellEvaluator(const FineCellEvaluator&) = delete;
  FineCellEvaluator& operator=(const FineCellEvaluator&) = delete;
  FineCellEvaluator(FineCellEvaluator&&) noexcept;
  FineCellEvaluator& operator=(FineCellEvaluator&&) noexcept;

  [[nodiscard]] FineCellEvaluation Evaluate(GridIndex index);
  [[nodiscard]] double hard_inflation_radius_m() const noexcept;
  [[nodiscard]] std::size_t evaluated_elevation_cells() const noexcept;
  [[nodiscard]] std::size_t cached_elevation_tiles() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct FineElevationChangeSet final {
  std::uint64_t base_raw_elevation_revision{};
  std::span<const GridIndex> changed_cells;
};

[[nodiscard]] double CircumscribedRadius(
    std::span<const Point2> envelope);

[[nodiscard]] bool CircleIntersectsCellArea(
    Point2 circle_center_m, double radius_m,
    const SparseGridGeometry& geometry, GridIndex cell) noexcept;

class FineTraversabilityBuilder final {
 public:
  [[nodiscard]] std::shared_ptr<const FineTraversabilitySnapshot> Derive(
      std::shared_ptr<const ElevationSnapshot> raw,
      const PlatformCapability& capability,
      const TraversabilityProfile& profile,
      std::shared_ptr<const FineTraversabilitySnapshot> previous = nullptr,
      std::optional<FineElevationChangeSet> merged_changes = std::nullopt)
      const;
};

}  // namespace lunar::incremental_navigation
